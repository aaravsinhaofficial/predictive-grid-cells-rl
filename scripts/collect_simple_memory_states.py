"""Native blackout trajectories for descriptive activity analysis of fixed finals.

States are AFTER the policy input and BEFORE the sampled action is executed.
All physical labels refer to that same pre-action instant. This collector does
not fit a projection, select trajectories, or qualify an attractor.
"""
from __future__ import annotations

import argparse
from collections import deque
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from code import train_memory as training
import numpy as np
import torch
from code.common import CodeSource, build_policy, build_population
from code.memory_task import MemoryClock, gate_observation
from lstm_code_swap import MatchedArena, summarize_episodes

SETTINGS = dict(seed=741022, episodes=128, n_envs=32, max_gap=60, stride=3)
PROTOCOL = ROOT / 'docs/research/simple_memory_training_protocol.json'
ANALYSIS_PROTOCOL = ROOT / 'docs/research/simple_memory_analysis_protocol.json'
SOURCES = ('scripts/collect_simple_memory_states.py', 'scripts/evaluate_simple_memory.py',
           'scripts/lstm_code_swap.py', *training.SOURCES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_array(value, dtype=None):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return np.array(array, dtype=dtype, copy=True)


def schema(arrays):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), bytes=v.nbytes,
                    sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in arrays.items()}


def transition_digest(fields):
    """Typed, shaped selected-slot bytes; no network/environment execution."""
    result = {}
    for key, value in fields.items():
        array = tensor_array(value)
        header = json.dumps(dict(dtype=str(array.dtype), shape=list(array.shape)), sort_keys=True).encode()
        result[key] = hashlib.sha256(header+b'\n'+array.tobytes()).hexdigest()
    return result


def physical_eligibility(prefix, endpoint, goal_position, goal_vector, *, half, wall_margin,
                         goal_radius, visible_steps=15, tolerance=1e-5):
    """Prespecified all-angle tests; neural state values never enter selection."""
    failures = []
    physical_keys = ('rgb', 'current', 'goal', 'pa', 'pr', 'pos', 'body_heading',
                     'travel_dir', 'velocity', 'goal_pos', 'known', 'blind', 'hit', 'done', 'action')
    if len(prefix['pos']) != visible_steps: failures.append('visible_prefix_length')
    if not bool(prefix['known'].all()) or bool(prefix['blind'].any()): failures.append('prefix_not_known_visible')
    if bool(prefix['hit'].any()) or bool(prefix['done'].any()): failures.append('prefix_hit_or_terminal')
    if 'hit_wall' in prefix and bool(prefix['hit_wall'].any()): failures.append('native_wall_contact')
    if not bool((prefix['episode_id'] == prefix['episode_id'][0]).all()): failures.append('prefix_crosses_episode')
    if not torch.equal(prefix['goal'], goal_vector[None].expand_as(prefix['goal'])):
        failures.append('native_goal_message_changed')
    if not torch.equal(prefix['goal_pos'], goal_position[None].expand_as(prefix['goal_pos'])):
        failures.append('goal_position_changed')
    finite = all(bool(torch.isfinite(prefix[key]).all()) for key in physical_keys)
    finite &= all(bool(torch.isfinite(value).all()) for value in (endpoint, goal_position, goal_vector))
    values = dict(radius=None, wall_clearance=None, goal_clearance=None, respawn_clearance=None)
    if not finite:
        failures.append('nonfinite_physical_or_input_field')
    else:
        p = endpoint.double(); points = prefix['pos'].double()
        radius = float((points-p).norm(dim=-1).max())
        distance = float((goal_position.double()-p).norm())
        wall = half-wall_margin-float(p.abs().max())-radius
        goal_clearance = distance-radius-goal_radius
        respawn = distance-float((points[0]-p).norm())-(goal_radius+.1)
        values.update(radius=radius, wall_clearance=wall, goal_clearance=goal_clearance, respawn_clearance=respawn)
        if not wall > tolerance: failures.append('all_angle_wall_clearance')
        if not goal_clearance > tolerance: failures.append('all_angle_goal_clearance')
        if not respawn >= -tolerance: failures.append('all_angle_native_respawn_clearance')
    return dict(eligible=not failures, failed_rules=failures, tolerance=tolerance, **values,
                selection_uses_neural_state=False)


@torch.inference_mode()
def collect(policy, cfg, *, device='cpu', seed=741022, episodes=128, n_envs=32,
            max_gap=60, stride=3, deadline=float('inf'), env_class=MatchedArena):
    """One native blind branch; private arena/action/clock RNGs, no state erasure.

    Returns lossless arrays, small metadata, and up to four onset snapshots. Sampling includes episode steps
    0, stride, 2*stride, ... in EVERY episode, regardless of reward/blackout.
    goal_pickup_* records the native inputs used for the current goal cache;
    unknown goals have zero pickup fields. The 512-vector goal itself is never
    changed by the collector. Uniform state arrays omit images/current codes;
    only the four onset contexts retain their bounded visible input prefixes.
    """
    if min(episodes, n_envs, stride, max_gap) < 1 or episodes % n_envs:
        raise ValueError('Positive sizes and whole episode batches required')
    if cfg['policy'] != 'lstm' or not policy.recurrent or cfg.get('extra_input', 'none') != 'none':
        raise ValueError('Native LSTM without extra inputs required')
    if cfg.get('code_noise', 0) != 0 or cfg['n_cells'] != 512:
        raise ValueError('Native deterministic 512-cell population required')
    started = time.monotonic()
    was_training = policy.training
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None
    policy.eval()
    try:
        n, length = n_envs, cfg['episode_len']
        env = env_class(cfg, n, 1, seed, device)
        obs = env.reset_all()
        population = build_population(cfg, device)
        source = CodeSource(cfg, population, device)
        clock = MemoryClock(n, device, seed + 2, visible_steps=cfg.get('task_visible_steps', 15))
        sampler = torch.Generator(device=device).manual_seed(seed + 1)
        state = policy.zero_state(n, device)
        goal = torch.zeros(n, 512, device=device)
        pickup = {k: torch.zeros(n, 2, device=device) for k in ('pos', 'dir', 'vel')}
        pa = torch.full((n,), -1, dtype=torch.long, device=device)
        pr = torch.zeros(n, device=device)
        phase_age = torch.ones(n, dtype=torch.long, device=device)
        episode_index = torch.zeros(n, dtype=torch.long, device=device)
        slot = torch.arange(n, device=device)
        count_names = ('known_steps', 'blind_steps', 'blind_goals', 'seen_known_goals')
        counts = torch.zeros(n, len(count_names), dtype=torch.long, device=device)
        rows, layouts, contexts, considered_contexts = [], [], [], set()
        candidate_ledger, selection_closed = [], None
        history = deque(maxlen=clock.visible_steps)
        samples = episodes * ((length + stride - 1) // stride)
        floats = dict(h=policy.nh, c=policy.nh, pos=2, body_heading=0, travel_dir=2,
                      velocity=2, goal_pos=2, goal_pickup_pos=2, goal_pickup_dir=2,
                      goal_pickup_vel=2, prev_reward=0, reward=0, hit_pos=2)
        integers = ('slot', 'episode_index', 'episode_id', 'native_episode_index',
                    'episode_step', 'global_step', 'phase', 'phase_age', 'phase_remaining',
                    'prev_action', 'action')
        arrays = {k: np.empty((samples, width) if width else (samples,), np.float32)
                  for k, width in floats.items()}
        arrays.update({k: np.empty(samples, np.int32) for k in integers})
        arrays.update({k: np.empty(samples, np.bool_) for k in ('known', 'blind', 'hit', 'done')})
        cursor = 0
        for step in range(length * (episodes // n)):
            if step % 100 == 0 and time.monotonic() >= deadline:
                raise TimeoutError('State collection deadline; no completion claim')
            episode_step = step % length
            if not bool((env.t == episode_step).all()):
                raise AssertionError('Unexpected asynchronous/native episode clock')
            if episode_step == 0:
                layouts.append(dict(episode_id=tensor_array(episode_index*n + slot, np.int32),
                    goal_pos=tensor_array(env.goal, np.float32),
                    land_pos=tensor_array(env.land_pos, np.float32),
                    stripe_phase=tensor_array(env.stripe_phase, np.float32)))
            blind, known = clock.mask(), clock.known.clone()
            if not torch.equal(known, env.n_goals > 0):
                raise AssertionError('Native discovery and observation clock disagree')
            if len(contexts) < 4:
                eligible = (blind & (phase_age == 1)).nonzero().flatten().cpu().tolist()
                for index in eligible:
                    key = (int(episode_index[index]), index)
                    if key in considered_contexts: continue
                    considered_contexts.add(key)  # A rejected first onset can never be replaced by a later one.
                    prefix = {name: torch.stack([row[name][index].detach().cpu().clone() for row in history])
                              for name in history[0]}
                    eligibility = physical_eligibility(prefix, obs['pos'][index].cpu(), env.goal[index].cpu(),
                        goal[index].cpu(), half=env.half, wall_margin=env.wall_margin,
                        goal_radius=env.goal_radius, visible_steps=clock.visible_steps)
                    ledger = dict(selected_slot=index, episode_index=key[0], episode_id=key[0]*n+index,
                                  global_step=step, status='selected' if eligibility['eligible'] else 'skipped',
                                  physical=eligibility)
                    candidate_ledger.append(ledger)
                    if not eligibility['eligible']: continue
                    context_runtime = dict(env=env, obs=obs, state=state, goal=goal, pa=pa, pr=pr,
                        clock=clock, counters=dict(collection_steps=step, frames=step*n*cfg['action_repeat'],
                                                  completed_episodes=len(rows)), recent_returns=[])
                    contexts.append(dict(selected_slot=index, episode_index=key[0], episode_id=key[0]*n+index,
                        global_step=step, phase_age=1, runtime=training.capture_runtime(context_runtime),
                        action_sampler_rng=sampler.get_state().cpu().clone(),
                        phase_ages=phase_age.cpu().clone(), episode_indices=episode_index.cpu().clone(),
                        visible_prefix=prefix, original_batch_size=n, physical_eligibility=eligibility,
                        native_continuation=[],
                        timing='Before the first blind policy input; complete native batch runtime, no optimization state.'))
                    if len(contexts) == 4:
                        selection_closed = dict(global_step=step, slot=index)
                        break
            rgb, current = gate_observation(obs['rgb'], source(obs['pos'], obs['dir'], obs['vel']), blind)
            continuation_rows = []
            for context in contexts:
                recorded = context['native_continuation']
                if len(recorded) >= 150 or (recorded and recorded[-1]['done']): continue
                index = context['selected_slot']
                record = dict(global_step=step, offset=step-context['global_step'],
                    input_hashes=transition_digest({key: value[index] for key, value in
                        dict(rgb=rgb, current=current, goal=goal, pa=pa, pr=pr).items()}),
                    h_before=state[0][index].cpu().clone(), c_before=state[1][index].cpu().clone(),
                    known=bool(known[index]), blind=bool(blind[index]), phase_age=int(phase_age[index]))
                continuation_rows.append((context, record))
            history_row = {key: value.detach().clone() for key, value in dict(
                rgb=rgb, current=current, goal=goal, pa=pa, pr=pr,
                h_before=state[0], c_before=state[1], pos=obs['pos'], body_heading=obs['hd'],
                travel_dir=obs['dir'], velocity=obs['vel'], goal_pos=env.goal,
                known=known, blind=blind, episode_id=episode_index*n+slot,
                episode_step=env.t, global_step=torch.full_like(slot, step)).items()} if len(contexts) < 4 else None
            if history_row is None: history.clear()
            logits, _, state = policy.step(rgb, current, goal, pa, pr, state)
            uniform = torch.rand(n, generator=sampler, device=device)
            action = (uniform[:, None] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(5)
            for context, record in continuation_rows:
                index = context['selected_slot']
                record.update(h_after=state[0][index].cpu().clone(), c_after=state[1][index].cpu().clone(),
                              action=int(action[index]), probabilities=logits[index].softmax(-1).cpu().clone())
            if history_row is not None:
                for key, value in dict(action=action, h_after=state[0], c_after=state[1]).items():
                    history_row[key] = value.detach().clone()
            selected = episode_step % stride == 0
            if selected:
                region = slice(cursor, cursor+n)
                values = dict(h=state[0], c=state[1], pos=obs['pos'], body_heading=obs['hd'],
                    travel_dir=obs['dir'], velocity=obs['vel'], goal_pos=env.goal,
                    goal_pickup_pos=pickup['pos'], goal_pickup_dir=pickup['dir'], goal_pickup_vel=pickup['vel'],
                    known=known, blind=blind, phase=torch.where(known, 1+blind.long(), 0),
                    phase_age=phase_age, phase_remaining=clock.remaining, slot=slot,
                    episode_index=episode_index, episode_id=episode_index*n+slot,
                    native_episode_index=env.ep_idx, episode_step=env.t,
                    global_step=torch.full_like(slot, step), prev_action=pa, prev_reward=pr, action=action)
                for key, value in values.items(): arrays[key][region] = tensor_array(value)
            obs, raw, done, info = env.step(action)
            hit = info['hit']
            for context, record in continuation_rows:
                index = context['selected_slot']
                post = {key: value[index].cpu().clone() for key,value in dict(
                    pos=obs['pos'], hd=obs['hd'], mdir=obs['dir'], vel=obs['vel'], t=env.t,
                    hit_pos=info['hit_pos'], hit_dir=info['hit_dir'], hit_vel=info['hit_vel']).items()}
                record.update(reward=float(raw[index]), hit=bool(hit[index]), done=bool(done[index]),
                              hit_pos=post['hit_pos'].clone(), post_physics=post,
                              post_physics_hashes=transition_digest({key:post[key] for key in ('pos','hd','mdir','vel','t')}))
                context['native_continuation'].append(record)
            if history_row is not None:
                for key, value in dict(hit=hit, done=done, reward=raw, hit_wall=info['hit_wall']).items():
                    history_row[key] = value.detach().clone()
                history.append(history_row)
            if selected:
                for key, value in dict(reward=raw, hit=hit, done=done, hit_pos=info['hit_pos']).items():
                    arrays[key][region] = tensor_array(value)
                cursor += n
            counts += torch.stack((known, blind, hit & blind & known, hit & ~blind & known), 1).long()
            if bool(hit.any()):
                goal = torch.where(hit[:, None], source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']), goal)
                for key in pickup:
                    pickup[key] = torch.where(hit[:, None], info['hit_' + key], pickup[key])
            clock.advance(hit, done, max_gap)
            same_phase = (known == clock.known) & (blind == clock.mask()) & ~hit & ~done
            phase_age = torch.where(same_phase, phase_age+1, torch.ones_like(phase_age))
            reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
            keep = ~done
            pa = torch.where(done, torch.full_like(action, -1), action)
            pr = reward * keep
            goal = goal * keep[:, None]
            pickup = {key: value * keep[:, None] for key, value in pickup.items()}
            state = tuple(value * keep[:, None] for value in state)
            stats = info['ep_stats']
            if stats is not None:
                for j, index in enumerate(stats['env'].cpu().tolist()):
                    row = {key: float(value[j]) for key, value in stats.items() if key != 'env'}
                    row.update(slot=index, episode_index=int(episode_index[index]),
                               episode_id=int(episode_index[index])*n+index,
                               **dict(zip(count_names, counts[index].cpu().tolist())))
                    rows.append(row)
            counts *= keep[:, None]
            episode_index += done.long()
            if bool(done.any()) != (episode_step == length-1) or (bool(done.any()) and not bool(done.all())):
                raise AssertionError('Unexpected native terminal schedule')
        assert cursor == samples and len(rows) == episodes
        assert len({row['episode_id'] for row in rows}) == episodes
        for key in layouts[0]: arrays['layout_' + key] = np.concatenate([x[key] for x in layouts])
        for key in ('K', 'phi', 'shift', 'is_pgc'):
            arrays['map_' + key] = tensor_array(getattr(population, key))
        assert all(np.isfinite(value).all() for value in arrays.values())
        assert torch.equal(cpu_rng, torch.get_rng_state()), 'Collector consumed global CPU RNG'
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device)), 'Collector consumed global CUDA RNG'
        uncompressed = sum(value.nbytes for value in arrays.values())
        if uncompressed > 200_000_000:
            raise ValueError('Collector exceeds its 200MB/model raw-array bound')
        metadata = dict(complete=True, seed=seed, episodes=episodes, n_envs=n, max_gap=max_gap,
            stride=stride, samples=samples, samples_per_episode=(length+stride-1)//stride,
            decisions=episodes*length, frames=episodes*length*cfg['action_repeat'],
            seconds=time.monotonic()-started, uncompressed_array_bytes=uncompressed,
            global_rng_unchanged=True, numerical_dtype='float32; integer labels and Boolean masks retained',
            state_timing='After policy input and before env.step; physical labels and previous action/reward are the inputs to that update.',
            transition_timing='action/reward/hit/done/hit_pos describe the immediately following native transition. hit_pos is pre-teleport, including on terminal steps.',
            phase_labels={'0': 'before first discovery', '1': 'visible after discovery', '2': 'joint blackout'},
            phase_age='One-based observation age since phase entry or most recent pickup/reset; the first blind input has age1. Before discovery, age is episode_step+1.',
                goal_pickup_fields='Exact native position/direction/velocity that produced the saved goal message; zero when unknown. No goal-code withdrawal.',
            sampling='All episode steps divisible by stride, including step0; no outcome/phase selection.',
            summary=summarize_episodes(rows, length),
            totals={key: sum(row[key] for row in rows) for key in count_names},
            onset_contexts=dict(count=len(contexts), requested=4,
                selection='First physically eligible first-blackout-of-episode contexts, chronological step then ascending slot. Rejected episodes never supply later onsets. Selection never uses neural outcomes.',
                candidate_ledger=candidate_ledger, selection_closed=selection_closed,
                visible_prefix='Only the selected slot: all preceding visible_steps actual native inputs, incoming/after h/c, labels/actions/outcomes. All are known, visible, same-episode observations. Original batch size/slot retained.',
                continuation='Selected-slot hashes, exact incoming/after h/c, action/reward and native post-physics for the next min(150,remaining episode) existing decisions; no new rollout.',
                rows=[{key: context[key] for key in ('selected_slot', 'episode_index', 'episode_id', 'global_step', 'phase_age')} for context in contexts]),
            scope='Native closed-loop rollout of one frozen model. Hidden h is the primary descriptive PCA activity; full h/c is retained. No ring claim or causal mechanism test.')
        return arrays, metadata, contexts
    finally:
        policy.train(was_training)


@torch.inference_mode()
def verify_visible_prefix(policy, context, device='cpu', strict=True):
    """Replay only archived history; replicate its slot to the original batch shape.

    There is no cross-example batch normalization in the native architecture.
    Equality is measured, not assumed from equal shapes or called bitwise when
    floating-point execution differs. This does not advance any environment.
    """
    prefix, n = context['visible_prefix'], context['original_batch_size']
    repeat = lambda value: value.to(device).unsqueeze(0).repeat(n, *([1]*value.ndim))
    state = tuple(repeat(prefix[key][0]) for key in ('h_before', 'c_before'))
    maximum = [0., 0.]; exact = True; qualified = True; finite = [True, True]
    for step in range(len(prefix['rgb'])):
        values = [repeat(prefix[key][step]) for key in ('rgb', 'current', 'goal', 'pa', 'pr')]
        _, _, state = policy.step(*values, state)
        for i, key in enumerate(('h_after', 'c_after')):
            expected = prefix[key][step].to(device)
            error = (state[i][0]-expected).abs().max()
            finite[i] &= bool(torch.isfinite(error))
            if finite[i]: maximum[i] = max(maximum[i], float(error))
            exact &= torch.equal(state[i][0], expected)
            qualified &= torch.allclose(state[i][0], expected, rtol=1e-4, atol=1e-5)
    result = dict(steps=len(prefix['rgb']), replay_batch_size=n, selected_slot=context['selected_slot'],
                  h_max_absolute_error=maximum[0] if finite[0] else None,
                  c_max_absolute_error=maximum[1] if finite[1] else None,
                  finite=all(finite), bitwise_equal=bool(exact), qualified=bool(qualified), atol=1e-5, rtol=1e-4,
                  scope='A failed replay excludes this context from prefix-based family mechanisms; it does not invalidate or remove uniform native PCA samples.')
    if strict and not qualified: raise AssertionError(result)
    return result


def load_final(run, device='cpu', threads=2):
    """Authenticate a complete 1B endpoint; never substitute an earlier stage."""
    run = Path(run)
    cfg = json.loads((run/'config.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    proof = json.loads((run/'provenance.json').read_text())
    producer_evidence = dict(producer_version=1)
    if proof.get('provenance_version') == 2:
        from scripts.simple_memory_endpoint import validate_v2_provenance
        producer_evidence = validate_v2_provenance(run, cfg, proof, root=ROOT)
    else:
        assert 'provenance_version' not in proof, 'Unknown training producer version'
    saved = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
    assert summary['complete'] and summary['frames'] == saved['frames'] == cfg['frames'] == 1_000_000_000
    assert saved['updates'] == summary['updates'] == 78125
    assert summary['final_sha256'] == sha(run/'final.pt')
    assert saved['provenance'] == summary['provenance'] == proof
    assert saved['config'] == {key: cfg[key] for key in saved['config']}
    assert proof['protocol_sha256'] == sha(PROTOCOL)
    for path, digest in proof['source_sha256'].items(): assert sha(ROOT/path) == digest, path
    assert cfg['tf32'] == 0, 'This collector uses the frozen full-FP32 setting'
    training.configure(dict(cfg, device=device, threads=threads))
    policy = build_policy(cfg, device)
    policy.load_state_dict(saved['policy'], strict=True)
    policy.requires_grad_(False)
    assert training.policy_hash(policy) == saved['policy_sha256'] == summary['final_policy_sha256']
    return policy, cfg, dict(checkpoint_sha256=sha(run/'final.pt'), config_sha256=sha(run/'config.json'),
        terminal_summary_sha256=sha(run/'summary.json'), training_provenance_sha256=sha(run/'provenance.json'),
        training_protocol_sha256=proof['protocol_sha256'], checkpoint_frames=saved['frames'],
        checkpoint_updates=saved['updates'], policy_sha256=saved['policy_sha256'], **producer_evidence)


def authenticate_analysis_protocol(path):
    path = Path(path)
    protocol = json.loads(path.read_text())
    sources = protocol.get('source_sha256', {})
    assert protocol.get('status') == 'frozen', 'Analysis protocol is not frozen; no model collection allowed'
    assert sources and 'scripts/collect_simple_memory_states.py' in sources, 'Collector must be in the frozen source closure'
    for name, expected in sources.items(): assert sha(ROOT/name) == expected, name
    return dict(analysis_protocol_sha256=sha(path), analysis_protocol_path=str(path))


def save_arrays(out, arrays, metadata, contexts):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('State output must be empty')
    path = out/'states.npz'
    with (out/'states.npz.tmp').open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    (out/'states.npz.tmp').replace(path)
    context_path = out/'onset_contexts.pt'
    training.atomic_torch(context_path, dict(schema=1, contexts=contexts,
        checkpoint_sha256=metadata.get('checkpoint_sha256'),
        scope='Read-only collection runtime before blind onset; not a training checkpoint.'))
    if path.stat().st_size + context_path.stat().st_size > 200_000_000:
        raise ValueError('Combined state/context artifacts exceed 200MB/model')
    metadata = dict(metadata, array_schema=schema(arrays), array_file=dict(
        path=path.name, sha256=sha(path), bytes=path.stat().st_size),
        contexts_file=dict(path=context_path.name, sha256=sha(context_path), bytes=context_path.stat().st_size))
    training.atomic_json(out/'summary.json', metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--analysis-protocol', type=Path, default=ANALYSIS_PROTOCOL)
    args = parser.parse_args(argv)
    out = args.out or args.run/'memory_states'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('State output must be empty before model execution')
    analysis_proof = authenticate_analysis_protocol(args.analysis_protocol)
    policy, cfg, proof = load_final(args.run, args.device, args.threads)
    before = training.policy_hash(policy)
    arrays, result, contexts = collect(policy, cfg, device=args.device, deadline=time.monotonic()+args.max_seconds, **SETTINGS)
    result['visible_prefix_replay'] = [verify_visible_prefix(policy, context, args.device, strict=False) for context in contexts]
    for context, validation in zip(contexts, result['visible_prefix_replay']):
        context['visible_prefix_validation'] = validation
    result['qualified_onset_contexts'] = sum(row['qualified'] for row in result['visible_prefix_replay'])
    assert training.policy_hash(policy) == before
    result.update(proof, **analysis_proof, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        condition=cfg['cond'], training_seed=cfg['seed'], training_task_max_gap=cfg['task_max_gap'],
        source_sha256={name: sha(ROOT/name) for name in dict.fromkeys(SOURCES)},
        software=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()),
        numerics=dict(device=args.device, threads=args.threads, dtype='float32', matmul_tf32=False,
            cudnn_tf32=False, deterministic_algorithms=True, cudnn_benchmark=False,
            gpu_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None),
        policy_unchanged=True)
    result = save_arrays(out, arrays, result, contexts)
    print(json.dumps({key: result[key] for key in ('complete', 'samples', 'frames', 'seconds', 'array_file')}), flush=True)


if __name__ == '__main__':
    main()
