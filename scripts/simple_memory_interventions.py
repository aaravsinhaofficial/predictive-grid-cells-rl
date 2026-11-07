"""Prespecified state/cue interventions at genuine native blackout onsets.

No diagnostic decoder or geometry controls the agent. A pulse replaces only
the current code on the selected slot's first blind input; free navigation
then continues with the native observation clock and actor.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from code import train_memory as training
from code.common import CodeSource, build_population
from code.memory_task import MemoryClock, gate_observation
from lstm_code_swap import MatchedArena
from lstm_travel_cue_pulse import semantic_nearest
from collect_simple_memory_states import transition_digest

CUES = ('none', 'ordinary', 'correct', 'wrong_plus', 'wrong_minus')
EDITS = ('none', 'plus30', 'minus30', 'normal_plus', 'normal_minus')
TIMES = (0, 1, 5, 15, 30, 60)


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def tensor_hash(value):
    a = np.ascontiguousarray(array(value))
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_edits(family, subset=None):
    """Both signed phase donors and a matched, tangent-orthogonal control.

    Invalid random controls stay invalid; no direction search or clipping.
    Per-phase masks retain valid prespecified rows, with every omission logged.
    """
    z = array(family['states']).astype(np.float64)
    scale = array(family['scale']).astype(np.float64)
    n, d = z.shape
    if n % 12 or d % 2 or np.any(scale <= 0):
        raise ValueError('A whole +/-30-degree family and positive full-state scales are required')
    subset = np.arange(n) if subset is None else np.asarray(subset, dtype=int)
    plus, minus = np.roll(z, -n//12, 0), np.roll(z, n//12, 0)
    tangent = (np.roll(z, -1, 0)-np.roll(z, 1, 0))/scale
    tn = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(tn, 1e-30)
    direction = np.random.default_rng(741025).standard_normal(z.shape)
    direction -= np.sum(direction*tangent, axis=1, keepdims=True)*tangent
    dn = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= np.maximum(dn, 1e-30)
    size = (np.linalg.norm((plus-z)/scale, axis=1) +
            np.linalg.norm((minus-z)/scale, axis=1))/2
    normal_plus, normal_minus = z+direction*size[:, None]*scale, z-direction*size[:, None]*scale
    normal_valid = ((tn[:, 0] > 1e-12) & (dn[:, 0] > 1e-12) & (size > 1e-12) &
                    (np.abs(normal_plus[:, :d//2]).max(1) <= 1) &
                    (np.abs(normal_minus[:, :d//2]).max(1) <= 1))
    values = dict(none=z, plus30=plus, minus30=minus,
                  normal_plus=normal_plus, normal_minus=normal_minus)
    valid = {name: normal_valid[subset] if name.startswith('normal') else np.ones(len(subset), bool)
             for name in EDITS}
    audit = dict(seed=741025, indices=subset.tolist(),
        matched_rms_norm=size[subset].tolist(), normal_valid=normal_valid[subset].tolist(),
        maximum_normal_tangent_dot=float(np.abs((direction*tangent).sum(1)).max()),
        hidden_bound_excess={name: np.maximum(np.abs(v[subset, :d//2]).max(1)-1, 0).tolist()
                             for name, v in values.items()},
        rule='No clipping, rescaling, replacement direction or outcome selection.')
    return {name: v[subset].copy() for name, v in values.items()}, valid, audit


@torch.inference_mode()
def recurrent_step(policy, z, inputs):
    h, c = policy.lstm(inputs, z.split(policy.nh, dim=1))
    return torch.cat((h, c), 1)


def phase_error(points, reference, theta, expected, scale):
    ref = reference/scale
    radius = float(np.sqrt(np.mean(np.sum((ref-ref.mean(0))**2, axis=1))))
    if len(points) == 0 or radius < 1e-12:
        return dict(n=len(points), radius=radius, defined=False)
    distance, phase = semantic_nearest(points/scale, ref, theta)
    error = np.angle(np.exp(1j*(phase-expected)))
    ordered = np.sort(np.mod(phase, 2*np.pi))
    coverage = float(1-np.diff(np.r_[ordered, ordered[0]+2*np.pi]).max()/(2*np.pi))
    return dict(n=len(points), radius=radius, defined=True,
        phase_mae_deg=float(np.rad2deg(np.abs(error)).mean()),
        signed_bias_deg=float(np.rad2deg(np.angle(np.exp(1j*error).mean()))),
        phase_resultant=float(np.abs(np.exp(1j*error).mean())), coverage=coverage,
        off_family_over_radius_mean=float(np.mean(distance/radius)),
        off_family_over_radius_p90=float(np.quantile(distance/radius, .9)))


@torch.inference_mode()
def cue_repair(policy, family, *, times=TIMES, stride=4, deadline=float('inf')):
    """One code pulse followed by the identical constant dark input.

    Geometry qualification is supplied separately; this returns descriptive
    errors even for collapsed/failed families and never declares a ring.
    """
    original = family['states']
    subset = np.arange(0, len(original), stride)
    edits, valid, audit = make_edits(family, subset)
    device, dtype = original.device, original.dtype
    active = {cue: {name: torch.as_tensor(z, dtype=dtype, device=device) for name, z in edits.items()}
              for cue in CUES}
    for cue in CUES:
        for name in EDITS:
            invalid = torch.as_tensor(~valid[name],device=device)
            active[cue][name][invalid] = original[subset][invalid]
    audit['invalid_row_execution'] = 'Invalid normal rows use no-edit padding for fixed batch shape and are excluded from every reported metric.'
    refs = {cue: original.clone() for cue in CUES}
    scale = array(family['scale']).astype(float)
    theta = array(family['theta']).astype(float)
    results, raw = {}, {name+'_valid':mask.copy() for name,mask in valid.items()}
    for t in range(max(times)+1):
        if time.monotonic() > deadline: raise TimeoutError('Cue-repair deadline')
        if t:
            for cue in CUES:
                inp = family['pulse_inputs'][cue] if t == 1 and cue != 'none' else family['dark_inputs']
                refs[cue] = recurrent_step(policy, refs[cue], inp)
                for name in EDITS:
                    active[cue][name] = recurrent_step(policy, active[cue][name], inp[subset])
        if t not in times: continue
        reference = array(refs['correct']).astype(float)
        probabilities = policy.pi(refs['correct'][subset, :policy.nh]).softmax(-1)
        rows = {}
        for cue in CUES:
            rows[cue] = {}
            own = array(refs[cue]).astype(float)
            own_prob = policy.pi(refs[cue][subset, :policy.nh]).softmax(-1)
            for name in EDITS:
                mask = valid[name]
                z = active[cue][name]
                data = array(z).astype(float)
                prob = policy.pi(z[:, :policy.nh]).softmax(-1)
                rows[cue][name] = dict(valid_rows=int(mask.sum()), requested_rows=len(mask),
                    correct_reference=phase_error(data[mask], reference, theta, theta[subset][mask], scale),
                    own_reference=phase_error(data[mask], own, theta, theta[subset][mask], scale),
                    raw_correct_reference=phase_error(data[mask], reference, theta, theta[subset][mask], np.ones_like(scale)),
                    raw_own_reference=phase_error(data[mask], own, theta, theta[subset][mask], np.ones_like(scale)),
                    actor_tv_correct=float(array((prob-probabilities).abs().sum(-1)/2)[mask].mean()) if mask.any() else None,
                    actor_tv_own=float(array((prob-own_prob).abs().sum(-1)/2)[mask].mean()) if mask.any() else None)
                raw[f't{t}_{cue}_{name}'] = array(z).copy()
            raw[f't{t}_{cue}_reference'] = array(refs[cue]).copy()
        results[str(t)] = rows
    contrasts = {}
    for t, rows in results.items():
        def error(cue, edit):
            return rows[cue][edit]['correct_reference'].get('phase_mae_deg')
        contrasts[t] = {}
        for other in ('ordinary', 'wrong'):
            alt = ('ordinary',) if other == 'ordinary' else ('wrong_plus', 'wrong_minus')
            effects = {}
            for edit in ('none', 'plus30', 'minus30'):
                numbers = [error(c, edit) for c in (*alt, 'correct')]
                effects[edit] = None if any(v is None for v in numbers) else float(np.mean(numbers[:-1])-numbers[-1])
            donor = None if any(effects[e] is None for e in ('plus30','minus30')) else (effects['plus30']+effects['minus30'])/2
            contrasts[t][other+'_minus_correct'] = dict(per_edit=effects, donor_mean=donor,
                donor_minus_no_edit=None if donor is None or effects['none'] is None else donor-effects['none'])
    return dict(complete=True, times=list(times), subset=subset.tolist(), edits=audit,
        rows=results, contrasts=contrasts, primary_update=15,
        primary_metric='Fixed h/c block RMS; raw-coordinate sensitivity is reported separately.',
        interpretation='State/cue responses require separate geometry and behavioral evidence for predictive stabilization.'), raw


@contextmanager
def preserve_rng(device):
    py, npstate = random.getstate(), np.random.get_state()
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try: yield
        finally:
            random.setstate(py)
            np.random.set_state(npstate)


def restore_context(cfg, context, device, tape, context_index):
    n = context['original_batch_size']
    runtime = dict(env=MatchedArena(cfg, n, 1, 741022, device),
        clock=MemoryClock(n, device, 741024, visible_steps=cfg.get('task_visible_steps',15)))
    training.restore_runtime(runtime, context['runtime'])
    sampler = torch.Generator(device=device).set_state(context['action_sampler_rng'].clone())
    if tape:
        base = 741030 + 1000*cfg['seed'] + 100*context_index + 10*tape
        runtime['env'].gen.manual_seed(base)
        runtime['env'].pair_seed = base
        sampler.manual_seed(base+1)
        runtime['clock'].generator.manual_seed(base+2)
    return runtime, sampler


@torch.inference_mode()
def native_fork(policy, cfg, context, *, edit=None, pulse=None, device='cpu', tape=0,
                context_index=0, horizon=150, max_gap=60, deadline=float('inf')):
    """Continue the archived full batch, altering only the selected recipient.

    Returns selected-slot traces for independent archived-continuation checks.
    The rest of the batch retains its original runtime and shared random tape.
    """
    with preserve_rng(device):
        runtime, sampler = restore_context(cfg, context, device, tape, context_index)
        env, clock = runtime['env'], runtime['clock']
        source = CodeSource(cfg, build_population(cfg, device), device)
        slot, n = context['selected_slot'], context['original_batch_size']
        others = torch.arange(n, device=device) != slot
        if not bool(clock.mask()[slot]) or context['phase_age'] != 1:
            raise ValueError('Intervention must precede the first blind observation')
        original_state = torch.cat(runtime['state'],1)[slot].clone()
        if edit is not None:
            edit = torch.as_tensor(edit, dtype=original_state.dtype, device=device)
            if edit.shape != original_state.shape or not torch.isfinite(edit).all():
                raise ValueError('Invalid full-state edit')
            if bool((edit[:policy.nh].abs() > 1).any()):
                raise ValueError('Unqualified hidden-bound-violating edit')
            for value, replacement in zip(runtime['state'], edit.split(policy.nh)):
                value[slot] = replacement
        if pulse is not None:
            pulse = torch.as_tensor(pulse, dtype=runtime['goal'].dtype, device=device)
            if pulse.shape != (512,) or not torch.isfinite(pulse).all():
                raise ValueError('Invalid single current-code pulse')
        initial_goal = runtime['goal'][slot].clone()
        initial_distance = float((runtime['obs']['pos'][slot]-env.goal[slot]).norm())
        score, hits, known_steps, blind_steps, first_hit = 0., 0, 0, 0, None
        before_hit_distance = initial_distance
        traces, control_hashes = [], []
        for t in range(horizon):
            if time.monotonic() > deadline: raise TimeoutError('Native-fork deadline')
            obs, state = runtime['obs'], runtime['state']
            blind, known = clock.mask(), clock.known.clone()
            rgb, code = gate_observation(obs['rgb'], source(obs['pos'],obs['dir'],obs['vel']),blind)
            if t == 0 and pulse is not None: code[slot] = pulse
            if t == 0:
                assert not bool(rgb[slot].any())
                assert torch.equal(runtime['goal'][slot], initial_goal)
            trace = dict(step=t, known=bool(known[slot]), blind=bool(blind[slot]),
                input_hashes=transition_digest({name:value[slot] for name,value in dict(
                    rgb=rgb,current=code,goal=runtime['goal'],pa=runtime['pa'],pr=runtime['pr']).items()}),
                h_before=array(state[0][slot]).copy(), c_before=array(state[1][slot]).copy(),
                pos=array(obs['pos'][slot]).copy(), body_heading=float(obs['hd'][slot]),
                travel_dir=array(obs['dir'][slot]).copy())
            logits, _, state = policy.step(rgb,code,runtime['goal'],runtime['pa'],runtime['pr'],state)
            uniform = torch.rand(n, generator=sampler, device=device)
            action = (uniform[:,None] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(5)
            trace.update(h_after=array(state[0][slot]).copy(),c_after=array(state[1][slot]).copy(),
                         action=int(action[slot]), probabilities=array(logits[slot].softmax(-1)).copy())
            control_hashes.append({name:tensor_hash(value[others]) for name,value in
                dict(h=state[0],c=state[1],action=action,pos=obs['pos']).items()})
            physical_goal = env.goal[slot].clone()
            next_obs, raw, done, info = env.step(action)
            hit = info['hit']
            trace.update(reward=float(raw[slot]),hit=bool(hit[slot]),done=bool(done[slot]),
                hit_pos=array(info['hit_pos'][slot]).copy(),
                post_physics_hashes=transition_digest({name:value[slot] for name,value in
                    dict(pos=env.pos,hd=env.hd,mdir=env.mdir,vel=next_obs['vel'],t=env.t).items()}))
            traces.append(trace)
            score += float(raw[slot]); hits += int(hit[slot])
            known_steps += int(known[slot]); blind_steps += int(blind[slot])
            if first_hit is None:
                before_hit_distance = float((info['hit_pos'][slot]-physical_goal).norm())
                if bool(hit[slot]): first_hit = t+1
            goal = runtime['goal']
            if bool(hit.any()):
                goal = torch.where(hit[:,None],source.goal(info['hit_pos'],info['hit_dir'],info['hit_vel']),goal)
            keep = ~done
            reward = raw.clamp(-cfg['reward_clip'],cfg['reward_clip']) if cfg['reward_clip']>0 else raw
            runtime.update(obs=next_obs,state=tuple(x*keep[:,None] for x in state),
                goal=goal*keep[:,None],pa=torch.where(done,torch.full_like(action,-1),action),pr=reward*keep)
            clock.advance(hit,done,max_gap)
            if bool(done[slot]): break
        return dict(complete=True,tape=tape,decisions=len(traces),score=score,hits=hits,
            known_steps=known_steps,blind_steps=blind_steps,time_to_first_hit=first_hit,
            first_hit_censored=first_hit is None,initial_distance=initial_distance,
            distance_reduction_before_hit=initial_distance-before_hit_distance,
            ended_at_native_terminal=bool(traces[-1]['done']) if traces else False,
            initial_goal_hash=tensor_hash(initial_goal),
            state_edit_l2=float((torch.cat(context['runtime']['state'],1)[slot].to(device)-
                                (original_state if edit is None else edit)).norm()),
            trace=traces,other_slot_hashes=control_hashes)


def compare_native_trace(actual, expected, *, exact_states=False):
    """Independent collector trace versus a restored no-edit continuation."""
    failures, maximum = [], {key:0. for key in ('h_before','c_before','h_after','c_after')}
    if len(actual) != len(expected): failures.append('trace_length')
    for i, (left,right) in enumerate(zip(actual,expected)):
        for key in ('input_hashes','action','reward','hit','done','post_physics_hashes'):
            if left[key] != right[key]: failures.append(f'{i}:{key}')
        for key in maximum:
            a,b = array(left[key]),array(right[key])
            maximum[key] = max(maximum[key],float(np.max(np.abs(a-b))))
            equal=np.array_equal(a,b) if exact_states else np.allclose(a,b,atol=1e-5,rtol=1e-4)
            if not equal: failures.append(f'{i}:{key}')
        if not np.array_equal(array(left['hit_pos']),array(right['hit_pos'])):
            failures.append(f'{i}:hit_pos')
    return dict(qualified=not failures,steps_compared=min(len(actual),len(expected)),
        failures=failures,maximum_absolute_state_error=maximum,
        rule='Exact inputs, actions, rewards and physics; '+('exact h/c.' if exact_states else 'h/c atol1e-5 rtol1e-4.'))


@torch.inference_mode()
def behavioral_forks(policy,cfg,family,*,device='cpu',context_index=0,
                     tapes=4,horizon=150,deadline=float('inf')):
    """All prespecified branches; no geometry-dependent branch selection."""
    context = family['original_context']
    pop = build_population(cfg,device)
    codes = family['current_codes']
    for code in codes.values():
        if not torch.equal(code[:,~pop.is_pgc],codes['ordinary'][:,~pop.is_pgc]):
            raise AssertionError('A cue branch altered an ordinary grid slot')
    if not pop.n_pgc and not all(torch.equal(code,codes['ordinary']) for code in codes.values()):
        raise AssertionError('Grid pulse codes are not exact shams')
    rt, slot = context['runtime'],context['selected_slot']
    source = CodeSource(cfg,pop,device)
    direct = source(*(rt['obs'][key].to(device) for key in ('pos','dir','vel')))[slot]
    if not torch.allclose(direct,codes['correct'][0],atol=1e-5,rtol=1e-4):
        raise AssertionError('Native-angle current cue does not match the actual recipient')
    edits, valid, audit = make_edits(family,[0])
    records, baselines, validations = [], {}, {}
    for tape in range(tapes):
        for edit in EDITS:
            if not valid[edit][0]:
                for cue in CUES:
                    records.append(dict(tape=tape,edit=edit,cue=cue,qualified=False,
                                        reason='Prespecified random normal violates bounds or is undefined'))
                continue
            for cue in CUES:
                result = native_fork(policy,cfg,context,device=device,tape=tape,
                    context_index=context_index,horizon=horizon,deadline=deadline,
                    edit=None if edit=='none' else edits[edit][0],
                    pulse=None if cue=='none' else family['current_codes'][cue][0])
                if edit=='none' and cue=='none':
                    baselines[tape] = result
                    if tape==0:
                        expected = context.get('native_continuation',[])[:horizon]
                        check = compare_native_trace(result['trace'],expected)
                        if not expected: check.update(qualified=False,reason='Missing independent native continuation')
                        validations['no_op_native_continuation'] = check
                base = baselines[tape]
                count = min(len(result['other_slot_hashes']),len(base['other_slot_hashes']))
                other_identical = result['other_slot_hashes'][:count] == base['other_slot_hashes'][:count]
                if not other_identical:
                    raise AssertionError('Intervention changed an unedited batch slot')
                result.update(tape=tape,edit=edit,cue=cue,qualified=True,
                              other_slots_identical_for_compared_steps=count)
                records.append(result)
    if cfg['cond']=='gc100':
        for tape in range(tapes):
            for edit in EDITS:
                rows = [r for r in records if r['tape']==tape and r['edit']==edit and r['cue']!='none' and r['qualified']]
                if not rows: continue
                for row in rows[1:]:
                    check = compare_native_trace(row['trace'],rows[0]['trace'],exact_states=True)
                    if not check['qualified']: raise AssertionError('Grid pulse sham failed')
        validations['grid_pulses_exact_sham'] = True
    else:
        validations['grid_pulses_exact_sham'] = None
    contrasts = {}
    for other in ('ordinary','wrong'):
        alternatives = ('ordinary',) if other=='ordinary' else ('wrong_plus','wrong_minus')
        effects = {}
        for edit in ('none','plus30','minus30'):
            paired = []
            for tape in range(tapes):
                rows = {r['cue']:r for r in records if r['edit']==edit and r['tape']==tape and r['qualified']}
                if all(cue in rows for cue in (*alternatives,'correct')):
                    paired.append(rows['correct']['score']-np.mean([rows[c]['score'] for c in alternatives]))
            effects[edit] = float(np.mean(paired)) if len(paired)==tapes else None
        donor = None if any(effects[e] is None for e in ('plus30','minus30')) else (effects['plus30']+effects['minus30'])/2
        contrasts['correct_minus_'+other] = dict(per_edit=effects,donor_mean=donor,
            donor_minus_no_edit=None if donor is None or effects['none'] is None else donor-effects['none'])
    qualified = validations.get('no_op_native_continuation',{}).get('qualified',False)
    return dict(complete=True,validation_qualified=qualified,validations=validations,
        context_index=context_index,episode_id=context['episode_id'],edits=audit,
        tapes=tapes,horizon=horizon,contrasts=contrasts,records=records,
        scope='Selected native anchors with fixed training weights; tapes/branches are not independent trained agents.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--states',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--activity',type=Path,help='Authenticated activity_analysis directory')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-seconds',type=float,default=7200)
    args=parser.parse_args(argv)
    from analyze_simple_memory_activity import authenticate,sha,PROTOCOL
    from collect_simple_memory_states import load_final
    from simple_memory_family import build_family
    from simple_memory_behavior_readout import analyze_behavior_readouts
    from threadpoolctl import threadpool_limits
    states=args.states or args.run/'memory_states'
    out=args.out or args.run/'intervention_analysis'
    if out.exists() and any(out.iterdir()): raise FileExistsError('Analysis output must be empty')
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['status']=='frozen' and protocol['source_sha256']
    assert 'scripts/simple_memory_interventions.py' in protocol['source_sha256']
    # Reuse full collection/model authentication; no policy execution occurs here.
    arrays,actor,cfg,proof=authenticate(args.run,states,PROTOCOL)
    del arrays,actor
    activity=args.activity or args.run/'activity_analysis'
    activity_summary=json.loads((activity/'summary.json').read_text())
    assert activity_summary['complete']
    assert activity_summary['provenance']['final_checkpoint_sha256']==proof['final_checkpoint_sha256']
    assert activity_summary['provenance']['analysis_protocol_sha256']==sha(PROTOCOL)
    assert activity_summary['numeric_artifact_sha256']==sha(activity/'activity_arrays.npz')
    assert activity_summary['provenance']['producer_sha256']==sha(ROOT/'scripts/analyze_simple_memory_activity.py')
    with np.load(activity/'activity_arrays.npz',allow_pickle=False) as archive:
        readouts={key:archive[key] for key in archive.files if '_readout_fold' in key}
    collection=json.loads((states/'summary.json').read_text())
    context_path=states/collection['contexts_file']['path']
    assert sha(context_path)==collection['contexts_file']['sha256']
    archived=torch.load(context_path,map_location='cpu',weights_only=False)
    assert archived['checkpoint_sha256']==proof['final_checkpoint_sha256']
    contexts=archived['contexts']
    assert len(contexts)<=4
    policy,cfg,_=load_final(args.run,args.device,args.threads)
    before=training.policy_hash(policy)
    out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.max_seconds
    summaries=[]
    for index,context in enumerate(contexts):
        if time.monotonic()>deadline: raise TimeoutError('Intervention deadline')
        family=build_family(policy,cfg,context,args.device)
        row=dict(index=index,episode_id=context['episode_id'],global_step=context['global_step'],
                 family=family['metadata'])
        if family['metadata']['qualified']:
            with threadpool_limits(limits=args.threads):
                cue,raw=cue_repair(policy,family,deadline=deadline)
            cue_path=out/f'context{index}_cue_arrays.npz'
            with cue_path.with_suffix('.npz.tmp').open('wb') as handle:
                np.savez_compressed(handle,**raw)
            cue_path.with_suffix('.npz.tmp').replace(cue_path)
            del raw
            behavior=behavioral_forks(policy,cfg,family,device=args.device,context_index=index,
                                     deadline=deadline)
            with threadpool_limits(limits=args.threads):
                behavior['held_out_angular_readouts']=analyze_behavior_readouts(behavior['records'],context,readouts)
            trace_path=out/f'context{index}_behavior_traces.pt'
            training.atomic_torch(trace_path,behavior)
            # Full lossless traces are archived separately; keep the report readable.
            behavior['records']=[{k:v for k,v in record.items() if k not in ('trace','other_slot_hashes')}
                                 for record in behavior['records']]
            row.update(qualified=behavior['validation_qualified'],cue_repair=cue,behavior=behavior,
                artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in (cue_path,trace_path)})
        else:
            row.update(qualified=False,reason='Native-prefix or family adapter qualification failed; no replacement')
        training.atomic_json(out/f'context{index}.json',row)
        summaries.append(row)
        print(json.dumps(dict(context=index,qualified=row['qualified'])),flush=True)
    assert training.policy_hash(policy)==before
    training.atomic_json(out/'summary.json',dict(complete=True,condition=cfg['cond'],training_seed=cfg['seed'],
        training_task_max_gap=cfg['task_max_gap'],created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        contexts=summaries,provenance=proof,policy_unchanged=True,
        activity_summary_sha256=sha(activity/'summary.json'),
        source_sha256={name:sha(ROOT/name) for name in protocol['source_sha256']},
        analysis_protocol_sha256=sha(PROTOCOL),
        scope='Prespecified cue/state causal interventions; geometry qualification is reported separately.'))


if __name__=='__main__': main()
