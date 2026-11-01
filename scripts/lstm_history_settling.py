"""Replace chronological recent history with repeated present-input updates.

Every decision starts h/c at zero, then applies the exact same current input
15 or 60 times before choosing an action. The physical environment advances
only once. This is a constructed frozen-network controller, not real memory.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
import torch.nn.functional as F
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena, summarize_episodes, verify_codes
from lstm_history_horizon import CONDITIONS, WindowBank


def repeated_current_states(cell, inputs, repetitions):
    """Share each input projection; stop each batch at its requested depth."""
    assert len(inputs) == len(repetitions) and list(repetitions) == sorted(set(repetitions))
    n = len(inputs[0])
    assert all(len(x) == n for x in inputs)
    x = torch.cat(inputs)
    drive = F.linear(x, cell.weight_ih, cell.bias_ih)
    h = x.new_zeros(len(x), cell.hidden_size)
    c = torch.zeros_like(h)
    outputs = []
    target = 0
    for iteration in range(1, repetitions[-1]+1):
        gates = drive + F.linear(h, cell.weight_hh, cell.bias_hh)
        i, f, g, o = gates.chunk(4, -1)
        c = f.sigmoid()*c + i.sigmoid()*g.tanh()
        h = o.sigmoid()*c.tanh()
        if iteration == repetitions[target]:
            outputs.append((h[:n], c[:n]))
            target += 1
            if target == len(repetitions): break
            h, c, drive = h[n:], c[n:], drive[n:]
    return outputs


def make_sequence(cell):
    """Copy the cell into an equivalent native one-layer sequence module."""
    sequence = torch.nn.LSTM(cell.input_size, cell.hidden_size).to(
        device=cell.weight_ih.device, dtype=cell.weight_ih.dtype)
    with torch.no_grad():
        sequence.weight_ih_l0.copy_(cell.weight_ih)
        sequence.weight_hh_l0.copy_(cell.weight_hh)
        sequence.bias_ih_l0.copy_(cell.bias_ih)
        sequence.bias_hh_l0.copy_(cell.bias_hh)
    return sequence.eval().requires_grad_(False)


def sequence_current_hidden(sequence, inputs, repetitions):
    """The native sequence graph repeats each branch's own present input."""
    n = len(inputs[0])
    x = torch.cat(inputs)
    series = x[None].expand(repetitions[-1], -1, -1).contiguous()
    hidden, _ = sequence(series)
    return [hidden[count-1, j*n:(j+1)*n] for j, count in enumerate(repetitions)]


@torch.inference_mode()
def validate_settling(device):
    torch.manual_seed(3881)
    cell = torch.nn.LSTMCell(11, 7).to(device)
    inputs = [torch.randn(3, 11, device=device) for _ in range(3)]
    repeats = (1, 3, 15)
    actual = repeated_current_states(cell, inputs, repeats)
    sequence_hidden = sequence_current_hidden(make_sequence(cell), inputs, repeats)
    errors = []
    for x, count, output, selected_h in zip(inputs, repeats, actual, sequence_hidden):
        state = (x.new_zeros(3, 7), x.new_zeros(3, 7))
        for _ in range(count): state = cell(x, state)
        errors.extend(float((a-b).abs().max()) for a, b in zip(output, state))
        errors.append(float((selected_h-state[0]).abs().max()))
    assert max(errors) < 2e-6, max(errors)
    return max(errors)


@torch.inference_mode()
def validate_checkpoints(seeds, device, repetitions, backend):
    """Check repeated manual updates against native cells at trained scales."""
    checks = {}
    for seed in seeds:
        for condition in CONDITIONS:
            path = ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
            net = build_policy(json.loads(path.read_text()), device).eval()
            net.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)['policy'])
            source = ROOT/f'results/lstm_navigation_phases_1B/{condition}_s{seed}_action_states.npz'
            data = np.load(source)
            tensors = [torch.from_numpy(data[key][:8]).to(device) for key in
                       ('embedding', 'current_code', 'goal_code', 'previous_action', 'previous_reward')]
            x = net._concat(*tensors, None)
            actual = repeated_current_states(net.lstm, [x for _ in repetitions], repetitions)
            selected = (sequence_current_hidden(make_sequence(net.lstm), [x for _ in repetitions], repetitions)
                        if backend == 'sequence' else [state[0] for state in actual])
            state_error, probability_error, selected_hidden_error = 0., 0., 0.
            for count, output, selected_h in zip(repetitions, actual, selected):
                expected = net.zero_state(len(x), device)
                for _ in range(count): expected = net.lstm(x, expected)
                state_error = max(state_error, *(float((a-b).abs().max()) for a, b in zip(output, expected)))
                selected_hidden_error = max(selected_hidden_error, float((selected_h-expected[0]).abs().max()))
                probability_error = max(probability_error, float((net.pi(selected_h).softmax(1)-net.pi(expected[0]).softmax(1)).abs().max()))
            assert max(state_error, selected_hidden_error) < 2e-4 and probability_error < 2e-5, (condition, seed, state_error, selected_hidden_error, probability_error)
            checks[f'{condition}/s{seed}'] = dict(max_state_error=state_error,
                selected_backend_max_hidden_error=selected_hidden_error, max_probability_error=probability_error,
                snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return checks


@torch.inference_mode()
def run(seed, a):
    started = time.perf_counter()
    torch.manual_seed(a.eval_seed)
    if a.device == 'mps': torch.mps.manual_seed(a.eval_seed)
    nets, configs, hashes = {}, {}, {}
    for condition in CONDITIONS:
        path = ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
        configs[condition] = json.loads(path.read_text())
        checkpoint = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        assert checkpoint['frames'] == 1_000_000_000
        hashes[condition] = hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()
        nets[condition] = build_policy(configs[condition], a.device).eval().requires_grad_(False)
        nets[condition].load_state_dict(checkpoint['policy'])
    cfg = configs['gc100']
    assert all(configs['pgc33'][key] == cfg[key] for key in ('arena_half', 'episode_len',
        'goal_radius', 'n_cells', 'nh', 'vision', 'action_repeat', 'v_max', 'rot_max',
        'reward_clip', 'img_noise', 'randomize_textures'))
    population = build_population(configs['pgc33'], a.device)
    variants = ('full', 'k1') + tuple(f'repeat{k}' for k in a.repetitions)
    n, nv = a.n_envs, len(variants)
    per_model, total = n*nv, n*nv*2
    env = MatchedArena(cfg, n, nv*2, a.eval_seed, a.device)
    obs = env.reset_all()
    full = {condition: net.zero_state(per_model, a.device) for condition, net in nets.items()}
    k1 = {condition: WindowBank(net.lstm, 1, n, a.device) for condition, net in nets.items()}
    sequences = {condition: make_sequence(net.lstm) for condition, net in nets.items()} if a.backend == 'sequence' else {}
    goals = torch.zeros(total, 512, device=a.device)
    pa = torch.full((total,), -1, dtype=torch.long, device=a.device)
    pr = torch.zeros(total, device=a.device)
    generator = torch.Generator(device=a.device).manual_seed(a.eval_seed+1)
    rows = {f'{condition}/{variant}': [] for condition in CONDITIONS for variant in variants}
    step, last = 0, started
    while len(rows['gc100/full']) < a.episodes:
        known = env.n_goals > 0
        policy = []
        for ci, condition in enumerate(CONDITIONS):
            net = nets[condition]
            sl = slice(ci*per_model, (ci+1)*per_model)
            code = population.activity(obs['pos'][sl], obs['dir'][sl], obs['vel'][sl], shift_scale=ci)
            x = net._concat(net.embed(obs['rgb'][sl]), code, goals[sl], pa[sl], pr[sl], None)
            full[condition] = net.lstm(x, full[condition])
            outputs = [net.pi(full[condition][0][:n]), net.pi(k1[condition].step(x[n:2*n], step)[0])]
            inputs = [x[j*n:(j+1)*n] for j in range(2, nv)]
            repeated = (sequence_current_hidden(sequences[condition], inputs, a.repetitions)
                        if a.backend == 'sequence' else
                        [state[0] for state in repeated_current_states(net.lstm, inputs, a.repetitions)])
            outputs.extend(net.pi(hidden) for hidden in repeated)
            policy.append(torch.cat(outputs))
        logits = torch.cat(policy)
        logits = torch.where(known[:, None], logits, logits[:n].repeat(nv*2, 1))
        uniform = torch.rand(n, generator=generator, device=a.device).repeat(nv*2)
        action = (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        obs, reward, done, info = env.step(action)
        hit = info['hit']
        if bool(hit.any()):
            for ci, condition in enumerate(CONDITIONS):
                sl = slice(ci*per_model, (ci+1)*per_model)
                value = population.activity(info['hit_pos'][sl], info['hit_dir'][sl], info['hit_vel'][sl], shift_scale=ci)
                goals[sl] = torch.where(hit[sl, None], value, goals[sl])
        for ci, condition in enumerate(CONDITIONS):
            local_done = done[ci*per_model:(ci+1)*per_model]
            full[condition] = tuple(state*(~local_done)[:, None] for state in full[condition])
            k1[condition].reset(local_done[n:2*n])
        pa, pr = action, reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            goals *= (~done)[:, None]
            pa = torch.where(done, torch.full_like(pa, -1), pa)
            pr *= ~done
            stats = {key: value.cpu().tolist() for key, value in info['ep_stats'].items()}
            for i, slot in enumerate(stats['env']):
                group = slot//n
                key = f'{CONDITIONS[group//nv]}/{variants[group%nv]}'
                rows[key].append(dict(slot=slot%n, episode=step//cfg['episode_len'],
                    **{name: stats[name][i] for name in ('ret', 'n_goals', 'first_goal_t', 'path_eff', 'path_eff_n')}))
        step += 1
        if time.perf_counter()-last > 25:
            print(f'settling s{seed}: {step}/{cfg["episode_len"]*(a.episodes//n)}; {time.perf_counter()-started:.0f}s', flush=True)
            last = time.perf_counter()
        if a.max_steps and step >= a.max_steps:
            print(f'BENCHMARK: {step} steps in {time.perf_counter()-started:.2f}s', flush=True)
            return
    first = [row['first_goal_t'] for row in rows['gc100/full']]
    assert all([row['first_goal_t'] for row in records] == first for records in rows.values())
    full_exact, k1_exact = None, None
    if (a.episodes, a.n_envs, a.eval_seed) == (128, 64, 737373):
        old = json.loads((ROOT/f'results/lstm_policy_hybrid_1B/s{seed}.json').read_text())
        for condition, variant in [('gc100', 'grid'), ('pgc33', 'pgc_after_discovery')]:
            assert rows[f'{condition}/full'] == old['variants'][variant]['rows'], (seed, condition, 'full baseline mismatch')
        full_exact = True
        reference = ROOT/f'results/lstm_history_horizon_1B/s{seed}.json'
        if reference.exists():
            prior = json.loads(reference.read_text())
            assert (prior['episodes'], prior['n_envs'], prior['eval_seed']) == (a.episodes, a.n_envs, a.eval_seed)
            for condition in CONDITIONS:
                assert rows[f'{condition}/k1'] == prior['variants'][f'{condition}/k1']['rows'], (seed, condition, 'K1 mismatch')
            k1_exact = True
    for condition in CONDITIONS:
        assert hashlib.sha256((ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt').read_bytes()).hexdigest() == hashes[condition]
    result = dict(seed=seed, episodes=a.episodes, n_envs=n, eval_seed=a.eval_seed, device=a.device,
        repetitions=a.repetitions, backend=a.backend, checkpoint_sha256=hashes,
        checks=dict(first_goal_times_identical=True, both_full_baselines_exact=full_exact,
                    both_k1_baselines_exact=k1_exact, checkpoint_hashes_unchanged=True),
        variants={key: summarize_episodes(records, cfg['episode_len']) for key, records in rows.items()},
        elapsed_seconds=time.perf_counter()-started)
    (a.out/f's{seed}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'DONE settling s{seed}: '+', '.join(f'{key}={value["mean_score"]:.2f}' for key, value in result['variants'].items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--n-envs', type=int, default=64)
    parser.add_argument('--eval-seed', type=int, default=737373)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 9)))
    parser.add_argument('--repetitions', nargs='+', type=int, default=[15, 60])
    parser.add_argument('--backend', choices=['loop', 'sequence'], default='loop')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'results/lstm_history_settling_1B')
    args = parser.parse_args()
    assert args.episodes % args.n_envs == 0
    assert args.repetitions == sorted(set(args.repetitions)) and min(args.repetitions) > 1
    torch.set_num_threads(2)
    args.out.mkdir(exist_ok=True, parents=True)
    error = validate_settling(args.device)
    trained = validate_checkpoints(args.seeds, args.device, args.repetitions, args.backend)
    print(f'VALIDATED repeated inputs: toy max error {error:.3g}; trained max state error '+
          f'{max(record["max_state_error"] for record in trained.values()):.3g}; selected-backend hidden error '+
          f'{max(record["selected_backend_max_hidden_error"] for record in trained.values()):.3g}', flush=True)
    protocol = dict(seeds=args.seeds, episodes=args.episodes, n_envs=args.n_envs, eval_seed=args.eval_seed,
        repetitions=args.repetitions, backend=args.backend, variants=['full', 'k1']+[f'repeat{k}' for k in args.repetitions],
        toy_replay_max_error=error, trained_replay_checks=trained,
        analysis_status='Exploratory follow-up motivated by the seed1 horizon result (large K1 loss, recovery at K15). The four comparisons below were specified before any settling scores were observed; this is not an overall preregistered study.',
        primary='Four paired score-gap contrasts for default depths: repeat15 minus K1, repeat60 minus K1, repeat15 minus chronological K15, repeat60 minus chronological K60. Holm t and exhaustive sign flips separately; display all per-model scores and paired population-gap intervals. Nonsignificance is not equivalence.',
        design='All branches follow common full Grid controller until first discovery. Each model then controls its own path. Every repeat branch zeros h/c at each decision and applies that decision input identically 15 or 60 times; only the final output selects one physical action. No chronological input sequence is replayed. All frozen weights, native spatial codes, external saved goal, previous action/reward and image remain available.',
        retained='Repeating the full current input also repeats its previous-action and previous-reward channels, including arrival reward just after pickup. Those values are not replaced or suppressed.',
        caveats=['Repeated current inputs are a constructed, potentially unfamiliar controller; success does not prove the native model never uses history.',
                 'Recovery would show that the advantage can operate without chronological recent inputs under this construction. Failure remains ambiguous between missing useful history and abnormal state initialization/updates.',
                 'Both h/c are recomputed each physical decision, but the external saved goal and previous action/reward still contain past information.',
                 'Repeating inputs changes internal computation depth per physical action and may itself create useful state; it does not identify the computation learned during training.',
                 'Full PGC begins from common Grid discovery experience. No neural states transfer between models.'],
        code_identity={seed: verify_codes(seed) for seed in args.seeds})
    if args.validate_only:
        print(json.dumps(trained, indent=2), flush=True)
        return
    if not args.max_steps: (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for seed in args.seeds:
        destination = args.out/f's{seed}.json'
        if destination.exists() and not args.max_steps:
            previous = json.loads(destination.read_text())
            assert (previous['episodes'], previous['n_envs'], previous['eval_seed'], previous['repetitions'], previous['backend']) == (args.episodes, args.n_envs, args.eval_seed, args.repetitions, args.backend)
            print(f'Using completed settling s{seed}', flush=True)
        else:
            run(seed, args)


if __name__ == '__main__': main()
