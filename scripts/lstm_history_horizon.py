"""Limit recurrent history while retaining native current and saved goal inputs.

All branches follow one grid controller until the first goal. Thereafter each
frozen model controls its own branch. Rolling windows are exact: K inputs,
starting from zero h/c; they are not periodic resets. No state crosses models.
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
import torch
import torch.nn.functional as F
import numpy as np
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena, summarize_episodes, verify_codes

VARIANTS = ('full', 'k1', 'k15', 'k60', 'pickup_reset')
CONDITIONS = ('gc100', 'pgc33')
HORIZONS = (1, 15, 60)


class WindowBank:
    """K restart cohorts; oldest cohort contains exactly the last K inputs."""
    def __init__(self, cell, k, n, device):
        self.cell, self.k = cell, k
        self.h = torch.zeros(k, n, cell.hidden_size, device=device)
        self.c = torch.zeros_like(self.h)

    def step(self, x, t):
        self.h[t % self.k] = 0
        self.c[t % self.k] = 0
        drive = F.linear(x, self.cell.weight_ih, self.cell.bias_ih)
        gates = drive[None] + F.linear(self.h, self.cell.weight_hh, self.cell.bias_hh)
        i, f, g, o = gates.chunk(4, -1)
        self.c = f.sigmoid() * self.c + i.sigmoid() * g.tanh()
        self.h = o.sigmoid() * self.c.tanh()
        selected = (t + 1) % self.k
        return self.h[selected], self.c[selected]

    def reset(self, done):
        self.h *= (~done)[None, :, None]
        self.c *= (~done)[None, :, None]


@torch.inference_mode()
def validate_bank(device):
    """Independent explicit replay, including asynchronous episode resets."""
    torch.manual_seed(10293)
    cell = torch.nn.LSTMCell(11, 7).to(device)
    x = torch.randn(19, 3, 11, device=device)
    errors = []
    for k in (1, 5, 25):
        bank = WindowBank(cell, k, 3, device)
        starts = [0, 0, 0]
        for t in range(len(x)):
            actual = bank.step(x[t], t)
            for b in range(3):
                state = (torch.zeros(1, 7, device=device), torch.zeros(1, 7, device=device))
                for j in range(max(starts[b], t-k+1), t+1):
                    state = cell(x[j, b:b+1], state)
                errors.extend(float((a[b:b+1]-s).abs().max()) for a, s in zip(actual, state))
            done = torch.tensor([t == 6, t == 10, False], device=device)
            bank.reset(done)
            for b, reset in enumerate(done.cpu().tolist()):
                if reset: starts[b] = t+1
    assert max(errors) < 2e-6, max(errors)
    return max(errors)


@torch.inference_mode()
def validate_checkpoints(seeds, device):
    """Verify shared-drive gates at the actual trained weights/input scales."""
    checks = {}
    for seed in seeds:
        for condition in CONDITIONS:
            path = ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
            net = build_policy(json.loads(path.read_text()), device).eval()
            net.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)['policy'])
            source = ROOT/f'results/lstm_navigation_phases_1B/{condition}_s{seed}_action_states.npz'
            data = np.load(source)
            tensors = [torch.from_numpy(data[key]).to(device) for key in
                       ('embedding', 'current_code', 'goal_code', 'previous_action', 'previous_reward')]
            x = net._concat(*tensors, None)
            bank = WindowBank(net.lstm, 1, len(x), device)
            expected = net.lstm(x, net.zero_state(len(x), device))
            actual = bank.step(x, 0)
            error = max(float((a-b).abs().max()) for a, b in zip(actual, expected))
            assert error < 2e-5, (condition, seed, error)
            checks[f'{condition}/s{seed}'] = dict(max_state_error=error,
                snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return checks


@torch.inference_mode()
def run(seed, a):
    started = time.perf_counter()
    torch.manual_seed(a.eval_seed)
    if a.device == 'mps': torch.mps.manual_seed(a.eval_seed)
    nets, configs, hashes = {}, {}, {}
    for c in CONDITIONS:
        path = ROOT / f'models/{c}/{c}_lstm_s{seed}_1000M.json'
        configs[c] = json.loads(path.read_text())
        ck = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        assert ck['frames'] == 1_000_000_000
        hashes[c] = hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()
        nets[c] = build_policy(configs[c], a.device).eval().requires_grad_(False)
        nets[c].load_state_dict(ck['policy'])
    cfg = configs['gc100']
    assert all(configs['pgc33'][key] == cfg[key] for key in ('arena_half', 'episode_len',
        'goal_radius', 'n_cells', 'nh', 'vision', 'action_repeat', 'v_max', 'rot_max',
        'reward_clip', 'img_noise', 'randomize_textures'))
    pop = build_population(configs['pgc33'], a.device)
    n, nv = a.n_envs, len(VARIANTS)
    per_model, total = n*nv, n*nv*2
    env = MatchedArena(cfg, n, nv*2, a.eval_seed, a.device)
    obs = env.reset_all()
    full = {c: net.zero_state(per_model, a.device) for c, net in nets.items()}
    pickup = {c: net.zero_state(n, a.device) for c, net in nets.items()}
    banks = {c: {k: WindowBank(net.lstm, k, n, a.device) for k in HORIZONS} for c, net in nets.items()}
    goals = torch.zeros(total, 512, device=a.device)
    pa = torch.full((total,), -1, dtype=torch.long, device=a.device)
    pr = torch.zeros(total, device=a.device)
    gen = torch.Generator(device=a.device).manual_seed(a.eval_seed+1)
    rows = {f'{c}/{v}': [] for c in CONDITIONS for v in VARIANTS}
    step, last = 0, started
    while len(rows['gc100/full']) < a.episodes:
        known = env.n_goals > 0
        policy = []
        for ci, c in enumerate(CONDITIONS):
            net = nets[c]
            sl = slice(ci*per_model, (ci+1)*per_model)
            code = pop.activity(obs['pos'][sl], obs['dir'][sl], obs['vel'][sl], shift_scale=ci)
            emb = net.embed(obs['rgb'][sl])
            x = net._concat(emb, code, goals[sl], pa[sl], pr[sl], None)
            full[c] = net.lstm(x, full[c])
            out = [net.pi(full[c][0][:n])]
            for j, k in enumerate(HORIZONS, 1):
                state = banks[c][k].step(x[j*n:(j+1)*n], step)
                out.append(net.pi(state[0]))
            pickup[c] = net.lstm(x[4*n:5*n], pickup[c])
            out.append(net.pi(pickup[c][0]))
            policy.append(torch.cat(out))
        logits = torch.cat(policy)
        # One common discovery path, including identical sampled actions.
        logits = torch.where(known[:, None], logits, logits[:n].repeat(nv*2, 1))
        uniform = torch.rand(n, generator=gen, device=a.device).repeat(nv*2)
        action = (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        obs, reward, done, info = env.step(action)
        hit = info['hit']
        if bool(hit.any()):
            for ci, c in enumerate(CONDITIONS):
                sl = slice(ci*per_model, (ci+1)*per_model)
                v = pop.activity(info['hit_pos'][sl], info['hit_dir'][sl], info['hit_vel'][sl], shift_scale=ci)
                goals[sl] = torch.where(hit[sl, None], v, goals[sl])
        for ci, c in enumerate(CONDITIONS):
            sl = slice(ci*per_model, (ci+1)*per_model)
            local_done = done[sl]
            full[c] = tuple(s*(~local_done)[:, None] for s in full[c])
            for j, k in enumerate(HORIZONS, 1): banks[c][k].reset(local_done[j*n:(j+1)*n])
            clear = (hit | done)[ci*per_model+4*n:ci*per_model+5*n]
            pickup[c] = tuple(s*(~clear)[:, None] for s in pickup[c])
        pa, pr = action, reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            goals *= (~done)[:, None]
            pa = torch.where(done, torch.full_like(pa, -1), pa)
            pr *= ~done
            es = {k: v.cpu().tolist() for k, v in info['ep_stats'].items()}
            for i, slot in enumerate(es['env']):
                group = slot//n
                key = f'{CONDITIONS[group//nv]}/{VARIANTS[group%nv]}'
                rows[key].append(dict(slot=slot%n, episode=step//cfg['episode_len'],
                    **{k: es[k][i] for k in ('ret', 'n_goals', 'first_goal_t', 'path_eff', 'path_eff_n')}))
        step += 1
        if time.perf_counter()-last > 25:
            print(f'history s{seed}: {step}/{cfg["episode_len"]*(a.episodes//n)}; {time.perf_counter()-started:.0f}s', flush=True)
            last = time.perf_counter()
        if a.max_steps and step >= a.max_steps:
            print(f'BENCHMARK: {step} steps in {time.perf_counter()-started:.2f}s', flush=True)
            return
    first = [r['first_goal_t'] for r in rows['gc100/full']]
    assert all([r['first_goal_t'] for r in row] == first for row in rows.values())
    exact = None
    if (a.episodes, a.n_envs, a.eval_seed) == (128, 64, 737373):
        old = json.loads((ROOT/f'results/lstm_policy_hybrid_1B/s{seed}.json').read_text())
        for c, v in [('gc100', 'grid'), ('pgc33', 'pgc_after_discovery')]:
            assert rows[f'{c}/full'] == old['variants'][v]['rows'], (seed, c, 'full baseline mismatch')
        exact = True
    for c in CONDITIONS:
        assert hashlib.sha256((ROOT/f'models/{c}/{c}_lstm_s{seed}_1000M.pt').read_bytes()).hexdigest() == hashes[c]
    result = dict(seed=seed, episodes=a.episodes, n_envs=n, eval_seed=a.eval_seed, device=a.device,
        checkpoint_sha256=hashes,
        checks=dict(first_goal_times_identical=True, both_full_baselines_exact=exact, checkpoint_hashes_unchanged=True),
        variants={key: summarize_episodes(row, cfg['episode_len']) for key, row in rows.items()},
        elapsed_seconds=time.perf_counter()-started)
    (a.out/f's{seed}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'DONE history s{seed}: '+', '.join(f'{k}={v["mean_score"]:.2f}' for k, v in result['variants'].items()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='mps')
    p.add_argument('--episodes', type=int, default=128)
    p.add_argument('--n-envs', type=int, default=64)
    p.add_argument('--eval-seed', type=int, default=737373)
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 9)))
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--out', type=Path, default=ROOT/'results/lstm_history_horizon_1B')
    a = p.parse_args()
    assert a.episodes % a.n_envs == 0
    torch.set_num_threads(2)
    a.out.mkdir(exist_ok=True, parents=True)
    error = validate_bank(a.device)
    if not a.max_steps:
        (a.out/'checkpoint_gate_validation.json').write_text(json.dumps(validate_checkpoints(a.seeds, a.device), indent=2)+'\n')
    protocol = dict(seeds=a.seeds, episodes=a.episodes, n_envs=a.n_envs, eval_seed=a.eval_seed,
        variants=VARIANTS, horizons_inputs=HORIZONS, bank_replay_max_error=error,
        primary='Four paired differences in population score gap versus full history: K1, K15, K60, pickup_reset. Holm t and exhaustive sign flips separately. Other contrasts descriptive/exploratory.',
        design='All branches follow common grid/full controller until first goal, then each model/variant controls its own trajectory. Full branches reproduce existing controller hybrid baselines. Exact rolling windows preserve every past input as originally experienced, including goal updates and previous action/reward. Both h and c are truncated. Pickup-reset clears both only at each reward/teleport. Native spatial inputs and externally saved goal remain available.',
        caveats=['K1 removes carried LSTM state, not external goal memory or explicit previous action/reward.',
                 'Zero initial states with a known goal are outside ordinary training resets; performance loss alone is not proof of useful memory.',
                 'PGC full branch shares grid discovery experience, not native PGC exploration.',
                 'No cross-model state transfer; same architecture does not mean same learned weights.',
                 'A finite-history failure does not establish a special attractor or link PCA offsets to reward.'],
        code_identity={s: verify_codes(s) for s in a.seeds})
    if not a.max_steps: (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for seed in a.seeds:
        if (a.out/f's{seed}.json').exists() and not a.max_steps:
            old = json.loads((a.out/f's{seed}.json').read_text())
            assert (old['episodes'], old['n_envs'], old['eval_seed']) == (a.episodes, a.n_envs, a.eval_seed)
            print(f'Using completed history s{seed}', flush=True)
        else: run(seed, a)


if __name__ == '__main__': main()
