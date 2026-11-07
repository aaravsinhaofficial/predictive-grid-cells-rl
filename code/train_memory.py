"""Native full-policy A2C with sensory blackouts and complete restart state.

Run with ``python -m code.train_memory``. The native task, network and A2C
loss are unchanged; code.memory_task owns the observation-availability rule.
SIGTERM/SIGINT finish the current optimizer update, atomically save its full
runtime, and return nonzero. There is no weights-only restart or extension.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from code.common import add_shared_args, resolve_cond, build_env, build_policy, build_population, CodeSource
from code.memory_task import MemoryClock, gate_observation, max_gap_at

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('code/train_memory.py', 'code/memory_task.py', 'code/common.py',
           'code/env.py', 'code/nets.py', 'code/codes.py', 'code/__init__.py', 'code/train.py')
OPERATIONAL = {'out', 'resume', 'device', 'max_seconds', 'protocol', 'threads'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n'); os.replace(tmp, path)


def atomic_torch(path, data):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp); os.replace(tmp, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return value


def policy_hash(policy):
    h = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); add_shared_args(ap)
    ap.set_defaults(policy='lstm', conv_arch='si', nh=256, encoder_norm='layernorm')
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--frames', type=int, default=1_000_000_000)
    ap.add_argument('--n_envs', type=int, default=32); ap.add_argument('--rollout', type=int, default=100)
    for name, default in [('gamma', .99), ('lr', 3e-4), ('rms_alpha', .99), ('rms_eps', 1e-5),
                          ('entropy', 8e-5), ('value_coef', .5), ('grad_clip', 40.), ('reward_clip', 1.)]:
        ap.add_argument('--' + name, type=float, default=default)
    ap.add_argument('--task_max_gap', type=int, default=60)
    ap.add_argument('--task_ramp_frames', type=int, default=200_000_000)
    ap.add_argument('--task_visible_steps', type=int, default=15)
    ap.add_argument('--clock_seed', type=int, default=None)
    ap.add_argument('--ckpt_every', type=int, default=100_000_000)
    ap.add_argument('--resume_every', type=int, default=200)
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tf32', type=int, default=0)
    ap.add_argument('--max_seconds', type=float, default=float('inf'))
    ap.add_argument('--protocol', default=None)
    ap.add_argument('--resume', action='store_true')
    cfg = resolve_cond(vars(ap.parse_args(argv)))
    if cfg['clock_seed'] is None: cfg['clock_seed'] = cfg['seed'] * 1000 + 2
    if cfg['cond'] not in ('gc100', 'pgc33') or cfg['policy'] != 'lstm' or cfg['extra_input'] != 'none':
        ap.error('This trainer supports native gc100/pgc33 LSTM with no extra inputs')
    if cfg['code_noise'] != 0 or cfg['code_mode'] != 'full' or cfg['n_cells'] != 512:
        ap.error('Native noiseless 512-cell population lookup required')
    if min(cfg['n_envs'], cfg['rollout'], cfg['frames'], cfg['resume_every'], cfg['ckpt_every']) < 1:
        ap.error('Positive budgets and sizes required')
    if cfg['task_max_gap'] < 0 or cfg['task_ramp_frames'] <= 0 or cfg['task_visible_steps'] < 1:
        ap.error('Invalid memory-clock settings')
    if cfg['frames'] % (cfg['n_envs'] * cfg['action_repeat']):
        ap.error('Frame endpoint must contain complete parallel native steps')
    return cfg


def configure(cfg):
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing a silent CPU run')
    if device.type not in ('cpu', 'cuda'): raise ValueError('Exact resume supports CPU/CUDA')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = bool(cfg['tf32'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    return device


def fresh_training(cfg):
    device = configure(cfg)
    population = build_population(cfg, device)
    env = build_env(cfg, cfg['n_envs'], cfg['seed'] * 1000 + 1, device, stagger=True)
    policy = build_policy(cfg, device); policy.train()
    assert all(p.requires_grad for p in policy.parameters())
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    runtime = dict(env=env, obs=env.reset_all(), state=policy.zero_state(cfg['n_envs'], device),
        goal=torch.zeros(cfg['n_envs'], cfg['n_cells'], device=device),
        pa=torch.full((cfg['n_envs'],), -1, dtype=torch.long, device=device),
        pr=torch.zeros(cfg['n_envs'], device=device),
        clock=MemoryClock(cfg['n_envs'], device, cfg['clock_seed'], visible_steps=cfg['task_visible_steps']),
        counters=dict(frames=0, updates=0, episodes=0, decisions=0, blind_decisions=0, goals=0, blind_goals=0),
        recent_returns=[])
    return policy, optimizer, CodeSource(cfg, population, device), runtime


def capture_runtime(runtime):
    env = runtime['env']; device = env.device
    return dict(environment_tensors={k: cpu_tree(v) for k, v in vars(env).items() if isinstance(v, torch.Tensor)},
        environment_scalars={k: v for k, v in vars(env).items() if isinstance(v, (str, bool, int, float))},
        environment_rng=env.gen.get_state().cpu().clone(), device_type=device.type,
        obs=cpu_tree(runtime['obs']), state=cpu_tree(runtime['state']), goal=cpu_tree(runtime['goal']),
        pa=cpu_tree(runtime['pa']), pr=cpu_tree(runtime['pr']), clock=runtime['clock'].state_dict(),
        counters=dict(runtime['counters']), recent_returns=list(runtime['recent_returns']),
        cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=torch.cuda.get_rng_state(device).cpu().clone() if device.type == 'cuda' else None,
        numpy_rng=np.random.get_state(), python_rng=random.getstate())


def restore_runtime(runtime, saved):
    env = runtime['env']; device = env.device
    assert saved['device_type'] == device.type, 'Exact resume cannot switch numerical backend'
    assert saved['environment_tensors'].keys() == {k for k, v in vars(env).items() if isinstance(v, torch.Tensor)}
    for k, v in saved['environment_tensors'].items(): setattr(env, k, v.clone().to(device))
    for k, v in saved['environment_scalars'].items(): setattr(env, k, v)
    env.gen.set_state(saved['environment_rng'].cpu().clone())
    runtime['obs'] = {k: v.clone().to(device) for k, v in saved['obs'].items()}
    runtime['state'] = tuple(v.clone().to(device) for v in saved['state'])
    for k in ('goal', 'pa', 'pr'): runtime[k] = saved[k].clone().to(device)
    runtime['clock'].load_state_dict(saved['clock'])
    runtime['counters'] = dict(saved['counters']); runtime['recent_returns'] = list(saved['recent_returns'])
    torch.set_rng_state(saved['cpu_rng'].cpu().clone())
    if device.type == 'cuda': torch.cuda.set_rng_state(saved['cuda_rng'].cpu().clone(), device)
    np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])


def collect_rollout(policy, source, runtime, cfg, length):
    env, clock = runtime['env'], runtime['clock']; n = cfg['n_envs']
    roll = {k: [] for k in ('rgb', 'code', 'goal', 'pa', 'pr', 'a', 'r', 'd', 'blind', 'hit', 'actor_logits', 'actor_values')}
    state0 = tuple(v.detach().clone() for v in runtime['state']); episodes = []
    with torch.no_grad():
        for _ in range(length):
            obs = runtime['obs']; blind = clock.mask()
            rgb, code = gate_observation(obs['rgb'], source(obs['pos'], obs['dir'], obs['vel']), blind)
            logits, value, state = policy.step(rgb, code, runtime['goal'], runtime['pa'], runtime['pr'], runtime['state'])
            action = torch.distributions.Categorical(logits=logits, validate_args=False).sample()
            for k, v in dict(rgb=rgb, code=code, goal=runtime['goal'], pa=runtime['pa'], pr=runtime['pr'],
                             a=action, blind=blind, actor_logits=logits, actor_values=value).items(): roll[k].append(v)
            # Always run native rendering/image-noise draws, including in darkness.
            obs, raw, done, info = env.step(action)
            reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
            hit = info['hit']
            goal = runtime['goal']
            if bool(hit.any()):
                goal = torch.where(hit[:, None], source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']), goal)
            keep = (~done).float()
            runtime.update(obs=obs, state=tuple(v * keep[:, None] for v in state), goal=goal * keep[:, None],
                           pa=torch.where(done, torch.full_like(action, -1), action), pr=reward * keep)
            runtime['counters']['frames'] += n * cfg['action_repeat']
            clock.advance(hit, done, max_gap_at(runtime['counters']['frames'], cfg['task_ramp_frames'], cfg['task_max_gap']))
            for k, v in (('r', reward), ('d', done.float()), ('hit', hit)): roll[k].append(v)
            if info['ep_stats'] is not None:
                es = info['ep_stats']
                for j in range(len(es['env'])):
                    episodes.append({k: float(v[j]) for k, v in es.items()})
        obs = runtime['obs']; mask = clock.mask()
        rgb, code = gate_observation(obs['rgb'], source(obs['pos'], obs['dir'], obs['vel']), mask)
        _, bootstrap, _ = policy.step(rgb, code, runtime['goal'], runtime['pa'], runtime['pr'], runtime['state'])
        bootstrap_inputs = dict(rgb=rgb, code=code, blind=mask, goal=runtime['goal'], pa=runtime['pa'], pr=runtime['pr'])
    return roll, state0, bootstrap, bootstrap_inputs, episodes


def replay(policy, roll, state0):
    length, n = len(roll['rgb']), len(roll['pa'][0])
    embeddings = policy.embed(torch.cat(roll['rgb'], 0)).view(length, n, -1)
    state = state0; logits, values = [], []
    for t in range(length):
        ll, vv, state = policy.step_from_embedding(embeddings[t], roll['code'][t], roll['goal'][t], roll['pa'][t], roll['pr'][t], state)
        logits.append(ll); values.append(vv)
        state = tuple(v * (1 - roll['d'][t])[:, None] for v in state)
    return torch.stack(logits), torch.stack(values), state


def discounted_returns(rewards, dones, bootstrap, gamma):
    target = bootstrap; returns = []
    for reward, done in zip(reversed(rewards), reversed(dones)):
        target = reward + gamma * (1 - done) * target
        returns.append(target)
    return torch.stack(returns[::-1])


def train_update(policy, optimizer, source, runtime, cfg, length=None, return_trace=False):
    length = cfg['rollout'] if length is None else int(length)
    roll, state0, boot, bootstrap_inputs, episodes = collect_rollout(policy, source, runtime, cfg, length)
    returns = discounted_returns(roll['r'], roll['d'], boot, cfg['gamma'])
    logits, values, _ = replay(policy, roll, state0)
    online = torch.stack(roll['actor_logits']); online_v = torch.stack(roll['actor_values'])
    # Different CNN batch sizes can have small floating-point discrepancies.
    torch.testing.assert_close(logits.detach(), online, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(values.detach(), online_v, rtol=1e-4, atol=1e-5)
    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
    advantage = (returns - values).detach()
    policy_loss = -(dist.log_prob(torch.stack(roll['a'])) * advantage).sum(0).mean()
    value_loss = .5 * (returns - values).square().sum(0).mean()
    entropy = dist.entropy().sum(0).mean()
    loss = policy_loss + cfg['value_coef'] * value_loss - cfg['entropy'] * entropy
    optimizer.zero_grad(set_to_none=True); loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite A2C loss')
    optimizer.step(); runtime['state'] = tuple(v.detach() for v in runtime['state'])
    counters = runtime['counters']; counters['updates'] += 1
    blind = torch.stack(roll['blind']); hits = torch.stack(roll['hit'])
    blind_n = int(blind.sum()); hit_n = int(hits.sum()); blind_hit_n = int((blind & hits).sum())
    counters['decisions'] += blind.numel(); counters['blind_decisions'] += blind_n
    counters['goals'] += hit_n; counters['blind_goals'] += blind_hit_n
    counters['episodes'] += len(episodes)
    runtime['recent_returns'] = (runtime['recent_returns'] + [r['ret'] for r in episodes])[-100:]
    metrics = dict(**counters, rollout_steps=length, loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()), entropy=float(entropy.detach()) / length, grad_norm=float(grad),
        replay_logit_max_error=float((logits.detach() - online).abs().max()),
        replay_value_max_error=float((values.detach() - online_v).abs().max()),
        batch_blind_fraction=blind_n / blind.numel(), cumulative_blind_fraction=counters['blind_decisions'] / counters['decisions'],
        batch_goals=hit_n, batch_blind_goals=blind_hit_n,
        max_gap=max_gap_at(counters['frames'], cfg['task_ramp_frames'], cfg['task_max_gap']),
        avg_return_100=float(np.mean(runtime['recent_returns'])) if runtime['recent_returns'] else None)
    if return_trace: return metrics, episodes, dict(roll=roll, state0=state0, bootstrap=boot, bootstrap_inputs=bootstrap_inputs, returns=returns)
    return metrics, episodes


def scientific_config(cfg):
    return {k: v for k, v in cfg.items() if k not in OPERATIONAL}


def provenance(cfg):
    sources = {p: sha(ROOT / p) for p in SOURCES}
    protocol_hash = sha(cfg['protocol']) if cfg['protocol'] else None
    if cfg['protocol']:
        protocol = json.loads(Path(cfg['protocol']).read_text())
        for name, expected in protocol.get('source_sha256', {}).items():
            assert sha(ROOT / name) == expected, f'Protocol source changed: {name}'
        for name, expected in protocol.get('training_config', {}).items():
            assert cfg[name] == expected, f'Protocol training setting changed: {name}'
        cohort = protocol.get('allowed_cohort')
        if cohort:
            assert cfg['cond'] in cohort['conditions'], 'Condition outside frozen cohort'
            assert cfg['seed'] in cohort['seeds'], 'Seed outside frozen cohort'
            assert cfg['task_max_gap'] in cohort['task_max_gaps'], 'Task outside frozen cohort'
            assert cfg['pgc_frac'] == cohort['pgc_fractions'][cfg['cond']], 'Population fraction changed'
            if cohort['code_seed_equals_seed']:
                assert cfg['code_seed'] == cfg['seed'], 'Map seed must match training seed'
            assert cfg['clock_seed'] == cfg['seed'] * cohort['clock_seed_multiplier'] + cohort['clock_seed_offset'], 'Clock seed changed'
    config = scientific_config(cfg)
    return dict(source_sha256=sources, protocol_sha256=protocol_hash,
                config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                deterministic_algorithms=True, cudnn_benchmark=False, cudnn_tf32=bool(cfg['tf32']), matmul_tf32=False,
                cudnn_version=torch.backends.cudnn.version(),
                cuda_device_name=torch.cuda.get_device_name(cfg['device']) if str(cfg['device']).startswith('cuda') else None,
                cuda_capability=list(torch.cuda.get_device_capability(cfg['device'])) if str(cfg['device']).startswith('cuda') else None)


def snapshot(policy, optimizer, runtime, cfg, proof):
    return dict(schema=1, policy=cpu_tree(policy.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
        runtime=capture_runtime(runtime), config=scientific_config(cfg), provenance=proof,
        frames=runtime['counters']['frames'], updates=runtime['counters']['updates'],
        policy_training=policy.training, policy_sha256=policy_hash(policy))


def restore(saved, policy, optimizer, runtime, cfg, proof):
    assert saved['schema'] == 1 and saved['config'] == scientific_config(cfg)
    assert saved['provenance'] == proof, 'Source/protocol/numerical runtime changed'
    policy.load_state_dict(saved['policy'], strict=True); optimizer.load_state_dict(cpu_tree(saved['optimizer']))
    policy.train(saved['policy_training']); restore_runtime(runtime, saved['runtime'])
    assert policy_hash(policy) == saved['policy_sha256']
    assert (saved['frames'], saved['updates']) == (runtime['counters']['frames'], runtime['counters']['updates'])


def main(argv=None):
    cfg = parse_args(argv); out = Path(cfg['out']); resume_path = out / 'resume.pt'
    if cfg['resume']:
        if not resume_path.is_file(): raise FileNotFoundError('Exact resume.pt required; no weights-only fallback')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError('New training requires an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    pending_signal = [0]
    def stop(signum, _frame): pending_signal[0] = signum
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    policy, opt, source, runtime = fresh_training(cfg); proof = provenance(cfg)
    if cfg['resume']:
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        restore(saved, policy, opt, runtime, cfg, proof)
    else:
        atomic_json(out/'config.json', dict(scientific_config(cfg), population=source.pop.describe()))
        atomic_json(out/'provenance.json', proof)
        atomic_torch(out/'initial.pt', snapshot(policy, opt, runtime, cfg, proof))
    attempt = len(list(out.glob('attempt_*.json')))
    record = dict(attempt=attempt, start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  restored_updates=runtime['counters']['updates'], restored_frames=runtime['counters']['frames'],
                  resume_sha256=sha(resume_path) if cfg['resume'] else None)
    if cfg['resume']:
        import shutil
        saved_input = out / f'resume_input_{attempt:03d}.pt'
        shutil.copyfile(resume_path, saved_input)
        record['resume_input'] = saved_input.name
    atomic_json(out/f'attempt_{attempt:03d}.json', record)
    # Every attempt keeps its raw log. Restored-update cutoffs identify rollback;
    # no earlier rows are deleted or silently counted twice.
    start = time.monotonic(); start_frames = runtime['counters']['frames']
    next_ckpt = (start_frames // cfg['ckpt_every'] + 1) * cfg['ckpt_every']
    status = 0
    with (out/f'metrics_{attempt:03d}.jsonl').open('x') as log, (out/f'episodes_{attempt:03d}.jsonl').open('x') as episodes_log:
        while runtime['counters']['frames'] < cfg['frames']:
            if pending_signal[0] or time.monotonic() - start >= cfg['max_seconds']:
                status = 128 + pending_signal[0] if pending_signal[0] else 124
                break
            left = (cfg['frames'] - runtime['counters']['frames']) // (cfg['n_envs'] * cfg['action_repeat'])
            metrics, episodes = train_update(policy, opt, source, runtime, cfg, min(cfg['rollout'], left))
            metrics.update(seconds=time.monotonic()-start, attempt=attempt)
            metrics['fps'] = (metrics['frames']-start_frames)/max(metrics['seconds'], 1e-9)
            log.write(json.dumps(metrics)+'\n'); log.flush()
            for row in episodes: episodes_log.write(json.dumps(dict(row, frames=metrics['frames'], update=metrics['updates']))+'\n')
            episodes_log.flush()
            if metrics['updates'] % 20 == 0: print(json.dumps(metrics), flush=True)
            if metrics['updates'] % cfg['resume_every'] == 0:
                atomic_torch(resume_path, snapshot(policy, opt, runtime, cfg, proof))
            if metrics['frames'] >= next_ckpt:
                atomic_torch(out/f'checkpoint_{metrics["frames"]:010d}.pt', snapshot(policy, opt, runtime, cfg, proof))
                next_ckpt = (metrics['frames']//cfg['ckpt_every']+1)*cfg['ckpt_every']
    if pending_signal[0] and not status: status = 128 + pending_signal[0]
    final = snapshot(policy, opt, runtime, cfg, proof); atomic_torch(resume_path, final)
    complete = runtime['counters']['frames'] == cfg['frames'] and not status
    record.update(complete=complete, exit_code=status, end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  seconds=time.monotonic()-start, final_frames=final['frames'], final_updates=final['updates'],
                  resume_sha256_after=sha(resume_path))
    atomic_json(out/f'attempt_{attempt:03d}.json', record)
    if complete:
        atomic_torch(out/'final.pt', final)
        atomic_json(out/'summary.json', dict(complete=True, **runtime['counters'], provenance=proof,
            final_sha256=sha(out/'final.pt'), final_policy_sha256=final['policy_sha256'],
            initial_sha256=sha(out/'initial.pt'), attempt_records=[p.name for p in sorted(out.glob('attempt_*.json'))]))
    else:
        atomic_json(out/f'interrupted_{attempt:03d}.json', record)
    print(json.dumps(record), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
