"""Private CUDA correctness/throughput fixtures; never a research training run.

No cloud resources are created. The two discarded full-policy benchmarks use
seed 90731, a one-frame curriculum, and forced-known clocks to exercise the
blackout workload without waiting for an untrained agent to find the goal.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import platform
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from code import train_memory as train  # Sets deterministic CUDA env before torch import.
import numpy as np
import torch

PRIVATE_SEED = 90731
BENCHMARK_FRAMES = 2_000_000
WARMUP_UPDATES = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_tree(expected, actual, path='root'):
    """Check every serialized tensor/RNG/config leaf, without numeric tolerance."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert expected.dtype == actual.dtype and expected.shape == actual.shape, path
        assert torch.equal(expected, actual), f'{path}: tensor values differ'
        return 1
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), path
        assert expected.dtype == actual.dtype and np.array_equal(expected, actual), path
        return 1
    assert type(expected) is type(actual), f'{path}: types differ'
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), f'{path}: keys differ'
        return sum(exact_tree(value, actual[key], f'{path}.{key}') for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), f'{path}: lengths differ'
        return sum(exact_tree(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(expected, actual)))
    assert expected == actual, f'{path}: values differ'
    return 1


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Global technical-preflight time budget exhausted')


def categorical_equivalence(device):
    """Validate only the removed distribution argument checks, preserving RNG."""
    devices = list(range(torch.cuda.device_count())) if torch.device(device).type == 'cuda' else []
    before = dict(cpu=torch.get_rng_state().clone(), cuda=[torch.cuda.get_rng_state(i).clone() for i in devices])
    cases = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(PRIVATE_SEED)
        for shape in ((32, 6), (100, 32, 6)):
            base = torch.randn(shape, device=device)
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = [torch.cuda.get_rng_state(i).clone() for i in devices]
            outputs = []
            for optimized in (False, True):
                torch.set_rng_state(cpu_rng)
                for index, state in zip(devices, cuda_rng): torch.cuda.set_rng_state(state, index)
                logits = base.clone().requires_grad_(True)
                distribution = (torch.distributions.Categorical(logits=logits, validate_args=False)
                                if optimized else torch.distributions.Categorical(logits=logits))
                actions = distribution.sample()
                log_prob, entropy = distribution.log_prob(actions), distribution.entropy()
                (log_prob.sum() + .13 * entropy.sum()).backward()
                outputs.append(train.cpu_tree(dict(normalized_logits=distribution.logits, probabilities=distribution.probs,
                    actions=actions, log_prob=log_prob, entropy=entropy, gradient=logits.grad,
                    cpu_rng=torch.get_rng_state(), cuda_rng=[torch.cuda.get_rng_state(i) for i in devices])))
            leaves = exact_tree(outputs[0], outputs[1], 'categorical')
            cases.append(dict(shape=list(shape), bitwise_equal=True, compared_leaves=leaves))
    exact_tree(before, dict(cpu=torch.get_rng_state(), cuda=[torch.cuda.get_rng_state(i) for i in devices]), 'global_rng')
    return dict(complete=True, cases=cases, global_rng_restored=True, private_seed=PRIVATE_SEED,
                compared=['normalized logits', 'probabilities', 'sampled actions', 'log probability', 'entropy',
                          'logit gradients', 'CPU RNG', 'CUDA RNG'], scientific_settings_changed=False)


def fixture_config(out, condition, device, n, rollout, frames):
    return train.parse_args(['--out', str(out), '--cond', condition,
        '--seed', str(PRIVATE_SEED), '--code_seed', str(PRIVATE_SEED),
        '--device', device, '--threads', '2', '--n_envs', str(n),
        '--rollout', str(rollout), '--frames', str(frames),
        '--task_max_gap', '60', '--task_ramp_frames', '1'])


def force_known(runtime, first_blind_boundary=False):
    """Explicit technical fixture only; do not call from scientific training."""
    clock = runtime['clock']
    forced = int((~clock.known).sum())
    clock.known.fill_(True)
    if first_blind_boundary:
        clock.remaining.fill_(1)
    return forced


def resume_fixture(out, condition, device, deadline):
    check_time(deadline)
    cfg = fixture_config(out, condition, device, n=2, rollout=4, frames=64)
    policy, optimizer, source, runtime = train.fresh_training(cfg)
    proof = train.provenance(cfg)
    before = train.cpu_tree(policy.state_dict())
    initial_hash = train.policy_hash(policy)
    assert all(p.requires_grad for p in policy.parameters())
    force_known(runtime, first_blind_boundary=True)
    # Keep this tiny serialization fixture away from terminal/pickup events;
    # use a nonzero native-position lookup to exercise saved goal-memory state.
    runtime['env'].t.zero_()
    runtime['env'].goal[:] = -runtime['env'].pos.sign() * .9
    obs = runtime['obs']
    runtime['goal'] = source.goal(obs['pos'], obs['dir'], obs['vel']).clone()
    synchronize(device); started = time.monotonic()
    first, _, trace = train.train_update(policy, optimizer, source, runtime, cfg, return_trace=True)
    assert not trace['roll']['blind'][0].any() and trace['roll']['blind'][1].all()
    assert trace['bootstrap_inputs']['blind'].all()
    assert trace['roll']['rgb'][1].count_nonzero() == trace['roll']['code'][1].count_nonzero() == 0
    assert trace['bootstrap_inputs']['rgb'].count_nonzero() == trace['bootstrap_inputs']['code'].count_nonzero() == 0
    assert math.isfinite(first['grad_norm']) and first['grad_norm'] > 0
    changed = [key for key, value in policy.state_dict().items()
               if not torch.equal(before[key], value.detach().cpu())]
    assert changed, 'Real RMSProp update did not change the full-policy fixture'
    checkpoint = out / 'technical_resume_fixture.pt'
    train.atomic_torch(checkpoint, train.snapshot(policy, optimizer, runtime, cfg, proof))
    check_time(deadline)
    expected_metrics, expected_episodes = train.train_update(policy, optimizer, source, runtime, cfg)
    expected = train.snapshot(policy, optimizer, runtime, cfg, proof)
    # Independent reconstruction and actual disk load, not a RAM-only copy.
    policy2, optimizer2, source2, runtime2 = train.fresh_training(cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    train.restore(saved, policy2, optimizer2, runtime2, cfg, proof)
    check_time(deadline)
    actual_metrics, actual_episodes = train.train_update(policy2, optimizer2, source2, runtime2, cfg)
    actual = train.snapshot(policy2, optimizer2, runtime2, cfg, proof)
    leaves = exact_tree(expected, actual)
    exact_tree(expected_metrics, actual_metrics, 'metrics')
    exact_tree(expected_episodes, actual_episodes, 'episodes')
    synchronize(device)
    result = dict(complete=True, private_seed=PRIVATE_SEED, config=train.scientific_config(cfg),
        provenance=proof, seconds=time.monotonic()-started,
        checkpoint_path=checkpoint.name, checkpoint_sha256=sha(checkpoint),
        initial_policy_sha256=initial_hash, final_policy_sha256=actual['policy_sha256'],
        bitwise_resume_equal=True, compared_leaves=leaves, changed_parameter_names=changed,
        trainable_parameters=sum(p.numel() for p in policy.parameters()),
        actor_replay_rtol=1e-4, actor_replay_atol=1e-5,
        max_replay_logit_error=max(m['replay_logit_max_error'] for m in (first, expected_metrics, actual_metrics)),
        max_replay_value_error=max(m['replay_value_max_error'] for m in (first, expected_metrics, actual_metrics)),
        fixture_changes=['known clock and first blind boundary forced', 'episode timer reset',
                         'goal moved away for the tiny fixture', 'nonzero lookup placed in goal memory'],
        research_model=False)
    train.atomic_json(out / 'resume_validation.json', result)
    return result


def throughput_fixture(out, condition, device, deadline):
    check_time(deadline)
    cfg = fixture_config(out, condition, device, n=32, rollout=100, frames=BENCHMARK_FRAMES)
    policy, optimizer, source, runtime = train.fresh_training(cfg)
    proof = train.provenance(cfg)
    initial_hash = train.policy_hash(policy)
    synchronize(device); started = time.monotonic()
    warm_seconds = None; warm_frames = None; forced = 0
    max_logit = max_value = 0.0
    with (out / 'throughput_updates.jsonl').open('x') as stream:
        while runtime['counters']['frames'] < BENCHMARK_FRAMES:
            check_time(deadline)
            forced += force_known(runtime)
            left = (BENCHMARK_FRAMES - runtime['counters']['frames']) // (cfg['n_envs'] * cfg['action_repeat'])
            metrics, _ = train.train_update(policy, optimizer, source, runtime, cfg, min(cfg['rollout'], left))
            synchronize(device); elapsed = time.monotonic() - started
            max_logit = max(max_logit, metrics['replay_logit_max_error'])
            max_value = max(max_value, metrics['replay_value_max_error'])
            row = {key: metrics[key] for key in ('frames', 'updates', 'decisions', 'blind_decisions',
                'batch_blind_fraction', 'max_gap', 'replay_logit_max_error', 'replay_value_max_error')}
            row.update(seconds=elapsed, frames_per_second=metrics['frames']/elapsed)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            if metrics['updates'] == WARMUP_UPDATES:
                warm_seconds, warm_frames = elapsed, metrics['frames']
            if metrics['updates'] % 20 == 0:
                print(json.dumps(dict(condition=condition, **row)), flush=True)
    assert runtime['counters']['frames'] == BENCHMARK_FRAMES
    assert runtime['counters']['blind_decisions'] > 0
    assert warm_seconds is not None and warm_frames is not None
    result = dict(complete=True, private_seed=PRIVATE_SEED, config=train.scientific_config(cfg),
        provenance=proof, seconds=elapsed, frames=BENCHMARK_FRAMES,
        updates=runtime['counters']['updates'], frames_per_second=BENCHMARK_FRAMES/elapsed,
        warmup_updates=WARMUP_UPDATES, warmup_frames=warm_frames, warmup_seconds=warm_seconds,
        post_warmup_frames_per_second=(BENCHMARK_FRAMES-warm_frames)/(elapsed-warm_seconds),
        initial_policy_sha256=initial_hash, final_policy_sha256=train.policy_hash(policy),
        forced_known_slot_events=forced, blind_decisions=runtime['counters']['blind_decisions'],
        decisions=runtime['counters']['decisions'], max_replay_logit_error=max_logit,
        max_replay_value_error=max_value, research_model=False, trained_weights_retained=False,
        fixture_changes=['one-frame curriculum reaches max_gap60 immediately',
                         'known=True at every rollout start; other clock and native physics updates unchanged'],
        updates_sha256=sha(out / 'throughput_updates.jsonl'))
    train.atomic_json(out / 'throughput.json', result)
    return result


def hardware(device):
    props = torch.cuda.get_device_properties(device)
    record = dict(hostname=socket.gethostname(), platform=platform.platform(),
        python=sys.version, numpy=np.__version__, torch=str(torch.__version__),
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        device=str(device), gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
        compute_capability=[props.major, props.minor], multiprocessors=props.multi_processor_count)
    proc = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=10)
    record['nvidia_smi'] = dict(exit_code=proc.returncode, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 600:
        parser.error('--max-seconds must be in (0,600]')
    if torch.device(args.device).type != 'cuda':
        parser.error('This GPU preflight refuses CPU fallback')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new private preflight output directory')
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); deadline = start + args.max_seconds
    result = dict(complete=False, purpose='Technical CUDA validation and cost estimation only',
        research_models=False, conditions=['gc100', 'pgc33'], private_seed=PRIVATE_SEED,
        target_frames_per_benchmark=BENCHMARK_FRAMES, max_seconds=args.max_seconds,
        started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256={p: sha(ROOT/p) for p in (*train.SOURCES, 'scripts/simple_memory_gpu_preflight.py')},
        cases={}, pending_cases=['gc100', 'pgc33'])
    train.atomic_json(out/'summary.json', result)
    code = 0
    def timeout(_signum, _frame):
        raise TimeoutError('Global technical-preflight time budget exhausted')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing a CPU benchmark')
        result['hardware'] = hardware(args.device)
        result['categorical_equivalence'] = categorical_equivalence(args.device)
        train.atomic_json(out/'summary.json', result)
        for condition in result['conditions']:
            folder = out/condition; folder.mkdir()
            case = result['cases'][condition] = {}
            case['resume'] = resume_fixture(folder, condition, args.device, deadline)
            train.atomic_json(out/'summary.json', result)
            gc.collect(); torch.cuda.empty_cache()
            case['throughput'] = throughput_fixture(folder, condition, args.device, deadline)
            result['pending_cases'].remove(condition)
            train.atomic_json(out/'summary.json', result)
            gc.collect(); torch.cuda.empty_cache()
        assert result['cases']['gc100']['resume']['initial_policy_sha256'] == result['cases']['pgc33']['resume']['initial_policy_sha256']
        assert result['cases']['gc100']['throughput']['initial_policy_sha256'] == result['cases']['pgc33']['throughput']['initial_policy_sha256']
        result['complete'] = True
    except BaseException as exc:
        code = 124 if isinstance(exc, TimeoutError) else 1
        result['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        result.update(exit_code=code, elapsed_seconds=time.monotonic()-start,
                      finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        train.atomic_json(out/'summary.json', result)
    print(json.dumps({key: result[key] for key in ('complete', 'exit_code', 'elapsed_seconds', 'pending_cases')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
