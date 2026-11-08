"""Versioned native-training continuation with explicit guard-amendment proof.

The original trainer and its protocol are immutable dependencies. The only
computational change is the separately reviewed replay verification guard.
Environment, policy, replay, loss, optimizer and RNG helpers remain v1.
Execution requires a frozen amendment, exact source/runtime bindings and an
explicit assignment within the same sixteen-model scientific cohort.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

from code import train_memory as base
from code.train_memory import collect_rollout, discounted_returns, replay, max_gap_at
from code.train_memory import torch, np
from code.simple_memory_replay_guard import GUARD_ID, validate_replay as replay_guard

ROOT = base.ROOT
SOURCES = (*base.SOURCES, 'code/train_memory_v2.py', 'code/simple_memory_replay_guard.py')
RUNTIME_KEYS = ('torch_version', 'cuda_version', 'cudnn_version', 'cuda_device_name',
                'cuda_capability', 'deterministic_algorithms', 'cudnn_benchmark',
                'cudnn_tf32', 'matmul_tf32')
PAYLOAD_KEYS = ('schema', 'policy', 'optimizer', 'runtime', 'config', 'frames',
                'updates', 'policy_training', 'policy_sha256')


def parse_args(argv=None):
    # Parse operational v2 flags separately: never add them to scientific cfg.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--amendment', required=True)
    ap.add_argument('--parent-run')
    options, rest = ap.parse_known_args(argv)
    cfg = base.parse_args(rest)
    if options.parent_run and cfg['resume']:
        ap.error('--parent-run and --resume are mutually exclusive')
    if not cfg['protocol']:
        ap.error('--protocol must identify the unchanged original protocol')
    return cfg, options


def logical_run_id(cfg):
    assert cfg['task_max_gap'] in (0, 60)
    task = 'native' if cfg['task_max_gap'] == 0 else 'blackout'
    return f"{cfg['cond']}/{task}/seed{cfg['seed']}"


def tree_digest(value):
    """Typed, shape-preserving digest, independent of torch archive metadata."""
    h = hashlib.sha256()
    def emit(tag, data):
        h.update(tag.encode() + b'\0' + str(len(data)).encode() + b':' + data)
    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            emit('torch', str(v.dtype).encode()); visit(a)
        elif isinstance(v, np.ndarray):
            emit('numpy', v.dtype.str.encode()); visit(tuple(v.shape))
            emit('bytes', np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            emit('dict', str(len(v)).encode())
            for k in sorted(v, key=lambda x: (type(x).__name__, str(x))): visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            emit(type(v).__name__, str(len(v)).encode())
            for item in v: visit(item)
        elif isinstance(v, np.generic):
            visit(v.item())
        else:
            assert v is None or isinstance(v, (str, int, float, bool)), type(v)
            emit(type(v).__name__, repr(v).encode())
    visit(value)
    return h.hexdigest()


def payload_digest(saved):
    assert set(saved) == set(PAYLOAD_KEYS) | {'provenance'}, 'Unexpected checkpoint schema'
    return tree_digest({k: saved[k] for k in PAYLOAD_KEYS})


def load_amendment(path, cfg):
    record = json.loads(Path(path).read_text())
    assert record['schema'] == 1 and record['status'] == 'frozen'
    assert record['kind'] == 'simple_memory_replay_guard_amendment'
    assert record['parent_protocol_sha256'] == base.sha(cfg['protocol'])
    assert set(record['source_sha256']) == set(SOURCES), 'Incomplete producer source union'
    for name, expected in record['source_sha256'].items():
        assert base.sha(ROOT/name) == expected, f'Amended source changed: {name}'
    original = json.loads(Path(cfg['protocol']).read_text())
    assert original['status'] == 'frozen'
    for name in base.SOURCES:
        assert record['source_sha256'][name] == original['source_sha256'][name]
    expected_ids = {f'{cond}/{task}/seed{seed}' for cond in ('gc100', 'pgc33')
                    for task in ('native', 'blackout') for seed in (1, 2, 3, 4)}
    bindings = record['run_bindings']
    assert set(bindings) == expected_ids, 'All sixteen logical models must remain registered'
    kinds = [b['kind'] for b in bindings.values()]
    assert {k: kinds.count(k) for k in set(kinds)} == {'v1_completed': 4, 'v1_resume': 4, 'v2_fresh': 8}
    binding = bindings[logical_run_id(cfg)]
    assert binding['kind'] != 'v1_completed', 'Completed v1 endpoint must not be retrained'
    return record, binding


def require_implemented_guard(amendment):
    if GUARD_ID is None:
        raise NotImplementedError('Replay guard awaits GPU diagnosis and prospective review; training disabled')
    assert amendment['guard']['id'] == GUARD_ID and amendment['guard']['status'] == 'frozen'


def provenance(cfg, amendment_path, lineage_sha256):
    amendment, _ = load_amendment(amendment_path, cfg)
    original = base.provenance(cfg)  # Includes exact original numerical-stack proof.
    assert set(amendment['numerical_runtime']) == set(RUNTIME_KEYS)
    assert amendment['numerical_runtime'] == {k: original[k] for k in RUNTIME_KEYS}, 'Numerical runtime differs from amendment'
    assert isinstance(lineage_sha256, str) and len(lineage_sha256) == 64
    return dict(original, provenance_version=2, producer_module='code.train_memory_v2',
                source_sha256=amendment['source_sha256'], base_provenance=original,
                amendment_sha256=base.sha(amendment_path), guard_id=GUARD_ID,
                logical_run_id=logical_run_id(cfg), lineage_sha256=lineage_sha256)


def restore_parent(saved, policy, optimizer, runtime, cfg, original_proof):
    """Strict v1 restore, then verify EVERY original checkpoint payload leaf."""
    before = payload_digest(saved)
    base.restore(saved, policy, optimizer, runtime, cfg, original_proof)
    restored = base.snapshot(policy, optimizer, runtime, cfg, original_proof)
    assert payload_digest(restored) == before, 'Migration changed full checkpoint state'
    assert payload_digest(saved) == before, 'Migration mutated the supplied parent'
    return before


def _safe_file(folder, name):
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts and name != '.'
    path = folder/rel
    assert path.is_file() and not path.is_symlink()
    assert path.resolve().is_relative_to(folder.resolve())
    return path


def authenticate_parent(parent, binding, cfg, original_proof):
    parent = Path(parent)
    required = {'resume.pt', 'initial.pt', 'config.json', 'provenance.json'}
    required.update(p.name for p in parent.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl', '.log'))
    evidence = binding['parent_evidence_sha256']
    assert required <= evidence.keys(), 'Parent raw logs/attempts omitted from lineage'
    for name, expected in evidence.items(): assert base.sha(_safe_file(parent, name)) == expected, name
    assert evidence['resume.pt'] == binding['parent_checkpoint_sha256']
    assert json.loads((parent/'provenance.json').read_text()) == original_proof
    original_cfg = json.loads((parent/'config.json').read_text())
    original_cfg.pop('population', None)
    assert original_cfg == base.scientific_config(cfg)
    if (parent/'summary.json').exists():
        assert not json.loads((parent/'summary.json').read_text()).get('complete'), 'Parent already completed'
    saved = torch.load(parent/'resume.pt', map_location='cpu', weights_only=False)
    initial = torch.load(parent/'initial.pt', map_location='cpu', weights_only=False)
    for snapshot in (saved, initial):
        assert snapshot['provenance'] == original_proof and snapshot['config'] == original_cfg
        assert snapshot['schema'] == 1
    assert initial['frames'] == initial['updates'] == 0
    assert 0 < saved['frames'] < cfg['frames'] and saved['updates'] > 0
    assert saved['runtime']['counters']['frames'] == saved['frames']
    assert saved['runtime']['counters']['updates'] == saved['updates']
    assert saved['frames'] == saved['updates']*cfg['n_envs']*cfg['rollout']*cfg['action_repeat']
    assert saved['runtime']['counters']['decisions'] == saved['frames']//cfg['action_repeat']
    last_updates = []
    for path in sorted(parent.glob('metrics_*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip(): last_updates.append(int(json.loads(line)['updates']))
    assert last_updates and max(last_updates) >= saved['updates']
    assert max(last_updates) == binding['last_logged_update']
    return saved, dict(parent_checkpoint_sha256=evidence['resume.pt'],
        parent_evidence_sha256=evidence, restored_frames=saved['frames'], restored_updates=saved['updates'],
        last_logged_parent_update=max(last_updates), rolled_back_logged_updates=max(last_updates)-saved['updates'],
        accepted_parent_update_cutoff=saved['updates'],
        parent_log_policy='All raw logs retained; parent rows beyond cutoff are rolled back, never double-counted.')


def train_update(policy, optimizer, source, runtime, cfg, length=None, return_trace=False):
    length = cfg['rollout'] if length is None else int(length)
    roll, state0, boot, bootstrap_inputs, episodes = collect_rollout(policy, source, runtime, cfg, length)
    returns = discounted_returns(roll['r'], roll['d'], boot, cfg['gamma'])
    logits, values, _ = replay(policy, roll, state0)
    online = torch.stack(roll['actor_logits']); online_v = torch.stack(roll['actor_values'])
    # Different CNN batch sizes can have small floating-point discrepancies.
    guard_record = replay_guard(policy, roll, state0, logits, values, online, online_v)
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
    metrics['replay_guard'] = guard_record
    if return_trace: return metrics, episodes, dict(roll=roll, state0=state0, bootstrap=boot, bootstrap_inputs=bootstrap_inputs, returns=returns)
    return metrics, episodes


def _next_attempt(out):
    indices = [int(p.stem.split('_')[1]) for p in out.glob('attempt_*.json')]
    return max(indices, default=-1) + 1


def _initialize(cfg, options, binding, out):
    policy, opt, source, runtime = base.fresh_training(cfg)
    original_proof = base.provenance(cfg)
    if cfg['resume']:
        lineage = json.loads((out/'lineage.json').read_text())
        assert lineage['logical_run_id'] == logical_run_id(cfg)
        assert lineage['kind'] == binding['kind']
        if binding['kind'] == 'v1_resume':
            assert lineage['parent_checkpoint_sha256'] == binding['parent_checkpoint_sha256']
            assert lineage['parent_evidence_sha256'] == binding['parent_evidence_sha256']
        assert lineage['amendment_sha256'] == base.sha(options.amendment)
        for name, expected in lineage.get('parent_evidence_sha256', {}).items():
            assert base.sha(_safe_file(out/'parent_evidence', name)) == expected
        proof = provenance(cfg, options.amendment, base.sha(out/'lineage.json'))
        assert json.loads((out/'provenance.json').read_text()) == proof
        initial = torch.load(out/'initial.pt', map_location='cpu', weights_only=False)
        assert payload_digest(initial) == lineage['initial_payload_sha256']
        saved = torch.load(out/'resume.pt', map_location='cpu', weights_only=False)
        expected = payload_digest(saved)
        base.restore(saved, policy, opt, runtime, cfg, proof)
        assert payload_digest(base.snapshot(policy, opt, runtime, cfg, proof)) == expected
        return policy, opt, source, runtime, proof
    lineage = dict(schema=1, logical_run_id=logical_run_id(cfg), kind=binding['kind'],
                   amendment_sha256=base.sha(options.amendment), parent_protocol_sha256=original_proof['protocol_sha256'])
    if options.parent_run:
        assert binding['kind'] == 'v1_resume'
        parent = Path(options.parent_run)
        saved, details = authenticate_parent(parent, binding, cfg, original_proof)
        digest = restore_parent(saved, policy, opt, runtime, cfg, original_proof)
        lineage.update(details, parent_payload_sha256=digest, migrated_payload_sha256=digest,
                       full_state_exact=True)
        evidence = out/'parent_evidence'; evidence.mkdir()
        for name, expected in details['parent_evidence_sha256'].items():
            dest = evidence/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_safe_file(parent, name), dest)
            assert base.sha(dest) == expected
        # This remains the actual original initialization, with its truthful v1 proof.
        shutil.copyfile(evidence/'initial.pt', out/'initial.pt')
        lineage['initial_payload_sha256'] = payload_digest(torch.load(out/'initial.pt', map_location='cpu', weights_only=False))
    else:
        assert binding['kind'] == 'v2_fresh', 'This logical model requires its exact registered v1 parent'
        lineage.update(restored_frames=0, restored_updates=0, accepted_parent_update_cutoff=0,
                       initial_payload_sha256=payload_digest(base.snapshot(policy, opt, runtime, cfg, original_proof)))
    base.atomic_json(out/'lineage.json', lineage)
    proof = provenance(cfg, options.amendment, base.sha(out/'lineage.json'))
    base.atomic_json(out/'config.json', dict(base.scientific_config(cfg), population=source.pop.describe()))
    base.atomic_json(out/'provenance.json', proof)
    current = base.snapshot(policy, opt, runtime, cfg, proof)
    if options.parent_run:
        assert payload_digest(current) == lineage['parent_payload_sha256']
        base.atomic_torch(out/'migration_boundary.pt', current)
    else:
        base.atomic_torch(out/'initial.pt', current)
    base.atomic_torch(out/'resume.pt', current)
    return policy, opt, source, runtime, proof


def main(argv=None):
    cfg, options = parse_args(argv)
    amendment, binding = load_amendment(options.amendment, cfg)
    require_implemented_guard(amendment)  # BEFORE output creation, RNG initialization, or environment work.
    out = Path(cfg['out']); resume_path = out/'resume.pt'
    if cfg['resume']:
        if not resume_path.is_file(): raise FileNotFoundError('Exact v2 resume.pt required')
        if (out/'summary.json').exists(): raise FileExistsError('Completed output cannot be resumed')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError('New v2 attempt requires an empty output directory; parent evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    policy, opt, source, runtime, proof = _initialize(cfg, options, binding, out)
    pending_signal = [0]
    def stop(signum, _frame): pending_signal[0] = signum
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    attempt = _next_attempt(out)
    record = dict(attempt=attempt, start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  producer_version=2, logical_run_id=logical_run_id(cfg), lineage_sha256=proof['lineage_sha256'],
                  restored_updates=runtime['counters']['updates'], restored_frames=runtime['counters']['frames'],
                  resume_sha256=base.sha(resume_path), resume_input=f'resume_input_{attempt:03d}.pt')
    shutil.copyfile(resume_path, out/record['resume_input'])
    base.atomic_json(out/f'attempt_{attempt:03d}.json', record)
    start = time.monotonic(); start_frames = runtime['counters']['frames']
    next_ckpt = (start_frames//cfg['ckpt_every']+1)*cfg['ckpt_every']; status = 0
    with (out/f'metrics_{attempt:03d}.jsonl').open('x') as log, (out/f'episodes_{attempt:03d}.jsonl').open('x') as ep_log:
        while runtime['counters']['frames'] < cfg['frames']:
            if pending_signal[0] or time.monotonic()-start >= cfg['max_seconds']:
                status = 128+pending_signal[0] if pending_signal[0] else 124
                break
            left = (cfg['frames']-runtime['counters']['frames'])//(cfg['n_envs']*cfg['action_repeat'])
            # Exceptions preserve the last valid checkpoint and the complete raw attempt log.
            # An incomplete rollout is never misrepresented as a completed optimizer update.
            try:
                metrics, episodes = train_update(policy, opt, source, runtime, cfg, min(cfg['rollout'], left))
            except AssertionError as exc:
                if hasattr(exc, 'record'):
                    base.atomic_json(out/f'rejected_update_{attempt:03d}.json', dict(
                        complete=False, guard=exc.record, attempt=attempt,
                        last_accepted_update=runtime['counters']['updates'],
                        unaccepted_collected_frames=runtime['counters']['frames'],
                        resume_sha256=base.sha(resume_path), lineage_sha256=proof['lineage_sha256']))
                raise
            metrics.update(seconds=time.monotonic()-start, attempt=attempt)
            metrics['fps'] = (metrics['frames']-start_frames)/max(metrics['seconds'], 1e-9)
            log.write(json.dumps(metrics)+'\n'); log.flush()
            for row in episodes: ep_log.write(json.dumps(dict(row, frames=metrics['frames'], update=metrics['updates']))+'\n')
            ep_log.flush()
            if metrics['updates']%20 == 0: print(json.dumps(metrics), flush=True)
            if metrics['updates']%cfg['resume_every'] == 0:
                base.atomic_torch(resume_path, base.snapshot(policy, opt, runtime, cfg, proof))
            if metrics['frames'] >= next_ckpt:
                base.atomic_torch(out/f'checkpoint_{metrics["frames"]:010d}.pt', base.snapshot(policy, opt, runtime, cfg, proof))
                next_ckpt = (metrics['frames']//cfg['ckpt_every']+1)*cfg['ckpt_every']
    if pending_signal[0] and not status: status = 128+pending_signal[0]
    final = base.snapshot(policy, opt, runtime, cfg, proof); base.atomic_torch(resume_path, final)
    complete = final['frames'] == cfg['frames'] and not status
    record.update(complete=complete, exit_code=status, end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  seconds=time.monotonic()-start, final_frames=final['frames'], final_updates=final['updates'],
                  resume_sha256_after=base.sha(resume_path))
    base.atomic_json(out/f'attempt_{attempt:03d}.json', record)
    if complete:
        base.atomic_torch(out/'final.pt', final)
        base.atomic_json(out/'summary.json', dict(complete=True, **runtime['counters'], provenance=proof,
            final_sha256=base.sha(out/'final.pt'), final_policy_sha256=final['policy_sha256'],
            initial_sha256=base.sha(out/'initial.pt'), lineage_sha256=proof['lineage_sha256'],
            attempt_records=[p.name for p in sorted(out.glob('attempt_*.json'))]))
    else:
        base.atomic_json(out/f'interrupted_{attempt:03d}.json', record)
    print(json.dumps(record), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
