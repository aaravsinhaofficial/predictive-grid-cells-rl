"""Bounded, single-owner supervisor for the exact frozen 16-job cohort. No retries."""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/research/simple_memory_training_protocol.json'
sys.path.insert(0, str(ROOT))
from scripts import ring_cloud as cloud


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def launch_args(job, protocol_path=PROTOCOL):
    p = argparse.ArgumentParser(add_help=False)
    for key in ('name', 'machine', 'zone', 'output-dir', 'command'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--hours', required=True, type=float)
    p.add_argument('--include', action='append', default=[])
    p.add_argument('--include-only', action='store_true')
    assert job['plan_argv'][1:3] == ['scripts/ring_cloud.py', 'plan']
    args = p.parse_args(job['plan_argv'][3:])
    assert args.name == job['name'] and args.output_dir == job['worker_output_dir']
    assert shlex.split(args.command) == job['train_argv'] and '--resume' not in job['train_argv']
    for flag, value in (('--cond', job['condition']), ('--seed', job['seed']), ('--code_seed', job['seed']),
                        ('--task_max_gap', job['task_max_gap']), ('--frames', 1_000_000_000),
                        ('--out', job['trainer_output_dir'])):
        assert job['train_argv'].count(flag) == 1 and job['train_argv'][job['train_argv'].index(flag)+1] == str(value)
    assert job['train_argv'].count('--protocol') == 1
    protocol_arg = job['train_argv'][job['train_argv'].index('--protocol')+1]
    for value in (args.output_dir, job['trainer_output_dir'], protocol_arg):
        assert not Path(value).is_absolute() and '..' not in Path(value).parts, 'Worker paths must be portable relative paths'
    assert (ROOT/protocol_arg).resolve() == Path(protocol_path).resolve()
    return args


def validate(plan_path, protocol_path):
    plan = json.loads(plan_path.read_text()); protocol = json.loads(protocol_path.read_text())
    assert plan['status'] == protocol['status'] == 'frozen', 'Plan and protocol must be frozen'
    for key in ('protocol_frozen', 'trainer_source_frozen', 'trainer_tests_complete', 'read_only_launcher_plans_validated'):
        assert plan['readiness'][key], key
    threshold = plan['hardware']['minimum_training_fps']
    assert math.isfinite(threshold) and threshold > 0
    assert protocol['source_sha256'] and not plan['package']['missing_inputs']
    for name, expected in protocol['source_sha256'].items():
        assert sha(ROOT/name) == expected, f'Frozen protocol source changed: {name}'
    assert sha(ROOT/'scripts/ring_cloud.py') == plan['package']['launcher_sha256']
    manifest = {x['path']: x['sha256'] for x in plan['package']['current_manifest']}
    assert set(manifest) == set(plan['package']['files'])
    for name, expected in manifest.items():
        assert sha(ROOT/name) == expected, f'Frozen package source changed: {name}'
    expected = {(c, s, g) for c in ('gc100', 'pgc33') for s in range(1, 5) for g in (0, 60)}
    assert len(plan['jobs']) == 16 and len({j['name'] for j in plan['jobs']}) == 16
    assert {(j['condition'], j['seed'], j['task_max_gap']) for j in plan['jobs']} == expected
    assert set(plan['design']['first_jobs']) == {j['name'] for j in plan['jobs'] if j['seed'] == 1}
    assert plan['design']['max_active_workers'] == protocol['budget']['max_concurrent_workers'] == 8
    for job in plan['jobs']:
        args = launch_args(job, protocol_path)
        assert args.include_only and set(args.include) == set(manifest)
        assert job['target_environment_frames'] == 1_000_000_000
        assert Path(job['trainer_output_dir']).parent == Path(args.output_dir)
        prepared = cloud.plan(args)
        assert {x['path']: x['sha256'] for x in prepared['package_manifest']} == manifest
    return plan


def cat(uri, tail=False):
    argv = ['gcloud', 'storage', 'cat'] + (['--range=-65536'] if tail else []) + [uri]
    result = cloud.run(argv, check=False, timeout=60)
    if result.returncode == 0:
        return result.stdout
    if re.search(r'No URLs matched|matched no objects|(?i:404|not found|does not exist)', result.stderr):
        return None
    raise RuntimeError(f'Cloud read failed: {result.stderr[-1500:]}')


def technical_row(text):
    if not text or '\n' not in text:
        return None
    # An upload can capture a partial final JSONL row. Use the last full row.
    row = json.loads(text[:text.rfind('\n')].splitlines()[-1])
    keys = ('frames', 'updates', 'fps', 'loss', 'grad_norm', 'replay_logit_max_error', 'replay_value_max_error')
    result = {key: row[key] for key in keys}
    assert all(math.isfinite(value) for value in result.values()), 'Nonfinite training health metric'
    assert result['frames'] >= 0 and result['fps'] > 0
    return result


def technical_viability(plan, job, record, status, metric, now=None):
    """Use filtered health metrics; enforce throughput after the existing 2M-frame warmup.

    The first-wave speed threshold applies only before promotion. Later checks
    reserve 600 seconds for completion/sync within the original provider deadline.
    """
    assert all(math.isfinite(value) for value in metric.values()), 'Nonfinite training health metric'
    assert 0 <= metric['frames'] <= 1_000_000_000 and metric['fps'] > 0
    now = time.time() if now is None else now
    deadline = dt.datetime.fromisoformat(record['provider_deadline_utc'])
    assert deadline.tzinfo is not None, 'Provider deadline must be timezone-aware'
    remaining = max(0, 1_000_000_000-metric['frames'])/metric['fps']
    available = deadline.timestamp()-now
    result = dict(observed_frames=metric['frames'], observed_cumulative_fps=metric['fps'],
                  provider_deadline_utc=record['provider_deadline_utc'], checked_at_utc=dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
                  projected_training_seconds=remaining, completion_sync_reserve_seconds=600,
                  available_seconds=available, margin_seconds=available-remaining-600,
                  throughput_warmup_frames=2_000_000,
                  projection_enforced=metric['frames'] >= 2_000_000)
    if result['projection_enforced']:
        if job['name'] in plan['design']['first_jobs'] and not status.get('technical_gate_passed'):
            assert metric['fps'] >= plan['hardware']['minimum_training_fps'], f'{job["name"]}: initial technical throughput gate failed'
        assert result['margin_seconds'] >= 0, f'{job["name"]}: projected training plus 600s completion reserve exceeds original provider deadline'
    return result


def matching_record(job, record, manifest, instance_id=None, protocol_path=PROTOCOL):
    assert record.get('instance_id') and re.fullmatch(r'[0-9]+', str(record['instance_id'])), 'Unbound instance ID; explicit root reconciliation required'
    assert instance_id is None or str(record['instance_id']) == str(instance_id), 'Bound instance ID changed; explicit root reconciliation required'
    args = launch_args(job, protocol_path)
    for key in ('name', 'machine', 'zone'):
        assert record[key] == getattr(args, key), f'Existing worker {key} mismatch'
    assert record['command'] == job['train_argv'] and record['output_dir'] == args.output_dir
    assert {x['path']: x['sha256'] for x in record['package_manifest']} == manifest


def ensure_cohort_budget(plan, ledger, protocol_path=PROTOCOL):
    names = {r['name'] for r in ledger['runs']}
    remaining = sum(cloud.plan(launch_args(j, protocol_path))['reserved_usd'] for j in plan['jobs'] if j['name'] not in names)
    assert cloud.committed(ledger) + remaining <= cloud.BUDGET, 'Remaining full-cohort reservations exceed cumulative budget; no new launch'


def finish(job, done, protocol_hash):
    error = None; result = None
    with cloud.locked_ledger() as ledger:
        record = cloud.record(ledger, job['name'])
        assert done['package_sha256'] == record['package_manifest_sha256'], 'Worker package mismatch'
        vm = cloud.describe(record['name'], record['zone'])
        if vm is not None:
            cloud.validate_identity(record, vm)
        with contextlib.redirect_stdout(io.StringIO()):
            cloud.collect(record)
        folder = ROOT/record['local_artifacts']/Path(job['trainer_output_dir']).name
        try:
            summary = json.loads((folder/'summary.json').read_text())
            assert done['exit_code'] == 0 and summary['complete'] is True and summary['frames'] == 1_000_000_000
            assert summary['provenance']['protocol_sha256'] == protocol_hash
            assert sha(folder/'final.pt') == summary['final_sha256'], 'Final checkpoint hash mismatch'
            import torch
            saved = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
            assert saved['frames'] == saved['runtime']['counters']['frames'] == summary['frames'], 'Saved frame counters disagree'
            assert saved['updates'] == saved['runtime']['counters']['updates'] == summary['updates']
            assert saved['runtime']['counters']['decisions'] == summary['decisions'] == 250_000_000
            assert saved['provenance'] == summary['provenance']
            config = json.loads((folder/'config.json').read_text())
            for key, value in (('cond', job['condition']), ('seed', job['seed']), ('code_seed', job['seed']),
                               ('task_max_gap', job['task_max_gap']), ('frames', 1_000_000_000)):
                assert config[key] == saved['config'][key] == value, f'Collected configuration mismatch: {key}'
            result = dict(complete=True, frames=summary['frames'], summary_path=str(folder/'summary.json'),
                          summary_sha256=sha(folder/'summary.json'), final_sha256=summary['final_sha256'])
        except Exception as exc:
            error = f'{job["name"]}: terminal validation failed: {exc}'
        if vm is not None:
            cloud.run(['gcloud', 'compute', 'instances', 'delete', record['name'], '--project', cloud.PROJECT,
                       '--zone', record['zone'], '--quiet'], timeout=180)
        cloud.refresh(record)  # Only this exact owned worker; closes its reservation after absence.
    if error:
        raise RuntimeError(error)
    return result


def cycle(plan, state, protocol_hash, protocol_path=PROTOCOL):
    ledger = json.loads(cloud.LEDGER.read_text())
    records = {r['name']: r for r in ledger['runs']}
    manifest = {x['path']: x['sha256'] for x in plan['package']['current_manifest']}
    for job in plan['jobs']:
        name = job['name']; status = state['jobs'].setdefault(name, {})
        if status.get('complete') or name not in records:
            continue
        record = records[name]; matching_record(job, record, manifest, status.get('instance_id'), protocol_path)
        status['instance_id'] = str(record['instance_id'])
        done = cat(record['gcs_prefix']+'/DONE.json')
        if done:
            status.update(finish(job, json.loads(done), protocol_hash))
            if name in plan['design']['first_jobs']:
                status.update(technical_gate_passed=True, technical_gate_basis='validated_1B_completion')
            continue
        metric = technical_row(cat(record['gcs_prefix']+'/outputs/'+Path(job['trainer_output_dir']).name+'/metrics_000.jsonl', True))
        if metric:
            status['technical'] = metric
            vm = cloud.describe(record['name'], record['zone'])
            assert vm is not None, f'{name}: cannot use stale metrics from an absent VM'
            cloud.validate_identity(record, vm)
            assert vm['status'] == 'RUNNING', f'{name}: technical monitoring requires a RUNNING VM'
            status['deadline_feasibility'] = technical_viability(plan, job, record, status, metric)
            if name in plan['design']['first_jobs'] and metric['frames'] >= 2_000_000:
                if not status.get('technical_gate_passed'):
                    status['technical_gate_basis'] = 'initial_throughput_and_original_deadline'
                status['technical_gate_passed'] = True
        if 'actual_upper_bound_usd' in record or time.time() > dt.datetime.fromisoformat(record['provider_deadline_utc']).timestamp()+90:
            raise RuntimeError(f'{name}: worker closed/expired without DONE; no retry')
    gate = all(state['jobs'].get(name, {}).get('technical_gate_passed') for name in plan['design']['first_jobs'])
    for job in plan['jobs']:
        ledger = json.loads(cloud.LEDGER.read_text()); names = {r['name'] for r in ledger['runs']}
        if job['name'] in names or (job['seed'] != 1 and not gate):
            continue
        if sum('actual_upper_bound_usd' not in r for r in ledger['runs']) >= 8:
            break
        ensure_cohort_budget(plan, ledger, protocol_path)
        args = launch_args(job, protocol_path)
        assert {x['path']: x['sha256'] for x in cloud.plan(args)['package_manifest']} == manifest
        with contextlib.redirect_stdout(io.StringIO()):
            cloud.launch(args)
        record = cloud.record(json.loads(cloud.LEDGER.read_text()), job['name'])
        assert record.get('instance_id') and 'actual_upper_bound_usd' not in record, f'{job["name"]}: launch failed; no retry'
        state['jobs'][job['name']]['instance_id'] = str(record['instance_id'])
        state['jobs'][job['name']]['launch_observed_at'] = cloud.now()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default='docs/research/simple_memory_cloud_plan.json')
    p.add_argument('--protocol', default='docs/research/simple_memory_training_protocol.json')
    p.add_argument('--out', default='results/ring_simple_memory_supervision')
    p.add_argument('--max-hours', type=float, default=72)
    args = p.parse_args(argv)
    assert 0 < args.max_hours <= 72
    plan_path, protocol_path = Path(args.plan).resolve(), Path(args.protocol).resolve()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(plan_sha256=sha(plan_path), protocol_sha256=sha(protocol_path), supervisor_sha256=sha(__file__))
        state = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else dict(binding=binding, jobs={})
        assert state['binding'] == binding and not state.get('halted'), 'Changed inputs or halted supervisor require explicit root review'
        deadline = time.monotonic()+args.max_hours*3600
        try:
            for status in state['jobs'].values():
                if status.get('complete'):
                    path = Path(status['summary_path']); summary = json.loads(path.read_text())
                    assert sha(path) == status['summary_sha256'] and summary['complete'] and summary['frames'] == 1_000_000_000
                    assert sha(path.parent/'final.pt') == status['final_sha256'] == summary['final_sha256']
            while time.monotonic() < deadline:
                started = time.monotonic()
                assert sha(plan_path) == binding['plan_sha256'] and sha(protocol_path) == binding['protocol_sha256']
                plan = validate(plan_path, protocol_path)
                cycle(plan, state, binding['protocol_sha256'], protocol_path)
                state.update(updated_at=cloud.now(), complete=sum(bool(x.get('complete')) for x in state['jobs'].values()) == 16)
                save(out/'status.json', state)
                print(json.dumps(dict(at=state['updated_at'], completed=sum(bool(x.get('complete')) for x in state['jobs'].values()), complete=state['complete'])), flush=True)
                if state['complete']:
                    return 0
                time.sleep(max(0, 60-(time.monotonic()-started)))
            raise TimeoutError('Supervisor time limit reached; existing provider deadlines remain active')
        except BaseException as exc:
            state.update(halted=True, updated_at=cloud.now(), error=f'{type(exc).__name__}: {exc}')
            save(out/'status.json', state); print(json.dumps(dict(halted=True, error=state['error'])), flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
