"""Single-owner recovery of the fixed 16-model cohort. No automatic attempt retries."""
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
sys.path.insert(0, str(ROOT))
from scripts import ring_cloud as cloud
from scripts.run_simple_memory_cohort import cat, save, sha

TARGET = 1_000_000_000
MODES = {'retained_v1': 4, 'resume_v2': 4, 'fresh_v2': 8}
VALIDATION_NAME = 'ring-20261105-simplemem-guardcheck-v2'
CORRUPTIONS = {'rgb', 'code', 'goal', 'initial_h', 'initial_c', 'reset',
               'large_logits', 'large_values', 'large_embeddings'}
HOST_LOGICAL = 'ring-20261105-simplemem-gc100-native-s4'
HOST_INSTANCE_ID = '2749035052516857068'


def local(value):
    p = Path(value)
    assert not p.is_absolute() and '..' not in p.parts, 'Paths must be portable and inside the repository'
    resolved = (ROOT/p).resolve()
    assert resolved.is_relative_to(ROOT) and resolved != ROOT and not (ROOT/p).is_symlink()
    return resolved


def check_hashes(manifest, required=()):
    assert isinstance(manifest, dict) and manifest and set(required) <= set(manifest), 'Missing hash manifest entries'
    for name, digest in manifest.items():
        assert sha(local(name)) == digest, f'Changed frozen input: {name}'


def continuation(row):
    """Exactly one prospectively registered host move; never an automatic retry."""
    if 'continuation' not in row: return None
    c = row['continuation']
    assert isinstance(c, dict) and c['kind'] == 'explicit_v2_host_recovery'
    assert row['logical_name'] == HOST_LOGICAL and row['mode'] == 'fresh_v2'
    assert (row['condition'], row['seed'], row['task_max_gap']) == ('gc100', 4, 0)
    assert row['name'] == HOST_LOGICAL+'-r1'
    assert c['source_worker_name'] == HOST_LOGICAL and str(c['source_instance_id']) == HOST_INSTANCE_ID
    assert c['source_run'] == f'results/ring_cloud_artifacts/{HOST_LOGICAL}/seed4'
    assert row['trainer_output_dir'] == 'results/ring_simple_memory_v2/gc100/native/seed4'
    assert row['worker_output_dir'] == 'results/ring_simple_memory_v2/gc100/native'
    assert c['source_attempt'] == 0 and c['attempt'] == 1
    assert 0 < c['start_frames'] == row['start_frames'] < TARGET
    assert c['start_frames'] == c['start_updates']*12800
    assert isinstance(c['source_files_sha256'], dict) and c['source_files_sha256']
    assert c['resume_sha256'] == c['source_files_sha256']['resume.pt']
    assert re.fullmatch('[0-9a-f]{64}', c['resume_payload_sha256'])
    return c


def expected_attempt(row):
    c = continuation(row)
    return c['attempt'] if c else 0


def _relative_file(folder, name):
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = Path(folder)/relative
    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(folder).resolve())
    return path


def validate_continuation(row, plan):
    """CPU metadata/hash validation of the exact stopped input and its copy."""
    c = continuation(row)
    if c is None: return
    import torch
    from code import train_memory_v2 as producer
    source, target = local(c['source_run']), local(row['trainer_output_dir'])
    files = c['source_files_sha256']
    required = {'config.json', 'provenance.json', 'lineage.json', 'initial.pt', 'resume.pt',
                'resume_input_000.pt', 'attempt_000.json', 'interrupted_000.json',
                'metrics_000.jsonl', 'episodes_000.jsonl'}
    assert required <= set(files)
    assert set(files) == {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    assert not any(Path(n).name.startswith('rejected_update_') for n in files)
    assert 'final.pt' not in files and 'summary.json' not in files
    assert {n for n in files if Path(n).name.startswith('attempt_')} == {'attempt_000.json'}
    for name, digest in files.items():
        assert sha(_relative_file(source, name)) == sha(_relative_file(target, name)) == digest
        assert row['package_manifest'].get(row['trainer_output_dir']+'/'+name) == digest
    worker = c['source_worker_result']
    assert worker['path'] == f'results/ring_cloud_artifacts/{HOST_LOGICAL}/WORKER_RESULT.json'
    check_hashes({worker['path']:worker['sha256']})
    assert json.loads(local(worker['path']).read_text())['exit_code'] == 143, 'Host move requires an orderly SIGTERM checkpoint'
    cfg, proof, lineage, attempt = (json.loads((source/n).read_text()) for n in
        ('config.json', 'provenance.json', 'lineage.json', 'attempt_000.json'))
    scientific = {k:v for k,v in cfg.items() if k != 'population'}
    amendment = json.loads(local(plan['bindings']['amendment']['path']).read_text())
    assert (cfg['cond'], cfg['seed'], cfg['code_seed'], cfg['task_max_gap']) == ('gc100', 4, 4, 0)
    assert proof['provenance_version'] == 2 and proof['producer_module'] == 'code.train_memory_v2'
    assert proof['source_sha256'] == amendment['source_sha256'] and proof['amendment_sha256'] == plan['bindings']['amendment']['sha256']
    assert proof['protocol_sha256'] == plan['bindings']['protocol']['sha256']
    assert proof['logical_run_id'] == lineage['logical_run_id'] == 'gc100/native/seed4'
    assert proof['config_sha256'] == hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
    assert {k:proof[k] for k in amendment['numerical_runtime']} == amendment['numerical_runtime']
    assert sha(source/'lineage.json') == proof['lineage_sha256'] == attempt['lineage_sha256']
    assert lineage['kind'] == 'v2_fresh' and lineage['restored_frames'] == lineage['restored_updates'] == 0
    assert not lineage.get('parent_evidence_sha256') and 'parent_checkpoint_sha256' not in lineage
    saved = torch.load(source/'resume.pt', map_location='cpu', weights_only=False)
    initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
    assert saved['config'] == initial['config'] == scientific and saved['provenance'] == initial['provenance'] == proof
    assert (saved['frames'], saved['updates']) == (c['start_frames'], c['start_updates'])
    assert all(saved['runtime']['counters'][k] == v for k,v in
        dict(frames=c['start_frames'], updates=c['start_updates'], decisions=c['start_frames']//4).items())
    assert producer.payload_digest(saved) == c['resume_payload_sha256']
    assert initial['frames'] == initial['updates'] == 0 and producer.payload_digest(initial) == lineage['initial_payload_sha256']
    assert attempt == json.loads((source/'interrupted_000.json').read_text())
    assert attempt['attempt'] == 0 and not attempt['complete'] and attempt['exit_code'] == 143
    assert attempt['restored_frames'] == attempt['restored_updates'] == 0
    assert (attempt['final_frames'], attempt['final_updates']) == (c['start_frames'], c['start_updates'])
    assert attempt['resume_sha256_after'] == c['resume_sha256']
    assert sha(source/'resume_input_000.pt') == attempt['resume_sha256']
    text = (source/'metrics_000.jsonl').read_text(); assert text.endswith('\n')
    metrics = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(metrics) == c['start_updates']
    for i,m in enumerate(metrics,1):
        assert (m['attempt'],m['updates'],m['frames'],m['decisions']) == (0,i,i*12800,i*3200)
        assert m['replay_guard']['accepted'] is True


def continuation_inputs(row, folder):
    """Portable final-attempt evidence; no reads from the original source folder."""
    c = continuation(row)
    if c is None: return {}
    import torch
    from code import train_memory_v2 as producer
    folder = Path(folder)
    result = {n:h for n,h in c['source_files_sha256'].items() if n != 'resume.pt'}
    for name,digest in result.items(): assert sha(_relative_file(folder,name)) == digest
    supplied = _relative_file(folder,'resume_input_001.pt')
    assert sha(supplied) == c['resume_sha256']
    saved = torch.load(supplied,map_location='cpu',weights_only=False)
    assert producer.payload_digest(saved) == c['resume_payload_sha256']
    assert (saved['frames'],saved['updates']) == (c['start_frames'],c['start_updates'])
    attempt_path = _relative_file(folder,'attempt_001.json'); attempt = json.loads(attempt_path.read_text())
    proof = json.loads((folder/'provenance.json').read_text())
    assert saved['provenance'] == proof
    assert attempt['attempt'] == 1 and attempt['producer_version'] == 2 and attempt['logical_run_id'] == 'gc100/native/seed4'
    assert attempt['complete'] is True and attempt['exit_code'] == 0
    assert attempt['lineage_sha256'] == proof['lineage_sha256']
    assert (attempt['restored_frames'],attempt['restored_updates']) == (c['start_frames'],c['start_updates'])
    assert (attempt['final_frames'],attempt['final_updates']) == (TARGET,78125)
    assert attempt['resume_input'] == supplied.name and attempt['resume_sha256'] == c['resume_sha256']
    summary_path = _relative_file(folder,'summary.json')
    assert json.loads(summary_path.read_text())['attempt_records'] == ['attempt_000.json','attempt_001.json']
    assert {p.name for p in folder.glob('attempt_*.json')} == {'attempt_000.json','attempt_001.json'}
    assert not list(folder.glob('rejected_update_*.json')) and not (folder/'interrupted_001.json').exists()
    result.update({supplied.name:sha(supplied),attempt_path.name:sha(attempt_path),summary_path.name:sha(summary_path)})
    return result


def validate_release(plan, amendment):
    """A report hash alone is insufficient: all prospective GPU checks must pass."""
    bindings = plan['bindings']; release = plan['gpu_release']
    assert release['status'] == 'complete', 'GPU validation release is still pending'
    check_hashes({release['summary_path']: release['summary_sha256'], release['done_path']: release['done_sha256']})
    protocol = json.loads(local(bindings['validation_protocol']['path']).read_text())
    report = json.loads(local(release['summary_path']).read_text())
    done = json.loads(local(release['done_path']).read_text())
    cloud_plan = json.loads(local(bindings['validation_cloud_plan']['path']).read_text())
    expected = cloud_plan['prepared_cloud_plan']
    assert protocol['status'] == cloud_plan['status'] == 'frozen' and expected['name'] == VALIDATION_NAME
    assert release['worker_name'] == VALIDATION_NAME and re.fullmatch(r'[0-9]+', str(release['instance_id']))
    assert done['exit_code'] == 0 and done['package_sha256'] == expected['package_manifest_sha256']
    assert report['complete'] is True and report['sources_unchanged'] is True and report['all_input_bytes_preserved'] is True
    assert report['manifest_sha256'] == bindings['validation_protocol']['sha256']
    assert report['amendment_sha256'] == bindings['amendment']['sha256'] == protocol['amendment_sha256']
    assert report['source_sha256'] == protocol['source_sha256']
    cases = report['cases']; ids = {k for k,v in amendment['run_bindings'].items() if v['kind'] == 'v1_resume'}
    assert len(cases) == 4 and {r['logical_run_id'] for r in cases} == ids
    assert sum(r['accepted_reconstructed'] for r in cases) == protocol['required_total_accepted_updates'] == 375
    prescribed = {r['logical_run_id']:r for r in protocol['cases']}
    for row in cases:
        spec = prescribed[row['logical_run_id']]
        assert row['complete'] is True and row['original_assertion_failure']
        assert row['accepted_reconstructed'] == spec['last_successful_update']-spec['start_update']
        assert row['failed_update'] == spec['last_successful_update']+1
        assert all(row[k] is True for k in ('original_update_metrics_exact', 'full_state_and_rng_exact', 'gradients_exact'))
        assert row['fallback']['fallback'] is True and row['fallback']['accepted'] is True
        assert row['reference_payload_sha256'] == row['guarded_payload_sha256']
        assert row['reference_gradients_sha256'] == row['guarded_gradients_sha256']
    known = next(r for r in cases if r['logical_run_id'] == 'pgc33/blackout/seed1')
    assert known['all_188_updates_full_boundary_exact'] is True
    assert set(known['corruption_controls']) == set(protocol['required_corruption_controls']) == CORRUPTIONS
    assert all(r['rejected'] is True for r in known['corruption_controls'].values())
    return expected


def launch_args(row, plan):
    p = argparse.ArgumentParser(add_help=False)
    for key in ('name', 'machine', 'zone', 'output-dir', 'command'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--include', action='append', default=[])
    p.add_argument('--include-only', action='store_true')
    assert row['plan_argv'][1:3] == ['scripts/ring_cloud.py', 'plan']
    args = p.parse_args(row['plan_argv'][3:])
    assert args.name == row['name'] and args.output_dir == row['worker_output_dir']
    assert args.hours == row['provider_hours'] and args.machine == 'n1-standard-8'
    assert args.zone in ('us-central1-a', 'us-central1-b')
    assert args.include_only and len(args.include) == len(set(args.include))
    assert set(args.include) == set(row['package_manifest'])
    argv = row['train_argv']
    assert shlex.split(args.command) == argv and argv[:3] == ['python', '-m', 'code.train_memory_v2']
    c = continuation(row)
    assert argv.count('--resume') == int(c is not None), 'Only the explicit host continuation may resume'
    expected = {'--cond': row['condition'], '--seed': row['seed'], '--code_seed': row['seed'],
        '--task_max_gap': row['task_max_gap'], '--frames': TARGET, '--out': row['trainer_output_dir'],
        '--protocol': plan['bindings']['protocol']['path'], '--amendment': plan['bindings']['amendment']['path']}
    if row['mode'] == 'resume_v2':
        expected['--parent-run'] = row['parent_run']
        assert row['name'] == row['logical_name']+'-r1'
    else:
        assert row['mode'] == 'fresh_v2' and '--parent-run' not in argv
        assert row['name'] == row['logical_name']+('-r1' if c else '')
    for flag, value in expected.items():
        assert argv.count(flag) == 1 and argv[argv.index(flag)+1] == str(value), f'Changed launch argument {flag}'
    for path in (args.output_dir, row['trainer_output_dir'], expected['--protocol'], expected['--amendment']):
        local(path)
    assert Path(row['trainer_output_dir']).parent == Path(args.output_dir)
    return args


def validate(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    assert plan['status'] == 'frozen' and plan['max_active_workers'] == 8
    assert plan['cumulative_ceiling_usd'] == cloud.BUDGET == 600 and cloud.RATE == 1.25
    assert plan['readiness'] and all(v is True for v in plan['readiness'].values())
    check_hashes(plan['operational_source_sha256'], ('scripts/run_simple_memory_recovery.py',
        'scripts/ring_cloud.py', 'scripts/run_simple_memory_cohort.py', 'scripts/simple_memory_endpoint.py',
        'scripts/run_simple_memory_analysis.py',
        'tests/test_run_simple_memory_recovery.py'))
    check_hashes(plan['validation_artifacts_sha256'])
    bindings = plan['bindings']
    assert {'protocol', 'amendment', 'original_plan', 'budget_envelope', 'analysis_registry',
            'validation_protocol', 'validation_cloud_plan'} <= set(bindings)
    check_hashes({v['path']: v['sha256'] for v in bindings.values()})
    protocol, amendment, original, envelope = (json.loads(local(bindings[k]['path']).read_text())
        for k in ('protocol', 'amendment', 'original_plan', 'budget_envelope'))
    assert protocol['status'] == amendment['status'] == original['status'] == 'frozen'
    assert amendment['parent_protocol_sha256'] == bindings['protocol']['sha256']
    assert amendment['guard']['status'] == 'frozen' and amendment['guard']['id']
    validation_worker = validate_release(plan, amendment)
    check_hashes(protocol['source_sha256'], ('code/train_memory.py',))
    check_hashes(amendment['source_sha256'], ('code/train_memory_v2.py',))
    old = {j['name']: j for j in original['jobs']}
    continuations = {r['logical_name']: r for r in envelope['continuations']}
    models = plan['models']
    assert len(models) == len({r['logical_name'] for r in models}) == 16
    assert {r['logical_name'] for r in models} == set(old)
    assert {(r['condition'], r['seed'], r['task_max_gap']) for r in models} == {
        (c, s, g) for c in ('gc100', 'pgc33') for s in range(1, 5) for g in (0, 60)}
    assert {m: sum(r['mode'] == m for r in models) for m in MODES} == MODES
    assert sum('continuation' in r for r in models) <= 1
    attempts = [r['name'] for r in models if r['mode'] != 'retained_v1']
    assert len(attempts) == len(set(attempts)) == 12
    assert len({r['trainer_output_dir'] for r in models if r['mode'] != 'retained_v1'}) == 12
    registered = amendment['run_bindings']
    expected_ids = {f'{r["condition"]}/{"native" if r["task_max_gap"] == 0 else "blackout"}/seed{r["seed"]}' for r in models}
    assert set(registered) == expected_ids
    for row in models:
        c = continuation(row)
        original_row = old[row['logical_name']]
        assert all(row[k] == original_row[k] for k in ('condition', 'seed', 'task_max_gap'))
        expected_mode = 'resume_v2' if row['logical_name'] in continuations else ('fresh_v2' if row['seed'] >= 3 else 'retained_v1')
        assert row['mode'] == expected_mode, 'Logical model changed recovery role'
        key = f'{row["condition"]}/{"native" if row["task_max_gap"] == 0 else "blackout"}/seed{row["seed"]}'
        registered_row = registered[key]
        assert registered_row['kind'] == {'retained_v1':'v1_completed', 'resume_v2':'v1_resume', 'fresh_v2':'v2_fresh'}[row['mode']]
        assert registered_row['logical_job_name'] == row['logical_name']
        if row['mode'] == 'retained_v1':
            assert row['run_dir'] == registered_row['original_run']
            folder = local(row['run_dir'])
            check_hashes(row['artifacts_sha256'], [str((folder/n).relative_to(ROOT))
                for n in ('final.pt', 'initial.pt', 'config.json', 'provenance.json', 'summary.json')])
            assert row['artifacts_sha256'][row['run_dir']+'/final.pt'] == registered_row['final_checkpoint_sha256']
            assert row['start_frames'] == TARGET
            continue
        assert row['start_frames'] == (continuations[row['logical_name']]['saved_frames'] if row['mode'] == 'resume_v2' else c['start_frames'] if c else 0)
        if row['mode'] == 'resume_v2':
            assert row['parent_run'] == registered_row['original_run']
            parent = local(row['parent_run'])
            required = [str((parent/n).relative_to(ROOT)) for n in ('resume.pt', 'initial.pt', 'config.json', 'provenance.json')]
            check_hashes(row['parent_artifacts_sha256'], required)
            prefix = row['parent_run'].rstrip('/')+'/'
            assert row['parent_artifacts_sha256'] == {prefix+k:v for k,v in registered_row['parent_evidence_sha256'].items()}
            assert row['parent_artifacts_sha256'][prefix+'resume.pt'] == registered_row['parent_checkpoint_sha256']
            target = local(row['trainer_output_dir'])
            assert not target.is_relative_to(parent) and not parent.is_relative_to(target), 'Child output overlaps immutable parent'
            assert row['provider_hours'] == continuations[row['logical_name']]['provider_hours']
        else:
            assert row['provider_hours'] == 25.5 and not row.get('parent_artifacts_sha256')
        check_hashes(row['package_manifest'], ('code/train_memory_v2.py', bindings['protocol']['path'], bindings['amendment']['path']))
        assert all(row['package_manifest'].get(k) == v for k, v in row.get('parent_artifacts_sha256', {}).items())
        validate_continuation(row,plan)
        args = launch_args(row, plan)
        # Preserve every scientific flag/value from the original frozen cohort.
        expected = list(original_row['train_argv']); expected[2] = 'code.train_memory_v2'
        expected[expected.index('--out')+1] = row['trainer_output_dir']
        expected += ['--amendment', bindings['amendment']['path']]
        if row['mode'] == 'resume_v2': expected += ['--parent-run', row['parent_run']]
        if c: expected += ['--resume']
        assert row['train_argv'] == expected, 'Scientific training argv changed'
        prepared = cloud.plan(args)
        assert {e['path']: e['sha256'] for e in prepared['package_manifest']} == row['package_manifest']
        assert prepared['reserved_usd'] == row['reserved_usd'] == round((row['provider_hours']+.25)*1.25, 4)
    extras = plan['other_reservations']
    assert len(extras) == len({r['name'] for r in extras}) == 9
    assert sum(r['purpose'] == 'analysis' and r['reserved_usd'] == 2.8125 for r in extras) == 8
    assert sum(r['purpose'] == 'validation' and r['reserved_usd'] == .9375 for r in extras) == 1
    assert not set(attempts) & {r['name'] for r in extras}
    registry = json.loads(local(bindings['analysis_registry']['path']).read_text())
    assert registry['status'] == 'frozen'
    analysis_rows = [r for r in extras if r['purpose'] == 'analysis']
    assert {r['name'] for r in analysis_rows} == {r['name'] for r in registry['jobs']}
    assert {(r['seed'], r['task_max_gap']) for r in analysis_rows} == {(s,g) for s in range(1,5) for g in (0,60)}
    for row in analysis_rows:
        registered = next(r for r in registry['jobs'] if r['name'] == row['name'])
        assert (row['seed'],row['task_max_gap']) == (registered['seed'],registered['task_max_gap'])
    validation = next(r for r in extras if r['purpose'] == 'validation')
    assert validation['name'] == VALIDATION_NAME and validation['expected_worker'] == {
        k:validation_worker[k] for k in ('name','zone','machine','command','output_dir','package_manifest_sha256')}
    assert validation['instance_id'] == plan['gpu_release']['instance_id']
    return plan


def authenticate(row, folder, plan):
    from scripts.simple_memory_endpoint import authenticate_run
    endpoint = authenticate_run(folder, root=ROOT, amendment_path=local(plan['bindings']['amendment']['path']))
    assert (endpoint['condition'], endpoint['training_seed'], endpoint['training_task_max_gap']) == (
        row['condition'], row['seed'], row['task_max_gap'])
    assert endpoint['frames'] == TARGET and endpoint['updates'] == 78125 and endpoint['decisions'] == 250_000_000
    expected = 1 if row['mode'] == 'retained_v1' else 2
    assert endpoint['producer_version'] == expected, 'Wrong producer version for logical endpoint'
    continuation_inputs(row,folder)
    return endpoint


def ensure_budget(plan, ledger):
    records = {r['name']: r for r in ledger['runs']}
    assert len(records) == len(ledger['runs']), 'Duplicate attempt name in ledger'
    pending = [r for r in plan['models'] if r['mode'] != 'retained_v1']+plan['other_reservations']
    for row in plan['models']:
        c = continuation(row)
        if c is None: continue
        assert c['source_worker_name'] in records, 'Original host missing from ledger'
        old = records[c['source_worker_name']]
        assert str(old.get('instance_id')) == str(c['source_instance_id']), 'Original host ID changed'
        assert old.get('status') == 'DELETED' and old.get('collected_at'), 'Original host must be collected and closed before continuation'
        assert 0 <= old.get('actual_upper_bound_usd',float('inf')) <= 1, 'Original host closeout exceeds the authorized envelope'
        assert old.get('local_artifacts') == str(Path(c['source_run']).parent)
        assert old.get('worker_result') == json.loads(local(c['source_worker_result']['path']).read_text())
    for row in pending:
        if row['name'] in records:
            existing = records[row['name']]
            assert existing['reserved_usd'] == row['reserved_usd'], 'Existing reservation mismatch'
            if row.get('purpose') == 'validation':
                assert str(existing.get('instance_id')) == row['instance_id'], 'Validation worker ID changed'
                assert all(existing.get(k) == v for k,v in row['expected_worker'].items()), 'Validation worker identity changed'
            if row.get('purpose') and 'actual_upper_bound_usd' in existing:
                assert existing.get('worker_result', {}).get('exit_code') == 0, 'A failed/unverified auxiliary job cannot satisfy a future reservation'
    missing = sum(r['reserved_usd'] for r in pending if r['name'] not in records)
    total = cloud.committed(ledger)+missing
    assert total <= 600, 'All remaining training, analysis and validation reservations exceed cumulative $600'
    return dict(committed_usd=cloud.committed(ledger), not_yet_launched_reserved_usd=missing, full_scope_upper_usd=total)


def matching_record(row, record, state, plan):
    args = launch_args(row, plan)
    assert record.get('instance_id') and re.fullmatch(r'[0-9]+', str(record['instance_id'])), 'Unbound attempt requires explicit root reconciliation'
    assert not state.get('instance_id') or str(record['instance_id']) == state['instance_id'], 'Instance ID changed'
    for key in ('name', 'machine', 'zone'):
        assert record[key] == getattr(args, key)
    assert record['command'] == row['train_argv'] and record['output_dir'] == row['worker_output_dir']
    assert record['duration_seconds'] == row['provider_hours']*3600 and record['reserved_usd'] == row['reserved_usd']
    assert record['gcs_prefix'] == f'gs://{cloud.BUCKET}/{cloud.PREFIX}/{row["name"]}'
    assert {e['path']: e['sha256'] for e in record['package_manifest']} == row['package_manifest']


def technical(text, row, record, now=None):
    if not text or '\n' not in text: return None
    raw = json.loads(text[:text.rfind('\n')].splitlines()[-1])
    keys = ('frames', 'updates', 'fps', 'seconds', 'attempt', 'loss', 'grad_norm', 'replay_logit_max_error', 'replay_value_max_error')
    m = {k: raw[k] for k in keys}
    assert all(math.isfinite(v) for v in m.values()), 'Nonfinite technical metric'
    assert row['start_frames'] < m['frames'] <= TARGET and m['fps'] > 0 and m['seconds'] > 0 and m['attempt'] == expected_attempt(row)
    assert m['frames'] == m['updates']*12800, 'Absolute original training counters changed'
    delta = m['frames']-row['start_frames']
    assert math.isclose(m['fps'], delta/m['seconds'], rel_tol=1e-9), 'FPS must describe this physical attempt only'
    now = time.time() if now is None else now
    deadline = dt.datetime.fromisoformat(record['provider_deadline_utc'])
    assert deadline.tzinfo is not None
    margin = deadline.timestamp()-now-(TARGET-m['frames'])/m['fps']-600
    m.update(new_frames=delta, provider_deadline_utc=record['provider_deadline_utc'],
        projected_training_seconds=(TARGET-m['frames'])/m['fps'], completion_sync_reserve_seconds=600,
        margin_seconds=margin, projection_enforced=delta >= 2_000_000)
    if m['projection_enforced']:
        assert margin >= 0, 'Actual FPS plus 600s completion reserve cannot fit the provider deadline'
    return m


def finish(row, done, plan):
    error = None; result = None
    with cloud.locked_ledger() as ledger:
        record = cloud.record(ledger, row['name'])
        matching_record(row, record, {}, plan)
        assert done['package_sha256'] == record['package_manifest_sha256'], 'DONE package mismatch'
        vm = cloud.describe(record['name'], record['zone'])
        if vm is not None: cloud.validate_identity(record, vm)
        with contextlib.redirect_stdout(io.StringIO()): cloud.collect(record)
        folder = ROOT/record['local_artifacts']/Path(row['trainer_output_dir']).name
        try:
            assert done['exit_code'] == 0, 'Worker failed; preserve this attempt, no retry'
            endpoint = authenticate(row, folder, plan)
            result = dict(complete=True, run_dir=str(folder.relative_to(ROOT)), endpoint=endpoint,
                done=done, completed_at=cloud.now())
        except Exception as exc: error = f'{row["name"]}: {type(exc).__name__}: {exc}'
        # A failed worker is still collected/deleted safely before the supervisor halts.
        if vm is not None:
            cloud.run(['gcloud', 'compute', 'instances', 'delete', record['name'], '--project', cloud.PROJECT,
                '--zone', record['zone'], '--quiet'], timeout=180)
        cloud.refresh(record)
    if error: raise RuntimeError(error)
    return result


def cycle(plan, state):
    ledger = json.loads(cloud.LEDGER.read_text()); records = {r['name']: r for r in ledger['runs']}
    for row in plan['models']:
        status = state['models'].setdefault(row['logical_name'], {})
        if status.get('complete'): continue
        if row['mode'] == 'retained_v1':
            status.update(complete=True, run_dir=row['run_dir'], endpoint=authenticate(row, local(row['run_dir']), plan))
            continue
        if row['name'] not in records: continue
        rec = records[row['name']]; matching_record(row, rec, status, plan)
        status['instance_id'] = str(rec['instance_id'])
        done = cat(rec['gcs_prefix']+'/DONE.json')
        if done:
            status.update(finish(row, json.loads(done), plan)); continue
        text = cat(rec['gcs_prefix']+'/outputs/'+Path(row['trainer_output_dir']).name+f'/metrics_{expected_attempt(row):03d}.jsonl', True)
        m = technical(text, row, rec)
        if m:
            vm = cloud.describe(rec['name'], rec['zone'])
            assert vm is not None, 'Absent VM cannot supply current technical health'
            cloud.validate_identity(rec, vm); assert vm['status'] == 'RUNNING'
            assert m['frames'] >= status.get('technical', {}).get('frames', row['start_frames'])
            status['technical'] = m
        if 'actual_upper_bound_usd' in rec or time.time() > dt.datetime.fromisoformat(rec['provider_deadline_utc']).timestamp()+90:
            raise RuntimeError(f'{row["name"]}: closed/expired without DONE; no retry')
    for row in plan['models']:
        if row['mode'] == 'retained_v1': continue
        ledger = json.loads(cloud.LEDGER.read_text())
        if state['models'].get(row['logical_name'], {}).get('complete'):
            assert any(r['name'] == row['name'] for r in ledger['runs']), 'Completed attempt missing from ledger; do not relaunch'
            continue
        if any(r['name'] == row['name'] for r in ledger['runs']): continue
        if sum('actual_upper_bound_usd' not in r for r in ledger['runs']) >= 8: break
        state['budget'] = ensure_budget(plan, ledger)
        check_hashes(row['package_manifest'])
        args = launch_args(row, plan)
        assert {e['path']: e['sha256'] for e in cloud.plan(args)['package_manifest']} == row['package_manifest']
        with contextlib.redirect_stdout(io.StringIO()): cloud.launch(args)
        rec = cloud.record(json.loads(cloud.LEDGER.read_text()), row['name'])
        matching_record(row, rec, {}, plan)
        assert 'actual_upper_bound_usd' not in rec, 'Launch failed; no automatic retry'
        state['models'][row['logical_name']].update(instance_id=str(rec['instance_id']), launched_at=cloud.now())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default='docs/research/simple_memory_recovery_host_plan.json')
    p.add_argument('--out', default='results/ring_simple_memory_recovery_supervision')
    p.add_argument('--max-hours', type=float, default=72)
    args = p.parse_args(argv); assert 0 < args.max_hours <= 72
    plan_path = local(args.plan); plan = validate(plan_path)
    out = local(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(plan_sha256=sha(plan_path), supervisor_sha256=sha(__file__))
        state = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else dict(binding=binding, models={})
        assert state['binding'] == binding and not state.get('halted'), 'Halted or changed supervisor needs explicit root reconciliation'
        assert set(state['models']) <= {r['logical_name'] for r in plan['models']}, 'Unexpected logical model in cached state'
        try:
            for row in plan['models']:
                cached = state['models'].get(row['logical_name'], {})
                if cached.get('complete'):
                    assert authenticate(row, local(cached['run_dir']), plan) == cached['endpoint'], 'Cached endpoint changed'
            deadline = time.monotonic()+args.max_hours*3600
            while time.monotonic() < deadline:
                started = time.monotonic()
                assert sha(plan_path) == binding['plan_sha256'] and sha(__file__) == binding['supervisor_sha256']
                check_hashes(plan['operational_source_sha256'])
                check_hashes({v['path']: v['sha256'] for v in plan['bindings'].values()})
                cycle(plan, state)
                completed = sum(bool(x.get('complete')) for x in state['models'].values())
                state.update(updated_at=cloud.now(), complete=completed == 16)
                save(out/'status.json', state)
                print(json.dumps(dict(at=state['updated_at'], completed=completed, complete=state['complete'])), flush=True)
                if state['complete']: return 0
                time.sleep(max(0, 60-(time.monotonic()-started)))
            raise TimeoutError('Supervisor time limit; all existing provider deadlines remain active')
        except BaseException as exc:
            state.update(halted=True, updated_at=cloud.now(), error=f'{type(exc).__name__}: {exc}')
            save(out/'status.json', state); print(json.dumps(dict(halted=True, error=state['error'])), flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
