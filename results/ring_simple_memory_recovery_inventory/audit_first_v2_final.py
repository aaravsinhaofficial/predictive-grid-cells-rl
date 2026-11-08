"""Local-only technical audit of the first complete guard-versioned endpoint.

Reads checkpoints on CPU. Never constructs/runs a policy or environment, reads
behavioral episode rows, calls the cloud, or changes existing evidence.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from code import train_memory_v2 as producer
from scripts.simple_memory_endpoint import authenticate_run

BASE = Path(__file__).resolve().parent
LOGICAL = 'ring-20261105-simplemem-pgc33-native-s2'
WORKER = LOGICAL + '-r1'
INSTANCE_ID = '2589352373224570581'
torch.set_num_threads(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def policy_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite_tensors(value, path='', counts=None):
    if counts is None:
        counts = dict(torch_tensors=0, floating_torch_tensors=0, numpy_arrays=0, floating_numpy_arrays=0)
    if isinstance(value, torch.Tensor):
        counts['torch_tensors'] += 1
        if value.is_floating_point() or value.is_complex():
            assert bool(torch.isfinite(value).all()), path
            counts['floating_torch_tensors'] += 1
    elif isinstance(value, np.ndarray):
        counts['numpy_arrays'] += 1
        if np.issubdtype(value.dtype, np.inexact):
            assert bool(np.isfinite(value).all()), path
            counts['floating_numpy_arrays'] += 1
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tensors(item, path + '/' + str(key), counts)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tensors(item, path + '/' + str(index), counts)
    return counts


def main():
    plan_path = ROOT / 'docs/research/simple_memory_recovery_runtime_plan.json'
    status_path = ROOT / 'results/ring_simple_memory_recovery_supervision/status.json'
    status_bytes = status_path.read_bytes()
    status, plan = json.loads(status_bytes), read(plan_path)
    row = next(r for r in plan['models'] if r['logical_name'] == LOGICAL)
    observed = status['models'][LOGICAL]
    assert observed.get('complete') is True, 'Collection/authentication not yet complete; not a worker failure'
    assert row['mode'] == 'resume_v2' and row['name'] == WORKER
    assert str(observed['instance_id']) == INSTANCE_ID
    assert status['binding'] == dict(plan_sha256=sha(plan_path), supervisor_sha256=sha(ROOT / 'scripts/run_simple_memory_recovery.py'))
    assert observed['done']['exit_code'] == 0
    run = ROOT / observed['run_dir']
    assert run == ROOT / 'results/ring_cloud_artifacts' / WORKER / 'seed2'
    endpoint = authenticate_run(run)
    assert endpoint == observed['endpoint']
    assert endpoint['producer_version'] == 2 and endpoint['producer_assignment'] == 'v1_resume'
    assert endpoint['logical_run_id'] == 'pgc33/native/seed2'
    assert (endpoint['frames'], endpoint['updates'], endpoint['decisions']) == (1_000_000_000, 78125, 250_000_000)
    final, parent, migration, initial = [torch.load(run / name, map_location='cpu', weights_only=False)
        for name in ('final.pt', 'parent_evidence/resume.pt', 'migration_boundary.pt', 'initial.pt')]
    lineage, terminal, proof = [read(run / name) for name in ('lineage.json', 'summary.json', 'provenance.json')]
    assert (parent['frames'], parent['updates']) == (880640000, 68800)
    assert producer.payload_digest(parent) == producer.payload_digest(migration) == lineage['parent_payload_sha256'] == lineage['migrated_payload_sha256']
    assert initial['frames'] == initial['updates'] == 0
    assert sha(run / 'initial.pt') == sha(run / 'parent_evidence/initial.pt') == terminal['initial_sha256']
    assert producer.payload_digest(initial) == lineage['initial_payload_sha256']
    initial_policy_sha = policy_digest(initial['policy'])
    assert initial_policy_sha == initial['policy_sha256']
    final_policy_sha = policy_digest(final['policy'])
    assert final_policy_sha == final['policy_sha256'] == terminal['final_policy_sha256'] == endpoint['policy_sha256']
    assert (final['runtime']['counters']['frames'], final['runtime']['counters']['updates'], final['runtime']['counters']['decisions']) == (1_000_000_000, 78125, 250_000_000)
    tensor_checks = finite_tensors(final)
    assert terminal['attempt_records'] == ['attempt_000.json']
    assert sorted(p.name for p in run.glob('attempt_*.json')) == ['attempt_000.json']
    assert sorted(p.name for p in run.glob('metrics_*.jsonl')) == ['metrics_000.jsonl']
    attempt = read(run / 'attempt_000.json')
    assert attempt['complete'] is True and attempt['exit_code'] == 0
    assert (attempt['restored_frames'], attempt['restored_updates']) == (880640000, 68800)
    assert (attempt['final_frames'], attempt['final_updates']) == (1_000_000_000, 78125)
    assert attempt['lineage_sha256'] == proof['lineage_sha256']
    assert sha(run / attempt['resume_input']) == attempt['resume_sha256']
    resume_input = torch.load(run / attempt['resume_input'], map_location='cpu', weights_only=False)
    assert producer.payload_digest(resume_input) == producer.payload_digest(migration)
    assert sha(run / 'resume.pt') == attempt['resume_sha256_after']
    resume_final = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
    assert producer.payload_digest(resume_final) == producer.payload_digest(final)
    metrics_text = (run / 'metrics_000.jsonl').read_text()
    assert metrics_text.endswith('\n'), 'Terminal metric log has an incomplete row'
    metrics = [json.loads(line) for line in metrics_text.splitlines() if line.strip()]
    assert len(metrics) == 78125 - 68800 == 9325
    fallback_records = []
    previous_seconds = 0.
    for offset, metric in enumerate(metrics, 1):
        assert metric['updates'] == 68800 + offset and metric['attempt'] == 0
        assert metric['frames'] == metric['updates'] * 12800
        assert metric['decisions'] == metric['updates'] * 3200
        assert metric['rollout_steps'] == 100
        assert metric['seconds'] > previous_seconds
        previous_seconds = metric['seconds']
        for key in ('loss', 'policy_loss', 'value_loss', 'entropy', 'grad_norm', 'replay_logit_max_error', 'replay_value_max_error', 'seconds', 'fps'):
            assert math.isfinite(metric[key]), (metric['updates'], key)
        assert math.isclose(metric['fps'], (metric['frames'] - 880640000) / metric['seconds'], rel_tol=1e-9)
        guard = metric['replay_guard']
        assert guard['accepted'] and guard['guard_id'] == producer.GUARD_ID
        assert guard['fallback_count'] == int(guard['fallback'])
        assert guard['tolerance'] == dict(rtol=1e-4, atol=1e-5)
        if guard['fallback']:
            assert guard['mode'] == 'bounded_cnn_batch_roundoff'
            assert all(guard[k] is True for k in ('actor_shaped_logits_and_values_exact', 'original_shaped_logits_and_values_exact', 'encoder_original_tolerance_passed', 'log_probability_original_tolerance_passed', 'critic_original_tolerance_passed', 'inputs_parameters_buffers_gradients_unmodified', 'rng_unmodified'))
            assert guard['learning_tensors_replaced'] is False
            assert guard['max_encoder_abs'] <= 1e-5 and guard['max_raw_logit_abs'] <= 1e-4
            fallback_records.append(dict(updates=metric['updates'], frames=metric['frames'], guard=guard))
        else:
            assert guard['mode'] == 'original'
    assert metrics[-1]['frames'] == 1_000_000_000 and metrics[-1]['updates'] == 78125
    assert any(r['updates'] == 68859 for r in fallback_records)
    rejected = sorted(p.name for p in run.glob('rejected_update_*.json'))
    interrupted = sorted(p.name for p in run.glob('interrupted_*.json'))
    assert not rejected and not interrupted, (rejected, interrupted)
    detail = BASE / 'first_v2_final_audit_local'
    detail.mkdir(exist_ok=False)
    (detail / 'supervisor_status_snapshot.json').write_bytes(status_bytes)
    bindings = {str((run / name).relative_to(ROOT)): sha(run / name) for name in
        ('config.json', 'provenance.json', 'summary.json', 'final.pt', 'initial.pt', 'lineage.json',
         'migration_boundary.pt', 'resume.pt', 'resume_input_000.pt', 'attempt_000.json', 'metrics_000.jsonl')}
    for name, digest in lineage['parent_evidence_sha256'].items():
        assert sha(run / 'parent_evidence' / name) == digest
        bindings[str((run / 'parent_evidence' / name).relative_to(ROOT))] = digest
    details = dict(complete=True, endpoint=endpoint, floating_tensor_checks=tensor_checks,
        accepted_guard_fallback_records=fallback_records, rejected_update_files=rejected,
        interrupted_attempt_files=interrupted, artifact_sha256=bindings,
        supervisor_status_snapshot_sha256=sha(detail / 'supervisor_status_snapshot.json'))
    details_path = detail / 'details.json'
    details_path.write_text(json.dumps(details, indent=2) + '\n')
    receipt = dict(schema=1, kind='first_guard_versioned_final_technical_audit', complete=True,
        audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='CPU-only local authentication of one completed training endpoint. No policy/environment execution, scientific outcome analysis, cloud access or resource mutation. The receipt does not certify the remaining cohort or its later analyses.',
        worker_name=WORKER, immutable_instance_id=INSTANCE_ID, logical_run_id=endpoint['logical_run_id'],
        producer_version=2, producer_assignment='v1_resume', worker_exit_code=observed['done']['exit_code'],
        supervisor_completed_at=observed['completed_at'],
        final_frames=1_000_000_000, final_updates=78125, final_decisions=250_000_000,
        restored_parent_frames=880640000, restored_parent_updates=68800,
        exact_full_parent_migration_payload=True, unchanged_original_initial_checkpoint=True,
        original_initial_policy_sha256=initial_policy_sha, finite_final_tensors=True,
        final_tensor_counts=tensor_checks, child_complete_metric_rows=9325,
        child_first_update=68801, child_last_update=78125, child_metric_continuity_exact=True,
        original_failure_update=68859, original_failure_crossed=True,
        accepted_guard_fallback_count=len(fallback_records),
        accepted_guard_fallback_updates=[r['updates'] for r in fallback_records],
        every_fallback_bound_and_invariance_check_passed=True,
        rejected_guard_record_count=0, interrupted_attempt_record_count=0,
        parent_rolled_back_log_rows_retained=lineage['rolled_back_logged_updates'],
        parent_cutoff_plus_child_updates=68800 + len(metrics),
        endpoint=endpoint, source_sha256=proof['source_sha256'],
        references_sha256={str(path.relative_to(ROOT)): sha(path) for path in
            (plan_path, ROOT / 'scripts/run_simple_memory_recovery.py', ROOT / 'scripts/simple_memory_endpoint.py',
             ROOT / 'scripts/run_simple_memory_analysis.py', Path(__file__), details_path,
             detail / 'supervisor_status_snapshot.json')},
        run_artifact_sha256=bindings)
    public = BASE / 'first_v2_final_audit.json'
    assert not public.exists()
    public.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(path=str(public.relative_to(ROOT)), sha256=sha(public),
        bytes=public.stat().st_size, child_rows=len(metrics), fallback_count=len(fallback_records),
        fallback_updates=[r['updates'] for r in fallback_records])))


if __name__ == '__main__':
    main()
