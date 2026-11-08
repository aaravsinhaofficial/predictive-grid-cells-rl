"""CPU-only technical audit of the fixed sixteen training endpoints.

No network calls, policy/environment construction, forwards, or outcome analysis.
Only technical metric fields are inspected; episode logs are hashed, never parsed.
The CLI writes a new directory and requires all sixteen endpoints by default.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from code import train_memory_v2 as producer
from scripts import run_simple_memory_recovery as recovery
from scripts.simple_memory_endpoint import checked_file, sha

FINAL_UPDATES = 78125
FRAMES_PER_UPDATE = 12800
DECISIONS_PER_UPDATE = 3200
FINAL_FRAMES = 1_000_000_000
FINAL_DECISIONS = 250_000_000
COLLECTION_SUMMARY = 'results/ring_simple_memory_recovery_inventory/collection/summary.json'
TECHNICAL = ('loss', 'policy_loss', 'value_loss', 'entropy', 'grad_norm',
             'replay_logit_max_error', 'replay_value_max_error', 'seconds', 'fps')
INVARIANTS = ('actor_shaped_logits_and_values_exact', 'original_shaped_logits_and_values_exact',
              'encoder_original_tolerance_passed', 'log_probability_original_tolerance_passed',
              'critic_original_tolerance_passed', 'inputs_parameters_buffers_gradients_unmodified',
              'rng_unmodified')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def logical_id(row):
    return f"{row['condition']}/{'native' if row['task_max_gap'] == 0 else 'blackout'}/seed{row['seed']}"


def finite_tree(value, counts=None):
    """Check saved tensors recursively, without interpreting behavioral counters."""
    counts = counts if counts is not None else dict(torch_tensors=0, floating_tensors=0, numpy_arrays=0)
    if isinstance(value, torch.Tensor):
        counts['torch_tensors'] += 1
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), 'Nonfinite saved tensor')
            counts['floating_tensors'] += 1
    elif isinstance(value, np.ndarray):
        counts['numpy_arrays'] += 1
        require(not np.issubdtype(value.dtype, np.inexact) or bool(np.isfinite(value).all()), 'Nonfinite saved array')
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v, counts)
    elif isinstance(value, (tuple, list)):
        for v in value: finite_tree(v, counts)
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), 'Nonfinite saved scalar')
    return counts


def policy_hash(policy):
    require(bool(policy), 'Empty policy')
    h = hashlib.sha256()
    for name, tensor in sorted(policy.items()):
        require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), 'Invalid policy tensor')
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_evidence(saved, updates):
    """Endpoint participation counters; not evidence of nonzero gradients each step."""
    optimizer=saved['optimizer']; groups=optimizer['param_groups']; policy=saved['policy']
    require(len(groups)==1, 'Expected one full-policy RMSprop parameter group')
    group=groups[0]; ids=group['params']
    require(len(ids)==len(policy)==20 and all(type(i) is int for i in ids) and len(set(ids))==20,
            'RMSprop does not cover all twenty unique policy parameters')
    # The frozen single-group optimizer serializes policy.parameters() as 0..19.
    # Require that same ordered mapping at initialization and every boundary.
    require(ids==list(range(20)), 'RMSprop parameter identity/order changed from the frozen initial group')
    expected=dict(lr=.0003,alpha=.99,eps=1e-5,momentum=0,weight_decay=0,centered=False)
    require(all(group.get(k)==v for k,v in expected.items()) and group['centered'] is False,
            'Frozen RMSprop hyperparameters changed')
    states=optimizer['state']
    require(set(states)==(set(ids) if updates else set()), 'Missing/extra RMSprop parameter states')
    for parameter_id, (name, parameter) in zip(ids,policy.items()):
        if not updates:
            continue
        state=states[parameter_id]
        require(set(state)=={'step','square_avg'}, 'Unexpected RMSprop state fields')
        step=state['step']; square=state['square_avg']
        require(isinstance(step,torch.Tensor) and step.ndim==0 and bool(torch.isfinite(step)) and
                step.item()==updates, 'RMSprop parameter step differs from checkpoint update')
        require(isinstance(square,torch.Tensor) and square.shape==parameter.shape and square.dtype==parameter.dtype,
                'RMSprop accumulator does not match policy parameter: '+name)
    return dict(parameter_groups=1,policy_parameter_count=20,unique_parameter_ids=ids,
        state_entries=len(states),all_parameter_steps_equal_checkpoint_updates=True if updates else None,
        checkpoint_updates=updates,initial_optimizer_state_empty=not bool(states),
        hyperparameters=expected,
        scope='Saved optimizer coverage and step counters; no assertion of nonzero gradients at every update.')


def checkpoint(path, cfg, proof, updates):
    """Load one checkpoint at a time; return compact evidence, never weights."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved['config'] == cfg and saved['provenance'] == proof, 'Checkpoint config/provenance mismatch')
    require(saved['schema'] == 1 and saved['updates'] == updates and
            saved['frames'] == updates * FRAMES_PER_UPDATE, 'Checkpoint counters mismatch')
    counters = saved['runtime']['counters']
    require((counters['updates'], counters['frames'], counters['decisions']) ==
            (updates, updates * FRAMES_PER_UPDATE, updates * DECISIONS_PER_UPDATE), 'Runtime counters mismatch')
    require(saved['policy_training'] is True, 'Checkpoint is not in the frozen training mode')
    if updates == 0:
        require(saved['optimizer']['state'] == {} and all(v == 0 for v in counters.values()),
                'Initial checkpoint is not fresh: optimizer/counters are nonzero')
    digest = policy_hash(saved['policy'])
    require(digest == saved['policy_sha256'], 'Saved policy digest mismatch')
    return dict(sha256=sha(path), payload_sha256=producer.payload_digest(saved), policy_sha256=digest,
                policy_tensor_count=len(saved['policy']),
                policy_schema={k:list(v.shape) for k,v in sorted(saved['policy'].items())},
                updates=updates, optimizer_participation=optimizer_evidence(saved,updates), finite=finite_tree(saved))


def guard_check(guard):
    require(guard['accepted'] is True and guard['guard_id'] == producer.GUARD_ID, 'Rejected/unknown replay guard')
    require(type(guard['fallback']) is bool and guard['fallback_count'] == int(guard['fallback']), 'Guard fallback count')
    require(guard['tolerance'] == dict(rtol=1e-4, atol=1e-5), 'Guard tolerance changed')
    if not guard['fallback']:
        require(guard['mode'] == 'original', 'Unknown normal guard mode')
        return False
    require(guard['mode'] == 'bounded_cnn_batch_roundoff', 'Unknown fallback mode')
    require(all(guard[k] is True for k in INVARIANTS) and guard['learning_tensors_replaced'] is False,
            'Fallback invariance failed')
    for key, limit in [('max_encoder_abs', 1e-5), ('max_raw_logit_abs', 1e-4)]:
        require(math.isfinite(guard[key]) and 0 <= guard[key] <= limit, 'Fallback bound failed: ' + key)
    return True


def metric_log(path, *, attempt, start, end, version, accepted_end=None):
    """Stream all raw rows, counting only the explicitly accepted prefix."""
    accepted_end = end if accepted_end is None else accepted_end
    require(start <= accepted_end <= end, 'Invalid accepted cutoff')
    digest = hashlib.sha256(); count = 0; previous_seconds = 0.; fallback = []
    max_encoder = max_logit = 0.
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line)
            require(line.endswith(b'\n') and line.strip(), 'Incomplete/empty technical metric row')
            m = json.loads(line); count += 1; update = start + count
            require((m['updates'], m['frames'], m['decisions'], m['attempt'], m['rollout_steps']) ==
                    (update, update * FRAMES_PER_UPDATE, update * DECISIONS_PER_UPDATE, attempt, 100),
                    'Noncontiguous metric counters/attempt')
            require(all(type(m[k]) in (int, float) and math.isfinite(m[k]) for k in TECHNICAL), 'Nonfinite technical metric')
            require(m['seconds'] > previous_seconds and m['fps'] > 0, 'Invalid metric clock/FPS')
            require(m['grad_norm'] >= 0 and m['replay_logit_max_error'] >= 0 and m['replay_value_max_error'] >= 0,
                    'Negative technical norm/error')
            require(math.isclose(m['fps'], (update-start)*FRAMES_PER_UPDATE/m['seconds'], rel_tol=1e-9),
                    'FPS is not attempt-local')
            previous_seconds = m['seconds']
            if version == 2:
                g = m['replay_guard']
                if guard_check(g):
                    fallback.append(update)
                    max_encoder = max(max_encoder, g['max_encoder_abs'])
                    max_logit = max(max_logit, g['max_raw_logit_abs'])
            else:
                require('replay_guard' not in m, 'v1 metric contains an amended guard')
    require(count == end-start, 'Missing or extra technical metric rows')
    return dict(sha256=digest.hexdigest(), raw_rows=count, accepted_rows=accepted_end-start,
                excluded_parent_suffix_rows=end-accepted_end, first_update=start+1, last_update=end,
                seconds=previous_seconds, fallback_updates=fallback, fallback_count=len(fallback),
                max_fallback_encoder_abs=max_encoder, max_fallback_raw_logit_abs=max_logit)


def audit_run(row, run, endpoint):
    """Detailed CPU-only checks after unchanged endpoint authentication."""
    run = Path(run); cfg = read(checked_file(run, 'config.json')); cfg.pop('population', None)
    proof = read(checked_file(run, 'provenance.json')); terminal = read(checked_file(run, 'summary.json'))
    version = 1 if row['mode'] == 'retained_v1' else 2
    require(terminal['complete'] is True and terminal['provenance'] == proof, 'Invalid terminal summary')
    require((terminal['frames'], terminal['updates'], terminal['decisions']) ==
            (FINAL_FRAMES, FINAL_UPDATES, FINAL_DECISIONS), 'Wrong terminal endpoint')
    require(endpoint['producer_version'] == version and
            (endpoint['frames'], endpoint['updates'], endpoint['decisions']) ==
            (FINAL_FRAMES, FINAL_UPDATES, FINAL_DECISIONS), 'Wrong authenticated endpoint')
    files = {}
    def file(name):
        path = checked_file(run, name); files[name] = sha(path); return path
    final = checkpoint(file('final.pt'), cfg, proof, FINAL_UPDATES)
    resume = checkpoint(file('resume.pt'), cfg, proof, FINAL_UPDATES)
    require(final['payload_sha256'] == resume['payload_sha256'], 'Final/resume full payload differs')
    require(final['sha256'] == terminal['final_sha256'] == endpoint['checkpoint_sha256'] and
            final['policy_sha256'] == terminal['final_policy_sha256'] == endpoint['policy_sha256'], 'Final digest mismatch')
    initial_proof = proof['base_provenance'] if row['mode'] == 'resume_v2' else proof
    initial = checkpoint(file('initial.pt'), cfg, initial_proof, 0)
    require(initial['policy_tensor_count'] == final['policy_tensor_count'] == 20 and
            initial['policy_schema'] == final['policy_schema'], 'Original policy tensor schema changed')
    require(initial['sha256'] == terminal['initial_sha256'], 'Initial checkpoint bytes changed')
    lineage = read(file('lineage.json')) if version == 2 else None
    if lineage:
        require(initial['payload_sha256'] == lineage['initial_payload_sha256'], 'Initial full payload changed')
        require(files['lineage.json'] == proof['lineage_sha256'] == terminal['lineage_sha256'], 'Lineage binding changed')
    host = row.get('continuation'); parent_log = None; origin = 0; start_payload = initial['payload_sha256']
    if row['mode'] == 'resume_v2':
        origin = row['start_frames']//FRAMES_PER_UPDATE
        parent = checkpoint(file('parent_evidence/resume.pt'), cfg, initial_proof, origin)
        migration = checkpoint(file('migration_boundary.pt'), cfg, proof, origin)
        require(parent['payload_sha256'] == migration['payload_sha256'] == lineage['parent_payload_sha256'] ==
                lineage['migrated_payload_sha256'], 'Migration full payload differs')
        require(files['parent_evidence/resume.pt'] == lineage['parent_checkpoint_sha256'], 'Parent checkpoint changed')
        require(sha(file('parent_evidence/initial.pt')) == initial['sha256'], 'Parent initialization differs')
        for name, digest in lineage['parent_evidence_sha256'].items():
            require(sha(file('parent_evidence/'+name)) == digest, 'Parent evidence changed: '+name)
        require(sorted(p.name for p in (run/'parent_evidence').glob('metrics_*.jsonl')) == ['metrics_000.jsonl'], 'Unexpected parent metric attempt')
        require(sorted(p.name for p in (run/'parent_evidence').glob('attempt_*.json')) == ['attempt_000.json'], 'Unexpected parent attempt')
        a = read(file('parent_evidence/attempt_000.json'))
        require((a['attempt'], a['restored_frames'], a['restored_updates']) == (0,0,0) and
                a.get('complete') is not True and not a.get('resume_sha256'), 'Parent origin changed')
        parent_log = metric_log(file('parent_evidence/metrics_000.jsonl'), attempt=0, start=0,
            end=lineage['last_logged_parent_update'], accepted_end=origin, version=1)
        require(parent_log['excluded_parent_suffix_rows'] == lineage['rolled_back_logged_updates'] and
                lineage['accepted_parent_update_cutoff'] == origin, 'Parent cutoff changed')
        start_payload = migration['payload_sha256']
    attempts = [0,1] if host else [0]
    expected_attempts = [f'attempt_{a:03d}.json' for a in attempts]
    require(terminal['attempt_records'] == expected_attempts and
            sorted(p.name for p in run.glob('attempt_*.json')) == expected_attempts, 'Unexpected attempt inventory')
    require(sorted(p.name for p in run.glob('metrics_*.jsonl')) == [f'metrics_{a:03d}.jsonl' for a in attempts], 'Unexpected metric inventory')
    require(sorted(p.name for p in run.glob('episodes_*.jsonl')) == [f'episodes_{a:03d}.jsonl' for a in attempts], 'Missing/extra episode evidence')
    require(not list(run.glob('rejected_update_*.json')), 'Rejected child guard record')
    require(sorted(p.name for p in run.glob('interrupted_*.json')) == (['interrupted_000.json'] if host else []), 'Unexpected interrupted attempt')
    require(sorted(p.name for p in run.glob('resume_input_*.pt')) ==
            ([f'resume_input_{a:03d}.pt' for a in attempts] if version == 2 else []), 'Unexpected supplied checkpoint inventory')
    logs = []; chains = []; start = origin
    for number in attempts:
        a = read(file(f'attempt_{number:03d}.json'))
        end = host['start_updates'] if host and number == 0 else FINAL_UPDATES
        require((a['attempt'], a['restored_updates'], a['restored_frames'], a['final_updates'], a['final_frames']) ==
                (number, start, start*FRAMES_PER_UPDATE, end, end*FRAMES_PER_UPDATE), 'Attempt boundary changed')
        stopping = bool(host and number == 0)
        require(a['complete'] is (not stopping) and a['exit_code'] == (143 if stopping else 0), 'Attempt terminal status changed')
        require(math.isfinite(a['seconds']) and a['seconds'] > 0, 'Invalid attempt duration')
        if version == 2:
            require(a['producer_version'] == 2 and a['logical_run_id'] == logical_id(row) and
                    a['lineage_sha256'] == proof['lineage_sha256'], 'Attempt identity changed')
            require(a['resume_input'] == f'resume_input_{number:03d}.pt', 'Supplied input filename changed')
            incoming = checkpoint(file(a['resume_input']), cfg, proof, start)
            require(incoming['sha256'] == a['resume_sha256'] and incoming['payload_sha256'] == start_payload,
                    'Attempt supplied full payload changed')
        else:
            require(a.get('resume_sha256') is None and 'resume_input' not in a, 'v1 fresh gained a resume')
        if stopping:
            require(read(file('interrupted_000.json')) == a, 'Host interruption record differs')
            boundary = checkpoint(file('resume_input_001.pt'), cfg, proof, end)
            require(a['resume_sha256_after'] == boundary['sha256'] == host['resume_sha256'] and
                    boundary['payload_sha256'] == host['resume_payload_sha256'], 'Host cutoff payload changed')
            for name, digest in host['source_files_sha256'].items():
                require(sha(file('resume_input_001.pt' if name == 'resume.pt' else name)) == digest,
                        'Host immutable origin evidence changed')
            next_payload = boundary['payload_sha256']
        else:
            require(a['resume_sha256_after'] == resume['sha256'], 'Final attempt resume binding differs')
            next_payload = final['payload_sha256']
        log = metric_log(file(f'metrics_{number:03d}.jsonl'), attempt=number, start=start, end=end, version=version)
        require(log['seconds'] <= a['seconds'], 'Metrics exceed attempt duration')
        file(f'episodes_{number:03d}.jsonl')  # Preserve bytes only; never parse outcomes.
        logs.append(dict(attempt=number, **log)); chains.append(dict(attempt=number, start_update=start,
            end_update=end, supplied_payload_sha256=start_payload, final_payload_sha256=next_payload))
        start, start_payload = end, next_payload
    accepted = (parent_log['accepted_rows'] if parent_log else 0) + sum(x['accepted_rows'] for x in logs)
    require(accepted == FINAL_UPDATES, 'Accepted update chain does not cover the endpoint')
    if parent_log:
        require(lineage['last_logged_parent_update']+1 in logs[0]['fallback_updates'], 'Original failing update not retained as accepted fallback')
    for name in ('config.json','provenance.json','summary.json'): file(name)
    return dict(complete=True, logical_run_id=logical_id(row), producer_version=version,
        initial=initial, final=final, exact_final_resume_payload=True, attempt_chains=chains,
        parent_log=parent_log, attempts=logs, accepted_updates=accepted,
        raw_metric_rows=(parent_log['raw_rows'] if parent_log else 0)+sum(x['raw_rows'] for x in logs),
        excluded_parent_suffix_rows=parent_log['excluded_parent_suffix_rows'] if parent_log else 0,
        fallback_count=sum(x['fallback_count'] for x in logs),
        fallback_updates=[u for x in logs for u in x['fallback_updates']],
        rejected_guard_count=0, artifact_sha256=files, endpoint=endpoint)


def registry(plan, root):
    """Read/hash only; never call the controller's launch or cloud planning code."""
    require(plan['status'] == 'frozen', 'Unfrozen recovery plan')
    for manifest in (plan['operational_source_sha256'], plan['validation_artifacts_sha256']):
        for name,digest in manifest.items(): require(sha(checked_file(root,name)) == digest, 'Changed plan-bound source/evidence: '+name)
    for binding in plan['bindings'].values():
        require(sha(checked_file(root,binding['path'])) == binding['sha256'], 'Changed plan binding')
    original = read(root/plan['bindings']['original_plan']['path'])
    amendment = read(root/plan['bindings']['amendment']['path'])
    rows = plan['models']; ids = [r['logical_name'] for r in rows]
    expected = {(c,s,g) for c in ('gc100','pgc33') for s in range(1,5) for g in (0,60)}
    require(len(ids) == len(set(ids)) == 16 and {(r['condition'],r['seed'],r['task_max_gap']) for r in rows} == expected,
            'Fixed sixteen-model registry changed')
    old = {r['name']:r for r in original['jobs']}
    require(set(ids) == set(old) and set(amendment['run_bindings']) == {logical_id(r) for r in rows}, 'Original registry differs')
    require({m:sum(r['mode']==m for r in rows) for m in recovery.MODES} == recovery.MODES, 'Producer assignments changed')
    require(sum('continuation' in r for r in rows) == 1, 'Host continuation registry changed')
    for row in rows:
        require(all(old[row['logical_name']][k] == row[k] for k in ('condition','seed','task_max_gap')), 'Logical identity changed')
        binding = amendment['run_bindings'][logical_id(row)]
        require(binding['logical_job_name'] == row['logical_name'] and binding['kind'] ==
                {'retained_v1':'v1_completed','resume_v2':'v1_resume','fresh_v2':'v2_fresh'}[row['mode']], 'Producer assignment differs')
        recovery.continuation(row)  # Pure validation of the one explicitly registered host move.
    return old


def worker_evidence(row, observed, run, record, original, plan):
    require(record['status'] == 'DELETED' and record.get('collected_at') and record.get('instance_id'), 'Worker completion/identity absent')
    require(Path(record['local_artifacts']) == run.parent.relative_to(ROOT), 'Collected worker directory changed')
    result_path = checked_file(run.parent,'WORKER_RESULT.json'); done = read(result_path)
    require(done == record['worker_result'] and done['exit_code'] == 0, 'Worker terminal evidence differs')
    manifest = record['package_manifest']; package = json_hash(manifest)
    require(len(manifest) == len({e['path'] for e in manifest}) and package == record['package_manifest_sha256'] == done['package_sha256'], 'Package digest differs')
    if row['mode'] == 'retained_v1':
        require(record['name'] == row['logical_name'] and record['command'] == original['train_argv'] and
                record['output_dir'] == original['worker_output_dir'], 'Original launch changed')
        for name,digest in row['artifacts_sha256'].items(): require(sha(checked_file(ROOT,name)) == digest, 'Retained endpoint changed')
        collection_path=checked_file(ROOT,COLLECTION_SUMMARY)
        require(sha(collection_path)==plan['validation_artifacts_sha256'][COLLECTION_SUMMARY], 'Collection registry changed')
        matches=[r for r in read(collection_path)['jobs'] if r['name']==row['logical_name']]
        require(len(matches)==1 and matches[0]['complete'] is True, 'Retained collection identity absent')
        entry=matches[0]; receipt_path=checked_file(collection_path.parent,entry['receipt'])
        require(sha(receipt_path)==entry['receipt_sha256'], 'Retained collection receipt changed')
        receipt=read(receipt_path)
        require(receipt['complete'] is True and receipt['name']==record['name'] and
                str(receipt['instance_id'])==str(record['instance_id']) and receipt['gcs_prefix']==record['gcs_prefix'] and
                (receipt['condition'],receipt['training_seed'],receipt['training_task_max_gap'])==
                (row['condition'],row['seed'],row['task_max_gap']) and
                receipt['final']['sha256']==sha(run/'final.pt'), 'Retained immutable worker identity differs')
    else:
        recovery.matching_record(row,record,observed,plan)
        require(done == observed['done'] and str(record['instance_id']) == observed['instance_id'], 'Supervisor/worker identity differs')
    # Frozen ring_cloud.WORKER substitutes sys.executable for python/python3.
    # The pinned T4 image resolves that interpreter to /usr/bin/python3.
    resolved_command=list(record['command'])
    if resolved_command[0] in ('python','python3'):
        resolved_command[0]='/usr/bin/python3'
    require(done['command'] == resolved_command, 'Worker command/interpreter differs')
    proof = read(run/'provenance.json'); packaged = {e['path']:e['sha256'] for e in manifest}
    require(all(packaged.get(k) == v for k,v in proof['source_sha256'].items()) and
            packaged.get(plan['bindings']['protocol']['path']) == proof['protocol_sha256'], 'Producer/package closure differs')
    return dict(worker_name=record['name'], instance_id=str(record['instance_id']), exit_code=0,
        package_manifest_sha256=package, worker_result_sha256=sha(result_path), worker_record_sha256=json_hash(record))


def audit_cohort(plan_path, status_path, ledger_path, out):
    torch.set_num_threads(1)
    plan_path, status_path, ledger_path, out = map(Path,(plan_path,status_path,ledger_path,out))
    input_bytes={p:p.read_bytes() for p in (plan_path,status_path,ledger_path)}
    snapshots=out/'input_snapshots';snapshots.mkdir(exist_ok=False)
    input_snapshots={}
    for label,path in [('plan',plan_path),('status',status_path),('ledger',ledger_path)]:
        dest=snapshots/(label+'.json');dest.write_bytes(input_bytes[path])
        input_snapshots[label]=dict(path=str(dest.relative_to(out)),sha256=sha(dest))
    plan, status, ledger = (json.loads(input_bytes[p]) for p in (plan_path,status_path,ledger_path))
    original = registry(plan,ROOT)
    require(status['binding'] == dict(plan_sha256=sha(plan_path), supervisor_sha256=sha(ROOT/'scripts/run_simple_memory_recovery.py')),
            'Supervisor status binding changed')
    require(set(status['models']) <= {r['logical_name'] for r in plan['models']}, 'Unexpected status logical ID')
    records = {r['name']:r for r in ledger['runs']}; require(len(records)==len(ledger['runs']), 'Duplicate ledger identity')
    models={}; initial_groups={str(s):{} for s in range(1,5)}
    for row in plan['models']:
        name=row['logical_name']; observed=status['models'].get(name,{})
        try:
            if observed.get('complete') is not True:
                require(row['mode'] != 'retained_v1', 'Retained final missing from supervisor')
                require(not observed.get('done') or observed['done'].get('exit_code') == 0, 'Known terminal worker failure')
                record=records.get(row.get('name'),{})
                require(not record.get('worker_result') or record['worker_result']['exit_code'] == 0, 'Known failed worker is not pending')
                require(record.get('status') != 'DELETED' or record.get('worker_result'), 'Deleted worker lacks terminal evidence')
                models[name]=dict(status='pending', logical_run_id=logical_id(row), reason='Supervisor has not authenticated a final endpoint')
                continue
            run=ROOT/observed['run_dir']
            expected=row['run_dir'] if row['mode']=='retained_v1' else str(Path('results/ring_cloud_artifacts')/row['name']/f"seed{row['seed']}")
            require(observed['run_dir']==expected, 'Collected endpoint routing differs')
            endpoint=recovery.authenticate(row,run,plan)
            require(endpoint==observed['endpoint'], 'Supervisor endpoint evidence changed')
            worker=worker_evidence(row,observed,run,records[row.get('name',name)],original[name],plan)
            receipt=audit_run(row,run,endpoint)
            receipt.update(worker=worker, supervisor_entry_sha256=json_hash(observed))
            if row['mode']=='resume_v2':
                log=row['original_worker_log']; require(sha(checked_file(ROOT,log['path']))==log['sha256'], 'Parent worker log changed')
                receipt['parent_worker_log_sha256']=log['sha256']
                parent_done_path=checked_file(ROOT,str(Path(row['parent_run']).parent/'WORKER_RESULT.json'))
                parent_done=read(parent_done_path); parent_record=records[name]
                require(parent_record['status']=='DELETED' and parent_record.get('instance_id') and
                        parent_done==parent_record['worker_result'] and parent_done['exit_code']==1 and
                        parent_done['package_sha256']==parent_record['package_manifest_sha256'], 'Original failure evidence differs')
                receipt['parent_worker_failure']=dict(exit_code=1,instance_id=str(parent_record['instance_id']),
                    worker_result_sha256=sha(parent_done_path),worker_record_sha256=json_hash(parent_record))
            target=out/(name+'.json'); target.write_text(json.dumps(receipt,indent=2)+'\n')
            models[name]=dict(status='complete',logical_run_id=logical_id(row),receipt=target.name,sha256=sha(target))
            initial_groups[str(row['seed'])][logical_id(row)]=receipt['initial']['policy_sha256']
        except (AssertionError, ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
            models[name]=dict(status='invalid',logical_run_id=logical_id(row),error=type(exc).__name__+': '+str(exc))
    comparisons={}
    for seed,hashes in initial_groups.items():
        comparisons[seed]=dict(completed_models=len(hashes),expected_models=4,policy_sha256=hashes,
            equal_among_available=len(set(hashes.values()))<=1,full_seed_complete=len(hashes)==4)
    counts={k:sum(v['status']==k for v in models.values()) for k in ('complete','pending','invalid')}
    valid=counts['invalid']==0 and all(c['equal_among_available'] for c in comparisons.values())
    complete=valid and counts['complete']==16
    return dict(schema=1,kind='fixed_cohort_training_technical_audit',audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        complete=complete,cohort_complete=complete,validation_passed=valid,counts=counts,models=models,
        initial_policy_equality_by_seed=comparisons,
        input_snapshots=input_snapshots,
        input_snapshot_scope='Exact local metadata evidence; ledger may contain infrastructure details and is not a public receipt.',
        scope='CPU-only saved-state/provenance/technical-metric audit. No policy or environment execution; no navigation or scientific outcome analysis. Pending models are not certified.',
        references_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(input_bytes[p]).hexdigest() if p in input_bytes else sha(p) for p in
            (plan_path,status_path,ledger_path,Path(__file__),ROOT/'tests/test_audit_simple_memory_training.py')},
        controller_source_sha256=sha(ROOT/'scripts/run_simple_memory_recovery.py'),
        endpoint_helper_sha256=sha(ROOT/'scripts/simple_memory_endpoint.py'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',default=str(ROOT/'docs/research/simple_memory_recovery_host_plan.json'))
    ap.add_argument('--status',default=str(ROOT/'results/ring_simple_memory_recovery_supervision/status.json'))
    ap.add_argument('--ledger',default=str(ROOT/'docs/research/ring_cloud_budget.json'))
    ap.add_argument('--out',required=True);ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args(argv);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    try:
        summary=audit_cohort(args.plan,args.status,args.ledger,out)
    except (AssertionError,ValueError,KeyError,OSError,TypeError,RuntimeError) as exc:
        summary=dict(complete=False,cohort_complete=False,validation_passed=False,error=type(exc).__name__+': '+str(exc))
    summary['allow_partial']=args.allow_partial
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    ok=summary['validation_passed'] and (summary['cohort_complete'] or args.allow_partial)
    print(json.dumps(dict(output=str(out/'summary.json'),validation_passed=summary['validation_passed'],
        cohort_complete=summary['cohort_complete'],counts=summary.get('counts'),exit_code=0 if ok else 1)))
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
