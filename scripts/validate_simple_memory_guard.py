"""Bounded GPU validation of the diagnosed replay-check amendment.

Reconstruct four preserved failed attempts, verify their accepted updates, and
compare a single guarded update with the original A2C arithmetic. This script
never produces a scientific training endpoint or changes an input checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from code import train_memory as base
from code import train_memory_v2 as amended
from code.simple_memory_replay_guard import validate_replay, ReplayGuardError
from scripts import diagnose_simple_memory_replay as diagnosis


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def reference_update():
    """Original arithmetic, minus exactly two assertions, ONLY for validation."""
    source = inspect.getsource(base.train_update)
    tree = ast.parse(source)
    function = tree.body[0]
    removed = []
    kept = []
    for statement in function.body:
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value.func) == 'torch.testing.assert_close'):
            removed.append(ast.unparse(statement))
        else:
            kept.append(statement)
    require(len(removed) == 2, 'Original verification statements changed')
    require(all('rtol=0.0001' in line and 'atol=1e-05' in line for line in removed),
            'Original verification tolerances changed')
    function.body = kept
    function.name = 'validation_reference_update'
    namespace = dict(vars(base))
    exec(compile(ast.fix_missing_locations(tree), '<validation-only-original-update>', 'exec'), namespace)
    return namespace[function.name], dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        removed_statements=removed, scope='Validation reference only; never used by cohort training')


def accepted_rows(run, start, end):
    rows = [json.loads(line) for line in (Path(run)/'metrics_000.jsonl').read_text().splitlines() if line.strip()]
    require(rows[-1]['updates'] == end, 'Archived final accepted update changed')
    selected = [row for row in rows if start < row['updates'] <= end]
    require([row['updates'] for row in selected] == list(range(start+1, end+1)),
            'Missing or duplicate accepted reconstruction rows')
    return selected


def gradient_digest(policy):
    return amended.tree_digest([parameter.grad for parameter in policy.parameters()])


def corruption_controls(policy, trace):
    """Deliberate faults on the retained real rollout; no backward or optimizer."""
    roll, state0 = trace['roll'], trace['state0']
    logits, values, state = base.replay(policy, roll, state0)
    logits, values = logits.detach(), values.detach()
    del state
    online, online_v = torch.stack(roll['actor_logits']), torch.stack(roll['actor_values'])
    require(validate_replay(policy, roll, state0, logits, values, online, online_v)['fallback'],
            'Corruption positive control did not exercise fallback')
    result = {}
    for kind in ('rgb', 'code', 'goal', 'initial_h', 'initial_c', 'reset', 'large_logits', 'large_values', 'large_embeddings'):
        bad_roll = {key: list(value) for key, value in roll.items()}
        bad_state = tuple(value.clone() for value in state0)
        ll, vv = logits.clone(), values.clone()
        if kind in ('rgb', 'code', 'goal'):
            bad_roll[kind][0] = bad_roll[kind][0].clone() + 1
        elif kind.startswith('initial_'):
            bad_state[0 if kind == 'initial_h' else 1].add_(0.5)
        elif kind == 'reset':
            bad_roll['d'][0] = 1-bad_roll['d'][0]
        elif kind == 'large_logits':
            ll.add_(0.01)
        elif kind == 'large_values':
            vv.add_(0.01)
        original_embed = policy.embed
        if kind == 'large_embeddings':
            require('embed' not in policy.__dict__, 'Unexpected policy instance encoder override')
            def corrupted_embed(rgb):
                output = original_embed(rgb)
                return output + 0.01 if len(rgb) > len(roll['pa'][0]) else output
            policy.embed = corrupted_embed
        try:
            try:
                validate_replay(policy, bad_roll, bad_state, ll, vv, online, online_v)
            except ReplayGuardError as error:
                result[kind] = dict(rejected=True, rejection_stage=error.record['rejection_stage'])
            else:
                raise RuntimeError('Guard accepted deliberate corruption: '+kind)
        finally:
            if kind == 'large_embeddings':
                del policy.embed
    require(len(result) == 9 and all(item['rejected'] for item in result.values()), 'Incomplete corruption controls')
    return result


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Validation deadline reached; no continuation')


def execute(manifest_path, out, max_seconds=900):
    require(0 < max_seconds <= 900, 'Validation must be bounded by 900 seconds')
    require(torch.cuda.is_available(), 'CUDA required: never replay real study checkpoints on local CPU')
    out = Path(out)
    require(not out.exists() or not any(out.iterdir()), 'Preserve previous validation artifacts')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + max_seconds
    result = dict(complete=False, started_utc=diagnosis.utc(), cases=[],
                  scope='Numerical validation on temporary copies only; no scientific continuation or final endpoint')
    base.atomic_json(out/'summary.json', result)
    preserved_inputs = {str(Path(manifest_path).resolve()): base.sha(manifest_path)}
    try:
        manifest = read(manifest_path)
        require(manifest['status'] == 'frozen' and manifest['maximum_seconds'] == 900,
                'Validation plan must be frozen')
        source_hashes = manifest['source_sha256']
        required_sources = set(amended.SOURCES) | {'scripts/validate_simple_memory_guard.py',
            'scripts/diagnose_simple_memory_replay.py', 'tests/test_validate_simple_memory_guard.py',
            'tests/test_simple_memory_replay_guard.py', 'tests/test_train_memory_v2.py'}
        require(set(source_hashes) == required_sources, 'Incomplete validation source closure')
        for name, digest in source_hashes.items():
            require(base.sha(ROOT/name) == digest, 'Changed validation source: '+name)
        amendment_path = ROOT/manifest['amendment_path']
        require(base.sha(amendment_path) == manifest['amendment_sha256'], 'Amendment changed after prospective freeze')
        amendment = read(amendment_path)
        preserved_inputs[str(amendment_path.resolve())] = base.sha(amendment_path)
        training_protocol_path = ROOT/'docs/research/simple_memory_training_protocol.json'
        require(base.sha(training_protocol_path) == amendment['parent_protocol_sha256'],
                'Original training protocol changed')
        preserved_inputs[str(training_protocol_path.resolve())] = amendment['parent_protocol_sha256']
        required_cases = {key for key, value in amendment['run_bindings'].items() if value['kind'] == 'v1_resume'}
        require(len(manifest['cases']) == 4 and {c['logical_run_id'] for c in manifest['cases']} == required_cases,
                'Validate all four preserved failures without selection')
        known_case = [c for c in manifest['cases'] if c['logical_run_id'] == 'pgc33/blackout/seed1']
        require(len(known_case) == 1 and known_case[0].get('diagnostic_boundary') and
                known_case[0].get('diagnostic_boundary_sha256'), 'Missing exact 188-update boundary control')
        require(sum(c['last_successful_update']-c['start_update'] for c in manifest['cases']) == 375,
                'Reconstruction scope changed')
        for case in manifest['cases']:
            parent = (ROOT/case['run']).resolve()
            require(parent.is_relative_to(ROOT/'results/ring_cloud_artifacts') and
                    not out.resolve().is_relative_to(parent), 'Validation output overlaps immutable parent')
            for name, digest in amendment['run_bindings'][case['logical_run_id']]['parent_evidence_sha256'].items():
                source_path = amended._safe_file(parent, name)
                require(base.sha(source_path) == digest, 'Parent evidence changed before validation')
                preserved_inputs[str(source_path.resolve())] = digest
            if case.get('diagnostic_boundary'):
                boundary_path = (ROOT/case['diagnostic_boundary']).resolve()
                require(boundary_path.is_relative_to(ROOT/'results/ring_cloud_artifacts') and
                        not out.resolve().is_relative_to(boundary_path.parent), 'Output overlaps diagnostic input')
                preserved_inputs[str(boundary_path)] = case['diagnostic_boundary_sha256']
        reference, description = reference_update()
        result.update(manifest_sha256=base.sha(manifest_path), amendment_sha256=base.sha(amendment_path),
                      source_sha256=source_hashes, reference=description)
        for case in manifest['cases']:
            check_deadline(deadline)
            run = ROOT/case['run']
            cfg = {key: value for key, value in read(run/'config.json').items() if key != 'population'}
            cfg.update(device='cuda:0', threads=2, protocol=str(ROOT/'docs/research/simple_memory_training_protocol.json'),
                       out=str(out), resume=True, max_seconds=max_seconds)
            proof = read(run/'provenance.json')
            runtime_versions = diagnosis.authenticate_cuda(proof)
            registered, binding = amended.load_amendment(amendment_path, cfg)
            amended.require_implemented_guard(registered)
            require(amended.logical_run_id(cfg) == case['logical_run_id'], 'Validation identity changed')
            saved, lineage = amended.authenticate_parent(run, binding, cfg, proof)
            require(saved['updates'] == case['start_update'] and binding['last_logged_update'] == case['last_successful_update'],
                    'Validation boundary changed')
            policy, optimizer, source, runtime = base.fresh_training(cfg)
            require(base.provenance(cfg) == proof, 'Numerical runtime/source changed')
            amended.restore_parent(saved, policy, optimizer, runtime, cfg, proof)
            row = dict(logical_run_id=case['logical_run_id'], complete=False,
                       parent_checkpoint_sha256=base.sha(run/'resume.pt'), software=runtime_versions,
                       restored_frames=saved['frames'], restored_updates=saved['updates'], accepted_reconstructed=0)
            result['cases'].append(row)
            base.atomic_json(out/'summary.json', result)
            path = out/(case['logical_run_id'].replace('/', '_')+'_accepted.jsonl')
            with path.open('x') as stream:
                for expected in accepted_rows(run, saved['updates'], binding['last_logged_update']):
                    check_deadline(deadline)
                    actual, _ = amended.train_update(policy, optimizer, source, runtime, cfg)
                    equality = {key: actual[key] == expected[key] for key in diagnosis.TECHNICAL}
                    require(all(equality.values()), 'Amended accepted update differs: '+str(expected['updates']))
                    require(actual['replay_guard']['fallback'] is False, 'Previously accepted update needed new guard fallback')
                    stream.write(json.dumps(dict(update=actual['updates'], technical={key: actual[key] for key in diagnosis.TECHNICAL},
                                                 matches_original=equality, fallback=False))+'\n')
                    stream.flush()
                    row['accepted_reconstructed'] += 1
                    if row['accepted_reconstructed'] % 20 == 0:
                        base.atomic_json(out/'summary.json', result)
            boundary = base.snapshot(policy, optimizer, runtime, cfg, proof)
            row['boundary_payload_sha256'] = amended.payload_digest(boundary)
            if case.get('diagnostic_boundary'):
                known_path = ROOT/case['diagnostic_boundary']
                require(base.sha(known_path) == case['diagnostic_boundary_sha256'], 'Preserved diagnostic boundary changed')
                known = torch.load(known_path, map_location='cpu', weights_only=False)['original_training_snapshot']
                require(amended.payload_digest(known) == row['boundary_payload_sha256'],
                        'All accepted-update policy/optimizer/environment/RNG state differs from exact prior reconstruction')
                row['all_188_updates_full_boundary_exact'] = True
                del known
            check_deadline(deadline)
            try:
                base.train_update(policy, optimizer, source, runtime, cfg)
            except AssertionError as error:
                row['original_assertion_failure'] = str(error)
            else:
                raise RuntimeError('Original failure did not recur; stop attribution')
            amended.restore_parent(boundary, policy, optimizer, runtime, cfg, proof)
            check_deadline(deadline)
            expected_metrics, _, reference_trace = reference(policy, optimizer, source, runtime, cfg, return_trace=True)
            expected_payload = amended.payload_digest(base.snapshot(policy, optimizer, runtime, cfg, proof))
            expected_gradients = gradient_digest(policy)
            amended.restore_parent(boundary, policy, optimizer, runtime, cfg, proof)
            check_deadline(deadline)
            actual_metrics, _ = amended.train_update(policy, optimizer, source, runtime, cfg)
            guard = actual_metrics.pop('replay_guard')
            require(guard['fallback'] is True, 'Known failed update did not exercise the proposed fallback')
            require(actual_metrics == expected_metrics, 'Guard changed original update metrics')
            actual_payload = amended.payload_digest(base.snapshot(policy, optimizer, runtime, cfg, proof))
            actual_gradients = gradient_digest(policy)
            require(actual_payload == expected_payload, 'Guard changed policy/optimizer/environment/RNG state')
            require(actual_gradients == expected_gradients, 'Guard changed original gradients')
            require(runtime['counters']['updates'] == case['last_successful_update']+1, 'Validation executed beyond one fault update')
            require(base.sha(run/'resume.pt') == row['parent_checkpoint_sha256'], 'Input checkpoint changed')
            row.update(failed_update=runtime['counters']['updates'], fallback=guard,
                       original_update_metrics_exact=True, full_state_and_rng_exact=True, gradients_exact=True,
                       reference_payload_sha256=expected_payload, guarded_payload_sha256=actual_payload,
                       reference_gradients_sha256=expected_gradients, guarded_gradients_sha256=actual_gradients,
                       accepted_log_sha256=base.sha(path))
            if case['logical_run_id'] == 'pgc33/blackout/seed1':
                amended.restore_parent(boundary, policy, optimizer, runtime, cfg, proof)
                before_controls = amended.payload_digest(base.snapshot(policy, optimizer, runtime, cfg, proof))
                before_gradients = gradient_digest(policy)
                row['corruption_controls'] = corruption_controls(policy, reference_trace)
                require(amended.payload_digest(base.snapshot(policy, optimizer, runtime, cfg, proof)) == before_controls and
                        gradient_digest(policy) == before_gradients, 'Corruption tests changed the validation model or RNG')
            row['complete'] = True
            base.atomic_json(out/'summary.json', result)
            del policy, optimizer, source, runtime, boundary, saved, reference_trace
            gc.collect()
            torch.cuda.empty_cache()
        require(all(row['complete'] for row in result['cases']) and len(result['cases']) == 4, 'Incomplete fault validation')
        result['complete'] = True
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        result.update(finished_utc=diagnosis.utc(), seconds=time.monotonic()-started)
        if 'source_sha256' in result:
            result['sources_unchanged'] = all(base.sha(ROOT/name) == digest for name, digest in result['source_sha256'].items())
            result['complete'] &= result['sources_unchanged']
        result['all_input_bytes_preserved'] = all(Path(path).is_file() and base.sha(path) == digest
                                                 for path, digest in preserved_inputs.items())
        result['input_sha256'] = preserved_inputs
        result['complete'] &= result['all_input_bytes_preserved']
        base.atomic_json(out/'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    args = parser.parse_args(argv)
    def alarm(_signum, _frame):
        raise TimeoutError('Hard validation wall-clock cap')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(min(args.max_seconds, 900))
    try:
        result = execute(args.manifest, args.out, args.max_seconds)
    finally:
        signal.alarm(0)
    print(json.dumps({key: result.get(key) for key in ('complete', 'seconds', 'error')}), flush=True)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
