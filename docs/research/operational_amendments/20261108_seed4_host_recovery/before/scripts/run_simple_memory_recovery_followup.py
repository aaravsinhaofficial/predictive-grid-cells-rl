"""Bounded analysis of all sixteen authenticated original/recovery endpoints.

Operational adapter only. No training or analysis algorithm is changed. The
original four completed producers and twelve guard-versioned producers retain
separate truthful provenance. No analysis launches until ALL sixteen are final;
no failed or expired analysis attempt is automatically retried.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
import time
import datetime as dt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ring_cloud as cloud
from scripts import run_simple_memory_followup as original
from scripts import run_simple_memory_recovery as recovery
from scripts.run_simple_memory_cohort import cat, save, sha

PLAN = ROOT/'docs/research/simple_memory_recovery_followup_runtime_plan.json'
RECOVERY_PLAN = ROOT/'docs/research/simple_memory_recovery_runtime_plan.json'
TRAINING_STATUS = ROOT/'results/ring_simple_memory_recovery_supervision/status.json'
ANALYSIS_PROTOCOL = ROOT/'docs/research/simple_memory_analysis_protocol.json'
ORIGINAL_PLAN = ROOT/'docs/research/simple_memory_followup_plan.json'
INPUTS = ('config.json', 'provenance.json', 'summary.json', 'final.pt')
OPERATIONS = ('scripts/run_simple_memory_recovery_followup.py',
    'scripts/run_simple_memory_recovery.py', 'scripts/run_simple_memory_followup.py',
    'scripts/run_simple_memory_analysis.py', 'scripts/simple_memory_endpoint.py',
    'scripts/ring_cloud.py', 'scripts/run_simple_memory_cohort.py',
    'tests/test_run_simple_memory_recovery_followup.py')
WORKER_HOURS = 2
RUNNER_MAX_SECONDS = 6600
WORKER_RESERVATION_USD = (WORKER_HOURS + .25) * 1.25  # Existing rate and upload allowance.
TOTAL_ANALYSIS_RESERVATION_USD = 8 * WORKER_RESERVATION_USD


def local(name):
    name = Path(name)
    assert not name.is_absolute() and '..' not in name.parts
    result = ROOT/name
    assert result.resolve().is_relative_to(ROOT.resolve()) and not result.is_symlink()
    return result


def check_files(manifest):
    assert isinstance(manifest, dict) and manifest
    for name, digest in manifest.items(): assert sha(local(name)) == digest, name


def draft_plan(recovery_path=RECOVERY_PLAN):
    """Read-only plan composition; caller explicitly writes a new draft file."""
    recovery_plan = json.loads(Path(recovery_path).read_text())
    template = json.loads(ORIGINAL_PLAN.read_text())
    analysis = json.loads(ANALYSIS_PROTOCOL.read_text())
    return dict(status='draft', purpose=__doc__,
        recovery_plan_path=str(Path(recovery_path).resolve().relative_to(ROOT)),
        recovery_plan_sha256=sha(recovery_path), recovery_supervisor_sha256=sha(ROOT/'scripts/run_simple_memory_recovery.py'),
        original_followup_plan_sha256=sha(ORIGINAL_PLAN),
        analysis_protocol_sha256=sha(ANALYSIS_PROTOCOL),
        training_protocol_sha256=sha(local(analysis['training_protocol'])),
        amendment_sha256=recovery_plan['bindings']['amendment']['sha256'],
        operational_source_sha256={p: sha(ROOT/p) for p in OPERATIONS},
        cumulative_ceiling_usd=600, per_worker_hours=WORKER_HOURS, runner_max_seconds=RUNNER_MAX_SECONDS,
        per_worker_reservation_usd=WORKER_RESERVATION_USD,
        max_active_workers=8, total_analysis_reservation_usd=TOTAL_ANALYSIS_RESERVATION_USD,
        worker_sources=['scripts/run_simple_memory_analysis.py', 'scripts/simple_memory_endpoint.py'],
        jobs=template['jobs'],
        readiness=dict(analysis_authentication_manifest_frozen=False, independent_operational_review=False),
        budget_rule='Before each launch use recovery.ensure_budget with the live ledger: all remaining training, validation and eight analysis reservations must fit cumulative $600.',
        input_rule='All16 finals authenticate before launch. Stage base four inputs; v2 also initial/lineage, and resumed v2 migration boundary plus every bound parent-evidence file. Preserve original evidence and failed attempts.',
        retry_rule='One immutable worker identity per original fixed analysis job; no retry after failed/expired/missing terminal evidence. No scientific-positive selection.')


def validated_plan(path=PLAN):
    plan = json.loads(Path(path).read_text())
    assert plan['status'] == 'frozen' and plan['readiness'] and all(v is True for v in plan['readiness'].values())
    assert plan['cumulative_ceiling_usd'] == cloud.BUDGET == 600
    assert plan['recovery_plan_sha256'] == sha(local(plan['recovery_plan_path']))
    assert plan['recovery_supervisor_sha256'] == sha(ROOT/'scripts/run_simple_memory_recovery.py')
    assert plan['original_followup_plan_sha256'] == sha(ORIGINAL_PLAN)
    assert plan['analysis_protocol_sha256'] == sha(ANALYSIS_PROTOCOL)
    assert set(OPERATIONS) <= set(plan['operational_source_sha256'])
    check_files(plan['operational_source_sha256'])
    analysis = json.loads(ANALYSIS_PROTOCOL.read_text())
    assert analysis['status'] == 'frozen'
    assert plan['training_protocol_sha256'] == sha(local(analysis['training_protocol']))
    training = json.loads(local(analysis['training_protocol']).read_text())
    recovered = recovery.validate(local(plan['recovery_plan_path']))
    assert recovered['bindings']['amendment']['sha256'] == plan['amendment_sha256']
    assert recovered['bindings']['protocol']['sha256'] == plan['training_protocol_sha256']
    amendment = json.loads(local(recovered['bindings']['amendment']['path']).read_text())
    manifests = (analysis['source_sha256'], analysis['validation_artifacts_sha256'],
                 training['source_sha256'], amendment['source_sha256'], plan['operational_source_sha256'])
    combined = {}
    for manifest in manifests:
        check_files(manifest)
        for name, digest in manifest.items():
            assert name not in combined or combined[name] == digest, f'Conflicting frozen closure: {name}'
            combined[name] = digest
    assert set(plan['worker_sources']) == {'scripts/run_simple_memory_analysis.py','scripts/simple_memory_endpoint.py'}
    template = json.loads(ORIGINAL_PLAN.read_text())
    assert plan['jobs'] == template['jobs'], 'Original eight paired analysis jobs/settings changed'
    assert len(plan['jobs']) == len({j['name'] for j in plan['jobs']}) == 8
    assert {(j['seed'],j['task']) for j in plan['jobs']} == {(s,t) for s in (1,2,3,4) for t in ('native','blackout')}
    assert plan['per_worker_hours'] == WORKER_HOURS and plan['runner_max_seconds'] == RUNNER_MAX_SECONDS
    assert plan['per_worker_reservation_usd'] == WORKER_RESERVATION_USD == 2.8125
    assert plan['max_active_workers'] == 8 and plan['total_analysis_reservation_usd'] == TOTAL_ANALYSIS_RESERVATION_USD == 22.5
    reservations = {r['name']:r for r in recovered['other_reservations'] if r['purpose'] == 'analysis'}
    assert set(reservations) == {j['name'] for j in plan['jobs']}
    assert all(r['reserved_usd'] == WORKER_RESERVATION_USD for r in reservations.values())
    assert {n for j in plan['jobs'] for n in j['training_jobs'].values()} == {r['logical_name'] for r in recovered['models']}
    for job in plan['jobs']:
        assert job['name'] == f"ring-20261105-simplemem-analysis-{job['task']}-s{job['seed']}"
        assert job['output_dir'] == f"results/ring_simple_memory_analysis/seed{job['seed']}_{job['task']}"
        assert job['machine'] == 'n1-standard-8' and job['task_max_gap'] == (60 if job['task']=='blackout' else 0)
    return plan, recovered


def check_poll_bindings(path, plan, recovered, binding):
    """Recheck small immutable bindings without rereading checkpoint packages.

    Full recovery validation runs once at startup. Recovery itself authenticates
    parent packages before launch; this controller authenticates all completed
    endpoints and each relocated pair before staging/launch. Waiting polls need
    only source/protocol/plan identity, not hundreds of MB of immutable parents.
    """
    assert sha(path) == binding['plan_sha256'], 'Followup plan changed during polling'
    assert sha(__file__) == binding['controller_sha256'], 'Followup source changed during polling'
    assert sha(local(plan['recovery_plan_path'])) == plan['recovery_plan_sha256'] == binding['recovery_plan_sha256']
    assert sha(ANALYSIS_PROTOCOL) == plan['analysis_protocol_sha256'] == binding['analysis_protocol_sha256']
    assert sha(ORIGINAL_PLAN) == plan['original_followup_plan_sha256']
    check_files(plan['operational_source_sha256'])
    check_files(recovered['operational_source_sha256'])
    check_files({v['path']:v['sha256'] for v in recovered['bindings'].values()})
    check_files(recovered['validation_artifacts_sha256'])
    release = recovered['gpu_release']
    check_files({release['summary_path']:release['summary_sha256'], release['done_path']:release['done_sha256']})
    analysis = json.loads(ANALYSIS_PROTOCOL.read_text())
    check_files(analysis['source_sha256'])
    check_files(analysis['validation_artifacts_sha256'])
    training = json.loads(local(analysis['training_protocol']).read_text())
    check_files(training['source_sha256'])
    amendment = json.loads(local(recovered['bindings']['amendment']['path']).read_text())
    check_files(amendment['source_sha256'])


def training_endpoints(plan, recovered, status):
    assert status['binding'] == dict(plan_sha256=plan['recovery_plan_sha256'],
        supervisor_sha256=plan['recovery_supervisor_sha256']), 'Recovery supervisor binding changed'
    models = {r['logical_name']:r for r in recovered['models']}
    assert len(models) == 16 and set(status['models']) <= set(models), 'Recovery cohort changed'
    if status.get('halted'): raise RuntimeError('Recovery supervisor halted; no analysis promotion')
    if not status.get('complete'): return None
    assert set(status['models']) == set(models), 'Missing final logical models'
    result = {}
    for name, model in models.items():
        row = status['models'][name]
        assert row['complete'] is True
        endpoint = row['endpoint']
        assert (endpoint['frames'],endpoint['updates'],endpoint['decisions']) == (1_000_000_000,78125,250_000_000)
        assert (endpoint['condition'],endpoint['training_seed'],endpoint['training_task_max_gap']) == (
            model['condition'],model['seed'],model['task_max_gap'])
        assert endpoint['producer_version'] == (1 if model['mode']=='retained_v1' else 2)
        folder = local(row['run_dir'])
        expected = (local(model['run_dir']) if model['mode']=='retained_v1' else
                    ROOT/'results/ring_cloud_artifacts'/model['name']/Path(model['trainer_output_dir']).name)
        assert folder.resolve() == expected.resolve(), 'Endpoint path does not match its registered attempt'
        for file,key in [('final.pt','checkpoint_sha256'),('config.json','config_sha256'),
                         ('summary.json','terminal_summary_sha256'),('provenance.json','training_provenance_sha256')]:
            assert sha(folder/file) == endpoint[key], f'Final input changed: {name}/{file}'
        result[name] = folder
    return result


def authenticate_endpoints(recovered, endpoints):
    result = {r['logical_name']:recovery.authenticate(r,endpoints[r['logical_name']],recovered) for r in recovered['models']}
    assert len(result) == 16
    return result


def required_inputs(run, endpoint):
    """Exact portable input allowlist, including the complete registered lineage."""
    from scripts.simple_memory_endpoint import checked_file
    run = Path(run); names = set(INPUTS)
    if endpoint['producer_version'] == 2:
        names.update(('initial.pt','lineage.json'))
        lineage = json.loads(checked_file(run,'lineage.json').read_text())
        assert sha(run/'lineage.json') == endpoint['lineage_sha256']
        assert lineage['kind'] == endpoint['producer_assignment']
        if lineage['kind'] == 'v1_resume':
            names.add('migration_boundary.pt')
            evidence = lineage['parent_evidence_sha256']
            assert {'resume.pt','initial.pt','config.json','provenance.json'} <= set(evidence)
            for name,digest in evidence.items():
                path = checked_file(run/'parent_evidence',name)
                assert sha(path) == digest
                names.add('parent_evidence/'+name)
        else:
            assert lineage['kind'] == 'v2_fresh' and not lineage.get('parent_evidence_sha256')
    else:
        assert endpoint['producer_version'] == 1
    return {name:sha(checked_file(run,name)) for name in sorted(names)}


def static_files(plan, recovered):
    analysis = json.loads(ANALYSIS_PROTOCOL.read_text())
    training = json.loads(local(analysis['training_protocol']).read_text())
    amendment_path = recovered['bindings']['amendment']['path']
    amendment = json.loads(local(amendment_path).read_text())
    files = (set(analysis['source_sha256']) | set(analysis['validation_artifacts_sha256']) |
             set(training['source_sha256']) | set(amendment['source_sha256']) | set(plan['worker_sources']) |
             {'docs/research/simple_memory_analysis_protocol.json',analysis['training_protocol'],amendment_path})
    return sorted(files)


def stage_inputs(plan, recovered, job, endpoints, expected):
    models = {r['logical_name']:r for r in recovered['models']}
    manifests = {}
    for condition,name in job['training_jobs'].items():
        source = endpoints[name]
        endpoint = recovery.authenticate(models[name],source,recovered)
        assert endpoint == expected[name], 'Final identity changed before staging'
        target = local(job['output_dir'])/condition
        target.mkdir(parents=True,exist_ok=True)
        manifest = required_inputs(source,endpoint)
        for filename,digest in manifest.items():
            dest = target/filename; dest.parent.mkdir(parents=True,exist_ok=True)
            assert dest.resolve().is_relative_to(target.resolve()), 'Nonlocal staging path'
            if dest.exists():
                assert dest.is_file() and not dest.is_symlink() and sha(dest) == digest, f'Staged evidence changed: {dest}'
            else:
                tmp = dest.with_name(dest.name+'.tmp')
                shutil.copyfile(Path(source)/filename,tmp)
                assert sha(tmp) == digest
                tmp.replace(dest)
        # Independently authenticate the relocated copies, not just source hashes.
        assert recovery.authenticate(models[name],target,recovered) == endpoint
        assert required_inputs(target,endpoint) == manifest
        manifests[condition] = manifest
    return manifests


def launch_args(plan, recovered, job, expected):
    base = Path(job['output_dir'])
    command = ['python','scripts/run_simple_memory_analysis.py','--runs',str(base/'gc100'),str(base/'pgc33'),
               '--out',str(base/'pipeline.json'),'--max-seconds',str(plan['runner_max_seconds'])]
    include = static_files(plan,recovered)
    for condition,name in job['training_jobs'].items():
        include += [str(base/condition/p) for p in required_inputs(local(base/condition),expected[name])]
    return argparse.Namespace(name=job['name'],machine=job['machine'],zone=job['zone'],hours=plan['per_worker_hours'],
        output_dir=str(base),command=shlex.join(command),include=sorted(set(include)),include_only=True)


def check_worker_limits(record):
    """The prepared reservation and existing immutable worker must agree."""
    assert record['duration_seconds'] == WORKER_HOURS * 3600 == 7200
    assert record['reserved_usd'] == WORKER_RESERVATION_USD == 2.8125


def match_record(plan, recovered, job, expected, record, instance_id=None):
    assert re.fullmatch('[0-9]+',str(record.get('instance_id',''))), 'Unbound analysis worker; explicit review required'
    if instance_id is not None: assert str(record['instance_id']) == str(instance_id)
    args = launch_args(plan,recovered,job,expected)
    for key in ('name','machine','zone','output_dir'): assert record[key] == getattr(args,key)
    assert record['command'] == shlex.split(args.command)
    assert record['gcs_prefix'] == f'gs://{cloud.BUCKET}/{cloud.PREFIX}/{args.name}'
    check_worker_limits(record)
    manifest = {x['path']:x['sha256'] for x in record['package_manifest']}
    assert set(manifest) == set(args.include)
    check_files(manifest)


# Reuse the unchanged terminal collection/identity/deletion and scientific
# pipeline validators. They take explicit job/inputs and consult no old plan.
finish = original.finish
verify_pair = original.verify_pair


def verify_completed_cache(plan, recovered, state):
    ledger = json.loads(cloud.LEDGER.read_text())
    for job in plan['jobs']:
        status = state['jobs'].get(job['name'],{})
        if not status.get('complete'): continue
        record = cloud.record(ledger,job['name'])
        match_record(plan,recovered,job,state['authenticated_training_endpoints'],record,status['instance_id'])
        folder = local(status['local_artifacts'])
        assert folder.resolve() == (ROOT/'results/ring_cloud_artifacts'/job['name']).resolve()
        assert sha(folder/'pipeline.json') == status['pipeline_sha256']
        assert verify_pair(folder,job,status['inputs']) == status['verification']


def cycle(plan, recovered, state):
    if not TRAINING_STATUS.is_file():
        state.update(phase='waiting_for_recovery_supervisor',training_complete_count=0)
        return
    training = json.loads(TRAINING_STATUS.read_text())
    endpoints = training_endpoints(plan,recovered,training)
    if endpoints is None:
        state.update(phase='waiting_for_all16_authenticated_finals',training_updated_at=training.get('updated_at'),
            training_complete_count=sum(bool(v.get('complete')) for v in training['models'].values()))
        return
    authenticated = authenticate_endpoints(recovered,endpoints)
    if 'authenticated_training_endpoints' in state:
        assert state['authenticated_training_endpoints'] == authenticated, 'Previously authenticated endpoint changed'
    else: state['authenticated_training_endpoints'] = authenticated
    for name,endpoint in authenticated.items(): assert endpoint == training['models'][name]['endpoint']
    state.update(phase='analysis',training_complete_count=16)
    ledger = json.loads(cloud.LEDGER.read_text()); records = {r['name']:r for r in ledger['runs']}
    assert len(records) == len(ledger['runs'])
    for job in plan['jobs']:
        name = job['name']; status = state['jobs'].setdefault(name,{})
        if status.get('complete'):
            assert name in records, 'Completed analysis missing from ledger; never relaunch'
            continue
        if name not in records: continue
        record = records[name]
        match_record(plan,recovered,job,authenticated,record,status.get('instance_id'))
        status['instance_id'] = str(record['instance_id'])
        inputs = {c:authenticated[n] for c,n in job['training_jobs'].items()}
        if 'inputs' not in status: status['inputs'] = inputs
        assert status['inputs'] == inputs
        done = cat(record['gcs_prefix']+'/DONE.json')
        if done:
            status.update(finish(job,json.loads(done),inputs)); continue
        vm = cloud.describe(record['name'],record['zone'])
        if vm is not None:
            cloud.validate_identity(record,vm)
            status.update(live_status=vm['status'],live_checked_at=cloud.now())
        if 'actual_upper_bound_usd' in record or time.time()>dt.datetime.fromisoformat(record['provider_deadline_utc']).timestamp()+90:
            raise RuntimeError(f'{name}: closed/expired without terminal artifact; no retry')
    for job in plan['jobs']:
        ledger = json.loads(cloud.LEDGER.read_text())
        if any(r['name']==job['name'] for r in ledger['runs']): continue
        if sum('actual_upper_bound_usd' not in r for r in ledger['runs']) >= plan['max_active_workers']: break
        state['budget'] = recovery.ensure_budget(recovered,ledger)
        staged = stage_inputs(plan,recovered,job,endpoints,authenticated)
        args = launch_args(plan,recovered,job,authenticated)
        prepared = cloud.plan(args)
        check_worker_limits(prepared)
        with contextlib.redirect_stdout(io.StringIO()): cloud.launch(args)
        record = cloud.record(json.loads(cloud.LEDGER.read_text()),job['name'])
        match_record(plan,recovered,job,authenticated,record)
        assert 'actual_upper_bound_usd' not in record, 'Launch failed; no retry'
        state['jobs'][job['name']].update(instance_id=str(record['instance_id']),staged_input_sha256=staged,
            inputs={c:authenticated[n] for c,n in job['training_jobs'].items()},launch_observed_at=cloud.now())
    state['complete'] = all(state['jobs'].get(j['name'],{}).get('complete') for j in plan['jobs'])
    if state['complete']: state['phase'] = 'complete'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--out',type=Path,default=ROOT/'results/ring_simple_memory_recovery_analysis_supervision')
    parser.add_argument('--max-hours',type=float,default=72)
    parser.add_argument('--write-draft',type=Path)
    args = parser.parse_args(argv)
    if args.write_draft:
        assert not args.write_draft.exists(), 'Do not overwrite an existing plan'
        save(args.write_draft,draft_plan())
        return 0
    assert 0 < args.max_hours <= 72
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,recovered = validated_plan(args.plan)
        binding = dict(plan_sha256=sha(args.plan),controller_sha256=sha(__file__),
            recovery_plan_sha256=plan['recovery_plan_sha256'],analysis_protocol_sha256=sha(ANALYSIS_PROTOCOL))
        path = args.out/'status.json'
        state = json.loads(path.read_text()) if path.exists() else dict(binding=binding,jobs={},complete=False)
        assert state['binding'] == binding and not state.get('halted'), 'Changed/halted controller requires explicit review'
        assert set(state['jobs']) <= {j['name'] for j in plan['jobs']}
        deadline = time.monotonic()+args.max_hours*3600
        try:
            verify_completed_cache(plan,recovered,state)
            while time.monotonic()<deadline:
                started = time.monotonic()
                check_poll_bindings(args.plan,plan,recovered,binding)
                cycle(plan,recovered,state)
                state['updated_at'] = cloud.now(); save(path,state)
                print(json.dumps({k:state.get(k) for k in ('updated_at','phase','training_complete_count','complete')}),flush=True)
                if state['complete']: return 0
                time.sleep(max(0,60-(time.monotonic()-started)))
            raise TimeoutError('Recovery followup monitor time limit; provider DELETE deadlines remain active')
        except BaseException as exc:
            state.update(halted=True,updated_at=cloud.now(),error=f'{type(exc).__name__}: {exc}')
            save(path,state); print(json.dumps(dict(halted=True,error=state['error'])),flush=True)
            return 1


if __name__ == '__main__': raise SystemExit(main())
