"""Wait for all16 exact training finals, then run eight bounded analysis pairs.

Operational orchestration only. Frozen scientific scripts/settings are never
edited; failed jobs are retained for explicit review and never retried here.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import ring_cloud as cloud
from scripts.run_simple_memory_cohort import cat,sha,save

PLAN=ROOT/'docs/research/simple_memory_followup_plan.json'
TRAINING_STATUS=ROOT/'results/ring_simple_memory_supervision/status.json'
TRAINING_PROCESS=ROOT/'results/ring_simple_memory_supervision/process.json'
TRAINING_PLAN=ROOT/'docs/research/simple_memory_cloud_plan.json'
ANALYSIS_PROTOCOL=ROOT/'docs/research/simple_memory_analysis_protocol.json'
INPUTS=('config.json','provenance.json','summary.json','final.pt')


def validated_plan(path=PLAN):
    plan=json.loads(Path(path).read_text())
    assert plan['status']=='frozen' and plan['cumulative_ceiling_usd']==cloud.BUDGET==600
    assert plan['analysis_protocol_sha256']==sha(ANALYSIS_PROTOCOL)
    assert plan['training_plan_sha256']==sha(TRAINING_PLAN)
    protocol=json.loads(ANALYSIS_PROTOCOL.read_text())
    assert protocol['status']=='frozen' and protocol['source_sha256']
    assert plan['training_protocol_sha256']==sha(ROOT/protocol['training_protocol'])
    assert plan['training_supervisor_sha256']==sha(ROOT/'scripts/run_simple_memory_cohort.py')
    required={'scripts/run_simple_memory_followup.py','scripts/run_simple_memory_analysis.py',
              'scripts/ring_cloud.py','scripts/run_simple_memory_cohort.py'}
    assert required<=set(plan['operational_source_sha256'])
    for path in set(protocol['source_sha256'])&set(plan['operational_source_sha256']):
        assert protocol['source_sha256'][path]==plan['operational_source_sha256'][path],'Conflicting source manifests'
    for manifest in (protocol['source_sha256'],protocol['validation_artifacts_sha256'],plan['operational_source_sha256']):
        for path,digest in manifest.items():assert sha(ROOT/path)==digest,path
    assert len(plan['jobs'])==8 and len({j['name'] for j in plan['jobs']})==8
    assert {(j['seed'],j['task']) for j in plan['jobs']}=={(s,t) for s in (1,2,3,4) for t in ('native','blackout')}
    assert plan['per_worker_hours']==1 and plan['runner_max_seconds']==2900
    assert plan['total_analysis_reservation_usd']==8*(1+.25)*cloud.RATE==12.5
    assert plan['max_active_workers']==8
    for job in plan['jobs']:
        assert job['name']==f"ring-20261105-simplemem-analysis-{job['task']}-s{job['seed']}"
        assert job['output_dir']==f"results/ring_simple_memory_analysis/seed{job['seed']}_{job['task']}"
        assert job['task_max_gap']==(60 if job['task']=='blackout' else 0)
        assert job['machine']=='n1-standard-8' and job['zone'] in ('us-central1-a','us-central1-b','us-central1-c')
        for condition in ('gc100','pgc33'):
            assert job['training_jobs'][condition]==f"ring-20261105-simplemem-{condition}-{job['task']}-s{job['seed']}"
    return plan


def live_training_supervisor(process_path=TRAINING_PROCESS):
    """Inspect the actual PID/command; a status file alone is not liveness."""
    info=json.loads(Path(process_path).read_text())
    pid=info['pid']
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    command=result.stdout.strip()
    alive=result.returncode==0 and 'scripts/run_simple_memory_cohort.py' in command
    return dict(pid=pid,alive=alive,command=command,checked_at=cloud.now())


def training_endpoints(plan,status):
    assert status['binding']==dict(plan_sha256=plan['training_plan_sha256'],
        protocol_sha256=plan['training_protocol_sha256'],supervisor_sha256=plan['training_supervisor_sha256']), 'Training supervisor binding changed'
    expected={name for j in plan['jobs'] for name in j['training_jobs'].values()}
    assert len(expected)==16 and set(status['jobs'])==expected,'Training cohort changed'
    if not status.get('complete'):
        return None
    assert not status.get('halted'),'Training supervisor halted despite completion flag'
    result={}
    for name in sorted(expected):
        row=status['jobs'][name]
        assert row['complete'] and row['frames']==1_000_000_000
        summary_path=Path(row['summary_path']).resolve()
        assert summary_path.is_relative_to(ROOT/'results/ring_cloud_artifacts'/name)
        assert sha(summary_path)==row['summary_sha256']
        assert sha(summary_path.parent/'final.pt')==row['final_sha256']
        result[name]=summary_path.parent
    return result


def authenticate_endpoints(plan,endpoints):
    from scripts.run_simple_memory_analysis import authenticate_run
    result={}
    for job in plan['jobs']:
        for condition,name in job['training_jobs'].items():
            endpoint=authenticate_run(endpoints[name])
            assert (endpoint['condition'],endpoint['training_seed'],endpoint['training_task_max_gap'])==(
                condition,job['seed'],job['task_max_gap']),'Training final identity changed'
            result[name]=endpoint
    assert len(result)==16
    return result


def static_files(plan):
    protocol=json.loads(ANALYSIS_PROTOCOL.read_text())
    training=json.loads((ROOT/protocol['training_protocol']).read_text())
    return sorted(set(protocol['source_sha256'])|set(protocol['validation_artifacts_sha256'])|
                  set(training['source_sha256'])|set(plan['worker_sources'])|
                  {'docs/research/simple_memory_analysis_protocol.json',protocol['training_protocol']})


def stage_inputs(job,endpoints):
    from scripts.run_simple_memory_analysis import authenticate_run
    records={}
    for condition,name in job['training_jobs'].items():
        source=Path(endpoints[name])
        endpoint=authenticate_run(source)
        cfg=json.loads((source/'config.json').read_text())
        assert (cfg['cond'],cfg['seed'],cfg['task_max_gap'])==(condition,job['seed'],job['task_max_gap'])
        target=ROOT/job['output_dir']/condition
        target.mkdir(parents=True,exist_ok=True)
        for file in INPUTS:
            src,dst=source/file,target/file
            if dst.exists():
                assert sha(dst)==sha(src),f'Staged input changed: {dst}'
            else:
                tmp=dst.with_suffix(dst.suffix+'.tmp')
                shutil.copyfile(src,tmp)
                assert sha(tmp)==sha(src)
                tmp.replace(dst)
        records[condition]=endpoint
    return records


def launch_args(plan,job):
    base=Path(job['output_dir'])
    command=['python','scripts/run_simple_memory_analysis.py','--runs',
             str(base/'gc100'),str(base/'pgc33'),'--out',str(base/'pipeline.json'),
             '--max-seconds',str(plan['runner_max_seconds'])]
    include=static_files(plan)+[str(base/c/file) for c in ('gc100','pgc33') for file in INPUTS]
    return argparse.Namespace(name=job['name'],machine=job['machine'],zone=job['zone'],
        hours=plan['per_worker_hours'],output_dir=str(base),command=shlex.join(command),
        include=include,include_only=True)


def whole_analysis_budget(plan,ledger):
    names={r['name'] for r in ledger['runs']}
    remaining=sum((plan['per_worker_hours']+.25)*cloud.RATE for j in plan['jobs'] if j['name'] not in names)
    bound=cloud.committed(ledger)+remaining
    if bound>cloud.BUDGET:raise RuntimeError('Remaining full-analysis reservations exceed cumulative600 budget')
    return bound


def match_record(plan,job,record,instance_id=None):
    assert re.fullmatch(r'[0-9]+',str(record.get('instance_id',''))),'Unbound worker identity; explicit review required'
    if instance_id is not None:assert str(record['instance_id'])==str(instance_id)
    args=launch_args(plan,job)
    for key in ('name','machine','zone','output_dir'):assert record[key]==getattr(args,key),key
    assert record['command']==shlex.split(args.command)
    assert record['gcs_prefix']==f'gs://{cloud.BUCKET}/{cloud.PREFIX}/{args.name}'
    assert record['duration_seconds']==3600 and record['reserved_usd']==1.5625
    manifest={x['path']:x['sha256'] for x in record['package_manifest']}
    assert set(manifest)==set(args.include)
    for path,digest in manifest.items():assert sha(ROOT/path)==digest,path


def verify_pair(folder,job,expected_inputs):
    from scripts.run_simple_memory_analysis import verify_pipeline
    proof=verify_pipeline(folder)
    pipeline=json.loads((Path(folder)/'pipeline.json').read_text())
    assert proof['models']==2 and proof['stages']==10,'Fixed Grid/PGC pair is incomplete'
    actual={row['endpoint']['condition']:row['endpoint'] for row in pipeline['runs']}
    assert len(pipeline['runs'])==2 and set(actual)=={'gc100','pgc33'}
    assert actual==expected_inputs,'Analysis pair differs from authenticated original inputs'
    assert all((e['training_seed'],e['training_task_max_gap'])==(job['seed'],job['task_max_gap']) for e in actual.values())
    return proof


def verify_completed_cache(plan,state):
    ledger=json.loads(cloud.LEDGER.read_text())
    for job in plan['jobs']:
        status=state['jobs'].get(job['name'],{})
        if not status.get('complete'):continue
        record=cloud.record(ledger,job['name'])
        match_record(plan,job,record,status['instance_id'])
        folder=ROOT/status['local_artifacts']
        assert folder.resolve()==(ROOT/'results/ring_cloud_artifacts'/job['name']).resolve()
        assert sha(folder/'pipeline.json')==status['pipeline_sha256']
        assert verify_pair(folder,job,status['inputs'])==status['verification']


def finish(job,done,expected_inputs):
    from scripts.run_simple_memory_analysis import verify_pipeline
    error=None;output=None
    with cloud.locked_ledger() as ledger:
        record=cloud.record(ledger,job['name'])
        assert done['package_sha256']==record['package_manifest_sha256']
        vm=cloud.describe(record['name'],record['zone'])
        if vm is not None:cloud.validate_identity(record,vm)
        try:
            with contextlib.redirect_stdout(io.StringIO()):cloud.collect(record)
            folder=ROOT/record['local_artifacts']
            assert done['exit_code']==0,'Analysis worker returned nonzero'
            proof=verify_pair(folder,job,expected_inputs)
            output=dict(complete=True,local_artifacts=str(folder.relative_to(ROOT)),
                pipeline_sha256=sha(folder/'pipeline.json'),verification=proof)
        except Exception as exc:
            error=f"{job['name']}: analysis validation failed: {exc}"
        # Only an identity-verified owned terminal worker is removed. Failed
        # output collection remains explicit; GCS output is not deleted.
        if vm is not None:
            cloud.run(['gcloud','compute','instances','delete',record['name'],'--project',cloud.PROJECT,
                       '--zone',record['zone'],'--quiet'],timeout=180)
        cloud.refresh(record)
    if error:raise RuntimeError(error)
    return output


def cycle(plan,state):
    training=json.loads(TRAINING_STATUS.read_text())
    endpoints=training_endpoints(plan,training)
    if endpoints is None:
        state.update(phase='waiting_for_all16_training_finals',training_updated_at=training['updated_at'],
            training_complete_count=sum(bool(v.get('complete')) for v in training['jobs'].values()),
            training_supervisor=live_training_supervisor(),training_halted=bool(training.get('halted')))
        # No launch, retry or training mutation while any final is missing.
        return
    state['phase']='analysis'
    # Authenticate every final before the first analysis launch. Later cycles
    # bind the same endpoints and still check all source/final file digests.
    if 'authenticated_training_endpoints' not in state:
        state['authenticated_training_endpoints']=authenticate_endpoints(plan,endpoints)
    expected=state['authenticated_training_endpoints']
    for name,folder in endpoints.items():
        for file,key in [('final.pt','checkpoint_sha256'),('config.json','config_sha256'),
                         ('summary.json','terminal_summary_sha256'),('provenance.json','training_provenance_sha256')]:
            assert sha(folder/file)==expected[name][key],f'Authenticated training input changed: {name}/{file}'
    ledger=json.loads(cloud.LEDGER.read_text());records={r['name']:r for r in ledger['runs']}
    for job in plan['jobs']:
        name=job['name'];status=state['jobs'].setdefault(name,{})
        if status.get('complete') or name not in records:continue
        record=records[name];match_record(plan,job,record,status.get('instance_id'))
        status['instance_id']=str(record['instance_id'])
        original={c:expected[n] for c,n in job['training_jobs'].items()}
        if 'inputs' not in status:status['inputs']=original
        assert status['inputs']==original
        done=cat(record['gcs_prefix']+'/DONE.json')
        if done:
            status.update(finish(job,json.loads(done),original))
            continue
        vm=cloud.describe(record['name'],record['zone'])
        if vm is not None:
            cloud.validate_identity(record,vm)
            status.update(live_status=vm['status'],live_checked_at=cloud.now())
        if 'actual_upper_bound_usd' in record or time.time()>dt.datetime.fromisoformat(record['provider_deadline_utc']).timestamp()+90:
            raise RuntimeError(f'{name}: closed/expired without terminal artifact; no retry')
    for job in plan['jobs']:
        ledger=json.loads(cloud.LEDGER.read_text());names={r['name'] for r in ledger['runs']}
        if job['name'] in names:continue
        if sum('actual_upper_bound_usd' not in r for r in ledger['runs'])>=plan['max_active_workers']:break
        state['full_remaining_analysis_bound_usd']=whole_analysis_budget(plan,ledger)
        inputs=stage_inputs(job,endpoints)
        assert inputs=={c:expected[n] for c,n in job['training_jobs'].items()}
        args=launch_args(plan,job)
        prepared=cloud.plan(args)
        assert prepared['reserved_usd']==1.5625 and prepared['duration_seconds']==3600
        with contextlib.redirect_stdout(io.StringIO()):cloud.launch(args)
        record=cloud.record(json.loads(cloud.LEDGER.read_text()),job['name'])
        assert record.get('instance_id') and 'actual_upper_bound_usd' not in record,'Launch failed; no retry'
        state['jobs'][job['name']].update(instance_id=str(record['instance_id']),
            inputs=inputs,launch_observed_at=cloud.now())
    state['complete']=all(state['jobs'].get(j['name'],{}).get('complete') for j in plan['jobs'])
    if state['complete']:state['phase']='complete'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--out',type=Path,default=ROOT/'results/ring_simple_memory_analysis_supervision')
    parser.add_argument('--max-hours',type=float,default=72)
    args=parser.parse_args(argv)
    assert 0<args.max_hours<=72
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=validated_plan(args.plan)
        binding=dict(plan_sha256=sha(args.plan),controller_sha256=sha(__file__),
                     analysis_protocol_sha256=sha(ANALYSIS_PROTOCOL))
        path=args.out/'status.json'
        state=json.loads(path.read_text()) if path.exists() else dict(binding=binding,jobs={},complete=False)
        assert state['binding']==binding and not state.get('halted'),'Changed or halted controller needs explicit review'
        deadline=time.monotonic()+args.max_hours*3600
        try:
            verify_completed_cache(plan,state)
            while time.monotonic()<deadline:
                started=time.monotonic()
                assert sha(args.plan)==binding['plan_sha256']
                plan=validated_plan(args.plan)
                cycle(plan,state)
                state['updated_at']=cloud.now();save(path,state)
                print(json.dumps({k:state.get(k) for k in ('updated_at','phase','training_complete_count','complete')}),flush=True)
                if state['complete']:return 0
                time.sleep(max(0,60-(time.monotonic()-started)))
            raise TimeoutError('Followup controller time budget expired; VM deadlines remain active')
        except BaseException as exc:
            state.update(halted=True,updated_at=cloud.now(),error=f'{type(exc).__name__}: {exc}')
            save(path,state);print(json.dumps(dict(halted=True,error=state['error'])),flush=True)
            return 1


if __name__=='__main__':raise SystemExit(main())
