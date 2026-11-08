"""Narrow operational guardian for the unchanged simple-memory training supervisor.

Only the two observed read-only GCS connection failures can trigger recovery.
This guardian never creates, deletes, retries or resumes a cloud worker itself.
Do not start it without the owner's review: a successful recovery clears the
specific archived connection halt and starts the same existing supervisor CLI.
Local deadline checks cannot run during host suspension. Passing remaining wall
time with a five-second launch reserve is not an awake wall-clock watchdog;
unchanged absolute provider DELETE deadlines independently bound cloud costs.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
SUPERVISION=ROOT/'results/ring_simple_memory_supervision'
ORIGINAL_PROCESS=SUPERVISION/'process_before_recovery_20261107T160447Z.json'
ORIGINAL_START='2026-11-07T13:02:09.281197+00:00'
SUPERVISOR='scripts/run_simple_memory_cohort.py'
FOLLOWUP_PLAN='docs/research/simple_memory_followup_plan.json'
LEDGER='docs/research/ring_cloud_budget.json'
PROJECT='connectome-ai-research'
BUCKET='connectome-ai-research-popcode'
STUDY='ring-20261105'
BACKOFF=(5,10,20,40,60)
MAX_RECOVERIES=3


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def stamp(value):
    parsed=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    require(parsed.utcoffset() is not None,'UTC offset missing from deadline evidence')
    return parsed.timestamp()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def relative(root,name):
    p=(Path(root)/name).resolve()
    require(not Path(name).is_absolute() and p.is_relative_to(Path(root).resolve()),'Nonlocal manifest path')
    return p


@contextlib.contextmanager
def exclusive(path):
    """No truncation of existing supervisor/ledger lock files."""
    with Path(path).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def original_deadline(process,original,now):
    require(original['started_utc']==ORIGINAL_START and original['max_hours']==72,'Changed original 72h anchor')
    require(process.get('original_started_utc',process['started_utc'])==ORIGINAL_START,'Changed original start in current process')
    deadline=stamp(ORIGINAL_START)+72*3600
    require(stamp(process['started_utc'])>=stamp(ORIGINAL_START),'Process predates original start')
    require(stamp(process['started_utc'])+process['max_hours']*3600<=deadline+.01,'Prior restart extended original deadline')
    require(now<deadline,'Original supervisor deadline expired; never extend')
    return deadline


def allowed_targets(plan,ledger):
    records={r['name']:r for r in ledger['runs']}
    targets={}
    for job in plan['jobs']:
        r=records.get(job['name'])
        if r is None:continue
        prefix=f"gs://{BUCKET}/{STUDY}/{job['name']}"
        require(r['gcs_prefix']==prefix,'Unbound worker storage prefix')
        targets[prefix+'/DONE.json']=(job['name'],'done')
        targets[prefix+'/outputs/'+Path(job['trainer_output_dir']).name+'/metrics_000.jsonl']=(job['name'],'metrics')
    return targets


def recognize_failure(error,targets):
    """Exact observed read classes; assertions, writes and generic timeouts fail."""
    timeout=re.fullmatch(r"TimeoutExpired: Command '(\[.*\])' timed out after 60(?:\.0)? seconds",error,re.S)
    if timeout:
        argv=ast.literal_eval(timeout.group(1))
        require(isinstance(argv,list) and all(isinstance(x,str) for x in argv),'Invalid timeout command')
        require(argv[:3]==['gcloud','storage','cat'] and
                (len(argv)==4 or len(argv)==5 and argv[3]=='--range=-65536'),'Timeout was not an allowed storage cat')
        uri=argv[-1];kind='storage_cat_timeout60'
    else:
        require(error.startswith('RuntimeError: Cloud read failed: ERROR: gcloud crashed (ConnectionError):'),
                'Unrecognized supervisor failure; explicit owner review required')
        require('NameResolutionError' in error and "Failed to resolve 'storage.googleapis.com'" in error and
                "host='storage.googleapis.com'" in error,'Not the observed storage DNS failure')
        match=re.search(r'/storage/v1/b/([^/]+)/o/([^?\s]+)\?',error)
        require(match is not None and match.group(1)==BUCKET,'DNS failure has no bound object')
        uri='gs://'+match.group(1)+'/'+unquote(match.group(2));kind='storage_dns_failure'
    require(uri in targets,'Failed read target is not an existing frozen owned job')
    name,object_type=targets[uri]
    return dict(kind=kind,target=uri,job=name,object_type=object_type,
                error_sha256=hashlib.sha256(error.encode()).hexdigest())


def process_state(pid):
    result=subprocess.run(['ps','-p',str(int(pid)),'-o','uid=,command='],capture_output=True,text=True,timeout=10)
    if result.returncode==1 and not result.stdout.strip():return dict(alive=False,pid=int(pid),command='')
    require(result.returncode==0 and result.stdout.strip(),'Unable to establish old PID identity')
    uid,command=result.stdout.strip().split(None,1)
    require(int(uid)==os.getuid(),'Recorded PID belongs to another user')
    tokens=shlex.split(command)
    require(any(token.endswith('/'+SUPERVISOR) or token==SUPERVISOR for token in tokens),
            'Recorded PID was reused or has unexpected command; never kill it')
    return dict(alive=True,pid=int(pid),command=command)


def whole_budget(training,followup,ledger):
    require(ledger['budget_usd']==followup['cumulative_ceiling_usd']==600 and
            ledger['reserved_rate_usd_per_hour']==1.25,'Budget/rate binding changed')
    require(len({r['name'] for r in ledger['runs']})==len(ledger['runs']),'Duplicate ledger jobs')
    names={r['name'] for r in ledger['runs']}
    committed=ledger['ancillary_reserve_usd']+sum(r.get('actual_upper_bound_usd',r['reserved_usd']) for r in ledger['runs'])
    missing_training=0.
    for job in training['jobs']:
        if job['name'] not in names:
            argv=job['plan_argv'];require(argv.count('--hours')==1,'Missing frozen worker cap')
            hours=float(argv[argv.index('--hours')+1]);require(0<hours<=25.5,'Changed worker cap')
            missing_training+=round((hours+.25)*1.25,4)
    missing_analysis=sum((followup['per_worker_hours']+.25)*1.25 for job in followup['jobs'] if job['name'] not in names)
    bound=committed+missing_training+missing_analysis
    require(all(math.isfinite(v) and v>=0 for v in (committed,missing_training,missing_analysis)) and bound<=600,
            'Combined full training and analysis budget exceeds600; no recovery')
    return dict(committed_upper_usd=committed,remaining_training_usd=missing_training,
                remaining_analysis_usd=missing_analysis,combined_upper_usd=bound,ceiling_usd=600)


def validate_local(root,supervision,process,status):
    """Read-only reuse of frozen validators, after verifying their source bytes."""
    root,supervision=Path(root),Path(supervision)
    plan_path=relative(root,process['plan']);protocol_path=relative(root,process['protocol'])
    followup=read(root/FOLLOWUP_PLAN);training=read(plan_path);protocol=read(protocol_path)
    require(training['status']==protocol['status']==followup['status']=='frozen','Inputs not frozen')
    binding=dict(plan_sha256=sha(plan_path),protocol_sha256=sha(protocol_path),supervisor_sha256=sha(root/SUPERVISOR))
    require(status['binding']==binding,'Supervisor binding changed')
    require(followup['training_plan_sha256']==binding['plan_sha256'] and
            followup['training_protocol_sha256']==binding['protocol_sha256'] and
            followup['training_supervisor_sha256']==binding['supervisor_sha256'],'Followup binding changed')
    analysis_path=root/'docs/research/simple_memory_analysis_protocol.json';analysis=read(analysis_path)
    require(analysis['status']=='frozen' and sha(analysis_path)==followup['analysis_protocol_sha256'],'Analysis protocol changed')
    manifests=[protocol['source_sha256'],analysis['source_sha256'],analysis['validation_artifacts_sha256'],
               followup['operational_source_sha256'],followup['validation_artifacts_sha256'],
               {x['path']:x['sha256'] for x in training['package']['current_manifest']}]
    for manifest in manifests:
        require(bool(manifest),'Empty frozen manifest')
        for name,digest in manifest.items():require(sha(relative(root,name))==digest,f'Frozen source/evidence changed: {name}')
    sys.path.insert(0,str(root))
    from scripts import run_simple_memory_cohort as cohort
    from scripts import run_simple_memory_followup as followup_module
    require(Path(cohort.__file__).resolve()==(root/SUPERVISOR).resolve(),'Wrong imported supervisor')
    cohort.validate(plan_path,protocol_path)
    followup_module.validated_plan(root/FOLLOWUP_PLAN)
    ledger=read(root/LEDGER)
    budget=whole_budget(training,followup,ledger)
    records={r['name']:r for r in ledger['runs']}
    manifest={x['path']:x['sha256'] for x in training['package']['current_manifest']}
    require(set(status['jobs'])<=set(j['name'] for j in training['jobs']),'Unexpected status job')
    for job in training['jobs']:
        record=records.get(job['name']);saved=status['jobs'].get(job['name'],{})
        if record is None:
            require(not saved.get('instance_id') and not saved.get('complete'),'Missing ledger identity')
            continue
        cohort.matching_record(job,record,manifest,saved.get('instance_id'),protocol_path)
        require(saved.get('instance_id')==str(record['instance_id']),'Missing status instance ID; explicit reconciliation required')
        if saved.get('complete'):
            summary_path=Path(saved['summary_path']);summary=read(summary_path)
            require(sha(summary_path)==saved['summary_sha256'] and summary['complete'] and summary['frames']==1_000_000_000,
                    'Completed scientific endpoint changed')
            require(sha(summary_path.parent/'final.pt')==saved['final_sha256']==summary['final_sha256'],'Completed checkpoint changed')
    return dict(training=training,followup=followup,ledger=ledger,budget=budget,binding=binding,
                ledger_sha256=sha(root/LEDGER),plan_path=str(plan_path),protocol_path=str(protocol_path))


class CloudReads:
    """Only GET-equivalent gcloud commands, bounded waits, retained evidence."""
    def __init__(self,out,deadline):self.out=Path(out);self.deadline=deadline;self.calls=[]
    def call(self,argv,allow_missing=False):
        valid=(argv[:3]==['gcloud','storage','cat'] or argv[:4] in
               (['gcloud','compute','instances','describe'],['gcloud','storage','objects','describe']))
        require(valid,'Guardian cloud interface is read-only')
        for attempt,backoff in enumerate((0,*BACKOFF)):
            if backoff:
                require(time.time()+backoff<self.deadline,'Recovery deadline exhausted')
                time.sleep(backoff)
            remaining=self.deadline-time.time();require(remaining>0,'Recovery deadline exhausted')
            row=dict(argv=argv,attempt=attempt,started_utc=utc())
            try:
                result=subprocess.run(argv,text=True,capture_output=True,timeout=min(45,remaining),
                    env=dict(os.environ,CLOUDSDK_STORAGE_PROCESS_COUNT='1',CLOUDSDK_STORAGE_THREAD_COUNT='1'))
                row.update(exit_code=result.returncode,stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                           stderr=result.stderr[-3000:],finished_utc=utc())
                self.calls.append(row);atomic(self.out/'cloud_reads.json',self.calls)
                if result.returncode==0:return result.stdout
                if allow_missing and re.search(r'No URLs matched|matched no objects|was not found|notFound|(?i:404|does not exist)',result.stderr):return None
                require(re.search(r'NameResolutionError|Failed to resolve|ConnectionError|ReadTimeout|ConnectTimeout',result.stderr),
                        'Read failed for an unrecognized reason: '+result.stderr[-500:])
            except subprocess.TimeoutExpired:
                row.update(timed_out=True,finished_utc=utc());self.calls.append(row);atomic(self.out/'cloud_reads.json',self.calls)
        raise RuntimeError('Bounded read-only reconnect attempts exhausted')


def terminal_done(text,record,protocol_sha,read_cloud,job):
    done=json.loads(text)
    require(done['exit_code']==0 and done['package_sha256']==record['package_manifest_sha256'],
            'Worker terminal failure/package mismatch; never retry')
    uri=record['gcs_prefix']+'/outputs/'+Path(job['trainer_output_dir']).name+'/summary.json'
    summary=json.loads(read_cloud(['gcloud','storage','cat',uri]))
    require(summary['complete'] is True and summary['frames']==1_000_000_000 and summary['updates']==78125 and
            summary['decisions']==250_000_000 and summary['provenance']['protocol_sha256']==protocol_sha,
            'Worker terminal scientific failure; no recovery')
    return dict(done=done,summary_sha256=hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest())


def verify_cloud(local,status,failure,read_cloud,now):
    records={r['name']:r for r in local['ledger']['runs']};rows=[];fresh=None
    from scripts.run_simple_memory_cohort import technical_row,technical_viability
    clock=now if callable(now) else lambda:now
    for job in local['training']['jobs']:
        name=job['name'];r=records.get(name);saved=status['jobs'].get(name,{})
        if r is None or saved.get('complete'):continue
        require(r.get('instance_id') and saved.get('instance_id')==str(r['instance_id']),'Missing immutable worker ID')
        raw=read_cloud(['gcloud','compute','instances','describe',name,'--project',PROJECT,'--zone',r['zone'],'--format=json'],allow_missing=True)
        vm=json.loads(raw) if raw is not None else None
        if vm is not None:
            require(str(vm['id'])==str(r['instance_id']) and vm['name']==name and vm.get('labels',{}).get('study')==STUDY,
                    'Owned VM identity mismatch')
            require(vm['zone'].rsplit('/',1)[-1]==r['zone'] and vm['machineType'].rsplit('/',1)[-1]==r['machine'],'VM machine/zone changed')
        done_text=read_cloud(['gcloud','storage','cat',r['gcs_prefix']+'/DONE.json'],allow_missing=True)
        row=dict(name=name,instance_id=str(r['instance_id']),vm=None if vm is None else
                 {key:vm[key] for key in ('id','name','status','zone','machineType','labels')})
        if done_text is not None:
            row['terminal']=terminal_done(done_text,r,local['binding']['protocol_sha256'],read_cloud,job)
            if name==failure['job']:fresh=dict(kind='successful_terminal_read',job=name)
        else:
            require(vm is not None and vm['status']=='RUNNING','Owned worker missing/not RUNNING without valid DONE; no retry')
            require('actual_upper_bound_usd' not in r and clock()<stamp(r['provider_deadline_utc']), 'Worker closed/expired without terminal evidence')
            uri=r['gcs_prefix']+'/outputs/'+Path(job['trainer_output_dir']).name+'/metrics_000.jsonl'
            metadata=json.loads(read_cloud(['gcloud','storage','objects','describe',uri,'--format=json']))
            times=[stamp(metadata[k]) for k in ('update_time','updated','updateTime') if k in metadata]
            require(times and len(set(times))==1,'Missing/ambiguous metric object update time')
            updated=times[0]
            require(-60<=clock()-updated<=900,'Metrics object is stale (over900s) or from the future')
            require(metadata['bucket']==BUCKET and metadata['name']==uri.split('/'+BUCKET+'/',1)[1], 'Metrics metadata identity changed')
            generation=str(metadata['generation']);require(re.fullmatch(r'[0-9]+',generation),'Missing object generation')
            # Fetch the exact newly described generation, rather than comparing
            # an old cat body with a concurrently uploaded object's timestamp.
            text=read_cloud(['gcloud','storage','cat','--range=-65536',uri+'#'+generation])
            metric=technical_row(text);require(metric is not None,'Missing complete fresh technical metric')
            require(metric['frames']>=saved.get('technical',{}).get('frames',0),'Regressed/stale metrics')
            viability=technical_viability(local['training'],job,r,saved,metric,now=clock())
            row.update(technical=metric,viability=viability,metric_object=dict(target=uri,generation=generation,updated_utc=metadata[next(k for k in ('update_time','updated','updateTime') if k in metadata)]))
            if name==failure['job']:
                fresh=dict(kind='successful_technical_read',job=name,target=uri,technical=metric,viability=viability)
        rows.append(row)
    require(fresh is not None,'Failed worker read not freshly verified')
    # Re-evaluate every bound after all potentially slow reconnect probes.
    jobs={job['name']:job for job in local['training']['jobs']}
    for row in rows:
        if 'technical' not in row:continue
        require(-60<=clock()-stamp(row['metric_object']['updated_utc'])<=900,'Metrics became stale during preflight')
        require(clock()<stamp(records[row['name']]['provider_deadline_utc']), 'Worker deadline expired during preflight')
        row['viability']=technical_viability(local['training'],jobs[row['name']],records[row['name']],status['jobs'][row['name']],row['technical'],now=clock())
        if row['name']==failure['job']:fresh['viability']=row['viability']
    return dict(checked_utc=utc(),owned_workers=rows,reconnected_read=fresh)


def restart_argv(process,supervision,deadline,now,python=sys.executable):
    remaining=math.floor(deadline-now)-5
    require(remaining>0,'Original deadline leaves no safe launch margin')
    return [python,'-u',SUPERVISOR,'--plan',process['plan'],'--protocol',process['protocol'],
            '--out',str(supervision),'--max-hours',format(remaining/3600,'.12f')]


def recover(supervision,out,history,*,root=ROOT,now=time.time,inspect_pid=process_state,local_validator=validate_local,
            cloud_verifier=verify_cloud,spawn=subprocess.Popen):
    """One narrow recovery. The caller holds the guardian lock for its lifetime."""
    root,supervision,out=Path(root),Path(supervision),Path(out)
    require(len(history['attempts'])<MAX_RECOVERIES,'Guardian recovery limit reached')
    require(history['guardian_source_sha256']==sha(__file__),'Guardian source changed while running')
    require(history['original_process_sha256']==sha(history['original_process_path']),'Original anchor changed while running')
    original=read(Path(history['original_process_path']))
    with exclusive(supervision/'supervisor.lock'):
        raw={name:(supervision/name).read_bytes() for name in ('status.json','process.json')}
        status,process=(json.loads(raw[name]) for name in ('status.json','process.json'))
        require(not inspect_pid(process['pid'])['alive'],'Old supervisor still alive; never duplicate')
        require(status.get('halted') is True and not status.get('complete'),'No recoverable halted supervisor')
        deadline=original_deadline(process,original,now())
        local=local_validator(root,supervision,process,status)
        failure=recognize_failure(status['error'],allowed_targets(local['training'],local['ledger']))
        attempt=dict(index=len(history['attempts'])+1,started_utc=utc(),status='preflight',failure=failure,
                     previous_pid=process['pid'],original_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
        history['attempts'].append(attempt);atomic(out/'guardian.json',history)
        evidence=out/f"recovery{attempt['index']:02d}";evidence.mkdir()
        for name,data in raw.items():(evidence/name).write_bytes(data)
        (evidence/'ledger.json').write_bytes((root/LEDGER).read_bytes())
        attempt['backup_sha256']={name:sha(evidence/name) for name in (*raw,'ledger.json')}
        atomic(out/'guardian.json',history)
        reader=CloudReads(evidence,min(deadline,stamp(history['deadline_utc']),now()+600))
        checked=cloud_verifier(local,status,failure,reader.call,now)
        atomic(evidence/'cloud_identity_and_read.json',checked)
        # No clearing if the owner, ledger or source files changed during probes.
        with exclusive((root/LEDGER).with_suffix('.lock')):
            require(all((supervision/name).read_bytes()==data for name,data in raw.items()),'Status/process changed during preflight')
            require(not inspect_pid(process['pid'])['alive'],'Old supervisor reappeared')
            require(sha(root/LEDGER)==local['ledger_sha256'],'Ledger changed during preflight; re-review required')
            require(history['guardian_source_sha256']==sha(__file__) and
                    history['original_process_sha256']==sha(history['original_process_path']), 'Guardian/anchor changed during preflight')
            final_local=local_validator(root,supervision,process,status)
            require(final_local['binding']==local['binding'] and final_local['budget']==local['budget'],'Bindings/budget changed')
            argv=restart_argv(process,supervision,deadline,now())
            attempt.update(status='spawn_intent',argv=argv,budget=local['budget'],evidence=str(evidence),
                           preflight_completed_utc=utc(),guardian_source_sha256=sha(__file__))
            atomic(out/'guardian.json',history)  # durable accounting BEFORE spawn
            cleared=dict(status);cleared.pop('halted',None);cleared.pop('error',None)
            cleared['connection_recovery']=dict(evidence=str(evidence),failure_sha256=failure['error_sha256'])
            atomic(supervision/'status.json',cleared)
    # Unchanged supervisor acquires its OWN exclusive lock. If a competing
    # owner wins this brief handoff, this attempt fails closed; never retry it.
    log=(evidence/'supervisor.log').open('xb')
    try:
        child=spawn(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:
        with exclusive(supervision/'supervisor.lock'):
            if (supervision/'status.json').read_bytes()==json.dumps(cleared,indent=2,allow_nan=False).encode()+b'\n':
                (supervision/'status.json').write_bytes(raw['status.json'])
        attempt['status']='spawn_failed';atomic(out/'guardian.json',history);raise
    finally:log.close()
    attempt.update(status='spawned_pending_confirmation',resumed_pid=child.pid,spawned_utc=utc())
    atomic(out/'guardian.json',history)
    process_new=dict(process,pid=child.pid,started_utc=utc(),original_started_utc=ORIGINAL_START,
                     max_hours=float(argv[-1]),original_deadline_utc=attempt['original_deadline_utc'],argv=argv,
                     guardian_recovery_evidence=str(evidence),guardian_source_sha256=sha(__file__))
    process_new.pop('ac_power_sleep_guard_pid',None)
    atomic(supervision/'process.json',process_new)
    original_deadline(process_new,original,now())
    sleep_seconds=max(1,math.ceil(deadline-now()))
    guard=spawn(['caffeinate','-s','-w',str(child.pid),'-t',str(sleep_seconds)],cwd=root,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    process_new['ac_power_sleep_guard_pid']=guard.pid;atomic(supervision/'process.json',process_new)
    time.sleep(1)
    require(child.poll() is None and inspect_pid(child.pid)['alive'],'Restart did not establish the expected live supervisor; manual review required')
    require(not read(supervision/'status.json').get('halted'),'Restart immediately halted; manual review required')
    attempt.update(status='recovered',ac_power_sleep_guard_pid=guard.pid,confirmed_utc=utc())
    atomic(out/'guardian.json',history)
    return attempt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supervision',type=Path,default=SUPERVISION)
    p.add_argument('--out',type=Path,default=SUPERVISION/'guardian')
    p.add_argument('--original-process',type=Path,default=ORIGINAL_PROCESS)
    p.add_argument('--max-hours',type=float,default=6)
    p.add_argument('--poll-seconds',type=float,default=30)
    args=p.parse_args(argv)
    require(0<args.max_hours<=6 and 0<args.poll_seconds<=60,'Guardian limits exceed reviewed bounds')
    require(args.supervision.resolve()==SUPERVISION.resolve(),'Only the original training supervisor is supported')
    require(args.out.resolve()==(SUPERVISION/'guardian').resolve(),'One canonical guardian lock/history is required')
    require(args.original_process.resolve()==ORIGINAL_PROCESS.resolve(),'Use the preserved original process anchor')
    args.out.mkdir(parents=True,exist_ok=True)
    with exclusive(args.out/'guardian.lock'):
        path=args.out/'guardian.json'
        if path.exists():
            history=read(path)
            require(history['guardian_source_sha256']==sha(__file__) and not history.get('halted'),'Changed/failed guardian requires owner review')
            require(all(a['status']=='recovered' for a in history['attempts']),'Unresolved prior recovery; never repeat spawn')
            require(history['original_process_sha256']==sha(args.original_process),'Original process anchor changed')
            require(stamp(history['deadline_utc'])<=stamp(history['started_utc'])+6*3600+1,
                    'Guardian history extends beyond its six-hour window')
        else:
            history=dict(schema=1,guardian_source_sha256=sha(__file__),started_utc=utc(),
                deadline_utc=dt.datetime.fromtimestamp(time.time()+args.max_hours*3600,dt.timezone.utc).isoformat(),
                original_process_path=str(args.original_process.resolve()),original_process_sha256=sha(args.original_process),
                attempts=[],complete=False)
            atomic(path,history)
        try:
            while time.time()<stamp(history['deadline_utc']):
                process=read(args.supervision/'process.json');status=read(args.supervision/'status.json')
                original_deadline(process,read(args.original_process),time.time())
                alive=process_state(process['pid'])
                history['last_check']=dict(utc=utc(),process=alive,status_sha256=sha(args.supervision/'status.json'))
                atomic(path,history)
                if status.get('complete'):
                    require(not status.get('halted'),'Completed status also halted')
                    history.update(complete=True,finished_utc=utc(),reason='Training supervisor completed');atomic(path,history);return 0
                if not alive['alive']:
                    recover(args.supervision,args.out,history)
                time.sleep(min(args.poll_seconds,max(0,stamp(history['deadline_utc'])-time.time())))
            history.update(finished_utc=utc(),reason='Guardian six-hour window ended; no deadline extension');atomic(path,history)
            return 0
        except BaseException as exc:
            history.update(halted=True,finished_utc=utc(),error=f'{type(exc).__name__}: {exc}')
            atomic(path,history)
            print(history['error'],flush=True);return 1


if __name__=='__main__':sys.exit(main())
