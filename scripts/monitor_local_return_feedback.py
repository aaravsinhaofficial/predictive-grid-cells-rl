"""Monitor, collect, audit and delete only the four approved feedback-acquisition workers."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'docs/research/local_return_feedback_launch.json'
AUDIT = ROOT/'results/local_return_feedback_design/cloud_audit'
PROTOCOL_HASH = 'ad969e782ec2943de2e59b2d72629002dea537f7c9c0fcf2ad3905ce051038c7'
NAMES = {f'ring-20261105-feedback-{pop}-{actor}-01' for pop in ('grid','pgc') for actor in ('retain','reset')}


def run(argv, timeout=180):
    env = dict(os.environ, CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD='0',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    return subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)


def event(**fields):
    print(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), **fields)), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds',type=int,default=15300)
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['protocol_sha256']==PROTOCOL_HASH
    assert {r['name'] for r in manifest['workers']}==NAMES
    assert manifest['provider_delete_seconds']==14400 and manifest['process_max_seconds']==12600
    assert all(r['artifact_subdir']=='experiment' for r in manifest['workers'])
    assert {(r['condition'],r['actor_mode']) for r in manifest['workers']} == {(c,a) for c in ('gc100','pgc33') for a in ('retained_actor','reset_actor')}
    assert manifest['no_automatic_retry_or_extension'] and manifest['planned_reservation_usd']==21.25
    AUDIT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();finished=set();last_collect={name:0. for name in NAMES}
    while len(finished)<4 and time.monotonic()-started<args.max_seconds:
        manifest=json.loads(MANIFEST.read_text())
        for worker in manifest['workers']:
            if worker.get('status')=='cancelled_before_reservation_or_launch':
                finished.add(worker['name'])
        ledger=json.loads((ROOT/'docs/research/ring_cloud_budget.json').read_text())
        records={r['name']:r for r in ledger['runs'] if r['name'] in NAMES}
        for worker in manifest['workers']:
            name=worker['name']
            if name in finished or name not in records:continue
            record=records[name]
            if record.get('status')=='launch_failed':
                event(name=name,status='launch_failed');finished.add(name);continue
            if not record.get('instance_id'):continue
            prefix=record['gcs_prefix'];dest=ROOT/worker['local_artifact_root'];folder=dest/'experiment'
            done=run(['gcloud','storage','cat',prefix+'/DONE.json'],60)
            if done.returncode==0:
                result=json.loads(done.stdout)
                collected=run([sys.executable,'scripts/ring_cloud.py','collect',name],660)
                (AUDIT/f'{name}_collect.log').write_text(collected.stdout+'\n'+collected.stderr)
                if collected.returncode:
                    event(name=name,status='collect_failed',tail=collected.stderr[-1000:]);continue
                if result.get('exit_code')==0:
                    summary=folder/'summary.json'
                    assert summary.exists(), 'Successful worker missing summary'
                    report=json.loads(summary.read_text())
                    assert report['completed'] and report['counters']['updates']==1024
                    assert report['counters']['trials']==32768
                    assert set(report['counters']['evaluation_records'])=={'baseline','final'}
                    assert report['protocol_sha256']==PROTOCOL_HASH
                    assert hashlib.sha256((folder/'final.pt').read_bytes()).hexdigest()==report['final_checkpoint_sha256']
                    auditor=ROOT/'scripts/audit_local_return_feedback.py'
                    if not auditor.exists():
                        event(name=name,status='awaiting_independent_auditor');continue
                    audited=run([sys.executable,str(auditor),'--folder',str(folder),'--condition',worker['condition'],
                                 '--actor-mode',worker['actor_mode'],'--require-complete','--out',str(ROOT/worker['audit_path'])],1800)
                    (AUDIT/f'{name}_audit.log').write_text(audited.stdout+'\n'+audited.stderr)
                    if audited.returncode:
                        event(name=name,status='final_audit_failed',tail=(audited.stdout+audited.stderr)[-1600:])
                        raise RuntimeError('Preserve collected failed audit for root review before deletion')
                    event(name=name,status='fixed_final_collected_and_audited',summary=str(summary.relative_to(ROOT)))
                else:
                    event(name=name,status='training_failed_collected',worker_result=result)
                    # Diagnose retained partial artifacts without requiring a complete endpoint.
                    # A diagnostic failure must not prevent deleting a failed, collected VM.
                    try:
                        audited=run([sys.executable,'scripts/audit_local_return_feedback.py',
                            '--folder',str(folder),'--condition',worker['condition'],
                            '--actor-mode',worker['actor_mode'],'--out',str(ROOT/worker['audit_path'])],1800)
                        (AUDIT/f'{name}_audit.log').write_text(audited.stdout+'\n'+audited.stderr)
                        event(name=name,status='failed_worker_partial_audit',audit_exit=audited.returncode)
                    except Exception as error:
                        (AUDIT/f'{name}_audit_exception.json').write_text(json.dumps(dict(
                            type=type(error).__name__,error=str(error),worker_result=result),indent=2)+'\n')
                        event(name=name,status='failed_worker_diagnostic_exception',error=str(error))
                deleted=run([sys.executable,'scripts/ring_cloud.py','delete',name],660)
                (AUDIT/f'{name}_delete.log').write_text(deleted.stdout+'\n'+deleted.stderr)
                if deleted.returncode:
                    event(name=name,status='delete_failed',tail=deleted.stderr[-1000:]);continue
                for kind in ('instances','disks'):
                    absent=run(['gcloud','compute',kind,'describe',name,'--project',manifest['project'],
                                '--zone',worker['zone'],'--format=json'],60)
                    (AUDIT/f'{name}_{kind}_absence.log').write_text(absent.stdout+'\n'+absent.stderr)
                    assert absent.returncode and ('was not found' in absent.stderr or 'notFound' in absent.stderr)
                finished.add(name);event(name=name,status='collected_audited_deleted_absence_verified')
                latest=json.loads(MANIFEST.read_text())
                row=next(r for r in latest['workers'] if r['name']==name)
                row['status']='complete_collected_audited_deleted' if result.get('exit_code')==0 else 'failed_collected_deleted'
                row['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
                if result.get('exit_code')==0:
                    row['audit_sha256']=hashlib.sha256((ROOT/row['audit_path']).read_bytes()).hexdigest()
                    row['final_checkpoint_sha256']=report['final_checkpoint_sha256']
                temp=MANIFEST.with_suffix('.tmp');temp.write_text(json.dumps(latest,indent=2)+'\n');temp.replace(MANIFEST)
                continue
            if time.monotonic()-last_collect[name]>=180:
                dest.mkdir(parents=True,exist_ok=True)
                copied=run(['gcloud','storage','rsync','--recursive',prefix+'/outputs',str(dest)],300)
                last_collect[name]=time.monotonic()
                metrics=folder/'metrics.jsonl'
                if metrics.exists():
                    lines=metrics.read_text().strip().splitlines()
                    if lines:
                        try:event(name=name,status='progress',metrics=json.loads(lines[-1]))
                        except json.JSONDecodeError:pass
                elif copied.returncode:event(name=name,status='waiting_for_first_upload')
            vm=run(['gcloud','compute','instances','describe',name,'--project',manifest['project'],
                    '--zone',worker['zone'],'--format=json(name,status,id)'],60)
            if vm.returncode and ('was not found' in vm.stderr or 'notFound' in vm.stderr):
                copied=run([sys.executable,'scripts/ring_cloud.py','collect',name],660)
                (AUDIT/f'{name}_provider_deleted_collect.log').write_text(copied.stdout+'\n'+copied.stderr)
                event(name=name,status='provider_deleted_before_success',collected=copied.returncode==0)
                finished.add(name)
        if len(finished)<4:time.sleep(30)
    event(status='monitor_stopped',finished=sorted(finished),pending=sorted(NAMES-finished))


if __name__=='__main__':main()
