"""Collect and promptly delete only the two frozen visual-memory pilot workers."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
NAMES=('ring-20261105-vision-grid-01','ring-20261105-vision-pgc-01')
ALLOWED_NAMES=NAMES+('ring-20261105-vision-long-grid-01','ring-20261105-vision-long-pgc-01')
ALLOWED_NAMES+=('ring-20261105-vision-noise-grid-01','ring-20261105-vision-noise-pgc-01')
ALLOWED_NAMES+=('ring-20261105-vision-noisy-eval-01',)
ALLOWED_NAMES+=('ring-20261105-vision-reset-eval-01',)
ALLOWED_NAMES+=('ring-20261105-reorient-grid-01','ring-20261105-reorient-pgc-01')
ALLOWED_NAMES+=('ring-20261105-reorient-pgc-02',)
ALLOWED_NAMES+=tuple(f'ring-20261105-reorient-ent-{condition}-{arm}-{attempt:02d}'
                    for condition in ('grid','pgc') for arm in ('original','explore')
                    for attempt in (1,2,3))
PROJECT='connectome-ai-research'
BASE='gs://connectome-ai-research-popcode/ring-20261105'


def experiment_folder(name):
    folder=ROOT/'results/ring_cloud_artifacts'/name
    if 'reorient-ent-' in name:
        manifest=ROOT/'docs/research/rl_blind_reorientation_entropy_launch.json'
        if manifest.exists():
            rows=json.loads(manifest.read_text())['workers']
            matches=[row for row in rows if row['name']==name]
            if matches:
                subdir=matches[0].get('artifact_subdir','')
                if subdir:
                    assert subdir=='experiment'
                    folder=folder/subdir
    return folder


def run(argv,timeout=180):
    env=dict(os.environ,CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD='0')
    return subprocess.run(argv,cwd=ROOT,env=env,text=True,capture_output=True,timeout=timeout)


def event(**fields):
    print(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),**fields)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--max-seconds',type=int,default=8000)
    ap.add_argument('--names',nargs='+',choices=ALLOWED_NAMES,default=NAMES)
    ap.add_argument('--audit-dir',type=Path,default=ROOT/'results/ring_visual_memory/cloud_audit');args=ap.parse_args()
    names=tuple(args.names)
    started=time.monotonic();last_collect={name:0. for name in names};finished=set()
    while len(finished)<len(names) and time.monotonic()-started<args.max_seconds:
        ledger=json.loads((ROOT/'docs/research/ring_cloud_budget.json').read_text())
        records={r['name']:r for r in ledger['runs'] if r['name'] in names}
        for name in names:
            if name in finished or name not in records:continue
            r=records[name]
            if not r.get('instance_id'):
                if r.get('status')=='launch_failed':finished.add(name);event(name=name,status='launch_failed')
                continue
            done=run(['gcloud','storage','cat',f'{BASE}/{name}/DONE.json'],60)
            if done.returncode==0:
                worker=json.loads(done.stdout)
                event(name=name,status='worker_finished',worker=worker)
                for action in ('collect','delete'):
                    if action=='delete' and worker.get('exit_code')==0:
                        folder=experiment_folder(name)
                        summary=folder/'summary.json'
                        complete=summary.exists() and json.loads(summary.read_text()).get('completed') is True
                        if 'vision-noise-' in name:
                            complete=complete and (folder/'final_noisy_evaluation.json').exists()
                        if 'vision-noisy-eval-' in name and complete:
                            report=json.loads(summary.read_text())
                            complete=report.get('all_models_received_identical_raw_noise') is True and len(report.get('models',{}))==4
                            reset_summary=folder/'final75_every_blind_reset/summary.json'
                            diagnostic_status=folder/'diagnostic_status.json'
                            complete=complete and reset_summary.exists() and diagnostic_status.exists()
                            if complete:
                                reset=json.loads(reset_summary.read_text())
                                statuses=json.loads(diagnostic_status.read_text())
                                complete=reset.get('completed') is True and len(reset.get('models',{}))==4
                                complete=complete and len(statuses)==2 and all(item['exit_code']==0 for item in statuses)
                        if 'vision-reset-eval-' in name and complete:
                            report=json.loads(summary.read_text())
                            environment=folder/'environment_audit.json'
                            complete=len(report.get('models',{}))==4 and environment.exists()
                            if complete:
                                complete=json.loads(environment.read_text()).get('smoke_returncode')==0
                        if 'reorient-ent-' in name and complete:
                            report=json.loads(summary.read_text())
                            complete=report.get('baseline_matches_parent_exactly') is True
                            complete=complete and report.get('training',{}).get('extension_frames')==50_073_600
                            complete=complete and all((folder/f).exists() for f in ('final.pt','final_evaluation.json','baseline_verification.json','resume_audit.json'))
                        if not complete:
                            event(name=name,status='completion_artifact_missing_retaining_worker')
                            break
                    result=run([sys.executable,'scripts/ring_cloud.py',action,name],300)
                    log=args.audit_dir;log.mkdir(parents=True,exist_ok=True)
                    (log/f'{name}_{action}.log').write_text(result.stdout+'\n'+result.stderr)
                    if result.returncode:
                        event(name=name,status=f'{action}_failed',tail=result.stderr[-1200:])
                        break
                else:
                    finished.add(name);event(name=name,status='collected_deleted')
                continue
            if time.monotonic()-last_collect[name]>=180:
                dest=ROOT/'results/ring_cloud_artifacts'/name;dest.mkdir(parents=True,exist_ok=True)
                result=run(['gcloud','storage','rsync','--recursive',f'{BASE}/{name}/outputs',str(dest)],180)
                last_collect[name]=time.monotonic()
                metrics=experiment_folder(name)/'metrics.jsonl'
                if metrics.exists():
                    lines=metrics.read_text().strip().splitlines()
                    if lines:
                        try:event(name=name,status='progress',metrics=json.loads(lines[-1]))
                        except json.JSONDecodeError:pass
            vm=run(['gcloud','compute','instances','describe',name,'--project',PROJECT,'--zone',r['zone'],'--format=json(name,status,id)'],60)
            if vm.returncode and ('was not found' in vm.stderr or 'notFound' in vm.stderr):
                result=run([sys.executable,'scripts/ring_cloud.py','collect',name],300)
                event(name=name,status='provider_deleted',artifacts_collected=result.returncode==0)
                finished.add(name)
        if len(finished)<len(names):time.sleep(30)
    event(status='monitor_stopped',finished=sorted(finished),pending=sorted(set(names)-finished))


if __name__=='__main__':main()
