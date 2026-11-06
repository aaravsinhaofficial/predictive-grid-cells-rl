from pathlib import Path
import os,subprocess,json,datetime as dt
ROOT=Path(Path('/tmp/feedback_reproduction_root.txt').read_text().strip());OUT=ROOT/'reproduced'
(OUT/'streams').rename(OUT/'streams_failed_relative_out')
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH='')
for label,args in [
 ('streams_absolute',['scripts/compare_feedback_training_streams.py','--out',str(OUT/'streams')]),
 ('actor_isolation',['scripts/check_feedback_baseline_actor_isolation.py','--out',str(OUT/'actor_isolation.json')]),
 ('completed_finals',['scripts/check_feedback_completed_finals.py','--out',str(OUT/'final_review.json')]),
 ('report',['scripts/plot_local_return_feedback.py','--audit-root',str(OUT/'audits'),'--out',str(OUT/'report')])]:
 command=['/opt/anaconda3/bin/python']+args;started=dt.datetime.now(dt.timezone.utc).isoformat();p=OUT/'logs'/f'{label}.log'
 with p.open('w')as f:r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 record=dict(label=label,command=command,returncode=r.returncode,started=started,ended=dt.datetime.now(dt.timezone.utc).isoformat(),log=str(p.relative_to(ROOT)))
 (OUT/'logs'/f'{label}.command.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
 if r.returncode:print(p.read_text()[-5000:],flush=True);raise RuntimeError(label)
print('ALL_REMAINING_COMMANDS_PASSED',flush=True)
