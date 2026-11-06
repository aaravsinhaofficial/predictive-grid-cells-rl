from pathlib import Path
import concurrent.futures,os,subprocess,json,datetime as dt,sys,hashlib
ROOT=Path(Path('/tmp/feedback_reproduction_root.txt').read_text().strip());OUT=ROOT/'reproduced';(OUT/'audits').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
PYTHON='/opt/anaconda3/bin/python';env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH='')
routing=json.loads((ROOT/'docs/research/local_return_feedback_portable_launch.json').read_text())
def command(label,args):
 started=dt.datetime.now(dt.timezone.utc).isoformat();p=OUT/'logs'/f'{label}.log'
 with p.open('w')as f:r=subprocess.run([PYTHON]+args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 record=dict(label=label,command=[PYTHON]+args,returncode=r.returncode,started=started,ended=dt.datetime.now(dt.timezone.utc).isoformat(),log=str(p.relative_to(ROOT)))
 (OUT/'logs'/f'{label}.command.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
 if r.returncode:print(p.read_text()[-4000:],flush=True);raise RuntimeError(label)
 return record
commands=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 futures=[]
 for w in routing['workers']:
  key=w['condition']+'_'+w['actor_mode'];args=['scripts/audit_local_return_feedback.py','--folder',str(Path(w['local_artifact_root'])/w['artifact_subdir']),'--condition',w['condition'],'--actor-mode',w['actor_mode'],'--protocol','docs/research/local_return_feedback_portable_protocol.json','--out','reproduced/audits/'+key+'.json','--require-complete'];futures.append(pool.submit(command,key,args))
 for f in concurrent.futures.as_completed(futures):commands.append(f.result())
for label,args in [
 ('references',['scripts/audit_local_return_feedback.py','--references','--folder','results/local_return_feedback_acquisition/references','--protocol','docs/research/local_return_feedback_portable_protocol.json','--out','reproduced/reference_audit.json','--require-complete']),
 ('streams',['scripts/compare_feedback_training_streams.py','--out','reproduced/streams']),
 ('actor_isolation',['scripts/check_feedback_baseline_actor_isolation.py','--out','reproduced/actor_isolation.json']),
 ('completed_finals',['scripts/check_feedback_completed_finals.py','--out','reproduced/final_review.json']),
 ('report',['scripts/plot_local_return_feedback.py','--audit-root','reproduced/audits','--out','reproduced/report'])]:commands.append(command(label,args))
(OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print('ALL_COMMANDS_PASSED',flush=True)
