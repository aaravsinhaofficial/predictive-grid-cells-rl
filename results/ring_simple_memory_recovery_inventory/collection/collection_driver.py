"""One authorized archive collection; two concurrent read-only GCS downloads.
No ledger write, VM API, policy construction, forward pass or optimizer update.
"""
from pathlib import Path
import concurrent.futures as cf
import copy,datetime,hashlib,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts import ring_cloud as cloud
from scripts.run_simple_memory_analysis import authenticate_run
import torch
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
INVENTORY=OUT.parent/'inventory.json'
PLAN=ROOT/'docs/research/simple_memory_cloud_plan.json'
PROTOCOL=ROOT/'docs/research/simple_memory_training_protocol.json'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def require(v,msg):
 if not v:raise RuntimeError(msg)


def inspect_checkpoint(path,cfg,proof,final_summary=None):
 s=torch.load(path,map_location='cpu',weights_only=False)
 require(s['schema']==1 and s['config']=={k:v for k,v in cfg.items() if k!='population'},'Checkpoint config mismatch')
 require(s['provenance']==proof,'Checkpoint proof mismatch')
 counters=s['runtime']['counters'];u=s['updates'];f=s['frames']
 require(f==counters['frames']==u*32*100*4 and u==counters['updates'],'Frame/update counters inconsistent')
 require(counters['decisions']==f//4,'Decision counter inconsistent')
 require(0<=u<=78125 and s['runtime']['device_type']=='cuda','Wrong endpoint budget/backend')
 require(all(float(v['step'])==u for v in s['optimizer']['state'].values()),'Optimizer step mismatch')
 require(len(s['optimizer']['state'])==len(s['policy'])==20,'Incomplete optimizer/policy state')
 h=hashlib.sha256()
 for k,v in sorted(s['policy'].items()):
  require(v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Nonfinite/wrong policy dtype')
  h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
 require(h.hexdigest()==s['policy_sha256'],'Policy tensor digest mismatch')
 rt=s['runtime']
 require(set(rt)=={'environment_tensors','environment_scalars','environment_rng','device_type','obs','state','goal','pa','pr','clock','counters','recent_returns','cpu_rng','cuda_rng','numpy_rng','python_rng'},'Runtime schema incomplete')
 require(len(rt['environment_tensors'])==28 and [list(x.shape) for x in rt['state']]==[[32,256],[32,256]],'Runtime environment/state schema')
 require(list(rt['obs']['rgb'].shape)==[32,3,84,84] and list(rt['goal'].shape)==[32,512],'Native input shape changed')
 require(list(rt['environment_rng'].shape)==list(rt['cuda_rng'].shape)==[16] and list(rt['cpu_rng'].shape)==[5056],'RNG state schema')
 require(set(rt['clock'])=={'n','groups','visible_steps','known','blind','remaining','rng','draw_steps'},'Private clock schema')
 if final_summary:
  require(final_summary['complete'] and f==1_000_000_000 and u==78125,'Not completed 1B')
  require(final_summary['final_sha256']==sha(path) and final_summary['final_policy_sha256']==h.hexdigest(),'Final summary digest mismatch')
 return dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,frames=f,updates=u,decisions=counters['decisions'],
             policy_sha256=h.hexdigest(),optimizer_states=20,optimizer_step=u,full_runtime_present=True,device_type='cuda')


def one(job,record,planjob):
 name=job['name'];r=copy.deepcopy(record);started=time.monotonic()
 receipt=dict(name=name,complete=False,started_utc=utc(),instance_id=str(r['instance_id']),gcs_prefix=r['gcs_prefix'],
  inventory_sha256=sha(INVENTORY),launcher_sha256=sha(cloud.__file__),driver_sha256=sha(__file__),ledger_write_operations=0,
  VM_API_operations=0,policy_forward_calls=0,record_is_deepcopy=True)
 dest=ROOT/'results/ring_cloud_artifacts'/name
 try:
  require(str(r['instance_id'])==str(job['instance_id']) and r['gcs_prefix']==job['gcs_prefix'],'Inventory/ledger identity mismatch')
  existing={p.relative_to(dest).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.rglob('*') if p.is_file()} if dest.exists() else {}
  receipt['preexisting_files']=existing
  # Canonical collect has no --delete, but can update same-name files. The
  # seven folders were absent at dispatch; fail before touching a raced writer.
  require(not existing,'Unexpected pre-existing output; preserve it for owner review')
  save(OUT/(name+'.json'),receipt)
  cloud.collect(r)
  receipt['collected_record_fields']={k:r[k] for k in ('collected_at','collected_bytes','local_artifacts')}
  worker=read(dest/'WORKER_RESULT.json')
  require(sha(dest/'WORKER_RESULT.json')==job['artifacts']['worker_result.json']['sha256'],'Worker result bytes differ from inventory')
  require(worker['exit_code']==job['done']['exit_code'] and worker['package_sha256']==r['package_manifest_sha256'],'Worker identity/exit changed')
  folder=dest/Path(planjob['trainer_output_dir']).name
  cfg=read(folder/'config.json');proof=read(folder/'provenance.json');protocol=read(PROTOCOL)
  require(proof['protocol_sha256']==sha(PROTOCOL) and proof['source_sha256']==protocol['source_sha256'],'Original frozen proof mismatch')
  for p,h in proof['source_sha256'].items():require(sha(ROOT/p)==h,'Frozen source changed: '+p)
  for k,v in protocol['training_config'].items():require(cfg[k]==v,'Frozen scientific config changed: '+k)
  for k,v in [('cond',planjob['condition']),('seed',planjob['seed']),('code_seed',planjob['seed']),('task_max_gap',planjob['task_max_gap'])]:require(cfg[k]==v,'Logical identity changed: '+k)
  scientific={k:v for k,v in cfg.items() if k!='population'}
  require(proof['config_sha256']==hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest(),'Config proof digest mismatch')
  require(proof['torch_version']=='2.9.1+cu129' and proof['cuda_version']=='12.9' and proof['cudnn_version']==91002 and proof['cuda_device_name']=='Tesla T4' and proof['cuda_capability']==[7,5],'Original numeric stack changed')
  require(proof['deterministic_algorithms'] and not proof['cudnn_tf32'] and not proof['matmul_tf32'] and not proof['cudnn_benchmark'],'Original FP32 settings changed')
  require(cfg['clock_seed']==1000*cfg['seed']+2 and cfg['pgc_frac']=={'gc100':0.,'pgc33':.33}[cfg['cond']],'Map/clock/population identity')
  receipt.update(condition=cfg['cond'],training_seed=cfg['seed'],training_task_max_gap=cfg['task_max_gap'],
                 config_sha256=sha(folder/'config.json'),provenance_sha256=sha(folder/'provenance.json'),
                 training_protocol_sha256=sha(PROTOCOL),worker_exit_code=worker['exit_code'])
  receipt['resume']=inspect_checkpoint(folder/'resume.pt',cfg,proof)
  if job['complete_1B']:
   summary=read(folder/'summary.json')
   require(sha(folder/'summary.json')==job['summary_source_sha256'],'Terminal summary bytes changed')
   receipt['final']=inspect_checkpoint(folder/'final.pt',cfg,proof,summary)
   receipt['frozen_final_authentication']=authenticate_run(folder)
   require(sha(folder/'initial.pt')==summary['initial_sha256'],'Original initialization bytes changed')
   require(receipt['resume']['frames']==receipt['final']['frames'],'Final resume boundary mismatch')
  else:
   require(not (folder/'final.pt').exists() and not (folder/'summary.json').exists(),'Failed run unexpectedly has terminal artifact')
   receipt['last_accepted_update']=job['latest_technical']['updates']
   receipt['last_accepted_frames']=job['latest_technical']['frames']
   receipt['updates_after_saved_resume']=job['latest_technical']['updates']-receipt['resume']['updates']
   require(0<=receipt['updates_after_saved_resume']<200,'Unexpected resume rollback distance')
  receipt['artifacts']={p.relative_to(dest).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.rglob('*')) if p.is_file()}
  receipt['complete']=True
 except BaseException as exc:
  receipt['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
 finally:
  receipt.update(finished_utc=utc(),seconds=time.monotonic()-started)
  save(OUT/(name+'.json'),receipt)
 return receipt


def main():
 inv=read(INVENTORY);ledger=read(cloud.LEDGER);ledger_memory=copy.deepcopy(ledger);plan=read(PLAN)
 records={r['name']:r for r in ledger['runs']};jobs={j['name']:j for j in plan['jobs']}
 require(inv['complete'] and len(inv['jobs'])==7,'Unexpected fixed collection cohort')
 result=dict(complete=False,started_utc=utc(),inventory_sha256=sha(INVENTORY),ledger_sha256_at_start=sha(cloud.LEDGER),
  maximum_parallel_downloads=2,ledger_write_operations=0,VM_API_operations=0,policy_forward_calls=0,jobs=[])
 save(OUT/'summary.json',result)
 with cf.ThreadPoolExecutor(max_workers=2) as pool:
  work={pool.submit(one,j,records[j['name']],jobs[j['name']]):j['name'] for j in inv['jobs']}
  for task in cf.as_completed(work):
   item=task.result();result['jobs'].append(dict(name=item['name'],complete=item['complete'],receipt=work[task]+'.json',receipt_sha256=sha(OUT/(work[task]+'.json'))))
   save(OUT/'summary.json',result)
   print(json.dumps(dict(name=item['name'],complete=item['complete'],frames=item.get('final',item.get('resume',{})).get('frames'),error=item.get('error',{}).get('message'))),flush=True)
 require(ledger==ledger_memory,'Original in-memory ledger was mutated')
 result.update(complete=all(x['complete'] for x in result['jobs']),finished_utc=utc(),ledger_sha256_at_end=sha(cloud.LEDGER),
               original_in_memory_ledger_unchanged=True,external_ledger_change_detected=sha(cloud.LEDGER)!=result['ledger_sha256_at_start'])
 save(OUT/'summary.json',result)
 return 0 if result['complete'] else 1


if __name__=='__main__':sys.exit(main())
