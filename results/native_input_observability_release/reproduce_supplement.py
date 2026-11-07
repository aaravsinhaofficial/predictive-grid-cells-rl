"""Independent fresh-checkout observer extraction and saved-data reproduction."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,os,re,subprocess,tarfile,tempfile,time
ROOT=Path('/Users/aaravsinha/predictive-grid-cells-rl');REL=ROOT/'results/native_input_observability_release';ARCH=REL/'archives';PY='/opt/anaconda3/bin/python';COMMIT='4f8f7dd7ee4139ed448325f217caf8d93f5a371e';started=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bh(b):return hashlib.sha256(b).hexdigest()
def log(s):print(s,flush=True)
def safe(name):
 p=PurePosixPath(name);assert name and str(p)==name and not p.is_absolute()and'..'not in p.parts and'\\'not in name and'\0'not in name
prior=json.loads((ROOT/'results/native_turn_release/reproduction_proof.json').read_text());assert prior['success'];prior_work=Path(prior['worktree'])
pm=json.loads((ROOT/'results/native_turn_release/archives/manifest.json').read_text());pe={e['path']:e for e in pm['files']}
body=(ARCH/'manifest.json').read_bytes();m=json.loads(body);idx=json.loads((ARCH/'index.json').read_text());entries={e['path']:e for e in m['files']}
assert bh(body)==idx['manifest_sha256']and m['source_commit']==idx['source_commit']==COMMIT and m['completion']['complete']
assert m['primary_archive_dependency']['manifest_sha256']==prior['manifest_sha256']and m['primary_archive_dependency']['index_sha256']==prior['index_sha256']
base=Path(tempfile.mkdtemp(prefix='native-observer-reproduce-'));work=base/'checkout';cmd=['git','worktree','add','--detach',str(work),COMMIT];run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert run.returncode==0,run.stderr
proof=dict(complete=False,source_commit=COMMIT,worktree=str(work),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),procedure_source=Path(__file__).read_text(),procedure_sha256=sha(__file__),manifest_sha256=bh(body),index_sha256=sha(ARCH/'index.json'),checkout_creation=dict(command=cmd,exit_code=run.returncode),primary_dependency=dict(source_commit=pm['source_commit'],manifest_sha256=prior['manifest_sha256'],index_sha256=prior['index_sha256'],proof_sha256=sha(ROOT/'results/native_turn_release/reproduction_proof.json')))
(REL/'reproduction_progress.json').write_text(json.dumps(proof,indent=2)+'\n')
linked=[];matching=[];new_commit_files=[]
for name,e in pe.items():
 safe(name);source=prior_work/name;assert source.is_file()and not source.is_symlink()and sha(source)==e['sha256'],name
 target=work/name
 if target.exists():
  assert target.is_file()and not target.is_symlink()
  (matching if sha(target)==e['sha256']else new_commit_files).append(name)
 else:target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target);linked.append(name)
proof['primary_dependency_reconstruction']=dict(method='Verified immutable primary members hard-linked from the previously independently extracted checkout; write guard forbids changes to all shared inputs. This copies no original-workspace data.',linked_files=len(linked),existing_identical_files=len(matching),new_commit_files_retained=new_commit_files,all_source_primary_members_rehashed=True)
log('Verified primary dependency; fresh checkout '+str(work))
seen=set();parts=[]
for part in idx['parts']:
 path=ARCH/part['path'];assert path.name==part['path']and path.stat().st_size==part['bytes']and sha(path)==part['sha256'];expected=set(part['members']);assert not seen&expected;got=set();embedded=False
 with tarfile.open(path,'r|gz')as tf:
  for member in tf:
   safe(member.name);assert member.isfile()and(member.uid,member.gid,member.mtime,member.mode)==(0,0,0,0o644)and not member.uname and not member.gname
   if member.name=='results/native_input_observability_release/manifest.json':assert not embedded and tf.extractfile(member).read()==body;embedded=True;continue
   assert member.name in expected and member.name not in got;e=entries[member.name];assert member.size==e['bytes'];target=work/member.name;assert target.resolve().is_relative_to(work.resolve())and not target.is_symlink()
   assert not any(p.is_symlink()for p in target.parents if p.is_relative_to(work))
   existing=target.exists()
   if existing:assert sha(target)==e['sha256'],('Unexpected existing supplement bytes',member.name)
   else:target.parent.mkdir(parents=True,exist_ok=True)
   h=hashlib.sha256();count=0;dst=None if existing else target.open('wb')
   try:
    with tf.extractfile(member)as stream:
     for b in iter(lambda:stream.read(1024*1024),b''):
      h.update(b);count+=len(b)
      if dst is not None:dst.write(b)
   finally:
    if dst is not None:dst.close()
   assert count==e['bytes']and h.hexdigest()==e['sha256']==sha(target);got.add(member.name)
  assert got==expected and embedded
 seen|=got;parts.append(dict(path=part['path'],sha256=part['sha256'],members=len(got),streamed_and_readback_verified=True))
assert seen==set(entries)and len(seen)==m['file_count'];tracked=set(filter(None,subprocess.check_output(['git','ls-tree','-r','--name-only','-z',COMMIT],cwd=work).decode().split('\0')))
for name in set(m['source_files'])|(set(entries)&tracked):assert bh(subprocess.check_output(['git','show',f'{COMMIT}:{name}'],cwd=work))==sha(work/name)==entries[name]['sha256'],name
proof['extraction']=dict(parts=parts,files=len(seen),uncompressed_bytes=m['uncompressed_bytes'],safe_regular_normalized_members=True,all_source_and_tracked_supplement_files_match_commit=True,source_files=len(m['source_files']),all_existing_overlap_bytes_identical=True)
log('Observer supplement extracted and pinned source checked.')
guard_source='''from pathlib import Path
import os,sys,json,runpy
checkout=Path(sys.argv[1]).resolve();original=Path(sys.argv[2]).resolve();prior=Path(sys.argv[3]).resolve();record=Path(sys.argv[4]);script=sys.argv[5];reads=set();writes=set();blocked=[];missing=set();output=checkout/'results/native_input_observability/report'
def hook(event,args):
 if event not in ('open','os.listdir','os.scandir')or not args or isinstance(args[0],int):return
 try:p=Path(os.fsdecode(args[0])).resolve()
 except(TypeError,ValueError):return
 if p.is_relative_to(original)or p.is_relative_to(prior):blocked.append(dict(event=event,path=str(p)));raise PermissionError('Non-checkout project access forbidden: '+str(p))
 if event=='open'and p.is_relative_to(checkout):
  mode=args[1]if len(args)>1 else None;flags=args[2]if len(args)>2 else 0;writing=(isinstance(mode,str)and any(x in mode for x in 'wax+'))or(isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR)))
  if writing and not p.is_relative_to(output):raise PermissionError('Shared input is read-only: '+str(p))
  (writes if writing else(reads if p.is_file()else missing)).add(str(p.relative_to(checkout)))
sys.addaudithook(hook)
for forbidden in(original,prior):
 try:(forbidden/'README.md').read_bytes()
 except PermissionError:pass
 else:raise AssertionError('Read guard self-test failed')
assert len(blocked)==2;blocked.clear();sys.dont_write_bytecode=True;sys.path[:0]=[str(checkout),str(checkout/'scripts')];os.chdir(checkout);sys.argv=[str(checkout/script)];success=False
try:runpy.run_path(str(checkout/script),run_name='__main__');success=True
finally:record.write_text(json.dumps(dict(success=success,both_forbidden_roots_self_tested=True,blocked_forbidden_accesses=blocked,checkout_reads=sorted(reads),checkout_writes=sorted(writes),nonexistent_read_probes=sorted(missing)),indent=2)+'\\n')
'''
guard=base/'guard.py';guard.write_text(guard_source);folder='results/native_input_observability/report';out=work/folder;saved=base/'archived_report';assert out.is_dir()and not saved.exists();out.rename(saved)
grecord=base/'guard.json';stdout=base/'report.stdout';stderr=base/'report.stderr';command=[PY,'-I','-B',str(guard),str(work),str(ROOT),str(prior_work),str(grecord),'scripts/report_native_input_observability.py'];env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(prior_work.parent/'matplotlib'))
with stdout.open('w')as a,stderr.open('w')as b:result=subprocess.run(command,cwd=work,env=env,stdout=a,stderr=b)
proof['command']=dict(command=command,exit_code=result.returncode,stdout=stdout.read_text(),stderr=stderr.read_text());(REL/'reproduction_progress.json').write_text(json.dumps(proof,indent=2)+'\n');assert result.returncode==0,proof['command']
g=json.loads(grecord.read_text());assert g['success']and g['both_forbidden_roots_self_tested']and not g['blocked_forbidden_accesses'];generated={str(p.relative_to(work))for p in out.iterdir()if p.is_file()};assert set(g['checkout_reads'])<=set(entries)|set(pe)|generated
proof['read_guard']=dict(source=guard_source,source_sha256=sha(guard),result=g,scope='Python file/directory audit guard; both original workspace and prior checkout are forbidden. Every linked primary input is write-protected. Not an OS sandbox.')
pattern=re.compile(rb'/(CreationDate|ModDate)\s*\((D:[^)]*)\)');norm=lambda b:pattern.sub(lambda m:b'/'+m[1]+b' (DATE_METADATA)',b);comparisons=[]
for name in('summary.json','validation.json','README.md','probability_sensitivity.png','probability_sensitivity.pdf','geometry_criteria.png','geometry_criteria.pdf'):
 before=(saved/name).read_bytes();after=(out/name).read_bytes();c=dict(path=folder+'/'+name,archive_sha256=bh(before),reproduced_sha256=bh(after),byte_identical=before==after)
 if before!=after:
  if name.endswith('.pdf'):assert pattern.search(before)and pattern.search(after)and norm(before)==norm(after);c['only_pdf_date_metadata_differs']=True
  elif name=='validation.json':a=json.loads(before);b=json.loads(after);b['created_utc']=a['created_utc'];assert a==b;c['only_created_utc_differs']=True
  else:raise AssertionError(('Scientific/presentation difference',name))
 comparisons.append(c)
s=json.loads((out/'summary.json').read_text());assert s['complete']and s['all_cohorts_complete']and len(s['rows'])==6656
assert all(s['populations'][c]['named']['geometry_qualified']==s['populations'][c]['named']['conditional_completed']==s['populations'][c]['named']['candidates']==0 for c in('gc100','pgc33'))
regenerated=base/'reproduced_report';out.rename(regenerated);saved.rename(out)
for name,e in entries.items():assert sha(work/name)==e['sha256']
for name in set(g['checkout_reads'])&set(pe):assert sha(work/name)==pe[name]['sha256']and sha(prior_work/name)==pe[name]['sha256']
assert sha(ARCH/'manifest.json')==proof['manifest_sha256']and sha(ARCH/'index.json')==proof['index_sha256']
assert sha(ROOT/'results/native_turn_release/reproduction_proof.json')==proof['primary_dependency']['proof_sha256']
proof.update(complete=True,no_scientific_rerun=True,named_cases=6656,conditional_tests=0,candidates=0,outputs=comparisons,comparison_totals=dict(outputs=len(comparisons),byte_identical=sum(x['byte_identical']for x in comparisons),pdf_date_metadata_only=sum(x.get('only_pdf_date_metadata_differs',False)for x in comparisons),validation_timestamp_only=sum(x.get('only_created_utc_differs',False)for x in comparisons)),all_supplement_members_restored_unchanged=True,all_read_primary_inputs_unchanged_in_both_checkouts=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started)
(REL/'reproduction_proof.json').write_text(json.dumps(proof,indent=2)+'\n');log(json.dumps(dict(complete=True,files=len(entries),outputs=proof['comparison_totals'],proof_sha256=sha(REL/'reproduction_proof.json'))))
