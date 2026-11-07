"""Independent disposable-checkout report reconstruction; no experiment runs."""
from pathlib import Path
import copy,datetime,hashlib,json,os,re,shutil,subprocess,time
ROOT=Path('/Users/aaravsinha/predictive-grid-cells-rl').resolve();RELEASE=ROOT/'results/native_turn_release';PY='/opt/anaconda3/bin/python';COMMIT='aa7ee742bd8aeb2e664471226c286fa0ed1495c2'
stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bsha(b):return hashlib.sha256(b).hexdigest()
def log(s):print(s,flush=True)
proof=json.loads((RELEASE/'reproduction_progress.json').read_text());assert proof['stage']=='reports_running';work=Path(proof['worktree']);base=work.parent
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()==COMMIT
manifest=json.loads((RELEASE/'archives/manifest.json').read_text());entries={e['path']:e for e in manifest['files']}
proof['report_procedure_source']=Path(__file__).read_text();proof['report_procedure_source_sha256']=sha(__file__);proof['resumed_report_utc']=stamp();started=time.monotonic()
guard_source='''import os,sys,json,runpy
from pathlib import Path
checkout=Path(sys.argv[1]).resolve();original=Path(sys.argv[2]).resolve();script=sys.argv[3];record=Path(sys.argv[4]);extra=sys.argv[5:];reads=set();writes=set();missing=set();blocked=[]
def hook(event,args):
 if event not in ('open','os.listdir','os.scandir') or not args or isinstance(args[0],int):return
 try:p=Path(os.fsdecode(args[0])).resolve()
 except(TypeError,ValueError):return
 if p.is_relative_to(original):
  blocked.append(dict(event=event,path=str(p)));raise PermissionError('Original workspace access forbidden: '+str(p))
 if event=='open' and p.is_relative_to(checkout):
  mode=args[1]if len(args)>1 else None;flags=args[2]if len(args)>2 else 0
  writing=(isinstance(mode,str)and any(c in mode for c in 'wax+'))or(isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR)))
  (writes if writing else(reads if p.is_file()else missing)).add(str(p.relative_to(checkout)))
sys.addaudithook(hook)
try:(original/'README.md').read_bytes()
except PermissionError:tested=True
else:raise AssertionError('guard failed')
assert len(blocked)==1;blocked.clear();sys.dont_write_bytecode=True;sys.path[:0]=[str(checkout),str(checkout/'scripts')];os.chdir(checkout);sys.argv=[str(checkout/script),*extra];success=False
try:runpy.run_path(str(checkout/script),run_name='__main__');success=True
finally:record.write_text(json.dumps(dict(success=success,guard_self_test=tested,blocked_original_workspace_accesses=blocked,checkout_reads=sorted(reads),checkout_writes=sorted(writes),nonexistent_read_probes=sorted(missing)),indent=2)+'\\n')
'''
guard=base/'guarded_report.py';guard.write_text(guard_source);proof['report_guard']=dict(source=guard_source,sha256=sha(guard),scope='Python file/directory audit guard installed before project imports; not an operating-system sandbox. Static integrity tensors may be read; no models or environment are executed.')
specs=[
 ('results/native_turn_teacher_preflight/report',[['scripts/report_native_turn_teacher_preflight.py']]),
 ('results/native_turn_original_memory/report',[['scripts/report_native_turn_original_memory.py']]),
 ('results/native_turn_distillation/report',[['scripts/report_native_turn_distillation.py']]),
 ('results/native_turn_geometry/report_original',[['scripts/report_native_turn_geometry.py','--endpoints','original'],['scripts/report_native_turn_geometry_failures.py','--report-directory','report_original']]),
 ('results/native_turn_geometry/report_all',[['scripts/report_native_turn_geometry.py','--endpoints','original','clean','noise'],['scripts/report_native_turn_geometry_failures.py','--report-directory','report_all']]),
 ('results/native_turn_imitation_decomposition',[['scripts/decompose_native_turn_imitation.py']]),
 ('results/native_turn_imitation_prefix',[['scripts/decompose_native_turn_prefix.py']]),
]
pdf_pattern=re.compile(rb'/(CreationDate|ModDate)\s*\((D:[^)]*)\)')
def pdfnorm(b):return pdf_pattern.sub(lambda m:b'/'+m[1]+b' (DATE_METADATA)',b)
reads_all=set();comparisons=[];call=len(proof['commands'])
for previous in proof['reports']:
 comparisons.extend(previous['outputs'])
 for c in previous['commands']:reads_all.update(c['guard']['checkout_reads'])
specs=specs[len(proof['reports']):]
for folder,commands in specs:
 out=work/folder;saved=base/'archived_reports'/folder;saved.parent.mkdir(parents=True,exist_ok=True);assert out.is_dir()and not saved.exists();out.rename(saved)
 report=dict(folder=folder,commands=[],preserved_archive_report=str(saved),outputs=[])
 for command in commands:
  record=base/f'guard_{call}.json';stdout=base/f'report_{call}.stdout';stderr=base/f'report_{call}.stderr'
  cmd=[PY,'-I','-B',str(guard),str(work),str(ROOT),command[0],str(record),*command[1:]]
  env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(base/'matplotlib'))
  log('START '+ ' '.join(command));t=time.monotonic()
  with stdout.open('w')as a,stderr.open('w')as b:p=subprocess.run(cmd,cwd=work,env=env,stdout=a,stderr=b)
  info=dict(command=cmd,cwd=str(work),exit_code=p.returncode,seconds=time.monotonic()-t,stdout=stdout.read_text(),stderr=stderr.read_text());proof['commands'].append(info);report['commands'].append(info)
  (RELEASE/'reproduction_progress.json').write_text(json.dumps(proof,indent=2)+'\n')
  assert p.returncode==0,(command,info['stderr'])
  g=json.loads(record.read_text());assert g['success']and g['guard_self_test']and not g['blocked_original_workspace_accesses'];readset=set(g['checkout_reads']);generated={str(x.relative_to(work))for x in out.rglob('*')if x.is_file()};assert readset<=set(entries)|generated,sorted(readset-set(entries)-generated)
  reads_all|=readset;info['guard']=g;call+=1;log('DONE '+command[0])
 if folder.endswith('report_all'):
  change=json.loads((saved/'presentation_addendum.json').read_text());p=out/'README.md';before=p.read_bytes();assert bsha(before)==change['previous_sha256'];needle=change['previous_text'].encode();assert before.count(needle)==1;after=before.replace(needle,change['updated_text'].encode());assert bsha(after)==change['updated_sha256']and after==(saved/'README.md').read_bytes();p.write_bytes(after);report['presentation_addendum']=dict(archive_sha256=sha(saved/'presentation_addendum.json'),previous_sha256=bsha(before),updated_sha256=bsha(after),exact_replacement_verified=True)
 names=sorted(p.name for p in out.iterdir()if p.is_file()and(p.suffix in('.png','.pdf','.md','.csv')or p.name in('summary.json','validation.json','failure_patterns.json')))
 assert 'summary.json'in names and 'README.md'in names
 pdfs={}
 for name in names:
  if not name.endswith('.pdf'):continue
  before=(saved/name).read_bytes();after=(out/name).read_bytes();assert pdfnorm(before)==pdfnorm(after),name
  pdfs[name]=(bsha(before),bsha(after))
 for name in names:
  before=(saved/name).read_bytes();after=(out/name).read_bytes();equal=before==after;cmp=dict(path=folder+'/'+name,archive_sha256=bsha(before),reproduced_sha256=bsha(after),byte_identical=equal)
  if not equal:
   if name.endswith('.pdf'):
    assert pdf_pattern.search(before)and pdf_pattern.search(after);cmp.update(only_pdf_creation_modification_dates_differ=True,archive_pdf_dates=[(a.decode(),b.decode())for a,b in pdf_pattern.findall(before)],reproduced_pdf_dates=[(a.decode(),b.decode())for a,b in pdf_pattern.findall(after)])
   elif name.endswith('.json'):
    a=json.loads(before);b=json.loads(after);allowed=[]
    if name=='summary.json'and 'figures'in a:
     for file,(old,new)in pdfs.items():
      assert a['figures'][file]==old and b['figures'][file]==new;b['figures'][file]=old
      if old!=new:allowed.append('figures.'+file+' (verified date-only PDF hash)')
    if 'created_utc'in a and 'created_utc'in b:
     assert name=='validation.json'or folder in('results/native_turn_imitation_decomposition','results/native_turn_imitation_prefix');b['created_utc']=a['created_utc'];allowed.append('created_utc')
    if name=='failure_patterns.json':
     assert a['validation_sha256']==sha(saved/'validation.json')and b['validation_sha256']==sha(out/'validation.json');b['validation_sha256']=a['validation_sha256'];allowed.append('validation_sha256 (verified timestamp-only validation)')
    assert a==b,(folder,name,'Unexpected JSON difference');assert allowed;cmp['only_explicit_metadata_differences']=allowed
   else:raise AssertionError(('Unexpected non-JSON difference',folder,name))
  report['outputs'].append(cmp);comparisons.append(cmp)
 # Generated audit bindings remain with generated output; archived audits are restored only to original archived bytes.
 target=base/'reproduced_reports'/folder;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();out.rename(target);saved.rename(out);report['reproduced_output']=str(target);report['archived_report_restored_exactly']=True
 proof['reports'].append(report);proof['stage']='reports_running';(RELEASE/'reproduction_progress.json').write_text(json.dumps(proof,indent=2)+'\n');log('COMPARE PASS '+folder)
for name,e in entries.items():assert sha(work/name)==e['sha256'],name
assert sha(RELEASE/'archives/manifest.json')==proof['manifest_sha256']and sha(RELEASE/'archives/index.json')==proof['index_sha256']
proof.update(success=True,stage='complete',finished_utc=stamp(),report_seconds=time.monotonic()-started,report_input_inventory=dict(checkout_files_read=len(reads_all),all_reads_manifest_members_or_verified_generated_reports=True,original_workspace_reads=0,read_paths=sorted(reads_all)),final_integrity=dict(all_extracted_manifest_members_restored_and_unchanged=True,manifest_index_unchanged=True,original_workspace_inputs_not_written_by_reproduction=True),comparison_totals=dict(outputs=len(comparisons),byte_identical=sum(c['byte_identical']for c in comparisons),pdf_date_metadata_only=sum(c.get('only_pdf_creation_modification_dates_differ',False)for c in comparisons),json_explicit_metadata_only=sum('only_explicit_metadata_differences'in c for c in comparisons)))
(RELEASE/'reproduction_proof.json').write_text(json.dumps(proof,indent=2)+'\n');(RELEASE/'reproduction_progress.json').write_text(json.dumps(proof,indent=2)+'\n');log('ALL REPRODUCTION PASSED '+json.dumps(proof['comparison_totals']))
