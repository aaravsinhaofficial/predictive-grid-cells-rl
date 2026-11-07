"""Verify the exploratory overlay and reproduce only its descriptive saved-data report."""
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile,subprocess,os,re,time,datetime
ROOT=Path('/Users/aaravsinha/predictive-grid-cells-rl');RELEASE=ROOT/'results/native_turn_counterbalanced_release';ARCH=RELEASE/'archives';PY='/opt/anaconda3/bin/python';started=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bh(b):return hashlib.sha256(b).hexdigest()
primary=json.loads((ROOT/'results/native_turn_release/reproduction_proof.json').read_text());assert primary['success'];work=Path(primary['worktree']);base=work.parent;commit=primary['source_commit']
body=(ARCH/'manifest.json').read_bytes();manifest=json.loads(body);index=json.loads((ARCH/'index.json').read_text());assert bh(body)==index['manifest_sha256']and manifest['source_commit']==index['source_commit']==commit
assert manifest['scientific_run_complete']is False and manifest['old16_input_equivalence']=='rejected'and(manifest['failed_input_equivalence_rows'],manifest['downstream_cell_failure_rows'])==(12,6)
assert manifest['primary_archive_dependency']['manifest_sha256']==primary['manifest_sha256']and manifest['primary_archive_dependency']['index_sha256']==primary['index_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()==commit
entries={e['path']:e for e in manifest['files']};assert len(entries)==manifest['file_count'];seen=set();parts=[];source_checks=0
proof=dict(complete=False,scope='Exploratory incomplete-equivalence overlay; descriptive report reproduction only. Scientific run remains incomplete and rejected.',source_commit=commit,primary_reproduction_proof_sha256=sha(ROOT/'results/native_turn_release/reproduction_proof.json'),primary_manifest_sha256=primary['manifest_sha256'],worktree=str(work),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),procedure_source=Path(__file__).read_text(),procedure_sha256=sha(__file__),manifest_sha256=bh(body),index_sha256=sha(ARCH/'index.json'))
for part in index['parts']:
 path=ARCH/part['path'];assert path.name==part['path']and path.stat().st_size==part['bytes']and sha(path)==part['sha256'];expected=set(part['members']);assert not seen&expected;got=set();embedded=False
 with tarfile.open(path,'r|gz')as tf:
  for m in tf:
   pp=PurePosixPath(m.name);assert not pp.is_absolute()and'..'not in pp.parts and str(pp)==m.name and'\\'not in m.name
   assert m.isfile()and(m.uid,m.gid,m.mtime,m.mode)==(0,0,0,0o644)and not m.uname and not m.gname
   if m.name=='results/native_turn_counterbalanced_release/manifest.json':assert not embedded and tf.extractfile(m).read()==body;embedded=True;continue
   assert m.name in expected and m.name not in got;e=entries[m.name];assert m.size==e['bytes'];out=work/m.name;assert out.resolve().is_relative_to(work.resolve());assert not out.is_symlink()and not any(p.is_symlink()for p in out.parents if p.is_relative_to(work))
   if out.exists():assert sha(out)==e['sha256'],('Unexpected prior bytes',m.name)
   out.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
   with tf.extractfile(m)as inp,out.open('wb')as dst:
    for b in iter(lambda:inp.read(1024*1024),b''):h.update(b);count+=len(b);dst.write(b)
   assert count==e['bytes']and h.hexdigest()==e['sha256']==sha(out);got.add(m.name)
  assert got==expected and embedded
 seen|=got;parts.append(dict(path=part['path'],sha256=part['sha256'],members=len(got),all_streamed_and_readback_hashes_match=True))
assert seen==set(entries)
for name in manifest['source_files']:
 assert bh(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=work))==sha(work/name)==entries[name]['sha256'];source_checks+=1
proof['extraction']=dict(parts=parts,files=len(seen),uncompressed_bytes=manifest['uncompressed_bytes'],all_safe_regular_normalized_members=True,all_existing_overlap_bytes_identical=True,source_files_exact_to_commit=source_checks)
folder='results/native_turn_counterbalanced_readout/report';output=work/folder;saved=base/'overlay_archived_report';assert output.exists()and not saved.exists();output.rename(saved)
guard=base/'overlay_guard.py';guard.write_text(primary['report_guard']['source']);guardlog=base/'overlay_guard.json';stdout=base/'overlay_report.stdout';stderr=base/'overlay_report.stderr';script='scripts/report_native_turn_counterbalanced_readout.py';cmd=[PY,'-I','-B',str(guard),str(work),str(ROOT),script,str(guardlog),'--report'];env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(base/'matplotlib'))
with stdout.open('w')as a,stderr.open('w')as b:r=subprocess.run(cmd,cwd=work,env=env,stdout=a,stderr=b)
proof['command']=dict(command=cmd,exit_code=r.returncode,stdout=stdout.read_text(),stderr=stderr.read_text());assert r.returncode==0,proof['command']
g=json.loads(guardlog.read_text());assert g['success']and g['guard_self_test']and not g['blocked_original_workspace_accesses'];primary_entries={e['path']:e for e in json.loads((ROOT/'results/native_turn_release/archives/manifest.json').read_text())['files']};generated={str(p.relative_to(work))for p in output.iterdir()if p.is_file()};assert set(g['checkout_reads'])<=set(entries)|set(primary_entries)|generated
proof['read_guard']=dict(source=guard.read_text(),source_sha256=sha(guard),result=g,all_project_inputs_in_verified_primary_or_overlay_or_generated_report=True)
pattern=re.compile(rb'/(CreationDate|ModDate)\s*\((D:[^)]*)\)');normalize=lambda b:pattern.sub(lambda m:b'/'+m[1]+b' (DATE_METADATA)',b);comparisons=[]
for p in sorted(output.iterdir()):
 if p.name=='artifacts.json':continue
 a=(saved/p.name).read_bytes();b=p.read_bytes();row=dict(path=folder+'/'+p.name,archive_sha256=bh(a),reproduced_sha256=bh(b),byte_identical=a==b)
 if a!=b:assert p.suffix=='.pdf'and pattern.search(a)and pattern.search(b)and normalize(a)==normalize(b),p.name;row['only_pdf_date_metadata_differs']=True
 comparisons.append(row)
s=json.loads((output/'summary.json').read_text());v=json.loads((output/'independent_validation.json').read_text());assert s['report_complete']and s['scientific_run_complete']is False and s['old16_input_equivalence']=='rejected'and s['failed_equivalence_comparisons']==12
assert v['arithmetic_audit_complete']and len(v['failed_equivalence_rows'])==12 and len(v['downstream_cell_failure_rows'])==6 and v['phase_rows']==196608
regenerated=base/'overlay_reproduced_report';output.rename(regenerated);saved.rename(output)
for name,e in entries.items():assert sha(work/name)==e['sha256']
# Shared primary data were read-only; verify all successfully read primary members.
for name in set(g['checkout_reads'])&set(primary_entries):assert sha(work/name)==primary_entries[name]['sha256']
proof.update(complete=True,scientific_run_complete=False,old16_input_equivalence='rejected',failed_equivalence_comparisons=12,downstream_cell_failure_rows=6,outputs=comparisons,comparison_totals=dict(outputs=len(comparisons),byte_identical=sum(x['byte_identical']for x in comparisons),pdf_date_metadata_only=sum(x.get('only_pdf_date_metadata_differs',False)for x in comparisons)),all_overlay_members_restored_unchanged=True,all_read_primary_members_unchanged=True,no_scientific_rerun=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started)
(RELEASE/'reproduction_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(complete=True,files=len(entries),outputs=proof['comparison_totals'],proof_sha256=sha(RELEASE/'reproduction_proof.json')),indent=2))
