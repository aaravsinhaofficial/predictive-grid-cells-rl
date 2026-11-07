"""Bounded observer-data supplement using the existing normalized safe writer."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import package_nonlinear_ring_outputs as archive
RELEASE=Path('results/native_input_observability_release');COMMIT='4f8f7dd7ee4139ed448325f217caf8d93f5a371e';PRIMARY='aa7ee742bd8aeb2e664471226c286fa0ed1495c2'
TREE='results/native_input_observability';PROTOCOL='docs/research/native_input_observability_protocol.json'
EXTRA=('scripts/report_native_input_observability.py','scripts/audit_native_input_observability_report.py','tests/test_report_native_input_observability.py','docs/research/native_input_observability_protocol_draft.json','docs/research/native_input_observability_review.json','docs/research/native_input_observability_report_review.json')
DENSE={f'results/native_turn_geometry_resolution/{c}/original/{n}.json'for c in('gc100','pgc33')for n in('summary','execution')}
archive.TREES=(TREE,);archive.MANIFEST_NAME=str(RELEASE/'manifest.json')
def read(p):return json.loads(Path(p).read_text())
def scope(report,validation,audit):
 assert report['complete']and report['all_cohorts_complete']and len(report['rows'])==report['expected_primary_rows']==6656
 assert validation['complete']and audit['complete']and audit['rows_independently_matched']==6656
 assert sum(v['numerical_geometry_rows_recomputed']for v in validation['cohorts'])==4160
 assert sum(v['numerical_normal_cases_recomputed']for v in validation['cohorts'])==0
 assert all(report['populations'][c]['named']['geometry_qualified']==report['populations'][c]['named']['conditional_completed']==report['populations'][c]['named']['candidates']==0 for c in('gc100','pgc33'))
def main():
 primary=ROOT/'results/native_turn_release/archives';pm=read(primary/'manifest.json');pi=read(primary/'index.json');proof=read(ROOT/'results/native_turn_release/reproduction_proof.json');assert proof['success']and pm['source_commit']==pi['source_commit']==PRIMARY
 assert archive.sha(primary/'manifest.json')==pi['manifest_sha256']==proof['manifest_sha256']and archive.sha(primary/'index.json')==proof['index_sha256']
 deps={e['path']:e for e in pm['files']};p=read(ROOT/PROTOCOL);out=ROOT/TREE/'report';report=read(out/'summary.json');validation=read(out/'validation.json');audit=read(out/'independent_review.json');scope(report,validation,audit)
 try:scope(dict(report,all_cohorts_complete=False),validation,audit)
 except AssertionError:pass
 else:raise AssertionError('Incomplete cohort accepted')
 assert validation['summary_sha256']==audit['summary_sha256']==archive.sha(out/'summary.json')and audit['full_array_validation_sha256']==archive.sha(out/'validation.json')
 assert report['protocol_sha256']==validation['protocol_sha256']==audit['protocol_sha256']==archive.sha(ROOT/PROTOCOL)
 sources=set(p['source_sha256'])|set(validation['arithmetic_source_sha256'])|set(EXTRA);names={PROTOCOL,*sources};names.update(archive.tree_files(ROOT,TREE))
 for mapping in(p['source_sha256'],validation['arithmetic_source_sha256']):
  for path,digest in mapping.items():archive.prove(ROOT,path,dict(sha256=digest))
 for path,digest in p['asset_sha256'].items():
  if path in deps:assert deps[path]['sha256']==digest;archive.prove(ROOT,path,deps[path])
  else:assert path in DENSE;archive.prove(ROOT,path,dict(sha256=digest));names.add(path)
 assert DENSE<=names
 for condition in('gc100','pgc33'):
  b=ROOT/TREE/condition;t=read(b/'summary.json');e=read(b/'execution.json');outer=read(ROOT/TREE/f'{condition}_outer_execution.json')
  assert t['complete']and e['complete']and e['exit_code']==0 and outer['complete']and outer['exit_code']==0
  assert t['protocol_sha256']==e['protocol_sha256']==archive.sha(ROOT/PROTOCOL)
  assert(t['primary_cases'],t['geometry_qualified_rows'],t['candidates'])==(3328,0,0)
  assert e['summary_sha256']==archive.sha(b/'summary.json')
  for path,row in t['artifacts'].items():archive.prove(ROOT,f'{TREE}/{condition}/{path}',row)
  for n in('screen','conditional'):assert t[n+'_summary_sha256']==archive.sha(b/(n+'_summary.json'))and read(b/(n+'_summary.json'))['complete']
  assert read(b/'conditional_summary.json')['rows']==[]
  for name in(f'{condition}.log',):assert(ROOT/TREE/name).is_file()
 execution=read(ROOT/TREE/'report_execution.json');assert execution['complete']and execution['exit_code']==0
 assert execution['summary_sha256']==archive.sha(out/'summary.json')and execution['validation_sha256']==archive.sha(out/'validation.json')
 assert(ROOT/TREE/'report_execution.log').is_file()and(ROOT/TREE/'report_execution_attempt1.log').is_file()
 tracked=set(filter(None,subprocess.check_output(['git','ls-tree','-r','--name-only','-z',COMMIT],cwd=ROOT).decode().split('\0')));entries=[]
 for name in sorted(names):
  path=archive.check_name_and_content(ROOT,name);entry=dict(path=name,bytes=path.stat().st_size,sha256=archive.sha(path));entries.append(entry)
  if name in tracked:assert entry['sha256']==archive.commit_sha(ROOT,COMMIT,name)
  if name in sources:assert name in tracked
  if name in deps:assert entry==deps[name]
 manifest=dict(version='native_input_observability_outputs_v1',source_commit=COMMIT,title='Fixed six-action observer: complete negative ring screen',
  completion=dict(complete=True,issues=[],passed_checks=['Both complete original cohorts and execution logs','6656 named rows;4160 numerical cases;zero conditional tests/candidates','Completed independent profile arithmetic and saved-JSON/figure audits','Frozen source/protocol binding;only four required dense terminal JSONs','Verified primary-archive dependency']),scientific_success_required_for_packaging=False,
  scope='Only fixed six-action observer data, its exact code/protocol/reviews and four required original1024 terminal metadata records; no dense arrays or counterbalanced data.',
  primary_archive_dependency=dict(required=True,source_commit=PRIMARY,manifest_sha256=archive.sha(primary/'manifest.json'),index_sha256=archive.sha(primary/'index.json'),scope='Shared original checkpoints,128-heading neural states and native input banks from the primary archive.'),
  result_trees=[TREE],additional_terminal_metadata=sorted(DENSE),source_files=sorted(sources),file_count=len(entries),uncompressed_bytes=sum(e['bytes']for e in entries),files=entries,
  packaging_procedure=dict(source=Path(__file__).read_text(),sha256=archive.sha(__file__),writer_source_sha256=archive.sha(ROOT/'scripts/package_nonlinear_ring_outputs.py')))
 dest=ROOT/RELEASE/'archives';index=archive.build(ROOT,manifest,dest)
 for part in index['parts']:
  old=part['path'];new=old.replace('nonlinear-ring-outputs-','native-input-observer-',1);(dest/old).rename(dest/new);part['path']=new
 (dest/'index.json').write_bytes(archive.json_bytes(index));verified=archive.verify_release(dest)
 print(json.dumps(dict(complete=True,files=len(entries),uncompressed_bytes=manifest['uncompressed_bytes'],parts=len(index['parts']),compressed_bytes=sum(x['bytes']for x in index['parts']),manifest_sha256=archive.sha(dest/'manifest.json'),index_sha256=archive.sha(dest/'index.json'),verified=verified),indent=2))
if __name__=='__main__':main()
