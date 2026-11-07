"""Small rejected-equivalence data overlay; reuses the verified archive writer."""
from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import package_nonlinear_ring_outputs as archive
RELEASE=Path('results/native_turn_counterbalanced_release');COMMIT='aa7ee742bd8aeb2e664471226c286fa0ed1495c2'
TREES=('results/native_turn_counterbalanced_readout','results/native_turn_counterbalanced_readout_design','results/native_turn_counterbalanced_readout_execution')
PROTOCOL='docs/research/native_turn_counterbalanced_readout_protocol.json'
EXTRA=('scripts/audit_native_turn_counterbalanced_readout.py','scripts/report_native_turn_counterbalanced_readout.py','scripts/review_native_turn_counterbalanced_report.py','tests/test_native_turn_counterbalanced_report.py','docs/research/native_turn_counterbalanced_report_review.json')
archive.TREES=TREES;archive.MANIFEST_NAME=str(RELEASE/'manifest.json')
def read(p):return json.loads(Path(p).read_text())
def check_scope(scientific,report,audit,review):
 assert scientific['complete']is False and report['report_complete']and report['scientific_run_complete']is False
 assert report['old16_input_equivalence']=='rejected'and report['failed_equivalence_comparisons']==12
 assert audit['arithmetic_audit_complete']and audit['scientific_run_complete']is False and audit['old16_input_equivalence']=='rejected'
 assert len(audit['failed_equivalence_rows'])==12 and len(audit['downstream_cell_failure_rows'])==6
 assert(audit['families'],audit['phase_rows'],audit['physics_banks'])==(1536,196608,128)
 assert review['complete']and review['scientific_run_complete']is False and review['old16_input_equivalence']=='rejected'
 assert review['failed_equivalence_comparisons']==12 and review['downstream_cell_failures_retained']==6

def main():
 primary=ROOT/'results/native_turn_release/archives';manifest=read(primary/'manifest.json');index=read(primary/'index.json');proof=read(ROOT/'results/native_turn_release/reproduction_proof.json')
 assert proof['success']and proof['source_commit']==manifest['source_commit']==index['source_commit']==COMMIT
 assert archive.sha(primary/'manifest.json')==index['manifest_sha256']==proof['manifest_sha256']
 assert archive.sha(primary/'index.json')==proof['index_sha256']
 deps={e['path']:e for e in manifest['files']};protocol=read(ROOT/PROTOCOL);b=ROOT/TREES[0];report=read(b/'report/summary.json');audit=read(b/'report/independent_validation.json');review=read(b/'report/final_review.json');scientific=read(b/'summary.json')
 check_scope(scientific,report,audit,review)
 for false in ('complete',):
  try:check_scope(dict(scientific,complete=True),report,audit,review)
  except AssertionError:pass
  else:raise AssertionError('Scientific-complete corruption was accepted')
 assert report['scientific_summary_sha256']==archive.sha(b/'summary.json')==audit['summary_sha256']
 assert report['audit_sha256']==archive.sha(b/'report/independent_validation.json')==review['arithmetic_audit_sha256']
 assert review['report_summary_sha256']==archive.sha(b/'report/summary.json')
 assert report['protocol_sha256']==archive.sha(ROOT/PROTOCOL)
 for path,digest in report['execution_evidence_sha256'].items():archive.prove(ROOT,path,dict(sha256=digest))
 for name in ('execution.json','attempt02/execution.json'):
  e=read(ROOT/TREES[2]/name);assert e['complete']is False and e['exit_code']==1 and not e['timeout']
 names={PROTOCOL,*EXTRA};sources=set(protocol['source_sha256'])|set(EXTRA)
 for path,digest in protocol['source_sha256'].items():archive.prove(ROOT,path,dict(sha256=digest));names.add(path)
 for path,digest in protocol['asset_sha256'].items():assert path in deps and deps[path]['sha256']==digest;archive.prove(ROOT,path,deps[path])
 for tree in TREES:names.update(archive.tree_files(ROOT,tree))
 entries=[]
 tracked=set(filter(None,subprocess.check_output(['git','ls-tree','-r','--name-only','-z',COMMIT],cwd=ROOT).decode().split('\0')))
 for name in sorted(names):
  p=archive.check_name_and_content(ROOT,name);entry=dict(path=name,bytes=p.stat().st_size,sha256=archive.sha(p));entries.append(entry)
  if name in tracked:assert entry['sha256']==archive.commit_sha(ROOT,COMMIT,name)
  if name in sources:assert name in tracked
  if name in deps:assert entry==deps[name]
 result=dict(version='counterbalanced_exploratory_incomplete_equivalence_v1',source_commit=COMMIT,
  title='Exploratory counterbalanced readout — incomplete equivalence',completion=dict(complete=True,issues=[],passed_checks=['Complete descriptive arithmetic audit; scientific run explicitly incomplete','All 1536 families and196608 rows retained','Twelve input-equivalence and six cell-state failures retained','Both failed execution attempts and logs retained','Every shared dependency matches the independently reproduced primary archive']),
  scientific_run_complete=False,old16_input_equivalence='rejected',failed_input_equivalence_rows=12,downstream_cell_failure_rows=6,scientific_success_required_for_packaging=False,
  primary_archive_dependency=dict(source_commit=COMMIT,manifest_sha256=archive.sha(primary/'manifest.json'),index_sha256=archive.sha(primary/'index.json'),required=True,scope='Extract the verified primary128 archives first; this small overlay reuses their frozen checkpoints, input banks and paired results.'),
  result_trees=list(TREES),source_files=sorted(sources),file_count=len(entries),uncompressed_bytes=sum(e['bytes']for e in entries),files=entries,
  packaging_procedure=dict(source=Path(__file__).read_text(),sha256=archive.sha(__file__),writer_source_sha256=archive.sha(ROOT/'scripts/package_nonlinear_ring_outputs.py')))
 out=ROOT/RELEASE/'archives';built=archive.build(ROOT,result,out)
 for part in built['parts']:
  old=part['path'];new=old.replace('nonlinear-ring-outputs-','counterbalanced-incomplete-equivalence-',1);(out/old).rename(out/new);part['path']=new
 (out/'index.json').write_bytes(archive.json_bytes(built));verified=archive.verify_release(out)
 print(json.dumps(dict(complete=True,files=len(entries),uncompressed_bytes=result['uncompressed_bytes'],parts=len(built['parts']),compressed_bytes=sum(p['bytes']for p in built['parts']),manifest_sha256=archive.sha(out/'manifest.json'),index_sha256=archive.sha(out/'index.json'),verified=verified),indent=2))
if __name__=='__main__':main()
