from pathlib import Path
import hashlib,json,re,copy,datetime as dt,platform,sys,torch,numpy,matplotlib,shutil
SOURCE=Path('/Users/aaravsinha/predictive-grid-cells-rl');ROOT=Path(Path('/tmp/feedback_reproduction_root.txt').read_text().strip());OUT=ROOT/'reproduced'
OLD=ROOT/'results/local_return_feedback_design/portable/cloud_audit';REPORT=ROOT/'results/local_return_feedback_acquisition/portable/report'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def identical_except(old,new,allowed):
 old=copy.deepcopy(old);new=copy.deepcopy(new);changes={}
 for key in allowed:
  if old.get(key)!=new.get(key):changes[key]=dict(archived=old.get(key),reproduced=new.get(key))
  old.pop(key,None);new.pop(key,None)
 assert old==new
 return changes
extraction=read(OUT/'extraction.json');routing=read(ROOT/'docs/research/local_return_feedback_portable_launch.json')
audits=[]
for w in routing['workers']:
 key=w['condition']+'_'+w['actor_mode'];old=OLD/(key+'.json');new=OUT/'audits'/(key+'.json')
 a=read(old);b=read(new);assert b['passed']and b['complete']and not b['failures']and not b['pending']
 changes=identical_except(a,b,('created_utc','folder'))
 audits.append(dict(condition=w['condition'],actor_mode=w['actor_mode'],archived_sha256=sha(old),reproduced_sha256=sha(new),all_checks_and_artifact_hashes_identical=True,metadata_differences=changes))
reference=read(OUT/'reference_audit.json');assert reference['passed']and reference['complete']
ref_changes=identical_except(read(ROOT/'results/local_return_feedback_design/reference_audit.json'),reference,('folder','protocol_sha256'))
streams_old=read(OLD/'paired_final_training_validation.json');streams_new=read(OUT/'streams/paired_final_training_validation.json')
assert streams_old['raw']['sha256']==streams_new['raw']['sha256']and streams_old['raw']['bytes']==streams_new['raw']['bytes']
assert sha(ROOT/streams_old['raw']['path'])==sha(ROOT/streams_new['raw']['path'])
sold=copy.deepcopy(streams_old);snew=copy.deepcopy(streams_new);old_raw_path=sold['raw'].pop('path');new_raw_path=snew['raw'].pop('path');stream_differences=identical_except(sold,snew,('utc',));stream_differences['raw/path']=dict(archived=old_raw_path,reproduced=new_raw_path)
other={}
for oldname,newname in [('baseline_actor_isolation.json','actor_isolation.json'),('completed_finals_independent_review.json','final_review.json')]:
 a=read(OLD/oldname);b=read(OUT/newname);assert b['complete'];changes=identical_except(a,b,('utc',))
 other[newname]=dict(archived_sha256=sha(OLD/oldname),reproduced_sha256=sha(OUT/newname),all_scientific_and_input_checks_identical=True,metadata_differences=changes)
a=read(REPORT/'summary.json');b=read(OUT/'report/summary.json');original_prefix=Path(a['arms'][0]['folder']).parents[3]
assert (original_prefix/'results/ring_cloud_artifacts'/a['arms'][0]['worker']/'experiment').as_posix()==a['arms'][0]['folder']
oldinputs=a.pop('input_sha256');newinputs=b.pop('input_sha256');assert len(oldinputs)==len(newinputs)
verified_inputs=[]
for k,digest in oldinputs.items():
 relative=Path(k).relative_to(original_prefix)
 if relative.parent==Path('results/local_return_feedback_design/portable/cloud_audit')and relative.name in {x['condition']+'_'+x['actor_mode']+'.json' for x in routing['workers']}:
  mapped=ROOT/'reproduced/audits'/relative.name;assert newinputs[str(mapped)]==sha(mapped)
 else:
  mapped=ROOT/relative;assert newinputs[str(mapped)]==digest==sha(mapped)
 verified_inputs.append(str(relative))
folders=[]
for x,y in zip(a['arms'],b['arms']):
 assert x['condition']==y['condition']and x['actor_mode']==y['actor_mode']
 rel=Path(x['folder']).relative_to(original_prefix);assert Path(y['folder'])==ROOT/rel
 folders.append(dict(condition=x['condition'],actor_mode=x['actor_mode'],archived=x.pop('folder'),reproduced=y.pop('folder')))
assert a==b,'Scientific summary differs'
files={}
for name in ['all_arms.csv','README.md','feedback_return.png','feedback_learning.png']:
 p=REPORT/name;q=OUT/'report'/name;assert p.read_bytes()==q.read_bytes(),name
 files[name]=dict(byte_identical=True,sha256=sha(p),bytes=p.stat().st_size)
for name in ['feedback_return.pdf','feedback_learning.pdf']:
 aa=(REPORT/name).read_bytes();bb=(OUT/'report'/name).read_bytes();pattern=rb'/CreationDate \(D:[^)]*\)'
 before=re.findall(pattern,aa);after=re.findall(pattern,bb);assert len(before)==len(after)==1
 assert re.sub(pattern,b'/CreationDate (REMOVED)',aa)==re.sub(pattern,b'/CreationDate (REMOVED)',bb)
 files[name]=dict(byte_identical=aa==bb,identical_except_pdf_creation_date=True,archived_sha256=sha(REPORT/name),reproduced_sha256=sha(OUT/'report'/name),archived_date=before[0].decode(),reproduced_date=after[0].decode())
for name,digest in extraction['all_extracted_member_sha256'].items():assert sha(ROOT/name)==digest,name
commands=[read(p) for p in sorted((OUT/'logs').glob('*.command.json'))]
assert len(commands)==10 and sum(x['returncode']==0 for x in commands)==9
record=dict(version='feedback_acquisition_isolated_extraction_reproduction_v1',utc=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,complete=True,
 tooling_commit='8175f0fd5df07db72ccc4a80d8321938d207bbbe',training_source_commit='f1e5e1725c4449ee6c8bd39cac022b9e94b0139f',protocol_sha256='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19',
 isolated_root=str(ROOT),isolation='Fresh git archive of exact tooling commit plus four hash-authenticated assets; no workspace model fallback, network, cloud, new policy rollout or optimization.',
 environment=dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,matplotlib=matplotlib.__version__,platform=platform.platform()),
 archives=extraction['archives'],archive_record_sha256_at_start=extraction['archive_record_sha256'],unique_archived_files=extraction['unique_archived_members'],all_archived_files_unchanged_after_reproduction=True,checkout_overlaps=extraction['checkout_overlaps_preserved'],
 pure_protocol_check=read(OUT/'pure_protocol_check.json'),strict_model_audits=audits,reference_audit=dict(passed=True,complete=True,all_saved_physics_and_data_checks_identical=True,metadata_differences=ref_changes),
 streams=dict(passed=True,all_named_generator_states_identical=True,raw_pt_byte_identical=True,raw_sha256=streams_new['raw']['sha256'],metadata_differences=stream_differences),other_checks=other,
 report=dict(all_scientific_fields_exact=True,all_input_bindings_verified=len(verified_inputs),only_summary_differences='Absolute experiment/input paths and SHA256 of re-audits with new date/path metadata.',folder_differences=folders,files=files,arms=[dict(condition=r['condition'],actor_mode=r['actor_mode'],baseline_successes=r['baseline']['successes'],final_successes=r['final']['successes'],trials=256,final_qualified=r['final_qualified'])for r in b['arms']]),
 commands=commands,guide_erratum=dict(original_guide_preserved=True,failed_documented_command='python scripts/compare_feedback_training_streams.py --out reproduced/streams',failure='Relative --out writes raw PT then fails Path.relative_to(ROOT).',corrected_command='python scripts/compare_feedback_training_streams.py --out "$PWD/reproduced/streams"',unchanged_analysis_source=True,failed_attempt_retained='reproduced/streams_failed_relative_out',failed_log='reproduced/logs/streams.log'),
 auxiliary_probe_note='A separate optional pure-protocol helper first imported Torch before adding the isolated root to sys.path, causing stdlib code to shadow repo code; it exited at import. Corrected private helper sets sys.path first and the exact protocol/bank check passes. No study source or asset changed.',
 scope='Read-only saved-data reproduction. Strict audits recompute saved physics/input/readout checks; no learned-policy environment rollout, CNN/LSTM trajectory replay, training, or cloud resource was run.')
(OUT/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(passed=True,audits=len(audits),unique_archived_files=record['unique_archived_files'],png_csv_readme_byte_identical=True,pdf_only_creation_date=True,arms=record['report']['arms'])),flush=True)
