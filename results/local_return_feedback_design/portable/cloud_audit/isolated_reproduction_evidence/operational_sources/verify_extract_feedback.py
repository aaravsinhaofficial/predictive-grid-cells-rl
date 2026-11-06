import hashlib,json,shutil,tarfile,datetime as dt
from pathlib import Path
SOURCE=Path('/Users/aaravsinha/predictive-grid-cells-rl')
ROOT=Path('/tmp/feedback_reproduction_root.txt').read_text().strip(); ROOT=Path(ROOT)
record=json.loads((SOURCE/'docs/research/feedback_acquisition_outputs_archive.json').read_text())
input_manifest=json.loads((SOURCE/'results/local_return_feedback_design/portable/cloud_package/manifest.json').read_text())
assert record['tooling_commit']=='8175f0fd5df07db72ccc4a80d8321938d207bbbe'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
seen={};overlaps=[];verified=[]
def extract(path,asset,rows):
 assert path.stat().st_size==asset['bytes'];assert sha(path)==asset['sha256'],path
 expected={r['path']:r for r in rows};assert len(expected)==len(rows)
 with tarfile.open(path,'r:gz')as archive:
  members=archive.getmembers();assert len(members)==len(rows);assert{m.name for m in members}==set(expected)
  for m in members:
   assert m.isfile()and not m.issym()and not m.islnk();rel=Path(m.name);assert not rel.is_absolute()and'..'not in rel.parts
   item=expected[m.name];assert m.size==item['bytes'];dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if m.name in seen:assert seen[m.name]==item['sha256'],m.name
   if dest.exists():
    prev=sha(dest)
    if prev!=item['sha256']:
     assert m.name not in seen
     assert not(m.name.startswith('scripts/')or m.name.startswith('code/')),m.name
     prior=ROOT/'reproduced/checkout_before_archives'/rel;prior.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(dest,prior)
     overlaps.append(dict(path=m.name,checkout_sha256=prev,archive_sha256=item['sha256']))
   h=hashlib.sha256();n=0
   with archive.extractfile(m)as inp,dest.open('wb')as out:
    for b in iter(lambda:inp.read(1024*1024),b''):out.write(b);h.update(b);n+=len(b)
   assert n==item['bytes']and h.hexdigest()==item['sha256'],m.name
   seen[m.name]=item['sha256']
 verified.append(dict(name=path.name,sha256=asset['sha256'],bytes=asset['bytes'],members=len(rows),all_members_exact=True))
 print(json.dumps(verified[-1]),flush=True)
extract(SOURCE/record['required_input_asset']['path'],record['required_input_asset'],input_manifest['package_manifest'])
for group,asset in record['assets'].items():
 path=Path(asset['path']);embedded=asset['manifest']
 with tarfile.open(path,'r:gz')as archive:
  data=archive.extractfile(embedded['path']).read();assert len(data)==embedded['bytes']and hashlib.sha256(data).hexdigest()==embedded['sha256']
  manifest=json.loads(data);assert manifest['tooling_commit']==record['tooling_commit']and manifest['complete_fixed_study']
 extract(path,asset,manifest['files']+[embedded])
for name,digest in seen.items():assert sha(ROOT/name)==digest,name
validation=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,isolated_root=str(ROOT),tooling_commit=record['tooling_commit'],archive_record_sha256=sha(SOURCE/'docs/research/feedback_acquisition_outputs_archive.json'),archives=verified,unique_archived_members=len(seen),checkout_overlaps_preserved=overlaps,all_extracted_member_sha256=seen)
(ROOT/'reproduced/extraction.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'passed':True,'unique_members':len(seen),'overlaps':len(overlaps),'root':str(ROOT)}),flush=True)
