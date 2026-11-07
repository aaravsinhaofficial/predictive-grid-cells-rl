from pathlib import Path,PurePosixPath
import datetime,hashlib,json,os,re,subprocess,sys,tarfile,tempfile,time
ROOT=Path('/Users/aaravsinha/predictive-grid-cells-rl').resolve();ARCH=ROOT/'results/native_turn_release/archives'
COMMIT='aa7ee742bd8aeb2e664471226c286fa0ed1495c2';PYTHON='/opt/anaconda3/bin/python'
started=time.monotonic();stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def bsha(b):return hashlib.sha256(b).hexdigest()
def log(s):print(s,flush=True)
def safe(name):
 assert isinstance(name,str) and name and '\\' not in name and not name.startswith('/')
 p=PurePosixPath(name);assert str(p)==name and all(q not in ('','.','..') for q in p.parts)
 return p
body=(ARCH/'manifest.json').read_bytes();manifest=json.loads(body);index=json.loads((ARCH/'index.json').read_text())
assert bsha(body)==index['manifest_sha256'] and manifest['source_commit']==index['source_commit']==COMMIT
assert manifest['completion']['complete'];entries={e['path']:e for e in manifest['files']};assert len(entries)==len(manifest['files'])==manifest['file_count']
base=Path(tempfile.mkdtemp(prefix='native-turn-reproduce-'));work=base/'checkout'
proof=dict(scope='Independent clean-checkout archive verification and saved-data report reproduction only; no training, neural evaluation, environment or scientific rerun.',source_commit=COMMIT,started_utc=stamp(),worktree=str(work),commands=[],procedure_source=Path(__file__).read_text(),procedure_source_sha256=sha(__file__),manifest_sha256=bsha(body),index_sha256=sha(ARCH/'index.json'))
setup=base/'setup.json';setup.write_text(json.dumps({'worktree':str(work),'base':str(base),'source_commit':COMMIT},indent=2))
cmd=['git','worktree','add','--detach',str(work),COMMIT]
p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert p.returncode==0,p.stderr
proof['checkout_creation']=dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
log('Fresh detached checkout: '+str(work))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip()==COMMIT
tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only','-z',COMMIT],cwd=work).decode().split('\0'))
seen=set();parts=[];manifest_member='results/native_turn_release/manifest.json';payload_bytes=0
for part in index['parts']:
 assert Path(part['path']).name==part['path'];archive=ARCH/part['path'];assert archive.stat().st_size==part['bytes']<1500000000 and sha(archive)==part['sha256']
 expected=set(part['members']);assert not(seen&expected) and expected<=set(entries);part_seen=set();embedded=False
 with tarfile.open(archive,'r|gz') as tf:
  for member in tf:
   safe(member.name);assert member.isfile() and not member.issym() and not member.islnk()
   assert (member.uid,member.gid,member.mtime,member.mode)==(0,0,0,0o644)
   if member.name==manifest_member:
    assert not embedded and member.size==len(body);assert tf.extractfile(member).read()==body;embedded=True;continue
   assert member.name in expected and member.name not in part_seen;entry=entries[member.name];assert member.size==entry['bytes']
   target=work/member.name;assert target.resolve().is_relative_to(work.resolve());target.parent.mkdir(parents=True,exist_ok=True)
   assert not target.is_symlink() and all(not parent.is_symlink() for parent in target.parents if parent.is_relative_to(work))
   h=hashlib.sha256();count=0
   with tf.extractfile(member) as inp,target.open('wb') as out:
    for block in iter(lambda:inp.read(1024*1024),b''):h.update(block);count+=len(block);out.write(block)
   assert count==entry['bytes'] and h.hexdigest()==entry['sha256'] and sha(target)==entry['sha256'];payload_bytes+=count;part_seen.add(member.name)
 assert embedded and part_seen==expected;seen|=part_seen;parts.append(dict(path=part['path'],compressed_sha256=part['sha256'],member_count=len(part_seen),all_streamed_and_readback_hashes_match=True))
 log('Verified and extracted '+part['path'])
assert seen==set(entries) and payload_bytes==manifest['uncompressed_bytes']
proof['extraction']=dict(files=len(seen),uncompressed_bytes=payload_bytes,parts=parts,all_regular_safe_normalized_members=True,embedded_manifests_identical=True,links_and_duplicates_rejected=True)
source_checked=[];other_tracked=[]
for name in sorted(set(manifest['source_files'])|(set(entries)&tracked)):
 expected=subprocess.check_output(['git','show',f'{COMMIT}:{name}'],cwd=work)
 assert bsha(expected)==sha(work/name)==entries[name]['sha256'],name
 (source_checked if name in manifest['source_files'] else other_tracked).append(name)
proof['source_integrity_before_reports']=dict(source_files=len(source_checked),other_tracked_archive_files=len(other_tracked),all_match_commit_bytes=True)
log('All archived source and tracked-file bytes match the pinned commit.')
proof['stage']='extraction_complete'
(ROOT/'results/native_turn_release/reproduction_progress.json').write_text(json.dumps(proof,indent=2)+'\n')
log('EXTRACTION COMPLETE '+str(work))
