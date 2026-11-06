from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tarfile,traceback

ROOT=Path('/private/tmp/goal-cue-memory-reproduction.7fX05W')
ASSETS=Path('/Users/aaravsinha/predictive-grid-cells-rl/results/goal_cue_memory_release/archives')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.package_goal_cue_memory_outputs import sha,verify_archive,MANIFEST_NAME

record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worktree':str(ROOT),
    'archive_directory':str(ASSETS),'source_commit':'569dbdd7a6295f168e7c3e824688d09366ea5fd4',
    'scope':'Only committed checkout files and the two release archive parts/index/manifest are used. No live root raw data is copied.',
    'parts':[],'complete':False,'failures':[]}
try:
    body=(ASSETS/'manifest.json').read_bytes();manifest=json.loads(body);index=json.loads((ASSETS/'index.json').read_text())
    assert hashlib.sha256(body).hexdigest()==index['manifest_sha256']
    assert index['source_commit']==manifest['source_commit']==record['source_commit']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==manifest['source_commit']
    rows={r['path']:r for r in manifest['files']};assert len(rows)==len(manifest['files'])
    seen=set();created=existing=0
    record['index_sha256']=sha(ASSETS/'index.json');record['manifest_sha256']=sha(ASSETS/'manifest.json')
    for part in index['parts']:
        path=ASSETS/part['path'];names=part['members']
        assert path.stat().st_size==part['bytes']and sha(path)==part['sha256']
        assert len(names)==len(set(names))and not seen.intersection(names)
        verify_archive(path,[rows[name]for name in names],body)
        print('SHA and every member verified:',path.name,flush=True)
        with tarfile.open(path,'r:gz')as archive:
            for member in archive:
                target=ROOT/member.name;assert target.resolve().is_relative_to(ROOT)
                check=ROOT
                for component in Path(member.name).parts:
                    check/=component;assert not check.is_symlink()
                expected=index['manifest_sha256']if member.name==MANIFEST_NAME else rows[member.name]['sha256']
                if target.exists():
                    assert target.is_file()and sha(target)==expected;existing+=1
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member)as source,target.open('xb')as destination:shutil.copyfileobj(source,destination)
                    assert sha(target)==expected;created+=1
        seen.update(names);record['parts'].append({'path':path.name,'sha256':part['sha256'],'bytes':part['bytes'],'members':len(names),'verified_and_safely_extracted':True})
        print('Safely extracted:',path.name,flush=True)
    assert seen==set(rows)
    for name,row in rows.items():assert(ROOT/name).stat().st_size==row['bytes']and sha(ROOT/name)==row['sha256']
    record.update(complete=True,file_count=len(rows),uncompressed_bytes=sum(r['bytes']for r in rows.values()),created_members=created,existing_members_verified=existing)
except BaseException as error:
    record['failures'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
finally:
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT/'extraction_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if not record['complete']:raise SystemExit(2)
