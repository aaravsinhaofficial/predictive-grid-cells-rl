from pathlib import Path
import copy,datetime,hashlib,json,re,subprocess

ROOT=Path('/private/tmp/goal-cue-memory-reproduction.7fX05W')
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worktree':str(ROOT),
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'audits':[],'reports':[],'complete':False,'differences':[]}

def differences(a,b,prefix=''):
    if type(a)is not type(b):return[{'path':prefix,'original':a,'reproduced':b}]
    if isinstance(a,dict):
        items=[]
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:items.append({'path':prefix+'/'+key,'original':a.get(key,'MISSING'),'reproduced':b.get(key,'MISSING')})
            else:items+=differences(a[key],b[key],prefix+'/'+key)
        return items
    if isinstance(a,list):
        if len(a)!=len(b):return[{'path':prefix,'original_length':len(a),'reproduced_length':len(b)}]
        return[item for i,(x,y)in enumerate(zip(a,b))for item in differences(x,y,prefix+f'/{i}')]
    return[]if a==b else[{'path':prefix,'original':a,'reproduced':b}]

assert read(OUT/'saved_checks_execution.json')['complete']
for phase in ('qualification','original_diagnostic'):
    for condition in ('gc100','pgc33'):
        original=ROOT/f'results/goal_cue_memory/{phase}/audits/{condition}.json'
        fresh=ROOT/f'reproduced/audits/{phase}_{condition}.json'
        a,b=read(original),read(fresh)
        assert a['passed']and a['complete']and b['passed']and b['complete']
        excluded={key:{'original':a.pop(key),'reproduced':b.pop(key)}for key in ('created_utc','folder')}
        delta=differences(a,b)
        result['audits'].append({'phase':phase,'condition':condition,'original_sha256':sha(original),'reproduced_sha256':sha(fresh),
            'exact_after_declared_execution_metadata_exclusion':not delta,'excluded_fields':excluded,'differences':delta})
        result['differences']+=delta
original=ROOT/'results/goal_cue_memory/original_diagnostic/audits/donor_equivalence.json'
fresh=ROOT/'reproduced/donor_equivalence.json';a,b=read(original),read(fresh)
excluded={'created_utc':{'original':a.pop('created_utc'),'reproduced':b.pop('created_utc')}}
delta=differences(a,b);result['differences']+=delta
result['donor_equivalence']={'original_sha256':sha(original),'reproduced_sha256':sha(fresh),'exact_after_timestamp_exclusion':not delta,'excluded_fields':excluded,'differences':delta}
for phase in ('qualification','original_diagnostic'):
    old=ROOT/f'results/goal_cue_memory/{phase}/report';new=ROOT/f'reproduced/{phase}_report'
    names={p.relative_to(old).as_posix()for p in old.rglob('*')if p.is_file()}
    assert names=={p.relative_to(new).as_posix()for p in new.rglob('*')if p.is_file()}
    for name in sorted(names):
        left,right=old/name,new/name;a,b=left.read_bytes(),right.read_bytes()
        item={'phase':phase,'file':name,'original_sha256':sha(left),'reproduced_sha256':sha(right),'byte_identical':a==b}
        if left.suffix=='.pdf':
            pattern=rb'/CreationDate\s*\([^)]*\)';ta,tb=re.findall(pattern,a),re.findall(pattern,b)
            assert len(ta)==len(tb)==1
            item['only_ignored_pdf_field']='CreationDate';item['original_creation_date']=ta[0].decode('ascii');item['reproduced_creation_date']=tb[0].decode('ascii')
            item['equal_ignoring_creation_date']=re.sub(pattern,b'/CreationDate (IGNORED)',a)==re.sub(pattern,b'/CreationDate (IGNORED)',b)
            equal=item['equal_ignoring_creation_date']
        else:equal=item['byte_identical']
        if not equal:
            item['differences']=differences(json.loads(a),json.loads(b))if left.suffix=='.json'else['File bytes differ beyond permitted PDF CreationDate']
            result['differences'].append({'phase':phase,'file':name,'detail':item['differences']})
        result['reports'].append(item)
manifest=read(ROOT/'results/goal_cue_memory_release/manifest.json')
for row in manifest['files']:
    p=ROOT/row['path'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
result['all_archive_inputs_unchanged_after_reproduction']=True
result['git_tracked_diff']=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert not result['git_tracked_diff']
result['complete']=not result['differences']
(OUT/'comparison_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'complete':result['complete'],'audits':len(result['audits']),'report_files':len(result['reports']),'differences':result['differences']}),flush=True)
if not result['complete']:raise SystemExit(2)
