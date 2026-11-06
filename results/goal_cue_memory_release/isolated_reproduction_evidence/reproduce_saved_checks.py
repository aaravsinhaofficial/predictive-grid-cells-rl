from pathlib import Path
import concurrent.futures,datetime,hashlib,json,os,subprocess,sys

ROOT=Path('/private/tmp/goal-cue-memory-reproduction.7fX05W')
OUT=Path(__file__).resolve().parent
PYTHON='/opt/anaconda3/bin/python'
env=dict(os.environ,OPENBLAS_NUM_THREADS='2')
record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worktree':str(ROOT),
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'no_new_policy_rollouts_or_training':True,'commands':[],'complete':False,'failures':[]}

def execute(name,command):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();log=OUT/(name+'.log')
    with log.open('wb')as output:result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
    item={'name':name,'command':command,'cwd':str(ROOT),'environment_overrides':{'OPENBLAS_NUM_THREADS':'2'},
        'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':result.returncode,
        'log_path':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
    print(json.dumps(item),flush=True);return item

def save():
    (OUT/'saved_checks_execution.json').write_text(json.dumps(record,indent=2)+'\n')

assert json.loads((OUT/'extraction_validation.json').read_text())['complete']
(ROOT/'reproduced/audits').mkdir(parents=True,exist_ok=True)
for phase in ('qualification','original_diagnostic'):
    commands=[]
    for condition in ('gc100','pgc33'):
        name=phase+'_'+condition
        script='scripts/audit_goal_cue_memory.py'if phase=='qualification'else'scripts/audit_goal_cue_memory_original_diagnostic.py'
        command=[PYTHON,script]
        if phase=='qualification':command+=['--phase','qualification']
        command+=['--condition',condition,'--folder',f'results/goal_cue_memory/{phase}/{condition}',
            '--out',f'reproduced/audits/{name}.json','--require-complete']
        commands.append((name,command))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as executor:
        results=list(executor.map(lambda item:execute(*item),commands))
    record['commands'].extend(results)
    record['failures'].extend(item['name']for item in results if item['exit_code'])
    save()
    if record['failures']:raise SystemExit(2)
for name,command in [
    ('donor_equivalence',[PYTHON,'scripts/check_goal_cue_donor_equivalence.py','--out','reproduced/donor_equivalence.json']),
    ('reports',[PYTHON,'-c',"from pathlib import Path; import scripts.report_goal_cue_qualification as qualification; import scripts.report_goal_original_diagnostic as memory; qualification.OUT=Path('reproduced/qualification_report'); qualification.main(); memory.OUT=Path('reproduced/original_diagnostic_report'); memory.main()"]),
]:
    result=execute(name,command);record['commands'].append(result)
    if result['exit_code']:record['failures'].append(name)
    save()
    if record['failures']:raise SystemExit(2)
record['complete']=True;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print('Saved-data checks and report regeneration complete',flush=True)
