"""Private guardian controls. No cloud commands, workers or real supervisors."""
import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import guard_simple_memory_supervisor as guard

NAME='ring-20261105-simplemem-gc100-native-s1'
PREFIX=f'gs://{guard.BUCKET}/{guard.STUDY}/{NAME}'
METRIC=PREFIX+'/outputs/seed1/metrics_000.jsonl'
TIMEOUT=f"TimeoutExpired: Command '{['gcloud','storage','cat','--range=-65536',METRIC]}' timed out after 60 seconds"
NOW=guard.stamp(guard.ORIGINAL_START)+5*3600


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=1)+'\n')


def local_fixture(tmp_path):
    supervision=tmp_path/'supervision';out=tmp_path/'guardian';out.mkdir()
    original=tmp_path/'original.json';write(original,dict(started_utc=guard.ORIGINAL_START,max_hours=72))
    process=dict(pid=10,started_utc=guard.ORIGINAL_START,max_hours=72,plan='plan.json',protocol='protocol.json')
    status=dict(binding={'private':'fixture'},jobs={NAME:{'instance_id':'123','technical':{'frames':100}}},halted=True,error=TIMEOUT)
    write(supervision/'process.json',process);write(supervision/'status.json',status)
    record=dict(name=NAME,instance_id='123',gcs_prefix=PREFIX,zone='us-central1-a',machine='n1-standard-8',
                package_manifest_sha256='package',provider_deadline_utc='2026-11-08T14:32:09+00:00',reserved_usd=32.1875)
    ledger=dict(budget_usd=600,reserved_rate_usd_per_hour=1.25,ancillary_reserve_usd=25,runs=[record])
    write(tmp_path/guard.LEDGER,ledger)
    job=dict(name=NAME,trainer_output_dir='results/ring_simple_memory/gc100/native/seed1',
             plan_argv=['python','scripts/ring_cloud.py','plan','--hours','25.5'])
    plan=dict(jobs=[job],design={'first_jobs':[NAME]},hardware={'minimum_training_fps':11200})
    local=dict(training=plan,ledger=ledger,budget={'combined_upper_usd':599.199},binding={'protocol_sha256':'protocol'},
               ledger_sha256=guard.sha(tmp_path/guard.LEDGER))
    history=dict(attempts=[],original_process_path=str(original),original_process_sha256=guard.sha(original),
        guardian_source_sha256=guard.sha(guard.__file__),deadline_utc=dt.datetime.fromtimestamp(NOW+6*3600,dt.timezone.utc).isoformat())
    return supervision,out,process,status,local,history


def test_only_two_bound_read_failure_classes_are_allowed():
    targets={METRIC:(NAME,'metrics'),PREFIX+'/DONE.json':(NAME,'done')}
    assert guard.recognize_failure(TIMEOUT,targets)['target']==METRIC
    dns=("RuntimeError: Cloud read failed: ERROR: gcloud crashed (ConnectionError): HTTPSConnectionPool(host='storage.googleapis.com', port=443): "
         "Max retries exceeded with url: /storage/v1/b/"+guard.BUCKET+"/o/"+guard.STUDY+'%2F'+NAME+"%2FDONE.json?alt=json "
         '(Caused by NameResolutionError("Failed to resolve \'storage.googleapis.com\'"))')
    assert guard.recognize_failure(dns,targets)['kind']=='storage_dns_failure'
    for error in ('AssertionError: technical throughput gate failed','RuntimeError: terminal validation failed',
                  'TimeoutError: Supervisor time limit reached',TIMEOUT.replace("'cat'","'cp'"),
                  TIMEOUT.replace(NAME,'unowned'),TIMEOUT.replace('60 seconds','180 seconds'),dns.replace('NameResolutionError','PermissionError')):
        with pytest.raises((RuntimeError,ValueError,SyntaxError)):guard.recognize_failure(error,targets)


def test_absolute_deadline_cannot_restart_or_extend(tmp_path):
    _,_,p,_,_,_=local_fixture(tmp_path)
    original=dict(started_utc=guard.ORIGINAL_START,max_hours=72)
    deadline=guard.original_deadline(p,original,NOW)
    assert deadline==guard.stamp(guard.ORIGINAL_START)+72*3600
    cmd=guard.restart_argv(p,tmp_path,deadline,NOW)
    assert float(cmd[-1])*3600 < deadline-NOW
    p.update(original_started_utc=guard.ORIGINAL_START,started_utc=dt.datetime.fromtimestamp(NOW,dt.timezone.utc).isoformat(),max_hours=72)
    with pytest.raises(RuntimeError,match='extended'):guard.original_deadline(p,original,NOW)
    p['max_hours']=1
    with pytest.raises(RuntimeError,match='expired'):guard.original_deadline(p,original,deadline)


def test_whole_budget_includes_unlaunched_training_and_analysis():
    train={'jobs':[{'name':'training','plan_argv':['--hours','25.5']}]}
    follow={'jobs':[{'name':'analysis'}],'per_worker_hours':1,'cumulative_ceiling_usd':600}
    ledger=dict(runs=[],ancillary_reserve_usd=25,budget_usd=600,reserved_rate_usd_per_hour=1.25)
    b=guard.whole_budget(train,follow,ledger)
    assert b['combined_upper_usd']==25+32.1875+1.5625
    ledger['ancillary_reserve_usd']=599
    with pytest.raises(RuntimeError,match='budget'):guard.whole_budget(train,follow,ledger)


def test_locks_and_reused_pid_fail_closed(tmp_path,monkeypatch):
    path=tmp_path/'lock'
    with guard.exclusive(path):
        with pytest.raises(BlockingIOError):
            with guard.exclusive(path):pass
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=f'{guard.os.getuid()} unrelated.py'))
    with pytest.raises(RuntimeError,match='reused'):guard.process_state(10)
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    assert not guard.process_state(10)['alive']


def test_recovery_preserves_exact_evidence_and_accounts_before_spawn(tmp_path,monkeypatch):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    raw={n:(supervision/n).read_bytes() for n in ('status.json','process.json')}
    spawned=[]
    def spawn(argv,**kwargs):
        on_disk=guard.read(out/'guardian.json')
        if not spawned:assert on_disk['attempts'][0]['status']=='spawn_intent'
        spawned.append(argv)
        return SimpleNamespace(pid=100+len(spawned),poll=lambda:None)
    monkeypatch.setattr(guard.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(guard,'utc',lambda:dt.datetime.fromtimestamp(NOW,dt.timezone.utc).isoformat())
    result=guard.recover(supervision,out,history,root=tmp_path,now=lambda:NOW,
        inspect_pid=lambda pid:{'alive':pid==101},local_validator=lambda *a:local,
        cloud_verifier=lambda *a:{'private':'read-only fixture'},spawn=spawn)
    assert result['status']=='recovered' and len(spawned)==2
    assert spawned[0][2]==guard.SUPERVISOR and spawned[1][0]=='caffeinate'
    assert float(spawned[0][-1])<67
    for n,data in raw.items():assert (out/'recovery01'/n).read_bytes()==data
    current=guard.read(supervision/'status.json')
    assert 'halted' not in current and current['jobs']==status['jobs']
    assert guard.read(supervision/'process.json')['pid']==101


def test_changed_status_during_probe_never_clears_or_spawns(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    def probe(*args):
        newer=dict(status,error='AssertionError: new scientific failure');write(supervision/'status.json',newer)
        return {}
    with pytest.raises(RuntimeError,match='changed during'):
        guard.recover(supervision,out,history,root=tmp_path,now=lambda:NOW,inspect_pid=lambda pid:{'alive':False},
            local_validator=lambda *a:local,cloud_verifier=probe,spawn=lambda *a,**k:pytest.fail('No spawn'))
    assert guard.read(supervision/'status.json')['error']=='AssertionError: new scientific failure'
    assert guard.read(out/'guardian.json')['attempts'][0]['status']=='preflight'


def test_recovery_limit_and_spawn_failure_do_not_silently_retry(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    history['attempts']=[{}]*3
    with pytest.raises(RuntimeError,match='limit'):guard.recover(supervision,out,history,root=tmp_path)
    history['attempts']=[];raw=(supervision/'status.json').read_bytes()
    def broken(*args,**kwargs):raise OSError('private spawn failure')
    with pytest.raises(OSError):
        guard.recover(supervision,out,history,root=tmp_path,now=lambda:NOW,inspect_pid=lambda pid:{'alive':False},
            local_validator=lambda *a:local,cloud_verifier=lambda *a:{},spawn=broken)
    assert (supervision/'status.json').read_bytes()==raw
    assert guard.read(out/'guardian.json')['attempts'][0]['status']=='spawn_failed'


def test_worker_missing_id_terminal_failure_and_read404_rejected(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    failure={'job':NAME}
    bad=copy.deepcopy(local);bad['ledger']['runs'][0].pop('instance_id')
    with pytest.raises(RuntimeError,match='ID'):
        guard.verify_cloud(bad,status,failure,lambda *a,**k:pytest.fail('No cloud read before identity'),NOW)
    vm=dict(id='123',name=NAME,labels={'study':guard.STUDY},status='RUNNING',zone='zones/us-central1-a',machineType='types/n1-standard-8')
    def failed(argv,**kw):return json.dumps(vm) if 'describe' in argv else json.dumps({'exit_code':1,'package_sha256':'package'})
    with pytest.raises(RuntimeError,match='terminal failure'):guard.verify_cloud(local,status,failure,failed,NOW)
    def absent(argv,**kw):return None
    with pytest.raises(RuntimeError,match='missing/not RUNNING'):guard.verify_cloud(local,status,failure,absent,NOW)


def test_connection_retry_is_bounded_read_only_and_waits_at_most60(tmp_path,monkeypatch):
    sleeps=[];calls=[]
    monkeypatch.setattr(guard.time,'sleep',sleeps.append)
    def failing(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1,stdout='',stderr='NameResolutionError private fixture')
    monkeypatch.setattr(guard.subprocess,'run',failing)
    client=guard.CloudReads(tmp_path,guard.time.time()+600)
    with pytest.raises(RuntimeError,match='exhausted'):client.call(['gcloud','storage','cat',METRIC])
    assert len(calls)==6 and sleeps==list(guard.BACKOFF) and max(sleeps)==60
    with pytest.raises(RuntimeError,match='read-only'):client.call(['gcloud','compute','instances','delete',NAME])
    assert len(calls)==6


def test_fresh_metrics_use_current_unchanged_viability_not_a_failure_whitelist(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    status['jobs'][NAME]['technical_gate_passed']=True
    vm=dict(id='123',name=NAME,labels={'study':guard.STUDY},status='RUNNING',zone='zones/us-central1-a',machineType='types/n1-standard-8')
    metric=dict(frames=300_000_000,updates=23437,fps=11100,loss=1.,grad_norm=2.,replay_logit_max_error=0.,replay_value_max_error=0.)
    def cloud(argv,**kwargs):
        if argv[:4]==['gcloud','compute','instances','describe']:return json.dumps(vm)
        if argv[:4]==['gcloud','storage','objects','describe']:
            return json.dumps(dict(bucket=guard.BUCKET,name=METRIC.split('/'+guard.BUCKET+'/',1)[1],generation='123456',
                update_time=dt.datetime.fromtimestamp(NOW,dt.timezone.utc).isoformat()))
        if argv[-1].endswith('DONE.json'):return None
        return json.dumps(metric)+'\n'
    result=guard.verify_cloud(local,status,{'job':NAME},cloud,NOW)
    assert result['reconnected_read']['viability']['margin_seconds']>0
    status['jobs'][NAME]['technical_gate_passed']=False
    with pytest.raises(AssertionError,match='initial technical'):guard.verify_cloud(local,status,{'job':NAME},cloud,NOW)
    status['jobs'][NAME]['technical_gate_passed']=True;metric['fps']=100
    with pytest.raises(AssertionError,match='original provider deadline'):guard.verify_cloud(local,status,{'job':NAME},cloud,NOW)


def test_ledger_change_during_probe_fails_before_status_clear(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    old=(supervision/'status.json').read_bytes()
    def probe(*args):
        ledger=guard.read(tmp_path/guard.LEDGER);ledger['ancillary_reserve_usd']+=1;write(tmp_path/guard.LEDGER,ledger)
        return {}
    with pytest.raises(RuntimeError,match='Ledger changed'):
        guard.recover(supervision,out,history,root=tmp_path,now=lambda:NOW,inspect_pid=lambda pid:{'alive':False},
            local_validator=lambda *a:local,cloud_verifier=probe,spawn=lambda *a,**k:pytest.fail('No spawn'))
    assert (supervision/'status.json').read_bytes()==old


def test_alternate_guardian_directory_cannot_reset_durable_limit(tmp_path):
    with pytest.raises(RuntimeError,match='canonical guardian'):
        guard.main(['--out',str(tmp_path/'second_guardian')])
    assert not (tmp_path/'second_guardian').exists()


def test_stale_metadata_and_postprobe_time_are_fail_closed(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    status['jobs'][NAME]['technical_gate_passed']=True
    vm=dict(id='123',name=NAME,labels={'study':guard.STUDY},status='RUNNING',zone='zones/us-central1-a',machineType='types/n1-standard-8')
    metric=dict(frames=300_000_000,updates=23437,fps=11100,loss=1.,grad_norm=2.,replay_logit_max_error=0.,replay_value_max_error=0.)
    clock=[NOW];updated=[NOW-901];reads=[]
    def cloud(argv,**kwargs):
        reads.append(argv)
        if argv[:4]==['gcloud','compute','instances','describe']:return json.dumps(vm)
        if argv[:4]==['gcloud','storage','objects','describe']:
            return json.dumps(dict(bucket=guard.BUCKET,name=METRIC.split('/'+guard.BUCKET+'/',1)[1],generation='4567',
                                   update_time=dt.datetime.fromtimestamp(updated[0],dt.timezone.utc).isoformat()))
        if argv[-1].endswith('DONE.json'):return None
        assert argv[-1]==METRIC+'#4567'
        clock[0]+=800
        return json.dumps(metric)+'\n'
    with pytest.raises(RuntimeError,match='stale'):guard.verify_cloud(local,status,{'job':NAME},cloud,lambda:clock[0])
    assert not any(a[-1]==METRIC+'#4567' for a in reads)
    updated[0]=NOW
    available=(1_000_000_000-metric['frames'])/metric['fps']+600+100
    local['ledger']['runs'][0]['provider_deadline_utc']=dt.datetime.fromtimestamp(NOW+available,dt.timezone.utc).isoformat()
    with pytest.raises(AssertionError,match='original provider deadline'):
        guard.verify_cloud(local,status,{'job':NAME},cloud,lambda:clock[0])


def test_original_anchor_mutation_during_probes_prevents_clear(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    old=(supervision/'status.json').read_bytes()
    def probe(*args):
        Path(history['original_process_path']).write_text('{"changed":true}')
        return {}
    with pytest.raises(RuntimeError,match='anchor changed'):
        guard.recover(supervision,out,history,root=tmp_path,now=lambda:NOW,inspect_pid=lambda pid:{'alive':False},
            local_validator=lambda *a:local,cloud_verifier=probe,spawn=lambda *a,**k:pytest.fail('No spawn'))
    assert (supervision/'status.json').read_bytes()==old


def test_runtime_guardian_source_binding_checked_before_any_probe(tmp_path):
    supervision,out,p,status,local,history=local_fixture(tmp_path)
    history['guardian_source_sha256']='changed'
    with pytest.raises(RuntimeError,match='source changed'):
        guard.recover(supervision,out,history,root=tmp_path,now=lambda:NOW,
                      local_validator=lambda *a:pytest.fail('No probe'))
    assert not history['attempts']
