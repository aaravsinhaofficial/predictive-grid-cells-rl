"""Synthetic optimizer/resume/guard tests; no trained-policy or study outcome."""
import copy
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
import pytest
import torch
import goal_cue_memory_run as run
import goal_cue_memory as task
from test_goal_cue_memory import fixture,equal


def state_fixture(seed=8801):
    net,cfg,source,_=fixture(seed,angles=2)
    cfg=dict(cfg,n_envs=2,img_noise=0.)
    opt=run.initializer.optimizer(net,cfg);streams=task.TrialStreams(seed+1)
    counters=run.fresh_counters();counters['evaluation_records']['baseline']={'synthetic':True}
    return net,cfg,source,opt,streams,counters


def one_update(net,cfg,source,opt,streams,counters,cap=0):
    plan=streams.draw_training(cfg['n_envs'],cfg,delay_cap=cap)
    metrics,rows=run.update(net,cfg,opt,source,plan)
    run.advance_counters(counters,metrics,cfg)
    return plan,metrics,rows


def test_complete_synthetic_update_and_serialized_resume_are_exact():
    net,cfg,source,opt,streams,counters=state_fixture()
    first=one_update(net,cfg,source,opt,streams,counters)
    assert first[1]['replay_error']==first[1]['value_replay_error']==0
    saved=run.snapshot(net,opt,streams,cfg,{'synthetic':True},counters,{'fixture':True})
    buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu',weights_only=True)
    expected=one_update(net,cfg,source,opt,streams,counters)
    final=run.snapshot(net,opt,streams,cfg,{'synthetic':True},counters,{'fixture':True})
    other,_,other_source,other_opt,other_streams,_=state_fixture()
    restored=run.restore(other,other_opt,other_streams,saved,cfg,{'fixture':True})
    actual=one_update(other,cfg,other_source,other_opt,other_streams,restored)
    actual_final=run.snapshot(other,other_opt,other_streams,cfg,{'synthetic':True},restored,{'fixture':True})
    assert equal(expected,actual) and equal(final,actual_final)
    assert final['streams']['delay_draws']==2 and final['counters']['delay_histograms']['block1'][0]==2
    corrupt=copy.deepcopy(saved);corrupt['counters']['delay_histograms']['block1'][0]+=1
    with pytest.raises(AssertionError):run.restore(other,other_opt,other_streams,corrupt,cfg,{'fixture':True})


def test_pending_evaluation_cannot_skip_a_fixed_boundary():
    counters=run.fresh_counters();assert run.pending_evaluation(counters)=='baseline'
    counters['evaluation_records']['baseline']={};assert run.pending_evaluation(counters) is None
    counters['updates']=1024;assert run.pending_evaluation(counters)=='block1'
    counters['updates']=1025
    with pytest.raises(AssertionError,match='Missing prescribed'):run.pending_evaluation(counters)
    counters['updates']=3072;counters['evaluation_records'].update(block1={},block2={})
    assert run.pending_evaluation(counters)=='block3'
    counters['evaluation_records']['block3']={};assert run.pending_evaluation(counters) is None


def make_parent_evidence(parent,counters):
    parent.mkdir();(parent/'metrics.jsonl').write_text('{"update":1}\n{"update":9999,"rolled_back":true}\n')
    for name in list(counters['evaluation_records']):
        before=parent/f'{name}_before_evaluation.pt';before.write_bytes(b'synthetic before')
        (parent/f'{name}.pt').write_bytes(b'synthetic after')
        (parent/f'{name}_evaluation_integrity.json').write_text('{}')
        folder=parent/f'{name}_evaluation';folder.mkdir();(folder/'summary.json').write_text('{"synthetic":true}')
        counters['evaluation_records'][name]=dict(checkpoint_before_sha256=run.sha(before),
            summary_sha256=run.sha(folder/'summary.json'))
    return parent


def test_resume_lineage_keeps_failed_tapes_and_rolled_back_metrics(tmp_path):
    counters=run.fresh_counters();counters['updates']=1024;counters['evaluation_records']['baseline']={}
    parent=make_parent_evidence(tmp_path/'parent',counters)
    checkpoint=parent/'latest.pt';checkpoint.write_bytes(b'exact supplied checkpoint')
    (parent/'failed_plan.pt').write_bytes(b'failed plan bytes')
    (parent/'failed_predraw_streams.pt').write_bytes(b'exact predraw bytes')
    pending=parent/'block1_evaluation';pending.mkdir();(pending/'partial.json').write_text('{"partial":true}')
    out=tmp_path/'new';out.mkdir();run.inherit_resume_artifacts(checkpoint,out,counters)
    evidence=out/'resume_parent_evidence'
    for name in ('failed_plan.pt','failed_predraw_streams.pt','metrics.jsonl'):
        assert (evidence/name).read_bytes()==(parent/name).read_bytes()
    assert (evidence/'checkpoint.pt').read_bytes()==checkpoint.read_bytes()
    assert (out/'parent_incomplete_block1_evaluation/partial.json').read_bytes()==(pending/'partial.json').read_bytes()
    assert not(out/'latest.pt').exists(),'Mutable intermediate latest must not be frozen in lineage'


def test_final_boundary_resume_evaluates_once_without_extra_learning(tmp_path,monkeypatch):
    # One synthetic update represents a final boundary; no actual study budget
    # or protocol is generated, and evaluation is a clearly marked stub.
    monkeypatch.setattr(run,'UPDATES_PER_BLOCK',1);monkeypatch.setattr(run,'TOTAL_UPDATES',1)
    monkeypatch.setattr(run,'TRAIN_SEED',8812)
    monkeypatch.setattr(run,'ENDPOINTS',{'baseline':0,'block3':1})
    net,cfg,source,opt,streams,counters=state_fixture(8811)
    one_update(net,cfg,source,opt,streams,counters)
    parent=make_parent_evidence(tmp_path/'parent',counters)
    checkpoint=parent/'latest.pt';initialization={'synthetic':True};provenance={'synthetic':True,'protocol_sha256':'synthetic'}
    torch.save(run.snapshot(net,opt,streams,cfg,initialization,counters,provenance),checkpoint)
    calls=[]
    def fake_evaluation(net,cfg,opt,streams,out,**kwargs):
        calls.append(out.name);out.mkdir();(out/'summary.json').write_text('{"synthetic":true}')
        return {'complete':True}
    def forbidden(*args,**kwargs):raise AssertionError('No further update at final boundary')
    monkeypatch.setattr(run,'checked_evaluation',fake_evaluation);monkeypatch.setattr(run,'update',forbidden)
    out=tmp_path/'resumed';out.mkdir()
    result=run.run_training(net,cfg,initialization,provenance,out,device='cpu',manifest=tmp_path/'unused',resume=checkpoint)
    assert result['completed'] and calls==['block3_evaluation']
    assert(out/'latest.pt').read_bytes()==checkpoint.read_bytes()
    assert(out/'final.pt').read_bytes()==(out/'block3.pt').read_bytes()
    assert run.sha(out/'resume_parent_evidence/checkpoint.pt')==run.sha(checkpoint)
    calls.clear();other=tmp_path/'already_complete';other.mkdir()
    result=run.run_training(net,cfg,initialization,provenance,other,device='cpu',manifest=tmp_path/'unused',resume=out/'final.pt')
    assert result['completed'] and not calls
    assert(other/'latest.pt').read_bytes()==(out/'block3_before_evaluation.pt').read_bytes()
    assert(other/'resume_parent_evidence/checkpoint.pt').read_bytes()==(out/'final.pt').read_bytes()
    assert(other/'resume_parent_evidence/ancestor_evidence/metrics.jsonl').read_bytes()==(parent/'metrics.jsonl').read_bytes()


def test_both_qualification_scores_and_authenticated_evidence_are_required(tmp_path):
    initial=json.loads(run.INITIALIZATION.read_text())
    (tmp_path/'audits').mkdir()
    for condition in ('gc100','pgc33'):
        folder=tmp_path/condition;(folder/'qualification_evaluation').mkdir(parents=True)
        row=next(r for r in initial['policies']if r['condition']==condition and r['actor_mode']=='retained_actor')
        report=dict(condition=condition,variant_order=['continuous'],checkpoint_policy_hash=row['full_initial_policy_hash'],
            variants={'continuous':dict(all_physical_checks=True,totals=dict(valid_trials=256,trials=256,invalid_trials=0,successes=200))})
        (folder/'qualification_evaluation/summary.json').write_text(json.dumps(report))
        for name in ('incoming.pt','qualification_after.pt','protocol.json'):(folder/name).write_text('synthetic fixture only')
        artifacts={str(p.relative_to(folder)):dict(sha256=run.sha(p),bytes=p.stat().st_size)for p in folder.rglob('*')if p.is_file()}
        audit=dict(phase='qualification',condition=condition,protocol_sha256='synthetic',passed=True,complete=True,
            failures=[],pending=[],auditor_sha256=run.sha(ROOT/'scripts/audit_goal_cue_memory.py'),artifacts=artifacts)
        (tmp_path/'audits'/f'{condition}.json').write_text(json.dumps(audit))
    assert run.qualification_gate(tmp_path,'synthetic')['both_originals_qualified']
    path=tmp_path/'pgc33/qualification_evaluation/summary.json';report=json.loads(path.read_text())
    report['variants']['continuous']['totals']['successes']=191;path.write_text(json.dumps(report))
    ap=tmp_path/'audits/pgc33.json';audit=json.loads(ap.read_text())
    audit['artifacts']['qualification_evaluation/summary.json']=dict(sha256=run.sha(path),bytes=path.stat().st_size)
    ap.write_text(json.dumps(audit))
    with pytest.raises(AssertionError,match='Both frozen original'):run.qualification_gate(tmp_path,'synthetic')
