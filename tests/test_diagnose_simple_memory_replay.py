"""Private random native-policy fixtures only; never load the study checkpoint."""
import copy
import json
from pathlib import Path
import sys
import time

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_simple_memory_replay as diag
from code import train_memory as train


def fixture(tmp_path):
    cfg=train.parse_args(['--out',str(tmp_path/'unused'),'--device','cpu','--threads','1',
                         '--cond','pgc33','--seed','91351','--code_seed','91351','--n_envs','2',
                         '--frames','32','--rollout','3','--task_ramp_frames','1'])
    net,opt,source,rt=train.fresh_training(cfg)
    rt['env'].t[0]=cfg['episode_len']-1
    rt['clock'].known.fill_(True);rt['clock'].remaining.fill_(1)
    rt['env'].goal[:]=-rt['env'].pos.sign()*.9
    return cfg,net,opt,source,rt


def test_hooks_preserve_actual_rollout_rng_and_distinguish_bootstrap(tmp_path):
    cfg,net,opt,source,rt=fixture(tmp_path)
    before=train.capture_runtime(rt)
    ordinary=train.collect_rollout(net,source,rt,cfg,3)
    expected_state=train.capture_runtime(rt)
    train.restore_runtime(rt,before)
    roll,state0,boot,bi,ep,actor,validation=diag.capture_rollout(net,source,rt,cfg,3)
    assert diag.tree_hash((roll,state0,boot,bi,ep))==diag.tree_hash(ordinary)
    assert diag.tree_hash(train.capture_runtime(rt))==diag.tree_hash(expected_state)
    assert validation['qualified'] and validation['rollout_calls']==3 and validation['bootstrap_calls']==1
    assert len(actor['encoder_calls'])==4 and actor['logits'].shape==(4,2,6)
    assert roll['d'][0][0]==1


def test_all_seven_replays_keep_parameters_inputs_and_rng_unchanged(tmp_path):
    cfg,net,opt,source,rt=fixture(tmp_path)
    roll,state0,boot,bi,ep,actor,_=diag.capture_rollout(net,source,rt,cfg,3)
    weights=train.policy_hash(net);optimizer=diag.tree_hash(opt.state_dict());rng=diag.tree_hash(train.capture_runtime(rt))
    results={}
    for name,kind,eg,rg in diag.PATHS:
        report,arrays=diag.replay_path(net,roll,state0,actor,kind=kind,encoder_grad=eg,recurrent_grad=rg)
        assert report['inputs_and_reset_rule_verified'] and report['original_logit_assert']['passed']
        assert report['original_value_assert']['passed']
        assert report['logits_requires_grad']==rg
        assert arrays['h_before'].shape==(3,2,256)
        assert train.policy_hash(net)==weights and diag.tree_hash(opt.state_dict())==optimizer
        assert diag.tree_hash(train.capture_runtime(rt))==rng
        results[name]=report
    assert results['B_same_batch_no_grad']['encoder_batch_sizes']==[2,2,2]
    assert results['D_actor_embedding_grad']['encoder_batch_sizes']==[]
    assert results['D_actor_embedding_grad']['differences']['embeddings']['bitwise_equal']


@pytest.mark.parametrize('corruption',['code','goal','state','done','rgb'])
def test_corrupted_inputs_or_reset_cannot_be_called_precision_only(tmp_path,corruption):
    cfg,net,opt,source,rt=fixture(tmp_path)
    roll,state0,boot,bi,ep,actor,_=diag.capture_rollout(net,source,rt,cfg,3)
    roll=copy.deepcopy(roll);state0=tuple(v.clone() for v in state0)
    if corruption in ('code','goal'):roll[corruption][1][0,0]+=1
    elif corruption=='state':state0[0][0,0]+=1
    elif corruption=='done':roll['d'][0][0]=1-roll['d'][0][0]
    else:roll['rgb'][0][0,0,0,0]+=1
    with pytest.raises(RuntimeError):diag.validate_actor_trace(roll,state0,actor,bi)


def test_comparator_retains_original_tolerance_and_first_divergence():
    expected=torch.tensor([[[0.,1.],[2.,3.]],[[4.,5.],[6.,7.]]])
    actual=expected.clone();actual[1,0,0]+=.01
    row=diag.compare_array(actual,expected)
    assert row['first_changed']==[1,0,0] and row['first_tolerance_violation']==[1,0,0]
    assert row['original_tolerance_violations']==1 and row['tolerance']['atol']==1e-5
    assert json.loads(json.dumps(row))==row
    assert not diag.original_assert(actual,expected)['passed']
    assert diag.original_assert(expected,expected)['passed']


def test_deadline_is_checked_inside_capture_and_removes_hooks(tmp_path):
    cfg,net,opt,source,rt=fixture(tmp_path)
    with pytest.raises(TimeoutError):diag.capture_rollout(net,source,rt,cfg,3,deadline=time.monotonic()-1)
    assert not net.encoder._forward_hooks and not net.lstm._forward_hooks and not net.lstm._forward_pre_hooks


def test_failed_authentication_retains_partial_technical_record(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    def reject(*args):raise RuntimeError('private mismatched source fixture')
    monkeypatch.setattr(diag,'authenticate_inputs',reject)
    out=tmp_path/'diagnostic'
    result=diag.execute(tmp_path,out,manifest,max_seconds=10)
    assert not result['complete'] and not result['input_authentication_complete']
    assert result['error']['type']=='RuntimeError' and result['producer_sha256']==diag.sha(diag.__file__)
    assert diag.read(out/'summary.json')['error']==result['error']


def test_real_execution_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='never train locally'):diag.authenticate_cuda({})


def test_original_multichunk_replay_coverage_on_small_private_fixture(tmp_path):
    cfg,net,opt,source,rt=fixture(tmp_path)
    original_embed=net.embed
    # Private instance only: exercise the4+2 batching branch at actor batch2.
    net.embed=lambda rgb,chunk=1024:original_embed(rgb,chunk=4)
    roll,state0,boot,bi,ep,actor,validation=diag.capture_rollout(net,source,rt,cfg,3)
    report,arrays=diag.replay_path(net,roll,state0,actor,kind='original')
    assert report['encoder_batch_sizes']==[4,2]
    assert report['inputs_and_reset_rule_verified'] and report['original_logit_assert']['passed']
    assert report['original_value_assert']['passed']
    assert arrays['embeddings'].shape==(3,2,256)
    assert all(p.grad is None for p in net.parameters())
    json.dumps(report)
