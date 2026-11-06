"""Synthetic inputs only. Never generates the study bank or loads source policies."""
import copy
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'tests'))
import local_return_feedback_evaluate as evaluator
import local_return_feedback as feedback
from test_local_return_feedback import fixture
from rl_visual_memory_extend import tree_equal


def test_lossless_compression_keeps_float32_rgb_and_detects_corruption(tmp_path):
    generator=torch.Generator().manual_seed(740101)
    value=dict(rgb=torch.rand(3,2,3,7,7,generator=generator),
        integer=torch.arange(6).reshape(3,2), nested={1:dict(state=torch.randn(8,generator=generator))})
    path=tmp_path/'raw.pt.gz';record=evaluator.save_raw(path,value)
    assert tree_equal(value,evaluator.load_raw(path,record))
    assert record['tensor_schema']['/rgb']['dtype']=='torch.float32'
    assert record['tensor_schema']['/rgb']['sha256']==feedback.task.tensor_hash(value['rgb'])
    with pytest.raises(AssertionError):evaluator.save_raw(path,value)
    wrong=copy.deepcopy(record);wrong['tensor_schema']['/rgb']['dtype']='torch.uint8'
    with pytest.raises(AssertionError):evaluator.load_raw(path,wrong)
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(AssertionError):evaluator.load_raw(path,record)


def test_full_input_evaluation_preserves_weights_mode_and_all_global_rng(tmp_path,monkeypatch):
    net,cfg,_,plan=fixture(740102);before=copy.deepcopy(net.state_dict())
    original=feedback.collect
    def consuming_global(*args,**kwargs):
        torch.rand(7)  # Evaluation must restore this without consuming learner RNG.
        return original(*args,**kwargs)
    monkeypatch.setattr(feedback,'collect',consuming_global)
    rng=evaluator.global_rng_snapshot();net.train()
    report=evaluator.evaluate_plan(net,cfg,plan,{'synthetic':True},tmp_path/'eval',actor_mode='reset_actor')
    assert report['complete'] and report['all_physical_checks'] and report['totals']['trials']==2
    assert not report['criterion_pass'] and not report['common_information_certificate_applies']
    assert net.training and tree_equal(before,net.state_dict())
    assert tree_equal(rng,evaluator.global_rng_snapshot())
    for index,result in enumerate(report['contexts']):
        raw=evaluator.load_raw(tmp_path/'eval'/f'context{index}'/'raw.pt.gz',result['raw'])
        assert raw['rgb'].dtype==torch.float32 and raw['rgb'].shape==(160,1,3,84,84)
        assert raw['code'].shape==(160,1,512) and raw['goal_world'].shape==(1,2)
        assert not raw['noise'].any() and not raw['reset_before'].any() and not raw['reset_after'].any()
        assert not raw['bootstrap'].any()
        assert torch.equal(raw['goal_world'],plan['goal'][index:index+1])


def test_failed_context_is_retained_without_replacement_or_denominator_loss(tmp_path,monkeypatch):
    net,cfg,_,plan=fixture(740103)
    def fail(*args,**kwargs):raise AssertionError('synthetic preparation failure')
    monkeypatch.setattr(feedback,'collect',fail)
    report=evaluator.evaluate_plan(net,cfg,plan,{'synthetic':True},tmp_path/'failed')
    assert report['complete'] and not report['all_physical_checks']
    assert report['totals']['trials']==report['totals']['invalid_trials']==2
    assert report['totals']['valid_trials']==0 and 'invalid trials contribute zero' in report['totals']['descriptive_mean_convention']
    assert report['totals']['successes']==0 and not report['criterion_pass']
    for i,result in enumerate(report['contexts']):
        assert not result['valid'] and result['failure_category']=='collection'
        raw=evaluator.load_raw(tmp_path/'failed'/f'context{i}'/'failed_plan.pt.gz',result['failed_plan'])
        assert int(raw['context'][0])==i


def test_summary_rejects_duplicates_missing_trials_and_invalid_successes():
    rows=[dict(context=c,heading_index=h,variant='feedback_native',hit=False) for c in range(2) for h in range(2)]
    assert evaluator.summarize(rows,2,2)['trials']==4
    with pytest.raises(AssertionError):evaluator.summarize(rows[:-1],2,2)
    with pytest.raises(AssertionError):evaluator.summarize(rows[:-1]+[rows[0]],2,2)
    broken=copy.deepcopy(rows);broken[0].update(invalid=True,hit=True)
    with pytest.raises(AssertionError):evaluator.summarize(broken,2,2)


def test_reference_policy_is_parameterless_and_never_uses_a_source_actor():
    net=evaluator.ReferencePolicy(256)
    assert list(net.parameters())==[] and dict(net.state_dict())=={}
    h,c=net.zero_state(2,'cpu')
    assert not net.pi(h).any() and not net.v(h).any()
    assert net.embed(torch.ones(2,3,84,84)).shape==(2,256)
    assert torch.equal(net.lstm(torch.ones(2,1287),(h,c))[0],h)
