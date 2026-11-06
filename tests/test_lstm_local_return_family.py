"""Native-physics/TinyPolicy checks; no completed-model neural outcomes."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import pytest
import torch
import lstm_local_return_family as family
from test_local_goal_return_training import TinyPolicy,config


def fixture(condition='pgc33',phases=8):
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(77201);net=TinyPolicy()
    cfg=config(condition);bank=family.task.TrialStreams(77202).draw(1,family.task.STAGES[3],cfg,headings=16)
    plan,metadata=family._densify_context(bank,0,phases)
    return net,cfg,bank,plan,metadata


def test_dense_and_sparse_phase_plans_preserve_physical_context_and_tapes():
    _,_,bank,plan,meta=fixture(phases=32)
    assert torch.equal(plan['heading'][::2],bank['heading'])
    assert meta['sealed_heading_anchor_bytes_preserved'] and bool((plan['delay']==60).all())
    selected=family.slice_plan(plan,[0,4,8,12,16,20,24,28])
    assert selected['heading_index'].tolist()==[0,4,8,12,16,20,24,28]
    for key in ('motor','state'):
        assert torch.equal(selected[key],bank[key][:,:,:1].expand_as(selected[key]))
    for key in ('endpoint','goal','pickup_angle','land_pos','stripe_phase','duration_a','duration_b','turn_a','turn_b'):
        assert torch.equal(selected[key],bank[key][:1].expand_as(selected[key]))
    assert torch.equal(selected['action'],bank['action'][:,:1].expand_as(selected['action']))
    assert selected['n_contexts']==1 and selected['headings']==8


def test_real_sealed_context_metadata_without_neural_execution():
    cfg=json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    cfg.update(n_envs=32,rollout=160,entropy=.01)
    plan,meta=family.dense_plan(cfg,0,phases=16)
    bank,record,_=family.evaluator.load_bank('final_hard')
    assert torch.equal(plan['heading'],bank['heading'][:16])
    assert meta['bank_sha256']==record['sha256'] and meta['no_new_random_draws']
    assert plan['state'].shape==(160,2,16,256)
    with pytest.raises(AssertionError):family.dense_plan(dict(cfg,entropy=.02),0,phases=16)


@pytest.mark.parametrize('condition',['gc100','pgc33'])
def test_actual_hooks_have_correct_cue_and_first_free_recurrence(condition):
    net,cfg,_,plan,_=fixture(condition);weights=copy.deepcopy(net.state_dict());rng=torch.random.get_rng_state().clone()
    result=family.collect_family(net,cfg,plan,chunk_size=4,provenance={'scope':'synthetic unit test'})
    assert result['audit']['first_free_clock']-result['audit']['cue_clock']==60
    assert result['audit']['all_phase_variant_delay_inputs_exact']
    assert result['audit']['grid_cue_identity_exact']==(condition=='gc100')
    assert torch.equal(torch.random.get_rng_state(),rng) and net.training
    assert all(torch.equal(weights[k],v) for k,v in net.state_dict().items())
    assert not net.lstm._forward_hooks and not net.lstm._forward_pre_hooks
    # Match original batch shape to require exact float32 recurrence, rather
    # than confusing GEMM batch-size roundoff with a one-update timing error.
    for start in (0,4):
        initial=result['pre_cue_state'][start:start+4].repeat(4,1);h,c=initial[:,:8],initial[:,8:]
        cue=torch.cat([result['cue_inputs'][name][start:start+4] for name in family.CUES])
        targets={key:torch.cat([result[key][name][start:start+4] for name in family.CUES])
                 for key in ('starts','after_first_common','first_free_states','first_free_logits')}
        with torch.no_grad():
            h,c=net.lstm(cue,(h,c))
            assert torch.equal(torch.cat((h,c),1),targets['starts'])
            x=result['dark_input'][None].expand(16,-1)
            for t in range(1,61):
                h,c=net.lstm(x,(h,c))
                if t==1:assert torch.equal(torch.cat((h,c),1),targets['after_first_common'])
            assert torch.equal(torch.cat((h,c),1),targets['first_free_states'])
            assert torch.equal(net.pi(h),targets['first_free_logits'])
    physics=result['nativephysics']
    assert torch.equal(physics['first_free_position'],plan['endpoint'])
    assert float(family._wrap(result['theta']-plan['heading']).abs().max())<1e-5
    assert float(family._wrap(physics['endpoint_heading_backderived']-result['theta']).abs().max())<2e-5


def test_chunking_and_arbitrary_selected_phases_retain_same_family():
    net,cfg,_,plan,_=fixture('pgc33')
    together=family.collect_family(net,cfg,plan,chunk_size=8)
    indices=torch.tensor([0,2,4,6]);sparse=family.slice_plan(plan,indices)
    separated=family.collect_family(net,cfg,sparse,chunk_size=2)
    assert torch.equal(separated['phase_indices'],indices)
    assert torch.equal(together['dark_input'],separated['dark_input'])
    for key in ('pre_cue_state','theta'):
        assert torch.allclose(together[key][indices],separated[key],atol=2e-6,rtol=0)
    for name in family.CUES:
        for key in ('cue_inputs','starts','first_free_states'):
            # Distinct GEMM batch shapes can round accumulating cell states
            # differently; inputs and clocks remain exact.
            assert torch.allclose(together[key][name][indices],separated[key][name],atol=2e-6,rtol=1e-6)


def test_failed_collection_removes_hooks_and_restores_mode():
    net,cfg,_,plan,_=fixture();net.train()
    original=family.task.collect
    def fail(*args,**kwargs):raise RuntimeError('synthetic producer failure')
    family.task.collect=fail
    try:
        with pytest.raises(RuntimeError,match='synthetic producer'):
            family.collect_family(net,cfg,plan,chunk_size=4)
    finally:family.task.collect=original
    assert net.training and not net.lstm._forward_hooks and not net.lstm._forward_pre_hooks


def test_phase_dependent_delay_input_is_rejected():
    net,cfg,_,plan,_=fixture();cue=int(family.task.schedule(plan,'cpu')['glimpse'][0]);clock=[0]
    def corrupt(module,args):
        t=clock[0];clock[0]+=1
        if t==cue+1:
            x,state=args;x=x.clone();x[1,-1]=1.
            return x,state
    handle=net.lstm.register_forward_pre_hook(corrupt)
    try:
        with pytest.raises(AssertionError,match='phase/cue dependent'):
            family.collect_family(net,cfg,plan,chunk_size=4)
    finally:handle.remove()
    assert not net.lstm._forward_hooks and not net.lstm._forward_pre_hooks
