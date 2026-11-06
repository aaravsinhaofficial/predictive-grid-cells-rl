"""Private synthetic/physical fixtures only; no original-policy outcomes."""
import math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_turn_original_memory as study
import native_turn_teacher_preflight as parent
from test_native_turn_teacher_preflight import fixture


def make_raw(condition='pgc33',kind='probe_glimpse'):
    cfg,net,pop,context=fixture(condition)
    raw,env,audit=parent.construct_history(net,pop,cfg,context,kind)
    state=net.zero_state(16,'cpu')
    with torch.no_grad():
        for x in raw['teacher_x']:state=net.lstm(x,state)
        target=net.pi(state[0]).softmax(1)
    physics=parent.action_progress(cfg,context,env)
    raw.update(teacher_first_choice_probabilities=target,first_update_physics=physics)
    reference=dict(audit=audit,pairs=parent.pair_metrics(target,physics['progress_m']))
    return net,pop,context,raw,reference


def test_only_measured_predictive_slots_rotated_and_turn_grid_shams():
    for condition in ('pgc33','gc100'):
        for kind in parent.KINDS:
            net,pop,context,raw,reference=make_raw(condition,kind)
            cue=reference['audit']['cue_index']
            for variant in study.VARIANTS:
                x,reset,operation=study.variant_history(raw,reference['audit'],pop,variant)
                assert torch.equal(x[-1],raw['student_native_x'][-1])
                if kind=='turn_only'or condition=='gc100':assert torch.equal(x,raw['student_native_x'])
                if variant.startswith('pre_cue_reset'):
                    assert reset==(cue if cue is not None else raw['labels'].index('suffix_0'))
                if variant.startswith('choice_reset'):assert reset==len(x)-1
            if condition=='pgc33'and kind=='probe_glimpse':
                plus,_,_=study.variant_history(raw,reference['audit'],pop,'cue_plus60')
                minus,_,_=study.variant_history(raw,reference['audit'],pop,'cue_minus60')
                assert not torch.equal(plus[cue],minus[cue])
                assert not torch.equal(plus[cue],raw['student_native_x'][cue])
                p=raw['physical'];idx=torch.as_tensor(pop.pgc_idx)
                expected=pop.activity(p['pos'][cue],study.rotate(p['mdir'][cue],math.pi/3),study.rotate(p['obs_vel'][cue],math.pi/3))
                assert torch.equal(plus[cue,:,256+idx],expected[:,idx])


def test_reset_occurs_before_declared_input_and_first_choice_once():
    net,pop,context,raw,reference=make_raw()
    for variant in ('pre_cue_reset_native','pre_cue_reset_ordinary','choice_reset_native'):
        x,reset,_=study.variant_history(raw,reference['audit'],pop,variant)
        result=study.replay_variant(net,x,context['initial_hc'],reset)
        state=net.zero_state(16,'cpu')
        with torch.no_grad():
            for item in x[reset:]:state=net.lstm(item,state)
        assert torch.equal(torch.cat(state,1),result['first_choice_state_after_input'])
        assert torch.equal(net.pi(state[0]),result['first_choice_logits'])
        assert torch.count_nonzero(result['reset_state_after'])==0
        assert torch.count_nonzero(result['reset_state_before'])>0
    erased=study.replay_variant(net,x,context['initial_hc'],len(x)-1)
    assert torch.count_nonzero(erased['first_choice_state_before_input'])==0
    assert torch.equal(erased['first_choice_logits'],erased['first_choice_logits'][:1].expand(16,-1))


def test_complete_context_keeps_all_variants_pairs_and_zero_state_bounds():
    for condition,kind in (('gc100','probe_glimpse'),('pgc33','probe_glimpse'),('pgc33','turn_only')):
        net,pop,context,raw,reference=make_raw(condition,kind)
        saved,report=study.evaluate_context(net,pop,raw,reference,0,kind,context['initial_hc'])
        assert set(saved['variants'])==set(study.VARIANTS)
        assert len(report['rows'])==8*16 and len(report['pairs'])==8*8
        assert report['zero_state_same_input_exact']and report['choice_reset_sham_exact']
        assert report['pre_cue_reset_ordinary_all_phases_exact']
        for key in ('first_choice_state_before_input','first_choice_state_after_input','first_choice_logits'):
            ordinary=saved['variants']['pre_cue_reset_ordinary'][key]
            assert torch.equal(ordinary,ordinary[:1].expand_as(ordinary))
            if condition=='gc100':assert torch.equal(ordinary,saved['variants']['pre_cue_reset_native'][key])
        assert all(p['progress_above_common_ceiling_m']<=1e-8 for p in report['pairs']if p['variant'].startswith('choice_reset'))


def test_own_prefix_replayed_after_weight_change_not_cached_teacher_state():
    _,net,_,_=fixture();gen=torch.Generator().manual_seed(998901)
    x=torch.randn(8,32,1287,generator=gen)*.1;done=torch.zeros(8,32,dtype=torch.bool);done[3,0]=True
    state=net.zero_state(32,'cpu');before=[]
    with torch.no_grad():
        for t in range(len(x)):
            before.append(torch.cat(state,1));state=net.lstm(x[t],state);state=tuple(v*(~done[t])[:,None]for v in state)
    data=dict(x=x,done=done,state_before=torch.stack(before));contexts=[]
    for i,(slot,t)in enumerate(((0,4),(1,6))):
        contexts.append(dict(context_id=i,slot=slot,step=t,preceding_native_x=x[:t,slot],preceding_done=done[:t,slot],
            native_x=x[t,slot],initial_hc=data['state_before'][t,slot].clone()))
    bank=dict(contexts=contexts);states,report=study.replay_own_prefixes(net,bank,data,True)
    assert all(r['original_cache_max_error']==0 for r in report['contexts'])
    with torch.no_grad():net.lstm.bias_ih.add_(.1)
    changed,report=study.replay_own_prefixes(net,bank,data,False)
    assert torch.count_nonzero(changed[0])==0  # Real preceding terminal reset.
    assert not torch.equal(changed[1],states[1])
    state=net.zero_state(32,'cpu')
    with torch.no_grad():
        for t in range(6):
            state=net.lstm(x[t],state);state=tuple(v*(~done[t])[:,None]for v in state)
    assert torch.equal(changed[1],torch.cat(state,1)[1])
    import pytest
    with pytest.raises(AssertionError):study.replay_own_prefixes(net,bank,data,True)


def test_common_input_js_floor_does_not_require_imitation_for_useful_progress():
    inputs=torch.zeros(4,1287);inputs[2:,0]=1
    targets=torch.tensor([[.9,.1,0,0,0,0],[.1,.9,0,0,0,0],[.8,.2,0,0,0,0],[.2,.8,0,0,0,0]])
    refs=[dict(context=i//2,kind='turn_only',case=i%2)for i in range(4)]
    common=torch.full((4,6),1/6)
    predictions={v:common.clone()for v in study.VARIANTS};predictions['native']=targets.clone()
    report=study.grouped_comparisons(inputs,targets,refs,predictions)
    assert report['groups']==2
    for group in report['records']:
        assert group['generalized_js_nats']>0
        assert group['variants']['native']['mean_teacher_forward_kl_nats']==0
        assert group['variants']['choice_reset_native']['memoryless_floor_minus_observed_kl_nats']<0
    # A history-specific translational choice can make progress without matching
    # the teacher's opposite-turn distributions: separate claims/metrics.
    progress=torch.tensor([[0.,0.,.01,-.01,0,0],[0,0,-.01,.01,0,0]])
    policy=torch.tensor([[.001,.001,.995,.001,.001,.001],[.001,.001,.001,.995,.001,.001]])
    assert float((policy*progress).sum(1).mean())>float(progress.mean(0).max())
    assert float(study.forward_kl(targets[:2],policy).mean())>report['records'][0]['generalized_js_nats']
