"""Evaluator tests use synthetic policies/plans, never learned study outcomes."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'tests'))
import pytest
import torch

import goal_cue_memory_evaluate as evaluator
from test_goal_cue_memory import equal, fixture


def test_variant_order_and_exact_context_local_quarter_turns():
    assert evaluator.ordered_variants(('donor_minus90', 'brief_native', 'self_swap')) == (
        'brief_native', 'donor_minus90', 'self_swap')
    assert evaluator.donor_indices('donor_plus90', 16).tolist() == list(range(4,16))+list(range(4))
    assert evaluator.donor_indices('donor_minus90', 16).tolist() == list(range(12,16))+list(range(12))
    assert evaluator.donor_indices('self_swap', 16).tolist() == list(range(16))
    for bad in (('donor_plus90',), ('no_cue', 'no_cue'), ('unregistered',)):
        with pytest.raises(AssertionError): evaluator.ordered_variants(bad)
    with pytest.raises(AssertionError): evaluator.donor_indices('donor_plus90', 2)


def test_all_eleven_branches_archive_literal_inputs_and_genuine_preinput_donors(tmp_path, monkeypatch):
    net, cfg, _, plan = fixture(78101, delay=0, angles=4)
    before = copy.deepcopy(net.state_dict()); net.train()
    rng = evaluator.global_rng_snapshot()
    original = evaluator.task.collect
    def consume_global(*args, **kwargs):
        torch.rand(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator.task, 'collect', consume_global)
    report = evaluator.evaluate_plan(net, cfg, plan, {'synthetic': True}, tmp_path/'all')
    assert report['complete'] and report['all_variants_valid']
    assert report['variant_order'] == list(evaluator.FULL_VARIANTS)
    assert net.training and equal(before, net.state_dict()) and equal(rng, evaluator.global_rng_snapshot())
    assert not report['gates']['qualification']['eligible'] and not report['gates']['useful_memory']['eligible']
    assert not report['gates']['common_information_certificate_applies']
    context = report['contexts'][0]
    saved_plan = evaluator.load_raw(tmp_path/'all/context0/input_plan.pt.gz', context['input_plan'])
    assert equal(saved_plan, evaluator.context_plan(plan, 0))
    raws = {}
    for name, result in context['variants'].items():
        assert result['valid'] and all(result['physical_checks'].values())
        assert not result['first_free_transition']['full_history_replay_performed']
        raw = evaluator.load_raw(tmp_path/f'all/context0/{name}/raw.pt.gz', result['raw']); raws[name] = raw
        assert raw['rgb'].dtype == torch.float32 and raw['rgb'].shape == (160,4,3,84,84)
        assert raw['goal_input'].shape == (160,4,512)
        assert torch.equal(raw['physical_goal'], plan['goal'])
        assert not raw['physics_active'][:15].any() and bool((raw['action'][:15] == -1).all())
        assert raw['evidence']['setup']['learning_reward'] == 0
        assert report['variants'][name]['totals']['trials'] == report['variants'][name]['totals']['valid_trials'] == 4
    native = raws['brief_native']
    for name in evaluator.DONOR_VARIANTS:
        raw = raws[name]; ids = evaluator.donor_indices(name, 4)
        assert torch.equal(raw['first_free_state_override'], torch.stack([
            native['evidence']['first_free'][int(i)]['state_before_input'] for i in ids]))
        assert bool(raw['state_override_mask'][15].all()) and int(raw['state_override_mask'].sum()) == 4
        for i, donor in enumerate(ids):
            item = raw['evidence']['first_free'][i]
            assert torch.equal(item['state_before_input'], native['evidence']['first_free'][int(donor)]['state_before_input'])
            assert torch.equal(item['state_before_intervention'], native['evidence']['first_free'][i]['state_before_input'])
        assert context['variants'][name]['configuration']['source_native_raw_sha256'] == context['variants']['brief_native']['raw']['sha256']
    for name in evaluator.WRONG_VARIANTS:
        ids = evaluator.donor_indices(name, 4)
        assert torch.equal(raws[name]['saved_goal'], native['saved_goal'][ids])
        assert not raws[name]['goal_input'][15:].any()
    assert context['variants']['self_swap']['self_swap_exact']
    for key in ('position','action','reward','logits','values'):
        assert torch.equal(raws['self_swap'][key], native[key])
    assert not raws['no_cue']['goal_input'].any()
    assert bool(raws['brief_reset']['reset_before'][15].all())
    assert bool(raws['continuous_reset']['reset_before'][15].all())
    assert raws['continuous_reset']['goal_input'][15].abs().sum() > 0


def test_failed_native_source_keeps_every_denominator_and_continues_independent_branches(tmp_path, monkeypatch):
    net,cfg,_,plan = fixture(78102, delay=0, angles=4)
    original = evaluator.task.collect
    def fail_native(*args, **kwargs):
        if kwargs['cue_mode'] == 'brief' and kwargs['goal_kind'] == 'native' and not kwargs['erase_after_cue']:
            raise RuntimeError('Synthetic native-source failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator.task, 'collect', fail_native)
    report = evaluator.evaluate_plan(net,cfg,plan,{'synthetic':True},tmp_path/'failed',
        variants=('brief_native','continuous','donor_plus90','wrong_plus90'))
    assert report['complete'] and not report['all_variants_valid']
    assert report['variants']['continuous']['totals']['valid_trials'] == 4
    for name in ('brief_native','donor_plus90','wrong_plus90'):
        totals = report['variants'][name]['totals']
        assert totals['trials'] == totals['invalid_trials'] == 4 and totals['successes'] == 0
        result = report['contexts'][0]['variants'][name]
        failed = evaluator.load_raw(tmp_path/f'failed/context0/{name}/failed_plan.pt.gz', result['failed_plan'])
        assert torch.equal(failed['goal'], plan['goal'])


def test_summaries_reject_denominator_corruption_and_gates_are_fixed_counts():
    rows = [dict(context=c,angle_index=a,hit=False) for c in range(16) for a in range(16)]
    assert evaluator.summarize(rows,16,16)['trials'] == 256
    with pytest.raises(AssertionError): evaluator.summarize(rows[:-1],16,16)
    with pytest.raises(AssertionError): evaluator.summarize(rows[:-1]+[rows[0]],16,16)
    broken=copy.deepcopy(rows);broken[0].update(hit=True,invalid=True)
    with pytest.raises(AssertionError): evaluator.summarize(broken,16,16)
    branches = {}
    for name, hits in [('continuous',192),('brief_native',167),('brief_reset',128)]:
        values=copy.deepcopy(rows)
        for row in values[:hits]: row['hit']=True
        branches[name]=dict(totals=evaluator.summarize(values,16,16),all_physical_checks=True)
    result=evaluator.gates(branches)
    assert result['qualification']['passed'] and result['useful_memory']['passed']
    branches['brief_reset']['totals']['successes']=129
    assert not evaluator.gates(branches)['useful_memory']['passed']
    branches['continuous']['totals']['valid_trials']=255
    assert not evaluator.gates(branches)['qualification']['passed']


def test_donor_progress_uses_actual_path_without_changing_recipient_target():
    goal=torch.tensor([[.75,0.],[0.,.75],[-.75,0.],[0.,-.75]])
    plan={'goal':goal,'start':torch.zeros(4,2)}
    # Four synthetic paths all follow their +90degree donor and remain away
    # from the physical recipient disk. These arrays do not run an agent.
    ids=evaluator.donor_indices('donor_plus90',4)
    position=torch.stack((goal[ids]*.4,goal[ids]*.8))
    raw={'position':position,'physics_active':torch.ones(2,4,dtype=torch.bool)}
    before=copy.deepcopy(plan)
    result=evaluator.donor_behavior(raw,plan,ids)
    assert equal(before,plan)
    assert all(row['first_target_disk']=='donor' and row['first_donor_disk_entry_step']==2 for row in result)
    assert all(row['progress_to_donor_m']>0 and row['progress_to_recipient_m']<0 for row in result)


def test_early_directional_progress_uses_fixed_fifteenth_step_and_holds_early_terminal():
    goal=torch.tensor([[.75,0.],[0.,.75],[-.75,0.],[0.,-.75]])
    plan={'goal':goal,'start':torch.zeros(4,2)}; ids=evaluator.donor_indices('donor_plus90',4)
    position=torch.zeros(20,4,2); free=torch.ones(20,4,dtype=torch.bool)
    position[:,0]=goal[ids[0]]*torch.linspace(.01,.8,20)[:,None]
    # Case1 reaches the actual recipient disk after3steps. Its terminal
    # position remains the early-horizon endpoint; no fabricated later path.
    position[:,1]=goal[1]*.8;position[:2,1]=goal[1]*torch.tensor([.2,.4])[:,None]
    free[3:,1]=False
    result=evaluator.donor_behavior({'position':position,'physics_active':free},plan,ids)
    expected=float(goal[ids[0]].norm()-(position[14,0]-goal[ids[0]]).norm())
    assert result[0]['early_15_free_progress_to_donor_m']==expected
    assert result[0]['early_observed_free_updates']==15 and not result[0]['early_terminal_position_held']
    assert result[0]['early_15_free_progress_to_donor_m']<result[0]['progress_to_donor_m']
    assert result[1]['early_observed_free_updates']==3 and result[1]['early_terminal_position_held']
    assert result[1]['first_recipient_disk_entry_step']==3
    assert result[1]['full_path_recipient_hit_censored']
    assert result[1]['early_15_free_progress_to_recipient_m']==result[1]['progress_to_recipient_m']
    rows=[dict(context=0,angle_index=i,hit=False,donor_behavior=row) for i,row in enumerate(result)]
    totals=evaluator.summarize(rows,1,4)['donor_behavior']
    assert totals['early_terminal_position_held_trials']==1
    assert totals['full_path_recipient_hit_censored_trials']==1
    assert totals['mean_early_15_free_progress_to_donor_m']==sum(r['early_15_free_progress_to_donor_m']for r in result)/4


def test_production_bank_authentication_precedes_any_collection_or_output(tmp_path, monkeypatch):
    calls=[]
    def reject(manifest, bank_name):
        calls.append((manifest,bank_name))
        raise AssertionError('Synthetic sealed-bank hash mismatch')
    def forbidden(*args,**kwargs):
        raise RuntimeError('Collection must not happen after failed authentication')
    monkeypatch.setattr(evaluator,'load_bank',reject)
    monkeypatch.setattr(evaluator,'evaluate_plan',forbidden)
    with pytest.raises(AssertionError,match='hash mismatch'):
        evaluator.evaluate_bank(None,{},tmp_path/'not_created',manifest=tmp_path/'sealed.json',
                                bank_name='qualification',variants=evaluator.QUALIFICATION_VARIANTS)
    assert calls==[(tmp_path/'sealed.json','qualification')]
    assert not (tmp_path/'not_created').exists()
