"""Causal timing and physical separation, using private synthetic policies only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'tests'))
import pytest
import torch
from test_local_goal_return_training import TinyPolicy, config
import local_goal_return_training as task
import lstm_local_return_causal as causal


def fixture():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(781923)
        net = TinyPolicy().eval().requires_grad_(False)
    cfg = config('gc100'); source = task.make_source(cfg, 'cpu')
    plan = task.TrialStreams(781925).draw(1, task.STAGES[3], cfg, headings=2)
    plan['delay'].fill_(60); cue = int(task.schedule(plan, 'cpu')['glimpse'][0]); count = [0]; captured = {}
    def hook(module, inputs):
        if count[0] == cue:
            captured['pre'] = torch.cat(inputs[1], 1)[:2].clone()
        count[0] += 1
    handle = net.lstm.register_forward_pre_hook(hook)
    try:
        roll, rows, audit, first = task.collect(net, cfg, source, plan,
            variants=causal.VARIANTS, sigma=0., retain_inputs=False)
    finally: handle.remove()
    return net, cfg, plan, captured['pre'], roll, rows, first


def test_observer_is_exact_sham_and_first_free_not_double_processed():
    net, cfg, plan, pre, roll, expected_rows, first = fixture()
    rows, audit, raw = causal.collect_intervention(net, cfg, plan, pre)
    assert rows == expected_rows and audit['first_free_state_exact']
    for key in ('action', 'position', 'raw_reward', 'terminal', 'active'):
        assert torch.equal(raw[key], torch.stack(roll[key]))
    for i in first:
        assert torch.equal(raw['first_free'][i]['state'], first[i]['state'])
        assert torch.equal(raw['snapshots'][60][i], first[i]['state'])
    assert not net.lstm._forward_pre_hooks and not net.lstm._forward_hooks


def test_edit_changes_memory_but_preserves_physics_until_free_and_reset_cuts_it():
    net, cfg, plan, pre, roll, _, _ = fixture()
    _, _, sham = causal.collect_intervention(net, cfg, plan, pre)
    _, checks, edited = causal.collect_intervention(net, cfg, plan, pre, pre+.4)
    free = int(task.schedule(plan, 'cpu')['free'][0]); n = len(pre)
    assert checks['precue_history_replay_max_abs_error'] == 0
    assert checks['applied_state_max_change'] > .39
    assert torch.equal(sham['cue_inputs'], edited['cue_inputs'])
    assert torch.equal(sham['position'][:free], edited['position'][:free])
    assert torch.equal(sham['action'][:free], edited['action'][:free])
    assert not torch.equal(sham['snapshots'][0][:n], edited['snapshots'][0][:n])
    for t in causal.TIMES:
        assert torch.equal(sham['snapshots'][t][-n:], edited['snapshots'][t][-n:])
    assert torch.equal(sham['action'][:, -n:], edited['action'][:, -n:])


def test_bad_history_fails_and_always_removes_hooks():
    net, cfg, plan, pre, _, _, _ = fixture()
    with pytest.raises(AssertionError, match='Native physical history changed'):
        causal.collect_intervention(net, cfg, plan, pre+1)
    assert not net.lstm._forward_pre_hooks and not net.lstm._forward_hooks


def test_cue_baseline_effect_does_not_count_as_edit_rescue():
    def record(native_error, ordinary_error, native_success, ordinary_success):
        return dict(phase={'60': {cue:{m:dict(phase_mae_deg=value) for m in ('raw','relative_rms')}
            for cue,value in [('native',native_error),('ordinary',ordinary_error)]}},
            navigation={cue:dict(success_fraction=value,mean_goal_progress_m=value)
                        for cue,value in [('native',native_success),('ordinary',ordinary_success)]})
    results = {'raw':dict(none=record(0,4,.8,.6), plus30=record(20,24,.5,.3))}
    answer = causal.rescue_contrasts(results)['raw']['plus30']
    assert answer['late_phase_error_reduction_deg']['raw'] == 0
    assert abs(answer['incremental_native_cue_navigation_rescue']['success_fraction']) < 1e-12
    results['raw']['plus30'] = record(10,24,.7,.3)
    answer = causal.rescue_contrasts(results)['raw']['plus30']
    assert answer['late_phase_error_reduction_deg']['raw'] == 10
    assert abs(answer['incremental_native_cue_navigation_rescue']['success_fraction']-.2) < 1e-12


def test_dense_certificate_uses_actual_free_tape_indices():
    cfg=config('gc100');plan=task.TrialStreams(781929).draw(1,task.STAGES[3],cfg,headings=4)
    plan['delay'].fill_(60);plan['goal'][:]=torch.tensor([.6,0.]);plan['endpoint'].zero_()
    plan['motor'].zero_();plan['motor'][:30]=1000.  # Deliberately impossible if indexed from0.
    x=torch.zeros(1287);x[1280]=1.;first={}
    for group in range(5):
        for i in range(4):
            first[group*4+i]=dict(input=x.clone(),state=torch.zeros(512),position=torch.zeros(2),
                heading=plan['heading'][i].clone(),motors=torch.zeros(3))
    raw=dict(first_free=first,goal=torch.zeros(20,512))
    result=causal.controlled_certificate(cfg,plan,[raw])
    assert result['certified_all_pairs'] and result['equally_weighted_pairs']==2
    assert result['free_start_update']>30 and result['delay_override']==60
    start=result['free_start_update'];plan['motor'][start:start+30]=1000.
    failed=causal.controlled_certificate(cfg,plan,[raw])
    assert not failed['certified_all_pairs'] and failed['common_information_success_upper_bound']==1
