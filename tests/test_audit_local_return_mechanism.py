"""Independent auditor corruption checks; synthetic networks, no study endpoints."""
from pathlib import Path
import copy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT))
import numpy as np
import pytest
import torch
import audit_local_return_mechanism as audit
import lstm_local_return_causal as causal
from test_lstm_local_return_causal import fixture


@pytest.fixture(scope='module')
def observed():
    net,cfg,plan,pre,_,_,_=fixture()
    rows,checks,raw=causal.collect_intervention(net,cfg,plan,pre)
    rows=causal.enrich_navigation_rows(rows,raw,plan)
    cues={name:raw['cue_inputs'][i*2:(i+1)*2] for i,name in enumerate(audit.CUES[:-1])}
    return net,cfg,plan,pre,rows,raw,cues


def replay(observed,raw=None):
    net,cfg,plan,pre,rows,original,cues=observed
    return audit.audit_replay(net.state_dict(),original if raw is None else raw,plan,pre,cues,None,
        torch.zeros(512,dtype=torch.bool),'gc100')


def test_saved_native_physics_and_recurrence_reconcile(observed):
    net,cfg,plan,pre,rows,raw,cues=observed
    assert audit.audit_trajectory(plan,cfg,raw,rows)['native_free_physics_max_abs_error_m']<2e-5
    assert replay(observed)['first_FREE_actor_reconstructed']


@pytest.mark.parametrize('damage',['motor','terminal','reward','previous_action','phase_id','position'])
def test_corrupted_physics_masks_or_global_phase_rejected(observed,damage):
    _,cfg,original,_,original_rows,original_raw,_=observed
    plan=copy.deepcopy(original);rows=copy.deepcopy(original_rows);raw=copy.deepcopy(original_raw)
    free=audit.clocks(plan)[3]
    if damage=='motor':plan['motor'][free,0,0]+=100
    elif damage=='terminal':raw['terminal'][free,0]=True
    elif damage=='reward':raw['raw_reward'][0,0]=10
    elif damage=='previous_action':raw['pa'][free,0]=4
    elif damage=='phase_id':rows[0]['heading_index']=99
    else:raw['position'][free,0,0]+=.1
    with pytest.raises(AssertionError):audit.audit_trajectory(plan,cfg,raw,rows)


@pytest.mark.parametrize('damage',['cue','intended_donor','state60','actor','reset'])
def test_replay_rejects_wrong_inputs_edits_actor_and_timing(observed,damage):
    raw=copy.deepcopy(observed[5])
    if damage=='cue':raw['cue_inputs'][0,800]+=.1
    elif damage=='intended_donor':raw['intended_precue_edit'][0,0]+=.1
    elif damage=='state60':raw['snapshots'][60][0,0]+=.1
    elif damage=='actor':raw['first_free'][0]['logits'][0]+=.1
    else:raw['snapshots'][0][-1,0]=.1
    with pytest.raises(AssertionError):replay(observed,raw)


def test_reset_independence_cross_edits_and_physical_prefix(observed):
    net,cfg,plan,pre,rows,sham,cues=observed
    _,_,edited=causal.collect_intervention(net,cfg,plan,pre,pre+.4)
    free=audit.clocks(plan)[3];audit.assert_reset_and_physics(sham,edited,2,free)
    edited['snapshots'][15][-1,0]+=.1
    with pytest.raises(AssertionError):audit.assert_reset_and_physics(sham,edited,2,free)


def test_independent_edit_reconstruction_and_collapse_accounting():
    from lstm_reorientation_causal_navigation import state_edits
    theta=np.arange(24)*2*np.pi/24
    curve=np.column_stack((np.cos(theta),np.sin(theta),.3*np.cos(2*theta)))
    subset=np.arange(0,24,2);scale=np.array([1.,2.,.5])
    expected,definition=state_edits(curve,scale,subset)
    actual,record=audit.fixed_edits(curve,scale,subset)
    assert actual.keys()==expected.keys()
    for name in actual:np.testing.assert_allclose(actual[name],expected[name],rtol=0,atol=2e-16)
    audit.assert_edit_definition(definition,record)
    assert np.array_equal(actual['plus30'],curve[(subset+2)%24])
    collapsed,invalid=audit.fixed_edits(np.ones_like(curve),scale,subset)
    assert set(collapsed)=={'none'} and len(invalid['invalid'])==6


def test_phase_and_rescue_arithmetic_independent():
    from lstm_reorientation_causal_navigation import phase_metrics
    theta=np.arange(24)*2*np.pi/24;curve=np.column_stack((np.cos(theta),np.sin(theta)))
    subset=np.arange(0,24,2);points=curve[(subset+2)%24]*1.03
    expected=phase_metrics(points,curve,curve,theta,theta[subset],np.ones(2),1.)
    audit.close(audit.phase_record(points,curve,curve,theta,theta[subset],np.ones(2)),expected,atol=1e-8)
    def record(native,ordinary):return dict(phase={'60':{c:{m:dict(phase_mae_deg=v) for m in audit.METRICS}
        for c,v in [('native',native),('ordinary',ordinary)]}},navigation={c:dict(success_fraction=v,mean_goal_progress_m=v)
        for c,v in [('native',native/100),('ordinary',ordinary/100)]})
    results={'raw':dict(none=record(1,4),plus30=record(12,24))}
    report=causal.rescue_contrasts(results);audit.audit_contrasts(results,report)
    report['raw']['plus30']['late_phase_error_reduction_deg']['raw']+=1
    with pytest.raises(AssertionError):audit.audit_contrasts(results,report)


def observable_fixture():
    row=dict(full_state_dimensions=512,observable_dimensions=24,observation_offsets=[0,5,10,15],evolution_updates=30,
        observable_geometry=dict(radius=1.,max_edge_over_radius=.01,nonlocal_distance_over_radius=dict(p10=.2)),
        evolved_observable_geometry=dict(radius=1.),radial_observable_normal_gain=dict(median=.1),
        radial_over_tangent_observable_gain=dict(median=.1),invariance={},candidate_gates={},
        centered_conditional_factor_candidate=True,noncentered_observable_candidate=True)
    keys=('observable_radius_RMS_at_least1e_minus4','max_edge_below10percent_radius','nonlocal_separation_above5percent',
        'evolved_radius_retains5percent','p90_invariance_residual_below5percent','phase_coverage80percent',
        'phase_degree_one','phase_order_preserved','observable_radial_kick_nontrivial_in80percent',
        'observable_tangent_kick_nontrivial_in80percent','radial_gain_at_most_half','radial_over_tangent_at_most_half')
    for label in ('centered','noncentered'):
        row['invariance'][label]=dict(residual_to_reference_curve_over_radius=dict(p90=.01),
            phase_transport=dict(circular_coverage=1.,winding_degree=1.,backward_edge_fraction=0.))
        row['candidate_gates'][label]={k:True for k in keys}
    return row


def test_observable_thresholds_conjunction_and_explicit_unmeasured_scope():
    row=observable_fixture();result=audit.audit_observable_gates(row,24)
    assert result['independently_recalculated_thresholds_per_centering']==10 and len(result['provenance_only_flags'])==2
    row['radial_observable_normal_gain']['median']=.9
    with pytest.raises(AssertionError):audit.audit_observable_gates(row,24)
    row=observable_fixture();row['centered_conditional_factor_candidate']=False
    with pytest.raises(AssertionError):audit.audit_observable_gates(row,24)


def test_behavior_gate_uses_both_fixed_thresholds_and_certificate():
    report={'checks':{'stage4_raw_evaluation':dict(variants=dict(blind_native=dict(trials=256,successes=170),
        blind_reset_after_glimpse=dict(trials=256,successes=130)),all_physical_checks=True,
        certificate=dict(all_pairs_certified=True),gates=dict(near_baseline_pass=False,hard_numeric_pass=True,hard_certified_memory_pass=True))}}
    assert audit.gate_from_saved(report)['hard_certified_memory_pass']
    report['checks']['stage4_raw_evaluation']['variants']['blind_reset_after_glimpse']['successes']=133
    with pytest.raises(AssertionError):audit.gate_from_saved(report)


@pytest.mark.parametrize('size,baseline',[(1e-15,0.),(1e-7,1000.),(1e-4,1e6)])
def test_nearest_semantic_tiny_edges_and_large_common_baseline(size,baseline):
    from lstm_travel_cue_pulse import semantic_nearest
    theta=np.arange(24)*2*np.pi/24
    curve=baseline+size*np.column_stack((np.cos(theta),np.sin(theta),.2*np.cos(2*theta)))
    points=.37*curve[::2]+.63*curve[1::2]
    expected_distance,expected_phase=semantic_nearest(points,curve,theta)
    distance,phase=audit.nearest_semantic(points,curve,theta)
    np.testing.assert_allclose(distance,expected_distance,atol=1e-14,rtol=1e-8)
    np.testing.assert_allclose(audit.wrap(phase-expected_phase),0.,atol=1e-9)
    # In the tiny case a tempting direct minimizing-fraction interpolation is
    # substantially wrong because the frozen semantic floor is much larger.
    if size<1e-14:
        direct=theta[::2]+.63*(2*np.pi/24)
        assert np.max(np.abs(audit.wrap(phase-direct)))>.1


def test_stale_success_beside_failure_remains_unresolved(tmp_path):
    (tmp_path/'primary.json').write_text('{}')
    audit.require_clean_context(tmp_path)
    (tmp_path/'analysis_failed.json').write_text('{"error":"precision"}')
    with pytest.raises(AssertionError,match='Unresolved failure'):
        audit.require_clean_context(tmp_path)


def test_navigation_missing_phase_or_invented_success_rejected(observed):
    rows=observed[4];summary={}
    for cue,variant in zip(audit.CUES,audit.VARIANTS):
        selected=[r for r in rows if r['variant']==variant]
        summary[cue]=dict(trials=2,successes=sum(r['hit'] for r in selected),success_fraction=sum(r['hit'] for r in selected)/2,
            mean_goal_progress_m=sum(r['progress_m'] for r in selected)/2,
            mean_first_action_goalward_progress_m=sum(r['first_action_goalward_progress_m'] for r in selected)/2,
            mean_path_length_m=sum(r['free_path_m'] for r in selected)/2,rows=selected)
    audit.audit_navigation(rows,summary,[0,1])
    with pytest.raises(AssertionError):audit.audit_navigation(rows[1:],summary,[0,1])
    summary['native']['successes']+=1
    with pytest.raises(AssertionError):audit.audit_navigation(rows,summary,[0,1])


def test_primary_summary_thresholds_and_candidate_conjunction():
    from lstm_continued_heading_memory import geometry_gates
    branch=dict(invariance={'30':{}},trajectory={'30':{}})
    for metric in audit.METRICS:
        branch['trajectory']['30'][metric]=dict(radius_retention=1.)
        branch['invariance']['30'][metric]=dict(geometry=dict(max_edge_over_radius=.01,
            nonlocal_distance_over_radius=dict(p10=.2)),horizons={'30':dict(
            phase_transport=dict(circular_coverage=1.,winding_degree=1.,backward_edge_fraction=0.),
            residual_to_fixed_curve_over_radius=dict(p90=.01),radial_normal_gain=dict(median=.1),
            radial_over_tangent_gain=dict(median=.1))})
    branch['candidate_gates']=geometry_gates(branch['invariance'],branch['trajectory'])
    branch['passes_both_metric_candidate_screen']=True
    assert audit.audit_primary_gates(branch)
    branch['trajectory']['30']['raw']['radius_retention']=.001
    with pytest.raises(AssertionError):audit.audit_primary_gates(branch)
