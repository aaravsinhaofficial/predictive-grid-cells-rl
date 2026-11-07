"""Saved-array corruption guards; no trained checkpoint or native outcomes."""
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import native_turn_geometry_resolution as study
import report_native_turn_geometry_resolution as report
from test_native_turn_geometry_controls import two_section_system


def circle(n=1024):
    t=np.arange(n)*2*np.pi/n
    return np.column_stack((.3*np.cos(t),.4*np.sin(t),2+.7*np.cos(t),3+.5*np.sin(t)))


def row(regime='fb_B',metric='raw',spacing=False,other=True,alias=None):
    return dict(context=0,prefix=0,kind='probe_glimpse',branch='native',regime=regime,
        window_updates=3,evolution_updates=2,observable='all_hidden',metric=metric,centered=True,
        gates={report.SPACING:spacing,'other':other},family='family',exact_sham_of=alias)


def test_independent_selection_keeps_other_metric_and_exact_exclusions():
    rows=[row(),row(metric='h_c_RMS',other=False),row(regime='fb_F',other=False),row(regime='fb_F',metric='h_c_RMS',other=False)]
    selected,cohort=report.selection(rows)
    assert selected==rows[:2]and cohort['spacing_only_rows']==1 and cohort['mandatory_metric_control_rows']==1
    assert cohort==study.eligibility(rows)[1]
    assert cohort['excluded_rows']==2


def test_hidden_and_angular_errors_cannot_hide_behind_common_cell_offset():
    original=circle(128);original[:,2:]+=1e8
    good=report.state_guard(original,original)
    assert good['passes'];report.compare(good,study.shared_state_agreement(original,original))
    for column in (0,2):
        changed=original.copy();changed[:,column]+=.001*np.cos(np.arange(128)*2*np.pi/128)
        guard=report.state_guard(changed,original)
        assert not guard['passes']
        report.compare(guard,study.shared_state_agreement(changed,original))


def test_only_visual_shared_inputs_receive_a_numerical_tolerance():
    x=torch.zeros((2,1024,1287));phase=torch.arange(1024,dtype=torch.float32)
    dense=dict(native_x=x,ordinary_x=x.clone(),target_choice_heading=phase,choice_heading=phase,reference_heading=phase,
        left_x=torch.zeros(1287),fb_x=torch.zeros((2,1287)),physical={'hd':phase[None]},audit={'phases':1024,'labels':['a','b']})
    coarse={k:(v[:,::8].clone()if k in ('native_x','ordinary_x')else v[::8].clone()if 'heading'in k else v.clone())for k,v in dense.items()if isinstance(v,torch.Tensor)}
    coarse.update(physical={'hd':phase[None,::8].clone()},audit={'phases':128,'labels':['a','b']})
    dense['native_x'][0,0,0]=5e-6
    check=report.input_guard(dense,coarse);assert check['passes'];report.compare(check,study.shared_input_audit(dense,coarse))
    dense['native_x'][0,0,256]=1e-8
    check=report.input_guard(dense,coarse);assert not check['passes'];report.compare(check,study.shared_input_audit(dense,coarse))


@pytest.mark.parametrize('center',[False,True])
def test_new_midpoint_wiggle_is_rejected_even_with_exact_shared_states(center):
    fine=circle();coarse=fine[::2]
    result=report.polygon_convergence(coarse,coarse,fine,fine,center)
    assert result['passes'];report.compare(result,study.convergence(coarse,coarse,fine,fine,center))
    corrupt=fine.copy();corrupt[1::2,0]+=.1
    for a,b in [(corrupt,fine),(fine,corrupt)]:
        result=report.polygon_convergence(coarse,coarse,a,b,center)
        assert not result['passes'];report.compare(result,study.convergence(coarse,coarse,a,b,center))


def test_exact_alias_cannot_copy_wrong_periodic_section():
    original=dict(row(),source_family='source',source_exact_sham_of=None,source_key=[],densities={'payload':'B'},source_gates={})
    alias=dict(original,source_family='alias',source_exact_sham_of='source')
    report.audit_alias(alias,original)
    wrong=dict(original,regime='fb_F',densities={'payload':'F'})
    with pytest.raises(AssertionError):report.audit_alias(alias,wrong)


@pytest.mark.parametrize('metric',['raw','h_c_RMS'])
def test_seeded_normals_and_both_densities_use_same_full_circle_anchors(metric):
    reference=circle()
    report.compare(report.tangent_check(reference,metric),study.tangent_agreement(reference,metric))
    for density in (512,1024):
        curve=reference[::1024//density]
        actual,meta=report.nominal_design(curve,metric)
        expected,old=study.old.make_perturbations(curve,report.scales(curve,metric),8)
        np.testing.assert_array_equal(actual,expected)
        for key,value in meta.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,old[key])
            else:report.compare(value,old[key])
        np.testing.assert_array_equal(curve[meta['anchor_indices']],reference[np.arange(8)*128])


def test_complete_probe_audit_rejects_seed_direction_or_anchor_corruption(tmp_path):
    curve,_,forward,backward,_,_=two_section_system()
    curve=curve.numpy();transition=lambda z,t:(forward,backward)[t%2](z)
    row={'observable':'full_state','metric':'raw'}
    for method in (None,'shared','clip_equalize'):
        probe,arrays=study.old.finite_probe(transition,curve,np.ones(4),lambda z:z,(0,),2,bounded=method)
        name=f'{method}.npz';np.savez_compressed(tmp_path/name,**arrays)
        out=report.audit_probe(tmp_path,probe,name,curve,'raw',arrays['first'][:len(curve)],arrays['last'][:len(curve)],{},row)
        assert out['physical_status']=='pass'
        corrupt={k:v.copy()for k,v in arrays.items()};corrupt['scaled_directions'][0,0,0]+=.1
        np.savez_compressed(tmp_path/'bad.npz',**corrupt)
        with pytest.raises(AssertionError):report.audit_probe(tmp_path,probe,'bad.npz',curve,'raw',arrays['first'][:len(curve)],arrays['last'][:len(curve)],{},row)
        bad=copy.deepcopy(probe);bad['perturbation_design']['anchor_indices'][1]+=1
        with pytest.raises(AssertionError):report.audit_probe(tmp_path,bad,name,curve,'raw',arrays['first'][:len(curve)],arrays['last'][:len(curve)],{},row)


def test_sampling_failure_is_distinct_from_geometry_or_recovery_failure():
    def fixture():return dict(convergence={'passes':True},densities={n:{'gates':{report.SPACING:True}}for n in ('512','1024')},
        resolved_geometry=True,physical_status='pass',broad_candidate=True)
    good=fixture();assert report.pair_status(good,good)=='screening_candidate'
    sample=fixture();sample['convergence']['passes']=False
    assert report.pair_status(good,sample)=='sampling_unresolved'
    sample=fixture();sample['resolved_geometry']=False
    assert report.pair_status(good,sample)=='geometry_failed'
    sample=fixture();sample['physical_status']='inconclusive';sample['broad_candidate']=False
    assert report.pair_status(good,sample)=='physical_test_inconclusive'
    sample=fixture();sample['physical_status']='failed_recovery';sample['broad_candidate']=False
    assert report.pair_status(good,sample)=='recovery_failed'


def test_physical_phase_transport_and_actual_motor_timing_match():
    first=circle(128);headings=np.arange(128)*2*np.pi/128;last=np.roll(first,-5,axis=0)
    for center in (False,True):
        actual=report.phase_audit(first,last,headings,np.full(128,.2),center)
        expected=study.old.physical_phase_comparison(first,last,headings,np.full(128,.2),center)
        report.compare(actual,expected)
    cfg={'action_repeat':4,'rot_max':2.4}
    family={'audit':{'left_physical':[]}}
    for t in (2,7,14,29,59):
        angle=study.old.parent.native_motion(cfg,[0]*min(t,8),2)+study.old.parent.native_motion(cfg,[0]*max(t-8,0),2)
        np.testing.assert_allclose(report.physical_rotation(family,cfg,dict(regime='left',evolution_updates=t)),angle,rtol=2e-15,atol=2e-15)
