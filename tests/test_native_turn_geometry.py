"""Synthetic and native-physics tests; no original policy outcomes."""
from pathlib import Path
import sys
import math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_turn_geometry as study
from test_native_turn_teacher_preflight import fixture


def test_coherent_fixed_prefixes_dense_phase_and_complete_common_inputs():
    cfg,net,pop,ctx=fixture()
    for prefix in (0,1):
        family=study.prepare_family(net,pop,cfg,ctx,'probe_glimpse',prefix)
        audit=family['audit'];x=family['native_x'];o=family['ordinary_x']
        assert len(family['target_choice_heading'])==128
        turn=[i for i,s in enumerate(audit['labels'])if s.startswith('turn_')]
        assert torch.equal(x[turn,:,1280:1286].argmax(-1),torch.full((14,128),prefix))
        assert all(study.same_rows(q)for q in (family['left_x'][None],family['fb_x'][0:1],family['fb_x'][1:2]))
        assert family['fb_x'][:,1280:1286].argmax(1).tolist()==[2,3]
        cue=audit['cue_index'];assert cue==len(x)-1 and study.same_rows(o[cue])
        mask=torch.ones(1287,dtype=torch.bool);mask[256+torch.as_tensor(pop.pgc_idx)]=False
        assert torch.equal(x[:,:,mask],o[:,:,mask])
        assert audit['choice_heading_error_max_rad']<2e-5
        assert max(p['position_error_m']for p in audit['fb_physical']if p['action']==3)<1e-6
        assert max(p['heading_error_rad']for p in audit['fb_physical'])<1e-6
        assert audit['fb_B_reference_updates_after_choice']==2 and audit['fb_F_reference_updates_after_choice']==3
        angle=family['reference_heading'];gap=study.wrap_pi(torch.roll(angle,-1)-angle)
        assert torch.max(torch.abs(gap-2*math.pi/128))<2e-5


def test_ordinary_reset_collapses_and_each_endpoint_replays_own_weights():
    cfg,net,pop,ctx=fixture()
    for kind in study.parent.KINDS:
        family=study.prepare_family(net,pop,cfg,ctx,kind,0,phases=16)
        net.double();first=study.elicit_branches(net,family,ctx['initial_hc'])
        assert np.array_equal(first['pre_cue_reset_ordinary'],np.repeat(first['pre_cue_reset_ordinary'][:1],16,axis=0))
        if kind=='turn_only':assert np.array_equal(first['native'],first['ordinary'])
        with torch.no_grad():net.lstm.bias_hh.add_(.02)
        second=study.elicit_branches(net,family,ctx['initial_hc'])
        assert not np.array_equal(first['native'],second['native'])
        # TinyNet's embedding has no parameters, so remaining family preparation
        # is unaffected by the recurrent dtype change.


def test_two_section_transition_order_is_actual_forward_then_backward():
    _,net,_,_=fixture();net.double()
    generator=torch.Generator().manual_seed(984899)
    x=torch.randn(2,1287,generator=generator,dtype=torch.float64)*.1
    z=torch.randn(4,16,generator=generator,dtype=torch.float64)*.1
    transition=study.transition_from_inputs(net,x)
    h,c=z.chunk(2,1)
    with torch.no_grad():
        h,c=net.lstm(x[0].expand(4,-1),(h,c));h,c=net.lstm(x[1].expand(4,-1),(h,c))
    actual=transition(transition(z,0),1)
    assert torch.allclose(actual,torch.cat((h,c),1),atol=2e-15,rtol=2e-15)


def test_block_purity_normals_and_initially_silent_later_visible_guard():
    theta=np.arange(128)*2*math.pi/128
    curve=np.zeros((128,6));curve[:,0]=np.cos(theta);curve[:,1]=np.sin(theta)
    curve[:,3]=.3*np.cos(theta);curve[:,4]=.3*np.sin(theta)
    starts,metadata=study.make_perturbations(curve,np.ones(6))
    directions=metadata['scaled_directions'];names=metadata['names']
    assert np.max(np.abs(metadata['normal_dot_tangent']))<1e-14
    assert np.count_nonzero(directions[names.index('hidden_only'),:,3:])==0
    assert np.count_nonzero(directions[names.index('cell_only'),:,:3])==0
    assert np.count_nonzero(directions[names.index('cell_radial'),:,:3])==0
    # A direction invisible at the first observation but visible later must not
    # become a zero-denominator apparent successful factor.
    first=starts[:,:3].copy();last=first.copy()
    index=names.index('cell_only');begin=128+index*16
    last[begin:begin+16,0]+=.1
    report,arrays=study.finite_normal_metrics(first,last,metadata)
    row=report['directions'][index]
    assert row['factor_leakage_count']==16 and not row['passes']


def test_physical_angle_transport_does_not_subtract_arclength_from_body_angle():
    theta=np.arange(128)*2*math.pi/128
    curve=np.column_stack((2*np.cos(theta),np.sin(theta),.3*np.cos(2*theta)))
    # One exact index shift is a known angle increment despite nonuniform arc.
    moved=np.roll(curve,-4,axis=0);angle=4*2*math.pi/128
    result=study.physical_phase_comparison(curve,moved,theta,np.full(128,angle),False)
    assert result['neural_minus_body_phase_change']['absolute_max_rad']<1e-12
    assert abs(result['neural_physical_label_displacement']['mean_rad']-angle)<1e-12


def test_total_observation_windows_and_periodic_sections_are_explicit():
    for window,evolution,offsets,kind in study.all_specs('left'):
        assert evolution+max(offsets)==window
        assert window in (3,8,15,30,60)
    for regime in ('fb_B','fb_F'):
        for window,evolution,offsets,kind in study.all_specs(regime):
            assert evolution%2==0 and window==evolution+max(offsets)
    # Exact zero common-state initialization cannot create a phase family.
    _,net,_,_=fixture();net.double();x=torch.zeros(2,1287,dtype=torch.float64)
    x[0,1282]=1;x[1,1283]=1
    states=study.trajectory(study.transition_from_inputs(net,x),np.zeros((128,16)),5)
    for z in states.values():assert np.array_equal(z,z[:1].repeat(128,0))


def test_staged_runner_retains_all_geometry_rows_and_full_timing(tmp_path):
    _,net,_,_=fixture();net.double();theta=np.arange(128)*2*math.pi/128
    curve=np.zeros((128,16));curve[:,0]=.2*np.cos(theta);curve[:,1]=.2*np.sin(theta)
    # A contracting state map is an explicit negative; no network weights are
    # run here. The actor is only a fixed synthetic observation.
    transition=lambda z,t:.9*z
    rows,states=study.analyze_curve(net,curve,transition,'left',theta,
        {t:np.zeros(128)for t in range(61)},tmp_path/'case')
    assert len(rows)==50
    assert set(states)=={0,1,2,3,7,8,14,15,29,30,59,60}
    assert all(not r['necessary_geometry_pass']and not r['normal_test_run']and not r['broad_candidate']for r in rows)
    assert all(r['window_updates']==r['evolution_updates']+max(r['observation_offsets'])for r in rows)
    assert (tmp_path/'case'/'result.json').exists()
