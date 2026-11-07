"""Independent private controls; no study weights, native outcomes or cloud use."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import analyze_simple_memory_dynamics as study


def circle(n=96):
    theta=np.arange(n)*2*np.pi/n
    basis=np.array([[1.,0.],[0.,1.],[.7,.2],[-.2,.7]])
    basis=np.linalg.qr(basis)[0]
    return .4*np.column_stack((np.cos(theta),np.sin(theta)))@basis.T,basis


def ring_transition(basis,decay=.8):
    b=torch.tensor(basis,dtype=torch.float64)
    def transition(z,step=0):
        xy=z@b;r=xy.norm(dim=1,keepdim=True)
        plane=xy@b.T
        return (xy/r*(.4+decay*(r-.4)))@b.T+decay*(z-plane)
    return transition


def test_true_full_state_ring_passes_fixed_geometry_and_bounded_normals():
    torch.set_num_threads(1)
    curve,basis=circle()
    result,states,arrays=study.analyze_regime(ring_transition(basis),curve,np.ones(4),
        'constant',arc_probes=48,anchors=4)
    assert result['conditional_geometry_candidate']
    assert result['tested_normal_recovery']
    assert result['reference_neural_updates']==result['future_neural_updates']==30
    assert result['broad_normal']['raw']['normal_directions']==7
    assert result['broad_normal']['raw']['hidden_bound_excess_count']==0
    assert result['invariance']['raw']['radial_construction']['hidden_bound_excess_count']==0
    np.testing.assert_allclose(states[60],curve,atol=1e-12)
    assert arrays['broad_raw_actual_physical_kicks'].shape==(8,2,4,4)


def test_shrinking_circle_fails_despite_smaller_perturbations_and_circular_pca():
    curve,_=circle()
    result,_,_=study.analyze_regime(lambda z,k:z*.97,curve,np.ones(4),
        'constant',arc_probes=48,anchors=4)
    row=result['invariance']['raw']['horizons']['30']
    assert row['radial_normal_gain']['median']<.5
    assert row['radial_over_tangent_gain']['median']>.99
    assert row['residual_to_fixed_curve_over_radius']['p90']>.5
    assert not result['conditional_geometry_candidate']
    assert not result['tested_normal_recovery']


def test_ring_near_hidden_bound_cannot_pass_using_nonphysical_radial_kicks():
    curve,basis=circle();center=np.zeros(4);center[0]=.99999-curve[:,0].max()
    shifted=curve+center;radial=ring_transition(basis)
    def transition(z,k):return radial(z-torch.as_tensor(center))+torch.as_tensor(center)
    result,_,_=study.analyze_regime(transition,shifted,np.ones(4),'constant',arc_probes=48,anchors=4)
    assert np.abs(shifted[:,:2]).max()<1
    assert result['invariance']['raw']['radial_construction']['hidden_bound_excess_count']>0
    assert not result['gates']['raw']['radial_starts_within_hidden_bounds']
    assert not result['conditional_geometry_candidate']


def test_forward_backward_two_sections_use_15_cycles_and_never_pool_sections():
    curve,basis=circle();radial=ring_transition(basis,.6)
    matrix=torch.tensor([[1.1,.2,0.,0.],[0.,.9,0.,0.],[0.,0.,1.,.1],[0.,0.,0.,1.]],dtype=torch.float64)
    inverse=torch.linalg.inv(matrix);shift=torch.tensor([.08,0.,0.,0.],dtype=torch.float64)
    def transition(z,k):
        return radial(z)@matrix.T+shift if k%2==0 else radial((z-shift)@inverse.T)
    result,states,_=study.analyze_regime(transition,curve,np.ones(4),
        'forward_backward',arc_probes=48,anchors=4)
    assert result['conditional_geometry_candidate'] and result['tested_normal_recovery']
    assert result['reference_map_steps']==result['future_map_steps']==15
    assert result['reference_neural_updates']+result['future_neural_updates']==60
    assert set(result['invariance']['raw']['horizons'])=={'1','7','15'}
    assert not np.allclose(states[1],states[2])
    np.testing.assert_allclose(states[30],curve,atol=1e-12)
    np.testing.assert_allclose(states[60],curve,atol=1e-12)


def test_period_two_point_and_collapsed_family_do_not_pass():
    curve,_=circle();anchor=torch.as_tensor(curve[:1])
    def point(z,k): return (anchor+(.05 if k%2==0 else 0)).expand_as(z)
    result,_,_=study.analyze_regime(point,curve,np.ones(4),'forward_backward',arc_probes=48,anchors=4)
    assert not result['conditional_geometry_candidate']
    assert not result['tested_normal_recovery']
    assert not result['gates']['raw']['reference_radius_RMS_at_least_1e_minus4']


def test_complete_input_laws_keep_goal_vision_reward_fixed_and_reject_start_token():
    x=np.zeros((4,1287),np.float32)
    x[:,:256]=.17;x[:,768:1280]=.31;x[:,1284]=1
    laws=study.input_laws(x)
    assert laws['forward_backward'][:,1280:1286].argmax(1).tolist()==[2,3]
    np.testing.assert_array_equal(laws['forward_backward'][:,:1280],np.repeat(x[:1,:1280],2,0))
    assert not laws['forward_backward'][:,-1].any()
    broken=x.copy();broken[1,800]+=1
    with pytest.raises(ValueError,match='identical'):study.input_laws(broken)
    broken=x.copy();broken[:,1280:1286]=0
    with pytest.raises(ValueError,match='PA'):study.input_laws(broken)
    broken=x.copy();broken[:,300]=1
    with pytest.raises(ValueError,match='zero'):study.input_laws(broken)


def test_frozen_transition_matches_native_lstm_and_preserves_source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(741099)
        cell=torch.nn.LSTMCell(5,2)
        x=torch.randn(2,5);z=torch.randn(7,4)
    before={k:v.clone()for k,v in cell.state_dict().items()}
    native=study.FrozenTransition(cell,x,native_float32=True)
    double=study.FrozenTransition(cell,x)
    for k in (0,1):
        expected=torch.cat(cell(x[k:k+1].expand(7,-1),z.chunk(2,1)),1)
        assert torch.equal(native(z.double(),k),expected.double())
        torch.testing.assert_close(double(z.double(),k),expected.double(),rtol=1e-5,atol=1e-7)
    assert all(torch.equal(before[k],v)for k,v in cell.state_dict().items())
    assert cell.training and cell.weight_ih.dtype==torch.float32


def test_local_cycle_jacobian_uses_correct_chain_and_spectra_distinguish_non_normality():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(741098)
        cell=torch.nn.LSTMCell(5,2);x=torch.randn(2,5)
    curve,_=circle();transition=study.FrozenTransition(cell,x)
    states=study.evolve(transition,curve,30)
    report,archive=study.local_spectra(transition,states,np.array([.5,.5,2.,2.]),'forward_backward')
    assert report['30']['autograd_max_absolute_error']<1e-10
    z=torch.tensor(states[30][0],dtype=torch.float64,requires_grad=True)
    def actual_two_step(v):
        a=torch.cat(transition.cell(transition.inputs[:1],v[None].chunk(2,1)),1)
        return torch.cat(transition.cell(transition.inputs[1:],a.chunk(2,1)),1)[0]
    jac=torch.autograd.functional.jacobian(actual_two_step,z).numpy()
    expected=np.linalg.eigvals(jac)
    observed=archive['t30_eigenvalues']
    assert np.abs(expected[:,None]-observed[None]).min(1).max()<1e-12
    assert report['30']['relative_rms']['coordinate_similarity_eigenvalue_set_error']<1e-10
    nonnormal,_=study.matrix_spectra(np.array([[.9,20.],[0.,.8]]),np.array([1.,0.]),np.array([2.,.5]))
    assert nonnormal['raw']['spectral_radius']==.9
    assert nonnormal['raw']['largest_singular_value']>20
    assert nonnormal['relative_rms']['largest_singular_value']<nonnormal['raw']['largest_singular_value']


def test_analytic_full_state_jvp_reports_transverse_and_phase_separately():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(741097)
        cell=torch.nn.LSTMCell(5,2);x=torch.randn(1,5)
    curve,_=circle();transition=study.FrozenTransition(cell,x)
    result=study.jvp_check(transition,curve,np.ones(4),'constant',anchors=4)
    assert len(result['directions'])==7
    assert result['finite_difference_max_absolute_error']<1e-8
    assert all('induced_phase_per_unit_tangent'in r and 'transverse_gain'in r for r in result['directions'])


def occupancy_fixture():
    n=72;curve,_=circle(n);angles=np.arange(n)*2*np.pi/n;ids=np.arange(0,n,2)
    phase=np.r_[0,angles[ids]];size=len(phase)
    direction=np.column_stack((np.cos(phase),np.sin(phase))).astype(np.float32)
    data=dict(episode_id=np.zeros(size,int),known=np.ones(size,bool),blind=np.r_[False,np.ones(size-1,bool)],
        global_step=np.r_[99,np.arange(200,200+size-1)],phase_age=np.ones(size,int),
        goal_pos=np.repeat([[.8,.8]],size,0),prev_action=np.full(size,2),prev_reward=np.zeros(size),
        pos=np.zeros((size,2)),velocity=direction*.1,travel_dir=direction,body_heading=phase,
        h=np.r_[curve[:1,:2],curve[ids,:2]],c=np.r_[curve[:1,2:],curve[ids,2:]])
    for key in ('goal_pickup_pos','goal_pickup_dir','goal_pickup_vel'):
        data[key]=np.repeat([[.4,.3]],size,0)
    context=dict(selected_slot=0,episode_id=0,global_step=100,runtime=dict(
        obs=dict(pos=torch.zeros(1,2),vel=torch.tensor([[.1,0.]]),dir=torch.tensor([[1.,0.]]),hd=torch.zeros(1)),
        environment_tensors=dict(goal=torch.tensor([[.8,.8]],dtype=torch.float64)),pa=torch.tensor([2])))
    return data,context,{t:curve.copy()for t in range(61)},np.ones(4),angles


def test_native_occupancy_requires_later_support_and_exact_goal_provenance():
    data,context,states,scale,theta=occupancy_fixture()
    result,arrays=study.native_occupancy(data,context,states,scale,theta)
    assert result['qualified'] and result['broad_native_support']
    assert result['compatible_rows']==36
    assert 0 not in arrays['row_indices']
    data['goal_pickup_pos'][1:,0]+=.001
    rejected,_=study.native_occupancy(data,context,states,scale,theta)
    assert not rejected['qualified'] and not rejected['broad_native_support']
    assert rejected['compatible_rows']==0
    assert 'causal effects'in rejected['scope']


def test_small_phase_coverage_is_unqualified_even_when_every_state_is_on_family():
    data,context,states,scale,theta=occupancy_fixture()
    data['travel_dir'][:]=[1,0];data['body_heading'][:]=0
    data['h'][:]=states[1][0,:2];data['c'][:]=states[1][0,2:]
    result,_=study.native_occupancy(data,context,states,scale,theta)
    assert result['compatible_rows']==36
    assert result['physical_phase_coverage']==0
    assert not result['qualified'] and not result['broad_native_support']


def test_unqualified_family_is_retained_without_search_and_cli_rejects_draft(tmp_path,monkeypatch):
    result,arrays=study.analyze_family(None,{'metadata':{'qualified':False,'reason':'prefix mismatch'}})
    assert not result['qualified'] and arrays=={}
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'status':'draft'}))
    monkeypatch.setattr(study,'PROTOCOL',protocol)
    with pytest.raises(RuntimeError,match='frozen'):
        study.main(['--run',str(tmp_path/'nonexistent'),'--states',str(tmp_path/'none'),'--out',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


def test_all_modules_integrate_on_small_private_lstm_without_mutating_weights():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(741096)
        cell=torch.nn.LSTMCell(1287,2)
    policy=SimpleNamespace(lstm=cell,n_code=512,n_actions=6)
    before={k:v.clone()for k,v in cell.state_dict().items()}
    curve,_=circle();inputs=np.zeros((96,1287),np.float32);inputs[:,:256]=.1;inputs[:,1282]=1
    family=dict(metadata={'qualified':True},states=curve.astype(np.float32),
        scale=np.ones(4,np.float32),theta=np.arange(96)*2*np.pi/96,dark_inputs=inputs)
    report,arrays=study.analyze_family(policy,family,arc_probes=48,anchors=4)
    assert report['qualified'] and set(report['regimes'])==set(study.REGIMES)
    assert all('local_spectra'in r and 'jvp'in r and 'native_float32_comparison'in r for r in report['regimes'].values())
    assert not any(r['conditional_ring_candidate']for r in report['regimes'].values())
    assert len(arrays)>20
    json.dumps(study.json_value(report),allow_nan=False)
    assert all(torch.equal(before[k],v)for k,v in cell.state_dict().items())


def test_native_256_unit_family_adapter_connects_to_both_dynamics_regimes(tmp_path):
    from tests.test_simple_memory_family import fixture
    from scripts.simple_memory_family import build_family
    cfg,policy,context=fixture(tmp_path)
    family=build_family(policy,cfg,context,'cpu',phases=12)
    report,arrays=study.analyze_family(policy,family,arc_probes=12,anchors=4,spectra=False)
    assert report['qualified'] and family['states'].shape==(12,512)
    assert set(report['regimes'])=={'constant','forward_backward'}
    assert arrays['constant_z_0'].shape==arrays['forward_backward_z_60'].shape==(12,512)
    assert all(r['reference_neural_updates']==r['future_neural_updates']==30 for r in report['regimes'].values())
    json.dumps(study.json_value(report),allow_nan=False)
