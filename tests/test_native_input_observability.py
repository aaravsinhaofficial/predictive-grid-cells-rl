"""Synthetic recurrent systems and private native physics; no trained queries."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_input_observability as study
from test_native_turn_teacher_preflight import fixture
from test_native_turn_geometry_controls import two_section_system


def random_observer(hidden=8):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(983515)
        cell=torch.nn.LSTMCell(1287,hidden,dtype=torch.float64).eval().requires_grad_(False)
        actor=torch.nn.Linear(hidden,6,dtype=torch.float64).eval().requires_grad_(False)
        base=torch.randn(1287,dtype=torch.float64)*.03
    return study.ActionObserver(cell,actor.weight,actor.bias,study.query_bank(base))


def test_factorized_six_queries_match_independent_cells_and_truthful_action_labels():
    observer=random_observer();gen=torch.Generator().manual_seed(982515)
    z=torch.randn(17,16,generator=gen,dtype=torch.float64)*.4
    before=z.clone();q,_=observer.profiles(z)
    assert observer.precision(z)['passes']
    torch.testing.assert_close(q.reshape(17,6,6).sum(-1),torch.ones(17,6,dtype=torch.float64))
    assert torch.equal(z,before)
    assert torch.equal(observer.bank[:,1280:1286],torch.eye(6,dtype=torch.float64))
    assert torch.count_nonzero(observer.bank[:,-1])==0
    held=observer.heldout(z)
    for j,tape in enumerate(study.HELDOUT):
        h,c=z.chunk(2,1)
        for action in tape:h,c=observer.cell(observer.bank[action].expand(17,-1),(h,c))
        expected=torch.nn.functional.linear(h,observer.weight,observer.bias).softmax(1)
        np.testing.assert_allclose(held[:,j],expected.numpy(),atol=1e-14,rtol=0)


def test_probability_units_floor_and_no_phase_fabrication_by_query():
    angle=np.arange(128)*2*np.pi/128
    q=np.full((128,6),1/6);q[:,0]+=.02*np.cos(angle);q[:,1]-=.02*np.cos(angle)
    q[:,2]+=.02*np.sin(angle);q[:,3]-=.02*np.sin(angle)
    profile=np.tile(q,(1,6));radius=study.old.flow.curve_geometry(profile)['radius']
    assert abs(study.probability_rms(profile)-radius/6)<1e-15
    assert study.geometry(profile,profile)['necessary_geometry_pass']
    near=1/6+(profile-1/6)*1e-4
    assert not study.geometry(near,near)['gates']['initial_probability_RMS_at_least_1e_minus4']
    observer=random_observer();constant=np.zeros((128,16));answer=observer.profiles(constant)[0].numpy()
    assert np.array_equal(answer,answer[:1].repeat(128,0))
    assert not study.geometry(answer,answer)['necessary_geometry_pass']


def test_render_bypass_preserves_native_physics_reward_hit_and_reset():
    cfg,_,_,ctx=fixture();n=6
    fast=study.fast_env(cfg,ctx,n);normal=study.build_env(dict(cfg,img_noise=0.),n,983201,'cpu',stagger=False)
    normal.reset_all();normal.noise_v=normal.noise_r=0.
    normal.goal[:]=ctx['goal_center'];normal.land_pos[:]=ctx['land_pos'];normal.stripe_phase[:]=ctx['stripe_phase']
    for k in study.old.parent.BOOKKEEPING:getattr(normal,k)[:]=ctx['native_bookkeeping'][k]
    for env in (fast,normal):
        env.pos.zero_();env.hd.zero_();env.mdir[:]=torch.tensor([1.,0.]);study.old.parent.brake(env)
        env.goal[0]=env.pos[0];env.t[1]=env.episode_len-1
    for j in range(3):
        _,rf,df,inf=fast.step((torch.arange(n)+j)%6)
        _,rn,dn,inn=normal.step((torch.arange(n)+j)%6)
        assert torch.equal(rf,rn)and torch.equal(df,dn)
        for key in ('hit','hit_wall','hit_pos'):assert torch.equal(inf[key],inn[key])
        for key in study.SNAPSHOT_KEYS:assert torch.equal(getattr(fast,key),getattr(normal,key)),key


def test_reference_timing_actual_motor_brakes_and_query_physics():
    cfg,net,pop,ctx=fixture()
    family=study.old.prepare_family(net,pop,cfg,ctx,'probe_glimpse',1,phases=16)
    refs=study.physical_references(cfg,ctx,family);origin=torch.tensor(family['audit']['reference_episode_t'])
    assert all(torch.count_nonzero(refs['left'][0][k])==0 for k in ('vf','vs','vr'))
    assert torch.count_nonzero(family['physical']['vf'][-1])>0
    assert torch.equal(refs['left'][7]['t'],origin+7)
    assert torch.equal(refs['fb_B'][0]['t'],origin+10)
    assert torch.equal(refs['fb_F'][0]['t'],origin+11)
    assert torch.equal(refs['fb_B'][0]['pos'],ctx['endpoint'].expand(16,-1))
    assert torch.equal(refs['fb_B'][0]['hd'],refs['fb_F'][0]['hd'])
    assert (refs['fb_B'][0]['vf']<0).all()and(refs['fb_F'][0]['vf']>0).all()
    bank=study.query_bank(family['left_x'])
    audit,terminal=study.physical_tapes(cfg,ctx,refs['left'][2],bank,tuple((a,)for a in range(6)))
    assert audit['valid']and audit['measurement_updates']==1
    assert torch.equal(terminal['t'],(origin+3).repeat_interleave(6))
    invalid={k:v.clone()for k,v in refs['left'][2].items()};invalid['goal'][0]=invalid['pos'][0]
    bad,_=study.physical_tapes(cfg,ctx,invalid,bank,tuple((a,)for a in range(6)))
    assert not bad['valid']and any(r['reason']=='goal_hit'for r in bad['issues'])


class SyntheticQuery:
    """Memory is hidden during delay; one common query reveals all four coordinates."""
    def profiles(self,state):
        z=torch.as_tensor(state,dtype=torch.float64);q=torch.full((len(z),6,6),1/6,dtype=torch.float64)
        for j in range(4):q[:,j,0]+=.02*z[:,j];q[:,j,1]-=.02*z[:,j]
        return q.reshape(len(z),36),torch.zeros(len(z),36,dtype=torch.float64)


def test_hidden_circle_is_readable_and_selectively_recovers_but_plane_does_not():
    curve,_,forward,backward,_,_=two_section_system();observer=SyntheticQuery()
    transition=lambda z,t:(forward if t%2==0 else backward)(z)
    result,arrays,_=study.finite_probe(transition,curve.numpy(),np.ones(4),observer,2)
    assert study.geometry(arrays['first'][:128],arrays['last'][:128])['necessary_geometry_pass']
    assert result['physical_conclusive']and result['all_normal_directions_pass']
    assert result['response_class_counts']['observable_normal']==320
    plane,pa,_=study.finite_probe(lambda z,t:.9*z,curve.numpy(),np.ones(4),observer,2)
    assert not study.geometry(pa['first'][:128],pa['last'][:128])['necessary_geometry_pass']
    assert not plane['all_normal_directions_pass']
    erased,ea,_=study.finite_probe(lambda z,t:torch.zeros_like(z),curve.numpy(),np.ones(4),observer,2)
    assert not study.geometry(ea['first'][:128],ea['last'][:128])['necessary_geometry_pass']
    assert not erased['all_normal_directions_pass']


def test_retention_guard_uses_coordinate_precision_not_large_cell_baseline():
    observer=random_observer();original=np.zeros((8,16));original[:,8:]=1e4
    tiny=original.copy();tiny[:,0]=1e-14
    assert study.retention_guard(tiny,original,observer)['passes']
    changed=original.copy();changed[:,0]=1e-8
    with pytest.raises(AssertionError,match='retention state arithmetic'):study.retention_guard(changed,original,observer)


def test_phase_only_or_invisible_fiber_is_transverse_inconclusive():
    curve,_,forward,backward,_,_=two_section_system()
    basis=torch.linalg.svd(curve,full_matrices=False).Vh[:2].T
    class PhaseQuery:
        def profiles(self,state):
            z=torch.as_tensor(state,dtype=torch.float64);p=z@basis
            p=p/torch.linalg.vector_norm(p,dim=1,keepdim=True)
            q=torch.full((len(z),6),1/6,dtype=torch.float64)
            q[:,:2]+=.02*p;q[:,2:4]-=.02*p
            q=q.repeat(1,6)
            return q,torch.zeros_like(q)
    result,arrays,_=study.finite_probe(lambda z,t:(forward if t%2==0 else backward)(z),
        curve.numpy(),np.ones(4),PhaseQuery(),2)
    assert study.geometry(arrays['first'][:128],arrays['last'][:128])['necessary_geometry_pass']
    assert result['physical_construction_conclusive']
    assert not result['initial_observation_conclusive']
    assert result['physical_status']=='inconclusive_transverse_observability'
    assert result['response_class_counts']['invisible']+result['response_class_counts']['phase_only']>0
