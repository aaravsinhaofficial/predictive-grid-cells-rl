"""Scientific safeguards for the original-agent goal-family assay."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('goal_manifold',ROOT/'scripts/lstm_goal_bearing_manifold.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_snapshot_is_valid_inward_pickup_and_preserves_ordinary_channels():
    cfg,_,_,pop=m.load_model('pgc33',1,'cpu')
    theta=np.arange(192)*2*np.pi/192
    for offset in m.OFFSETS:
        center,pickup,arrival,ahead=m.snapshot_geometry([.30,-.30],theta,offset)
        assert np.max(np.abs(pickup))<1.17
        assert np.max(np.abs(center))<1.17
        assert np.max(np.abs(ahead))<1.17
        np.testing.assert_allclose(np.linalg.norm(pickup-center,axis=1),.29,atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(ahead-center,axis=1),.09,atol=1e-14)
        np.testing.assert_allclose(center-pickup,.29*arrival,atol=1e-14)
        pos=torch.tensor(pickup,dtype=torch.float32);direction=torch.tensor(arrival,dtype=torch.float32)
        native=pop.activity(pos,direction,direction*.35)
        ordinary=pop.activity(pos,direction,direction*.35,shift_scale=0)
        assert torch.equal(native[:,~pop.is_pgc],ordinary[:,~pop.is_pgc])
        assert not torch.equal(native[:,pop.is_pgc],ordinary[:,pop.is_pgc])
    # The offset contexts separate observer bearings of pickup and predicted point.
    center,pickup,arrival,ahead=m.snapshot_geometry([0.,0.],theta,np.pi/2)
    bearing=lambda p:np.arctan2(p[:,1],p[:,0])
    assert np.rad2deg(m.angular_error(bearing(pickup),bearing(ahead))).mean()>10


def test_normal_controls_match_state_swap_magnitude_and_are_locally_orthogonal():
    rng=np.random.default_rng(17);theta=np.arange(192)*2*np.pi/192
    basis=np.linalg.qr(rng.normal(size=(512,2)))[0]
    z=np.stack((np.cos(theta),np.sin(theta)),1)@basis.T
    perturbed,tangent,normal=m.perturb_families(z,42)
    np.testing.assert_allclose(np.sum(tangent*normal,axis=1),0,atol=1e-14)
    for i,j in [(1,3),(2,4)]:
        np.testing.assert_allclose(np.linalg.norm(perturbed[i]-perturbed[0],axis=1),np.linalg.norm(perturbed[j]-perturbed[0],axis=1),rtol=3e-6)


def test_phase_preservation_does_not_disguise_a_contracting_point_as_ring():
    theta=np.arange(192)*2*np.pi/192
    z=np.zeros((192,512));z[:,:2]=np.stack((np.cos(theta),np.sin(theta)),1)
    decoder=m.fit_phase_readout(z,theta)
    radius=m.family_geometry(z)['radius']
    ring=z.copy();point=z*.8**30
    # Both preserve angle mathematically; only the ring preserves state separation.
    np.testing.assert_allclose(np.arctan2(point[:,1],point[:,0]),np.arctan2(ring[:,1],ring[:,0]),atol=1e-14)
    assert m.family_geometry(ring)['radius']/radius==1
    assert m.family_geometry(point)['radius']/radius<.002
    assert m.phase_metrics(point,theta,decoder)['heldout_valid_fraction']==0
    assert m.phase_metrics(ring,theta,decoder)['heldout_mae_deg']<1e-10


def test_replay_goal_substitution_changes_only_saved_goal_slice():
    cfg,_,net,pop=m.load_model('pgc33',1,'cpu')
    generator=torch.Generator().manual_seed(17)
    x=torch.randn(12,1287,generator=generator)
    state=torch.randn(12,512,generator=generator)
    native=m.step_full(net,x,state)
    direct=net.lstm(x,(state[:,:256],state[:,256:]))
    assert torch.equal(native,torch.cat(direct,1))
    altered=x.clone();altered[:,768:1280]=0
    assert torch.equal(altered[:,:768],x[:,:768])
    assert torch.equal(altered[:,1280:],x[:,1280:])
