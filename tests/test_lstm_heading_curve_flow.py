import numpy as np
import torch

from scripts.lstm_heading_curve_flow import (
    curve_geometry,invariance_probe,nearest_closed_segments,uniform_arc_samples,
)


def circle(n=192):
    theta=np.arange(n)*2*np.pi/n
    return np.column_stack((np.cos(theta),np.sin(theta),np.zeros(n)))


def test_full_state_nearest_segment_keeps_out_of_plane_distance():
    curve=circle();query=uniform_arc_samples(curve,64);query[:,2]=.3
    distance,phase,closest,_=nearest_closed_segments(query,curve)
    np.testing.assert_allclose(distance,.3,atol=1e-12)
    np.testing.assert_allclose(closest[:,2],0,atol=0)
    assert np.unique(np.floor(phase*32)).size>=31


def test_true_rotating_ring_has_invariance_and_radial_recovery():
    def transition(z):
        r=z[:,:2].norm(dim=1,keepdim=True)
        xy=z[:,:2]/r*(1+.7*(r-1))
        angle=.02;c=np.cos(angle);s=np.sin(angle)
        rotated=torch.stack((c*xy[:,0]-s*xy[:,1],s*xy[:,0]+c*xy[:,1]),1)
        return torch.cat((rotated,.7*z[:,2:]),1)
    out=invariance_probe(None,None,circle(),np.ones(3),96,transition)
    row=out['horizons']['75']
    assert row['residual_to_fixed_curve_over_radius']['p90']<.001
    assert row['radial_over_tangent_gain']['median']<.01
    assert abs(row['phase_transport']['winding_degree']-1)<1e-10
    assert row['phase_transport']['circular_coverage']>.98


def test_shrinking_linear_plane_is_not_radially_attracting_ring():
    def transition(z):
        return z*torch.tensor([.99,.99,.2],dtype=z.dtype)
    out=invariance_probe(None,None,circle(),np.ones(3),96,transition)
    row=out['horizons']['75']
    assert row['radial_normal_gain']['median']<.5
    assert .99<row['radial_over_tangent_gain']['median']<1.01
    assert row['residual_to_fixed_curve_over_radius']['p90']>.5


def test_closed_folded_line_fails_nonlocal_separation():
    curve=circle();curve[:,1]=0
    geometry=curve_geometry(curve)
    assert geometry['nonlocal_distance_over_radius']['p10']<1e-7


def test_collapsed_point_geometry_remains_finite():
    result=curve_geometry(np.ones((96,5)))
    assert result['radius']==0
    assert result['nonlocal_distance_over_radius']['median']==0
