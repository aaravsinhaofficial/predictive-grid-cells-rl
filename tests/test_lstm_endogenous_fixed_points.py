import numpy as np
import torch
from scripts.lstm_endogenous_fixed_points import minimize_residual,clusters


def test_residual_search_preserves_ring_and_collapses_point_without_geometry_loss():
    theta=torch.arange(16,dtype=torch.float64)*2*np.pi/16
    start=torch.stack((1.2*theta.cos(),1.2*theta.sin(),torch.full_like(theta,.1)),1)
    def ring(x):
        radius=x[:,:2].norm(dim=1,keepdim=True)
        return torch.cat((x[:,:2]*(.8+.2/radius),x[:,2:]*.8),1)
    points,report=minimize_residual(ring,start,adam_steps=150,lbfgs_steps=60)
    assert max(report['residual_rms'])<1e-8
    torch.testing.assert_close(points[:,:2].norm(dim=1),torch.ones(16,dtype=torch.float64),atol=1e-7,rtol=0)
    assert len(np.unique(clusters(points.numpy(),1e-3)))==16
    collapsed,report=minimize_residual(lambda x:.8*x,start,adam_steps=150,lbfgs_steps=60)
    assert max(report['residual_rms'])<1e-8
    assert len(np.unique(clusters(collapsed.numpy(),1e-3)))==1


def test_full_state_clusters_do_not_silently_discard_output_null_dimensions():
    points=np.zeros((10,5));points[:,4]=np.arange(10)
    assert len(np.unique(clusters(points,1e-3)))==10
    assert len(np.unique(clusters(points[:,:2],1e-3)))==1
