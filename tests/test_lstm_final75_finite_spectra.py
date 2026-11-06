"""Spectral-product and angular-normal tests independent of final outcomes."""
import numpy as np
import torch
from scripts.lstm_final75_finite_spectra import (
    compose,periodic_tangent,metric_spectrum,validate_product,stable_full_jacobian,advance)


def test_noncommuting_product_uses_actual_temporal_order():
    a=torch.tensor([[1.,2.],[0.,1.]],dtype=torch.float64)
    b=torch.tensor([[1.,0.],[3.,1.]],dtype=torch.float64)
    v=torch.tensor([.2,-.7],dtype=torch.float64)
    product=compose([a,b,a])
    torch.testing.assert_close(product@v,a@(b@(a@v)),atol=1e-15,rtol=1e-15)
    assert not torch.allclose(compose([a,b]),a@b)


def test_known_slow_and_contracting_modes_and_exact_normal_projection():
    angle=.37;q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    tangent,radial=q[:,0],q[:,1]
    matrix=q@np.diag([1.,.8**30])@q.T
    summary,arrays=metric_spectrum(matrix,tangent,tangent,radial,np.ones(2),np.ones((1,1)))
    np.testing.assert_allclose(np.sort(np.abs(arrays['eigenvalues'])),[.8**30,1.],atol=1e-14)
    np.testing.assert_allclose(arrays['singular_values'],[1.,.8**30],atol=1e-14)
    assert abs(summary['maximum_transverse_gain']-.8**30)<1e-14
    assert abs(summary['tangent_total_gain']-1)<1e-14
    assert abs(summary['tangent_final_alignment']-1)<1e-14
    assert abs(summary['radial_normal_gain']-.8**30)<1e-14


def test_both_endpoint_metric_transform_and_different_tangents():
    matrix=np.array([[.9,.7],[-.2,.5]])
    scale=np.array([.4,7.]);t0=np.array([1.,.3]);t1=np.array([.2,1.]);radial=np.array([-.3,1.])
    summary,arrays=metric_spectrum(matrix,t0,t1,radial,scale,np.ones((1,1)))
    m=np.diag(1/scale)@matrix@np.diag(scale)
    u0=t0/scale;u0/=np.linalg.norm(u0);u1=t1/scale;u1/=np.linalg.norm(u1)
    normal=(np.eye(2)-np.outer(u1,u1))@m@(np.eye(2)-np.outer(u0,u0))
    np.testing.assert_allclose(summary['maximum_transverse_gain'],np.linalg.svd(normal,compute_uv=False)[0],rtol=1e-13)
    np.testing.assert_allclose(np.sort_complex(arrays['eigenvalues']),np.sort_complex(np.linalg.eigvals(matrix)),atol=1e-13)
    assert summary['tangent_final_alignment']<1


def test_periodic_derivative_uses_actual_unequal_angular_spacing():
    base=np.linspace(-np.pi,np.pi,768,endpoint=False);theta=base+.003*np.sin(3*base)
    curve=np.column_stack([np.cos(theta),np.sin(theta)])
    expected=np.column_stack([-np.sin(theta),np.cos(theta)])
    derivative=periodic_tangent(curve,theta)
    np.testing.assert_allclose(derivative,expected,atol=1.2e-5,rtol=0)


def test_analytic_full_lstm_jacobian_product_matches_native_autograd():
    torch.set_num_threads(2);torch.manual_seed(7101)
    cell=torch.nn.LSTMCell(3,256).double().requires_grad_(False)
    x=torch.randn(1,3,dtype=torch.float64);z=torch.randn(512,dtype=torch.float64)
    drive=torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih)
    jac,_=stable_full_jacobian(cell,drive,z)
    def native(state):
        h,c=cell(x,(state[:256][None],state[256:][None]))
        return torch.cat((h,c),1)[0]
    auto=torch.autograd.functional.jacobian(native,z,vectorize=True)
    torch.testing.assert_close(jac,auto,atol=1e-14,rtol=1e-12)
    matrices=[];state=z.clone()
    for _ in range(3):
        matrix,_=stable_full_jacobian(cell,drive,state);matrices.append(matrix)
        state=advance(cell,drive,state[None])[0]
    result=validate_product(cell,drive,z,compose(matrices),3)
    assert result['autograd_jvp_relative_error']<1e-12
    assert result['centered_difference_relative_error']<1e-8
