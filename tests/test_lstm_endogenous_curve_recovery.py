import numpy as np
import torch
from scripts.lstm_endogenous_curve_recovery import advance, decomposition


def test_exact_lstm_differential_including_input_family():
    torch.manual_seed(173)
    cell=torch.nn.LSTMCell(11,7).double()
    x=torch.randn(3,11,dtype=torch.float64);z=torch.randn(3,14,dtype=torch.float64)
    dx=torch.randn_like(x);dz=torch.randn_like(z)
    drive=torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih)
    drive_theta=torch.nn.functional.linear(dx,cell.weight_ih,None)
    actual,dactual=advance(cell,drive,z,dz[:,:,None],drive_theta)
    def f(xx,zz):
        h,c=cell(xx,(zz[:,:7],zz[:,7:]));return torch.cat((h,c),1)
    expected,dexpected=torch.autograd.functional.jvp(f,(x,z),(dx,dz))
    torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-12)
    torch.testing.assert_close(dactual[:,:,0],dexpected,atol=1e-13,rtol=1e-12)


def test_sheared_ring_normal_recovers_despite_pointwise_error():
    # Local Jacobian of r'=1+.8(r-1), theta'=theta+.6(r-1).
    jac=torch.tensor([[.8,0.],[.6,1.]],dtype=torch.float64)
    tangent=torch.tensor([[0.,1.]],dtype=torch.float64)
    normal=torch.tensor([[1.,0.]],dtype=torch.float64)
    for _ in range(15):normal=normal@jac.T
    phase,residual=decomposition(normal,tangent)
    assert float(normal.norm())>2.8  # old pointwise metric falsely suggests failure
    assert np.isclose(float(residual),.8**15)
    assert np.isclose(float(phase),.6*(1-.8**15)/(1-.8))


def test_corrected_projection_does_not_hide_expanding_transverse_mode():
    tangent=torch.tensor([[0.,1.]],dtype=torch.float64)
    vector=torch.tensor([[1.1**15,4.]],dtype=torch.float64)
    phase,residual=decomposition(vector,tangent)
    assert float(residual)>4 and float(phase)==4
