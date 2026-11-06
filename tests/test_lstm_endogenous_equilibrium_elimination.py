import torch
from scripts.lstm_endogenous_equilibrium_elimination import equilibrium_map,stable_full_jacobian
from scripts.lstm_endogenous_curve_recovery import advance


def test_elimination_recovers_known_fixed_point_and_handles_forget_rounding():
    cell=torch.nn.LSTMCell(2,256).double().requires_grad_(False)
    for p in cell.parameters():p.zero_()
    cell.bias_ih[256:512]=2.;cell.bias_ih[512:768]=.4
    drive=cell.bias_ih[None]
    h,c,d=equilibrium_map(cell,drive,torch.zeros(1,256,dtype=torch.float64))
    z=torch.cat((h,c),1)
    torch.testing.assert_close(advance(cell,drive,z),z,atol=1e-12,rtol=0)
    extreme=drive.clone();extreme[:,256:512]=45
    hh,cc,dd=equilibrium_map(cell,extreme,h)
    assert torch.sigmoid(extreme[:,256:512]).min()==1
    assert dd.min()>0 and cc.min()>1e18
    assert torch.isfinite(cc).all()


def test_stable_full_jacobian_matches_autograd_away_from_rounding():
    torch.manual_seed(718)
    cell=torch.nn.LSTMCell(2,256).double().requires_grad_(False)
    z=torch.randn(512,dtype=torch.float64)*.1
    drive=torch.randn(1,1024,dtype=torch.float64)*.1
    jac,_=stable_full_jacobian(cell,drive,z)
    expected=torch.autograd.functional.jacobian(lambda v:advance(cell,drive,v[None])[0],z,vectorize=True)
    torch.testing.assert_close(jac,expected,atol=1e-12,rtol=1e-12)
