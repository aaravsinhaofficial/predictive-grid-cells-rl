import numpy as np
import torch
from scripts.native_lstm_equilibrium_branches import EliminatedLSTM, deflated_residual, qualify, solve_start, initializers


def test_exact_hidden_derivative_and_full_state_derivative():
    rng=np.random.default_rng(31);n=7
    model=EliminatedLSTM(rng.normal(size=(4*n,n))*.2,rng.normal(size=4*n)*.2)
    h=rng.normal(size=n)*.1
    residual,jac,c,_=model.eliminated(h)
    eps=1e-6;eye=np.eye(n)
    numeric=np.column_stack([(model.eliminated(h+eps*v)[0]-model.eliminated(h-eps*v)[0])/(2*eps) for v in eye])
    np.testing.assert_allclose(jac,numeric,atol=1e-9,rtol=1e-7)
    z=np.r_[h,c];eye=np.eye(2*n)
    full=np.column_stack([(model.advance(z+eps*v)-model.advance(z-eps*v))/(2*eps) for v in eye])
    np.testing.assert_allclose(model.full_jacobian(z),full,atol=1e-9,rtol=1e-7)
    cell=torch.nn.LSTMCell(1,n).double()
    with torch.no_grad():
        cell.weight_ih.zero_();cell.bias_hh.zero_()
        cell.weight_hh.copy_(torch.tensor(model.w));cell.bias_ih.copy_(torch.tensor(model.drive))
        hh,cc=cell(torch.zeros(1,1,dtype=torch.float64),(torch.tensor(h[None]),torch.tensor(c[None])))
    np.testing.assert_allclose(model.advance(z),np.r_[hh.numpy()[0],cc.numpy()[0]],atol=1e-12,rtol=1e-12)


def test_saturated_forget_gate_still_has_stable_equilibrium_residual():
    n=4;drive=np.zeros(4*n);drive[n:2*n]=45;drive[2*n:3*n]=.4
    model=EliminatedLSTM(np.zeros((4*n,n)),drive)
    h=np.ones(n)*.5
    check,z=qualify(model,h)
    assert check['qualified'] and check['max_abs_c']>1e18
    assert check['min_one_minus_f']>0
    np.testing.assert_allclose(model.stable_residual(z),0,atol=1e-12)


def test_deflated_analytic_derivative_and_unstable_root_recovery():
    # One-unit LSTM with i=o=f=.5 and candidate tanh(3h) has an unstable root0
    # plus stable nonzero equilibria. Deflating the positive one must preserve0.
    w=np.zeros((4,1));w[2,0]=3
    model=EliminatedLSTM(w,np.zeros(4));pos=solve_start(model,np.array([.4]),np.empty((0,1)),False,100)[1][:1]
    h=np.array([.1]);known=pos[None]
    _,jac=deflated_residual(model,h,known);eps=1e-6
    numeric=(deflated_residual(model,h+eps,known)[0]-deflated_residual(model,h-eps,known)[0])/(2*eps)
    np.testing.assert_allclose(jac[:,0],numeric,rtol=1e-6,atol=1e-7)
    outcome,z,_,_=solve_start(model,h,known,True,100)
    assert outcome['selected']['qualified']
    assert abs(z[0])<1e-8
    assert np.max(np.abs(np.linalg.eigvals(model.full_jacobian(z))))>1
    assert outcome['nfev']<=100


def test_fixed_initialization_counts_bounds_and_reproducibility():
    native=np.zeros((64,5));known=np.ones((1,5))*.1
    a,labels=initializers(native,known,100)
    b,_=initializers(native,known,100)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(64,5) and len(labels)==64
    assert labels.count('native')==16 and labels.count('zero')==1 and labels.count('uniform')==23
    assert np.abs(a).max()<=1
