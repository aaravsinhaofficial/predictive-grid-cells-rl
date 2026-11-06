from types import SimpleNamespace
import torch
from scripts.lstm_endogenous_quotient_points import compare,quotient_step,reduce_state
from scripts.lstm_endogenous_saturation_audit import candidate,sequence_test


def test_sign_quotient_preserves_saturated_outputs_even_when_forced_small_c_does_not():
    cell=torch.nn.LSTMCell(2,256).double().requires_grad_(False)
    for p in cell.parameters():p.zero_()
    cell.bias_ih[512]=1
    actor=torch.nn.Linear(256,2,bias=False).double().requires_grad_(False)
    actor.weight.zero_();actor.weight[0,0]=1
    state=torch.zeros(4,512,dtype=torch.float64);state[:,256]=1e20
    mask,values=candidate(state);inputs=torch.zeros(60,2,dtype=torch.float64)
    with torch.no_grad():
        q=compare(SimpleNamespace(lstm=cell,pi=actor),state,inputs,mask,values.sign())
        clamp=sequence_test(cell,actor,state,inputs,mask,values)
    assert q['passes'] and q['hidden_max']<1e-12
    assert not clamp['passes']


def test_quotient_retains_every_unsaturated_coordinate_and_all_hidden_coordinates():
    cell=torch.nn.LSTMCell(2,256).double().requires_grad_(False)
    state=torch.randn(3,512,dtype=torch.float64)
    mask=torch.zeros(256,dtype=torch.bool);mask[7]=True
    q=reduce_state(state,mask)
    assert q.shape==(3,511)
    torch.testing.assert_close(q[:,:256],state[:,:256])
    drive=torch.randn(3,1024,dtype=torch.float64)
    result=quotient_step(cell,drive,q,mask,torch.ones(256,dtype=torch.float64))
    assert result.shape==(3,511)
