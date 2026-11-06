import torch
from scripts.lstm_endogenous_saturation_audit import candidate,sequence_test


def toy_cell(forget):
    cell=torch.nn.LSTMCell(2,256).double().requires_grad_(False)
    for p in cell.parameters():p.zero_()
    cell.bias_ih[256]=forget
    cell.bias_ih[512]=1
    actor=torch.nn.Linear(256,2,bias=False).double().requires_grad_(False)
    actor.weight.zero_();actor.weight[0,0]=1
    state=torch.zeros(4,512,dtype=torch.float64);state[:,256]=20
    return cell,actor,state


def test_saturated_accumulator_can_be_removed_only_with_output_check():
    cell,actor,state=toy_cell(20.)
    mask,values=candidate(state)
    assert mask.sum()==1
    with torch.no_grad():r=sequence_test(cell,actor,state,torch.zeros(60,2,dtype=torch.float64),mask,values)
    assert r['passes']
    assert r['max_selected_c_step_drift']>.3
    assert r['hidden_max_abs']<1e-6


def test_initial_saturation_alone_does_not_imply_future_invisibility():
    cell,actor,state=toy_cell(-2.)
    mask,values=candidate(state)
    with torch.no_grad():r=sequence_test(cell,actor,state,torch.zeros(60,2,dtype=torch.float64),mask,values)
    assert not r['passes']
    assert r['hidden_max_abs']>.1
