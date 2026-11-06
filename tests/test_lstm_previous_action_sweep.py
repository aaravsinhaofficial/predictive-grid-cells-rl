import pytest
import torch
from scripts.lstm_previous_action_sweep import action_input


def test_action_intervention_preserves_every_other_input_and_original():
    x=torch.arange(1287,dtype=torch.float64)[None,:]
    saved=x.clone()
    for action in range(6):
        changed=action_input(x,action)
        assert torch.equal(x,saved)
        assert torch.equal(changed[:,:1280],x[:,:1280])
        assert torch.equal(changed[:,1286:],x[:,1286:])
        assert changed[:,1280:1286].sum()==1
        assert changed[0,1280+action]==1
        assert torch.equal(action_input(changed,action),changed)
    with pytest.raises(AssertionError):action_input(x,6)
