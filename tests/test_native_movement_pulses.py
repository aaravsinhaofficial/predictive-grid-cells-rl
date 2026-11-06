import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from native_movement_pulses import crossing,select_cases,summarize_case


def test_first_passage_preserves_reversals():
    phase=np.array([0.,2.,1.,3.,1.,4.]);states=np.c_[np.arange(6.),np.arange(6.)**2]
    result,bracket=crossing(phase,states,1.5)
    assert bracket['left']==0 and bracket['right']==1
    np.testing.assert_array_equal(result,[.75,.75])


def test_mechanical_selection_all_episodes_and_no_later_replacement():
    data={'x':torch.zeros(10,32,1)}
    segment=dict(slot=0,known_goal=True,repeated_same_sign_pairs=1,start_step=0,end_step=5)
    cycles=[dict(slot=0,segment_start=0,sign=1,start_step=0,end_step=2),dict(slot=0,segment_start=0,sign=1,start_step=3,end_step=5)]
    cases,missing=select_cases(data,dict(segments=[segment],cycles=cycles))
    assert cases==[] and len(missing)==32
    assert missing[0]['reason']=='first repeated crossing reaches terminal boundary'


def test_response_uses_same_mode_sham_and_terminal_mask():
    initial=torch.zeros(21,512);initial[1::7]=.1
    state=torch.zeros(5,21,512)
    for mode in range(3):
        state[:,mode*7:(mode+1)*7]=mode*20.
        state[:,mode*7+1]+=.05
    active=torch.ones(5,21,dtype=torch.bool);active[2:,1]=False
    raw=dict(initial_state=initial,initial_position=torch.zeros(2),goal=torch.ones(2),
        closed=dict(state=state,logits=torch.zeros(5,21,6),pos=torch.zeros(5,21,2),hit=torch.zeros(5,21,dtype=torch.bool),path_step=torch.zeros(5,21),active=active),
        replay=dict(state=state,logits=torch.zeros(5,21,6)))
    rows=summarize_case(raw,dict(scale=torch.ones(512),slot=0,anchor=0,horizon=5))
    for mode in range(3):
        row=rows[mode*7+1]
        assert abs(row['responses']['closed'][0]['scaled_state_gain']-.5)<1e-4
    assert not rows[1]['responses']['closed'][1]['paired_before_terminal']
    assert rows[8]['responses']['closed'][1]['paired_before_terminal']
