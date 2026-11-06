import copy
import numpy as np
import pytest
import torch
from scripts import lstm_endogenous_reorientation_observable as adapter


def archive():
    a=dict(checkpoint_sha256=np.array('correct'),times=np.array(adapter.TIMES),theta=np.zeros(768),
        ordinary_input=np.zeros((1,1287)),native_input=np.zeros((1,1287)))
    for cue in adapter.CUES:
        a[cue+'_states']=np.zeros((7,768,512));a[cue+'_glimpse_inputs']=np.zeros((768,1287))
    return a


def test_all_four_branches_shared_input_and_grid_sham_validation():
    a=archive();adapter.validate_archive(a,'correct','gc100')
    bad=copy.deepcopy(a);bad['ordinary_input']=np.zeros((768,1287))
    with pytest.raises(AssertionError):adapter.validate_archive(bad,'correct','pgc33')
    bad=copy.deepcopy(a);bad['native_input'][0,0]=1
    with pytest.raises(AssertionError):adapter.validate_archive(bad,'correct','pgc33')
    bad=copy.deepcopy(a);bad['plus60_states'][0,0,0]=1
    with pytest.raises(AssertionError):adapter.validate_archive(bad,'correct','gc100')
    adapter.validate_archive(bad,'correct','pgc33')
    with pytest.raises(AssertionError):adapter.validate_archive(a,'incorrect','gc100')


def test_replay_checks_saved15_to60_and_rejects_wrong_indexing():
    states=np.stack([np.full((3,4),float(t)) for t in adapter.TIMES])
    errors=adapter.replay_error(lambda z:z+1,states)
    assert errors=={'30':0.,'45':0.,'60':0.}
    wrong=states.copy();wrong[-1]+=1
    with pytest.raises(AssertionError):adapter.replay_error(lambda z:z+1,wrong)
