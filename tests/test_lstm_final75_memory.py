import copy
import pytest
from scripts.lstm_final75_memory import training_design,validate_checkpoint


def checkpoint(arm):
    design=training_design(arm)
    return dict(completed=True,original_frames=1_000_000_000,extra_frames=75_072_000,
                extension_frames=50_048_000,config=dict(cond='gc100',seed=1),provenance=copy.deepcopy(design)),design


def test_accepts_both_frozen_arms_with_matching_parent_provenance():
    for arm in ('v2','v3'):
        data,design=checkpoint(arm);validate_checkpoint(data,arm,'gc100',design)


def test_rejects_old_endpoint_and_changed_parent():
    data,design=checkpoint('v2');data['extra_frames']=25_024_000
    with pytest.raises(AssertionError):validate_checkpoint(data,'v2','gc100',design)
    data,design=checkpoint('v3');data['provenance']['parent_sha256']['gc100']['final.pt']='wrong'
    with pytest.raises(AssertionError):validate_checkpoint(data,'v3','gc100',design)


def test_rejects_wrong_training_arm_and_incomplete_endpoint():
    data,design=checkpoint('v2')
    with pytest.raises(AssertionError):validate_checkpoint(data,'v3','gc100',design)
    data['completed']=False
    with pytest.raises(AssertionError):validate_checkpoint(data,'v2','gc100',design)
