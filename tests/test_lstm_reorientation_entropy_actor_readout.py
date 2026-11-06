from pathlib import Path
import pytest
from scripts import lstm_reorientation_entropy_actor_readout as adapter
from tests.test_lstm_reorientation_entropy_mechanism import metadata
from tests.test_lstm_endogenous_reorientation_observable import archive


def test_honest90m_metadata_and_producer_validation(monkeypatch):
    calls=[];monkeypatch.setattr(adapter.producer,'validate_producer',lambda *args:calls.append(args))
    data=metadata('gc100','explore');a=archive()
    adapter.validate_inputs(data,'gc100','explore',a,'correct',Path('producer'))
    assert calls==[(Path('producer'),'correct','gc100','explore')]
    assert data['extra_frames']==90_132_480
    a['plus60_states'][0,0,0]=1
    with pytest.raises(AssertionError):adapter.validate_inputs(data,'gc100','explore',a,'correct',Path('producer'))


def test_rejects40m_and_wrong_entropy_arm_without_relabeling(monkeypatch):
    monkeypatch.setattr(adapter.producer,'validate_producer',lambda *args:None)
    data=metadata();data['extra_frames']=40_058_880
    with pytest.raises(AssertionError):adapter.validate_inputs(data,'pgc33','explore',archive(),'correct',Path('.'))
    assert data['extra_frames']==40_058_880
    data=metadata()
    with pytest.raises(AssertionError):adapter.validate_inputs(data,'pgc33','original',archive(),'correct',Path('.'))


def test_exact_inherited_helper_and_complete_cohort():
    from scripts import lstm_reorientation_actor_readout as original
    assert adapter.original.decompose.__code__.co_code==original.decompose.__code__.co_code
    assert len(adapter.MODELS)*2*len(original.CUES)*len(original.TIMES)==128
    assert adapter.producer.resolve_checkpoint.__code__.co_code==__import__('scripts.lstm_reorientation_entropy_mechanism',fromlist=['resolve_checkpoint']).resolve_checkpoint.__code__.co_code
