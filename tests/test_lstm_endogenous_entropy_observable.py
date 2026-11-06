from pathlib import Path
import pytest
from scripts import lstm_endogenous_entropy_observable as adapter
from tests.test_lstm_endogenous_reorientation_observable import archive
from tests.test_lstm_reorientation_entropy_mechanism import metadata


def test_genuine90m_validation_delegation_and_grid_shams(monkeypatch):
    calls=[]
    monkeypatch.setattr(adapter.producer,'validate_producer',lambda *args:calls.append(args))
    data=metadata('gc100','explore');a=archive()
    adapter.validate_inputs(data,'gc100','explore',a,'correct',Path('test-producer'))
    assert calls==[(Path('test-producer'),'correct','gc100','explore')]
    assert data['extra_frames']==90_132_480
    a['minus60_states'][0,0,0]=1
    with pytest.raises(AssertionError):adapter.validate_inputs(data,'gc100','explore',a,'correct',Path('test-producer'))


def test_old40m_metadata_is_rejected_without_relabeling(monkeypatch):
    monkeypatch.setattr(adapter.producer,'validate_producer',lambda *args:None)
    data=metadata();data['extra_frames']=40_058_880
    with pytest.raises(AssertionError):adapter.validate_inputs(data,'pgc33','explore',archive(),'correct',Path('.'))
    assert data['extra_frames']==40_058_880


def test_fixed_cohort_and_original_observable_numerics_unchanged():
    assert len(adapter.MODELS)*2*len(adapter.CUES)*len(adapter.METRICS)==64
    from scripts.lstm_endogenous_observable_flow import observable_probe
    assert adapter.observable_probe.__code__.co_code==observable_probe.__code__.co_code
    assert adapter.original_observable.TIMES==(0,1,5,15,30,45,60)
