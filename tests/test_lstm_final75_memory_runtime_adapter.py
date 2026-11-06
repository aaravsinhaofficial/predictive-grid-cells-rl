import copy
import pytest
from scripts import lstm_final75_memory_runtime_adapter as adapter


def test_only_verified_runtime_field_is_removed(monkeypatch):
    design={'version':'test'}
    runtime=dict(torch_version='2.9.1+cu129',condition='gc100',device='cuda',protocol_sha256='expected')
    data={'provenance':dict(design,runtime=runtime)}
    monkeypatch.setattr(adapter,'sha',lambda path:'expected')
    received=[]
    monkeypatch.setattr(adapter,'BASE_VALIDATOR',lambda d,a,c,p:received.append(d))
    assert adapter.checked_metadata(data,'v2','gc100',design)==runtime
    assert received[0]['provenance']==design and 'runtime' in data['provenance']
    bad=copy.deepcopy(data);bad['provenance']['runtime']['protocol_sha256']='wrong'
    with pytest.raises(AssertionError):adapter.checked_metadata(bad,'v2','gc100',design)
    bad=copy.deepcopy(data);bad['provenance']['runtime']['extra']='unverified'
    with pytest.raises(AssertionError):adapter.checked_metadata(bad,'v2','gc100',design)
