import json
from pathlib import Path
import pytest
from scripts import audit_reorientation_entropy_mechanism_outputs as adapter


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def causal_files(tmp_path,arm='explore',frames=90_132_480):
    primary=tmp_path/'primary';folder=primary/'causal/context0'
    inherited={'source_sha256':{'numerical.py':'fixed'},'fixed_cohort':'original40Mdesign'}
    frozen=tmp_path/'frozen.json';write(frozen,inherited)
    envelope={'adapter':{'genuine':'90M'},'inherited_numerical_protocol':inherited,
        'inherited_cohort_override':'Numerical protocol unchanged; actual prospective cohort is four fixed90M endpoints, not40M'}
    inputs={'condition':'pgc33','arm':arm,'genuine_extra_frames':frames,'checkpoint_sha256':'exact',
        'state_archive':str(primary/'context0_states.npz')}
    write(folder/'protocol.json',envelope);write(folder/'inputs.json',inputs)
    return primary,folder,frozen,envelope,inputs


def test_unwraps_only_verified_manifest_without_relabeling_inputs(tmp_path,monkeypatch):
    primary,folder,frozen,envelope,inputs=causal_files(tmp_path)
    monkeypatch.setattr(adapter,'CAUSAL_PROTOCOL',frozen)
    monkeypatch.setattr(adapter.producer,'protocol',lambda:{'genuine':'90M'})
    original_read=adapter.original.read
    before={path:path.read_bytes() for path in (folder/'protocol.json',folder/'inputs.json')}
    def existing_auditor(*args):
        assert adapter.original.read(folder/'protocol.json')==envelope['inherited_numerical_protocol']
        assert adapter.original.read(folder/'inputs.json')==inputs
        return {'context':0,'opportunity':{'phases':192,'phases_with_goal_impossible_under_any_action_sequence':0}}
    monkeypatch.setattr(adapter.original,'audit_causal',existing_auditor)
    result=adapter.audit_causal_envelope(folder,primary,0,'exact','pgc33','explore')
    assert result['genuine_extra_frames']==90_132_480
    assert adapter.original.read is original_read
    assert all(path.read_bytes()==content for path,content in before.items())


@pytest.mark.parametrize('wrong',('frames','arm','producer','inherited','archive'))
def test_rejects_incorrect_genuine_provenance_before_numerical_audit(tmp_path,monkeypatch,wrong):
    primary,folder,frozen,envelope,inputs=causal_files(tmp_path)
    monkeypatch.setattr(adapter,'CAUSAL_PROTOCOL',frozen)
    monkeypatch.setattr(adapter.producer,'protocol',lambda:{'genuine':'90M'})
    if wrong=='frames':inputs['genuine_extra_frames']=40_058_880
    elif wrong=='arm':inputs['arm']='original'
    elif wrong=='producer':envelope['adapter']={'genuine':'40M'}
    elif wrong=='inherited':envelope['inherited_numerical_protocol']={'source_sha256':{}}
    elif wrong=='archive':inputs['state_archive']=str(primary/'context1_states.npz')
    write(folder/'protocol.json',envelope);write(folder/'inputs.json',inputs)
    called=[];monkeypatch.setattr(adapter.original,'audit_causal',lambda *a:called.append(a))
    with pytest.raises(AssertionError):adapter.audit_causal_envelope(folder,primary,0,'exact','pgc33','explore')
    assert not called


def test_stale_audited_raw_artifact_rejected(tmp_path,monkeypatch):
    folder=tmp_path/'experiment';folder.mkdir()
    final=folder/'final.pt';final.write_bytes(b'checkpoint')
    raw=folder/'final_evaluation.json';raw.write_bytes(b'initial')
    audit=dict(passed=True,complete=True,required_complete=True,failures=[],pending=[],condition='pgc33',arm='explore',folder=str(folder),
        auditor_sha256=adapter.sha(adapter.ROOT/'scripts/audit_rl_blind_reorientation_entropy.py'),
        artifacts={p.name:{'bytes':p.stat().st_size,'sha256':adapter.sha(p)} for p in (final,raw)})
    path=tmp_path/'audit.json';write(path,audit)
    assert adapter.check_completed_training_audit(path,folder,'pgc33','explore',adapter.sha(final))['all_training_artifact_hashes_verified']
    raw.write_bytes(b'changed')
    with pytest.raises(AssertionError):adapter.check_completed_training_audit(path,folder,'pgc33','explore',adapter.sha(final))
