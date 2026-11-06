"""Small synthetic bank-authentication fixtures, never the proposed study bank."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import pytest
import torch
import goal_cue_memory_bank as bank


def synthetic_bank(tmp_path,monkeypatch):
    monkeypatch.setattr(bank,'BANKS',{'qualification':8701,'memory':8702})
    monkeypatch.setattr(bank,'CONTEXTS',2);monkeypatch.setattr(bank,'ANGLES',4)
    monkeypatch.setattr(bank,'DELAY',3)
    cfg=ROOT/'models/gc100/gc100_lstm_s1_1000M.json'
    target=tmp_path/cfg.relative_to(ROOT);target.parent.mkdir(parents=True)
    target.write_bytes(cfg.read_bytes())
    manifest=tmp_path/'synthetic_behavior.json'
    design=bank.create_banks(manifest,root=tmp_path)
    return manifest,design


def test_saved_bank_exact_authentication_and_context_slice(tmp_path,monkeypatch):
    before=torch.get_rng_state().clone();manifest,design=synthetic_bank(tmp_path,monkeypatch)
    assert torch.equal(before,torch.get_rng_state())
    plan,record,_=bank.load_bank(manifest,'memory',root=tmp_path)
    assert record['seed']==8702 and len(plan['goal'])==8
    for index in range(2):
        part=bank.context_plan(plan,index);assert part['n_contexts']==1
        assert part['context'].tolist()==[index]*4
        assert torch.equal(part['motor'],plan['motor'][:,:,index*4:(index+1)*4])
        assert torch.equal(part['action'],plan['action'][:,index*4:(index+1)*4])
        assert torch.equal(part['goal'],plan['goal'][index*4:(index+1)*4])
    with pytest.raises(AssertionError,match='overwrite'):bank.create_banks(manifest,root=tmp_path)
    path=tmp_path/record['path'];changed=copy.deepcopy(plan);changed['goal'][0,0]+=1e-7
    torch.save(changed,path)
    with pytest.raises(AssertionError):bank.load_bank(manifest,'memory',root=tmp_path)


def test_bank_schema_or_seed_corruption_is_not_accepted(tmp_path,monkeypatch):
    manifest,design=synthetic_bank(tmp_path,monkeypatch)
    bad=copy.deepcopy(design);bad['banks']['memory']['seed']+=1
    manifest.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):bank.load_bank(manifest,'memory',root=tmp_path)
    bad=copy.deepcopy(design);schema=bad['banks']['memory']['tensor_schema']
    schema['/goal']['sha256']='0'*64;manifest.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):bank.load_bank(manifest,'memory',root=tmp_path)


def test_loading_never_calls_seed_regeneration(tmp_path,monkeypatch):
    manifest,_=synthetic_bank(tmp_path,monkeypatch)
    def forbidden(*args,**kwargs):raise AssertionError('Loading must not draw a bank')
    # Replacing draw changes the class-source pin if patched directly. The
    # independent generator constructor is enough to trap any attempted redraw.
    monkeypatch.setattr(bank.task.TrialStreams,'__init__',forbidden)
    plan,_,_=bank.load_bank(manifest,'qualification',root=tmp_path)
    assert len(plan['goal'])==8
