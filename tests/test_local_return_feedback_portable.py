"""Portable authentication tests; no learned policy or physical trial is executed."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import local_return_feedback_portable_evaluate as portable
import local_return_feedback_portable_run as runner
import local_return_feedback_run as frozen
import local_return_feedback_evaluate as original


def test_sealed_load_uses_exact_saved_bytes_without_regeneration(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Worker must not regenerate sealed bank')
    monkeypatch.setattr(portable.feedback.TrialStreams,'draw',forbidden)
    # Patching the class alters inspect.getsource only if its source changes; the
    # class source remains the exact frozen file and the draw call must not occur.
    before=torch.random.get_rng_state().clone()
    plan,record,_=portable.load_bank()
    assert original.tensor_schema(plan)==record['tensor_schema']
    assert torch.equal(before,torch.random.get_rng_state())


def test_changed_origin_validation_rejected(monkeypatch):
    real=portable.sha
    monkeypatch.setattr(portable,'sha',lambda p:'0'*64 if str(p).endswith(portable.ORIGIN_PATH) else real(p))
    with pytest.raises(AssertionError,match='Creation-time bank validation changed'):portable.load_bank()


def test_changed_manifest_rejected(monkeypatch):
    real=portable.sha
    monkeypatch.setattr(portable,'sha',lambda p:'0'*64 if Path(p)==portable.BEHAVIOR else real(p))
    with pytest.raises(AssertionError,match='manifest changed'):portable.load_bank()


def test_changed_bank_file_rejected(monkeypatch):
    real=portable.sha
    monkeypatch.setattr(portable,'sha',lambda p:'0'*64 if Path(p).name=='feedback_hard.pt' else real(p))
    with pytest.raises(AssertionError,match='bank bytes changed'):portable.load_bank()


def test_changed_tensor_rejected(monkeypatch):
    real=torch.load
    def changed(*a,**kw):
        plan=real(*a,**kw);plan['motor'][0,0,0]+=1e-4;return plan
    monkeypatch.setattr(portable.torch,'load',changed)
    with pytest.raises(AssertionError,match='tensor bytes changed'):portable.load_bank()


def test_runner_computations_and_fixed_design_unchanged():
    a=ast.parse(Path(frozen.__file__).read_text());b=ast.parse(Path(runner.__file__).read_text())
    funcs=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    fa,fb=funcs(a),funcs(b);assert set(fa)==set(fb)
    for name in set(fa)-{'protocol'}:assert fa[name]==fb[name],name
    for key in ('VERSION','UPDATES','TRIALS_PER_BATCH','TRAIN_SEED','MAX_SECONDS','PROVIDER_SECONDS','COUNTERS'):
        assert getattr(runner,key)==getattr(frozen,key)
    parent=json.loads(runner.FROZEN_V2_PROTOCOL.read_text());parent.pop('frozen_utc')
    new=runner.protocol();assert new.pop('portability')['same_saved_bank']
    for name,digest in parent['source_sha256'].items():assert new['source_sha256'][name]==digest
    for name,digest in parent['input_assets'].items():assert new['input_assets'][name]==digest
    new['source_sha256']=parent['source_sha256'];new['input_assets']=parent['input_assets']
    assert new==parent


def test_evaluation_delegates_to_unchanged_v2_plan(monkeypatch,tmp_path):
    captured={}
    def fake(net,cfg,plan,reference,out,**kw):
        captured.update(plan=plan,reference=reference,kwargs=kw);return {'sentinel':True}
    monkeypatch.setattr(original,'evaluate_plan',fake)
    result=portable.evaluate_bank(object(),{},tmp_path,device='cpu',actor_mode='reset_actor')
    assert result=={'sentinel':True}
    record=json.loads(portable.BEHAVIOR.read_text())['bank']
    assert original.tensor_schema(captured['plan'])==record['tensor_schema']
    assert captured['reference']['bank_sha256']==record['sha256']
    assert captured['kwargs']==dict(device='cpu',controller='policy',actor_mode='reset_actor')
