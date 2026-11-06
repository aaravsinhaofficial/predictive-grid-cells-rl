"""Authenticate and use the unchanged sealed bank without cross-platform re-drawing.

Creation-time exact seed regeneration remains recorded in the hash-pinned origin
validation. Runtime authentication requires exact file and tensor bytes; no
regenerated values, numerical tolerance, or replacement bank enters evaluation.
The frozen v2 evaluate_plan function still performs every policy/physical step.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import torch
import local_return_feedback as feedback
import local_return_feedback_evaluate as original

ROOT=original.ROOT
BEHAVIOR=original.BEHAVIOR
ORIGIN_PATH='results/local_return_feedback_design/bank_validation.json'
ORIGIN_SHA256='701f999c646acb6265378b8b9abfff2ecedadc0cd53251417df108dbf883507c'
sha=original.sha
global_rng_snapshot=original.global_rng_snapshot
restore_global_rng=original.restore_global_rng
preserved_rng=original.preserved_rng

def load_bank(manifest=BEHAVIOR):
    manifest=Path(manifest)
    origin_path=ROOT/ORIGIN_PATH
    assert sha(origin_path)==ORIGIN_SHA256, 'Creation-time bank validation changed'
    origin=json.loads(origin_path.read_text())
    assert origin['version']=='feedback_acquisition_bank_creation_v2' and origin['complete']
    assert origin['exact_private_seed_regeneration_passed'] and origin['global_cpu_rng_preserved']
    assert not origin['model_instantiation_or_forward_pass'] and not origin['environment_rollout']
    assert not origin['reference_or_policy_outcomes']
    assert str(manifest.resolve().relative_to(ROOT))==origin['manifest_path']
    assert sha(manifest)==origin['manifest_sha256'], 'Sealed bank manifest changed'
    for path,digest in origin['source_sha256'].items():assert sha(ROOT/path)==digest,path
    design=json.loads(manifest.read_text())
    assert design['version']=='feedback_acquisition_behavior_v2'
    assert design['seed']==original.BANK_SEED and design['stage']==asdict(feedback.STAGE)
    assert design['n_contexts']==original.CONTEXTS and design['headings']==original.HEADINGS
    assert design['success_threshold']==original.SUCCESS_THRESHOLD
    for name,cls in [('generator',feedback.TrialStreams),('stage',feedback.Stage)]:
        assert design[name+'_class_sha256']==hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
    record=design['bank'];assert record==origin['bank']
    path=Path(record['path']);assert not path.is_absolute() and '..' not in path.parts
    path=ROOT/path
    assert sha(path)==record['sha256'] and path.stat().st_size==record['bytes'], 'Sealed bank bytes changed'
    plan=torch.load(path,map_location='cpu',weights_only=True)
    assert original.tensor_schema(plan)==record['tensor_schema'], 'Sealed tensor bytes changed'
    assert plan['stage']==asdict(feedback.STAGE) and plan['n_contexts']==original.CONTEXTS
    assert plan['headings']==original.HEADINGS and plan['batch']==0
    return plan,record,design


def evaluate_bank(net,cfg,out,*,device='cpu',manifest=BEHAVIOR,controller='policy',actor_mode=None):
    plan,bank,_=load_bank(manifest)
    reference=dict(manifest_path=str(Path(manifest).resolve().relative_to(ROOT)),
        manifest_sha256=sha(manifest),bank_path=bank['path'],bank_sha256=bank['sha256'])
    return original.evaluate_plan(net,cfg,plan,reference,out,device=device,
        controller=controller,actor_mode=actor_mode)
