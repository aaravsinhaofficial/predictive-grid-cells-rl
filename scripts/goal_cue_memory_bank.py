"""Prospective bank creation and exact portable authentication; no model calls.

Import is inert. Creation must be separately authorized before any study outcome.
Workers authenticate the saved bytes; they never regenerate a sealed bank.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
import goal_cue_memory as task
from local_return_feedback_evaluate import tensor_schema
from rl_visual_memory_extend import tree_equal

BEHAVIOR = ROOT/'docs/research/goal_cue_memory_behavior_protocol.json'
VERSION = 'native_goal_cue_memory_behavior_v1'
BANKS = {'qualification':691801, 'memory':691901}
CONTEXTS, ANGLES, DELAY = 16, 16, 45


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def context_plan(plan, index):
    assert 0 <= index < plan['n_contexts']
    width = plan['angles_per_context']; sl = slice(index*width,(index+1)*width)
    result = {}
    for key,value in plan.items():
        if key == 'motor': result[key] = value[:,:,sl].clone()
        elif key == 'action': result[key] = value[:,sl].clone()
        elif isinstance(value,torch.Tensor): result[key] = value[sl].clone()
        else: result[key] = copy.deepcopy(value)
    result['n_contexts'] = 1
    return result


def create_banks(manifest=BEHAVIOR, *, root=ROOT):
    """Pure private draws and serialization only; no environment or policy."""
    root = Path(root); manifest = Path(manifest)
    assert not manifest.exists(), 'Never overwrite a prospective bank manifest'
    folder = root/'results/goal_cue_memory_design/banks'
    assert not folder.exists(), 'Never replace earlier bank bytes'
    cfg_path = root/'models/gc100/gc100_lstm_s1_1000M.json'
    cfg = json.loads(cfg_path.read_text()); before = torch.get_rng_state().clone()
    records = {}; folder.mkdir(parents=True)
    for name,seed in BANKS.items():
        streams = task.TrialStreams(seed)
        plan = streams.draw(CONTEXTS,cfg,delay_steps=DELAY,angles_per_context=ANGLES)
        task.validate_plan(cfg,plan)
        check = task.TrialStreams(seed).draw(CONTEXTS,cfg,delay_steps=DELAY,angles_per_context=ANGLES)
        assert tree_equal(plan,check), 'Creation-time exact seed validation failed'
        path = folder/f'{name}.pt'; torch.save(plan,path)
        assert tree_equal(torch.load(path,map_location='cpu',weights_only=True),plan)
        records[name] = dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size,
            tensor_schema=tensor_schema(plan),seed=seed,delay_steps=DELAY,n_contexts=CONTEXTS,angles_per_context=ANGLES,
            creation_validation=dict(exact_private_seed_regeneration=True,exact_serialization_roundtrip=True,
                no_policy_environment_or_training=True,final_private_streams=tensor_schema(streams.snapshot())))
    assert torch.equal(before,torch.get_rng_state())
    design = dict(version=VERSION,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        banks=records,qualification_successes=192,memory_successes=167,memory_erasure_cost=39,trials=CONTEXTS*ANGLES,
        common_information_50_percent_certificate=False,
        bank_authentication='Exact saved file bytes/SHA and uncompressed tensor schema/SHA. No loading-time regeneration, substitution or tolerance-based authentication.',
        sampling='16 independent contexts, each16 evenly spaced goal bearings with a private uniform angular offset. Independent uniform pickup offset is fixed relative to goal within each family. Common room and tapes within each context.',
        global_cpu_rng_preserved=True,config_path=str(cfg_path.relative_to(root)),config_sha256=sha(cfg_path),
        generator_class_sha256=hashlib.sha256(inspect.getsource(task.TrialStreams).encode()).hexdigest(),
        collector_source_sha256=sha(task.__file__),generator_source_sha256=sha(__file__),torch_version=str(torch.__version__))
    manifest.parent.mkdir(parents=True,exist_ok=True); manifest.write_text(json.dumps(design,indent=2)+'\n')
    return design


def load_bank(manifest=BEHAVIOR, bank_name='memory', *, root=ROOT):
    root = Path(root); manifest = Path(manifest); design = json.loads(manifest.read_text())
    assert design['version'] == VERSION and set(design['banks']) == set(BANKS)
    assert (design['qualification_successes'],design['memory_successes'],design['memory_erasure_cost'],design['trials']) == (192,167,39,CONTEXTS*ANGLES)
    assert not design['common_information_50_percent_certificate']
    assert design['generator_class_sha256'] == hashlib.sha256(inspect.getsource(task.TrialStreams).encode()).hexdigest()
    assert design['collector_source_sha256'] == sha(task.__file__)
    assert design['generator_source_sha256'] == sha(__file__)
    cfg_name = Path(design['config_path']); assert not cfg_name.is_absolute() and '..' not in cfg_name.parts
    assert sha(root/cfg_name) == design['config_sha256']
    cfg = json.loads((root/cfg_name).read_text())
    record = design['banks'][bank_name]; relative = Path(record['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = root/relative; assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    assert (record['seed'],record['delay_steps'],record['n_contexts'],record['angles_per_context']) == (BANKS[bank_name],DELAY,CONTEXTS,ANGLES)
    assert all(record['creation_validation'][key] for key in ('exact_private_seed_regeneration','exact_serialization_roundtrip','no_policy_environment_or_training'))
    plan = torch.load(path,map_location='cpu',weights_only=True)
    assert tensor_schema(plan) == record['tensor_schema']
    assert (plan['delay_steps'],plan['n_contexts'],plan['angles_per_context'],plan['batch']) == (DELAY,CONTEXTS,ANGLES,0)
    assert torch.equal(plan['context'],torch.arange(CONTEXTS).repeat_interleave(ANGLES))
    assert torch.equal(plan['angle_index'],torch.arange(ANGLES).repeat(CONTEXTS))
    task.validate_plan(cfg,plan)
    return plan,record,design


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create',action='store_true',required=True)
    parser.add_argument('--manifest',type=Path,default=BEHAVIOR)
    args=parser.parse_args(); result=create_banks(args.manifest)
    print(json.dumps(dict(complete=True,banks={k:{f:r[f] for f in ('path','sha256','bytes')}for k,r in result['banks'].items()})))
