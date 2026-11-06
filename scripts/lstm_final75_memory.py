"""Frozen clean heading/travel assays for fixed75.072M v2/v3 endpoints.

Reuse the unchanged25M analysis functions and gates. Only checkpoint validation
and orchestration differ. State noise is OFF during context collection and
deterministic geometry for both the noiseless and noise-trained arms.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy
from lstm_heading_memory_probe import sha,save
import lstm_continued_heading_memory as heading
import lstm_travel_memory_probe as travel

TRAINING_PROTOCOLS={
    'v2':ROOT/'docs/research/rl_visual_memory_extension_protocol.json',
    'v3':ROOT/'docs/research/rl_visual_memory_noise_protocol.json'}
VERSIONS={'v2':'visual_memory_extension_v2','v3':'visual_memory_state_noise_v3'}
PARENTS={'gc100':ROOT/'results/ring_cloud_artifacts/ring-20261105-vision-grid-01/final.pt',
         'pgc33':ROOT/'results/ring_cloud_artifacts/ring-20261105-vision-pgc-01/final.pt'}
SOURCES=('scripts/lstm_final75_memory.py','scripts/lstm_continued_heading_memory.py',
         'scripts/lstm_travel_memory_probe.py','scripts/lstm_heading_curve_flow.py',
         'scripts/lstm_heading_current_release.py','scripts/lstm_heading_memory_probe.py',
         'scripts/lstm_endogenous_curve_recovery.py','scripts/plot_lstm_heading_memory.py',
         'scripts/rl_visual_memory.py','scripts/lstm_code_swap.py',
         'code/env.py','code/nets.py','code/codes.py','code/common.py')
NATIVE_ELIGIBILITY=heading.eligible_window


def training_design(arm):
    design=json.loads(TRAINING_PROTOCOLS[arm].read_text())
    design.pop('frozen_at',None)
    return design


def validate_checkpoint(data,arm,condition,design):
    assert data.get('completed') is True,'Require completed final checkpoint'
    assert data.get('original_frames')==1_000_000_000
    assert data.get('extra_frames')==75_072_000
    assert data.get('extension_frames')==50_048_000
    assert data['config']['cond']==condition and data['config']['seed']==1
    assert data['provenance']['version']==VERSIONS[arm]
    assert data['provenance']==design,'Final checkpoint provenance differs from frozen arm design'
    assert data['provenance']['parent_sha256'][condition]['final.pt']==design['parent_sha256'][condition]['final.pt']


def add_heading_checkpoint_metadata(path,checkpoint,arm):
    with np.load(path) as archive:
        fields={key:archive[key] for key in archive.files}
    fields.update(checkpoint_path=np.array(str(checkpoint)),checkpoint_sha256=np.array(sha(checkpoint)),
                  training_arm=np.array(arm),extra_frames=np.array(75_072_000),extension_frames=np.array(50_048_000))
    temp=path.with_name(path.stem+'_metadata_tmp.npz')
    np.savez_compressed(temp,**fields);temp.replace(path)


def run(data,cfg,checkpoint,arm,args):
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
    assert not net.lstm._forward_hooks,'Clean collection must not install a training-noise hook'
    hargs=SimpleNamespace(device=args.device,out=args.out/'heading',smoke_only=False)
    hargs.out.mkdir(parents=True,exist_ok=True)
    contexts=heading.collect_contexts(net,cfg,checkpoint,hargs)
    heading_results=[]
    for i,window in enumerate(contexts):
        result=heading.analyze_context(data,cfg,window,i,hargs)
        result.update(training_arm=arm,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                      extra_frames=75_072_000,extension_frames=50_048_000,
                      wrapper_source_sha256=sha(__file__),training_state_noise_during_assay=False)
        save(hargs.out/f'context{i}.json',result)
        add_heading_checkpoint_metadata(hargs.out/f'context{i}_states.npz',checkpoint,arm)
        heading_results.append(result)
    targs=SimpleNamespace(device=args.device,out=args.out/'travel',cohort=f'{arm}_final75M')
    targs.out.mkdir(parents=True,exist_ok=True)
    def eligible(window,slot,radius):
        ok,reason=NATIVE_ELIGIBILITY(window,slot,radius)
        if not ok:return ok,reason
        clear=travel.path_clearance(window[0]['pos'][slot],window[0]['goal_center'][slot],cfg)
        return (True,'accepted') if clear['eligible'] else (False,'all_angle_forward_path_clearance')
    with patch.object(heading,'eligible_window',eligible):
        travel_contexts=heading.collect_contexts(net,cfg,checkpoint,targs)
    travel_results=[]
    for i,window in enumerate(travel_contexts):
        result=travel.analyze(dict(checkpoint=str(checkpoint)),window,i,targs)
        result.update(training_arm=arm,extension_frames=50_048_000,
                      wrapper_source_sha256=sha(__file__),training_state_noise_during_assay=False)
        save(targs.out/f'{cfg["cond"]}_s1/context{i}.json',result);travel_results.append(result)
    return dict(training_arm=arm,condition=cfg['cond'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        extra_frames=75_072_000,extension_frames=50_048_000,clean_native_collection=True,
        heading=[dict(context=r['context'],candidates={name:b['passes_both_metric_candidate_screen'] for name,b in r['branches'].items()}) for r in heading_results],
        travel=[dict(context=r['context'],candidate=r['ordinary_release']['passes_both_metric_candidate_screen']) for r in travel_results])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['v2','v3']);p.add_argument('--condition',choices=['gc100','pgc33'])
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='mps')
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_final75_memory')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    designs={arm:training_design(arm) for arm in VERSIONS}
    parent_hashes={condition:sha(path) for condition,path in PARENTS.items()}
    for design in designs.values():
        assert design['total_learning_frames_beyond_1B']==75_072_000
        assert design['added_learning_frames']==50_048_000
        assert {c:design['parent_sha256'][c]['final.pt'] for c in PARENTS}==parent_hashes
    protocol=dict(version='final75M_clean_memory_assays_v1',training_arms=VERSIONS,
        checkpoint_requirements='Fixed final.pt only;completed=True; original1B+75.072M extra;50.048M extension; seed1; exact frozen training provenance and fixed25.024M parent hashes',
        collection='Fresh clean continued-policy episodes with final15-visible/30–60-blind schedule,seed98201,n_envs32. Two native heading contexts and two geometrically eligible travel contexts; same frozen selection methods as25M. Training state-noise hooks OFF for both arms.',
        analysis='Exact unchanged heading/travel functions, inputs, scales, geometry, readouts, radial kicks and candidate gates. Primary1–4s; no new topology loss or post-outcome threshold selection.',
        metadata='Heading and travel NPZs include checkpoint_path/checkpoint_sha256 for independent observable/pulse verification',
        parent_checkpoint_sha256=parent_hashes,
        training_protocol_sha256={arm:sha(path) for arm,path in TRAINING_PROTOCOLS.items()},
        source_sha256={name:sha(ROOT/name) for name in SOURCES})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen75M wrapper/source changed'
    else:save(path,protocol)
    if args.freeze_only:return
    if args.arm is None or args.condition is None or args.checkpoint is None:p.error('--arm,--condition,--checkpoint required')
    assert args.checkpoint.name=='final.pt','Use the fixed final checkpoint, not a selected intermediate block'
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    validate_checkpoint(data,args.arm,args.condition,designs[args.arm])
    save(args.out/'checkpoint_audit.json',dict(arm=args.arm,condition=args.condition,checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha(args.checkpoint),parent_sha256=parent_hashes[args.condition],
        provenance_matches_frozen_training=True,training_state_noise_during_assay=False))
    result=run(data,data['config'],args.checkpoint,args.arm,args)
    save(args.out/'summary.json',result)


if __name__=='__main__':main()
