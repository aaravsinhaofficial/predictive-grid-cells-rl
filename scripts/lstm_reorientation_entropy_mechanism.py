"""Prospective fixed90M adapter: unchanged task-native geometry and cue navigation.

The genuine entropy-extension metadata is validated directly. No checkpoint is
relabeled as a40M checkpoint, and no frozen numerical source or gate is changed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy
from lstm_endogenous_curve_recovery import sha,save
import lstm_reorientation_mechanism as primary
import lstm_reorientation_causal_navigation as causal
from audit_rl_blind_reorientation_entropy import check_runtime_design

TRAINING_PROTOCOL=ROOT/'docs/research/rl_blind_reorientation_entropy_protocol.json'
TRAINING_HASH='cb05dc84f44e5cf076a7be29bedb6e1235f5a73ce797eabdfda874d966c1a4bb'
ROUTING=ROOT/'docs/research/rl_blind_reorientation_entropy_launch.json'
OUT=ROOT/'results/lstm_reorientation_entropy_mechanism'
ARMS={'original':8e-5,'explore':.01}
CONDITIONS=('gc100','pgc33')
SOURCES=tuple(dict.fromkeys((*primary.SOURCES,*causal.SOURCES,
    'scripts/lstm_reorientation_entropy_mechanism.py',
    'scripts/audit_rl_blind_reorientation_entropy.py',
    'tests/test_lstm_reorientation_entropy_mechanism.py')))


def training_design():
    assert sha(TRAINING_PROTOCOL)==TRAINING_HASH,'Frozen extension training protocol changed'
    data=json.loads(TRAINING_PROTOCOL.read_text());assert data.pop('frozen_at')
    return data


def resolve_checkpoint(condition,arm,routing_path=ROUTING):
    """Resolve the physical worker and artifact child; never infer retry names."""
    assert condition in CONDITIONS and arm in ARMS
    routing=json.loads(Path(routing_path).read_text())
    assert routing['frozen_protocol_sha256']==TRAINING_HASH
    matches=[r for r in routing['workers'] if (r['condition'],r['arm'])==(condition,arm)]
    assert len(matches)==1
    row=matches[0];name=Path(row['name']);child=Path(row['artifact_subdir'])
    assert len(name.parts)==1 and name.name.startswith('ring-')
    assert not child.is_absolute() and child.parts and '..' not in child.parts
    return ROOT/'results/ring_cloud_artifacts'/name/child/'final.pt'


def validate_final(data,condition,arm):
    """Validate real90M metadata without mutating data or its provenance."""
    assert condition in CONDITIONS and arm in ARMS
    expected=training_design()
    check_runtime_design(data['provenance'],expected,condition,arm)
    assert data['completed'] is True and data['original_frames']==1_000_000_000
    assert data['parent_extra_frames']==40_058_880 and data['extension_frames']==50_073_600
    assert data['extra_frames']==90_132_480 and data['extension_updates']==2445
    assert data['total_updates']==4601 and data['warmup_updates']==200
    assert data['extension_warmup_updates']==0 and data['stage_evaluations_completed']==5
    cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    cfg.update(rollout=160,entropy=ARMS[arm]);assert data['config']==cfg
    assert {float(s['step']) for s in data['optimizer']['state'].values()}=={4601.}
    assert data['runtime_snapshot']['noise_replays']==4601
    assert 0<=data['runtime_snapshot']['noise_max_replay_error']<3e-4
    return cfg


def protocol():
    return dict(version='fixed90M_matched_entropy_task_native_mechanism_v1',
        cohort='All four fixed final90,132,480-extra-learning-frame endpoints: original/explore entropy by Grid/PGC; original training seed1; exactly two physically eligible native contexts each. No stage or candidate selection.',
        training_protocol_sha256=TRAINING_HASH,
        numerical_design_sha256=sha(ROOT/'docs/research/reorientation_mechanism_protocol.json'),
        scope_addendum_sha256=sha(ROOT/'docs/research/reorientation_mechanism_review_addendum.md'),
        primary_implementation_manifest_sha256=sha(ROOT/'results/lstm_reorientation_mechanism/implementation_protocol.json'),
        causal_implementation_manifest_sha256=sha(ROOT/'results/lstm_reorientation_mechanism/causal_navigation/protocol.json'),
        validation='Require actual completed90M metadata, all5fixedblocks, exact40M parent hashes, unchanged architecture/config except frozen entropy,4601optimizer/noise-replayupdates, genuine six-field CUDA runtime provenance. Never rewrite checkpoint metadata.',
        routing='Resolve explicit condition/arm row.name plus row.artifact_subdir from the launch routing at execution; record routing snapshot/hash. Worker retries are infrastructure metadata, not outcome selection.',
        collection='Call unchanged collect_contexts: fresh CPU32env clean native histories under final30–60delay, seed168201, minimum2700updates/max6000 if needed; first2distinct-slot contexts selected only by frozen wall/goal clearance. No training noise, preserve all native phase coverage.',
        primary='Call unchanged elicit/analyze:768actualtravelphases, complete512(h,c), native/ordinary/wrong±60singlecues, identical subsequent ordinary drive within each model; raw and h/cRMS metrics; every original gate and failure retained.',
        causal='Call unchanged causal.analyze on both contexts regardless geometry:192angles,pre-cue±30phase and±2%radial/isotropic edits, radial15control,correct/ordinary/wrong±60cue/reset;30freeactions with exactmatchedtapes and firsthitstop. At t60 firstfreeaction uses pi(h) directly;29laterinputupdates. Noiseless forceddelay and native noisyfreemovement.',
        conditional_observable_rescue='If the independently frozen full-hidden observable ring-factor criteria qualify while full-state fails, retain all primary full-state causal outcomes and additionally measure cue repair in that SAME fixed1024-coordinate observation map [h(z),h(F^5z),h(F^10z),h(F^15z)] under ordinary delay input, at prescribed15/30/45anchors (latest60). Use dense unperturbed correct-cue and own-cue reference families with full-observable nearest segments; no representation fitting/neuron selection. Retain centered and noncentered versions, with perturbations using the unperturbed reference center, and all cues/contexts of the qualifying model. Couple to already prescribed actor/navigation effects. This is a conditional follow-up plan, not an implemented additional primary endpoint or new gate.',
        interpretation='Separate trained dynamics from acute within-model cue repair; nativecontexts/goalcode differ acrossmodels; one originalseedpair andnestedcontexts,not trained-seed significance. Observable adapter and behavioral evaluation remain separate.',
        source_sha256={p:sha(ROOT/p) for p in SOURCES})


def preflight(condition,arm,checkpoint,routing):
    assert checkpoint.name=='final.pt' and checkpoint.resolve()==resolve_checkpoint(condition,arm,routing).resolve()
    design=training_design()
    for p,digest in {**design['source_sha256'],**design['original_asset_sha256']}.items():
        assert sha(ROOT/p)==digest,p
    for item in design['parent_assets'].values():
        for name,digest in item['sha256'].items():assert sha(ROOT/item['folder']/name)==digest
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=validate_final(data,condition,arm)
    return data,cfg


def validate_producer(folder,checkpoint_hash,condition,arm):
    meta=json.loads((folder/'extension_adapter.json').read_text())
    assert (meta['condition'],meta['arm'],meta['checkpoint_sha256'])==(condition,arm,checkpoint_hash)
    assert meta['genuine_extra_frames']==90_132_480 and meta['adapter_source_sha256']==sha(__file__)
    assert meta['adapter_protocol']==protocol()
    inherited=json.loads((folder/'implementation_protocol.json').read_text())
    assert inherited==json.loads((ROOT/'results/lstm_reorientation_mechanism/implementation_protocol.json').read_text())
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--arm',choices=tuple(ARMS))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--routing',type=Path,default=ROUTING)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--mode',choices=('primary','causal'),default='primary')
    p.add_argument('--context',type=int,choices=(0,1));p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    manifest=protocol();path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:save(path,manifest)
    if args.freeze_only:return
    assert args.condition and args.arm
    checkpoint=args.checkpoint or resolve_checkpoint(args.condition,args.arm,args.routing)
    data,cfg=preflight(args.condition,args.arm,checkpoint,args.routing)
    folder=args.out/args.arm/args.condition;folder.mkdir(parents=True,exist_ok=True)
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy'])
    digest=sha(checkpoint)
    if args.mode=='primary':
        assert not (folder/'summary.json').exists(),'Do not overwrite a completed endpoint analysis'
        save(folder/'implementation_protocol.json',json.loads((ROOT/'results/lstm_reorientation_mechanism/implementation_protocol.json').read_text()))
        save(folder/'extension_adapter.json',dict(condition=args.condition,arm=args.arm,checkpoint=str(checkpoint),
            checkpoint_sha256=digest,genuine_extra_frames=data['extra_frames'],adapter_source_sha256=sha(__file__),
            adapter_protocol=manifest,routing_sha256=sha(args.routing),routing_snapshot=json.loads(args.routing.read_text()),
            checkpoint_provenance=data['provenance'],backend='cpu',torch_version=str(torch.__version__),
            torch_threads=2,started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        contexts=primary.collect_contexts(net,cfg,checkpoint,folder)
        result=[primary.analyze(net,data,cfg,w,checkpoint,i,folder) for i,w in enumerate(contexts)]
        save(folder/'summary.json',dict(condition=args.condition,arm=args.arm,checkpoint=str(checkpoint),
            checkpoint_sha256=digest,extra_frames=data['extra_frames'],contexts=result,
            adapter_source_sha256=sha(__file__),numerical_source_sha256=sha(Path(primary.__file__))))
    else:
        assert args.context is not None
        validate_producer(folder,digest,args.condition,args.arm)
        states=folder/f'context{args.context}_states.npz';archive=np.load(states)
        assert str(archive['checkpoint_sha256'])==digest
        assert len(archive['theta'])==768 and np.array_equal(archive['times'],causal.TIMES)
        producer=json.loads(states.with_name(states.name.replace('_states.npz','.json')).read_text())
        assert producer['checkpoint_sha256']==digest and producer['source_sha256']==sha(Path(primary.__file__))
        target=folder/'causal'/f'context{args.context}';target.mkdir(parents=True,exist_ok=True)
        assert not (target/'result.json').exists(),'Do not overwrite a completed causal result'
        save(target/'protocol.json',dict(adapter=manifest,
            inherited_numerical_protocol=json.loads((ROOT/'results/lstm_reorientation_mechanism/causal_navigation/protocol.json').read_text()),
            inherited_cohort_override='Numerical protocol unchanged; actual prospective cohort is four fixed90M endpoints, not40M'))
        save(target/'inputs.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
            state_archive=str(states),state_archive_sha256=sha(states),arm=args.arm,
            condition=args.condition,genuine_extra_frames=data['extra_frames']))
        try:result=causal.analyze(net.double(),cfg,archive,target)
        except ValueError as exc:
            save(target/'invalid_intervention.json',dict(reason=str(exc),preserve_context=True));raise
        print(json.dumps(result),flush=True)
    assert sha(checkpoint)==digest,'Read-only analysis changed checkpoint'


if __name__=='__main__':main()
