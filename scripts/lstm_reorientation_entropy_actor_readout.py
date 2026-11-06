"""Prospective90M adapter to the exact frozen40M linear actor decomposition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from lstm_endogenous_curve_recovery import sha,save
import lstm_reorientation_actor_readout as original
import lstm_reorientation_entropy_mechanism as producer
from lstm_endogenous_reorientation_observable import validate_archive

OUT=ROOT/'results/lstm_reorientation_entropy_actor_readout'
INPUT=producer.OUT
PARENT_PROTOCOL=ROOT/'results/lstm_reorientation_actor_readout/protocol.json'
PARENT_PROTOCOL_HASH='494b501009e1f6dce9c45b7ff4a19f4c23a0a8b1efd78038aa037428e1a5db54'
PRODUCER_PROTOCOL_HASH='18728e4b6b3d6f324f180dfced93ad980bcb462916ec971ef58cc5d030ec6562'
MODELS=tuple((arm,condition) for arm in producer.ARMS for condition in producer.CONDITIONS)
SOURCES=tuple(dict.fromkeys((*producer.SOURCES,*original.SOURCES,
    'scripts/lstm_reorientation_entropy_actor_readout.py',
    'tests/test_lstm_reorientation_entropy_actor_readout.py')))


def protocol():
    assert sha(PARENT_PROTOCOL)==PARENT_PROTOCOL_HASH
    inherited=json.loads(PARENT_PROTOCOL.read_text());assert inherited==original.protocol()
    assert sha(producer.OUT/'protocol.json')==PRODUCER_PROTOCOL_HASH
    assert json.loads((producer.OUT/'protocol.json').read_text())==producer.protocol()
    return dict(version='fixed90M_matched_entropy_actor_readout_v1',
        cohort='All4fixed90,132,480-extra-frame models(original/exploreentropy×Grid/PGC),2nativecontexts,4cues,4times0/15/30/60updates:128rows,768phases each. No stage/context/outcome selection.',
        prospective_prediction='Stronger entropy may reduce dominant average action margins and allow existing direction-related scores to alter choices. Hidden actor-sensitive rowspace energy, centered-score variation, mean action margins, exact probabilities and linearized softmax sensitivity remain separate quantities. Useful-memory interpretation still relies on the separately frozen cue/state/navigation interventions.',
        timing='Prepared and frozen before any90Mfinal readout outcomes; motivated by completed40M follow-up finding directional scores overwhelmed by a large default-action margin.',
        method='Call EXACT frozen40M decompose(hidden,weight,bias,theta) helper without changes: learned linear actor, action-common shift removal, exactSVDrow/nullspace, meanvsdirectional margins, fixedFourier harmonic share, exactsoftmax and separate linearized sensitivity. Same definitions/nullthresholds, no new fitting or gates.',
        metadata='Use producer.resolve_checkpoint/preflight/validate_final/validate_producer; explicit actualretryworker/artifactchild; genuine completed90Mconfig/runtime/parentframes. Never rewrite metadata as40M. Validate every archive with unchanged fullshape/Grididentity checks.',
        read_only='Saved hiddenstates and learned pi.weight/pi.bias only; no recurrent flow, environment, training, state edits, sampledactions or actor normalization. Input/checkpoint hashes verified before/after; all Gridcueidentities exact.',
        scope='One originaltrainingseedpair with two entropy continuations; nativecontexts differ betweenmodels/arms. Angles/contexts/cues are nested repeated measurements, not independent agents. No claim of entropy optimum or generalization from fixedFourier projection.',
        old_decomposition_protocol_sha256=PARENT_PROTOCOL_HASH,
        primary90M_protocol_sha256=PRODUCER_PROTOCOL_HASH,
        training90M_protocol_sha256=producer.TRAINING_HASH,
        source_sha256={p:sha(ROOT/p) for p in SOURCES})


def validate_inputs(data,condition,arm,archive,checkpoint_hash,folder):
    producer.validate_final(data,condition,arm)
    producer.validate_producer(folder,checkpoint_hash,condition,arm)
    validate_archive(archive,checkpoint_hash,condition)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--routing',type=Path,default=producer.ROUTING);p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    design=protocol();manifest=args.out/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==design
    elif args.freeze_only:save(manifest,design)
    else:raise RuntimeError('Freeze reviewed protocol before computing outcomes')
    if args.freeze_only:return
    torch.set_num_threads(2);rows=[];sources={}
    for arm,condition in MODELS:
        checkpoint=producer.resolve_checkpoint(condition,arm,args.routing)
        data,cfg=producer.preflight(condition,arm,checkpoint,args.routing)
        digest=sha(checkpoint);folder=INPUT/arm/condition
        w=data['policy']['pi.weight'].double().numpy();b=data['policy']['pi.bias'].double().numpy()
        sources[str(checkpoint.relative_to(ROOT))]=digest
        for context in range(2):
            path=folder/f'context{context}_states.npz';archive_hash=sha(path)
            with np.load(path) as loaded:a={key:loaded[key] for key in loaded.files}
            validate_inputs(data,condition,arm,a,digest,folder)
            sources[str(path.relative_to(ROOT))]=archive_hash;records={}
            for cue in original.CUES:
                records[cue]={}
                for t in original.TIMES:
                    hidden=a[cue+'_states'][list(a['times']).index(t),:,:256]
                    record=original.decompose(hidden,w,b,a['theta']);records[cue][t]=record
                    rows.append(dict(condition=condition,arm=arm,context=context,cue=cue,
                        time_updates=t,time_seconds=t/15,genuine_extra_frames=data['extra_frames'],
                        checkpoint_sha256=digest,archive_sha256=archive_hash,**record))
            if condition=='gc100':assert all(records[cue]==records['native'] for cue in original.CUES)
            assert sha(path)==archive_hash
        assert sha(checkpoint)==digest
    assert len(rows)==128
    save(args.out/'summary.json',dict(protocol_sha256=sha(manifest),source_sha256=sources,rows=rows,
        complete=True,all128rows_retained=True,grid_cue_shams_exact=True,no_environment_or_state_updates=True,
        routing_sha256=sha(args.routing),routing_snapshot=json.loads(args.routing.read_text()),
        backend='cpu',torch_threads=2,torch_version=str(torch.__version__)))


if __name__=='__main__':main()
