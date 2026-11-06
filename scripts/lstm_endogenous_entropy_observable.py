"""Honest90M metadata adapter to the unchanged task-native observable probe."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from code.common import build_policy
from lstm_endogenous_curve_recovery import advance,sha,save
from lstm_endogenous_observable_flow import observable_probe
from rl_visual_memory_extend import tree_equal
import lstm_endogenous_reorientation_observable as original_observable
import lstm_reorientation_entropy_mechanism as producer

OUT=ROOT/'results/lstm_reorientation_entropy_observable'
PRODUCER=ROOT/'results/lstm_reorientation_entropy_mechanism'
CUES=original_observable.CUES
METRICS=('raw_full_state_kicks','h_c_RMS_full_state_kicks')
MODELS=tuple((arm,condition) for arm in producer.ARMS for condition in producer.CONDITIONS)
SOURCES=tuple(dict.fromkeys((*producer.SOURCES,
    'scripts/lstm_endogenous_reorientation_observable.py','scripts/lstm_endogenous_entropy_observable.py',
    'scripts/lstm_endogenous_observable_flow.py','tests/test_lstm_endogenous_entropy_observable.py')))


def protocol():
    inherited=json.loads((PRODUCER/'protocol.json').read_text())
    assert inherited==producer.protocol(),'Prospective90M primary adapter changed'
    return dict(version='fixed90M_task_native_future_observable_v1',
        cohort='All four fixed90,132,480-extra-frame endpoints: bothpopulations×original/explore entropy;2contexts×4cuebranches×2constructionmetrics=64checks. Centered and noncentered outcomes separate; nested checks, not64trainedagents.',
        metadata='Use honest producer preflight/validate_final/validate_producer and explicit actualworker/artifactchild routing. No40M metadata rewriting. Source/checkpoint/context hashes retained.',
        method='Exactly the40M observable_probe, archive validation and15→60replay checks. Same384fullstate arc samples,±2%radial/tangent kicks, per-cue t0 h/cRMS plus rawcoordinates, all256h at fourtimes, identical commonordinarydelay input. No learned map,selectedneurons or gatechanges.',
        timing='Reference15; evolve30 to45; observe offsets0/5/10/15 at both ends; laststate60 alreadyprocessedfirstfreeinput, before itsactor action.',
        guards='All original observable nontriviality, closedfamily,coverage/order,invariance andradial-recovery gates retained. ExactGridshams preserved. No cue/context/model selection.',
        interpretation='Secondary conditional ring-factor nomination; traineddynamics andwithin-modelcue effects stay distinct. Separate frozen primary/causal analyses test fullstate andactualnavigation. Any prospective observable rescue uses the same fixedmap as specified in producerprotocol.',
        producer_protocol_sha256=sha(PRODUCER/'protocol.json'),
        parent40M_observable_protocol_sha256=sha(ROOT/'results/lstm_reorientation_observable/protocol.json'),
        original_observable_protocol_sha256=sha(ROOT/'results/lstm_endogenous_ring/observable_history/protocol.json'),
        source_sha256={name:sha(ROOT/name) for name in SOURCES})


def validate_inputs(data,condition,arm,archive,checkpoint_hash,folder):
    producer.validate_final(data,condition,arm)
    producer.validate_producer(folder,checkpoint_hash,condition,arm)
    original_observable.validate_archive(archive,checkpoint_hash,condition)


def run(condition,arm,context,out,routing):
    checkpoint=producer.resolve_checkpoint(condition,arm,routing)
    data,cfg=producer.preflight(condition,arm,checkpoint,routing)
    digest=sha(checkpoint);folder=PRODUCER/arm/condition
    path=folder/f'context{context}_states.npz'
    with np.load(path) as archive:a={k:archive[k] for k in archive.files}
    validate_inputs(data,condition,arm,a,digest,folder)
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy']);net.double()
    x=torch.tensor(a['ordinary_input'],dtype=torch.float64)
    drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
    transition=lambda z:advance(net.lstm,drive,z)
    record=dict(condition=condition,arm=arm,context=context,extra_frames=data['extra_frames'],
        extension_frames=data['extension_frames'],checkpoint=str(checkpoint),checkpoint_sha256=digest,
        archive=str(path),archive_sha256=sha(path),source_sha256=sha(__file__),
        producer_metadata_sha256=sha(folder/'extension_adapter.json'),
        producer_protocol_sha256=sha(PRODUCER/'protocol.json'),routing_sha256=sha(routing),
        reference_updates=15,latest_update=60,branches={})
    with torch.no_grad():
        for cue in CUES:
            states=a[cue+'_states'].astype(np.float64)
            errors=original_observable.replay_error(transition,states)
            scale=np.repeat(np.sqrt(np.mean(states[0].reshape(-1,2,256)**2,axis=(0,2))),256).clip(1e-8)
            metrics={name:observable_probe(transition,states[original_observable.TIMES.index(15)],units,
                lambda z:z[:,:256],lambda z:net.pi(z[:,:256]).softmax(1))
                for name,units in zip(METRICS,(np.ones(512),scale))}
            record['branches'][cue]=dict(replay_max_abs_errors=errors,metrics=metrics)
    if condition=='gc100':
        assert all(tree_equal(record['branches']['native'],record['branches'][cue]) for cue in CUES[1:])
    record['grid_shams_exact']=condition=='gc100'
    assert sha(checkpoint)==digest
    target=out/arm/condition/f'context{context}.json';target.parent.mkdir(parents=True,exist_ok=True);save(target,record)
    print(json.dumps(dict(condition=condition,arm=arm,context=context,
        centered_candidates={cue:{m:r['centered_conditional_factor_candidate'] for m,r in branch['metrics'].items()}
            for cue,branch in record['branches'].items()})),flush=True)


def summarize(out,require_complete=False):
    rows=[];missing=[]
    for arm,condition in MODELS:
        for context in (0,1):
            path=out/arm/condition/f'context{context}.json'
            if not path.exists():missing.append(f'{arm}/{condition}/context{context}');continue
            record=json.loads(path.read_text())
            assert record['source_sha256']==sha(__file__) and record['extra_frames']==90_132_480
            assert (record['arm'],record['condition'],record['context'])==(arm,condition,context)
            assert tuple(record['branches'])==CUES
            for cue,branch in record['branches'].items():
                assert tuple(branch['metrics'])==METRICS
                for metric,r in branch['metrics'].items():
                    rows.append(dict(arm=arm,condition=condition,context=context,cue=cue,metric=metric,
                        centered_candidate=r['centered_conditional_factor_candidate'],
                        noncentered_candidate=r['noncentered_observable_candidate'],
                        radial_over_tangent=r['radial_over_tangent_observable_gain']['median'],
                        radial_gain=r['radial_observable_normal_gain']['median'],
                        centered_departure_p90=r['invariance']['centered']['residual_to_reference_curve_over_radius']['p90'],
                        radius_retention=r['invariance']['centered']['evolved_radius_retention'],
                        actor_probability_rms=r['actor']['evolved']['family_probability_RMS'],
                        centered_gates=r['candidate_gates']['centered'],
                        replay_max_abs_error=max(branch['replay_max_abs_errors'].values())))
    if require_complete:assert not missing and len(rows)==64
    save(out/'summary.json',dict(protocol_sha256=sha(out/'protocol.json'),missing=missing,
        all64_checks_complete=len(rows)==64,branch_metric_checks=len(rows),
        centered_candidates=sum(r['centered_candidate'] for r in rows),
        noncentered_candidates=sum(r['noncentered_candidate'] for r in rows),rows=rows,
        inference='Fourfixedmodels,oneoriginalseedpair;contexts,cuebranches andmetrics are nested. RetainedGridshamduplicates are not independentreplications.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition',choices=producer.CONDITIONS)
    p.add_argument('--arm',choices=tuple(producer.ARMS));p.add_argument('--routing',type=Path,default=producer.ROUTING)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--summarize-only',action='store_true');p.add_argument('--require-complete',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    design=protocol();path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==design,'Frozen90Mobservable adapter changed'
    else:
        assert args.freeze_only,'Review/freeze before analyzing final90Mstates'
        save(path,design)
    if args.freeze_only:return
    if not args.summarize_only:
        if args.condition is None or args.arm is None:p.error('--condition and --arm required')
        with threadpool_limits(limits=2):
            for context in (0,1):run(args.condition,args.arm,context,args.out,args.routing)
    summarize(args.out,args.require_complete)


if __name__=='__main__':main()
