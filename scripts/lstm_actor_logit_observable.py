"""Prospective fixed learned-action-score observable dynamics, all cohorts.

The sole new observation is Q(W_pi h+b_pi), retaining all six centered scores.
Every transition, full-state kick, distance/geometry calculation and candidate
gate is the unchanged observable_probe. No decoder or representation is fitted.
"""
from __future__ import annotations
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
import lstm_endogenous_reorientation_observable as forty
import lstm_reorientation_entropy_mechanism as ninety

OUT=ROOT/'results/lstm_actor_logit_observable'
METRICS=('raw_full_state_kicks','h_c_RMS_full_state_kicks')
PARENTS=('results/lstm_endogenous_ring/observable_history/protocol.json',
    'results/lstm_reorientation_observable/protocol.json',
    'results/lstm_reorientation_entropy_observable/protocol.json',
    'results/lstm_reorientation_entropy_mechanism/protocol.json',
    'results/lstm_travel_memory/original/protocol.json',
    'results/lstm_travel_memory/heldout_original/protocol.json',
    'results/lstm_reorientation_actor_readout/protocol.json')
SOURCES=tuple(dict.fromkeys((*ninety.SOURCES,*forty.SOURCES,
    'scripts/lstm_endogenous_travel_observable.py','scripts/lstm_travel_memory_probe.py',
    'scripts/lstm_travel_memory_heldout.py','scripts/lstm_actor_logit_observable.py',
    'tests/test_lstm_actor_logit_observable.py','tests/test_lstm_endogenous_observable_flow.py')))


def cases():
    result=[]
    for condition in ninety.CONDITIONS:
        for seed in range(1,9):
            group='original' if seed<=2 else 'heldout_original'
            for context in ((0,) if seed<=2 else (0,1)):
                result.append(dict(cohort='original1B',arm='original',condition=condition,seed=seed,context=context,
                    subgroup='pilot' if seed<=2 else 'heldout',
                    archive=f'results/lstm_travel_memory/{group}/{condition}_s{seed}/context{context}_states.npz',
                    checkpoint=f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt',
                    output=f'original1B/{condition}_s{seed}/context{context}.json'))
    for condition,worker in forty.WORKERS.items():
        for context in (0,1):
            result.append(dict(cohort='forty',arm='parent',condition=condition,seed=1,context=context,
                subgroup='fixed40M',archive=f'results/lstm_reorientation_mechanism/final/{condition}/context{context}_states.npz',
                checkpoint=f'results/ring_cloud_artifacts/{worker}/final.pt',output=f'forty/{condition}/context{context}.json'))
    for arm in ninety.ARMS:
        for condition in ninety.CONDITIONS:
            for context in (0,1):
                result.append(dict(cohort='ninety',arm=arm,condition=condition,seed=1,context=context,
                    subgroup='fixed90M',archive=f'results/lstm_reorientation_entropy_mechanism/{arm}/{condition}/context{context}_states.npz',
                    checkpoint='resolved from exact condition/arm routing final.pt',output=f'ninety/{arm}/{condition}/context{context}.json'))
    return result


def centered_actor_logits(state,weight,bias):
    scores=torch.nn.functional.linear(state[:,:weight.shape[1]],weight,bias)
    return scores-scores.mean(dim=1,keepdim=True)


def source_manifest(path):
    manifest=json.loads(Path(path).read_text())
    for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return manifest


def protocol():
    source_manifest(ROOT/'results/lstm_travel_memory/original/protocol.json')
    source_manifest(ROOT/'results/lstm_travel_memory/heldout_original/protocol.json')
    assert json.loads((ninety.OUT/'protocol.json').read_text())==ninety.protocol()
    assert json.loads((forty.OUT/'protocol.json').read_text())==forty.protocol()
    known=[case for case in cases() if case['cohort']!='ninety']
    assets=sorted({case['archive'] for case in known}|{case['checkpoint'] for case in known}|
        {f'models/{case["condition"]}/{case["condition"]}_lstm_s{case["seed"]}_1000M.json' for case in known})
    return dict(version='fixed_learned_actor_logit_observable_v1',
        timing='Prospectively frozen before any dynamics in this new observation map. Original and40M full-state/all-hidden outcomes and40M actor covariance,directional logits/output saturation are already known. The original heldout label describes the earlier collection cohort,not newly untouched data for this follow-up. No new-map outcome informed cohort/map/gates.',
        cases=cases(),counts=dict(original1B=56,original_pilot=8,original_heldout=48,forty=32,ninety=64,total=152),
        cohort='All16original1Bmodels: seeds1/2 one existing movement context each; seeds3–8 two existing held-out movement contexts each,28contexts total. Both40Mfinals and allfour90Mfinals:two contexts and allfour cue branches each. Raw/RMS and centered/noncentered retained. No new contexts or outcome selection.',
        map='Observe Q(W_pi h+b_pi) with Q=I−11T/6. Retain the literal6centered learned scores at each offset0/5/10/15:24coordinates, rank≤20. Bias retained; only the per-state action-common score removed. No fitting, whitening, score normalization, PCA, selected neurons/actions or2D projection.',
        dynamics='Use identical observable_probe:384uniform FULL512-state arc points;±2%full-radius tangent/radial kicks; same common ordinary native-context input; reference15→45 after30updates; future observations through60. Original cohort ordinary-release branch only; newtask cohorts allfour single-cue initializations.',
        metrics='Raw full-state and per-cue t0 h/c block-RMS constructions. Both noncentered and phase-family-mean-subtracted SCORE families. This second centering differs from mandatory action-common Qcentering; both are reported.',
        gates='EVERY existing gate and number unchanged. In particular observable radius RMS≥1e−4 now means centered-logit units per coordinate; kick nontriviality≥1e−4 of reference radius; invariance p90≤.05radius; radial gain and radial/angular≤.5. No gate relaxation or claim that a hidden-unit amplitude and logit-unit amplitude are interchangeable.',
        validation='Genuine1B/40M/90M metadata validated separately. Existing original/40M checkpoint/config/archive hashes locked here;90M validators require exact final budgets, parents, routing and producer. Replay every saved15→60branch before probes. Grid cue shams and checkpoint/archive immutability checked.',
        controls='Known attracting rotating ring with a phase-dependent actor-null hidden spectator: exact actor map retains the ring despite all-hidden deformation. Contracting point/plane and vanishing actor output fail; exactQlinear and softmax invariance. Existing rotating-ring+shared-drift,shrinking-plane,output-silent tests reused.',
        interpretation='A positive result nominates a conditional actor-score ring factor under the tested input, not an autonomous full-state attractor or demonstrated useful navigation. Finite future scores need not form a closed Markov quotient; mean action preferences and softmax saturation still matter. Repeated contexts/cues/metrics are not independent trained agents.',
        conditional_followup='If any candidate qualifies, retain all existing outcomes and test correct/ordinary/wrong cue repair in the SAME fixed24score map with dense unperturbed correct/own-cue references at prescribed15/30/45anchors through60. Couple to actual actor probabilities and the already specified navigation interventions. No standalone ring proof from projected geometry; no follow-up map fit or new gate.',
        parent_protocol_sha256={name:sha(ROOT/name) for name in PARENTS},
        existing_input_sha256={name:sha(ROOT/name) for name in assets},
        source_sha256={name:sha(ROOT/name) for name in SOURCES})


def validate_metadata(data,cfg,case):
    if case['cohort']=='original1B':
        assert data['frames']==1_000_000_000
        expected=json.loads((ROOT/f'models/{case["condition"]}/{case["condition"]}_lstm_s{case["seed"]}_1000M.json').read_text())
        assert cfg==expected and cfg['cond']==case['condition'] and cfg['seed']==case['seed']
    elif case['cohort']=='forty':
        forty.validate_checkpoint(data,case['condition'])
        expected=json.loads((ROOT/f'models/{case["condition"]}/{case["condition"]}_lstm_s1_1000M.json').read_text())
        expected['rollout']=160;assert cfg==expected
    else:
        assert case['cohort']=='ninety'
        assert cfg==ninety.validate_final(data,case['condition'],case['arm'])


def validate_original_archive(a,case,digest):
    assert str(a['checkpoint_sha256'].item())==digest
    assert tuple(a['times'])==forty.TIMES and a['theta'].shape==(768,)
    assert a['states'].shape==(7,768,512) and np.isfinite(a['states']).all()
    assert a['ordinary_input'].shape==(1,1287) and np.isfinite(a['ordinary_input']).all()
    meta=json.loads((ROOT/case['archive']).with_name(f'context{case["context"]}.json').read_text())
    assert (meta['condition'],meta['seed'],meta['context'])==(case['condition'],case['seed'],case['context'])
    assert meta['checkpoint_sha256']==digest and meta['source_sha256']==sha(ROOT/'scripts/lstm_travel_memory_probe.py')
    if case['seed']>=3:
        assert meta['heldout_source_sha256']==sha(ROOT/'scripts/lstm_travel_memory_heldout.py')
        inherited=json.loads((ROOT/'results/lstm_travel_memory/heldout_original/protocol.json').read_text())
        assert inherited['checkpoints'][f'{case["condition"]}_s{case["seed"]}']==digest


def run(case,out,routing):
    condition=case['condition'];path=ROOT/case['archive'];folder=path.parent
    if case['cohort']=='ninety':
        checkpoint=ninety.resolve_checkpoint(condition,case['arm'],routing)
        data,cfg=ninety.preflight(condition,case['arm'],checkpoint,routing)
    else:
        checkpoint=ROOT/case['checkpoint'];data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=data.get('config') or json.loads(checkpoint.with_suffix('.json').read_text())
    digest=sha(checkpoint);archive_hash=sha(path);validate_metadata(data,cfg,case)
    with np.load(path) as loaded:a={key:loaded[key] for key in loaded.files}
    if case['cohort']=='original1B':
        validate_original_archive(a,case,digest);branches={'ordinary_current_released':a['states']}
    else:
        if case['cohort']=='ninety':ninety.validate_producer(folder,digest,condition,case['arm'])
        else:
            manifest=source_manifest(folder/'implementation_protocol.json')
            assert manifest['design_sha256']==sha(ROOT/'docs/research/reorientation_mechanism_protocol.json')
        forty.validate_archive(a,digest,condition);branches={cue:a[cue+'_states'] for cue in forty.CUES}
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy']);net.double()
    assert net.pi.weight.shape==(6,256) and net.pi.bias.shape==(6,)
    drive=torch.nn.functional.linear(torch.tensor(a['ordinary_input'],dtype=torch.float64),net.lstm.weight_ih,net.lstm.bias_ih)
    transition=lambda z:advance(net.lstm,drive,z)
    output=lambda z:centered_actor_logits(z,net.pi.weight,net.pi.bias)
    record=dict(case=case,checkpoint=str(checkpoint),checkpoint_sha256=digest,archive_sha256=archive_hash,
        original_frames=1_000_000_000,extra_frames=0 if case['cohort']=='original1B' else data['extra_frames'],
        source_sha256=sha(__file__),protocol_sha256=sha(out/'protocol.json'),
        observation='Six literal centered learned logits at each of four future offsets;24coordinates.',
        reference_updates=15,latest_update=60,branches={},backend='cpu',torch_version=str(torch.__version__))
    if case['cohort']=='ninety':record.update(routing_sha256=sha(routing),producer_metadata_sha256=sha(folder/'extension_adapter.json'))
    with torch.no_grad():
        for cue,values in branches.items():
            states=values.astype(np.float64);errors=forty.replay_error(transition,states)
            scale=np.repeat(np.sqrt(np.mean(states[0].reshape(-1,2,256)**2,axis=(0,2))),256).clip(1e-8)
            metrics={name:observable_probe(transition,states[forty.TIMES.index(15)],units,output,
                lambda z:net.pi(z[:,:256]).softmax(1)) for name,units in zip(METRICS,(np.ones(512),scale))}
            assert all(r['observable_dimensions']==24 for r in metrics.values())
            record['branches'][cue]=dict(replay_max_abs_errors=errors,metrics=metrics)
    if condition=='gc100' and case['cohort']!='original1B':
        assert all(tree_equal(record['branches']['native'],record['branches'][cue]) for cue in forty.CUES[1:])
    assert sha(checkpoint)==digest and sha(path)==archive_hash
    target=out/case['output'];assert not target.exists(),'Keep completed new-map results; do not overwrite'
    target.parent.mkdir(parents=True,exist_ok=True);save(target,record)
    print(json.dumps(dict(case=case['output'],centered_candidates={cue:{metric:r['centered_conditional_factor_candidate'] for metric,r in b['metrics'].items()} for cue,b in record['branches'].items()})),flush=True)


def summarize(out,require_complete=False):
    rows=[];missing=[]
    for case in cases():
        path=out/case['output']
        if not path.exists():missing.append(case['output']);continue
        record=json.loads(path.read_text());assert record['case']==case and record['source_sha256']==sha(__file__)
        assert record['protocol_sha256']==sha(out/'protocol.json')
        expected=('ordinary_current_released',) if case['cohort']=='original1B' else forty.CUES
        assert tuple(record['branches'])==expected
        for cue,branch in record['branches'].items():
            assert tuple(branch['metrics'])==METRICS
            for metric,r in branch['metrics'].items():
                rows.append(dict(cohort=case['cohort'],subgroup=case['subgroup'],arm=case['arm'],condition=case['condition'],seed=case['seed'],context=case['context'],cue=cue,metric=metric,
                    checkpoint_sha256=record['checkpoint_sha256'],archive_sha256=record['archive_sha256'],
                    centered_candidate=r['centered_conditional_factor_candidate'],noncentered_candidate=r['noncentered_observable_candidate'],
                    radial_over_tangent=r['radial_over_tangent_observable_gain']['median'],radial_gain=r['radial_observable_normal_gain']['median'],
                    centered_departure_p90=r['invariance']['centered']['residual_to_reference_curve_over_radius']['p90'],
                    radius_retention=r['invariance']['centered']['evolved_radius_retention'],actor_probability_rms=r['actor']['evolved']['family_probability_RMS'],
                    centered_gates=r['candidate_gates']['centered'],noncentered_gates=r['candidate_gates']['noncentered'],
                    replay_max_abs_error=max(branch['replay_max_abs_errors'].values())))
    if require_complete:assert not missing and len(rows)==152
    save(out/'summary.json',dict(protocol_sha256=sha(out/'protocol.json'),complete=not missing and len(rows)==152,
        missing=missing,checks=len(rows),expected_checks=152,rows=rows,
        centered_candidates=sum(r['centered_candidate'] for r in rows),noncentered_candidates=sum(r['noncentered_candidate'] for r in rows),
        inference='Originalpilot/heldout/newtaskcohorts separate; nestedcontext/cue/metric rows and duplicatedGridshams are not independenttrainedmodels. Existingfullstate/fullhiddenresults remain primary.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--cohort',choices=('original1B','forty','ninety'));p.add_argument('--condition',choices=ninety.CONDITIONS)
    p.add_argument('--seed',type=int);p.add_argument('--arm',choices=tuple(ninety.ARMS));p.add_argument('--context',type=int,choices=(0,1))
    p.add_argument('--routing',type=Path,default=ninety.ROUTING);p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--summarize-only',action='store_true');p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True);design=protocol();manifest=args.out/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==design,'Frozen actor-score protocol/source/input changed'
    elif args.freeze_only:save(manifest,design)
    else:raise RuntimeError('Review and freeze before any new-map outcomes')
    if args.freeze_only:return
    if not args.summarize_only:
        if args.cohort is None:p.error('Choose a fixed cohort explicitly')
        selected=[case for case in cases() if case['cohort']==args.cohort and
            (args.condition is None or case['condition']==args.condition) and
            (args.seed is None or case['seed']==args.seed) and
            (args.arm is None or case['arm']==args.arm) and
            (args.context is None or case['context']==args.context)]
        assert selected,'No prescribed case matches requested filters'
        with threadpool_limits(limits=2):
            for case in selected:run(case,args.out,args.routing)
    summarize(args.out,args.require_complete)


if __name__=='__main__':main()
