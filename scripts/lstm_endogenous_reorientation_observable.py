"""Task-native adapter to the unchanged future-hidden-activity ring-factor test.

All four single-cue branches share the ordinary, black, previous-LEFT delay
input. Every hidden unit is observed; no neurons or decoder plane are selected.
The primary full-state and separate free-navigation causal tests are unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from lstm_endogenous_observable_flow import load_net,observable_probe
from lstm_endogenous_curve_recovery import advance,sha,save
from rl_visual_memory_extend import tree_equal

OUT=ROOT/'results/lstm_reorientation_observable'
INPUT=ROOT/'results/lstm_reorientation_mechanism/final'
CUES=('native','ordinary','plus60','minus60')
TIMES=(0,1,5,15,30,45,60)
WORKERS={'gc100':'ring-20261105-reorient-grid-01','pgc33':'ring-20261105-reorient-pgc-02'}
SOURCES=('scripts/lstm_endogenous_reorientation_observable.py',
    'scripts/lstm_endogenous_observable_flow.py','scripts/lstm_endogenous_curve_recovery.py',
    'scripts/lstm_heading_curve_flow.py','scripts/lstm_reorientation_mechanism.py',
    'code/common.py','code/nets.py')


def protocol():
    return dict(version='task_native_reorientation_future_observable_v1',
        role='Secondary conditional ring-factor control; original full-state and free-navigation causal outcomes stay separate.',
        cohort='Both fixed completed40,058,880-learning-frame seed1 endpoints; both prospectively collected contexts; native/ordinary/+60/-60 single-cue branches; raw and h/c-RMS metrics. All32 branch/metric checks retained, centered and noncentered reported separately; not independent model replications.',
        inputs='Use only ordinary_input, shape1x1287, identical to native_input alias: common black RGB, held ordinary endpoint code, native saved goal, previousLEFT and zero reward. Phase-specific glimpse inputs are never the autonomous drive.',
        timing='Start at saved post-glimpse time15, evolve30updates to45, observe all256h at offsets0,5,10,15 both before/after. Latest neural state60 has processed the first free-action input, before its actor choice; no free movement is simulated here.',
        observation='Unchanged observable_probe: all256 hidden outputs at four future times,1024 coordinates; actor probabilities reported separately. No fitted decoder, neuron selection or selected2D projection.',
        perturbation='Unchanged384 uniform full-state arc samples; full tangent and radial projected off tangent, each plus/minus2% full-curve radius; raw metric and per-cue t0 h/c block RMS metric exactly as final75 adapter.',
        centering='Report original and common-phase-mean-subtracted observable curves. Perturbed branches use the unperturbed center; never independently remove perturbation means.',
        criteria='Use every existing observable_probe gate unchanged, including observable nontriviality, curve order/coverage/invariance, radial gain<=.5 and radial/tangent<=.5. A positive conditional-factor nomination requires separate task behavior and cue causality interpretation.',
        selection='No checkpoint, context, branch, neuron or outcome selection; no gate tuning. Degenerate families remain and fail the existing guards.',
        numerical_validation='Replay each saved branch from15 through60 under the sole shared ordinary input and compare archived30/45/60 states before probes; Grid cue/state/output shams exact.',
        source_sha256={name:sha(ROOT/name) for name in SOURCES},
        parent_observable_protocol_sha256=sha(ROOT/'results/lstm_endogenous_ring/observable_history/protocol.json'),
        task_protocol_sha256=sha(ROOT/'docs/research/rl_blind_reorientation_protocol.json'),
        primary_design_sha256=sha(ROOT/'docs/research/reorientation_mechanism_protocol.json'))


def validate_checkpoint(data,condition):
    assert data['completed'] is True and data['original_frames']==1_000_000_000
    assert data['extra_frames']==40_058_880 and data['total_updates']==2156
    assert data['warmup_updates']==200 and data['stage_evaluations_completed']==4
    assert data['config']['cond']==condition and data['config']['seed']==1
    expected=json.loads((ROOT/'docs/research/rl_blind_reorientation_protocol.json').read_text());expected.pop('frozen_at')
    assert data['provenance']==expected


def validate_archive(a,checkpoint_hash,condition):
    assert str(a['checkpoint_sha256'].item())==checkpoint_hash
    assert tuple(a['times'])==TIMES and a['theta'].shape==(768,)
    assert a['ordinary_input'].shape==(1,1287),'Autonomous input must be shared across phases'
    assert np.array_equal(a['ordinary_input'],a['native_input'])
    for cue in CUES:
        assert a[cue+'_states'].shape==(len(TIMES),768,512)
        assert a[cue+'_glimpse_inputs'].shape==(768,1287)
        assert np.isfinite(a[cue+'_states']).all()
    assert np.isfinite(a['ordinary_input']).all()
    if condition=='gc100':
        for cue in CUES[1:]:
            assert np.array_equal(a[cue+'_states'],a['native_states'])
            assert np.array_equal(a[cue+'_glimpse_inputs'],a['native_glimpse_inputs'])


def replay_error(transition,states):
    z=torch.tensor(states[TIMES.index(15)],dtype=torch.float64);errors={}
    with torch.no_grad():
        for t in range(16,61):
            z=transition(z)
            if t in (30,45,60):
                error=float(np.max(np.abs(z.numpy()-states[TIMES.index(t)])))
                assert np.allclose(z.numpy(),states[TIMES.index(t)],atol=1e-9,rtol=1e-11),(t,error)
                errors[str(t)]=error
    return errors


def run(condition,context,out):
    checkpoint=ROOT/'results/ring_cloud_artifacts'/WORKERS[condition]/'final.pt'
    folder=INPUT/condition;archive=folder/f'context{context}_states.npz'
    producer=json.loads((folder/'implementation_protocol.json').read_text())
    assert producer['design_sha256']==sha(ROOT/'docs/research/reorientation_mechanism_protocol.json')
    for path,digest in producer['source_sha256'].items():assert sha(ROOT/path)==digest
    net,data,cfg=load_net(checkpoint);validate_checkpoint(data,condition)
    with np.load(archive) as loaded:a={key:loaded[key] for key in loaded.files}
    validate_archive(a,sha(checkpoint),condition)
    x=torch.tensor(a['ordinary_input'],dtype=torch.float64)
    drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
    transition=lambda z:advance(net.lstm,drive,z)
    record=dict(condition=condition,context=context,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        archive=str(archive),archive_sha256=sha(archive),source_sha256=sha(__file__),
        producer_manifest_sha256=sha(folder/'implementation_protocol.json'),
        reference_updates=15,latest_update=60,branches={})
    with torch.no_grad():
        for cue in CUES:
            states=a[cue+'_states'].astype(np.float64)
            error=replay_error(transition,states)
            scale=np.repeat(np.sqrt(np.mean(states[0].reshape(-1,2,256)**2,axis=(0,2))),256).clip(1e-8)
            results={name:observable_probe(transition,states[TIMES.index(15)],units,
                lambda z:z[:,:256],lambda z:net.pi(z[:,:256]).softmax(1))
                for name,units in [('raw_full_state_kicks',np.ones(512)),('h_c_RMS_full_state_kicks',scale)]}
            record['branches'][cue]=dict(replay_max_abs_errors=error,metrics=results)
    if condition=='gc100':
        assert all(tree_equal(record['branches']['native'],record['branches'][c]) for c in CUES[1:])
    record['grid_shams_exact']=condition=='gc100'
    target=out/condition/f'context{context}.json';target.parent.mkdir(parents=True,exist_ok=True);save(target,record)
    print(json.dumps(dict(condition=condition,context=context,
        centered_candidates={cue:{metric:r['centered_conditional_factor_candidate'] for metric,r in b['metrics'].items()}
            for cue,b in record['branches'].items()})),flush=True)


def summarize(out,require_complete=False):
    rows=[];missing=[]
    for condition in WORKERS:
        for context in (0,1):
            path=out/condition/f'context{context}.json'
            if not path.exists():missing.append(f'{condition}/context{context}');continue
            result=json.loads(path.read_text());assert result['source_sha256']==sha(__file__)
            for cue,branch in result['branches'].items():
                for metric,r in branch['metrics'].items():
                    rows.append(dict(condition=condition,context=context,cue=cue,metric=metric,
                        centered_candidate=r['centered_conditional_factor_candidate'],
                        noncentered_candidate=r['noncentered_observable_candidate'],
                        radial_over_tangent=r['radial_over_tangent_observable_gain']['median'],
                        radial_gain=r['radial_observable_normal_gain']['median'],
                        centered_departure_p90=r['invariance']['centered']['residual_to_reference_curve_over_radius']['p90'],
                        radius_retention=r['invariance']['centered']['evolved_radius_retention'],
                        actor_probability_rms=r['actor']['evolved']['family_probability_RMS'],
                        centered_gates=r['candidate_gates']['centered'],
                        replay_max_abs_error=max(branch['replay_max_abs_errors'].values())))
    if require_complete:assert not missing and len(rows)==32
    save(out/'summary.json',dict(protocol_sha256=sha(out/'protocol.json'),missing=missing,
        all32_checks_complete=len(rows)==32,branch_metric_checks=len(rows),
        centered_candidates=sum(r['centered_candidate'] for r in rows),
        noncentered_candidates=sum(r['noncentered_candidate'] for r in rows),rows=rows,
        inference='Two originalseed1 populations, two nested native contexts and four controlled cue conditions; duplicated Grid shams are retained, not independent replications.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition',choices=tuple(WORKERS))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--summarize-only',action='store_true');p.add_argument('--require-complete',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    design=protocol();path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==design,'Frozen observable adapter changed'
    else:
        assert args.freeze_only,'Freeze source/protocol before analyzing final checkpoints'
        save(path,design)
    if args.freeze_only:return
    if not args.summarize_only:
        if args.condition is None:p.error('--condition required')
        with threadpool_limits(limits=2):
            for context in (0,1):run(args.condition,context,args.out)
    summarize(args.out,args.require_complete)


if __name__=='__main__':main()
