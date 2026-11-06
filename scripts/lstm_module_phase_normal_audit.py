"""Prospective broad-direction perturbations of every positive module factor.

Discovery labels remain unchanged. This audit adds fixed random full-state,
hidden-only, and cell-only directions to the existing radial/tangent probes.
No decoder, circular fit, or independently centered perturbed family is used.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import lstm_heading_curve_flow as flow
import lstm_module_phase_ring as primary
from lstm_endogenous_curve_recovery import advance

PARENT=ROOT/'results/lstm_module_phase_observable'
EVOLUTIONS=(15,30,75)
OFFSETS=(0,5,10,15)
RNG_SEED=2026110619


def label(row):
    return '_'.join(str(row[k]) for k in ('condition','seed','module','axis','regime','anchor_updates','metric','observable'))


def summary(values,valid=None):
    x=np.asarray(values,dtype=float)
    if valid is not None:x=x[np.asarray(valid,dtype=bool)]
    x=x[np.isfinite(x)]
    if not len(x):return dict(count=0,median=None,p90=None,maximum=None,minimum=None)
    return dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()),minimum=float(x.min()))


def make_starts(curve,scale,count=384,anchor_count=8):
    scaled=np.asarray(curve,dtype=float)/scale
    points=flow.uniform_arc_samples(scaled,count)
    ids=np.arange(anchor_count)*count//anchor_count
    tangent=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-30)
    q=tangent[ids]
    n=points.shape[1];half=n//2
    radius=flow.curve_geometry(scaled)['radius'];amplitude=.02*radius
    rng=np.random.default_rng(RNG_SEED)
    directions=[];names=[]
    for index in range(16):
        v=rng.normal(size=(anchor_count,n))
        v-=np.sum(v*q,axis=1,keepdims=True)*q
        directions.append(v);names.append(f'random_full_{index:02d}')
    for name,sl in [('hidden_only',slice(0,half)),('cell_only',slice(half,n))]:
        v=np.zeros((anchor_count,n));v[:,sl]=rng.normal(size=(anchor_count,half))
        component=q[:,sl]
        v[:,sl]-=np.sum(v[:,sl]*component,axis=1,keepdims=True)*component/np.maximum(np.sum(component*component,axis=1,keepdims=True),1e-30)
        directions.append(v);names.append(name)
    radial=points[ids]-points.mean(0)
    radial-=np.sum(radial*q,axis=1,keepdims=True)*q
    directions.extend([radial,q]);names.extend(['existing_radial','existing_tangent'])
    directions=np.stack(directions)
    lengths=np.linalg.norm(directions,axis=-1)
    directions/=np.maximum(lengths[:,:,None],1e-30)
    perturbed=points[ids][None,None]+amplitude*directions[:,None]*np.array([1.,-1.])[None,:,None,None]
    starts=np.concatenate((points,perturbed.reshape(-1,n)))*scale
    metadata=dict(count=count,anchor_count=anchor_count,anchor_indices=ids.tolist(),names=names,
        amplitude_scaled_norm=amplitude,full_scaled_radius=radius,direction_norm_before_normalizing=lengths.tolist(),
        normal_dot_slice_tangent=np.sum(directions[:-1]*q[None],axis=-1).tolist(),
        hidden_bound_overshoot_count=(np.abs(starts[count:,:half])>1).sum(axis=1).tolist())
    return starts,metadata


def measure_observations(observed,metadata):
    count=metadata['count'];p=metadata['anchor_count'];ids=np.asarray(metadata['anchor_indices'])
    nd=len(metadata['names']);shape=(nd,2,p)
    first=np.concatenate([observed[t] for t in OFFSETS],axis=1)
    reference=first[:count];radius=flow.curve_geometry(reference)['radius']
    initial=first[count:]
    d0,phase0,closest0,_=flow.nearest_closed_segments(initial,reference)
    phase_base0=flow.nearest_closed_segments(reference[ids],reference)[1]
    base0=np.broadcast_to(reference[ids],(nd,2,p,reference.shape[1])).reshape(initial.shape)
    total0=np.linalg.norm(initial-base0,axis=1)
    meaningful=d0>=1e-4*max(radius,1e-30)
    tangent_valid=total0.reshape(shape)[-1]>=1e-4*max(radius,1e-30)
    arrays=dict(initial_normal_distance=d0.reshape(shape),initial_total_distance=total0.reshape(shape),
                initial_normal_nontrivial=meaningful.reshape(shape),initial_arc_position=phase0.reshape(shape),
                initial_base_arc_position=phase_base0,initial_phase_component_norm=np.linalg.norm(closest0-base0,axis=1).reshape(shape))
    records={}
    for horizon in EVOLUTIONS:
        last=np.concatenate([observed[horizon+t] for t in OFFSETS],axis=1)
        baseline=last[:count];perturbed=last[count:]
        d1,phase1,closest1,_=flow.nearest_closed_segments(perturbed,baseline)
        phase_base1=flow.nearest_closed_segments(baseline[ids],baseline)[1]
        base1=np.broadcast_to(baseline[ids],(nd,2,p,baseline.shape[1])).reshape(perturbed.shape)
        total1=np.linalg.norm(perturbed-base1,axis=1)
        phase_component=np.linalg.norm(closest1-base1,axis=1)
        normal_gain=d1/np.maximum(d0,1e-30)
        total_gain=total1/np.maximum(total0,1e-30)
        tangent_gain=total_gain.reshape(shape)[-1]
        tangent_denom=np.mean(tangent_gain,axis=0)
        ratio=normal_gain.reshape(shape)/np.maximum(tangent_denom[None,None],1e-30)
        valid=meaningful.reshape(shape)&tangent_valid.all(axis=0)[None,None]
        arc_delta=(phase1.reshape(shape)-phase_base1[None,None]+.5)%1-.5
        for name,value in dict(normal_distance=d1.reshape(shape),normal_gain=normal_gain.reshape(shape),
            total_gain=total_gain.reshape(shape),normal_over_tangent_gain=ratio,valid_for_ratio=valid,
            arc_position=phase1.reshape(shape),base_arc_position=phase_base1,arc_displacement=arc_delta,
            phase_component_norm=phase_component.reshape(shape),total_distance=total1.reshape(shape)).items():
            arrays[f'{name}_{horizon}']=value
        directions=[]
        for index,name in enumerate(metadata['names']):
            mask=valid[index]
            directions.append(dict(name=name,initial_normal_nontrivial_fraction=float(meaningful.reshape(shape)[index].mean()),
                paired_tangent_nontrivial_fraction=float(tangent_valid.all(axis=0).mean()),
                normal_gain=summary(normal_gain.reshape(shape)[index],mask),
                normal_over_tangent_gain=summary(ratio[index],mask),total_gain=summary(total_gain.reshape(shape)[index]),
                absolute_arc_displacement=summary(np.abs(arc_delta[index])),
                initial_normal_distance_over_radius=summary(d0.reshape(shape)[index]/max(radius,1e-30))))
        normals=directions[:-1]
        informative=[r for r in normals if r['initial_normal_nontrivial_fraction']>=.8 and r['normal_gain']['count']>0]
        passing=[r for r in informative if r['normal_gain']['median']<=.5 and r['normal_over_tangent_gain']['median']<=.5]
        records[str(horizon)]=dict(directions=directions,normal_directions_total=len(normals),
            informative_normal_directions=len(informative),normal_directions_passing_old_median_gains=len(passing),
            all_19_normal_directions_pass=bool(len(passing)==len(normals)),
            baseline_common_mean_shift_over_initial_radius=float(np.linalg.norm(baseline.mean(0)-reference.mean(0))/max(radius,1e-30)),
            baseline_radius_retention=float(flow.curve_geometry(baseline)['radius']/max(radius,1e-30)),
            tangent_gain=summary(tangent_gain,tangent_valid))
    return dict(observable_radius=radius,observable_dimensions=reference.shape[1],horizons=records),arrays


@torch.inference_mode()
def finite_probe(transition,curve,scale,output,count=384,anchor_count=8):
    starts,metadata=make_starts(curve,np.asarray(scale),count,anchor_count)
    state=torch.tensor(starts,dtype=torch.float64)
    needed=set(OFFSETS)|{t+h for h in EVOLUTIONS for t in OFFSETS}
    observed={}
    for t in range(max(needed)+1):
        if t in needed:observed[t]=output(state).detach().cpu().numpy().copy()
        if t<max(needed):state=transition(state)
    result,arrays=measure_observations(observed,metadata)
    result['perturbation_design']=metadata
    arrays['full_state_starts']=starts
    # Retain full observed baseline/kick values for an independent recomputation.
    arrays.update({f'output_t{t}':value for t,value in observed.items()})
    return result,arrays


def synthetic_controls():
    phase=np.arange(128)*2*np.pi/128
    curve=np.zeros((128,6));curve[:,0]=np.cos(phase);curve[:,1]=np.sin(phase)
    def output(z):return z[:,[0,1,2,4,5]]
    def ring(z):
        out=z.clone();r=z[:,:2].norm(dim=1,keepdim=True)
        out[:,:2]=z[:,:2]/r*(1+.8*(r-1))
        out[:,2:]*=.8;out[:,3]=z[:,3]+.01
        return out
    def plane(z):return .95*z
    positive,_=finite_probe(ring,curve,np.ones(6),output,count=128)
    negative,_=finite_probe(plane,curve,np.ones(6),output,count=128)
    good=positive['horizons']['30'];bad=negative['horizons']['30']
    assert good['all_19_normal_directions_pass']
    assert good['tangent_gain']['median']>.9
    radial=next(r for r in bad['directions'] if r['name']=='existing_radial')
    assert radial['normal_over_tangent_gain']['median']>.9
    assert not bad['all_19_normal_directions_pass']
    return dict(analytic_ring_with_unobserved_clock=positive,shrinking_plane=negative,
                interpretation='Checks only; the analytic ring is not evidence about trained navigation networks.')


def candidate_rows(parent):
    return [row for model in parent['results'] for row in model['rows']
            if row['centered_candidate'] or row['noncentered_candidate']]


def run_case(row,args,net):
    name=label(row);folder=args.out/name
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    folder.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    archive=np.load(primary.ROOT/f"results/lstm_module_phase_ring/{row['condition']}_s{row['seed']}/states.npz",allow_pickle=False)
    regime=primary.REGIMES.index(row['regime']);family=row['module']*2+row['axis']
    curve=archive[f"states_{row['anchor_updates']}"][regime,family]
    scale=np.ones(512) if row['metric']=='raw_full_state_kicks' else np.repeat(archive['scales'][family],256)
    native=np.load(ROOT/f"results/original_goal_bearing_manifold/pilot/{row['condition']}_s{row['seed']}/context0_states.npz",allow_pickle=False)
    x=torch.tensor(native['history_x'][14:15],dtype=torch.float64)
    if row['regime']=='all_goal_withheld':x[:,768:1280]=0
    drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
    transition=lambda z:advance(net.lstm,drive,z)
    def output(z):
        if row['observable']=='all_hidden':return z[:,:256]
        logits=net.pi(z[:,:256]);return logits-logits.mean(1,keepdim=True)
    result,arrays=finite_probe(transition,curve,scale,output)
    result.update(case=name,original_discovery_centered_candidate=row['centered_candidate'],
        original_discovery_noncentered_candidate=row['noncentered_candidate'],original_case={k:row[k]for k in
        ('condition','seed','module','axis','regime','anchor_updates','metric','observable')},seconds=time.monotonic()-began)
    np.savez_compressed(folder/'all_perturbations.npz',**arrays)
    primary.save(folder/'result.json',result)
    print(json.dumps(dict(case=name,seconds=result['seconds'],passes={h:r['normal_directions_passing_old_median_gains'] for h,r in result['horizons'].items()})),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/lstm_module_phase_normal_audit')
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    parent=json.loads((PARENT/'summary.json').read_text());rows=candidate_rows(parent)
    design=dict(parent_sha256=primary.sha(PARENT/'summary.json'),candidate_ids=[label(r)for r in rows],
        case_count=len(rows),cohort='Every centered OR noncentered positive case from completed four-model fixed-observable screen; original labels unchanged.',
        baseline_points=384,uniform_arc_anchor_points=8,seed=RNG_SEED,
        directions='16 independent seeded full-state random vectors projected off slice tangent; one hidden-only and one cell-only similarly projected within their block; existing radial and tangent controls.',
        full_state_amplitude='Both signs,2percent of originalfullstatecurve radius in the original case raw/RMS metric.',
        observation_offsets=list(OFFSETS),evolution_updates=list(EVOLUTIONS),
        comparison='Distance to contemporaneous evolved full-dimensional observable baseline polygon. No independent centering of perturbed values. Arc displacement is observed polygon arclength, not a decoded physical angle.',
        guards='Initial normal displacement>=1e-4 observable radius; paired tangent displacement>=1e-4 radius. Retain all raw gains; report uninformative kicks separately.',
        summaries='Per-direction median normalgain<=.5 and relative-to-tangent gain<=.5, alongsidep90/max andeveryfinitekick. Broad pass requiresall19 testednormaldirections informative in80percent andmeetingmedianconditions.',
        limitations='Finite set of normals, not exhaustive contraction; a surviving transverse mode may be a second torus phase. Synthetic cross-module elicitation and near-saturated actor readout remain.',
        source_sha256={p:primary.sha(ROOT/p)for p in ['scripts/lstm_module_phase_normal_audit.py','scripts/lstm_heading_curve_flow.py',
            'scripts/lstm_module_phase_observable.py','scripts/lstm_endogenous_observable_flow.py','scripts/lstm_endogenous_curve_recovery.py']})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==design,'Changed frozen design; use a new output directory'
    else:primary.save(path,design)
    with threadpool_limits(limits=1):
        controls=synthetic_controls();primary.save(args.out/'controls.json',controls)
        if args.controls_only:return
        nets={};results=[]
        for row in rows:
            key=row['condition'],row['seed']
            if key not in nets:
                _,_,net,_=primary.base.load_model(*key,'cpu');net.double();nets[key]=net
            results.append(run_case(row,args,nets[key]))
        primary.save(args.out/'summary.json',dict(case_count=len(results),results=results,
            broad_pass_counts={str(h):sum(r['horizons'][str(h)]['all_19_normal_directions_pass']for r in results)for h in EVOLUTIONS}))


if __name__=='__main__':main()
