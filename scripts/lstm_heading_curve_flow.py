"""Decoder-free, full-state slow-curve classification of original heading families.

Preselected context0, seeds1/2, both populations. Real visible heading histories
initialize the original LSTM. Long common-input evolution and uniform-arc probes
test whether a dense observed closed family is approximately invariant.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from lstm_heading_memory_probe import elicit,sha,save,wrap
from lstm_endogenous_curve_recovery import advance,describe

SAMPLES=(0,1,15,75,150,450,1500)
ANCHORS=(75,450,1500)
FLOW=(1,15,75)


def curve_geometry(curve):
    """Closed observed polygon; large bridges or folding remain explicit."""
    x=np.asarray(curve,dtype=np.float64)
    centered=x-x.mean(0); radius=float(np.sqrt(np.mean(np.sum(centered**2,1))))
    segment=np.roll(x,-1,axis=0)-x;length=np.linalg.norm(segment,axis=1)
    arc=np.r_[0.,np.cumsum(length)];total=float(arc[-1])
    # Nonlocal points have >=10% separation along the observed polygon.
    d2=np.maximum(np.sum(x*x,1)[:,None]+np.sum(x*x,1)[None,:]-2*x@x.T,0)
    arcdiff=np.abs(arc[:-1,None]-arc[None,:-1]); arcdiff=np.minimum(arcdiff,total-arcdiff)
    d2[arcdiff<.1*max(total,1e-30)]=np.inf
    nonlocal_distance=np.sqrt(d2.min(1))
    nonlocal_distance=np.where(np.isfinite(nonlocal_distance),nonlocal_distance,0.)
    singular=np.linalg.svd(centered,compute_uv=False);v=singular**2
    frac=v/max(float(v.sum()),1e-40)
    cosine=np.sum(segment*np.roll(segment,1,0),1)/np.maximum(length*np.roll(length,1),1e-30)
    return dict(radius=radius,arc_length=total,
                max_edge_over_radius=float(length.max()/max(radius,1e-30)),
                max_edge_over_arc=float(length.max()/max(total,1e-30)),
                closing_edge_over_radius=float(length[-1]/max(radius,1e-30)),
                edge_length_quantiles=np.quantile(length,[0,.1,.5,.9,1]).tolist(),
                consecutive_segment_cosine=describe(cosine),
                nonlocal_distance_over_radius=describe(nonlocal_distance/max(radius,1e-30)),
                covariance_participation_ratio=float(1/max(float((frac**2).sum()),1e-40)) if v.sum()>1e-40 else 0.,
                first_five_pc_fractions=frac[:5].tolist())


def nearest_closed_segments(points,curve):
    """Exact nearest point on every segment of a full-dimensional polygon.

    Uses pairwise dot products, never a PCA/readout projection. Returns distance,
    normalized arc position, closest full-state point and segment identity.
    """
    points=np.asarray(points,dtype=np.float64);curve=np.asarray(curve,dtype=np.float64)
    # Centering suppresses cancellation when the family is small around a large
    # common state, particularly for long-lived cell-state baselines.
    center=curve.mean(0);p=points-center;s=curve-center
    d=np.roll(s,-1,axis=0)-s; d2=np.sum(d*d,1)
    projection=(p@d.T-np.sum(s*d,1)[None,:])/np.maximum(d2[None,:],1e-40)
    t=np.clip(projection,0,1)
    distances=np.maximum(np.sum(p*p,1)[:,None]+np.sum(s*s,1)[None,:]-2*p@s.T
                         -2*t*(p@d.T-np.sum(s*d,1)[None,:])+t*t*d2[None,:],0)
    segment=distances.argmin(1);idx=np.arange(len(p));fraction=t[idx,segment]
    closest=curve[segment]+fraction[:,None]*d[segment]
    length=np.sqrt(d2);cumulative=np.r_[0.,np.cumsum(length)]
    phase=(cumulative[segment]+fraction*length[segment])/max(float(cumulative[-1]),1e-30)
    return np.linalg.norm(points-closest,axis=1),np.mod(phase,1),closest,segment


def uniform_arc_samples(curve,count):
    x=np.asarray(curve,dtype=np.float64)
    d=np.roll(x,-1,axis=0)-x;length=np.linalg.norm(d,axis=1)
    cumulative=np.r_[0.,np.cumsum(length)];total=cumulative[-1]
    target=np.arange(count)*total/count
    index=np.searchsorted(cumulative,target,side='right')-1
    index=np.clip(index,0,len(x)-1)
    f=(target-cumulative[index])/np.maximum(length[index],1e-40)
    return x[index]+f[:,None]*d[index]


def phase_transport(phase,initial_phase):
    phase=np.asarray(phase);initial_phase=np.asarray(initial_phase)
    delta=wrap((phase-initial_phase)*2*np.pi)
    ordered=np.sort(np.mod(phase,1));largest=float(np.diff(np.r_[ordered,ordered[0]+1]).max())
    steps=wrap((np.roll(phase,-1)-phase)*2*np.pi)/(2*np.pi)
    return dict(mean_displacement_rad=float(np.angle(np.exp(1j*delta).mean())),
                displacement_concentration=float(abs(np.exp(1j*delta).mean())),
                absolute_displacement_rad=describe(np.abs(delta)),
                circular_coverage=1-largest,
                winding_degree=float(steps.sum()),
                backward_edge_fraction=float((steps<-1e-5).mean()),
                occupied_arc_bins_32=int(len(np.unique(np.floor(phase*32).astype(int)%32))))


def invariance_probe(cell,drive,curve,scale,count,transition=None):
    scaled=curve/scale
    geometry=curve_geometry(scaled)
    points=uniform_arc_samples(scaled,count)
    _,initial_phase,_,_=nearest_closed_segments(points,scaled)
    tangent=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    tangent_norm=np.linalg.norm(tangent,axis=1)
    unit=tangent/np.maximum(tangent_norm[:,None],1e-30)
    radial=points-points.mean(0)
    radial-=np.sum(radial*unit,axis=1)[:,None]*unit
    radial_norm=np.linalg.norm(radial,axis=1)
    radial/=np.maximum(radial_norm[:,None],1e-30)
    amplitude=.02*geometry['radius']
    starts=np.concatenate((points,points+amplitude*radial,points-amplitude*radial))
    initial_radial_distance=np.stack([nearest_closed_segments(starts[j*count:(j+1)*count],points)[0] for j in (1,2)])
    state=torch.tensor(starts*scale,dtype=torch.float64)
    rows={}
    with torch.no_grad():
        for k in range(1,max(FLOW)+1):
            state=advance(cell,drive,state) if transition is None else transition(state)
            if k in FLOW:
                all_evolved=state.numpy()/scale
                evolved=all_evolved[:count]
                distance,phase,closest,segment=nearest_closed_segments(evolved,scaled)
                displacement=np.linalg.norm(evolved-points,axis=1)
                row=dict(residual_to_fixed_curve_over_radius=describe(distance/max(geometry['radius'],1e-30)),
                         residual_over_total_displacement=describe(distance/np.maximum(displacement,1e-30)),
                         total_displacement_over_radius=describe(displacement/max(geometry['radius'],1e-30)),
                         phase_transport=phase_transport(phase,initial_phase))
                evolved_tangent=np.roll(evolved,-1,axis=0)-np.roll(evolved,1,axis=0)
                tangent_gain=np.linalg.norm(evolved_tangent,axis=1)/np.maximum(tangent_norm,1e-30)
                radial_gains=[]
                for j in (1,2):
                    residual=nearest_closed_segments(all_evolved[j*count:(j+1)*count],evolved)[0]
                    radial_gains.append(residual/np.maximum(initial_radial_distance[j-1],1e-30))
                row['radial_normal_gain']=describe(np.array(radial_gains))
                row['observed_tangent_length_gain']=describe(tangent_gain)
                row['radial_over_tangent_gain']=describe(np.array(radial_gains)/np.maximum(tangent_gain[None,:],1e-30))
                rows[str(k)]=row
    return dict(geometry=geometry,horizons=rows,radial_amplitude_over_radius=.02,
                radial_direction_norm_before_normalizing_over_radius=describe(radial_norm/max(geometry['radius'],1e-30)))


def candidate_screen(record):
    """Prespecified 30s reference, 5s evolution; decoder-independent nomination."""
    checks={}
    for metric in ('raw','relative_rms'):
        r=record['invariance']['450'][metric];g=r['geometry'];f=r['horizons']['75']
        checks[metric]=dict(family_retains_5percent_initial_radius=record['trajectory']['450'][metric]['radius_retention']>=.05,
            no_large_closing_bridges=g['max_edge_over_radius']<=.10,
            nonlocal_segments_separated=g['nonlocal_distance_over_radius']['p10']>=.05,
            fixed_curve_residual_p90_at_most_5percent=f['residual_to_fixed_curve_over_radius']['p90']<=.05,
            projected_phase_coverage_at_least_80percent=f['phase_transport']['circular_coverage']>=.8,
            projected_phase_map_degree_one=abs(f['phase_transport']['winding_degree']-1)<.1,
            projected_phase_order_preserved=f['phase_transport']['backward_edge_fraction']<=.05,
            targeted_radial_gain_at_most_half=f['radial_normal_gain']['median']<=.5,
            radial_recovers_twice_as_fast_as_tangent=f['radial_over_tangent_gain']['median']<=.5)
    return checks


def model_context(condition,seed,args):
    started=time.monotonic();context=0
    stem=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    checkpoint=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    net=build_policy(cfg,args.embed_device).eval().requires_grad_(False)
    net.load_state_dict(checkpoint['policy'],strict=True)
    source=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'
    archive=np.load(source);pop=build_population(cfg,'cpu')
    elicitation=SimpleNamespace(angles=args.phases//2,prepare=30,embed_device=args.embed_device)
    state,dark_x,_,_,physics=elicit(net,pop,cfg,archive,context,elicitation)
    net.cpu().double();cell=net.lstm
    drive=torch.nn.functional.linear(dark_x,cell.weight_ih,cell.bias_ih)
    initial=state.clone();scales=torch.stack([state[:,:256].square().mean().sqrt(),state[:,256:].square().mean().sqrt()]).clamp_min(1e-8)
    relative=scales.repeat_interleave(256).numpy()
    metric_scales={'raw':np.ones(512),'relative_rms':relative}
    initial_geometry={name:curve_geometry(initial.numpy()/s) for name,s in metric_scales.items()}
    snapshots={};trajectory={};max_speed={name:np.zeros(args.phases) for name in metric_scales}
    with torch.no_grad():
        for step in range(max(SAMPLES)+1):
            following=advance(cell,drive,state)
            delta=(following-state).numpy()
            for name,s in metric_scales.items():max_speed[name]=np.maximum(max_speed[name],np.linalg.norm(delta/s,axis=1))
            if step in SAMPLES:
                z=state.numpy().copy();snapshots[step]=z
                row={}
                for name,s in metric_scales.items():
                    geom=curve_geometry(z/s);geom['radius_retention']=geom['radius']/initial_geometry[name]['radius']
                    geom['step_speed_over_radius']=describe(np.linalg.norm(delta/s,axis=1)/max(geom['radius'],1e-30))
                    geom['step_speed_over_trajectory_max_so_far']=describe(np.linalg.norm(delta/s,axis=1)/np.maximum(max_speed[name],1e-30))
                    row[name]=geom
                h=z[:,:256];c=z[:,256:];h0=initial[:,:256].numpy();c0=initial[:,256:].numpy()
                logits=net.pi(state[:,:256]);probabilities=logits.softmax(1)
                row['behavioral_expression']=dict(
                    hidden_radius_retention=float(np.linalg.norm(h-h.mean(0))/np.linalg.norm(h0-h0.mean(0))),
                    cell_radius_retention=float(np.linalg.norm(c-c.mean(0))/np.linalg.norm(c0-c0.mean(0))),
                    action_logit_family_rms=float((logits-logits.mean(0)).square().mean().sqrt()),
                    action_probability_family_rms=float((probabilities-probabilities.mean(0)).square().mean().sqrt()),
                    value_family_sd=float(net.v(state[:,:256]).std()))
                trajectory[str(step)]=row
                print(f'{condition} s{seed} flow {step/15:g}s',flush=True)
            state=following
    invariance={}
    for anchor in ANCHORS:
        invariance[str(anchor)]={name:invariance_probe(cell,drive,snapshots[anchor],scale,args.remesh) for name,scale in metric_scales.items()}
    # Does later observed activity stay on an earlier *fixed* curve? This
    # complements remeshed self-invariance with actual heading-history states.
    cross_time={}
    for anchor,future in ((75,450),(75,1500),(450,1500)):
        key=f'{anchor}_to_{future}';cross_time[key]={}
        for name,s in metric_scales.items():
            curve=snapshots[anchor]/s;points=snapshots[future]/s
            radius=trajectory[str(anchor)][name]['radius']
            distance,phase,_,_=nearest_closed_segments(points,curve)
            _,phase0,_,_=nearest_closed_segments(curve,curve)
            cross_time[key][name]=dict(residual_to_earlier_curve_over_radius=describe(distance/max(radius,1e-30)),
                                       projected_phase_transport=phase_transport(phase,phase0))
    record=dict(condition=condition,seed=seed,context=0,phases=args.phases,remesh_points=args.remesh,
        step_seconds=cfg['action_repeat']/60.,checkpoint_sha256=sha(stem.with_suffix('.pt')),
        context_archive_sha256=sha(source),source_sha256=sha(__file__),physics_checks=physics,
        trajectory=trajectory,invariance=invariance,cross_time=cross_time,
        explanation='A closed parameterized image of initial headings is not presumed an invariant ring. Dense full-state polygons are checked for closure bridges, nonlocal folding, fixed-curve invariance and phase transport. The uniform-arc starting states interpolate nearby observed states; interpolation gap metrics are reported. No phase decoder enters this analysis.',
        seconds=time.monotonic()-started)
    record['candidate_gates']=candidate_screen(record)
    record['passes_both_metric_candidate_screen']=all(all(g.values()) for g in record['candidate_gates'].values())
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    save(folder/'context0.json',record)
    np.savez_compressed(folder/'context0_states.npz',times=np.array(SAMPLES),states=np.stack([snapshots[t] for t in SAMPLES]),relative_rms_scale=relative,dark_input=dark_x.numpy())
    print(json.dumps(dict(condition=condition,seed=seed,seconds=record['seconds'],candidate=record['passes_both_metric_candidate_screen'],
                          raw_30s_5s_residual_p90=invariance['450']['raw']['horizons']['75']['residual_to_fixed_curve_over_radius']['p90'],
                          relative_30s_5s_residual_p90=invariance['450']['relative_rms']['horizons']['75']['residual_to_fixed_curve_over_radius']['p90'])),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases',type=int,default=768);p.add_argument('--remesh',type=int,default=384)
    p.add_argument('--embed-device',default='mps');p.add_argument('--out',type=Path,default=ROOT/'results/lstm_heading_memory/curve_flow')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    assert args.phases>=192 and args.phases%4==0
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(experiment='Decoder-independent original-LSTM heading slow-curve classification',
        models=[(c,s) for c in ('gc100','pgc33') for s in (1,2)],context=0,
        phases=args.phases,uniform_arc_probe_points=args.remesh,visible_updates=30,flow_samples=SAMPLES,
        reference_updates=ANCHORS,invariance_horizons=FLOW,
        primary='At30s, take the dense full512-state observed family and interpolate uniform-arc points; evolve5s under identical native black-image/pure-turn/current-code/goal-code input, then project onto the FIXED30s curve using exact full-dimensional nearest line segments. Repeat raw and initial h/c-RMS-relative metrics.',
        candidate_criteria='In BOTH metrics: radius>=5% of initial; largest edge<=10% of radius; p10 nearest nonlocal(>=10%arc) separation>=5%radius; after5s p90 residual-to-fixed-curve<=5%radius; projected arc coverage>=80%; phase-map winding degree within.1of1; reversed phase edges<=5%; targeted radial gain<=.5 and radial/tangent gain<=.5. Nomination for spectral/attraction validation, not a proven attractor.',
        radial_test='At each reference curve, uniform-arc centered-radius vectors are projected off the local tangent. Inward and outward kicks of2%radius are evolved alongside the baseline. Measure distance to the EVOLVED baseline curve and divide by initial distance; compare with observed tangent-length gain. This rejects simple isotropically shrinking planes even if most random512-dimensional normals contract.',
        guards='No2D readout; no target circle is fitted. Closure is a measured image of the circular physical heading family and can fail bridge/folding tests. Report hidden/cell/actor variation to expose behaviorally silent residual cell memory. All outcomes retained.',
        literature='Ságodi etal arXiv2408.00109v2 S7.6.3/S7.7 uses long autonomous trajectories, a slow-speed cutoff, periodic spline reconstruction, flow reversals and a spectral gap. Here the known initial heading order gives an independent curve parameter; full-state nearest-segment invariance tests replace output-angle reconstruction. The projection itself is not proof of topology.',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_heading_curve_flow.py','scripts/lstm_heading_memory_probe.py','scripts/lstm_endogenous_curve_recovery.py','code/env.py','code/nets.py','code/codes.py']})
    frozen=args.out/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==json.loads(json.dumps(protocol)),'Frozen protocol/source changed; choose a new output directory'
    else:save(frozen,protocol)
    if args.freeze_only:return
    results=[model_context(c,s,args) for c in ('gc100','pgc33') for s in (1,2)]
    save(args.out/'summary.json',dict(candidates=sum(r['passes_both_metric_candidate_screen'] for r in results),
                                    results=[{k:r[k] for k in ('condition','seed','candidate_gates','passes_both_metric_candidate_screen')} for r in results]))


if __name__=='__main__':main()
