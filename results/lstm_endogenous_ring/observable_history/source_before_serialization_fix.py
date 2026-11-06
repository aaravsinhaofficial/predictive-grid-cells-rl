"""Secondary conditional-ring-factor control in fixed observable coordinates.

Phi(z) contains EVERY hidden output at four future times, not a fitted plane.
Only a common phase-family mean is removed in the centered sensitivity. The
primary full-state screen is unchanged. Shared drift and phase-dependent drift
are kept distinct, and actual full-state radial/tangent kicks remain causal.
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
from code.common import build_policy
import lstm_heading_curve_flow as flow
from lstm_endogenous_curve_recovery import advance,describe,sha,save

OBSERVATIONS=(0,5,10,15)
EVOLUTION=30


def observable_probe(transition,curve,scale,output,actor=None,count=384):
    """Generic full-state perturbations; output is fixed before any results."""
    curve=np.asarray(curve,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    scaled=curve/scale;points=flow.uniform_arc_samples(scaled,count)
    full_geometry=flow.curve_geometry(scaled)
    tangent=np.roll(points,-1,0)-np.roll(points,1,0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-30)
    radial=points-points.mean(0);radial-=np.sum(radial*tangent,axis=1)[:,None]*tangent
    radial_norm=np.linalg.norm(radial,axis=1)
    radial/=np.maximum(radial_norm[:,None],1e-30)
    amplitude=.02*full_geometry['radius']
    starts=np.concatenate((points,points+amplitude*radial,points-amplitude*radial,points+amplitude*tangent,points-amplitude*tangent))*scale
    state=torch.tensor(starts,dtype=torch.float64)
    observed={};actor_values={}
    needed=set(OBSERVATIONS)|{EVOLUTION+t for t in OBSERVATIONS}
    with torch.no_grad():
        for step in range(EVOLUTION+max(OBSERVATIONS)+1):
            if step in needed:
                observed[step]=output(state).detach().numpy()
                if actor is not None:actor_values[step]=actor(state).detach().numpy()
            if step<EVOLUTION+max(OBSERVATIONS):state=transition(state)
    first=np.concatenate([observed[t]for t in OBSERVATIONS],1)
    last=np.concatenate([observed[EVOLUTION+t]for t in OBSERVATIONS],1)
    initial=first[:count];evolved=last[:count]
    geometry=flow.curve_geometry(initial);radius=geometry['radius']
    _,initial_phase,_,_=flow.nearest_closed_segments(initial,initial)
    initial_radial=[];radial_gain=[];tangent_gain=[];radial_total=[];actor_rows={}
    for j in (1,2):
        begin=first[j*count:(j+1)*count];end=last[j*count:(j+1)*count]
        d0=flow.nearest_closed_segments(begin,initial)[0]
        d1=flow.nearest_closed_segments(end,evolved)[0]
        initial_radial.append(d0)
        radial_gain.append(d1/np.maximum(d0,1e-30))
        radial_total.append(np.linalg.norm(end-evolved,axis=1)/np.maximum(np.linalg.norm(begin-initial,axis=1),1e-30))
    for j in (3,4):
        d0=np.linalg.norm(first[j*count:(j+1)*count]-initial,axis=1)
        d1=np.linalg.norm(last[j*count:(j+1)*count]-evolved,axis=1)
        tangent_gain.append(d1/np.maximum(d0,1e-30))
    initial_radial=np.asarray(initial_radial);radial_gain=np.asarray(radial_gain);tangent_gain=np.asarray(tangent_gain)
    nontrivial=initial_radial>=1e-4*max(radius,1e-30)
    tangent_nontrivial=np.stack([np.linalg.norm(first[j*count:(j+1)*count]-initial,axis=1)>=1e-4*max(radius,1e-30)for j in (3,4)])
    if actor is not None:
        for label,offset in [('reference',0),('evolved',EVOLUTION)]:
            values=np.concatenate([actor_values[offset+t]for t in OBSERVATIONS],1)
            baseline=values[:count]
            actor_rows[label]=dict(family_probability_RMS=float(np.sqrt(np.mean((baseline-baseline.mean(0))**2))),
                radial_probability_change_max=float(np.abs(values[count:3*count]-np.tile(baseline,(2,1))).max()),
                tangent_probability_change_max=float(np.abs(values[3*count:5*count]-np.tile(baseline,(2,1))).max()))
    invariance={};gates={}
    evolved_geometry=flow.curve_geometry(evolved)
    for label,center in [('noncentered',False),('centered',True)]:
        ref=initial-initial.mean(0) if center else initial
        current=evolved-evolved.mean(0) if center else evolved
        distance,phase,_,_=flow.nearest_closed_segments(current,ref)
        transport=flow.phase_transport(phase,initial_phase)
        invariance[label]=dict(residual_to_reference_curve_over_radius=describe(distance/max(radius,1e-30)),phase_transport=transport,
            reference_radius=radius,evolved_radius_retention=evolved_geometry['radius']/max(radius,1e-30))
        gates[label]=dict(observable_radius_RMS_at_least1e_minus4=radius/np.sqrt(initial.shape[1])>=1e-4,
            max_edge_below10percent_radius=geometry['max_edge_over_radius']<=.1,
            nonlocal_separation_above5percent=geometry['nonlocal_distance_over_radius']['p10']>=.05,
            evolved_radius_retains5percent=evolved_geometry['radius']/max(radius,1e-30)>=.05,
            p90_invariance_residual_below5percent=float(np.quantile(distance,.9))/max(radius,1e-30)<=.05,
            phase_coverage80percent=transport['circular_coverage']>=.8,
            phase_degree_one=abs(transport['winding_degree']-1)<.1,
            phase_order_preserved=transport['backward_edge_fraction']<=.05,
            observable_radial_kick_nontrivial_in80percent=float(nontrivial.mean())>=.8,
            observable_tangent_kick_nontrivial_in80percent=float(tangent_nontrivial.mean())>=.8,
            radial_gain_at_most_half=float(np.median(radial_gain))<=.5,
            radial_over_tangent_at_most_half=float(np.median(radial_gain/np.maximum(tangent_gain.mean(0)[None,:],1e-30)))<=.5)
    return dict(observation_offsets=OBSERVATIONS,evolution_updates=EVOLUTION,full_state_dimensions=curve.shape[1],observable_dimensions=initial.shape[1],
        full_state_geometry=full_geometry,observable_geometry=geometry,evolved_observable_geometry=evolved_geometry,
        initial_radial_observable_distance_over_radius=describe(initial_radial/max(radius,1e-30)),
        radial_observable_normal_gain=describe(radial_gain),radial_observable_total_gain=describe(radial_total),tangent_observable_total_gain=describe(tangent_gain),
        radial_over_tangent_observable_gain=describe(radial_gain/np.maximum(tangent_gain.mean(0)[None,:],1e-30)),
        radial_direction_norm_before_normalizing_over_radius=describe(radial_norm/max(full_geometry['radius'],1e-30)),
        common_observable_mean_shift_over_radius=float(np.linalg.norm(evolved.mean(0)-initial.mean(0))/max(radius,1e-30)),
        invariance=invariance,actor=actor_rows,candidate_gates=gates,centered_conditional_factor_candidate=all(gates['centered'].values()),
        noncentered_observable_candidate=all(gates['noncentered'].values()))


def load_net(checkpoint):
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=data.get('config') or json.loads(checkpoint.with_suffix('.json').read_text())
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy']);net.double()
    return net,data,cfg


def archive_probe(checkpoint,archive_path,out,continued=False):
    net,data,cfg=load_net(checkpoint)
    if continued:assert data.get('completed') and data['extra_frames']==25_024_000
    a=np.load(archive_path);times=list(a['times']);idx=times.index(15)
    if 'native_states' in a:
        variants=[('native_current','native_states','native_input'),('ordinary_current','ordinary_states','ordinary_input')]
    else:variants=[('saved_input','states','dark_input')]
    record=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),archive=str(archive_path),archive_sha256=sha(archive_path),
        source_sha256=sha(__file__),condition=cfg['cond'],continued=continued,reference_updates=15,latest_update=60,branches={})
    for label,state_key,input_key in variants:
        curve=a[state_key][idx].astype(np.float64);initial=a[state_key][0].astype(np.float64)
        scale=np.repeat(np.sqrt(np.mean(initial.reshape(-1,2,256)**2,axis=(0,2))),256)
        scale=np.maximum(scale,1e-8)
        x=torch.tensor(a[input_key],dtype=torch.float64)
        drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
        transition=lambda z:advance(net.lstm,drive,z)
        output=lambda z:z[:,:256]
        actor=lambda z:net.pi(z[:,:256]).softmax(1)
        record['branches'][label]={name:observable_probe(transition,curve,s,output,actor)for name,s in [('raw_full_state_kicks',np.ones(512)),('h_c_RMS_full_state_kicks',scale)]}
    out.parent.mkdir(parents=True,exist_ok=True);save(out,record)
    print(json.dumps(dict(archive=str(archive_path),conditional_candidates={name:{metric:r['centered_conditional_factor_candidate']for metric,r in branch.items()}for name,branch in record['branches'].items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/observable_history')
    p.add_argument('--originals',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--label',default='continued');p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(role='Secondary conditional ring-factor control; primary full-state outcomes unchanged.',
        observed_coordinates='All 256 hidden outputs at 0,5,10,15 future updates; actor probabilities reported separately. No fit, learned decoder, selected neuron subset, or 2D projection.',
        times='Saved curve at 15 updates (1 s), evolve 30 updates (2 s), observe for 15 more updates (1 s); latest state at 60 updates (4 s).',
        perturbations='384 uniform full-state arc points; centered radial projected off full tangent and full tangent, each ±2% full-curve radius. Same physical starts for centered/noncentered reports; raw and h/c-RMS constructions separately.',
        centering='Remove only the common phase-family mean from each unperturbed observable curve. Perturbations share the baseline center; never independently center perturbed families.',
        guard='At least 80% initial radial/tangent kicks must produce observable displacement >=1e-4 reference radius; observable family RMS radius >=1e-4 hidden units.',
        nomination='Closure/separation/coverage/order gates as primary; centered p90 fixed-curve residual <=5% radius, radial gain <=.5, radial/tangent gain <=.5. Both physical kick metrics reported; no behavioral ring claim from nomination.',
        source_sha256=sha(__file__),parents={p:sha(ROOT/p)for p in ['scripts/lstm_heading_curve_flow.py','scripts/lstm_endogenous_curve_recovery.py']})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    if args.freeze_only:return
    with threadpool_limits(limits=2):
        if args.originals:
            for condition in ('gc100','pgc33'):
                for seed in (1,2):
                    checkpoint=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt'
                    archive=ROOT/f'results/lstm_heading_memory/curve_flow/{condition}_s{seed}/context0_states.npz'
                    archive_probe(checkpoint,archive,args.out/f'original/{condition}_s{seed}/native.json')
                    if condition=='pgc33':
                        archive=ROOT/f'results/lstm_heading_memory/current_release/{condition}_s{seed}/context0_states.npz'
                        archive_probe(checkpoint,archive,args.out/f'original/{condition}_s{seed}/ordinary.json')
        else:
            if args.checkpoint is None or args.archive is None:p.error('--checkpoint and --archive required without --originals')
            archive_probe(args.checkpoint,args.archive,args.out/args.label/f'{args.archive.stem}.json',continued=True)


if __name__=='__main__':main()
