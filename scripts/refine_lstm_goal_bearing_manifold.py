"""Descriptive precision/coordinate audit of the original-agent goal-family pilot.

Uses all cached pilot contexts; no new selected behavioral outcomes. Exact
angular JVP defines a conditional state-family tangent, not an assumed ring.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import lstm_goal_bearing_manifold as m


def elicited(net,pop,inputs,initial,observer,theta,offset,ordinary):
    u=torch.stack((theta.cos(),theta.sin()),1)
    direction=torch.stack(((theta+offset).cos(),(theta+offset).sin()),1)
    pickup=observer+.75*u-.29*direction
    goal=pop.activity(pickup,direction,direction*.35,shift_scale=0 if ordinary else 1)
    return m.condition_states(net,inputs,initial,goal),goal


def mean_errors(branch,scale,n):
    z=branch.reshape(9,n,512)
    dif=(z[1:]-z[:1])/scale
    return dif.square().sum(2).sqrt()


def semantics(archive,variant):
    # Arrival0 has the same center/pickup/ahead bearing; test ±90 without refitting.
    train=archive[f'offset0_{variant}_states'].astype(np.float64)
    theta=archive['theta'];center=train.mean(0);x=train-center
    y=np.stack((np.cos(theta),np.sin(theta)),1)
    gram=x@x.T;alpha=np.trace(gram)/len(gram)*.01
    weights=x.T@np.linalg.solve(gram+alpha*np.eye(len(gram)),y)
    result={}
    for offset in (1,2):
        state=archive[f'offset{offset}_{variant}_states']
        pred=(state-center)@weights
        phase=np.arctan2(pred[:,1],pred[:,0])
        bearings=archive[f'offset{offset}_{variant}_bearings_center_pickup_ahead']
        result[str(offset)]={name:float(np.rad2deg(m.angular_error(phase,target)).mean()) for name,target in zip(('true_center','ordinary_pickup','predictive_point'),bearings)}
        result[str(offset)]['amplitude_median']=float(np.median(np.linalg.norm(pred,axis=1)))
    return result


def run(directory):
    started=time.perf_counter();summary=json.loads((directory/'summary.json').read_text())
    cfg,checkpoint,net,pop=m.load_model(summary['condition'],summary['seed'],'cpu')
    assert m.sha(checkpoint)==summary['checkpoint_sha256']
    results=[]
    for ci in range(len(summary['contexts'])):
        a=np.load(directory/f'context{ci}_states.npz')
        x=torch.tensor(a['history_x'])
        initial=torch.tensor(np.r_[a['history_h_before'][0],a['history_c_before'][0]])
        observer=torch.tensor(a['history_pos'][14],dtype=torch.float64)
        theta=torch.tensor(a['theta'],dtype=torch.float64)
        variants=['native','ordinary_goal'] if summary['condition']=='pgc33' else ['native']
        context=dict(context=ci,semantics={v:semantics(a,v) for v in variants},families=[])
        for offset_index,offset in enumerate(m.OFFSETS):
            for variant in variants:
                label=f'offset{offset_index}_{variant}';ordinary=variant=='ordinary_goal'
                fun=lambda angle:elicited(net,pop,x[:15],initial,observer,angle,offset,ordinary)[0]
                state,tangent=torch.autograd.functional.jvp(fun,theta,torch.ones_like(theta))
                goal=elicited(net,pop,x[:15],initial,observer,theta,offset,ordinary)[1].detach()
                prior=a[label+'_states'];identity=float((state-torch.tensor(prior)).abs().max())
                assert identity<3e-4,identity
                finite=(fun(theta+.001)-fun(theta-.001))/(.002)
                tangent_rel=float((finite-tangent).norm()/tangent.norm())
                assert tangent_rel<.015,tangent_rel
                base=state.detach()
                # Explicit relative-RMS metric prevents the numerically larger c block
                # from silently defining both the tangent and the noise control.
                scales=torch.stack((base[:,:256].square().mean().sqrt(),base[:,256:].square().mean().sqrt()))
                scale=torch.repeat_interleave(scales,256)
                tangent_scaled=tangent/scale
                tangent_scaled/=tangent_scaled.norm(dim=1,keepdim=True)
                normal=[]
                for k in range(3):
                    gen=torch.Generator().manual_seed(864201+ci*10+offset_index*100+k)
                    q=torch.randn(base.shape,generator=gen)
                    q-=torch.sum(q*tangent_scaled,1,keepdim=True)*tangent_scaled
                    q/=q.norm(dim=1,keepdim=True)
                    normal.append(q)
                directions=torch.stack((tangent_scaled,-tangent_scaled,*[s*q for q in normal for s in (1,-1)]))
                assert float(torch.max(torch.abs(torch.sum(torch.stack(normal)*tangent_scaled,2))))<2e-6
                amplitude=.01*np.sqrt(512)
                perturb=torch.cat((base[None],base[None]+amplitude*directions*scale),0)
                row=dict(variant=variant,offset_deg=float(np.rad2deg(offset)),native_reconstruction_max_error=identity,
                         angular_jvp_finite_difference_relative_error=tangent_rel,block_rms_scales=scales.tolist(),
                         metric='Euclidean after dividing each h/c block by its own absolute RMS across the elicited family.',
                         perturbation_relative_rms=.01,continuations={})
                with torch.no_grad():
                    for mode in ('goal_held','goal_withheld'):
                        branch=perturb.reshape(-1,512).clone()
                        initial_error=mean_errors(branch,scale,len(theta))
                        initial_raw=(perturb[1:]-perturb[:1]).square().sum(2).sqrt()
                        records={}
                        for t in range(16):
                            if t in m.HORIZONS:
                                current=branch.reshape(9,len(theta),512)
                                errors=mean_errors(branch,scale,len(theta))/initial_error
                                raw_errors=(current[1:]-current[:1]).square().sum(2).sqrt()/initial_raw
                                phase_state=current[0].numpy()
                                records[str(t)]=dict(tangent_error_gain=float(errors[:2].mean()),
                                    normal_error_gain=float(errors[2:].mean()),
                                    tangent_raw_coordinate_error_gain=float(raw_errors[:2].mean()),
                                    normal_raw_coordinate_error_gain=float(raw_errors[2:].mean()),
                                    newly_fitted_phase=m.phase_metrics(phase_state,a['theta'],m.fit_phase_readout(phase_state.astype(np.float64),a['theta'])))
                            if t==15:break
                            xx=x[15+t].repeat(9*len(theta),1)
                            xx[:,768:1280]=goal.repeat(9,1) if mode=='goal_held' else 0
                            branch=m.step_full(net,xx,branch)
                        row['continuations'][mode]=records
                    if ci==0:
                        dense_theta=torch.arange(768,dtype=torch.float64)*(2*np.pi/768)
                        dense=fun(dense_theta).numpy()
                        row['geometry_resolution']={str(n):m.family_geometry(dense[::768//n]) for n in (96,192,384,768)}
                        frozen=base.clone()
                        decoder=m.fit_phase_readout(base.numpy().astype(np.float64),a['theta'])
                        for _ in range(15):
                            frozen_x=x[14].repeat(len(theta),1);frozen_x[:,768:1280]=0
                            frozen=m.step_full(net,frozen_x,frozen)
                        row['artificial_fixed_input_goal_withheld']=dict(
                            note='The complete endpoint physical input vector is repeated15times; this is an artificial stationary-input diagnostic, not a valid native action trajectory.',
                            radius_retention=m.family_geometry(frozen.numpy())['radius']/m.family_geometry(base.numpy())['radius'],
                            fixed_readout=m.phase_metrics(frozen.numpy(),a['theta'],decoder),
                            newly_fitted_readout=m.phase_metrics(frozen.numpy(),a['theta'],m.fit_phase_readout(frozen.numpy().astype(np.float64),a['theta'])))
                context['families'].append(row)
        results.append(context)
        print(f'{directory.name}: refined context{ci}',flush=True)
    payload=dict(condition=summary['condition'],seed=summary['seed'],contexts=results,elapsed_seconds=time.perf_counter()-started,
                 source_sha256=m.sha(Path(__file__)),primary_pilot_summary_sha256=m.sha(directory/'summary.json'),
                 role='Descriptive angular-JVP, metric-scale, heldout-readout and resolution audit; no seed-level hypothesis tests.')
    m.save_json(directory/'refinement.json',payload)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/original_goal_bearing_manifold/pilot')
    args=ap.parse_args();torch.set_num_threads(2)
    for directory in sorted(args.root.glob('*_s*')):
        if (directory/'summary.json').exists():run(directory)

if __name__=='__main__':main()
