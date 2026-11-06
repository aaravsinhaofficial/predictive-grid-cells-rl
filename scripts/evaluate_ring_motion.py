"""Causal input replacement during motion through a learned periodic state set.

All replicas see the same physical trajectory. Each disturbed path is compared
with its own unperturbed input-condition baseline. This isolates amplification
of a state disturbance from the direct effect of changing spatial codes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ring_task_model import load_checkpoint, wrap_angle
from ring_dynamics import ring_radius, perturbations, tangents


@torch.no_grad()
def run(checkpoint,out,n_phase=128,steps=120,device='cpu',cue_gain=1.):
    model,task,ckpt=load_checkpoint(checkpoint,device)
    config=model.config
    phase=torch.arange(n_phase,device=device)*2*math.pi/n_phase
    variants=['native'] if ckpt['condition']=='grid' else ['native','ordinary','reverse','restore_at_15']
    results=[]
    arrays={}
    for speed in (-2.,-1.,-.5,.5,1.,2.):
        velocity=torch.full_like(phase,speed)
        def make_input(theta,condition,gain=None):
            measured=theta-config.delay_steps*config.dt*speed
            x=task.inputs(measured,velocity,torch.ones_like(theta),condition,
                          delayed_velocity=velocity,shuffled_velocity=-velocity)
            x[...,:config.n_code] *= cue_gain if gain is None else gain
            return x
        # Native warm-up lets all conditions start from the exact same state.
        h=torch.zeros(n_phase,model.hidden_size,device=device)
        for t in range(-100,0):
            theta=phase+t*config.dt*speed
            h=model.step(make_input(theta,ckpt['condition'],gain=1.),h)
        curve=h.cpu().numpy()
        radius=ring_radius(curve)
        tangent,normal=perturbations(curve,.02,seed=419)
        offsets=[torch.zeros_like(h),torch.as_tensor(tangent,device=device,dtype=h.dtype),
                 torch.as_tensor(normal,device=device,dtype=h.dtype)]
        state={variant:torch.cat([h+delta for delta in offsets]) for variant in variants}
        initial_distance={key:float(torch.linalg.norm(offsets[j],dim=-1).mean())
                          for j,key in enumerate(('none','tangent','normal'))}
        for t in range(steps):
            theta=phase+t*config.dt*speed
            x_native=make_input(theta,ckpt['condition'])
            x_plain=make_input(theta,'grid')
            native_previous=state['native'][:n_phase]
            for variant in variants:
                condition=ckpt['condition']
                if variant=='ordinary' or (variant=='restore_at_15' and t<15):
                    condition='grid'
                elif variant=='reverse':
                    condition='reverse'
                x=make_input(theta,condition).repeat(3,1)
                state[variant]=model.step(x,state[variant])
            if t+1 in (1,5,15,30,60,120,steps):
                native_base=state['native'][:n_phase]
                tangent_now=torch.as_tensor(tangents(native_base.cpu().numpy()),device=device,dtype=h.dtype)
                # Same-state update difference exposes the pathway changed by the code.
                delta=model.step(x_native,native_previous)-model.step(x_plain,native_previous)
                tangent_component=(delta*tangent_now).sum(-1)
                drive_energy=float(delta.square().sum())
                tangent_energy=float(tangent_component.square().sum())
                for variant in variants:
                    base,ht,hn=state[variant].chunk(3)
                    outputs=model.readout(base)
                    decoded=torch.atan2(outputs[:,1],outputs[:,0])
                    phase_error=wrap_angle(decoded-theta)
                    row=dict(speed=speed,variant=variant,cue_gain=cue_gain,step=t+1,seconds=(t+1)*config.dt,
                        baseline_mae_deg=float(phase_error.abs().mean()*180/math.pi),
                        signed_lag_seconds=float(phase_error.mean()/speed),
                        baseline_radius=ring_radius(base.cpu().numpy()),
                        predictive_drive_tangent_energy_fraction=tangent_energy/max(drive_energy,1e-20))
                    for label,perturbed in (('tangent',ht),('normal',hn)):
                        difference=perturbed-base
                        y=model.readout(perturbed)
                        dphi=wrap_angle(torch.atan2(y[:,1],y[:,0])-decoded)
                        row[label+'_state_gain']=float(difference.norm(dim=-1).mean())/initial_distance[label]
                        row[label+'_phase_error_deg']=float(dphi.abs().mean()*180/math.pi)
                        row[label+'_output_error']=float((y-outputs).norm(dim=-1).mean())
                    results.append(row)
        for variant in variants:
            arrays[f'speed{speed}_{variant}']=state[variant].cpu().numpy()
    result=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        seed=ckpt['seed'],condition=ckpt['condition'],config=ckpt['config'],rows=results,
        n_phase=n_phase,cue_gain=cue_gain,perturbation_radius_fraction=.02,
        protocol='Matched forced constant-speed trajectories; native warm-up; code interventions and full-state perturbations at step zero; own counterfactual baselines.')
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    summary={}
    for variant in variants:
        rows=[r for r in results if r['variant']==variant and r['step']==60]
        summary[variant]={k:float(np.mean([r[k] for r in rows])) for k in
                          ('baseline_mae_deg','normal_state_gain','tangent_state_gain','normal_phase_error_deg')}
    print(json.dumps(summary,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--n-phase',type=int,default=128)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cue-gain',type=float,default=1.)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    run(a.checkpoint,a.out,a.n_phase,a.steps,a.device,a.cue_gain)
