"""Does predictive spatial input reanchor a learned direction-memory ring?

Initialization supplies the same short heading cue as training. A phase
perturbation replaces memory by another measured state on that conditional
curve. Every condition then receives the same physical position/direction.
Cue-off persistence and cue-on correction are explicitly separate tests.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ring_heading_model import load_checkpoint
from ring_dynamics import ring_radius, tangents, wrap
from evaluate_ring_dynamics import jacobian


@torch.no_grad()
def run(checkpoint,out,n_phase=72,steps=200,device='cpu'):
    if n_phase%36:
        raise ValueError('n_phase must be divisible by36 for exact10degree phase perturbations')
    model,task,ckpt=load_checkpoint(checkpoint,device)
    model.requires_grad_(False)
    phase=torch.arange(n_phase,device=device)*2*math.pi/n_phase
    direction=torch.stack((phase.cos(),phase.sin()),-1)
    zero=torch.zeros(n_phase,device=device)
    one=torch.ones_like(zero)
    positions=[(-.8,-.8),(-.8,.8),(.8,-.8),(.8,.8),(0.,0.),(10.,10.)]
    variants=['native','blank','ordinary','reverse','restore_20']
    if ckpt['condition']=='predictive':
        variants.append('self_generated')
    rows=[]
    archive={}
    for context,xy in enumerate(positions):
        pos=torch.tensor(xy,device=device).expand(n_phase,2)
        h=torch.zeros(n_phase,model.hidden_size,device=device)
        cue=task.inputs(pos,direction,zero,zero,one,one,ckpt['condition'],shuffled_direction=-direction)
        for _ in range(model.config.initial_heading_steps):
            h=model.step(cue,h)
        cue_end=h.clone()
        blank=torch.zeros(n_phase,model.input_size,device=device)
        for _ in range(40):
            h=model.step(blank,h)
        radius=ring_radius(h.cpu().numpy())
        curve=h.clone()
        # These are actual states elicited by adjacent initial directions.
        plus=curve.roll(-n_phase//36,0)
        minus=curve.roll(n_phase//36,0)
        states={v:torch.cat((curve,plus,minus)) for v in variants}
        decoded0=model.readout(curve)
        initial_angle=torch.atan2(decoded0[:,1],decoded0[:,0])
        initial_displacements=[]
        for s in (plus,minus):
            y=model.readout(s)
            initial_displacements.append(wrap((torch.atan2(y[:,1],y[:,0])-initial_angle).cpu().numpy()))
        initial_displacement=float(np.mean(np.abs(initial_displacements)))
        initial_state_distance=np.stack([(s-curve).norm(dim=-1).cpu().numpy() for s in (plus,minus)])
        # Neutral dynamics are measured with full-state Jacobians at the curve.
        matrices=jacobian(model,curve).cpu().numpy()
        tangent=tangents(curve.cpu().numpy())
        spectral=[]
        for j,tan in zip(matrices,tangent):
            ev,v=np.linalg.eig(j)
            top=int(np.argmax(np.abs(ev)))
            spectral.append((float(np.abs(ev[top])),float(abs(np.dot(tan,v[:,top]))/np.linalg.norm(v[:,top]))))
        for t in range(steps+1):
            if t in (0,1,5,10,20,40,60,100,200,steps):
                for variant,state in states.items():
                    base,hp,hm=state.chunk(3)
                    outputs=[model.readout(s) for s in (base,hp,hm)]
                    angles=[torch.atan2(y[:,1],y[:,0]).cpu().numpy() for y in outputs]
                    error=wrap(angles[0]-phase.cpu().numpy())
                    differences=np.array([wrap(a-angles[0]) for a in angles[1:]])
                    state_distance=np.stack([(s-base).norm(dim=-1).cpu().numpy() for s in (hp,hm)])
                    initial_abs=np.abs(np.array(initial_displacements))
                    local_gain=np.abs(differences)/np.maximum(initial_abs,1e-5)
                    ordered=np.sort(np.mod(angles[0],2*np.pi))
                    max_gap=np.diff(np.r_[ordered,ordered[0]+2*np.pi]).max()
                    rows.append(dict(context=context,position=list(xy),variant=variant,step=t,
                        seconds=t*model.config.dt,baseline_mae_deg=float(np.mean(np.abs(error))*180/np.pi),
                        phase_displacement_deg=float(np.mean(np.abs(differences))*180/np.pi),
                        full_state_phase_perturbation_gain_mean=float(np.mean(state_distance)/np.mean(initial_state_distance)),
                        full_state_phase_perturbation_gain_median=float(np.median(state_distance/np.maximum(initial_state_distance,1e-10))),
                        phase_displacement_median_deg=float(np.median(np.abs(differences))*180/np.pi),
                        phase_displacement_rms_deg=float(np.sqrt(np.mean(differences**2))*180/np.pi),
                        phase_local_gain_median=float(np.median(local_gain)),
                        phase_local_gain_q10=float(np.quantile(local_gain,.1)),
                        phase_local_gain_q90=float(np.quantile(local_gain,.9)),
                        phase_displacement_retained=float(np.mean(np.abs(differences))/max(initial_displacement,1e-10)),
                        circle_coverage=float(1-max_gap/(2*np.pi)),
                        state_radius_retained=ring_radius(base.cpu().numpy())/max(radius,1e-10),
                        radius_from_initial_cue=ring_radius(base.cpu().numpy())/ring_radius(cue_end.cpu().numpy()),
                        initial_memory_mae_deg=float(np.mean(np.abs(wrap(initial_angle.cpu().numpy()-phase.cpu().numpy())))*180/np.pi),
                        neutral_spectral_radius=float(np.mean([s[0] for s in spectral])),
                        neutral_slow_mode_tangent_alignment=float(np.mean([s[1] for s in spectral]))))
            if t==steps:
                break
            for variant,state in states.items():
                if variant=='self_generated':
                    # The preview now follows each replica's own memory. It
                    # contains no independent reference to the true direction.
                    estimate=model.readout(state)
                    angle=torch.atan2(estimate[:,1],estimate[:,0])
                    own_direction=torch.stack((angle.cos(),angle.sin()),-1)
                    x=task.inputs(pos.repeat(3,1),own_direction,zero.repeat(3),zero.repeat(3),
                                  one.repeat(3),zero.repeat(3),'predictive')
                    states[variant]=model.step(x,state)
                    continue
                visible=one
                condition=ckpt['condition']
                if variant=='blank':
                    visible=zero
                elif variant=='ordinary' or (variant=='restore_20' and t<20):
                    condition='grid'
                elif variant=='reverse':
                    condition='reverse' if ckpt['condition']=='predictive' else ckpt['condition']
                x=task.inputs(pos,direction,zero,zero,visible,zero,condition,shuffled_direction=-direction)
                states[variant]=model.step(x.repeat(3,1),state)
        archive[f'context{context}_initial']=curve.cpu().numpy()
        archive[f'context{context}_cue_end']=cue_end.cpu().numpy()
        for variant,state in states.items():
            archive[f'context{context}_{variant}_final']=state.cpu().numpy()
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        condition=ckpt['condition'],seed=ckpt['seed'],config=ckpt['config'],rows=rows,
        protocol=dict(phase_perturbation_deg=10,initial_heading_cue_steps=model.config.initial_heading_steps,
            n_phase=n_phase,continuation_steps=steps,physical_positions=[list(xy) for xy in positions],
            blank_settling_steps=40,stationary_direction_retained=True,
            training_positions='initial[-1.25,1.25]^2 plus forwardmovement',
            distant_position='[10,10] held out beyond training reach',
            purpose='Cue-free memory versus input-driven correction of a phase disturbance'))
    result['variant_labels']={'ordinary':('Remove explicit heading cue' if ckpt['condition']=='grid_heading' else 'Predictive slots become ordinary'),
        'reverse':('Reverse predictive spatial shift' if ckpt['condition']=='predictive' else 'No-op control'),
        'self_generated':'Preview direction comes from each replica own previous decoded memory; no independent direction reference'}
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),**archive)
    summary={}
    for variant in variants:
        selected=[r for r in rows if r['context']<5 and r['step']==100 and r['variant']==variant]
        summary[variant]={k:float(np.mean([r[k] for r in selected])) for k in
                          ('baseline_mae_deg','phase_displacement_retained','circle_coverage','state_radius_retained')}
    print(json.dumps(dict(condition=ckpt['condition'],seed=ckpt['seed'],summary=summary),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--n-phase',type=int,default=72)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    run(a.checkpoint,a.out,a.n_phase,a.steps,a.device)
