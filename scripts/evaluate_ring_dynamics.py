"""Cue-free memory, full-state perturbations and spectra of a trained ring task.

This evaluates learned dynamics, not whether an input-driven PCA trace is round.
The spatial phase orders initialization cues; no circle is imposed on state.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ring_dynamics import ring_radius, tangents, perturbations, trajectory_measurements, wrap
from ring_task_model import load_checkpoint, phase_error


@torch.no_grad()
def initialize(model, task, phase, condition, cue_steps=8, settle_steps=20):
    h = phase.new_zeros(len(phase), model.hidden_size)
    zero = torch.zeros_like(phase)
    cue = task.inputs(phase, zero, torch.ones_like(phase), condition,
                      shuffled_velocity=zero)
    neutral = cue*0
    for _ in range(cue_steps):
        h = model.step(cue, h)
    for _ in range(settle_steps):
        h = model.step(neutral, h)
    return h


@torch.no_grad()
def continue_states(model, states, steps, times):
    x = states.new_zeros(len(states), model.input_size)
    path = []
    for step in range(steps+1):
        if step in times:
            path.append(states.cpu().numpy())
        if step < steps:
            states = model.step(x, states)
    return np.stack(path)


def jacobian(model, states, x=None):
    """Analytic h(t+1)/h(t), with current input held fixed."""
    if x is None:
        x = states.new_zeros(len(states), model.input_size)
    pre = model.input(x)+model.recurrent(states)
    gain = 1-pre.tanh().square()
    alpha = model.config.leak
    eye = torch.eye(model.hidden_size, device=states.device, dtype=states.dtype)
    return (1-alpha)*eye + alpha*gain[..., :, None]*model.recurrent.weight


@torch.no_grad()
def analyze(checkpoint, out, n_phase=128, steps=300, device='cpu'):
    model, task, ckpt = load_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    phase = torch.arange(n_phase, device=device)*2*math.pi/n_phase
    cue_end = initialize(model, task, phase, ckpt['condition'], settle_steps=0)
    early_times = [0,1,5,10,20]
    early_path = continue_states(model,cue_end,20,early_times)
    h = torch.as_tensor(early_path[-1],device=device)
    curve = h.cpu().numpy()
    radius = ring_radius(curve)
    times = sorted(set(t for t in (0,1,5,10,20,40,60,100,150,300,600,steps) if t<=steps))
    base = continue_states(model,h,steps,times)
    rows_by_size = {}
    archive = dict(initial_phase=phase.cpu().numpy(), base=base, times=np.array(times),
                   early_path=early_path,early_times=np.array(early_times))
    for size in (0.02,0.10,0.30):
        tangent, normal = perturbations(curve,size,seed=419)
        ht = h + torch.as_tensor(tangent,device=device,dtype=h.dtype)
        hn = h + torch.as_tensor(normal,device=device,dtype=h.dtype)
        t_path,n_path = [continue_states(model,s,steps,times) for s in (ht,hn)]
        rows = trajectory_measurements(base,t_path,n_path)
        for t,row in zip(times,rows):
            row['step'] = t
            row['seconds'] = t*model.config.dt
        rows_by_size[str(size)] = rows
        archive[f'tangent_{size}'] = t_path
        archive[f'normal_{size}'] = n_path
    memory = []
    target = torch.stack((phase.cos(),phase.sin()),-1)
    for t,state in zip(times,base):
        output = model.readout(torch.as_tensor(state,device=device))
        error = phase_error(output,target).cpu().numpy()
        estimate = torch.atan2(output[:,1],output[:,0]).cpu().numpy()
        ordered = np.sort(np.mod(estimate,2*np.pi))
        largest_gap = np.diff(np.r_[ordered,ordered[0]+2*np.pi]).max()
        adjacent = wrap(np.roll(estimate,-1)-estimate)
        quantized = np.floor(np.mod(estimate,2*np.pi)/(2*np.pi/36)).astype(int)
        single_next = model.step(torch.zeros(n_phase,model.input_size,device=device),
                                 torch.as_tensor(state,device=device))
        memory.append(dict(step=t,seconds=t*model.config.dt,
            angular_mae_deg=float(np.mean(np.abs(error))*180/np.pi),
            angular_rmse_deg=float(np.sqrt(np.mean(error**2))*180/np.pi),
            circle_coverage=float(1-largest_gap/(2*np.pi)),
            radius_ratio=ring_radius(state)/radius,
            radius_from_cue_end=ring_radius(state)/ring_radius(early_path[0]),
            occupied_10deg_bins=int(len(np.unique(quantized))),
            phase_map_total_winding=float(adjacent.sum()/(2*np.pi)),
            largest_neighbor_phase_gap_deg=float(np.max(np.abs(adjacent))*180/np.pi),
            fixed_point_residual_radius=float((single_next-torch.as_tensor(state,device=device)).norm(dim=-1).mean()/radius),
            output_radius_mean=float(output.norm(dim=-1).mean())))
    # Local eigenvalues, tangent response, and restricted normal dynamics.
    js = jacobian(model,h).cpu().numpy().astype(np.float64)
    tangent = tangents(curve)
    spectra=[]
    for j,t in zip(js,tangent):
        eigen = np.linalg.eigvals(j)
        q = np.linalg.qr(t[:,None], mode='complete')[0][:,1:]
        normal_eigen = np.linalg.eigvals(q.T@j@q)
        jt = j@t
        spectra.append(dict(spectral_radius=float(np.abs(eigen).max()),
            tangent_gain=float(t@jt), tangent_shear=float(np.linalg.norm(jt-t*(t@jt))),
            normal_spectral_radius=float(np.abs(normal_eigen).max()),
            largest_singular_value=float(np.linalg.svd(j,compute_uv=False)[0])))
    archive['jacobians']=js
    # Independently differentiate the implementation for one state.
    with torch.enable_grad():
        h0=h[0].detach().clone().requires_grad_(True)
        neutral=h0.new_zeros(model.input_size)
        auto=torch.autograd.functional.jacobian(lambda state:model.step(neutral,state),h0)
    jac_error=float(np.abs(auto.cpu().numpy()-js[0]).max())
    assert jac_error<2e-6, jac_error
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        condition=ckpt['condition'],seed=ckpt['seed'],config=ckpt['config'],steps_completed=ckpt['steps_completed'],
        validation=ckpt['validation'],n_phase=n_phase,cue_steps=8,initial_blank_steps=20,
        initial_ring_radius=radius,memory=memory,perturbation=rows_by_size,
        early_cue_removal=[dict(step=t,radius_ratio=ring_radius(state)/ring_radius(early_path[0]))
                           for t,state in zip(early_times,early_path)],
        spectra={key:float(np.mean([row[key] for row in spectra])) for key in spectra[0]},
        per_phase_spectra=spectra,jacobian_autograd_max_error=jac_error,
        notes=['Full hidden state; no PCA truncation.',
               'Same zero input for all states during continuation.',
               'Transverse directions are orthogonal to empirical curve tangent.',
               'Restricted normal spectra are local projections, not global stability proofs.'])
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),**archive)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('condition','seed','validation','spectra')},indent=2))
    print(json.dumps(dict(memory_final=memory[-1],normal=size_summary(rows_by_size,'0.1'))))
    return result


def size_summary(rows,size):
    return dict(initial=rows[size][0],final=rows[size][-1])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--n-phase',type=int,default=128)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    analyze(a.checkpoint,a.out,a.n_phase,a.steps,a.device)
