"""Post-screen coordinate/long-transient audit of ALL original pilot contexts.

The frozen first screen used block-RMS coordinates. This descriptive addendum
checks the same physical perturbations in raw coordinates, isotropic raw-state
controls, longer recovery, and coordinate-invariant local Jacobian spectra.
No context is selected by its result.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits
import lstm_goal_bearing_manifold as base
from refine_lstm_goal_bearing_manifold import elicited
from lstm_endogenous_curve_recovery import advance,decomposition,describe,sha,save

TIMES=(0,1,15,75,150,450,1500)


def evaluate(condition,seed,context,out):
    started=time.monotonic()
    cfg,checkpoint,net,pop=base.load_model(condition,seed,'cpu');net.double();cell=net.lstm
    p=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context{context}_states.npz'
    a=np.load(p);inputs=torch.tensor(a['history_x'][:15],dtype=torch.float64)
    initial=torch.tensor(np.r_[a['history_h_before'][0],a['history_c_before'][0]],dtype=torch.float64)
    observer=torch.tensor(a['history_pos'][14],dtype=torch.float64)
    theta=torch.arange(96,dtype=torch.float64)*(2*np.pi/96)
    fun=lambda angle:elicited(net,pop,inputs,initial,observer,angle,0.,False)[0]
    z,t=torch.autograd.functional.jvp(fun,theta,torch.ones_like(theta));z=z.detach();t=t.detach()
    original=z.clone();scale=torch.stack((z[:,:256].square().mean().sqrt(),z[:,256:].square().mean().sqrt())).repeat_interleave(256)
    x=inputs[-1:].repeat(96,1);x[:,768:1280]=0
    drive=torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih)
    with torch.no_grad():
        for _ in range(75):z,dd=advance(cell,drive,z,t[:,:,None]);t=dd[:,:,0]
        qlist=[];names=[]
        gen=torch.Generator().manual_seed(190731+context+seed*100)
        for metric,s in [('block_RMS',scale),('raw',torch.ones_like(scale))]:
            unit=t/s;unit/=unit.norm(dim=1,keepdim=True)
            for j in range(3):
                q=torch.randn(z.shape,generator=gen,dtype=z.dtype)
                q-=(q*unit).sum(1,keepdim=True)*unit
                q/=q.norm(dim=1,keepdim=True)
                qlist.append(q*s);names.append(f'{metric}_{j}')
        directions=torch.stack((t,*qlist),-1);initial_d=directions.clone();start=z.clone()
        spectra=[]
        for phase in (0,24,48,72):
            identity=torch.eye(512,dtype=torch.float64)[None]
            _,jac=advance(cell,drive[phase:phase+1],z[phase:phase+1],identity)
            jac=jac[0].numpy()
            values,vectors=scipy.linalg.eig(jac,check_finite=False)
            tangent=t[phase].numpy()
            # This is alignment with a complex eigenspace; magnitude handles
            # its arbitrary complex phase without reporting only a real part.
            alignment=np.abs(vectors.conj().T@tangent)/(np.linalg.norm(vectors,axis=0)*np.linalg.norm(tangent))
            best=int(np.argmax(alignment));near=int(np.argmin(np.abs(values-1)))
            spectra.append(dict(phase_index=phase,spectral_radius=float(np.abs(values).max()),
                eigenvalues_modulus_at_least_099=int(np.sum(np.abs(values)>=.99)),
                eigenvalues_modulus_greater_than_one=int(np.sum(np.abs(values)>1+1e-8)),
                tangent_best_eigenvector_alignment=float(alignment[best]),
                tangent_best_eigenvalue=[float(values[best].real),float(values[best].imag)],
                nearest_one_eigenvalue=[float(values[near].real),float(values[near].imag)],
                nearest_one_tangent_alignment=float(alignment[near]),
                all_eigenvalues=np.column_stack((values.real,values.imag)).tolist()))
        rows={}
        for step in range(max(TIMES)+1):
            if step in TIMES:
                row={}
                for metric,s in [('block_RMS',scale),('raw',torch.ones_like(scale))]:
                    tangent=directions[:,:,0]/s;initial_t=initial_d[:,:,0]/s
                    gains={}
                    for group in ('block_RMS','raw'):
                        total=[];normal=[]
                        for k,name in enumerate(names,1):
                            if not name.startswith(group):continue
                            perturb=directions[:,:,k]/s;first=initial_d[:,:,k]/s
                            _,current_perp=decomposition(perturb,tangent)
                            _,initial_perp=decomposition(first,initial_t)
                            total.append((perturb.norm(dim=1)/first.norm(dim=1)).numpy())
                            normal.append((current_perp/initial_perp.clamp_min(1e-30)).numpy())
                        gains[group]=dict(pointwise_gain=describe(total),transverse_gain=describe(normal))
                    radius=float(((z-z.mean(0))/s).square().mean().sqrt())
                    original_radius=float(((original-original.mean(0))/s).square().mean().sqrt())
                    row[metric]=dict(tangent_norm_gain=describe((tangent.norm(dim=1)/initial_t.norm(dim=1)).numpy()),
                        family_radius_from_original=radius/original_radius,normal_direction_sets=gains)
                rows[str(step)]=row
            if step<max(TIMES):z,directions=advance(cell,drive,z,directions)
        finite={}
        for rms in (1e-5,1e-3,1e-2):
            # One prespecified phase, all three block-RMS normals; comparisons
            # share physical directions rather than re-drawing a null basis.
            epsilon=rms*np.sqrt(512)
            q=initial_d[0,:,1:4].T
            plus=start[:1]+epsilon*q;minus=start[:1]-epsilon*q
            baseline=start[:1].clone()
            dd=initial_d[:1].clone()
            for _ in range(15):
                plus=advance(cell,drive[:1],plus);minus=advance(cell,drive[:1],minus)
                baseline,dd=advance(cell,drive[:1],baseline,dd)
            fd=(plus-minus)/(2*epsilon);linear=dd[0,:,1:4].T
            finite[str(rms)]=dict(relative_derivative_error=float((fd-linear).norm()/linear.norm()),
                central_pair_scaled_gain=describe((fd/scale).norm(dim=1).numpy()),
                linear_scaled_gain=describe((linear/scale).norm(dim=1).numpy()))
    result=dict(condition=condition,seed=seed,context=context,source_sha256=sha(__file__),
        checkpoint_sha256=sha(checkpoint),context_sha256=sha(p),settle_steps=75,local_spectra=spectra,
        longer_recovery=rows,finite_amplitude_check=finite,seconds=time.monotonic()-started,
        interpretation='Descriptive all-context sensitivity after the primary block-RMS screen. Local eigenvalues are instantaneous at a slow trajectory state, not eigenvalues of an identified fixed point. Both coordinate systems and both isotropic direction sets are retained.')
    folder=out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True);save(folder/f'context{context}.json',result)
    print(json.dumps(dict(condition=condition,seed=seed,context=context,seconds=result['seconds'],
        raw_1s_normal=rows['15']['raw']['normal_direction_sets']['raw']['transverse_gain']['median'],
        raw_5s_normal=rows['75']['raw']['normal_direction_sets']['raw']['transverse_gain']['median'],
        slow_mode_counts=[s['eigenvalues_modulus_at_least_099']for s in spectra])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/metric_audit');p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(role='Descriptive audit prompted by known raw-versus-block metric sensitivity; no change to frozen primary results',conditions=['gc100','pgc33'],seeds=[1,2],contexts=[0,1,2,3],angles=96,settle=75,
        probe_times=TIMES,metrics=['block_RMS','raw'],normal_direction_sets=['block_RMS_isotropic','raw_isotropic'],
        spectral_phases=[0,24,48,72],source_sha256=sha(__file__),primary_source_sha256=sha(ROOT/'scripts/lstm_endogenous_curve_recovery.py'))
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
    else:save(path,protocol)
    if args.freeze_only:return
    with threadpool_limits(limits=2):
        for c in protocol['conditions']:
            for seed in protocol['seeds']:
                for context in protocol['contexts']:evaluate(c,seed,context,args.out)


if __name__=='__main__':main()
