"""Validated saturation quotient, with no phase projection or circular loss.

Eight fixed discovery phases identify persistently saturated c coordinates.
All h and all other c coordinates remain. Their reduced dynamics is validated
on 56 held-out phases before any unconstrained residual minimization.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits
import lstm_goal_bearing_manifold as base
from refine_lstm_goal_bearing_manifold import elicited
from lstm_endogenous_curve_recovery import advance,sha,save
from lstm_endogenous_fixed_points import minimize_residual,clusters
from lstm_endogenous_saturation_audit import candidate


def quotient_step(cell,drive,q,mask,sign):
    h=q[:,:256];c=torch.zeros_like(h);c[:,~mask]=q[:,256:]
    gates=drive+torch.nn.functional.linear(h,cell.weight_hh,cell.bias_hh)
    i,f,g,o=gates.chunk(4,1)
    c=f.sigmoid()*c+i.sigmoid()*g.tanh()
    h=o.sigmoid()*torch.where(mask,sign,c.tanh())
    return torch.cat((h,c[:,~mask]),1)


def reduce_state(z,mask):
    return torch.cat((z[:,:256],z[:,256:][:,~mask]),1)


def compare(net,start,inputs,mask,sign):
    full=start.clone();q=reduce_state(start,mask)
    hmax=torch.zeros(len(start),dtype=start.dtype);pmax=hmax.clone()
    for x in inputs:
        drive=torch.nn.functional.linear(x[None],net.lstm.weight_ih,net.lstm.bias_ih)
        full=advance(net.lstm,drive,full);q=quotient_step(net.lstm,drive,q,mask,sign)
        hmax=torch.maximum(hmax,(full[:,:256]-q[:,:256]).abs().max(1).values)
        pmax=torch.maximum(pmax,(net.pi(full[:,:256]).softmax(1)-net.pi(q[:,:256]).softmax(1)).abs().max(1).values)
    return dict(hidden_error_per_phase=hmax.tolist(),actor_error_per_phase=pmax.tolist(),hidden_max=float(hmax.max()),actor_max=float(pmax.max()),passes=bool((hmax<=1e-5).all() and (pmax<=1e-6).all()))


def sequences(net,pop,w,other,theta):
    xx=torch.tensor(w['x'],dtype=torch.float64)
    native_initial=torch.tensor(np.r_[w['h_before'][0],w['c_before'][0]],dtype=torch.float64)
    observer=torch.tensor(w['pos'][14],dtype=torch.float64)
    _,goal=elicited(net,pop,xx[:15],native_initial,observer,theta,0.,False)
    _,ordinary_goal=elicited(net,pop,xx[:15],native_initial,observer,theta,0.,True)
    native=xx[15:75].clone();withheld=native.clone();withheld[:,768:1280]=0
    native_goal=native.clone();native_goal[:,768:1280]=goal[:1]
    ordinary=native.clone();ordinary[:,768:1280]=ordinary_goal[:1]
    ordinary[:,256:768]=pop.activity(torch.tensor(w['pos'][15:75],dtype=torch.float64),torch.tensor(w['direction'][15:75],dtype=torch.float64),torch.tensor(w['velocity'][15:75],dtype=torch.float64),shift_scale=0)
    alternate=torch.tensor(other['x'][15:75],dtype=torch.float64)
    stress=torch.stack([native[t] if (t//5)%2==0 else alternate[t]for t in range(60)])
    fixed=xx[14:15].repeat(60,1);fixed[:,768:1280]=0
    return dict(recorded_native=native,goal_withheld=withheld,common_native_goal=native_goal,ordinary_current_and_goal=ordinary,switched_native_context_stress=stress,fixed_common_input=fixed)


def run(condition,seed,context,args):
    cfg,checkpoint,net,pop=base.load_model(condition,seed,'cpu');net.double()
    source=ROOT/f'results/lstm_endogenous_ring/saturation_audit/{condition}_s{seed}'
    data=np.load(source/'native_4s_contexts.npz')
    windows=[{k.split('__',1)[1]:data[k]for k in data if k.startswith(f'{j}__')}for j in range(2)]
    a=np.load(source/f'context{context}_settle75.npz');state=torch.tensor(a['state']);theta=torch.tensor(a['theta'])
    discovery=torch.arange(64)%8==0;heldout=~discovery
    with torch.no_grad():
        seq=sequences(net,pop,windows[context],windows[1-context],theta)
        mask,values=candidate(state[discovery]);sign=values.sign()
        initial_candidates=mask.clone();min_signed=torch.full((256,),float('inf'),dtype=torch.float64)
        for xx in seq.values():
            z=state[discovery].clone()
            for x in xx:
                z=advance(net.lstm,torch.nn.functional.linear(x[None],net.lstm.weight_ih,net.lstm.bias_ih),z)
                min_signed=torch.minimum(min_signed,(z[:,256:]*sign).min(0).values)
        mask=mask&(min_signed>=8)
        validation={name:compare(net,state[heldout],xx,mask,sign)for name,xx in seq.items()}
        passed=bool(mask.any()) and all(t['passes']for t in validation.values())
    record=dict(condition=condition,seed=seed,context=context,source_sha256=sha(__file__),checkpoint_sha256=sha(checkpoint),
        source_state_sha256=sha(source/f'context{context}_settle75.npz'),initial_candidate_count=int(initial_candidates.sum()),
        candidate_indices=mask.nonzero().flatten().tolist(),candidate_count=int(mask.sum()),candidate_signs=sign[mask].tolist(),
        discovery_phase_indices=discovery.nonzero().flatten().tolist(),heldout_phase_indices=heldout.nonzero().flatten().tolist(),
        validation=validation,passes_heldout_validation=passed,optimization_attempted=passed)
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True);stem=folder/f'context{context}'
    if passed:
        drive=torch.nn.functional.linear(seq['fixed_common_input'][:1],net.lstm.weight_ih,net.lstm.bias_ih)
        step=lambda q:quotient_step(net.lstm,drive,q,mask,sign)
        initial=reduce_state(state,mask)
        solution,optimization=minimize_residual(step,initial,3000,300)
        with torch.no_grad():
            residual=step(solution)-solution;raw=residual.square().mean(1).sqrt()
            scale=torch.cat((state[:,:256].square().mean().sqrt().expand(256),state[:,256:].square().mean().sqrt().expand(256-int(mask.sum()))))
            scaled=(residual/scale).square().mean(1).sqrt();fixed=(raw<=1e-8)&(scaled<=1e-6);slow=(raw<=1e-5)&(scaled<=1e-4)
            # Restore the eliminated amplitudes from the actual initializer.
            # A candidate may leave the validated saturation sector during optimization.
            restored=state.clone();restored[:,:256]=solution[:,:256];restored[:,256:][:,~mask]=solution[:,256:]
            post=compare(net,restored,seq['fixed_common_input'],mask,sign)
            cp=clusters((solution/scale).numpy(),1e-3)
            sv=scipy.linalg.svdvals((solution-solution.mean(0)).numpy(),check_finite=False);vf=sv**2/max(float(np.sum(sv**2)),1e-40)
            spectra=[]
        if bool(slow.any()):
            representatives=sorted(set(int(np.where((cp==c)&slow.numpy())[0][0])for c in np.unique(cp[slow.numpy()])))
            for index in representatives:
                jac=torch.autograd.functional.jacobian(lambda q:step(q[None])[0],solution[index],vectorize=True).detach().numpy()
                ev,vec=scipy.linalg.eig(jac,check_finite=False)
                t=(solution[(index+1)%64]-solution[(index-1)%64]).numpy()
                alignment=np.abs(vec.conj().T@t)/(np.linalg.norm(vec,axis=0)*max(np.linalg.norm(t),1e-30))
                output=np.linalg.norm(net.pi.weight.detach().numpy()@vec[:256],axis=0)/np.linalg.norm(vec,axis=0)
                spectra.append(dict(initializer=index,eigenvalues=np.column_stack((ev.real,ev.imag)).tolist(),actor_logit_gain=output.tolist(),neighbor_tangent_alignment=alignment.tolist(),near_neutral_count=int(np.sum(abs(ev)>=.99))))
        record['optimization']=optimization;record['residual_raw_rms']=raw.tolist();record['residual_scaled_rms']=scaled.tolist()
        record.update(fixed_count=int(fixed.sum()),slow_count=int(slow.sum()),scaled_cluster_count=len(np.unique(cp)),post_optimization_native_amplitude_validation=post,
            geometry=dict(radius_rms=float((solution-solution.mean(0)).square().mean().sqrt()),participation_ratio=float(1/max(float(np.sum(vf**2)),1e-40)),first_five_pc_fractions=vf[:5].tolist()),spectra=spectra)
        np.savez_compressed(stem.with_suffix('.npz'),theta=theta.numpy(),mask=mask.numpy(),sign=sign.numpy(),initial=initial.numpy(),solution=solution.numpy(),residual=residual.numpy(),restored=restored.numpy())
    save(stem.with_suffix('.json'),record)
    print(json.dumps({k:record[k]for k in ('condition','seed','context','candidate_count','passes_heldout_validation','fixed_count','slow_count')if k in record}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/quotient_points');p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(conditions=['gc100','pgc33'],seeds=[1,2],contexts=[0,1],settled_updates=75,discovery_phases=list(range(0,64,8)),
        selection='On eight discovery phases: shared initial c sign and |c|>=8 throughout every six60-step future. Every h and other c retained.',
        quotient='Replace tanh(new_c_j) by the fixed saturation sign only for selected j. Remove those c coordinates; retain every h. No phase-based subspace.',
        validation='All six futures on the56otherphases; maximum h error<=1e-5 and actor-probability error<=1e-6. No optimization unless combined quotient passes.',
        optimization='Only original recurrent residual in the retained physical coordinates; Adam3000/LBFGS300 as full-state search; no phase/geometry term.',
        post_optimization='Restore removed c amplitudes from actual initializers and compare full and quotient dynamics over4seconds commoninput; failure invalidates an optimized quotient candidate.',
        fixed_threshold=dict(raw_RMS=1e-8,scaled_RMS=1e-6),slow_threshold=dict(raw_RMS=1e-5,scaled_RMS=1e-4),
        interpretation='Conditional empirical quotient on tested input sector; all failed full-state and initial-saturation tests retained. A ring additionally requires connected phase geometry and preferential radial/normal attraction.',
        source_sha256=sha(__file__),parents={p:sha(ROOT/p)for p in ['scripts/lstm_endogenous_fixed_points.py','scripts/lstm_endogenous_saturation_audit.py']})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    if args.freeze_only:return
    with threadpool_limits(limits=2):
        for condition in protocol['conditions']:
            for seed in protocol['seeds']:
                for context in protocol['contexts']:run(condition,seed,context,args)


if __name__=='__main__':main()
