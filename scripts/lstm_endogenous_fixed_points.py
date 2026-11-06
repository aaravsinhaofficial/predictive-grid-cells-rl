"""Unconstrained fixed/slow-point search in original full-state navigation LSTMs.

Only the one-step dynamical residual is minimized. Goal angle labels are never
part of the objective, and no circular geometry or recurrent structure is added.
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
from scipy.cluster.hierarchy import linkage,fcluster
import torch
from threadpoolctl import threadpool_limits
import lstm_goal_bearing_manifold as base
from refine_lstm_goal_bearing_manifold import elicited
from lstm_endogenous_curve_recovery import advance,sha,save


def minimize_residual(step,start,adam_steps=3000,lbfgs_steps=300):
    """Independent initializers share a batch; only ||F(z)-z||² is optimized."""
    z=start.clone().detach().requires_grad_(True)
    optimizer=torch.optim.Adam([z],lr=.05,eps=1e-12)
    best=start.clone();best_q=torch.full((len(z),),float('inf'),dtype=z.dtype)
    history=[]
    def retain():
        nonlocal best,best_q
        with torch.no_grad():
            q=(step(z)-z).square().mean(1)
            better=q<best_q
            best=torch.where(better[:,None],z,best);best_q=torch.minimum(best_q,q)
            return q
    retain()
    for iteration in range(adam_steps):
        if iteration in (adam_steps//3,2*adam_steps//3):
            for group in optimizer.param_groups:group['lr']*=.2
        optimizer.zero_grad();residual=step(z)-z
        loss=residual.square().mean(1).sum();loss.backward();optimizer.step()
        if iteration%10==0:
            q=retain()
            if not torch.isfinite(z).all() or float(z.abs().max())>1000:
                history.append(dict(adam_iteration=iteration,status='nonfinite_or_state_magnitude_above1000'));break
            if iteration%250==0:history.append(dict(adam_iteration=iteration,residual_rms_median=float(best_q.sqrt().median()),residual_rms_max=float(best_q.sqrt().max())))
            if bool((best_q.sqrt()<1e-9).all()):break
    retain()
    z=torch.nn.Parameter(best.clone())
    optimizer=torch.optim.LBFGS([z],lr=1,max_iter=lbfgs_steps,max_eval=lbfgs_steps*2,
        tolerance_grad=1e-14,tolerance_change=1e-18,history_size=30,line_search_fn='strong_wolfe')
    evaluations=0
    def closure():
        nonlocal evaluations
        optimizer.zero_grad();loss=(step(z)-z).square().mean(1).sum();loss.backward();evaluations+=1
        retain();return loss
    optimizer.step(closure);retain()
    return best.detach(),dict(adam_history=history,lbfgs_evaluations=evaluations,
        residual_rms=best_q.sqrt().tolist(),objective='Mean squared raw full-state one-step residual; summed independent initializers; no goal or geometry loss.')


def clusters(points,threshold):
    if len(points)<2:return np.ones(len(points),dtype=int)
    tree=linkage(points,method='complete',metric='euclidean')
    return fcluster(tree,t=threshold*np.sqrt(points.shape[1]),criterion='distance')


def run(condition,seed,context,regime,args):
    started=time.monotonic();cfg,checkpoint,net,pop=base.load_model(condition,seed,'cpu');net.double();cell=net.lstm
    path=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context{context}_states.npz'
    a=np.load(path);inputs=torch.tensor(a['history_x'],dtype=torch.float64)
    native_initial=torch.tensor(np.r_[a['history_h_before'][0],a['history_c_before'][0]],dtype=torch.float64)
    observer=torch.tensor(a['history_pos'][14],dtype=torch.float64)
    theta=torch.arange(args.angles,dtype=torch.float64)*(2*np.pi/args.angles)
    with torch.no_grad():initial,_=elicited(net,pop,inputs[:15],native_initial,observer,theta,0.,False)
    scale=torch.stack((initial[:,:256].square().mean().sqrt(),initial[:,256:].square().mean().sqrt())).repeat_interleave(256)
    x=inputs[14:15].clone()
    if regime=='goal_withheld':x[:,768:1280]=0
    drive=torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih)
    step=lambda z:advance(cell,drive,z)
    solution,optimization=minimize_residual(step,initial,args.adam_steps,args.lbfgs_steps)
    with torch.no_grad():
        residual=step(solution)-solution
        raw=residual.square().mean(1).sqrt();scaled=(residual/scale).square().mean(1).sqrt()
        fixed=(raw<=1e-8)&(scaled<=1e-6)
        slow=(raw<=1e-5)&(scaled<=1e-4)
        cluster_scaled=clusters((solution/scale).numpy(),1e-3)
        cluster_raw=clusters(solution.numpy(),1e-3)
        cluster_h=clusters(solution[:,:256].numpy(),1e-3)
        states=solution.numpy();labels=theta.numpy()
        geometry={}
        for name,values in [('raw',states),('block_RMS',(solution/scale).numpy()),('h',states[:,:256]),('c',states[:,256:])]:
            centered=values-values.mean(0);sv=scipy.linalg.svdvals(centered,check_finite=False);var=sv**2;frac=var/max(var.sum(),1e-40)
            neighbor=np.linalg.norm(np.roll(values,-1,axis=0)-values,axis=1)
            geometry[name]=dict(radius_rms=float(np.sqrt(np.mean(centered**2))),participation_ratio=float(1/max(np.sum(frac**2),1e-40)),
                first_five_pc_fractions=frac[:5].tolist(),ordered_neighbor_max_over_median=float(neighbor.max()/max(np.median(neighbor),1e-40)))
        probs=net.pi(solution[:,:256]).softmax(1)
        actor_range=float((probs-probs.mean(0)).abs().max())
        # Retain every initializer; eigendecomposition covers one deterministic
        # representative per scaled-state cluster plus quarter-circle phases.
        representatives=sorted(set([int(np.where(cluster_scaled==k)[0][0])for k in np.unique(cluster_scaled)]+[0,args.angles//4,args.angles//2,3*args.angles//4]))
        spectra=[]
        for index in representatives:
            _,jac=advance(cell,drive,solution[index:index+1],torch.eye(512,dtype=torch.float64)[None])
            values,vectors=scipy.linalg.eig(jac[0].numpy(),check_finite=False)
            mode_norm=np.linalg.norm(vectors,axis=0)
            output=np.linalg.norm(net.pi.weight.detach().numpy()@vectors[:256],axis=0)/mode_norm
            h_fraction=np.linalg.norm(vectors[:256],axis=0)/mode_norm
            tangent=states[(index+1)%len(states)]-states[(index-1)%len(states)]
            alignment=np.abs(vectors.conj().T@tangent)/(mode_norm*max(np.linalg.norm(tangent),1e-30))
            near=np.abs(values)>=.99
            spectra.append(dict(initializer=index,qualified_fixed=bool(fixed[index]),qualified_slow=bool(slow[index]),
                spectral_radius=float(np.abs(values).max()),unstable_modes=int(np.sum(np.abs(values)>1+1e-6)),
                near_neutral_modes=int(np.sum(near)),near_neutral_actor_gain_max=float(output[near].max()) if near.any() else 0.,
                eigenvalues=np.column_stack((values.real,values.imag)).tolist(),actor_logit_gain=output.tolist(),
                hidden_fraction=h_fraction.tolist(),neighbor_tangent_alignment=alignment.tolist()))
        # Dynamics of the optimizer solutions, without any further fitting.
        relaxation=solution.clone();flow={}
        for t in range(151):
            if t in (0,15,75,150):
                flow[str(t)]=dict(radius_raw_rms=float((relaxation-relaxation.mean(0)).square().mean().sqrt()),
                    distance_from_solution_raw_rms=float((relaxation-solution).square().mean().sqrt()))
            if t<150:relaxation=step(relaxation)
        # Shared ACTUAL subsequent physical replay can reveal a state difference
        # that is invisible to the actor under the fixed-point input itself.
        replay=solution.clone();future=[]
        for xx in inputs[15:30]:
            xxx=xx[None].expand(len(solution),-1).clone()
            xxx[:,768:1280]=0 if regime=='goal_withheld'else x[:,768:1280]
            replay=advance(cell,torch.nn.functional.linear(xxx,cell.weight_ih,cell.bias_ih),replay)
            pp=net.pi(replay[:,:256]).softmax(1)
            future.append(float((pp-pp.mean(0)).abs().max()))
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    stem=folder/f'context{context}_{regime}'
    np.savez_compressed(stem.with_suffix('.npz'),theta=labels,initial=initial.numpy(),solution=states,residual=residual.numpy(),input=x.numpy(),scale=scale.numpy())
    record=dict(condition=condition,seed=seed,context=context,regime=regime,initializers=args.angles,
        checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),context_sha256=sha(path),fixed_count=int(fixed.sum()),slow_count=int(slow.sum()),
        residual_raw_rms=raw.tolist(),residual_block_rms=scaled.tolist(),cluster_scaled_labels=cluster_scaled.tolist(),cluster_raw_labels=cluster_raw.tolist(),
        cluster_h_labels=cluster_h.tolist(),scaled_cluster_count=len(np.unique(cluster_scaled)),raw_cluster_count=len(np.unique(cluster_raw)),h_cluster_count=len(np.unique(cluster_h)),
        geometry=geometry,actor_max_probability_deviation_from_family_mean=actor_range,shared_future_actor_deviation=future,
        optimization=optimization,spectra=spectra,solution_flow=flow,seconds=time.monotonic()-started,
        interpretation='Clusters and spectra retain every optimizer solution. Spectra at unqualified residuals are descriptive only. Multiple neutral saturated-c solutions alone are not a one-dimensional ring; actor observability and full-state dimensionality are recorded.')
    save(stem.with_suffix('.json'),record)
    print(json.dumps({k:record[k]for k in ['condition','seed','context','regime','fixed_count','slow_count','scaled_cluster_count','h_cluster_count','actor_max_probability_deviation_from_family_mean','seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/fixed_points');p.add_argument('--angles',type=int,default=64)
    p.add_argument('--adam-steps',type=int,default=3000);p.add_argument('--lbfgs-steps',type=int,default=300);p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(experiment='Unconstrained original-LSTM fixed/slow-point landscape',conditions=['gc100','pgc33'],seeds=[1,2],contexts=[0,1],regimes=['goal_withheld','actual_goal_held'],angles=args.angles,
        initializer='64 valid goal-conditioned states after15actual physical-history updates; evenly spaced goal angles; no selected neural geometry.',
        common_input='One recorded natural endpoint visual/current/action/reward input per context; held goal is that SAME actual snapshot across all initializers; withheld goal iszero.',
        objective='Raw full512-state dynamical residual only, no goal/angle/circle term.',adam_steps=args.adam_steps,lbfgs_steps=args.lbfgs_steps,
        fixed_threshold=dict(raw_RMS=1e-8,block_RMS=1e-6),slow_threshold=dict(raw_RMS=1e-5,block_RMS=1e-4),
        cluster_threshold_relative_rms=1e-3,representatives='First initializer per block-RMS cluster, plus four preselected quarter-circle initializers; all solutions retained.',
        interpretation='No requirement of a continuum of exact fixed points: ordered saddles/stable points can support a slow ring. A ring claim additionally needs connected/invariant phase geometry and preferential transverse recovery. Multiple saturated-c neutral modes are examined for actor visibility.',
        sources={'scripts/lstm_endogenous_fixed_points.py':sha(__file__),'scripts/lstm_endogenous_curve_recovery.py':sha(ROOT/'scripts/lstm_endogenous_curve_recovery.py'),'code/nets.py':sha(ROOT/'code/nets.py')})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
    else:save(path,protocol)
    if args.freeze_only:return
    with threadpool_limits(limits=2):
        for c in protocol['conditions']:
            for seed in protocol['seeds']:
                for context in protocol['contexts']:
                    for regime in protocol['regimes']:run(c,seed,context,regime,args)


if __name__=='__main__':main()
