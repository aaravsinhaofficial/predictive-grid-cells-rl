"""Final conditioning audit: exact constant-input LSTM equilibrium elimination.

At equilibrium c=i(h)g(h)/(1-f(h)). We solve only h=o(h)tanh(c(h)),
using sigmoid(-forget_logit) for 1-f to avoid subtractive cancellation.
No angle labels, circular projection, or geometric regularizer enter the fit.
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
from lstm_endogenous_curve_recovery import advance,sha,save
from lstm_endogenous_fixed_points import minimize_residual,clusters


def equilibrium_map(cell,drive,h):
    gates=drive+torch.nn.functional.linear(h,cell.weight_hh,cell.bias_hh)
    i,f,g,o=gates.chunk(4,1)
    denominator=(-f).sigmoid()
    c=i.sigmoid()*g.tanh()/denominator
    return o.sigmoid()*c.tanh(),c,denominator


def stable_full_jacobian(cell,drive,z):
    """Full-state Jacobian; stable f derivative does not round 1-f to zero."""
    h,c=z[:256],z[256:]
    gates=drive[0]+torch.nn.functional.linear(h,cell.weight_hh,cell.bias_hh)
    ai,af,ag,ao=gates.chunk(4)
    i,f,g,o=ai.sigmoid(),af.sigmoid(),ag.tanh(),ao.sigmoid()
    fi=(-af).sigmoid();tc=(f*c+i*g).tanh()
    wi,wf,wg,wo=cell.weight_hh.chunk(4)
    dcdh=(c*f*fi)[:,None]*wf+(g*i*(1-i))[:,None]*wi+(i*(1-g*g))[:,None]*wg
    dhdh=(o*(1-tc*tc))[:,None]*dcdh+(tc*o*(1-o))[:,None]*wo
    dcdc=torch.diag(f);dhdc=torch.diag(o*(1-tc*tc)*f)
    return torch.cat((torch.cat((dhdh,dhdc),1),torch.cat((dcdh,dcdc),1)),0),fi


def run(condition,seed,context,regime,args):
    cfg,checkpoint,net,pop=base.load_model(condition,seed,'cpu');net.double();cell=net.lstm
    source=ROOT/f'results/lstm_endogenous_ring/fixed_points/{condition}_s{seed}/context{context}_{regime}.npz'
    a=np.load(source);initial=torch.tensor(a['initial']);x=torch.tensor(a['input'])
    drive=torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih)
    map_h=lambda h:equilibrium_map(cell,drive,h)[0]
    hidden,optimization=minimize_residual(map_h,initial[:,:256],3000,300)
    with torch.no_grad():
        hnext,c,denom=equilibrium_map(cell,drive,hidden);solution=torch.cat((hidden,c),1)
        hidden_residual=(hnext-hidden).square().mean(1).sqrt()
        gates=drive+torch.nn.functional.linear(hidden,cell.weight_hh,cell.bias_hh)
        gi,gf,gg,go=gates.chunk(4,1)
        stable_c_delta=gi.sigmoid()*gg.tanh()-denom*c
        ordinary_residual=advance(cell,drive,solution)-solution
        raw=ordinary_residual.square().mean(1).sqrt()
        stable_residual=torch.cat((hnext-hidden,stable_c_delta),1).square().mean(1).sqrt()
        qualified=(hidden_residual<=1e-8)&(stable_residual<=1e-8)
        slow=(hidden_residual<=1e-5)&(stable_residual<=1e-5)
        labels=clusters(hidden.numpy(),1e-3)
        representatives=sorted(set([0,16,32,48]+[int(np.where((labels==k)&qualified.numpy())[0][0])for k in np.unique(labels[qualified.numpy()])]))
        spectra=[]
        for index in representatives:
            jac,stable_df=stable_full_jacobian(cell,drive,solution[index])
            values,vectors=scipy.linalg.eig(jac.numpy(),check_finite=False)
            hfrac=np.linalg.norm(vectors[:256],axis=0)/np.linalg.norm(vectors,axis=0)
            out=np.linalg.norm(net.pi.weight.detach().numpy()@vectors[:256],axis=0)/np.linalg.norm(vectors,axis=0)
            spectra.append(dict(initializer=index,qualified_fixed=bool(qualified[index]),eigenvalues=np.column_stack((values.real,values.imag)).tolist(),
                near_neutral_count=int(np.sum(np.abs(values)>=.99)),unstable_count=int(np.sum(np.abs(values)>1+1e-6)),
                hidden_fraction=hfrac.tolist(),actor_logit_gain=out.tolist(),spectral_radius=float(np.abs(values).max())))
        # True full-state flow starts from the original family, not the eliminated-c map.
        native=initial.clone();flow={}
        for t in range(1501):
            if t in (0,60,150,1500):
                hd=torch.cdist(native[:,:256],hidden)/16
                cd=torch.cdist(native[:,256:],c)/16
                flow[str(t)]=dict(native_max_abs_c=float(native[:,256:].abs().max()),
                    nearest_solution_hidden_RMS_median=float(hd.min(1).values.median()),nearest_solution_hidden_RMS_max=float(hd.min(1).values.max()),
                    nearest_solution_c_RMS_median=float(cd.min(1).values.median()),
                    paired_hidden_RMS_median=float((native[:,:256]-hidden).square().mean(1).sqrt().median()),
                    paired_c_RMS_median=float((native[:,256:]-c).square().mean(1).sqrt().median()))
            if t<1500:native=advance(cell,drive,native)
        sv=scipy.linalg.svdvals((hidden-hidden.mean(0)).numpy(),check_finite=False);vf=sv**2/max(float(np.sum(sv**2)),1e-40)
    record=dict(condition=condition,seed=seed,context=context,regime=regime,source_sha256=sha(__file__),checkpoint_sha256=sha(checkpoint),
        initializer_sha256=sha(source),optimization=optimization,fixed_count=int(qualified.sum()),slow_count=int(slow.sum()),
        hidden_residual_RMS=hidden_residual.tolist(),stable_full_residual_RMS=stable_residual.tolist(),naive_float64_full_residual_RMS=raw.tolist(),
        minimum_stable_one_minus_f=float(denom.min()),forget_sigmoid_rounds_to_one_count=int((gf.sigmoid()==1).sum()),
        forget_sigmoid_rounds_to_one_count_by_state=(gf.sigmoid()==1).sum(1).tolist(),
        max_absolute_subtractive_cancellation_error=float(((1-gf.sigmoid())-denom).abs().max()),
        c_abs_max=float(c.abs().max()),c_abs_max_per_state=c.abs().max(1).values.tolist(),
        implied_cell_relaxation_time_seconds_max=float(1/denom.min()/15),native_initial_c_abs_max=float(initial[:,256:].abs().max()),
        hidden_cluster_count=len(np.unique(labels)),hidden_cluster_labels=labels.tolist(),spectra=spectra,native_forward_reachability=flow,
        hidden_geometry=dict(radius_RMS=float((hidden-hidden.mean(0)).square().mean().sqrt()),participation_ratio=float(1/max(float(np.sum(vf**2)),1e-40)),first_five_pc_fractions=vf[:5].tolist()),
        interpretation='Algebraically eliminated c is a mathematical equilibrium condition, not an assumption that native c relaxes instantly. Stable residuals expose cancellation; the actual full LSTM is simulated for100s to check accessibility. Unqualified optimizer points are not fixed points; remote fixed points are not a behavioral ring.')
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True);stem=folder/f'context{context}_{regime}'
    save(stem.with_suffix('.json'),record)
    np.savez_compressed(stem.with_suffix('.npz'),initial=initial.numpy(),solution=solution.numpy(),input=x.numpy(),stable_one_minus_f=denom.numpy(),theta=a['theta'])
    print(json.dumps({k:record[k]for k in ('condition','seed','context','regime','fixed_count','slow_count','hidden_cluster_count','c_abs_max','minimum_stable_one_minus_f')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/equilibrium_elimination');p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(conditions=['gc100','pgc33'],seeds=[1,2],contexts=[0,1],regimes=['goal_withheld','actual_goal_held'],initializers=64,
        scope='Final original-agent equilibrium conditioning refinement, using same16lockedinputcases after full-state and validated-quotient searches stalled.',
        exact_identity='c=i(h)g(h)/sigmoid(-forget_logit(h)); optimize h=o(h)tanh(c(h)) only.',
        objective='Hidden consistency residual, no phase labels or circle regularization. Adam3000/LBFGS300 as previous searches.',
        qualification='HiddenRMS<=1e-8 and stablefullRMS<=1e-8 (slow1e-5). Stable c residual=ig-sigmoid(-f_logit)*c. Ordinary float64 F(z)-z also reported.',
        spectra='Full-state Jacobian uses stable f derivative f*sigmoid(-f_logit), avoiding derivative cancellation. One representative per qualified hidden cluster plus four preassigned phases.',
        accessibility='Original full-state recurrence from original64native-conditionedstates, samefixedinput,60/150/1500updates (4/10/100s); compare h and c to solutions separately.',
        source_sha256=sha(__file__),parents={p:sha(ROOT/p)for p in ['scripts/lstm_endogenous_fixed_points.py','scripts/lstm_endogenous_curve_recovery.py']})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    if args.freeze_only:return
    with threadpool_limits(limits=2):
        for condition in protocol['conditions']:
            for seed in protocol['seeds']:
                for context in protocol['contexts']:
                    for regime in protocol['regimes']:run(condition,seed,context,regime,args)


if __name__=='__main__':main()
