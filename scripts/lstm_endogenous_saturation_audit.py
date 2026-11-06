"""Causal observability audit before removing any saturated LSTM coordinates.

All hidden coordinates remain. A common-sign c coordinate is eligible only from
its saturation, never from a circular decoder or phase geometry. Every proposed
removal is checked by clamping the selected c coordinates after EVERY update.
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
from threadpoolctl import threadpool_limits
import lstm_goal_bearing_manifold as base
from refine_lstm_goal_bearing_manifold import elicited
from lstm_endogenous_curve_recovery import advance,sha,save


def candidate(state,threshold=8.):
    c=state[:,256:]
    sign=c[0].sign()
    mask=(c.abs().min(0).values>=threshold)&((c.sign()==sign).all(0))
    return mask,sign*threshold


def clamp_state(state,mask,values):
    return torch.cat((state[:,:256],torch.where(mask,values,state[:,256:])),1)


def sequence_test(cell,actor,state,inputs,mask,values):
    full=state.clone();reduced=clamp_state(state,mask,values)
    larger=state.clone();larger[:,256:]=torch.where(mask,larger[:,256:]*1.5,larger[:,256:])
    trajectory=[];max_hidden=0.;max_probability=0.;max_larger_hidden=0.;max_larger_probability=0.
    min_abs_c=float('inf');max_derivative=0.;max_drift=0.;hidden_rms=0.
    for t,x in enumerate(inputs):
        drive=torch.nn.functional.linear(x[None],cell.weight_ih,cell.bias_ih)
        old=full
        full=advance(cell,drive,full)
        reduced=clamp_state(advance(cell,drive,reduced),mask,values)
        larger=advance(cell,drive,larger)
        pp=actor(full[:,:256]).softmax(1)
        errors=[float((reduced[:,:256]-full[:,:256]).abs().max()),float((actor(reduced[:,:256]).softmax(1)-pp).abs().max()),
                float((larger[:,:256]-full[:,:256]).abs().max()),float((actor(larger[:,:256]).softmax(1)-pp).abs().max())]
        max_hidden=max(max_hidden,errors[0]);max_probability=max(max_probability,errors[1])
        max_larger_hidden=max(max_larger_hidden,errors[2]);max_larger_probability=max(max_larger_probability,errors[3])
        hidden_rms=max(hidden_rms,float((reduced[:,:256]-full[:,:256]).square().mean().sqrt()))
        if mask.any():
            selected=full[:,256:][:,mask]
            min_abs_c=min(min_abs_c,float(selected.abs().min()))
            max_derivative=max(max_derivative,float((1-selected.tanh().square()).max()))
            max_drift=max(max_drift,float((full[:,256:]-old[:,256:])[:,mask].abs().max()))
        if t in (0,14,29,59):trajectory.append(dict(step=t+1,hidden_max_abs=errors[0],actor_max_abs=errors[1],hidden_larger_max_abs=errors[2],actor_larger_max_abs=errors[3]))
    return dict(hidden_max_abs=max_hidden,actor_max_abs=max_probability,hidden_rms_max=hidden_rms,
        hidden_larger_max_abs=max_larger_hidden,actor_larger_max_abs=max_larger_probability,
        min_abs_selected_c=None if not mask.any() else min_abs_c,max_sech_squared_c=max_derivative,max_selected_c_step_drift=max_drift,
        passes=(max_hidden<=1e-5 and max_probability<=1e-6 and max_larger_hidden<=1e-5 and max_larger_probability<=1e-6),trajectory=trajectory)


def optimization_gate_audit(condition,seed,net,out):
    rows=[];cell=net.lstm
    for context in (0,1):
        for regime in ('goal_withheld','actual_goal_held'):
            p=ROOT/f'results/lstm_endogenous_ring/fixed_points/{condition}_s{seed}/context{context}_{regime}.npz'
            a=np.load(p);z=torch.tensor(a['solution']);x=torch.tensor(a['input'])
            gates=torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih)+torch.nn.functional.linear(z[:,:256],cell.weight_hh,cell.bias_hh)
            i,f,g,o=gates.chunk(4,1);ii=i.sigmoid();one_minus_f=(-f).sigmoid();write=ii*g.tanh()
            stable_delta=write-one_minus_f*z[:,256:];residual=advance(cell,torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih),z)-z
            saturated=z[:,256:].abs()>=8;slow_forget=one_minus_f<=1e-6
            total=float(residual.square().sum())
            rows.append(dict(context=context,regime=regime,raw_residual_rms_median=float(residual.square().mean(1).sqrt().median()),
                residual_energy_fraction_c=float(residual[:,256:].square().sum())/total,
                residual_energy_fraction_saturated_c=float((stable_delta.square()*saturated).sum())/total,
                residual_energy_fraction_saturated_slow_forget_c=float((stable_delta.square()*(saturated&slow_forget)).sum())/total,
                saturated_cells_per_state_median=float(saturated.sum(1).double().median()),
                saturated_slow_forget_cells_per_state_median=float((saturated&slow_forget).sum(1).double().median()),
                saturated_c_drift_abs_median=float(stable_delta[saturated].abs().median()) if saturated.any() else None,
                max_c=float(z[:,256:].abs().max()),max_hidden_residual=float(residual[:,:256].abs().max()),
                stable_delta_reconstruction_error=float((stable_delta-residual[:,256:]).abs().max())))
    save(out/'full_state_optimization_gate_audit.json',rows)


def run(condition,seed,args):
    start=time.monotonic();cfg,checkpoint,net,pop=base.load_model(condition,seed,'cpu')
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    collection_args=SimpleNamespace(device='cpu',eval_seed=864201,n_envs=16,prepare=15,future=60,max_steps=12000,contexts=2)
    cache=folder/'native_4s_contexts.npz'
    if cache.exists():
        data=np.load(cache);windows=[{k.split('__',1)[1]:data[k] for k in data if k.startswith(f'{j}__')}for j in range(2)]
    else:
        windows,metadata=base.collect_contexts(condition,seed,collection_args,cfg,net,pop)
        np.savez_compressed(cache,**{f'{j}__{k}':v for j,w in enumerate(windows)for k,v in w.items()});save(folder/'collection.json',metadata)
    net.double();theta=torch.arange(64,dtype=torch.float64)*2*np.pi/64
    with torch.no_grad():
        optimization_gate_audit(condition,seed,net,folder)
        for context,w in enumerate(windows):
            inputs=torch.tensor(w['x'],dtype=torch.float64)
            initial=torch.tensor(np.r_[w['h_before'][0],w['c_before'][0]],dtype=torch.float64)
            observer=torch.tensor(w['pos'][14],dtype=torch.float64)
            state,goal=elicited(net,pop,inputs[:15],initial,observer,theta,0.,False)
            _,ordinary_goal=elicited(net,pop,inputs[:15],initial,observer,theta,0.,True)
            raw_native=inputs[15:75].clone();withdrawn=raw_native.clone();withdrawn[:,768:1280]=0
            common_native=raw_native.clone();common_native[:,768:1280]=goal[:1]
            ordinary=raw_native.clone();ordinary[:,768:1280]=ordinary_goal[:1]
            ordinary[:,256:768]=pop.activity(torch.tensor(w['pos'][15:75],dtype=torch.float64),torch.tensor(w['direction'][15:75],dtype=torch.float64),torch.tensor(w['velocity'][15:75],dtype=torch.float64),shift_scale=0)
            alternate=torch.tensor(windows[1-context]['x'][15:75],dtype=torch.float64)
            stress=torch.stack([raw_native[t] if (t//5)%2==0 else alternate[t]for t in range(60)])
            sequences=dict(recorded_native=raw_native,goal_withheld=withdrawn,common_native_goal=common_native,ordinary_current_and_goal=ordinary,switched_native_context_stress=stress)
            for settled in (0,75):
                z=state.clone()
                fixed_x=inputs[14:15].clone();fixed_x[:,768:1280]=0
                drive=torch.nn.functional.linear(fixed_x,net.lstm.weight_ih,net.lstm.bias_ih)
                for _ in range(settled):z=advance(net.lstm,drive,z)
                mask,values=candidate(z)
                tests={name:sequence_test(net.lstm,net.pi,z,xx,mask,values)for name,xx in sequences.items()}
                record=dict(condition=condition,seed=seed,context=context,settled_updates=settled,source_sha256=sha(__file__),checkpoint_sha256=sha(checkpoint),
                    native_context_sha256=sha(cache),candidate_indices=mask.nonzero().flatten().tolist(),candidate_signs=values[mask].sign().tolist(),
                    candidate_count=int(mask.sum()),retained_dimensions=512-int(mask.sum()),tests=tests,
                    passes_all_futures=bool(mask.any()) and all(t['passes']for t in tests.values()),
                    interpretation='Empirical finite-domain approximate quotient only if every future passes. Exact input-output equivalence or global coordinate inactivity is not inferred from a finite replay.')
                stem=folder/f'context{context}_settle{settled}'
                save(stem.with_suffix('.json'),record)
                np.savez_compressed(stem.with_suffix('.npz'),state=z.numpy(),mask=mask.numpy(),values=values.numpy(),common_input=fixed_x.numpy(),theta=theta.numpy())
                print(json.dumps({k:record[k]for k in ('condition','seed','context','settled_updates','candidate_count','passes_all_futures')}),flush=True)
    print(json.dumps(dict(condition=condition,seed=seed,seconds=time.monotonic()-start)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/saturation_audit');p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(conditions=['gc100','pgc33'],seeds=[1,2],contexts=2,angles=64,settled_updates=[0,75],
        candidate='Common c sign and min |c|>=8 across all 64 phases at intervention start. All h and all other c retained.',
        manipulation='Replace selected c with their common sign times8 before and after every update; separate +50% magnitude kick without repeated clamping.',
        maximum_hidden_error=1e-5,maximum_actor_probability_error=1e-6,
        future='60 uninterrupted native steps (4 seconds) after the15-step elicitation; exact native, goal withdrawn, common native snapshot, ordinary current+snapshot, and every5steps switching between two recorded contexts.',
        collection='First two qualifying windows from unchanged collector, eval_seed864201,16envs,15prepare+60future,max12000steps. Selection uses task validity, never hidden geometry or performance contrasts.',
        analytical_bound='For |c|>=8, sech(c)^2<=4.502e-7. This instantaneous derivative is not treated as a future-output bound; explicit replay tests amplification and loss of saturation.',
        source_sha256=sha(__file__),parents={p:sha(ROOT/p)for p in ['scripts/lstm_goal_bearing_manifold.py','scripts/lstm_endogenous_curve_recovery.py','scripts/lstm_endogenous_fixed_points.py']})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    if args.freeze_only:return
    with threadpool_limits(limits=2),torch.no_grad():
        for condition in protocol['conditions']:
            for seed in protocol['seeds']:run(condition,seed,args)


if __name__=='__main__':main()
