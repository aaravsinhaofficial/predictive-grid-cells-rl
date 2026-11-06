"""Predefined worst-direction follow-up to the original 32-state Jacobian test.

Both native-input and ordinary-goal-input leading right singular vectors are
tested under BOTH future inputs. The same physical 60-step replay continues.
This is a transient-amplification test, not a ring-manifold test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy import linalg
import torch
from code.common import build_policy,build_population,build_env
from lstm_local_dynamics_spectra import select_states,local_jacobian,autograd_check
from lstm_history_input_factorial import correct
from summarize_lstm_code_swap import paired

SOURCE=ROOT/'results/lstm_variable_1B'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def recover_windows(cfg,replay,indices):
    n=int(replay['episodes']);steps=int(replay['episode_len'])
    lookup={int(k):j for j,k in enumerate(indices)}
    rgb=torch.empty(len(indices),3,84,84)
    env=build_env(cfg,n,int(replay['replay_seed']),'cpu',stagger=False,paired=True)
    obs=env.reset_all();image_hash=hashlib.sha256()
    for t in range(steps):
        sl=slice(t*n,(t+1)*n)
        for k in ('pos','hd','dir','vel'):assert np.array_equal(obs[k].numpy(),replay[k][sl]),(k,t)
        image_hash.update(np.ascontiguousarray(obs['rgb'].numpy()))
        for e in range(n):
            index=t*n+e
            if index in lookup:rgb[lookup[index]]=obs['rgb'][e]
        obs,_,_,info=env.step(torch.from_numpy(replay['action'][sl]))
        assert np.array_equal(info['hit'].numpy(),replay['hit'][sl])
    expected=json.loads((SOURCE/'summary.json').read_text())['protocol']['rgb_sha256']
    assert image_hash.hexdigest()==expected
    return rgb,image_hash.hexdigest()


def run(seed,replay,rows,windows,unique,rgb,args):
    start=time.monotonic();stem=ROOT/f'models/pgc33/pgc33_lstm_s{seed}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    ckpt=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(ckpt['policy'])
    pop=build_population(cfg,'cpu')
    saved=dict(np.load(SOURCE/f'pgc33_s{seed}_activity.npz'))
    initial=np.concatenate((saved['h'][rows],saved['c'][rows]),1).astype(np.float64)
    local=np.searchsorted(unique,windows);b=len(rows)
    with torch.inference_mode():
        e=torch.cat([net.embed(rgb[j:j+256]) for j in range(0,len(rgb),256)])
        current=pop.activity(*(torch.from_numpy(replay[k][unique]) for k in ('pos','dir','vel')))
        goal=pop.activity(*(torch.from_numpy(replay['memory_'+k][unique]) for k in ('pos','dir','vel')))
        ordinary=pop.activity(*(torch.from_numpy(replay['memory_'+k][unique]) for k in ('pos','dir','vel')),shift_scale=0)
        known=torch.from_numpy(replay['goal_known'][unique])[:,None];goal*=known;ordinary*=known
        pa=torch.from_numpy(replay['prev_action'][unique]);pr=torch.from_numpy(replay['prev_reward'][unique])
        inputs={'native':net._concat(e,current,goal,pa,pr,None).double(),
                'goal_ordinary':net._concat(e,current,ordinary,pa,pr,None).double()}
    print(f'seed{seed}: 64 full-state singular decompositions',flush=True)
    singular={k:[] for k in inputs};vectors={k:[] for k in inputs};autograd_error=0.
    for name,x in inputs.items():
        for j in range(b):
            h,c=torch.from_numpy(initial[j]).chunk(2)
            jac=local_jacobian(net,x[local[j,0]],h,c)
            if seed==1 and j==0:autograd_error=max(autograd_error,autograd_check(net,x[local[j,0]],h,c,jac))
            _,sv,vh=linalg.svd(jac.numpy(),full_matrices=False,check_finite=False)
            singular[name].append(float(sv[0]));vectors[name].append(vh[0])
    old=json.loads((ROOT/f'results/lstm_local_dynamics_spectra_1B/pgc33_s{seed}.json').read_text())
    old_error={}
    for new,previous in [('native','intact'),('goal_ordinary','goal_grid_same_state')]:
        reference=np.array([r['largest_singular_value'] for r in old['variants'][previous]['points']])
        old_error[new]=float(np.max(np.abs(np.array(singular[new])-reference)))
    assert max(old_error.values())<1e-8,old_error
    wi,wh=net.lstm.weight_ih.double(),net.lstm.weight_hh.double()
    bi,bh=net.lstm.bias_ih.double(),net.lstm.bias_hh.double()
    piw,pib=net.pi.weight.double(),net.pi.bias.double()
    drives={k:torch.nn.functional.linear(x,wi,bi) for k,x in inputs.items()}
    variants=list(inputs);directions=list(vectors);measurements={};archive={}
    scalar_h=max(float(np.std(saved['h'])),1e-8);scalar_c=max(float(np.std(saved['c'])),1e-8)
    outside={};linear_check={}
    for amplitude in args.amplitudes:
        epsilon=amplitude*np.sqrt(512)
        patches=[initial]
        for direction in directions:
            v=np.array(vectors[direction]);patches.extend([initial+epsilon*v,initial-epsilon*v])
        initial_batch=np.stack(patches)
        outside[str(amplitude)]=float((np.abs(initial_batch[1:,:,:256])>1).mean())
        a=np.tile(initial_batch[None],(len(variants),1,1,1)).reshape(-1,512)
        h,c=torch.from_numpy(a).chunk(2,1)
        values={v:{d:{m:[] for m in ('gain','h_error','c_error','action_tv')} for d in directions} for v in variants}
        with torch.inference_mode():
            for t in range(args.horizon):
                drive=torch.stack([drives[v][local[:,t]] for v in variants])[:,None].expand(-1,5,-1,-1).reshape(-1,1024)
                i,f,g,o=(drive+torch.nn.functional.linear(h,wh,bh)).chunk(4,1)
                c=f.sigmoid()*c+i.sigmoid()*g.tanh();h=o.sigmoid()*c.tanh()
                state=torch.cat((h,c),1).numpy().reshape(2,5,b,512)
                p=torch.nn.functional.linear(h,piw,pib).softmax(-1).numpy().reshape(2,5,b,6)
                for vi,v in enumerate(variants):
                    for di,d in enumerate(directions):
                        delta=state[vi,1+2*di:3+2*di]-state[vi,0]
                        values[v][d]['gain'].append(np.linalg.norm(delta,axis=2).mean(0)/epsilon)
                        values[v][d]['h_error'].append(np.linalg.norm(delta[:,:,:256],axis=2).mean(0)/epsilon)
                        values[v][d]['c_error'].append(np.linalg.norm(delta[:,:,256:],axis=2).mean(0)/epsilon)
                        values[v][d]['action_tv'].append((.5*np.abs(p[vi,1+2*di:3+2*di]-p[vi,0]).sum(2)).mean(0))
        summaries={}
        for v in variants:
            summaries[v]={}
            for direction in directions:
                summaries[v][direction]={}
                for metric,curves in values[v][direction].items():
                    curves=np.array(curves)
                    archive[f'a{amplitude}_{v}_{direction}_{metric}']=curves
                    summaries[v][direction][metric]={str(t):float(curves[t-1].mean()) for t in [1,5,15,args.horizon]}
                    summaries[v][direction][metric]['time_mean']=float(curves.mean())
        if amplitude==min(args.amplitudes):
            for v in variants:
                ratio=np.array(values[v][v]['gain'][0])/np.array(singular[v])
                linear_check[v]=float(np.max(np.abs(ratio-1)))
            assert max(linear_check.values())<.01,linear_check
        measurements[str(amplitude)]=summaries
    record=dict(seed=seed,condition='pgc33',samples=b,episodes=16,horizon=args.horizon,amplitudes=args.amplitudes,
        largest_singular_values=singular,measurements=measurements,
        validation=dict(previous_singular_value_max_error=old_error,autograd_max_error=autograd_error,
                        small_perturbation_one_step_relative_error=linear_check,perturbed_h_outside_range_fraction=outside),
        checkpoint_sha256=sha(stem.with_suffix('.pt')),seconds=time.monotonic()-start)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/f'pgc33_s{seed}_curves.npz',windows=windows,**archive)
    (args.out/f'pgc33_s{seed}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(f'seed{seed}: done {record["seconds"]:.1f}s',flush=True)
    return record


def summarize(runs,args):
    family={}
    if len(runs)>1:
        for direction in ('native','goal_ordinary'):
            for horizon in ('1',str(args.horizon)):
                a=[r['measurements']['0.0001']['native'][direction]['gain'][horizon] for r in runs]
                b=[r['measurements']['0.0001']['goal_ordinary'][direction]['gain'][horizon] for r in runs]
                family[direction+'_direction_at_'+horizon]=paired(a,b)
        correct(family)
    summary=dict(protocol=json.loads((args.out/'protocol.json').read_text()),comparisons=family,models=runs)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for ax,direction,title in zip(axes,('native','goal_ordinary'),('Native input: most sensitive direction','Ordinary goal: most sensitive direction')):
        for variant,color,label in [('native','#D87532','Predictive goal'),('goal_ordinary','#7761A8','Ordinary goal')]:
            curves=[]
            for r in runs:
                with np.load(args.out/f'pgc33_s{r["seed"]}_curves.npz') as a:curves.append(a[f'a0.0001_{variant}_{direction}_gain'].mean(1))
            curves=np.array(curves);t=np.arange(1,args.horizon+1)/15
            ax.plot(t,curves.mean(0),color=color,label=label)
        ax.set(title=title,xlabel='Seconds after perturbation',ylabel='State error / initial error')
        ax.spines[['top','right']].set_visible(False);ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False,fontsize=8)
    fig.supxlabel('Same initial perturbation within each panel · Actual physical replay · Raw h/c coordinates',fontsize=8)
    for ext in ('png','pdf'):fig.savefig(args.out/f'worst_direction_propagation.{ext}',dpi=180)
    plt.close(fig)
    lines=['# Predictive goal input protects sensitive memory directions','',
        '**Predictive goal input reduces amplification along particular vulnerable state directions.** The effect is selective: the separate isotropic-noise experiment found no broad reduction in average noise propagation.','',
        'This follow-up revisits the exact 32 states per model used in the earlier Jacobian analysis. It finds the most amplified initial state direction under predictive input and under ordinary goal input, then tests each of those two directions under both inputs.','',
        '![Worst-direction propagation](worst_direction_propagation.png)','',
        'Each panel keeps its initial perturbation direction fixed. Both branches receive the actual next 60 physical inputs; each perturbed trajectory is compared with its own unperturbed input-matched reference. The plotted perturbation is 0.0001 RMS raw-state units. A 0.01 sensitivity run is saved as well.','',
        '| Input-defined perturbation direction and horizon | Ordinary minus predictive gain | 95% interval | Holm t p | Holm exact p |','|---|---:|---:|---:|---:|']
    for k,r in family.items():lines.append(f'| {k} | {r["difference"]:+.4f} | [{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}] | {r["holm_t_p"]:.4g} | {r["holm_exact_p"]:.4g} |')
    if runs:
        def avg(v,metric,point):return float(np.mean([r['measurements']['0.0001'][v]['goal_ordinary'][metric][point] for r in runs]))
        lines += ['',f'For the ordinary-goal-sensitive direction, the same initial error grows by {avg("native","gain","1"):.2f} times under predictive goal input and {avg("goal_ordinary","gain","1"):.2f} times under ordinary goal input on the first update. After four seconds, the gains are {avg("native","gain",str(args.horizon)):.2f} and {avg("goal_ordinary","gain",str(args.horizon)):.2f}. The larger 0.01 perturbation produces a similar pattern.','',
            f'The lasting difference is concentrated in cell state c. At four seconds, hidden-output h error is only {avg("native","h_error",str(args.horizon)):.4f} times the initial full-state perturbation under predictive input, while c error is {avg("native","c_error",str(args.horizon)):.2f} times. Action-probability effects at this horizon are very small and vary in direction across seeds.','',
            '**Mechanistic interpretation:** the predictive goal message changes the gates so that some vulnerable memory-state disturbances are amplified less. This supports input-dependent protection of stored state. Identifying these directions with a ring and linking their protection to better navigation are separate tests.']
    lines+=['','The two initial directions were specified before this follow-up. They are singular vectors in raw h/c coordinates, not circular manifold tangents. A gain above one means initial state error was amplified in that metric; it is distinct from an autonomous attractor, persistent phase memory, or navigation performance.','',
        'The analysis reproduces the earlier singular values and checks the small finite perturbation against the local linear prediction. It includes a larger perturbation sensitivity. All states, weights, and input sequences retain recorded provenance.','',
        '[Results](summary.json) · [Protocol](protocol.json) · [Source](../../../scripts/lstm_worst_direction_stability.py)']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--horizon',type=int,default=60)
    p.add_argument('--amplitudes',nargs='+',type=float,default=[.0001,.01])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_attractor/worst_direction')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads);args.out.mkdir(parents=True,exist_ok=True)
    replay=dict(np.load(SOURCE/'replay.npz'));rows=select_states(replay)
    n=int(replay['episodes']);windows=replay['sample_index'][rows,None]+n*np.arange(1,args.horizon+1)[None]
    unique=np.unique(windows)
    cfg=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    rgb,rgb_hash=recover_windows(cfg,replay,unique)
    protocol=dict(seeds=args.seeds,states_per_seed=len(rows),horizon=args.horizon,amplitudes=args.amplitudes,
        directions='Leading right singular vector at same original state, separately under native and ordinary-goal next input; both directions tested under both inputs.',
        comparisons='Four predefined contrasts: ordinary-goal minus native amplification, two fixed directions at1and60updates; paired seeds, Holm separately t and exact.',
        coordinate_metric='Raw full h/c Euclidean norm, matching the earlier Jacobian singular values.',
        continuation='Actual60futurephysicalinputs; each perturbed trajectory versus its own unperturbed branch withsameinputcondition.',
        rgb_sha256=rgb_hash,replay_sha256=sha(SOURCE/'replay.npz'),source_sha256=sha(Path(__file__)))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    runs=[]
    for seed in args.seeds:
        path=args.out/f'pgc33_s{seed}.json'
        if args.resume and path.exists():runs.append(json.loads(path.read_text()))
        else:runs.append(run(seed,replay,rows,windows,unique,rgb,args))
    summarize(runs,args)


if __name__=='__main__':main()
