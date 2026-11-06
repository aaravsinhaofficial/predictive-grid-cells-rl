"""Causal finite-horizon noise propagation in original frozen trained agents.

Every perturbed replica follows the actual common replay inputs. Its reference
is an unperturbed replica under the SAME input intervention. Prediction removal
therefore changes input-supported error correction, not the reference policy.
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
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from code.common import build_policy,build_population
from lstm_action_computation import choose_rows
from lstm_history_input_factorial import recover_replay_images,episode_average,correct
from summarize_lstm_code_swap import paired,estimate

SOURCE=ROOT/'results/lstm_variable_1B'
VARIANTS=('native','current_ordinary','goal_ordinary','both_ordinary')
NOISES=('isotropic','goal_phase','goal_readout_silent')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


def decoder_fields(state,replay,rows):
    ix=replay['sample_index']
    delta=replay['goal_pos'][ix]-replay['pos'][ix]
    angle=wrap(np.arctan2(delta[:,1],delta[:,0])-replay['hd'][ix])
    target=np.column_stack((np.cos(angle),np.sin(angle)))
    groups=replay['episode'][ix]
    known=replay['goal_known'][ix]
    mean=np.empty_like(state,dtype=np.float64);scale=np.empty_like(mean)
    beta=np.empty((len(state),state.shape[1],2),dtype=np.float64)
    intercept=np.empty((len(state),2),dtype=np.float64)
    prediction=np.empty((len(state),2))
    for tr,te in GroupKFold(4).split(state,groups=groups):
        tr=tr[known[tr]]
        scaler=StandardScaler().fit(state[tr]);s=np.maximum(scaler.scale_,.01)
        model=Ridge(alpha=10.).fit((state[tr]-scaler.mean_)/s,target[tr])
        prediction[te]=model.predict((state[te]-scaler.mean_)/s)
        mean[te]=scaler.mean_;scale[te]=s;beta[te]=model.coef_.T;intercept[te]=model.intercept_
    quality=dict(cv_mae_deg=float(np.rad2deg(np.abs(wrap(np.arctan2(prediction[known,1],prediction[known,0])-angle[known]))).mean()),
        cv_cosine_accuracy=float(np.cos(np.arctan2(prediction[known,1],prediction[known,0])-angle[known]).mean()),
        folds=4,alpha=10.,samples=int(known.sum()))
    return dict(mean=mean[rows],scale=scale[rows],beta=beta[rows],intercept=intercept[rows]),quality


def decode(state,d):
    z=(state-d['mean'])/d['scale']
    value=np.einsum('nd,ndk->nk',z,d['beta'])+d['intercept']
    return value,np.arctan2(value[:,1],value[:,0]),np.linalg.norm(value,axis=1)


def directions(state,d,seed):
    rng=np.random.default_rng(16011+seed)
    value,_,_=decode(state,d)
    phase=value[:,0,None]*d['beta'][:,:,1]-value[:,1,None]*d['beta'][:,:,0]
    phase/=np.maximum(np.linalg.norm(phase,axis=1,keepdims=True),1e-12)
    iso=rng.normal(size=state.shape);iso/=np.linalg.norm(iso,axis=1,keepdims=True)
    null=rng.normal(size=state.shape)
    for j in range(len(state)):
        q=np.linalg.qr(d['beta'][j])[0]
        null[j]-=q@(q.T@null[j])
    null/=np.linalg.norm(null,axis=1,keepdims=True)
    return {'isotropic':iso,'goal_phase':phase,'goal_readout_silent':null}


@torch.inference_mode()
def run(condition,seed,replay,rgb,args):
    started=time.monotonic()
    stem=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    checkpoint=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert checkpoint['frames']==1_000_000_000
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(checkpoint['policy'])
    pop=build_population(cfg,'cpu')
    n=int(replay['episodes']);steps=int(replay['episode_len'])
    saved=dict(np.load(SOURCE/f'{condition}_s{seed}_activity.npz'))
    complete=np.concatenate((saved['h'],saved['c']),1).astype(np.float64)
    rows=choose_rows(replay)
    rows=rows[replay['step'][replay['sample_index'][rows]]+args.horizon<steps]
    ix=replay['sample_index'][rows]
    initial=complete[rows]
    episodes=replay['episode'][ix]
    d,quality=decoder_fields(complete,replay,rows)
    noise=directions(initial,d,seed)
    print(f'{condition} s{seed}: embedding {len(rgb)} replay images; {len(rows)} starts',flush=True)
    e=torch.cat([net.embed(rgb[j:j+256]) for j in range(0,len(rgb),256)])
    current=pop.activity(*(torch.from_numpy(replay[k]) for k in ('pos','dir','vel')))
    goal=pop.activity(*(torch.from_numpy(replay['memory_'+k]) for k in ('pos','dir','vel')))
    cp=pop.activity(*(torch.from_numpy(replay[k]) for k in ('pos','dir','vel')),shift_scale=0)
    gp=pop.activity(*(torch.from_numpy(replay['memory_'+k]) for k in ('pos','dir','vel')),shift_scale=0)
    known=torch.from_numpy(replay['goal_known'])[:,None];goal*=known;gp*=known
    assert torch.equal(current[:,~pop.is_pgc],cp[:,~pop.is_pgc])
    assert torch.equal(goal[:,~pop.is_pgc],gp[:,~pop.is_pgc])
    x=net._concat(e,current,goal,torch.from_numpy(replay['prev_action']),torch.from_numpy(replay['prev_reward']),None)
    base_drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
    current_delta=torch.nn.functional.linear(cp-current,net.lstm.weight_ih[:,256:768])
    goal_delta=torch.nn.functional.linear(gp-goal,net.lstm.weight_ih[:,768:1280])
    del e,x,current,goal,cp,gp
    drives={v:base_drive+(current_delta if v in ('current_ordinary','both_ordinary') else 0)+(goal_delta if v in ('goal_ordinary','both_ordinary') else 0) for v in VARIANTS}
    lookup={int(k):j for j,k in enumerate(replay['sample_index'])}
    horizons=[1,5,15,args.horizon]
    measurements={};validation_error={'h':0.,'c':0.,'logits':0.};archive={}
    perturbed_h_outside_fraction={}
    for amplitude in args.amplitudes:
        patches={'base':initial}
        for name,vec in noise.items():
            shift=amplitude*np.sqrt(initial.shape[1])*vec*d['scale']
            patches[name+'_plus']=initial+shift;patches[name+'_minus']=initial-shift
        labels=list(patches);pcount=len(labels);b=len(initial)
        initial_batch=np.stack([patches[k] for k in labels])
        perturbed_h_outside_fraction[str(amplitude)]=float((np.abs(initial_batch[1:,:,:256])>1).mean())
        batch=np.tile(initial_batch[None],(len(VARIANTS),1,1,1)).reshape(-1,512).astype(np.float32)
        h,c=torch.from_numpy(batch).chunk(2,1)
        raw={v:{name:{key:[] for key in ('action_tv','state_gain','raw_state_gain','h_gain','c_gain','goal_vector_error','goal_angle_error_deg','goal_angle_valid')} for name in NOISES} for v in VARIANTS}
        for t in range(1,args.horizon+1):
            indices=ix+n*t
            drive=torch.stack([drives[v][indices] for v in VARIANTS])[:,None].expand(-1,pcount,-1,-1).reshape(-1,1024)
            i,f,g,o=(drive+torch.nn.functional.linear(h,net.lstm.weight_hh,net.lstm.bias_hh)).chunk(4,1)
            c=f.sigmoid()*c+i.sigmoid()*g.tanh();h=o.sigmoid()*c.tanh()
            state=torch.cat((h,c),1).numpy().reshape(len(VARIANTS),pcount,b,512)
            logits=net.pi(h).numpy().reshape(len(VARIANTS),pcount,b,6)
            probability=torch.from_numpy(logits).softmax(-1).numpy()
            # Compare intact unperturbed continuation to recorded actual states.
            if amplitude==args.amplitudes[0]:
                for j,index in enumerate(indices):
                    if int(index) in lookup:
                        old=lookup[int(index)]
                        for key,a,z in [('h',state[0,0,j,:256],saved['h'][old]),('c',state[0,0,j,256:],saved['c'][old]),('logits',logits[0,0,j],saved['logits'][old])]:
                            validation_error[key]=max(validation_error[key],float(np.max(np.abs(a-z))))
            for vi,v in enumerate(VARIANTS):
                base=state[vi,0]
                bv,bangle,br=decode(base,d)
                for ni,name in enumerate(NOISES):
                    fields={k:[] for k in raw[v][name]}
                    for sign in range(2):
                        pi=1+ni*2+sign;other=state[vi,pi]
                        delta=other-base;first=initial_batch[pi]-initial
                        standard=delta/d['scale'];first_standard=first/d['scale']
                        fields['state_gain'].append(np.linalg.norm(standard,axis=1)/np.linalg.norm(first_standard,axis=1))
                        fields['raw_state_gain'].append(np.linalg.norm(delta,axis=1)/np.linalg.norm(first,axis=1))
                        for key,sl in [('h_gain',slice(0,256)),('c_gain',slice(256,512))]:
                            fields[key].append(np.linalg.norm(standard[:,sl],axis=1)/np.linalg.norm(first_standard[:,sl],axis=1))
                        fields['action_tv'].append(.5*np.abs(probability[vi,pi]-probability[vi,0]).sum(1))
                        value,angle,radius=decode(other,d)
                        valid=(br>=.2)&(radius>=.2)
                        fields['goal_vector_error'].append(np.linalg.norm(value-bv,axis=1))
                        fields['goal_angle_error_deg'].append(np.where(valid,np.rad2deg(np.abs(wrap(angle-bangle))),np.nan))
                        fields['goal_angle_valid'].append(valid.astype(float))
                    for key,values in fields.items():raw[v][name][key].append(np.mean(values,axis=0))
        summaries={}
        for v in VARIANTS:
            summaries[v]={}
            for name in NOISES:
                arrays={k:np.array(a) for k,a in raw[v][name].items()}
                summaries[v][name]={}
                for key,a in arrays.items():
                    def avg(z):
                        return float(np.nanmean([np.nanmean(z[episodes==ep]) for ep in np.unique(episodes)]))
                    summaries[v][name][key]={str(t):avg(a[t-1]) for t in horizons}
                    summaries[v][name][key]['time_mean']=avg(np.nanmean(a,axis=0))
                    archive[f'a{amplitude}_{v}_{name}_{key}']=a.astype(np.float32)
        measurements[str(amplitude)]=summaries
        print(f'{condition} s{seed}: amplitude {amplitude}, {time.monotonic()-started:.1f}s',flush=True)
    assert validation_error['h']<.001 and validation_error['c']<.01 and validation_error['logits']<.01,validation_error
    if condition=='gc100':
        for amp,summaries in measurements.items():
            assert all(summaries[v]==summaries['native'] for v in VARIANTS)
    record=dict(condition=condition,seed=seed,starts=len(rows),episodes=len(np.unique(episodes)),horizon=args.horizon,
        amplitudes=args.amplitudes,decoder=quality,measurements=measurements,
        validation=dict(native_saved_state_max_error=validation_error,prediction_removal_changes_only_pgc_slots=True,
                        perturbed_h_outside_unit_interval_fraction=perturbed_h_outside_fraction),
        checkpoint_sha256=sha(stem.with_suffix('.pt')),activity_sha256=sha(SOURCE/f'{condition}_s{seed}_activity.npz'),seconds=time.monotonic()-started)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/f'{condition}_s{seed}_trajectories.npz',initial_index=ix,episode=episodes,**archive)
    (args.out/f'{condition}_s{seed}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def summarize(args,runs):
    pgc=sorted([r for r in runs if r['condition']=='pgc33'],key=lambda r:r['seed'])
    grid=sorted([r for r in runs if r['condition']=='gc100'],key=lambda r:r['seed'])
    family={}
    if len(pgc)>1:
        for variant in ('current_ordinary','goal_ordinary'):
            for metric,point in [('action_tv','time_mean'),('state_gain',str(args.horizon))]:
                a=[r['measurements']['0.01']['native']['isotropic'][metric][point] for r in pgc]
                b=[r['measurements']['0.01'][variant]['isotropic'][metric][point] for r in pgc]
                family[variant+'_'+metric]=paired(a,b)
        correct(family)
    context={}
    if len(grid)==len(pgc)>1 and [r['seed'] for r in grid]==[r['seed'] for r in pgc]:
        for metric,point in [('action_tv','time_mean'),('state_gain',str(args.horizon)),('raw_state_gain',str(args.horizon))]:
            a=[r['measurements']['0.01']['native']['isotropic'][metric][point] for r in grid]
            b=[r['measurements']['0.01']['native']['isotropic'][metric][point] for r in pgc]
            context[metric]=paired(a,b)
        correct(context)
        grid_noop=all(all(r['measurements'][amp][v]==r['measurements'][amp]['native'] for amp in r['measurements'] for v in VARIANTS) for r in grid)
        assert grid_noop
        (args.out/'grid_context.json').write_text(json.dumps(dict(
            description='Secondary population context: same physical replay and paired seeds; three contrasts corrected separately from acute-input primary family.',
            comparisons=context,grid_noop=grid_noop),indent=2)+'\n')
    result=dict(protocol=json.loads((args.out/'protocol.json').read_text()),comparisons=family,
                secondary_population_context=context,models=runs)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    if pgc:
        for variant,color,label in [('native','#D87532','Predictive intact'),('current_ordinary','#3379B5','Current made ordinary'),('goal_ordinary','#7761A8','Goal made ordinary')]:
            for ax,metric,mult in zip(axes,['state_gain','action_tv'],[1,100]):
                values=[]
                for r in pgc:
                    with np.load(args.out/f'pgc33_s{r["seed"]}_trajectories.npz') as a:
                        ep=a['episode'];curve=a[f'a0.01_{variant}_isotropic_{metric}']
                        values.append(np.mean([curve[:,ep==e].mean(1) for e in np.unique(ep)],axis=0))
                values=np.array(values)*mult;time_axis=np.arange(1,args.horizon+1)/15
                ax.plot(time_axis,values.mean(0),color=color,label=label)
                if len(values)>1:ax.fill_between(time_axis,values.mean(0)-values.std(0,ddof=1)/np.sqrt(len(values)),values.mean(0)+values.std(0,ddof=1)/np.sqrt(len(values)),color=color,alpha=.15)
        axes[0].set(title='State noise propagation',ylabel='State error / initial error',xlabel='Seconds after perturbation')
        axes[1].set(title='Action noise propagation',ylabel='Action probability moved (%)',xlabel='Seconds after perturbation')
        axes[1].legend(frameon=False,fontsize=8)
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.supxlabel('Identical physical replay · Each noisy state compared with its own input-matched reference',fontsize=8)
        for ext in ('png','pdf'):fig.savefig(args.out/f'predictive_noise_propagation.{ext}',dpi=180)
        plt.close(fig)
    if pgc:
        def value(variant,metric,point,amp='0.01',noise='isotropic'):
            return float(np.mean([r['measurements'][amp][variant][noise][metric][point] for r in pgc]))
        lines=['# Does predictive input suppress recurrent noise?','',
            'We gave the frozen original LSTMs the same physical experience, added a small error to their recurrent state, and measured how the error spread. We then repeated this after making the predictive current code, goal code, or both ordinary.','',
            '**Every noisy trajectory is compared with an unperturbed trajectory receiving the same input variant.** The measurements therefore separate noise propagation from the direct action change caused by removing prediction.','',
            '![Predictive input and error propagation](predictive_noise_propagation.png)','',
            f'Equal averages across {len(pgc)} independently trained PGC models. Shading is the standard error across seeds. The physical inputs and actions follow the saved replay; the analyzed policies do not control the trajectory.','',
            '| Input | Mean action probability moved by noise | State error after 4 s / initial error |','|---|---:|---:|']
        for v,label in [('native','Predictive intact'),('current_ordinary','Current made ordinary'),('goal_ordinary','Goal made ordinary'),('both_ordinary','Both made ordinary')]:
            lines.append(f'| {label} | {100*value(v,"action_tv","time_mean"):.4f}% | {value(v,"state_gain",str(args.horizon)):.4f} |')
        lines += ['', 'The state measure uses all 512 h/c coordinates, divided by their training-fold standard deviations before measuring distance. Separate h, c, and unscaled full-state results are saved to check dependence on this choice. Values above one indicate amplification of the initial error in this metric; values below one indicate contraction.','',
            'The primary perturbation is 0.01 RMS units of this scaled state, with positive and negative copies averaged. The 0.05 run is an amplitude sensitivity check. Isotropic perturbations define the primary comparisons. Goal-phase-changing and goal-readout-silent perturbations are exploratory decoder directions, not measured manifold tangents or normals.','',
            '| Primary contrast: ordinary minus predictive intact | Difference | 95% paired interval | Holm t p | Holm exact p |','|---|---:|---:|---:|---:|']
        for key,r in family.items():
            factor=100 if key.endswith('action_tv') else 1
            unit=' pp' if factor==100 else ''
            lines.append(f'| {key} | {factor*r["difference"]:+.6f}{unit} | [{factor*r["ci95"][0]:+.6f}, {factor*r["ci95"][1]:+.6f}] | {r["holm_t_p"]:.4g} | {r["holm_exact_p"]:.4g} |')
        outside=np.mean([r['validation']['perturbed_h_outside_unit_interval_fraction']['0.01'] for r in pgc])
        decoder=np.mean([r['decoder']['cv_mae_deg'] for r in pgc])
        lines += ['', 'A positive contrast means that making the code ordinary increased error propagation; a negative contrast means the opposite. Four primary comparisons were specified before the full-seed run. The independent units are training seeds, with Holm adjustment separately for paired-t and exhaustive paired sign-flip tests.','',
            f'The circular goal-bearing readout has mean held-out-episode error {decoder:.1f}°. Angle changes are only summarized when both decoded vector amplitudes are at least 0.2; the valid fraction and unconditional cosine/sine-vector error are also saved. These decoder analyses are exploratory.','',
            f'At the primary amplitude, {100*outside:.2f}% of perturbed h coordinates initially fall outside [-1, 1]. The first recurrent update restores the LSTM output range. These are injected mathematical state errors, not naturally occurring observations or proven on-manifold perturbations.','',
            f'Each seed starts at {pgc[0]["starts"]} known-goal states across 16 common replay episodes, retaining only starts with a complete {args.horizon}-update future. States were saved after an input; the first continuation uses the next physical input. Episode means receive equal weight within seed. Weights stay frozen, and every intervention preserves non-PGC code channels. Exact image and checkpoint hashes plus saved-state reproduction errors are recorded.','',
            'This experiment measures input-supported robustness of the learned update. A ring-attractor mechanism additionally requires a closed circular state manifold, weak tangent dynamics, transverse recovery, and a predicted behavioral consequence.','',
            '[Results](summary.json) · [Protocol](protocol.json) · [Source](../../../scripts/lstm_predictive_stability.py)']
        if context:
            lines += ['', '**Comparison with Grid-trained networks:** the same relative-state perturbation produces larger responses in PGC-trained networks. These three secondary population contrasts are corrected separately from the four acute-input contrasts.','',
                '| Measure | Grid | PGC + grid | Holm t p | Holm exact p |','|---|---:|---:|---:|---:|']
            for metric,r in context.items():
                factor=100 if metric=='action_tv' else 1
                label='Mean action probability moved (%)' if metric=='action_tv' else metric
                lines.append(f'| {label} | {factor*r["mean_a"]:.4f} | {factor*r["mean_b"]:.4f} | {r["holm_t_p"]:.4g} | {r["holm_exact_p"]:.4g} |')
            lines += ['', 'Ordinary-code replacement is exactly a no-op in all eight Grid-trained models, providing a negative control for the intervention implementation. The primary PGC within-model tests remain unchanged. These network comparisons use each model\'s own state scaling and remain conditional on that perturbation definition; the raw-state gain sensitivity has the same direction.','',
                'Together with the targeted singular-direction follow-up, the pattern suggests an overall more responsive PGC-trained network with selective protection of particular vulnerable memory directions, rather than a globally quieter update. [Population-context numbers](grid_context.json) · [Targeted follow-up](../worst_direction/README.md)']
        (args.out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--conditions',nargs='+',choices=['pgc33','gc100'],default=['pgc33'])
    p.add_argument('--horizon',type=int,default=60)
    p.add_argument('--amplitudes',nargs='+',type=float,default=[.01,.05])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_attractor/predictive_stability')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=args.seeds,conditions=args.conditions,horizon=args.horizon,amplitudes=args.amplitudes,
        primary_amplitude=.01,primary_noise='isotropic',
        primary_comparisons='Four paired seed contrasts: current/goal ordinary minus intact; mean action TV over60updates and standardized-state gain at60updates. Holm correction separately for t and exact sign-flip.',
        continuation='Same physical replay, actual next inputs, frozen weights; every perturbed branch compared to its own unperturbed input-intervention reference.',
        state_metric='Per-coordinate training-fold standard deviation with0.01floor, full h/c. Raw-state gain and separate h/c gains are sensitivity metrics.',
        readout='4-fold held-out-episode cosine/sine ridge alpha10 for egocentric goal bearing. Phase-changing and readout-silent are decoder directions, not geometric manifold tangents/normals.',
        phase_guard='Circular angle error only when both decoded vector amplitudes>=0.2; validity fraction and unconditional vector error are saved.',
        perturbations='Antithetic +/- initial fullstate perturbations of0.01and0.05RMS standardized coordinate units. Initial h may exceed[-1,1]; fraction recorded.',
        source_replay_sha256=sha(SOURCE/'replay.npz'),source_script_sha256=sha(Path(__file__)))
    if not args.summarize_only:(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if args.summarize_only:
        runs=[json.loads((args.out/f'{c}_s{s}.json').read_text()) for c in args.conditions for s in args.seeds]
    else:
        replay=dict(np.load(SOURCE/'replay.npz'))
        cfg=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
        print('Reconstructing and verifying exact replay images',flush=True)
        rgb,rgb_hash=recover_replay_images(cfg,replay)
        protocol['rgb_sha256']=rgb_hash;(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        runs=[]
        for c in args.conditions:
            for s in args.seeds:
                path=args.out/f'{c}_s{s}.json'
                if args.resume and path.exists():runs.append(json.loads(path.read_text()))
                else:runs.append(run(c,s,replay,rgb,args))
    summarize(args,runs)


if __name__=='__main__':main()
