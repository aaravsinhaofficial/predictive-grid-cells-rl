"""Matched goal-only gate rescue during closed-loop navigation.

Current predictive code remains intact in every condition. No injected noise.
Primary: eight seeds,64environments x2episodes per condition. Benchmark: a
separate16environment x1episode run; never included in primary statistics.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import torch.nn.functional as F
from code.common import build_policy,build_population
from lstm_code_swap import MatchedArena,gates_and_state,state_from_gates,summarize_episodes,verify_codes

VARIANTS=('intact','goal_ordinary','restore_forget','restore_candidate_output')
RESTORE=((0,1,2,3),(),(1,),(2,3))
PRIMARY=(('goal_ordinary','intact'),('goal_ordinary','restore_forget'),
         ('goal_ordinary','restore_candidate_output'),('restore_forget','restore_candidate_output'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def masks_for(n,device):
    return [torch.tensor([g in chosen for chosen in RESTORE],device=device).repeat_interleave(n)[:,None] for g in range(4)]


def update(net,x_native,x_plain,state,masks):
    ng,ns=gates_and_state(net,x_native,state)
    pg,ps=gates_and_state(net,x_plain,state)
    gates=tuple(torch.where(m,a,b) for m,a,b in zip(masks,ng,pg))
    return state_from_gates(gates,state[1]),(ng,pg,ns,ps)


def validate_update(net,x_native,x_plain,state,masks,parts):
    ng,pg,ns,ps=parts
    # All/no restoration are exact identities for the explicit update.
    all_native=state_from_gates(tuple(ng),state[1]);no_native=state_from_gates(tuple(pg),state[1])
    assert all(torch.equal(a,b) for a,b in zip(all_native,ns))
    assert all(torch.equal(a,b) for a,b in zip(no_native,ps))
    max_lstm=0.
    for x,reference in ((x_native,ns),(x_plain,ps)):
        direct=net.lstm(x,state)
        for a,b in zip(direct,reference):
            max_lstm=max(max_lstm,float((a-b).abs().max()))
            assert torch.allclose(a,b,atol=2e-4,rtol=2e-5)
    # Independent selection BEFORE nonlinearities must match selection AFTER.
    recurrent=F.linear(state[0],net.lstm.weight_hh,net.lstm.bias_hh)
    native=F.linear(x_native,net.lstm.weight_ih,net.lstm.bias_ih)+recurrent
    plain=F.linear(x_plain,net.lstm.weight_ih,net.lstm.bias_ih)+recurrent
    selected=[torch.where(m,a,b) for m,a,b in zip(masks,native.chunk(4,1),plain.chunk(4,1))]
    i,f,g,o=selected
    independent=state_from_gates((i.sigmoid(),f.sigmoid(),g.tanh(),o.sigmoid()),state[1])
    actual=state_from_gates(tuple(torch.where(m,a,b) for m,a,b in zip(masks,ng,pg)),state[1])
    n=len(x_native)//len(VARIANTS)
    assert all(torch.equal(a[:n],b[:n]) for a,b in zip(actual,ns))
    assert all(torch.equal(a[n:2*n],b[n:2*n]) for a,b in zip(actual,ps))
    max_pre=max(float((a-b).abs().max()) for a,b in zip(actual,independent))
    assert max_pre<1e-6,max_pre
    return max_lstm,max_pre


@torch.inference_mode()
def run(seed,args):
    started=time.perf_counter()
    path=ROOT/f'models/pgc33/pgc33_lstm_s{seed}_1000M.json'
    cfg=json.loads(path.read_text());ckpath=path.with_suffix('.pt');ckhash=sha(ckpath)
    ckpt=torch.load(ckpath,map_location='cpu',weights_only=True)
    assert ckpt['frames']==1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device=='mps':torch.mps.manual_seed(args.eval_seed)
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(ckpt['policy'])
    pop=build_population(cfg,args.device)
    n=args.n_envs;groups=len(VARIANTS);total=n*groups
    env=MatchedArena(cfg,n,groups,args.eval_seed,args.device);obs=env.reset_all()
    for k in ('pos','hd','dir','rgb'):
        assert all(torch.equal(obs[k][:n],obs[k][j*n:(j+1)*n]) for j in range(1,groups))
    native_goal=torch.zeros(total,512,device=args.device);plain_goal=torch.zeros_like(native_goal)
    pa=torch.full((total,),-1,dtype=torch.long,device=args.device);pr=torch.zeros(total,device=args.device)
    state=net.zero_state(total,args.device);masks=masks_for(n,args.device)
    generator=torch.Generator(device=args.device).manual_seed(args.eval_seed+1)
    rows={v:[] for v in VARIANTS};step=0;last=started
    errors=dict(lstm=0.,independent_preactivation=0.,unchanged_goal_channels=0.,current_code_difference=0.)
    checks=0;history=[]
    while len(rows['intact'])<args.episodes:
        current=pop.activity(obs['pos'],obs['dir'],obs['vel'])
        e=net.embed(obs['rgb'])
        xn=net._concat(e,current,native_goal,pa,pr,None)
        xp=net._concat(e,current,plain_goal,pa,pr,None)
        previous=state
        state,parts=update(net,xn,xp,previous,masks)
        if step%200==0:
            a,b=validate_update(net,xn,xp,previous,masks,parts)
            errors['lstm']=max(errors['lstm'],a);errors['independent_preactivation']=max(errors['independent_preactivation'],b)
            assert torch.equal(xn[:,:768],xp[:,:768])
            assert torch.equal(xn[:,1280:],xp[:,1280:])
            assert torch.equal(native_goal[:,~pop.is_pgc],plain_goal[:,~pop.is_pgc])
            checks+=1
        logits=net.pi(state[0])
        uniforms=torch.rand(n,generator=generator,device=args.device).repeat(groups)
        actions=(uniforms[:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        obs,reward,done,info=env.step(actions)
        if bool(info['hit'].any()):
            native=pop.activity(info['hit_pos'],info['hit_dir'],info['hit_vel'])
            plain=pop.activity(info['hit_pos'],info['hit_dir'],info['hit_vel'],shift_scale=0)
            assert torch.equal(native[:,~pop.is_pgc],plain[:,~pop.is_pgc])
            native_goal=torch.where(info['hit'][:,None],native,native_goal)
            plain_goal=torch.where(info['hit'][:,None],plain,plain_goal)
        pa,pr=actions,reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            keep=(~done).float();native_goal*=keep[:,None];plain_goal*=keep[:,None]
            state=tuple(z*keep[:,None] for z in state)
            pa=torch.where(done,torch.full_like(pa,-1),pa);pr*=keep
            stats={k:v.cpu().tolist() for k,v in info['ep_stats'].items()}
            for j,slot in enumerate(stats['env']):
                rows[VARIANTS[slot//n]].append(dict(slot=slot%n,episode=step//cfg['episode_len'],
                    **{k:stats[k][j] for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')}))
        step+=1
        if time.perf_counter()-last>25:
            elapsed=time.perf_counter()-started;history.append(dict(step=step,seconds=elapsed))
            print(f'goal-gate s{seed}: {step}/{cfg["episode_len"]*(args.episodes//n)}; {elapsed:.1f}s',flush=True);last=time.perf_counter()
    first=[(r['slot'],r['episode'],r['first_goal_t']) for r in rows['intact']]
    assert all([(r['slot'],r['episode'],r['first_goal_t']) for r in records]==first for records in rows.values())
    assert sha(ckpath)==ckhash
    prior={}
    oldpath=ROOT/f'results/lstm_code_swap_1B/pgc33_s{seed}.json'
    if (args.episodes,n,args.eval_seed)==(128,64,737373) and oldpath.exists():
        old=json.loads(oldpath.read_text())
        for v,oldv in [('intact','intact'),('goal_ordinary','goal_grid')]:
            prior[v]=dict(rows_equal=rows[v]==old['variants'][oldv]['rows'],
                          score_difference=float(np.mean([r['ret'] for r in rows[v]])-old['variants'][oldv]['mean_score']),
                          note='Historical baseline diagnostic; cross-device floating-point trajectories need not be identical.')
    if args.device=='mps':torch.mps.synchronize()
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    result=dict(seed=seed,condition='pgc33',benchmark=args.benchmark,checkpoint_sha256=ckhash,
        episodes=args.episodes,n_envs=n,eval_seed=args.eval_seed,device=args.device,
        variants={v:summarize_episodes(a,cfg['episode_len']) for v,a in rows.items()},
        validation=dict(checks=checks,max_errors=errors,all_gates_native_identity=True,no_gates_plain_identity=True,
            first_goal_times_identical=True,current_prediction_always_intact=True,checkpoint_unchanged=True,
            historical_baseline=prior),elapsed_seconds=elapsed,decisions=step,
        aggregate_env_decisions_per_second=step*total/elapsed,progress=history,
        software=dict(torch=torch.__version__,cuda=torch.version.cuda))
    write_json(args.out/f'pgc33_s{seed}.json',result)
    print(f'DONE s{seed}: '+', '.join(f'{v}={a["mean_score"]:.3f}' for v,a in result['variants'].items())+f'; {elapsed:.1f}s',flush=True)
    return result


def summarize(args):
    import scipy.stats as st
    runs=[json.loads((args.out/f'pgc33_s{s}.json').read_text()) for s in args.seeds]
    assert all(r['benchmark']==args.benchmark for r in runs)
    comparisons={}
    if len(runs)>1 and not args.benchmark:
        for reference,intervention in PRIMARY:
            a=np.array([r['variants'][reference]['mean_score'] for r in runs]);b=np.array([r['variants'][intervention]['mean_score'] for r in runs]);d=b-a
            perms=np.array([np.mean(d*np.array(signs)) for signs in itertools.product((-1,1),repeat=len(d))])
            se=st.sem(d);ci=st.t.interval(.95,len(d)-1,loc=d.mean(),scale=se) if se>0 else (d.mean(),d.mean())
            comparisons[intervention+'_minus_'+reference]=dict(mean_a=float(a.mean()),mean_b=float(b.mean()),difference=float(d.mean()),
                differences=d.tolist(),ci95=list(map(float,ci)),paired_t_p=float(st.ttest_rel(b,a).pvalue) if se>0 else (1. if d[0]==0 else 0.),
                exact_signflip_p=float(np.mean(np.abs(perms)>=abs(d.mean())-1e-12)))
        for source,target in [('paired_t_p','holm_t_p'),('exact_signflip_p','holm_exact_p')]:
            running=0.
            for i,key in enumerate(sorted(comparisons,key=lambda k:comparisons[k][source])):
                running=max(running,min(1.,(len(comparisons)-i)*comparisons[key][source]));comparisons[key][target]=running
    result=dict(protocol=json.loads((args.out/'protocol.json').read_text()),comparisons=comparisons,models=runs)
    write_json(args.out/'summary.json',result)
    if args.benchmark:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['goal_ordinary','restore_forget','restore_candidate_output','intact']
    neural_names=['plain','forget','candidate_output','native'];labels=['Ordinary goal','Forget restored','Candidate + output','Intact']
    colors=['#9299A1','#587DA5','#D87532','#6F573B']
    neuralpath=ROOT/'results/ring_attractor/gate_stability/summary.json'
    neural=json.loads(neuralpath.read_text())['models'] if neuralpath.exists() else None
    fig,axes=plt.subplots(1,2 if neural else 1,figsize=(9,3.6) if neural else (5,3.6),layout='constrained',squeeze=False)
    panels=[(axes[0,-1],np.array([[r['variants'][v]['mean_score'] for v in order] for r in runs]),'Navigation','Episode reward')]
    if neural:
        by={r['seed']:r for r in neural};assert all(s in by for s in args.seeds)
        values=np.array([[by[s]['measurements']['0.0001'][v]['c_error']['60'] for v in neural_names] for s in args.seeds])
        panels.insert(0,(axes[0,0],values,'Memory disturbance after 4 s','Cell-state error / initial error'))
    for ax,values,title,ylabel in panels:
        ax.bar(np.arange(4),values.mean(0),color=colors,yerr=st.sem(values,axis=0) if len(values)>1 else None,capsize=3)
        ax.set(xticks=np.arange(4),xticklabels=labels,title=title,ylabel=ylabel);ax.tick_params(axis='x',labelrotation=35,labelsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Goal code changed; current predictive code always intact · Same four gate conditions',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(args.out/f'goal_gate_memory_and_navigation.{ext}',dpi=180)
    plt.close(fig)
    lines=['# Goal-only gate rescue: memory protection and navigation','',
        'The four conditions change only how the saved goal code enters the LSTM. Current predictive code stays intact. The behavioral experiment adds no state noise; the memory panel comes from the separately measured targeted perturbation experiment.','',
        '![Matched gate conditions](goal_gate_memory_and_navigation.png)','',
        '| Condition | Mean episode reward |','|---|---:|']
    for v,label in zip(order,labels):lines.append(f'| {label} | {np.mean([r["variants"][v]["mean_score"] for r in runs]):.3f} |')
    lines+=['','| Planned score contrast | Difference | 95% interval | Holm t p | Holm exact p |','|---|---:|---:|---:|---:|']
    for k,r in comparisons.items():lines.append(f'| {k} | {r["difference"]:+.3f} | [{r["ci95"][0]:+.3f}, {r["ci95"][1]:+.3f}] | {r["holm_t_p"]:.4g} | {r["holm_exact_p"]:.4g} |')
    lines+=['','Independent units are training seeds. Four score contrasts receive Holm correction separately for paired-t and exhaustive sign-flip tests. The four branches share episode layouts, per-step physical noise, and action-sampling uniforms; each branch evolves its own state and trajectory. Episodes before goal discovery are identical. Weights remain frozen.','',
        'This compares the functions of the same goal-input gate pathways. The neural assay injects a targeted disturbance on a common forced trajectory; the behavioral assay measures unperturbed closed-loop navigation. It does not itself show that the measured memory protection causes a navigation advantage.','',
        '[Results](summary.json) · [Protocol](protocol.json) · [Source](../../../scripts/lstm_goal_gate_behavior.py)']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--n-envs',type=int,default=64)
    p.add_argument('--eval-seed',type=int,default=737373)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_attractor/goal_gate_behavior')
    args=p.parse_args();assert args.episodes%args.n_envs==0
    if args.benchmark and 'benchmark' not in str(args.out):args.out=args.out/'benchmark'
    torch.set_num_threads(args.threads);args.out.mkdir(parents=True,exist_ok=True)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.summarize_only:summarize(args);return
    protocol=dict(role='benchmark' if args.benchmark else 'primary',seeds=args.seeds,
        episodes_per_seed_variant=args.episodes,n_envs_per_variant=args.n_envs,eval_seed=args.eval_seed,device=args.device,
        variants=VARIANTS,restored_gate_indices=dict(zip(VARIANTS,RESTORE)),
        intervention='Current mixed predictive code intact in every branch. Only169predictive goal slots are made ordinary; selected gates recover native predictive goal influence, computed fromsameactualprecedingh/c. Noaddedstate noise.',
        primary_score_comparisons=PRIMARY,test_family='Four paired training-seed score comparisons; Holm separately t andexactsignflips; othermetrics descriptive.',
        episode_pairing='MatchedArena common layouts, physicalnoise, actionuniforms; eachbranchownsitsstateandtrajectory.',
        benchmark_excluded_from_primary=args.benchmark,code_identity={s:verify_codes(s) for s in args.seeds},source_sha256=sha(Path(__file__)))
    oldpath=args.out/'protocol.json'
    if oldpath.exists():
        old=json.loads(oldpath.read_text())
        for k in ('role','episodes_per_seed_variant','n_envs_per_variant','eval_seed','device','variants'):
            assert old[k]==json.loads(json.dumps(protocol[k])),(k,old[k],protocol[k])
    write_json(oldpath,protocol)
    if args.prepare_only:
        print(f'Prepared {protocol["role"]} protocol at {oldpath}; no evaluation launched.',flush=True)
        return
    for seed in args.seeds:
        path=args.out/f'pgc33_s{seed}.json'
        if path.exists():
            old=json.loads(path.read_text())
            assert (old['episodes'],old['n_envs'],old['eval_seed'],old['device'],old['benchmark'])==(args.episodes,args.n_envs,args.eval_seed,args.device,args.benchmark)
            print(f'Using completed seed{seed}',flush=True)
        else:run(seed,args)
    summarize(args)


if __name__=='__main__':main()
