"""Matched-state goal-code interventions and immediate goalward action effects.

No fitted readout and no policy rollout. The physical states are selected from
the previous common scripted replay, without looking at neural results.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from code.common import build_env, build_policy, build_population
from lstm_goal_geometry import goal_response
from summarize_lstm_code_swap import estimate, paired, holm

SOURCE = ROOT / 'results/lstm_variable_1B'
OUT = ROOT / 'results/lstm_action_computation_1B'
VARIANTS = {'gc100': ('intact', 'lookahead_goal', 'center_goal'),
            'pgc33': ('intact', 'plain_goal', 'mirror_goal', 'reverse_goal', 'center_pgc_goal')}
ON_VARIANTS={**VARIANTS,'pgc33':VARIANTS['pgc33']+('current_grid','both_grid','current_reverse')}
ACTION_NAMES=('turn_left','turn_right','forward','backward','strafe_left','strafe_right')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_rows(replay):
    ix = replay['sample_index']
    rows = []
    for e in range(int(replay['episodes'])):
        available = np.flatnonzero((replay['episode'][ix] == e)
            & replay['goal_known'][ix] & (replay['step'][ix] < 1349))
        rows.extend(available[::8].tolist())
    return np.sort(rows)


@torch.inference_mode()
def reconstruct(cfg, replay, rows):
    n = int(replay['episodes'])
    next_index = replay['sample_index'][rows] + n
    wanted = {int(index): j for j, index in enumerate(next_index)}
    env = build_env(cfg, n, int(replay['replay_seed']), 'cpu', stagger=False, paired=True)
    obs = env.reset_all()
    rgb = torch.empty(len(rows), 3, 84, 84)
    kin = {k: np.empty(len(rows), dtype=np.float32) for k in ('vf','vs','vr')}
    image_hash = hashlib.sha256()
    for t in range(int(replay['episode_len'])):
        sl = slice(t*n, (t+1)*n)
        for key in ('pos','hd','vel','dir'):
            assert np.array_equal(obs[key].numpy(), replay[key][sl]), (key,t)
        image_hash.update(np.ascontiguousarray(obs['rgb'].numpy()))
        for e in range(n):
            index = t*n + e
            if index in wanted:
                j=wanted[index];rgb[j]=obs['rgb'][e]
                for key in kin: kin[key][j]=float(getattr(env,key)[e])
        obs, _, _, info = env.step(torch.from_numpy(replay['action'][sl]))
        assert np.array_equal(info['hit'].numpy(),replay['hit'][sl])
    expected=json.loads((SOURCE/'summary.json').read_text())['protocol']['rgb_sha256']
    assert image_hash.hexdigest()==expected
    return rgb,next_index,kin,image_hash.hexdigest()


def utilities(cfg, replay, ix, kin):
    """Exact deterministic environment step for each possible action, no pickup reset."""
    p=replay['pos'][ix].astype(float);g=replay['goal_pos'][ix].astype(float)
    hd=replay['hd'][ix].astype(float)
    targets=np.array([[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
    dt=cfg['action_repeat']/60.; k=1-math.exp(-dt/.25)
    vf=kin['vf'][:,None]+(cfg['v_max']*targets[None,:,0]-kin['vf'][:,None])*k
    vs=kin['vs'][:,None]+(cfg['v_max']*targets[None,:,1]-kin['vs'][:,None])*k
    vr=kin['vr'][:,None]+(cfg['rot_max']*targets[None,:,2]-kin['vr'][:,None])*k
    theta=hd[:,None]+dt*vr
    step=np.stack((vf*np.cos(theta)-vs*np.sin(theta),vf*np.sin(theta)+vs*np.cos(theta)),-1)*dt
    future=(p[:,None,:]+step).clip(-cfg['arena_half']+.08,cfg['arena_half']-.08)
    distance=np.linalg.norm(g-p,axis=1)
    target_distance=np.linalg.norm(g[:,None,:]-future,axis=2)
    remaining=np.maximum(distance-cfg['goal_radius'],0)
    future_remaining=np.maximum(target_distance-cfg['goal_radius'],0)
    # Millimeters of expected next-step progress, with the zero-noise physical update.
    progress=1000*(remaining[:,None]-future_remaining)
    unit=(g-p)/distance[:,None]
    f=np.stack((np.cos(hd),np.sin(hd)),-1);left=np.stack((-np.sin(hd),np.cos(hd)),-1)
    intent=targets[None,:,0,None]*f[:,None,:]+targets[None,:,1,None]*left[:,None,:]
    radial=np.sum(intent*unit[:,None,:],axis=2)
    return progress,radial,distance


def motion_utility(hd,direction):
    f=np.stack((np.cos(hd),np.sin(hd)),-1);left=np.stack((-np.sin(hd),np.cos(hd)),-1)
    target=np.stack((np.zeros_like(f),np.zeros_like(f),f,-f,left,-left),axis=1)
    return np.sum(target*direction[:,None,:],axis=2)


def held_go_capture(data,go,cfg,steps=23):
    """Noiseless actual kinematics with a repeated translation; no goal respawn."""
    p=data['pos'].astype(float).copy();hd=data['hd'].astype(float).copy()
    vf=data['vf'].astype(float).copy();vs=data['vs'].astype(float).copy();vr=data['vr'].astype(float).copy()
    g=data['goal_center'].astype(float);hit=np.zeros(len(p),bool)
    targets={2:(1,0),3:(-1,0),4:(0,1),5:(0,-1)};tf,ts=targets[go]
    dt=cfg['action_repeat']/60.;k=1-math.exp(-dt/.25);lim=cfg['arena_half']-.08
    for _ in range(steps):
        vf+=(tf*cfg['v_max']-vf)*k;vs+=(ts*cfg['v_max']-vs)*k;vr+=(0-vr)*k
        hd+=vr*dt
        step=np.stack((vf*np.cos(hd)-vs*np.sin(hd),vf*np.sin(hd)+vs*np.cos(hd)),1)*dt
        before=p+step;after=before.clip(-lim,lim);wall=np.any(before!=after,axis=1)
        vf[wall]*=.2;vs[wall]*=.2;p=after
        hit|=np.linalg.norm(g-p,axis=1)<cfg['goal_radius']
    return hit


@torch.inference_mode()
def validate_held_go_capture(data,go,cfg):
    rows=np.linspace(0,len(data['pos'])-1,min(12,len(data['pos'])),dtype=int)
    subset={k:v[rows] for k,v in data.items()}
    predicted=held_go_capture(subset,go,cfg)
    env=build_env(cfg,len(rows),474747,'cpu',stagger=False,paired=True)
    env.reset_all();env.noise_v=0.;env.noise_r=0.;env.img_noise=0.
    for k in ('pos','hd','vf','vs','vr'):getattr(env,k)[:]=torch.from_numpy(subset[k])
    # Goal-independent kinematics: prevent automatic pickup respawn, track the real disk externally.
    env.goal[:]=100.
    actual=np.zeros(len(rows),bool)
    for _ in range(23):
        _,_,_,info=env.step(torch.full((len(rows),),go,dtype=torch.long))
        actual|=np.linalg.norm(subset['goal_center']-info['hit_pos'].numpy(),axis=1)<cfg['goal_radius']
    assert np.array_equal(predicted,actual),(predicted,actual)
    return dict(samples=len(rows),labels_match=True)


def add_ray_metrics(record,data,arrays,cfg):
    """Exploratory turn/go diagnostic, aligned to each policy's favored body axis."""
    intact=arrays['intact_probability'];mean_probability=intact.mean(0)
    go=int(2+np.argmax(mean_probability[2:]));turn=int(np.argmax(mean_probability[:2]))
    offset={2:0.,3:np.pi,4:np.pi/2,5:-np.pi/2}[go]
    angle=data['hd']+offset;axis=np.stack((np.cos(angle),np.sin(angle)),1)
    delta=data['goal_center']-data['pos'];distance=np.linalg.norm(delta,axis=1)
    projection=(axis*delta).sum(1);lateral=np.abs(axis[:,0]*delta[:,1]-axis[:,1]*delta[:,0])
    ray_hit=(projection>0)&(lateral<cfg['goal_radius'])
    inertial_hit=held_go_capture(data,go,cfg)
    bearing=(np.arctan2(delta[:,1],delta[:,0])-angle+np.pi)%(2*np.pi)-np.pi
    ids=data['episode']*64+data['slot'];unique=np.unique(ids)
    def avg(values,mask):
        means=[float(values[mask&(ids==ep)].mean()) for ep in unique if np.any(mask&(ids==ep))]
        return float(np.mean(means)) if means else None
    record['action_strategy']=dict(dominant_translation=ACTION_NAMES[go],dominant_turn=ACTION_NAMES[turn],
        dominant_translation_index=go,dominant_turn_index=turn,mean_action_probability=mean_probability.tolist(),
        dominant_pair_probability_mass=float(mean_probability[go]+mean_probability[turn]),
        go_axis_offset_from_facing_degrees=float(np.degrees(offset)),
        basis='Action identities selected from native intact probabilities averaged across saved states; not selected from intervention outcomes.')
    record['validation']['held_go_capture_vs_environment']=validate_held_go_capture(data,go,cfg)
    record['near_inertial_capture']={}
    edges=np.linspace(-np.pi,np.pi,13)
    for name in record['variants']:
        prob=arrays[name+'_probability']
        for subset,mask in [('all',np.ones(len(ids),bool)),('near',distance<.6),('far',distance>=.6)]:
            hits=mask&ray_hit;miss=mask&~ray_hit
            g_hit=avg(prob[:,go],hits);t_miss=avg(prob[:,turn],miss)
            record['variants'][name][subset].update(go_probability_ray_hit=g_hit,
                go_probability_ray_miss=avg(prob[:,go],miss),turn_probability_ray_miss=t_miss,
                balanced_ray_agreement=None if g_hit is None or t_miss is None else .5*(g_hit+t_miss),
                ray_hit_samples=int(hits.sum()),ray_miss_samples=int(miss.sum()))
        counts=[];means=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(bearing>=lo)&(bearing<hi);counts.append(int(m.sum()));means.append(avg(prob[:,go],m))
        record['variants'][name]['go_probability_by_bearing']=dict(edges_degrees=np.degrees(edges).tolist(),
            samples=counts,mean_go_probability=means)
        near=distance<.6;hits=near&inertial_hit;miss=near&~inertial_hit
        gh=avg(prob[:,go],hits);tm=avg(prob[:,turn],miss)
        record['near_inertial_capture'][name]=dict(go_probability_capture=gh,turn_probability_miss=tm,
            go_probability_miss=avg(prob[:,go],miss),balanced_capture_agreement=.5*(gh+tm) if gh is not None and tm is not None else None,
            capture_samples=int(hits.sum()),miss_samples=int(miss.sum()),
            horizon_decisions=23,horizon_seconds=23*cfg['action_repeat']/60.)
    arrays['ray_hit']=ray_hit;arrays['go_bearing_error_radians']=bearing
    arrays['held_go_capture']=inertial_hit
    return record,arrays


@torch.inference_mode()
def validate_utilities(cfg,replay,ix,kin,progress):
    env=build_env(cfg,6,929292,'cpu',stagger=False,paired=True)
    env.noise_v=0.;env.noise_r=0.;env.img_noise=0.
    largest=0.
    for j in np.linspace(0,len(ix)-1,min(10,len(ix)),dtype=int):
        env.reset_all()
        env.pos[:]=torch.from_numpy(replay['pos'][ix[j]])
        env.goal[:]=torch.from_numpy(replay['goal_pos'][ix[j]])
        env.hd[:]=float(replay['hd'][ix[j]])
        for key in kin:getattr(env,key)[:]=float(kin[key][j])
        _,_,_,info=env.step(torch.arange(6))
        dist=np.linalg.norm(replay['goal_pos'][ix[j]].astype(float)-replay['pos'][ix[j]].astype(float))
        after=np.linalg.norm(replay['goal_pos'][ix[j]].astype(float)-info['hit_pos'].numpy().astype(float),axis=1)
        reference=1000*(max(0,dist-cfg['goal_radius'])-np.maximum(0,after-cfg['goal_radius']))
        largest=max(largest,float(np.max(np.abs(reference-progress[j]))))
    assert largest < .0003,largest
    return largest


@torch.inference_mode()
def run_model(c,s,replay,rows,ix,rgb,progress,radial,distance,out):
    stem=ROOT/f'models/{c}/{c}_lstm_s{s}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    ckpt=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert ckpt['frames']==1_000_000_000
    net=build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(ckpt['policy'],strict=True)
    pop=build_population(cfg,'cpu')
    with np.load(SOURCE/f'{c}_s{s}_activity.npz') as archive:
        state=tuple(torch.from_numpy(archive[k][rows].copy()) for k in ('h','c'))
    e=net.embed(rgb)
    current=pop.activity(*(torch.from_numpy(replay[k][ix]) for k in ('pos','dir','vel')))
    pa=torch.from_numpy(replay['prev_action'][ix]);pr=torch.from_numpy(replay['prev_reward'][ix])
    hit,direction,center=[torch.from_numpy(replay[k][ix]) for k in ('memory_pos','memory_dir','goal_pos')]
    assert replay['goal_known'][ix].all()
    output={}; arrays={}
    episodes=replay['episode'][ix]
    for name in VARIANTS[c]:
        goal=goal_response(pop,name,hit,direction,center)
        logits,_,_=net.step_from_embedding(e,current,goal,pa,pr,state)
        prob=logits.softmax(1).numpy()
        arrays[name+'_probability']=prob
        expected_progress=(prob*progress).sum(1)
        expected_radial=(prob*radial).sum(1)
        output[name]={}
        for subset,mask in [('all',np.ones(len(rows),bool)),('near',distance<.6),('far',distance>=.6)]:
            # Equal episode weights within each seed, excluding episodes without this subset.
            episode_progress=[float(expected_progress[mask&(episodes==ep)].mean())
                              for ep in range(16) if np.any(mask&(episodes==ep))]
            episode_radial=[float(expected_radial[mask&(episodes==ep)].mean())
                              for ep in range(16) if np.any(mask&(episodes==ep))]
            output[name][subset]=dict(samples=int(mask.sum()),episodes=len(episode_progress),
                expected_progress_mm=float(np.mean(episode_progress)),
                expected_radial_intent=float(np.mean(episode_radial)),
                episode_progress_mm=episode_progress,episode_radial_intent=episode_radial)
    # Common actor-centered input makes TV independent of any shared-logit shifts.
    for name in VARIANTS[c]:
        tv=.5*np.abs(arrays[name+'_probability']-arrays['intact_probability']).sum(1)
        output[name]['action_tv_vs_intact']=float(tv.mean())
    np.savez_compressed(out/f'{c}_s{s}_actions.npz',**arrays)
    record=dict(condition=c,seed=s,variants=output,checkpoint_sha256=digest(stem.with_suffix('.pt')),
                activity_sha256=digest(SOURCE/f'{c}_s{s}_activity.npz'))
    (out/f'{c}_s{s}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(c,s,{k:round(v['all']['expected_progress_mm'],4) for k,v in output.items()},flush=True)


def summarize(out,protocol):
    runs={(c,s):json.loads((out/f'{c}_s{s}.json').read_text()) for c in VARIANTS for s in range(1,9)}
    estimates={}
    for c in VARIANTS:
        for name in runs[c,1]['variants']:
            estimates[f'{c}/{name}']={}
            for subset in ('all','near','far'):
                metrics=['expected_progress_mm','expected_radial_intent']
                if 'expected_motion_intent' in runs[c,1]['variants'][name][subset]:metrics.append('expected_motion_intent')
                if 'balanced_ray_agreement' in runs[c,1]['variants'][name][subset]:metrics +=[
                    'balanced_ray_agreement','go_probability_ray_hit','go_probability_ray_miss','turn_probability_ray_miss']
                estimates[f'{c}/{name}'][subset]={metric:estimate([runs[c,s]['variants'][name][subset][metric]
                     for s in range(1,9)]) for metric in metrics}
    contrasts={}
    for metric in estimates['pgc33/intact']['all']:
        contrasts[metric]={}
        for subset in ('all','near','far'):
            family={}
            pairs=[('pgc_minus_grid','pgc33/intact','gc100/intact'),
                ('predictive_minus_plain','pgc33/intact','pgc33/plain_goal'),
                ('predictive_minus_reverse','pgc33/intact','pgc33/reverse_goal'),
                ('mirror_minus_predictive','pgc33/mirror_goal','pgc33/intact')]
            if 'pgc33/current_grid' in estimates:pairs +=[
                ('intact_minus_current_grid','pgc33/intact','pgc33/current_grid'),
                ('intact_minus_both_grid','pgc33/intact','pgc33/both_grid'),
                ('intact_minus_current_reverse','pgc33/intact','pgc33/current_reverse')]
            for label,left,right in pairs:
                # The shared paired helper returns b-a.
                family[label]=paired(estimates[right][subset][metric]['values'],estimates[left][subset][metric]['values'])
            holm(family)
            running=0.
            for j,key in enumerate(sorted(family,key=lambda k:family[k]['exact_signflip_p'])):
                running=max(running,min(1.,(len(family)-j)*family[key]['exact_signflip_p']))
                family[key]['holm_exact_p']=running
            contrasts[metric][subset]=family
    summary=dict(protocol=protocol,estimates=estimates,comparisons=contrasts,models=list(runs.values()))
    if 'near_inertial_capture' in runs['pgc33',1]:
        capture_estimates={f'{c}/{name}':{m:estimate([runs[c,s]['near_inertial_capture'][name][m] for s in range(1,9)])
            for m in ('go_probability_capture','turn_probability_miss','go_probability_miss','balanced_capture_agreement')}
            for c in VARIANTS for name in runs[c,1]['variants']}
        comparisons={}
        for label,left,right in pairs:
            comparisons[label]=paired(capture_estimates[right]['balanced_capture_agreement']['values'],
                                      capture_estimates[left]['balanced_capture_agreement']['values'])
        holm(comparisons);running=0.
        for j,key in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['exact_signflip_p'])):
            running=max(running,min(1.,(len(comparisons)-j)*comparisons[key]['exact_signflip_p']));comparisons[key]['holm_exact_p']=running
        summary['near_inertial_capture']=dict(estimates=capture_estimates,comparisons=comparisons,
            caveat='Dominant-go diagnostic, not an optimal-action oracle. PGC seed5 uses a meaningful third action, so its dominant-pair diagnostic ignores that alternative.',
            exclude_seed5_sensitivity={key:paired(np.delete(v['a'],4),np.delete(v['b'],4)) for key,v in comparisons.items()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out,summary)
    if 'near_inertial_capture' in summary:write_turn_go_report(out,summary)
    return summary


def write_turn_go_report(out,summary):
    e=summary['near_inertial_capture']['estimates'];c=summary['near_inertial_capture']['comparisons']
    names=['intact','current_grid','current_reverse'];labels=['Predictive live code','Ordinary live code','Reversed live code']
    values=np.array([e['pgc33/'+name]['balanced_capture_agreement']['values'] for name in names])*100
    fig,ax=plt.subplots(figsize=(7.8,3.7),layout='constrained')
    for seed in range(8):ax.plot(range(3),values[:,seed],color='.72',lw=1,marker='o',ms=4,alpha=.7)
    means=values.mean(1);errors=values.std(1,ddof=1)/np.sqrt(8)
    ax.errorbar(np.arange(3)+.07,means,yerr=errors,fmt='s',color='#D87532',ms=8,capsize=4,lw=2)
    ax.set(xticks=range(3),xticklabels=labels,ylabel='Balanced go / turn agreement (%)',
           title='Near-goal turn / go choices depend on live prediction')
    ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Same preceding RNN state · 8 PGC models · lines: seeds · squares: mean ± SEM',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'turn_go_capture.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# A simple turn-or-go computation','','**Removing live motion prediction makes the RNN more likely to keep going when its preferred movement would miss the nearby goal.** This is a matched-state action effect, not a percentage of navigation performance explained.','',
        '![Same-state live-code changes in near-goal turn/go agreement](turn_go_capture.png)','',
        '## What the policies do','',
        'Fifteen of sixteen native models put at least **99.87%** of action probability on one turn direction and one movement direction in these snapshots. The movement can be forward, backward, or sideways; these are alternative body-axis strategies. PGC seed 5 is an exception: its dominant pair has **88.55%** mass, with substantial backward movement in addition to strafing. The [full trajectory analysis](../../lstm_navigation_phases_1B/README.md) independently finds the same exception.','',
        'The RNN’s action head is linear. Its go-versus-turn log-odds is exactly:',
        '```text','(go weights − turn weights) · hidden activity + (go bias − turn bias)','```',
        'For the models dominated by this pair, that one changing readout controls almost all choices. This does **not** mean the recurrent state itself is one-dimensional.','',
        '## The diagnostic','',
        'At saved native-policy states within **0.6 m** of the goal center, we ask: would holding this model’s preferred movement for **23 decisions (1.53 seconds)** reach the goal disk? We simulate the actual inertia, residual rotation, wall collisions, and 0.3 m pickup radius, with noise off. The hypothetical continuation never controls the real policy.','',
        'We then measure the current probability of **go when that continuation hits**, and **turn when it misses**. The diagnostic averages these two numbers equally. It is not an optimal-action oracle: a miss could require more than 1.53 seconds, and an actual policy can turn again. It also overlooks the extra movement option of PGC seed 5.','',
        'Only the live population code is changed for the main comparison. The previous h/c state, goal snapshot, visual embedding, previous action/reward, and all 343 ordinary live cells stay fixed. No learned decoder is fitted.','',
        '| Near-goal action probability | Predictive live code | Ordinary live code | Reversed live code |',
        '|---|---:|---:|---:|']
    for metric,label in [('go_probability_capture','Go when continuation hits'),('turn_probability_miss','Turn when continuation misses'),
                         ('balanced_capture_agreement','Balanced agreement')]:
        vals=[e['pgc33/'+name][metric]['mean']*100 for name in names]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2f}%' for v in vals)+' |')
    lines+=['','The largest component of the ordinary-code effect is **less turning when continued movement would miss**. This is consistent with anticipatory steering, rather than simply making the next movement faster.','',
        '| Same-state comparison | Agreement difference (percentage points) | 95% paired interval | Holm t p | Holm exact p |',
        '|---|---:|---|---:|---:|']
    for key,label in [('intact_minus_current_grid','Predictive − ordinary live code'),
                      ('intact_minus_current_reverse','Predictive − reversed live code'),
                      ('intact_minus_both_grid','Predictive − both codes ordinary'),
                      ('predictive_minus_plain','Predictive − ordinary goal code')]:
        r=c[key];lines.append(f'| {label} | {100*r["difference"]:+.2f} | [{100*r["ci95"][0]:+.2f}, {100*r["ci95"][1]:+.2f}] | {r["holm_t_p"]:.4g} | {r["holm_exact_p"]:.5f} |')
    sensitivity=summary['near_inertial_capture']['exclude_seed5_sensitivity']['intact_minus_current_grid']
    lines+=['','The live ordinary-code comparison has the same sign in all eight seeds. The paired-t and exact tests disagree at 0.05 after correction: with eight seeds, the smallest two-sided exact p is 0.0078125, and multiplying it by the seven-comparison family gives 0.0546875. Both results are reported. The diagnostics were exploratory additions after observing the policies’ concentrated action choices.','',
        f'Excluding the three-action seed 5 leaves a **{100*sensitivity["difference"]:.2f}-point** live-code effect, with an unadjusted paired interval [{100*sensitivity["ci95"][0]:.2f}, {100*sensitivity["ci95"][1]:.2f}]. This is a structural sensitivity check, not another selected primary test.','',
        'The simpler straight-ray diagnostic agrees: removing live prediction reduces near-goal balanced agreement by 1.78 points; its far-state effect is only 0.095 points with an interval spanning zero. This is not itself a test of the near-versus-far interaction. The original one-step distance-progress metric is inconclusive; it gives little credit to useful turns. Both diagnostics and all variants remain in the numerical summary.','',
        '**Scope:** this identifies an input-sensitive steering computation inside already trained PGC networks. It does not establish why training selected these strategies, or show that this snapshot metric mediates the score advantage. The separate closed-loop interventions supply the navigation evidence.','',
        'The input itself encodes a position 20 cm ahead along motion. This result does not show that the RNN internally simulates a future trajectory.','',
        '[All action metrics and controls](README.md) · [Full results](summary.json) · [Protocol](protocol.json)','']
    (out/'turn-go.md').write_text('\n'.join(lines))


def write_report(out,summary):
    onpolicy=out.name=='on_policy'
    fig,axes=plt.subplots(1,2,figsize=(11,5.6 if onpolicy else 3.5),layout='constrained')
    labels=[('predictive_minus_plain','Original − ordinary'),('predictive_minus_reverse','Original − reversed'),
            ('mirror_minus_predictive','Reflected − original')]
    if onpolicy:
        labels=[('predictive_minus_plain','Original − goal ordinary'),('predictive_minus_reverse','Original − goal reversed'),
                ('mirror_minus_predictive','Goal reflected − original'),
                ('intact_minus_current_grid','Original − current ordinary'),
                ('intact_minus_both_grid','Original − both ordinary'),
                ('intact_minus_current_reverse','Original − current reversed')]
    for ax,subset in zip(axes,('near','far')):
        stats=summary['comparisons']['expected_progress_mm'][subset]
        for j,(key,label) in enumerate(labels):
            r=stats[key];mu=r['difference'];lo,hi=r['ci95']
            ax.errorbar(mu,j,xerr=[[mu-lo],[hi-mu]],fmt='o',color='#D87532',capsize=3)
            jitter=np.linspace(-.11,.11,8)
            ax.scatter(r['differences'],j+jitter,color='#D87532',alpha=.3,s=16)
        ax.axvline(0,color='.55',lw=1);ax.set(yticks=range(len(labels)),yticklabels=[x[1] for x in labels],
            ylim=(len(labels)-.5,-.5),title='Near goal (< 0.6 m)' if subset=='near' else 'Far from goal (≥ 0.6 m)',
            xlabel='Expected next-step progress difference (mm)')
        ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('PGC models · same preceding state · 8 seeds · marginal 95% paired intervals',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'goalward_action_effect.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    where='native policy experience' if onpolicy else 'shared scripted experience'
    lines=[f'# Immediate goalward action effects on {where}', '',
        '**This measures whether changing population-code inputs immediately shifts actions toward the true goal. It is not a navigation-score test.**','',
        '![Same-state goal-code effects on immediate progress](goalward_action_effect.png)','',
        'Each comparison holds the previous hidden/cell state, visual embedding, previous action, and reward fixed. Goal variants change only the saved goal code; the 343 ordinary goal slots remain intact. We compare the native predictive code, ordinary code, reflected predictive shift, reversed predictive shift, and oracle-centered predictive slots. Grid-model controls use the native pickup, an ahead point, or true center.','',
        'For each of six actions, we calculate one noiseless environment step, including current linear/rotational velocity, inertia, updated heading, and wall clipping. We measure distance reduction to the **goal disk**, then average it using the RNN’s action probabilities. No decoder is fitted. An independent environment-step check verifies this calculation.','',
        '| Comparison | All states: change (mm) | Near states: change (mm) | Near 95% interval | Near Holm t / exact p |',
        '|---|---:|---:|---|---|']
    for key,label in labels:
        a=summary['comparisons']['expected_progress_mm']['all'][key];n=summary['comparisons']['expected_progress_mm']['near'][key]
        lines.append(f'| {label} | {a["difference"]:+.4f} | {n["difference"]:+.4f} | [{n["ci95"][0]:+.4f}, {n["ci95"][1]:+.4f}] | {n["holm_t_p"]:.4g} / {n["holm_exact_p"]:.4g} |')
    if onpolicy:lines+=['','Current-only variants remove or reverse the shift of the 169 live predictive cells while retaining the original goal input. The both-ordinary variant removes both shifts. Native current/goal inputs are retained exactly where not intervened on; CPU/MPS reconstruction differences are checked. These additional current-code tests were added after inspecting the first seed’s goal-only results, and before inspecting the other native-policy models.']
    lines+=['',f'Intervals can include both helpful and harmful effects. The seed, not the individual state, is the replicate. {"Seven" if onpolicy else "Four"} comparisons (including the native population contrast) are corrected within each metric/subset; no global correction across metrics/subsets. This is exploratory.','',
        '**Interpretation limits:** a useful turn may have little immediate distance benefit. Holding the preceding state intact also retains information from earlier predictive inputs. The result therefore cannot determine whether sustained predictive input helps over many decisions, nor exclude useful delayed steering.','',
        ('These snapshots come from each native model’s own trajectories. Native grid versus PGC comparisons therefore confound visited states with policy differences. Within-model goal interventions use identical states.' if onpolicy else
         'All sixteen networks see the same scripted physical replay, but their actions do not control it. Their recurrent states reflect this unusual action history. Selection takes every eighth known-goal moving sample within each of sixteen episodes, for 430 states per model. Exact full-image replay reconstruction was verified.'),'',
        'The summary also reports expected target translation toward the center'+(' and alignment with actual motion direction' if onpolicy else '')+', separate, simpler action-intent metrics that likewise undervalue turns.','',
        ('Reproduction requires the native `*_action_states.npz` files from [the navigation-phase rollout](../../lstm_navigation_phases_1B/README.md), plus the original 1B checkpoints. Raw archives and weights remain local.' if onpolicy else 'Reproduction requires the original shared replay/activity archives and 1B checkpoints. Raw arrays and weights remain local.'),'',
        '```sh',f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_action_computation.py'+(' --on-policy' if onpolicy else ''),'```','',
        '[Numerical results](summary.json) · [Protocol](protocol.json)'+(' · [Shared replay probe](../README.md)' if onpolicy else ' · [Native-policy probe](on_policy/README.md)'),'']
    if onpolicy:
        lines[4:4]=['The [turn-or-go analysis](turn-go.md) tests whether the live predictive input changes steering appropriately when continued movement would hit or miss the nearby goal.','']
    (out/'README.md').write_text('\n'.join(lines))


@torch.inference_mode()
def on_policy_model(c,s,source,out):
    path=source/f'{c}_s{s}_action_states.npz'
    data=dict(np.load(path))
    stem=ROOT/f'models/{c}/{c}_lstm_s{s}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    ckpt=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert ckpt['frames']==1_000_000_000
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(ckpt['policy'],strict=True)
    pop=build_population(cfg,'cpu')
    current=pop.activity(*(torch.from_numpy(data[k]) for k in ('pos','dir','vel')))
    current_error=float(np.max(np.abs(current.numpy()-data['current_code'])))
    # Device trig implementations may differ slightly between MPS source and CPU probe.
    assert current_error < .0001,current_error
    replay=dict(pos=data['pos'],hd=data['hd'],goal_pos=data['goal_center'])
    ix=np.arange(len(data['slot']));kin={k:data[k] for k in ('vf','vs','vr')}
    progress,radial,distance=utilities(cfg,replay,ix,kin)
    utility_error=validate_utilities(cfg,replay,ix,kin,progress)
    episodes=data['episode']*64+data['slot'];episode_ids=np.unique(episodes)
    state=tuple(torch.from_numpy(data[k]) for k in ('h_before','c_before'))
    e=torch.from_numpy(data['embedding']);current=torch.from_numpy(data['current_code'])
    ordinary_current=pop.activity(*(torch.from_numpy(data[k]) for k in ('pos','dir','vel')),shift_scale=0)
    reverse_current=pop.activity(torch.from_numpy(data['pos']),-torch.from_numpy(data['dir']),torch.from_numpy(data['vel']))
    if c=='pgc33':
        ordinary_current=torch.where(pop.is_pgc[None],ordinary_current,current)
        reverse_current=torch.where(pop.is_pgc[None],reverse_current,current)
    motion=motion_utility(data['hd'],data['dir'])
    pa=torch.from_numpy(data['previous_action']);pr=torch.from_numpy(data['previous_reward'])
    hit,direction,center=[torch.from_numpy(data[k]) for k in ('memory_pos','memory_dir','goal_center')]
    output={};arrays={}
    for name in ON_VARIANTS[c]:
        goal_name='intact' if name in ('current_grid','current_reverse') else ('plain_goal' if name=='both_grid' else name)
        goal=goal_response(pop,goal_name,hit,direction,center)
        cur=ordinary_current if name in ('current_grid','both_grid') else (reverse_current if name=='current_reverse' else current)
        if goal_name=='intact':
            goal_error=float(np.max(np.abs(goal.numpy()-data['goal_code'])))
            assert goal_error < .0001,goal_error
            # The true original input is retained; no numerical reconstruction drift in baseline.
            goal=torch.from_numpy(data['goal_code'])
        elif c=='pgc33':
            # Preserve the native MPS-produced ordinary slots exactly.
            goal=torch.where(pop.is_pgc[None],goal,torch.from_numpy(data['goal_code']))
        logits,_,_=net.step_from_embedding(e,cur,goal,pa,pr,state)
        prob=logits.softmax(1).numpy();arrays[name+'_probability']=prob
        values=dict(expected_progress_mm=(prob*progress).sum(1),expected_radial_intent=(prob*radial).sum(1),
                    expected_motion_intent=(prob*motion).sum(1))
        output[name]={}
        for subset,mask in [('all',np.ones(len(ix),bool)),('near',distance<.6),('far',distance>=.6)]:
            output[name][subset]=dict(samples=int(mask.sum()),episodes=sum(np.any(mask&(episodes==ep)) for ep in episode_ids))
            for key,value in values.items():
                em=[float(value[mask&(episodes==ep)].mean()) for ep in episode_ids if np.any(mask&(episodes==ep))]
                output[name][subset][key]=float(np.mean(em))
    for name in ON_VARIANTS[c]:output[name]['action_tv_vs_intact']=float(.5*np.abs(arrays[name+'_probability']-arrays['intact_probability']).sum(1).mean())
    record=dict(condition=c,seed=s,variants=output,checkpoint_sha256=digest(stem.with_suffix('.pt')),
                source_snapshot_sha256=digest(path),validation=dict(current_reconstruction_maxerror=current_error,
                goal_reconstruction_maxerror=goal_error,utility_vs_environment_maxerror_mm=utility_error))
    record,arrays=add_ray_metrics(record,data,arrays,cfg)
    np.savez_compressed(out/f'{c}_s{s}_actions.npz',**arrays,distance=distance,progress_mm=progress,radial_intent=radial,episode_id=episodes)
    # NumPy integer booleans in episode counts should be ordinary JSON integers.
    for v in output.values():
        for subset in ('all','near','far'):v[subset]['episodes']=int(v[subset]['episodes'])
    (out/f'{c}_s{s}.json').write_text(json.dumps(record,indent=2)+'\n')
    print('ONPOLICY',c,s,{k:round(v['all']['expected_progress_mm'],4) for k,v in output.items()},flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    ap.add_argument('--on-policy',action='store_true')
    ap.add_argument('--snapshot-source',type=Path,default=ROOT/'results/lstm_navigation_phases_1B')
    args=ap.parse_args()
    if args.on_policy:
        args.out=args.out/'on_policy';args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
        protocol=dict(selection='Native intact-policy states at decisions 600/900/1200, first32environment slots, known goal only. Saved by separate behavior analysis.',
            state_timing='Saved h/c BEFORE this input, plus exact native embedding/current/goal input. Other goal variants are recomputed on CPU.',
            source=str(args.snapshot_source.relative_to(ROOT)),metrics='Expected noiseless one-step goal-disk progress in mm and target translation toward true center. Equal sampled-episode weights within each seed.',
            subsets='All; near (<0.6m from true center); far (>=0.6m).',
            statistics='Exploratory follow-up after scripted-replay analysis. Eight paired seed means. Holm within seven contrasts for each metric/subset; not across metrics/subsets.',
            current_extension='Current-only, both-input and current-reversal variants added after inspecting initial seed3 goal-only results, before other on-policy models. Additional metric is translation alignment with actual motion direction.',
            ray_extension='Exploratory diagnostic added after seeing two-action concentration in native snapshots3–8. Select dominant translation and turn identities from intact probabilities. Straight ray intersects true goal disk when forward projection>0 and absolute lateral distance<0.3m. Balanced agreement averages go-probability on ray hits and turn-probability on misses, each averaged within episode then over sampled episodes. This is not an optimal action/value oracle; it ignores inertia and future turning.',
            inertial_extension='Also label near states by whether23noiselessdecisions (1.533seconds) holdingdominanttranslation wouldhittruegoaldisk, retaininginitialvf/vs/vr andwallcollisions. Norespawn ornewpolicydecisions. Samebalancedgo/turn diagnostic, notoptimalcontrol. PGCseed5isathree-actionexception; exclusionofpairedseed5reportedassensitivity.',
            limitations=['Native models visit different physical states; the native contrast is descriptive.',
                         'Goal-code interventions within each model hold its previous state and current experience fixed.',
                         'One-step utility undercounts useful turn actions and is not navigation performance.'])
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for c in VARIANTS:
            for s in args.seeds:on_policy_model(c,s,args.snapshot_source,args.out)
        if all((args.out/f'{c}_s{s}.json').exists() for c in VARIANTS for s in range(1,9)):summarize(args.out,protocol)
        return
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    replay=dict(np.load(SOURCE/'replay.npz'));rows=choose_rows(replay)
    cfg=json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    protocol=dict(selection='Every eighth saved moving known-goal state within each of 16 episodes, excluding final input. No neural result used.',
        states=len(rows),saved_rows=rows.tolist(),state_timing='Saved post-update h_t/c_t; intervene on actual next input at t+1.',
        metrics='Expected noiseless one-step progress towards the true goal disk in mm, and target translation projected towards true center. Equal episode weight per seed.',
        subsets='All; near center (<0.6m); far (>=0.6m).',
        statistics='Exploratory. Eight training seeds. Four comparisons Holm within each metric/subset; not corrected across metrics/subsets.',
        limitations=['Scripted physical experience; policy does not select actions.','One-step progress undercounts turn actions whose benefit is delayed.',
                    'Same-state goal changes are causal instantaneous input effects, not new closed-loop navigation scores.',
                    'Different native policies have their own recurrent states/weights; native comparison is descriptive.'],
        replay_sha256=digest(SOURCE/'replay.npz'))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rgb,ix,kin,image_hash=reconstruct(cfg,replay,rows)
    protocol['reconstructed_full_rgb_sha256']=image_hash
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    progress,radial,distance=utilities(cfg,replay,ix,kin)
    protocol['utility_vs_environment_maxerror_mm']=validate_utilities(cfg,replay,ix,kin,progress)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    np.savez_compressed(args.out/'state_inputs.npz',rows=rows,next_index=ix,progress_mm=progress,radial_intent=radial,distance=distance,**kin)
    for c in VARIANTS:
        for s in args.seeds:run_model(c,s,replay,rows,ix,rgb,progress,radial,distance,args.out)
    if all((args.out/f'{c}_s{s}.json').exists() for c in VARIANTS for s in range(1,9)):
        summarize(args.out,protocol)


if __name__=='__main__':main()
