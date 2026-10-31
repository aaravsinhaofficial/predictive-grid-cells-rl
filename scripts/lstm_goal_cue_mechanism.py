"""Test which geometry in the saved goal code supports frozen PGC navigation.

Only goal messages change. Readout, recurrence, live encoding and task stay
intact. Radial/tangent/center conditions explicitly use the hidden true center.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena, summarize_episodes, verify_codes
from lstm_goal_geometry import reflection

VERSION = 1
VARIANTS = {
    'pgc33': ('intact', 'all_at_ahead', 'all_at_hit', 'coherent_center',
              'pgc_radial', 'pgc_tangent', 'pgc_reverse_radial'),
    'gc100': ('intact', 'coherent_center'),
}
PRIMARY = {
    'all_ahead_minus_intact': ('pgc33/intact', 'pgc33/all_at_ahead'),
    'radial_minus_intact': ('pgc33/intact', 'pgc33/pgc_radial'),
    'radial_minus_tangent': ('pgc33/pgc_tangent', 'pgc33/pgc_radial'),
    'radial_minus_reverse': ('pgc33/pgc_reverse_radial', 'pgc33/pgc_radial'),
}
LABELS = {'intact':'Original', 'all_at_ahead':'All slots ahead',
          'all_at_hit':'All slots at pickup', 'coherent_center':'All slots at center',
          'pgc_radial':'Predictive slots inward', 'pgc_tangent':'Predictive slots sideways',
          'pgc_reverse_radial':'Predictive slots outward'}


def cue_points(hit, direction, center):
    _, radial, radius = reflection(hit, direction, center)
    tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=1)
    sign = torch.where((direction*tangent).sum(1, keepdim=True) >= 0, 1., -1.)
    tangent *= sign
    return {'ahead':hit+.2*direction, 'radial':hit+.2*radial,
            'tangent':hit+.2*tangent, 'reverse_radial':hit-.2*radial,
            'radius':radius, 'radial_direction':radial, 'tangent_direction':tangent}


def goal_response(pop, variant, hit, direction, center):
    if variant == 'intact': return pop.activity(hit, direction)
    if variant == 'all_at_hit': return pop.activity(hit, direction, shift_scale=0)
    if variant == 'all_at_ahead': return pop.activity(hit+.2*direction, direction, shift_scale=0)
    if variant == 'coherent_center': return pop.activity(center, direction, shift_scale=0)
    points = cue_points(hit, direction, center)
    point = points[variant.removeprefix('pgc_')]
    ordinary = pop.activity(hit, direction, shift_scale=0)
    shifted = pop.activity(point, direction, shift_scale=0)
    return torch.where(pop.is_pgc[None], shifted, ordinary)


def synthetic_checks(cfg):
    pop = build_population(cfg, 'cpu')
    rng = np.random.default_rng(20261031)
    center = torch.tensor(rng.uniform(-.8,.8,(256,2)),dtype=torch.float32)
    angles = torch.tensor(rng.uniform(-np.pi,np.pi,256),dtype=torch.float32)
    radial = torch.stack((angles.cos(),angles.sin()),1)
    hit = center-.29*radial
    directions = torch.roll(radial,1,0)
    pts = cue_points(hit,directions,center)
    assert torch.allclose((pts['radial']-center).norm(dim=1),torch.full((256,),.09),atol=1e-6)
    assert torch.allclose((pts['reverse_radial']-center).norm(dim=1),torch.full((256,),.49),atol=1e-6)
    assert (pts['radial_direction']*pts['tangent_direction']).sum(1).abs().max()<1e-6
    for key in ('radial','tangent','reverse_radial'):
        assert torch.allclose((pts[key]-hit).norm(dim=1),torch.full((256,),.2),atol=1e-6)
    original = goal_response(pop,'intact',hit,directions,center)
    if cfg['cond']=='pgc33':
        assert pop.is_pgc.sum()==169
        for name in ('pgc_radial','pgc_tangent','pgc_reverse_radial','all_at_hit'):
            changed=goal_response(pop,name,hit,directions,center)
            assert torch.equal(original[:,~pop.is_pgc],changed[:,~pop.is_pgc])
        ahead=goal_response(pop,'all_at_ahead',hit,directions,center)
        assert torch.equal(original[:,pop.is_pgc],ahead[:,pop.is_pgc])
    return dict(synthetic_samples=256, fixed_shift_length_checked=True,
                orthogonal_tangent_checked=True, selected_slot_identity_checked=True)


@torch.inference_mode()
def run(seed,condition,args):
    start=time.perf_counter()
    config_path=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
    cfg=json.loads(config_path.read_text())
    checks=synthetic_checks(cfg)
    checkpoint=config_path.with_suffix('.pt')
    before_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert ckpt['frames']==1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device=='mps':torch.mps.manual_seed(args.eval_seed)
    net=build_policy(cfg,args.device).eval()
    net.load_state_dict(ckpt['policy'],strict=True)
    net.requires_grad_(False)
    pop=build_population(cfg,args.device)
    variants=VARIANTS[condition]
    n,groups=args.n_envs,len(variants)
    env=MatchedArena(cfg,n,groups,args.eval_seed,args.device)
    obs=env.reset_all()
    goal_code=torch.zeros(n*groups,512,device=args.device)
    pa=torch.full((n*groups,),-1,dtype=torch.long,device=args.device)
    pr=torch.zeros(n*groups,device=args.device)
    state=net.zero_state(n*groups,args.device)
    gen=torch.Generator(device=args.device).manual_seed(args.eval_seed+1)
    rows={v:[] for v in variants}
    hits={v:[] for v in variants}
    step=0; last=start
    while len(rows['intact'])<args.episodes:
        code=pop.activity(obs['pos'],obs['dir'],obs['vel'])
        logits,_,state=net.step(obs['rgb'],code,goal_code,pa,pr,state)
        uniform=torch.rand(n,generator=gen,device=args.device).repeat(groups)
        action=(uniform[:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        center=env.goal.clone()
        known=env.n_goals>0
        obs,reward,done,info=env.step(action)
        hit=info['hit']
        if bool(hit.any()):
            for j,name in enumerate(variants):
                sl=slice(j*n,(j+1)*n)
                message=goal_response(pop,name,info['hit_pos'][sl],info['hit_dir'][sl],center[sl])
                goal_code[sl]=torch.where(hit[sl,None],message,goal_code[sl])
            record=hit&~done
            if bool(record.any()):
                idx=record.nonzero().squeeze(1)
                hp,hd,gc=info['hit_pos'][idx],info['hit_dir'][idx],center[idx]
                pts=cue_points(hp,hd,gc)
                metrics=torch.stack([pts['radius'],(pts['ahead']-gc).norm(dim=1),
                    (pts['radial']-gc).norm(dim=1),(pts['tangent']-gc).norm(dim=1),
                    (pts['reverse_radial']-gc).norm(dim=1),
                    (hd*pts['radial_direction']).sum(1),known[idx].float()],1)
                data=torch.cat((hp,gc,hd,metrics),1).cpu().numpy()
                for i,slot in enumerate(idx.cpu().tolist()):
                    hits[variants[slot//n]].append(np.r_[slot%n,step//cfg['episode_len'],step%cfg['episode_len'],data[i]])
        pa=action;pr=reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            keep=(~done).float();goal_code*=keep[:,None]
            state=tuple(z*keep[:,None] for z in state)
            pa=torch.where(done,torch.full_like(action,-1),action);pr*=keep
            es={k:v.cpu().tolist() for k,v in info['ep_stats'].items()}
            for i,slot in enumerate(es['env']):
                rows[variants[slot//n]].append(dict(slot=slot%n,episode=step//cfg['episode_len'],
                    **{k:es[k][i] for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')}))
        step+=1
        if time.perf_counter()-last>25:
            print(f'{condition} s{seed} goal cues: {step}/{cfg["episode_len"]*(args.episodes//n)}; {time.perf_counter()-start:.0f}s',flush=True)
            last=time.perf_counter()
    first=[r['first_goal_t'] for r in rows['intact']]
    assert all([r['first_goal_t'] for r in rr]==first for rr in rows.values())
    baseline_checks={}
    oldpath=ROOT/f'results/lstm_goal_geometry_1B/{condition}_s{seed}.json'
    if (args.episodes,args.n_envs,args.eval_seed,args.device)==(128,64,737373,'mps') and oldpath.exists():
        old=json.loads(oldpath.read_text())
        matched={'intact':'intact'}
        if condition=='pgc33':matched['all_at_hit']='plain_goal'
        else:matched['coherent_center']='center_goal'
        for new,previous in matched.items():
            assert rows[new]==old['variants'][previous]['rows'],(condition,seed,new,'baseline mismatch')
            baseline_checks[new]=len(rows[new])
    columns=['slot','episode','step','hit_x','hit_y','goal_x','goal_y','dir_x','dir_y',
             'hit_distance','ahead_distance','radial_distance','tangent_distance','reverse_radial_distance',
             'arrival_radial_cosine','was_known']
    hit_arrays={v:np.asarray(hits[v],dtype=np.float64).reshape(-1,len(columns)) for v in variants}
    original_first=hit_arrays['intact'][hit_arrays['intact'][:,-1]==0]
    assert all(np.array_equal(array[array[:,-1]==0],original_first) for array in hit_arrays.values())
    np.savez_compressed(args.out/f'{condition}_s{seed}_hits.npz',columns=np.array(columns),**hit_arrays)
    geometry={}
    for v,array in hit_arrays.items():
        geometry[v]={}
        for label,mask in [('all',np.ones(len(array),bool)),('first',array[:,-1]==0),('return',array[:,-1]==1)]:
            sample=array[mask]
            geometry[v][label]=dict(n=len(sample),**{key:float(sample[:,i].mean()) for i,key in enumerate(columns) if i>=9 and len(sample)})
            if len(sample):
                displacement=sample[:,5:7]-sample[:,3:5]
                arrival=sample[:,7:9]
                projection=float(np.mean(np.sum(displacement*arrival,axis=1)))
                direction_power=float(np.mean(np.sum(arrival*arrival,axis=1)))
                optimal=projection/direction_power
                hit_mse=float(np.mean(sample[:,9]**2))
                ahead_mse=float(np.mean(sample[:,10]**2))
                optimum_mse=float(np.mean(np.sum((displacement-optimal*arrival)**2,axis=1)))
                geometry[v][label].update(
                    center_projection_along_arrival_m=projection,
                    hit_mean_squared_center_error_m2=hit_mse,
                    ahead_mean_squared_center_error_m2=ahead_mse,
                    ahead_squared_error_reduction=1-ahead_mse/hit_mse,
                    hit_mean_inside_margin_m=float(np.mean(cfg['goal_radius']-sample[:,9])),
                    ahead_mean_inside_margin_m=float(np.mean(cfg['goal_radius']-sample[:,10])),
                    ahead_inside_goal_fraction=float(np.mean(sample[:,10]<cfg['goal_radius'])),
                    optimum_scalar_shift_m=optimal,
                    optimum_scalar_mean_squared_error_m2=optimum_mse,
                    optimum_scalar_squared_error_reduction=1-optimum_mse/hit_mse)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before_hash
    result=dict(version=VERSION,condition=condition,seed=seed,checkpoint_frames=ckpt['frames'],
        checkpoint_sha256=before_hash,episodes=args.episodes,n_envs=n,eval_seed=args.eval_seed,device=args.device,
        variants={v:summarize_episodes(rows[v],cfg['episode_len']) for v in variants},geometry=geometry,
        checks={**checks,'first_goal_records_identical':True,'prior_baseline_exact_episode_counts':baseline_checks,
                'first_hit_positions_directions_identical':True,'checkpoint_unchanged':True},elapsed_seconds=time.perf_counter()-start)
    (args.out/f'{condition}_s{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE '+condition+f' s{seed}: '+', '.join(f'{v}={result["variants"][v]["mean_score"]:.2f}' for v in variants),flush=True)


def protocol(args):
    return dict(version=VERSION,seeds=args.seeds,conditions=args.conditions,episodes_per_seed_variant=args.episodes,
        n_envs=args.n_envs,eval_seed=args.eval_seed,device=args.device,variants=VARIANTS,
        primary_score_contrasts=PRIMARY,primary_family='Four paired training-seed score contrasts; Holm separately for t and exact sign-flip.',
        intervention='Only the goal snapshot changes and is refreshed at every pickup. Live code and trained RNN unchanged.',
        oracle_conditions=['coherent_center','pgc_radial','pgc_tangent','pgc_reverse_radial'],
        tangent='Unit perpendicular to inward radial direction, choosing the sign aligned with the arrival tangential component; positive perpendicular if exactly tied.',
        shift_m=.2,goal_radius_m=.3,
        interpretation='Input interventions test use by frozen trained models; out-of-distribution messages and changed trajectories limit claims about training mechanisms.',
        primary_selected_before_full_rollouts=True)


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from summarize_lstm_code_swap import estimate,paired,holm
    from summarize_lstm_goal_geometry import holm_signflips
    proto=json.loads((out/'protocol.json').read_text())
    assert proto['seeds']==list(range(1,9))
    assert proto['primary_score_contrasts']=={key:list(value) for key,value in PRIMARY.items()}
    runs={(c,s):json.loads((out/f'{c}_s{s}.json').read_text()) for c in VARIANTS for s in range(1,9)}
    baseline_checks=0;first_checks=0
    for (condition,seed),r in runs.items():
        assert (r['episodes'],r['n_envs'],r['eval_seed'],r['checkpoint_frames'])==(128,64,737373,1_000_000_000)
        assert r['checks']['checkpoint_unchanged'] and r['checks']['first_goal_records_identical']
        assert r['checks']['first_hit_positions_directions_identical']
        assert set(r['variants'])==set(VARIANTS[condition])
        reference=r['variants']['intact']['rows']
        assert len(reference)==128
        for variant,v in r['variants'].items():
            assert len(v['rows'])==128
            assert [x['first_goal_t'] for x in v['rows']]==[x['first_goal_t'] for x in reference]
            assert v['mean_score']==np.mean([x['ret'] for x in v['rows']])
            first_checks+=128*(variant!='intact')
        baseline_checks+=sum(r['checks']['prior_baseline_exact_episode_counts'].values())
    assert baseline_checks==4096
    metrics=('mean_score','mean_path_eff','post_discovery_goal_rate','mean_first_goal_step','frac_found_goal')
    estimates={};comparisons={}
    for metric in metrics:
        values={f'{c}/{v}':[runs[c,s]['variants'][v][metric] for s in range(1,9)] for c in VARIANTS for v in VARIANTS[c]}
        estimates[metric]={k:estimate(v) for k,v in values.items()}
        comparisons[metric]={k:paired(values[a],values[b]) for k,(a,b) in PRIMARY.items()}
        holm(comparisons[metric]);holm_signflips(comparisons[metric])
    exploratory={}
    score_values={k:v['values'] for k,v in estimates['mean_score'].items()}
    for key,a,b in [('all_hit_minus_intact','pgc33/intact','pgc33/all_at_hit'),
                     ('pgc_center_minus_intact','pgc33/intact','pgc33/coherent_center'),
                     ('grid_center_minus_intact','gc100/intact','gc100/coherent_center'),
                     ('pgc_minus_grid','gc100/intact','pgc33/intact')]:
        exploratory[key]=paired(score_values[a],score_values[b])
    holm(exploratory);holm_signflips(exploratory)
    geometry={}
    for c in VARIANTS:
        geometry[c]={phase:{k:estimate([runs[c,s]['geometry']['intact'][phase][k] for s in range(1,9)])
                     for k in runs[c,1]['geometry']['intact'][phase]} for phase in ('all','first','return')}
    result=dict(protocol=proto,estimates=estimates,primary_score=comparisons['mean_score'],
                secondary_metrics={k:v for k,v in comparisons.items() if k!='mean_score'},
                exploratory_score=exploratory,geometry_intact=geometry,
                validation=dict(models=16,first_goal_records_checked=first_checks,
                                previous_baseline_records_exact=baseline_checks,all_checkpoints_unchanged=True),
                checkpoint_sha256={f'{c}/s{s}':r['checkpoint_sha256'] for (c,s),r in runs.items()},
                geometric_reference=dict(assumptions='Ideal straight boundary crossings through a circular goal, uniform impact parameter, negligible sampling overshoot. This is a reference model, not a fitted description of all trajectories.',
                    radius_m=.3,expected_center_projection_m=.3*np.pi/4,
                    pickup_squared_center_error_m2=.3**2,
                    ahead_20cm_squared_center_error_m2=.3**2+.2**2-2*.2*(.3*np.pi/4),
                    optimum_scalar_squared_center_error_m2=.3**2-(.3*np.pi/4)**2,
                    derivation='Average sqrt(r^2-b^2) over b uniform on [-r,r] is pi*r/4.'),
                statistics='Eight paired training seeds. Four primary score contrasts, Holm t and sign-flip separately. Secondary metric families and four replication/descriptive score contrasts labeled exploratory. No equivalence claims from nonsignificance.',
                limitations=['True-center directions are privileged oracle diagnostics.',
                             'Changing code geometry can be outside the learned input distribution.',
                             'Different actions change subsequent experience; first-discovery times remain exactly matched.',
                             'First-hit geometry shows information available in the input, not how a hidden unit computes it.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
    for ax,c,color in zip(axes,VARIANTS,('#D87532','#3379B5')):
        variants=VARIANTS[c]
        for y,v in enumerate(variants):
            e=estimates['mean_score'][f'{c}/{v}']
            ax.barh(y,e['mean'],height=.55,color=color,alpha=1 if v=='intact' else .7)
            ax.errorbar(e['mean'],y,xerr=e['sem'],fmt='none',color='#253442',capsize=3)
            ax.text(e['mean']+e['sem']+1.2,y,f'{e["mean"]:.1f}',va='center',fontsize=10)
        ax.set(yticks=range(len(variants)),yticklabels=[LABELS[v] for v in variants],
               ylim=(len(variants)-.5,-.5),xlim=(0,125),xlabel='90-second score (mean ± SEM)',
               title='PGC-trained RNNs' if c=='pgc33' else 'Grid-trained RNNs')
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.supxlabel('8 paired training seeds · goal message only · center / inward / sideways / outward use the true center',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(out/f'goal_cue_scores.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.4),layout='constrained')
    for i,(key,label,color) in enumerate([('hit_distance','Pickup','#8B939B'),('ahead_distance','20 cm ahead','#D87532'),('radial_distance','20 cm inward\n(oracle)','#A5BAA6')]):
        e=geometry['pgc33']['first'][key]
        ax.bar(i,100*e['mean'],color=color,width=.55)
        ax.errorbar(i,100*e['mean'],yerr=100*e['sem'],fmt='none',color='#253442',capsize=3)
        ax.text(i,100*(e['mean']+e['sem'])+1,f'{100*e["mean"]:.1f} cm',ha='center')
    ax.set(xticks=[0,1,2],xticklabels=['Pickup','20 cm ahead','20 cm inward\n(oracle)'],ylim=(0,36),ylabel='Distance to hidden center',title='The first pickup is near the goal boundary')
    fig.supxlabel('First discoveries only · original PGC models · mean ± SEM across 8 seeds',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(out/f'first_hit_geometry.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    score=lambda c,v:estimates['mean_score'][f'{c}/{v}']['mean']
    text=['# Which part of the goal message helps?', '',
          '**Making the predictive goal shift point exactly inward improves all eight PGC models. Moving the ordinary pickup marker to the unchanged predictive point hurts all eight.** The trained RNN and live spatial code remain intact.','',
          '![Navigation scores for goal-code geometry interventions](goal_cue_scores.png)','',
          'The ordinary 343 goal slots normally mark the pickup. The predictive 169 slots mark a point 20 cm ahead along the arrival direction. Both are continuously supplied and refreshed at each pickup. No neuron is silenced and there is no retraining.','',
          '| PGC goal message | Score |','|---|---:|']
    text += [f'| {LABELS[v]} | {score("pgc33",v):.2f} |' for v in VARIANTS['pgc33']]
    text += ['', '**The spatial direction is useful, and the ordinary pickup anchor also matters.** These results support learned use of both parts of the goal message. They do not identify a literal vector-subtraction algorithm inside the RNN or show that this one factor explains every part of the trained-population advantage.', '',
             '“All slots ahead” moves only the ordinary anchor slots: the original predictive slots remain exactly unchanged. “All slots at pickup” removes the predictive shift. “All slots at center” supplies one coherent true-center code. The last three conditions keep ordinary slots at the pickup and place predictive slots 20 cm inward, sideways, or outward.', '',
             'Inward, sideways, outward, and center are **oracle diagnostics** using the otherwise hidden true center. Sideways chooses the perpendicular sign matching the arrival direction. All three direction tests preserve the 20 cm shift length; their distances to the goal center differ. Altered messages may fall outside the learned input distribution.', '',
             '## Preselected score comparisons','',
             '| Comparison | Difference | 95% paired interval | Holm t p | Holm exact p |',
             '|---|---:|---:|---:|---:|']
    for key,c in comparisons['mean_score'].items():
        text.append(f'| {key.replace("_"," ")} | {c["difference"]:+.2f} | [{c["ci95"][0]:+.2f}, {c["ci95"][1]:+.2f}] | {c["holm_t_p"]:.4g} | {c["holm_signflip_p"]:.4g} |')
    text += ['', 'The four comparisons form one Holm family. Eight training seeds, not individual pickups, are statistical replicates. Other comparisons and behavior measures are exploratory; all values and intervals are in [summary.json](summary.json). No nonsignificant difference is treated as equivalence.', '',
             '## Information available at the first discovery','',
             'Reward occurs on entering a **30 cm goal disk**, not on reaching its center. The saved pickup is therefore usually near its edge. Arrival direction points roughly inward, so the predictive point steps into the target area.', '',
             '![Geometry of first goal discoveries](first_hit_geometry.png)','',
             f'For the original PGC policies, the first pickup is **{100*geometry["pgc33"]["first"]["hit_distance"]["mean"]:.1f} cm** from the hidden center. Moving 20 cm along the arrival direction reduces the mean error to **{100*geometry["pgc33"]["first"]["ahead_distance"]["mean"]:.1f} cm**. The mean arrival cosine toward the center is **{geometry["pgc33"]["first"]["arrival_radial_cosine"]["mean"]:.3f}**. This uses first discoveries only, before any goal-message intervention has influenced behavior.', '',
             f'Grid-policy first discoveries show the same environmental cue: **{100*geometry["gc100"]["first"]["hit_distance"]["mean"]:.1f} cm → {100*geometry["gc100"]["first"]["ahead_distance"]["mean"]:.1f} cm**. Thus the extra geometry is available in the mixed code; it is not merely a result of PGC policies already making more successful return trips. This alone does not identify the RNN computation.', '',
             f'For PGC first discoveries, the ordinary pickup marker lies only **{100*geometry["pgc33"]["first"]["hit_mean_inside_margin_m"]["mean"]:.2f} cm inside the reward boundary** on average. The predictive marker has a mean inward margin of **{100*geometry["pgc33"]["first"]["ahead_mean_inside_margin_m"]["mean"]:.2f} cm**; **{100*geometry["pgc33"]["first"]["ahead_inside_goal_fraction"]["mean"]:.1f}%** of those ahead points are inside the disk. Aiming toward an interior point could tolerate larger steering errors, but these geometry measurements alone do not show that the RNN aims at that point.', '',
             f'A simple reference calculation explains the scale: with straight boundary crossings and a uniformly distributed impact parameter, the average center displacement along arrival is `π × radius / 4`, or **23.6 cm** here. The measured projection at first discovery is **{100*geometry["pgc33"]["first"]["center_projection_along_arrival_m"]["mean"]:.1f} cm** for PGC policies and **{100*geometry["gc100"]["first"]["center_projection_along_arrival_m"]["mean"]:.1f} cm** for grid policies. The reference assumes ideal crossing geometry; the measured trajectories need not satisfy those assumptions exactly. A 20 cm shift is consequently on the scale of an inward center estimate, rather than the actual pickup location.', '',
             '| First-hit center estimate | Grid-policy pickups | PGC-policy pickups |',
             '|---|---:|---:|',
             f'| Pickup squared error (m²) | {geometry["gc100"]["first"]["hit_mean_squared_center_error_m2"]["mean"]:.5f} | {geometry["pgc33"]["first"]["hit_mean_squared_center_error_m2"]["mean"]:.5f} |',
             f'| 20 cm ahead squared error (m²) | {geometry["gc100"]["first"]["ahead_mean_squared_center_error_m2"]["mean"]:.5f} | {geometry["pgc33"]["first"]["ahead_mean_squared_center_error_m2"]["mean"]:.5f} |',
             f'| Squared error reduction | {100*geometry["gc100"]["first"]["ahead_squared_error_reduction"]["mean"]:.1f}% | {100*geometry["pgc33"]["first"]["ahead_squared_error_reduction"]["mean"]:.1f}% |',
             f'| Best scalar shift along arrival | {100*geometry["gc100"]["first"]["optimum_scalar_shift_m"]["mean"]:.1f} cm | {100*geometry["pgc33"]["first"]["optimum_scalar_shift_m"]["mean"]:.1f} cm |', '',
             'The best scalar shift minimizes squared center error on these observed first pickups; it is a descriptive geometry fit, not a trained policy or an out-of-sample performance result. Each seed is weighted equally. The externally saved PGC goal message retains this arrival-related cue at every later step. An ordinary goal snapshot contains only pickup position; a grid RNN could still retain arrival information in its recurrent history.', '',
             'The ideal crossing model predicts `radius² + shift² − 2 × shift × (π × radius / 4)` for the ahead-point squared error: **0.03575 m²** at a 20 cm shift and 30 cm radius. This numerical reference is close to the observed errors, but its assumptions are an approximation and it is not a model of the learned policy.', '',
             'This is one candidate factor for these frozen 1B models. Earlier 300M training controls with predictive live code and ordinary goal snapshots can still learn useful behavior; these goal-message interventions do not explain every predictive-code benefit across training budgets or architectures.', '',
             '## Checks and reproduction','',
             f'All {first_checks:,} intervention episode first-goal records match their original branch exactly. All {baseline_checks:,} repeated original/plain-goal/grid-center episode records match the earlier goal-geometry experiment exactly. Original checkpoint hashes are unchanged. Means of eight seed means are plotted; score is 10 points per pickup in 90 seconds.', '',
             '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_goal_cue_mechanism.py',
             'python scripts/lstm_goal_cue_mechanism.py --summarize','```','',
             'Reproduction requires the local frozen checkpoints. Raw pickup arrays remain local; JSON summaries and per-episode behavior records are included. [Protocol](protocol.json) · [Previous goal geometry](../lstm_goal_geometry_1B/summary.json)']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    for key,c in comparisons['mean_score'].items():print(key,c['difference'],c['ci95'],c['holm_t_p'],c['holm_signflip_p'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='mps')
    ap.add_argument('--episodes',type=int,default=128)
    ap.add_argument('--n-envs',type=int,default=64)
    ap.add_argument('--eval-seed',type=int,default=737373)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    ap.add_argument('--conditions',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    ap.add_argument('--out',type=Path,default=ROOT/'results/lstm_goal_cue_mechanism_1B')
    ap.add_argument('--protocol-only',action='store_true')
    ap.add_argument('--summarize',action='store_true')
    args=ap.parse_args()
    if args.summarize:
        summarize(args.out)
        return
    assert args.episodes%args.n_envs==0
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    desired=protocol(args)
    proto_path=args.out/'protocol.json'
    if proto_path.exists():assert json.loads(proto_path.read_text())==json.loads(json.dumps(desired)),'Protocol changed'
    else:proto_path.write_text(json.dumps(desired,indent=2)+'\n')
    if args.protocol_only:return
    for seed in args.seeds:
        verify_codes(seed)
        for condition in args.conditions:
            path=args.out/f'{condition}_s{seed}.json'
            if path.exists():
                r=json.loads(path.read_text())
                assert (r['version'],r['episodes'],r['n_envs'],r['eval_seed'],r['device'])==(VERSION,args.episodes,args.n_envs,args.eval_seed,args.device)
                assert tuple(r['variants'])==VARIANTS[condition]
                print(f'Using completed {condition} s{seed}',flush=True)
            else:run(seed,condition,args)


if __name__=='__main__':main()
