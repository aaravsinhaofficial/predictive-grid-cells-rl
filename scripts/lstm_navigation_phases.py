"""Locate the navigation gain in frozen 1B LSTMs without changing their policy.

Replays the exact code-swap variants and random streams. Measures progress and
return-trip phases. This is a behavioral localization analysis, not by itself
a causal mediation test. Raw step arrays stay local; seed summaries are saved.
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy import stats
from code.common import build_population, build_policy
from lstm_code_swap import MatchedArena, PGC_VARIANTS, GRID_VARIANTS, summarize_episodes

FIELDS = ('distance_before', 'distance_after', 'distance_progress', 'speed',
          'heading_toward_goal_cos', 'motion_toward_goal_cos', 'action', 'entropy',
          'hit', 'wall', 'known', 'age', 'goal_count', 'absolute_turn_rate')
IX = {k:i for i,k in enumerate(FIELDS)}
DT = 4 / 60


def mean(x):
    return float(np.mean(x)) if len(x) else None


def describe_trace(a, episode_len, n, variants):
    """Keep exposure counts; successful return summaries are explicitly conditional."""
    result = {}
    for j, variant in enumerate(variants):
        x = a[:, j*n:(j+1)*n]
        known = x[..., IX['known']] > 0
        distance = x[..., IX['distance_before']]
        phases = {'all_return': known,
                  'near_under_0.6m': known & (distance < .6),
                  'middle_0.6_to_0.9m': known & (distance >= .6) & (distance < .9),
                  'far_over_0.9m': known & (distance >= .9),
                  'first_second_after_respawn': known & (x[..., IX['age']] < 15)}
        phase_rows = {}
        for label, mask in phases.items():
            rows = x[mask]
            turn_valid=mask & (x[...,IX['hit']]==0) & (((np.arange(len(x))+1)%episode_len!=0)[:,None])
            phase_rows[label] = dict(steps=int(mask.sum()), seconds=float(mask.sum()*DT),
                mean_distance_progress_m_per_second=mean(rows[:, IX['distance_progress']]/DT),
                mean_speed_m_per_second=mean(rows[:, IX['speed']]),
                mean_heading_toward_goal_cos=mean(rows[:, IX['heading_toward_goal_cos']]),
                mean_motion_toward_goal_cos=mean(rows[:, IX['motion_toward_goal_cos']]),
                turning_action_fraction=mean((rows[:, IX['action']]<2).astype(float)),
                slow_step_fraction=mean((rows[:, IX['speed']]<.05).astype(float)),
                wall_fraction=mean(rows[:, IX['wall']]),
                mean_policy_entropy=mean(rows[:, IX['entropy']]),
                mean_absolute_turn_rate=mean(x[...,IX['absolute_turn_rate']][turn_valid]),
                turn_rate_nonreset_steps=int(turn_valid.sum()))
        trips=[];first_returns=[]
        for ep in range(len(x)//episode_len):
            for slot in range(n):
                s=x[ep*episode_len:(ep+1)*episode_len,slot]
                hits=np.flatnonzero(s[:, IX['hit']] > 0)
                if len(hits) and hits[0] < episode_len-300:
                    start=int(hits[0]+1)
                    # Exactly 300 decisions remain observable, even if no return succeeds.
                    assert start+300<=episode_len
                    candidates=hits[(hits>=start)&(hits<start+300)]
                    success=bool(len(candidates))
                    end=int(candidates[0]+1) if success else start+300
                    z=s[start:end]
                    near=np.flatnonzero(z[:,IX['distance_after']]<.6)
                    first_near=-1 if z[0,IX['distance_before']]<.6 else (int(near[0]) if len(near) else None)
                    before=(first_near+1)*DT if first_near is not None else len(z)*DT
                    first_returns.append(dict(slot=slot,episode=ep,first_goal_step=int(hits[0]),
                        success_within_20s=success,capped_return_seconds=len(z)*DT,
                        before_near_seconds=before,after_first_near_seconds=len(z)*DT-before,
                        start_distance=float(z[0,IX['distance_before']]),reached_near=first_near is not None))
                for ordinal, start in enumerate(hits + 1, 1):
                    if start >= episode_len: continue
                    future=hits[hits>=start]
                    end=int(future[0]+1) if len(future) else episode_len
                    z=s[start:end]; success=bool(len(future))
                    near=np.flatnonzero(z[:, IX['distance_after']] < .6)
                    first_near=-1 if z[0,IX['distance_before']] < .6 else (int(near[0]) if len(near) else None)
                    # Crossing back out after a near step; a goal hit ends the leg.
                    exits=int(np.sum((z[:, IX['distance_after']] >= .6) &
                                     (z[:, IX['distance_before']] < .6)))
                    travelled=float(np.sum(z[:,IX['speed']])*DT)
                    ideal=max(0.,float(z[0,IX['distance_before']])-.3)
                    trips.append(dict(slot=slot,episode=ep,return_ordinal=ordinal,
                        success=success,steps=len(z),seconds=len(z)*DT,
                        start_distance=float(z[0,IX['distance_before']]),
                        path_length=travelled,ideal_path=ideal,
                        excess_path=travelled-ideal if success else None,
                        path_eff=min(1.,ideal/max(travelled,1e-9)) if success else None,
                        reached_near=first_near is not None,
                        before_near_seconds=(first_near+1)*DT if first_near is not None else len(z)*DT,
                        after_first_near_seconds=(len(z)-first_near-1)*DT if first_near is not None else 0.,
                        near_exit_count=exits,
                        captured_without_near_exit=success and first_near is not None and exits==0))
        complete=[t for t in trips if t['success']]
        reached=[t for t in trips if t['reached_near']]
        trip_summary=dict(total_return_attempts=len(trips),complete_return_attempts=len(complete),
            censored_return_attempts=len(trips)-len(complete),
            reached_near_attempts=len(reached),
            mean_complete_return_seconds=mean([t['seconds'] for t in complete]),
            mean_complete_before_near_seconds=mean([t['before_near_seconds'] for t in complete]),
            mean_complete_after_first_near_seconds=mean([t['after_first_near_seconds'] for t in complete]),
            mean_complete_excess_path_m=mean([t['excess_path'] for t in complete]),
            mean_complete_path_eff=mean([t['path_eff'] for t in complete]),
            fraction_near_attempts_with_exit=mean([t['near_exit_count']>0 for t in reached]),
            mean_near_exit_count=mean([t['near_exit_count'] for t in reached]),
            first_pass_capture_fraction=mean([t['captured_without_near_exit'] for t in reached]))
        first_summary=dict(eligible_episodes=len(first_returns),
            **{k:mean([r[k] for r in first_returns]) for k in ['success_within_20s','capped_return_seconds',
                      'before_near_seconds','after_first_near_seconds','reached_near']})
        actions=x[...,IX['action']].astype(int)
        counts=np.bincount(actions[known],minlength=6)
        frequencies=counts/counts.sum()
        turn=int(np.argmax(counts[:2]));move=int(np.argmax(counts[2:])+2)
        near=known&(distance<.6);far=known&(distance>=.9)
        strategy=dict(action_counts=counts.tolist(),action_fractions=frequencies.tolist(),
            preferred_turn=turn,preferred_translation=move,pair_fraction=float(frequencies[turn]+frequencies[move]),
            near_turn_fraction=float(np.mean(actions[near]<2)),near_translation_fraction=float(np.mean(actions[near]>=2)),
            near_preferred_translation_fraction=float(np.mean(actions[near]==move)),far_turn_fraction=float(np.mean(actions[far]<2)))
        result[variant]=dict(phases=phase_rows,trip_summary=trip_summary,trips=trips,
                            first_return_20s=first_summary,first_return_20s_rows=first_returns,action_strategy=strategy)
    return result


@torch.inference_mode()
def run(seed, condition, args):
    start=time.perf_counter()
    snapshots_only=getattr(args,'snapshots_only',False)
    cfgpath=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
    cfg=json.loads(cfgpath.read_text()); checkpoint=cfgpath.with_suffix('.pt')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert ckpt['frames']==1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device=='mps':torch.mps.manual_seed(args.eval_seed)
    net=build_policy(cfg,args.device).eval();net.load_state_dict(ckpt['policy'],strict=True);net.requires_grad_(False)
    pop=build_population(dict(cfg,pgc_frac=.33,cond='pgc33'),args.device)
    variants=PGC_VARIANTS if condition=='pgc33' else GRID_VARIANTS
    if snapshots_only:variants=('intact',)
    groups,n=len(variants),args.n_envs;total=groups*n
    env=MatchedArena(cfg,n,groups,args.eval_seed,args.device);obs=env.reset_all()
    current_grid=torch.tensor([v in ('current_grid','both_grid') or (condition=='gc100' and v=='intact') for v in variants],device=args.device).repeat_interleave(n)
    goal_grid=torch.tensor([v in ('goal_grid','both_grid') or (condition=='gc100' and v=='intact') for v in variants],device=args.device).repeat_interleave(n)
    state=net.zero_state(total,args.device)
    mixed_goal=torch.zeros(total,512,device=args.device);grid_goal=torch.zeros_like(mixed_goal)
    previous_action=torch.full((total,),-1,dtype=torch.long,device=args.device)
    previous_reward=torch.zeros(total,device=args.device)
    generator=torch.Generator(device=args.device).manual_seed(args.eval_seed+1)
    rows={v:[] for v in variants}; traces=[]
    snapshots=[]
    memory_pos=torch.zeros(total,2,device=args.device)
    memory_dir=torch.zeros_like(memory_pos);memory_vel=torch.zeros_like(memory_pos)
    age=torch.zeros(total,device=args.device)
    last=time.perf_counter()
    for iteration in range(cfg['episode_len']*(args.episodes//n)):
        mixed_code=pop.activity(obs['pos'],obs['dir'],obs['vel'])
        grid_code=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0)
        code=torch.where(current_grid[:,None],grid_code,mixed_code)
        goal=torch.where(goal_grid[:,None],grid_goal,mixed_goal)
        embedding=net.embed(obs['rgb'])
        if iteration % cfg['episode_len'] in (600,900,1200):
            slots=torch.arange(min(32,n),device=args.device)
            slots=slots[env.n_goals[slots]>0]
            if len(slots):
                snapshots.append(dict(
                    h_before=state[0][slots].clone(),c_before=state[1][slots].clone(),
                    embedding=embedding[slots].clone(),
                    pos=obs['pos'][slots].clone(),hd=obs['hd'][slots].clone(),
                    dir=obs['dir'][slots].clone(),vel=obs['vel'][slots].clone(),
                    vf=env.vf[slots].clone(),vs=env.vs[slots].clone(),vr=env.vr[slots].clone(),
                    goal_center=env.goal[slots].clone(),
                    memory_pos=memory_pos[slots].clone(),memory_dir=memory_dir[slots].clone(),
                    memory_vel=memory_vel[slots].clone(),
                    current_code=code[slots].clone(),goal_code=goal[slots].clone(),
                    previous_action=previous_action[slots].clone(),previous_reward=previous_reward[slots].clone(),
                    slot=slots.clone(),episode=torch.full_like(slots,iteration//cfg['episode_len']),
                    step=torch.full_like(slots,iteration%cfg['episode_len'])))
        logits,values,state=net.step_from_embedding(embedding,code,goal,previous_action,previous_reward,state)
        probs=logits.softmax(1)
        uniforms=torch.rand(n,generator=generator,device=args.device).repeat(groups)
        actions=(uniforms[:,None]>probs.cumsum(1)).sum(1).clamp_max(5)
        # Copy quantities reset in-place by env.step; observe() position is already a copy.
        oldpos=obs['pos'];oldgoal=env.goal.clone();counts=env.n_goals.clone()
        toward=oldgoal-oldpos;distance=toward.norm(dim=1);unit=toward/distance.clamp_min(1e-9)[:,None]
        facing=torch.stack((obs['hd'].cos(),obs['hd'].sin()),1)
        obs,raw_reward,done,info=env.step(actions)
        after=(oldgoal-info['hit_pos']).norm(dim=1)
        displacement=info['hit_pos']-oldpos;speed=displacement.norm(dim=1)/env.dt
        motion=displacement/(speed*env.dt).clamp_min(1e-9)[:,None]
        if not snapshots_only:
            traces.append(torch.stack((distance,after,distance-after,speed,(facing*unit).sum(1),
                (motion*unit).sum(1),actions.float(),-(probs*probs.clamp_min(1e-12).log()).sum(1),
                info['hit'].float(),info['hit_wall'].float(),(counts>0).float(),age,counts.float(),env.vr.abs()),1))
        hit=info['hit']
        if bool(hit.any()):
            hm=pop.activity(info['hit_pos'],info['hit_dir'],info['hit_vel'])
            hg=pop.activity(info['hit_pos'],info['hit_dir'],info['hit_vel'],shift_scale=0)
            mixed_goal=torch.where(hit[:,None],hm,mixed_goal);grid_goal=torch.where(hit[:,None],hg,grid_goal)
            memory_pos=torch.where(hit[:,None],info['hit_pos'],memory_pos)
            memory_dir=torch.where(hit[:,None],info['hit_dir'],memory_dir)
            memory_vel=torch.where(hit[:,None],info['hit_vel'],memory_vel)
        age=torch.where(hit|done,torch.zeros_like(age),age+1)
        previous_action=actions;previous_reward=raw_reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            keep=(~done).float();state=tuple(z*keep[:,None] for z in state)
            mixed_goal*=keep[:,None];grid_goal*=keep[:,None]
            memory_pos*=keep[:,None];memory_dir*=keep[:,None];memory_vel*=keep[:,None]
            previous_action=torch.where(done,torch.full_like(actions,-1),actions);previous_reward*=keep
            es={k:v.cpu().tolist() for k,v in info['ep_stats'].items()}
            for j,slot in enumerate(es['env']):
                rows[variants[slot//n]].append(dict(slot=slot%n,episode=iteration//cfg['episode_len'],
                    **{k:es[k][j] for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')}))
        if time.perf_counter()-last>25:
            print(f'{condition} s{seed} phases {iteration+1}/{cfg["episode_len"]*(args.episodes//n)}; {time.perf_counter()-start:.0f}s',flush=True);last=time.perf_counter()
    a=torch.stack(traces).cpu().numpy() if traces else None
    del traces
    behavioral={v:summarize_episodes(rows[v],cfg['episode_len']) for v in variants}
    prior=json.loads((ROOT/f'results/lstm_code_swap_1B/{condition}_s{seed}.json').read_text())
    assert prior['episodes']==args.episodes and prior['n_envs']==n and prior['eval_seed']==args.eval_seed
    for v in variants:
        assert rows[v]==prior['variants'][v]['rows'],f'Original behavior mismatch {condition} s{seed} {v}'
    result=dict(condition=condition,seed=seed,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        variants=behavioral,phase_analysis=describe_trace(a,cfg['episode_len'],n,variants) if a is not None else None,
        baseline_records_match_exactly=True,episodes=args.episodes,n_envs=n,eval_seed=args.eval_seed,
        elapsed_seconds=time.perf_counter()-start)
    if a is not None:
        np.savez_compressed(args.out/f'{condition}_s{seed}_trace.npz',trace=a,fields=np.array(FIELDS),variants=np.array(variants))
    if snapshots:
        snap={k:torch.cat([s[k] for s in snapshots]).cpu().numpy() for k in snapshots[0]}
        temp=args.out/f'.{condition}_s{seed}_action_states.tmp.npz'
        np.savez_compressed(temp,**snap)
        temp.replace(args.out/f'{condition}_s{seed}_action_states.npz')
        result['action_snapshot_count']=len(snap['slot'])
        result['action_snapshot_timing']='Native intact branch only. h/c BEFORE the current input; embedding and sensory/state/memory fields correspond to that same input. Decisions600/900/1200, slots0..31, known goal only; episode/slot identify repeated observations.'
    suffix='_action_states_meta' if snapshots_only else ''
    (args.out/f'{condition}_s{seed}{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'DONE {condition} s{seed} '+', '.join(f'{v}={z["mean_score"]:.2f}' for v,z in behavioral.items())+f';{time.perf_counter()-start:.0f}s',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='mps');ap.add_argument('--episodes',type=int,default=128)
    ap.add_argument('--n-envs',type=int,default=64);ap.add_argument('--eval-seed',type=int,default=737373)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    ap.add_argument('--conditions',nargs='+',default=['pgc33','gc100'])
    ap.add_argument('--snapshots-only',action='store_true',help='Native branch only; preserve full phase results and validate old native records.')
    ap.add_argument('--out',type=Path,default=ROOT/'results/lstm_navigation_phases_1B')
    args=ap.parse_args();assert args.episodes%args.n_envs==0
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol_name='snapshot_protocol.json' if args.snapshots_only else 'protocol.json'
    (args.out/protocol_name).write_text(json.dumps(dict(
        seeds=args.seeds,episodes=args.episodes,n_envs=args.n_envs,eval_seed=args.eval_seed,
        phase_thresholds_m=[.6,.9],respawn_phase_steps=15,fields=FIELDS,
        scope='Descriptive localization of behavior on the existing evaluation set. Thresholds chosen before inspecting phase data. No model or input changes beyond the previously tested code-swap variants. Native condition comparisons do not hold trajectories fixed.',
        checks='Require exact equality of all original per-episode records for all variants.',
        censoring='All step exposures and censored return attempts retained. Complete-trip phase durations are conditional on successful returns; selection limits causal interpretation.'),indent=2)+'\n')
    for seed in args.seeds:
        for condition in args.conditions:
            path=args.out/f'{condition}_s{seed}_action_states.npz' if args.snapshots_only else args.out/f'{condition}_s{seed}.json'
            if path.exists():print(f'Reusing {condition} s{seed}',flush=True);continue
            run(seed,condition,args)

if __name__=='__main__':main()
