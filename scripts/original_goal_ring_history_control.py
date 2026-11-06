"""Measured arrival-history control using only observed ordinary map rates.

Collect actual native Gridactor first pickups. Decode recent current-code
positions and the saved ordinary pickup code; reconstruct the held last moving
direction exactly as ArenaEnv defines it. Compare with independently inverted
PGC ahead rates. No new behavioral intervention or training.
"""
import argparse
from collections import deque
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from rl_goal_memory import GoalCircleArena
from original_rl_goal_ring import MapInverse,sha,save_json


def arrival_from_decoded_positions(position,threshold=.05*4/60):
    displacements=position[1:]-position[:-1]
    lengths=np.linalg.norm(displacements,axis=1)
    moving=np.flatnonzero(lengths>threshold)
    if not len(moving):return None,None
    last=int(moving[-1])
    return displacements[last]/lengths[last],last


@torch.inference_mode()
def collect(args):
    started=time.perf_counter();cfgpath=ROOT/'models/gc100/gc100_lstm_s1_1000M.json';cfg=json.loads(cfgpath.read_text())
    checkpoint=cfgpath.with_suffix('.pt');weight_hash=sha(checkpoint)
    weight=torch.load(checkpoint,map_location='cpu',weights_only=True);assert weight['frames']==1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device=='mps':torch.mps.manual_seed(args.eval_seed)
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(weight['policy'])
    ordinary=build_population(cfg,args.device);mixed=build_population(dict(cfg,pgc_frac=.33,cond='pgc33'),args.device)
    ordinary_idx=(~mixed.is_pgc).nonzero().flatten();pgc_idx=mixed.is_pgc.nonzero().flatten()
    env=GoalCircleArena(cfg,args.n_envs,args.eval_seed,args.device,radius=.65,paired=True)
    obs=env.reset_all();state=net.zero_state(args.n_envs,args.device)
    goal=torch.zeros(args.n_envs,512,device=args.device)
    pa=torch.full((args.n_envs,),-1,dtype=torch.long,device=args.device);pr=torch.zeros(args.n_envs,device=args.device)
    generator=torch.Generator(device=args.device).manual_seed(args.eval_seed+1)
    history=deque(maxlen=30);records=[];episode_rows=[]
    for step in range(cfg['episode_len']*(args.episodes//args.n_envs)):
        code=ordinary.activity(obs['pos'],obs['dir'],obs['vel'])
        history.append(code[:,ordinary_idx].cpu().numpy())
        logits,_,state=net.step(obs['rgb'],code,goal,pa,pr,state,None)
        action=(torch.rand(args.n_envs,generator=generator,device=args.device)[:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        before_count=env.n_goals.clone();before_t=env.t.clone();true_goal=env.goal.clone()
        obs,reward,done,info=env.step(action)
        first=info['hit']&(before_count==0)
        if bool(info['hit'].any()):
            hit_code=ordinary.activity(info['hit_pos'],info['hit_dir'],info['hit_vel'])
            goal=torch.where(info['hit'][:,None],hit_code,goal)
        if bool(first.any()):
            ids=first.nonzero().flatten()
            observed_mixed=mixed.activity(info['hit_pos'][ids],info['hit_dir'][ids],info['hit_vel'][ids])
            for j,i in enumerate(ids.cpu().tolist()):
                length=min(int(before_t[i])+1,30)
                recent=np.stack([x[i]for x in list(history)[-length:]]+[observed_mixed[j,ordinary_idx].cpu().numpy()])
                records.append(dict(slot=i,episode=step//cfg['episode_len'],step=step,ordinary_history=recent,
                    observed_ahead=observed_mixed[j,pgc_idx].cpu().numpy(),true_hit=info['hit_pos'][i].cpu().numpy(),
                    true_direction=info['hit_dir'][i].cpu().numpy(),true_goal=true_goal[i].cpu().numpy()))
        pa,pr=action,reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            stats=info['ep_stats']
            for j,i in enumerate(stats['env'].cpu().tolist()):episode_rows.append(dict(slot=i,episode=step//cfg['episode_len'],first_goal_step=int(stats['first_goal_t'][j])))
            keep=(~done).float();goal*=keep[:,None];state=tuple(x*keep[:,None]for x in state)
            pa=torch.where(done,torch.full_like(pa,-1),pa);pr*=keep
    assert sha(checkpoint)==weight_hash
    assert len(records)==sum(r['first_goal_step']>=0 for r in episode_rows)
    assert len({(r['slot'],r['episode'])for r in records})==len(records)
    assert len(episode_rows)==args.episodes
    return cfg,records,episode_rows,dict(seconds=time.perf_counter()-started,checkpoint_sha256=weight_hash)


def analyze(cfg,records,episodes,collection,args):
    pop=build_population(dict(cfg,pgc_frac=.33,cond='pgc33'),'cpu')
    ordinary_idx=(~pop.is_pgc).nonzero().flatten();pgc_idx=pop.is_pgc.nonzero().flatten()
    inverse_ordinary=MapInverse(pop,ordinary_idx);inverse_ahead=MapInverse(pop,pgc_idx)
    lengths=[len(r['ordinary_history'])for r in records]
    observed=torch.tensor(np.concatenate([r['ordinary_history']for r in records]))
    positions,_=inverse_ordinary.decode(observed)
    ahead,_=inverse_ahead.decode(torch.tensor(np.stack([r['observed_ahead']for r in records])))
    results=[];cursor=0;data={}
    for i,(record,length)in enumerate(zip(records,lengths)):
        decoded=positions[cursor:cursor+length].numpy();cursor+=length
        direction,last=arrival_from_decoded_positions(decoded)
        row=dict(slot=record['slot'],episode=record['episode'],step=record['step'],history_positions=length,
            history_direction_available=direction is not None)
        if direction is not None:
            estimate=decoded[-1]+.20*direction
            expected=ahead[i].numpy()
            z=estimate/np.linalg.norm(estimate);zp=expected/np.linalg.norm(expected)
            rates=pop.activity(torch.tensor(np.stack((.65*z,.65*zp))),shift_scale=0)
            row.update(direction_error_deg=float(np.rad2deg(abs(np.arctan2(np.cross(direction,record['true_direction']),np.dot(direction,record['true_direction']))))),
                last_moving_interval=last,intervals_since_last_moving=length-2-last,
                point_difference_m=float(np.linalg.norm(estimate-expected)),ring_state_difference=float(np.linalg.norm(z-zp)),
                ring_phase_difference_deg=float(np.rad2deg(abs(np.arctan2(np.cross(z,zp),np.dot(z,zp))))),
                regenerated_goal_code_max_difference=float((rates[0]-rates[1]).abs().max()),
                regenerated_goal_code_rmse=float((rates[0]-rates[1]).square().mean().sqrt()),
                decoded_pickup_error_m=float(np.linalg.norm(decoded[-1]-record['true_hit'])))
        results.append(row)
        for key in ('ordinary_history','observed_ahead','true_hit','true_direction','true_goal'):data[f'{i}_{key}']=record[key]
    valid=[r for r in results if r['history_direction_available']]
    metrics={k:dict(mean=float(np.mean([r[k]for r in valid])),maximum=float(np.max([r[k]for r in valid])))for k in ('direction_error_deg','point_difference_m','ring_state_difference','ring_phase_difference_deg','regenerated_goal_code_max_difference','regenerated_goal_code_rmse','decoded_pickup_error_m')}
    result=dict(stage='Measured observed-rate arrival-history control; no new behavioral intervention',
        original_actor='gc100_lstm_s1_1000M',episodes=len(episodes),first_pickups=len(records),undiscovered_episodes=len(episodes)-len(records),
        included_all_first_pickups=True,history_available=len(valid),history_missing=len(records)-len(valid),
        metrics=metrics,rows=results,episode_rows=episodes,collection=collection,
        inversion=dict(ordinary=inverse_ordinary.diagnostics(),predictive=inverse_ahead.diagnostics()),
        interpretation='History decoder uses only observed ordinary map rates and known movement threshold. True positions/directions/goals are diagnostic only. Any missing history is reported, not replaced by the true direction. This demonstrates available information, not that the original Grid LSTM learned this inverse/history computation.')
    np.savez_compressed(args.out/'observed_rates.npz',**data);save_json(args.out/'results.json',result)
    lines=['# Grid history can provide the same arrival cue','',
        'We ran the original frozen Grid actor in the same circular-goal square arena. At every first pickup, we saved up to30recent ordinary-grid observations and the actual pickup snapshot. A known-map inverse recovered their positions. The last displacement exceeding the original movement threshold supplied arrival direction; smaller movements retained that direction. No true direction was supplied to this reconstruction.','',
        f'Included **{len(records)}firstpickups from{len(episodes)}episodes**. History supplied a direction for**{len(valid)}/{len(records)}**; missing histories were not filled using ground truth.','',
        f'The resulting ahead point differed from the independently decoded PGC ahead point by at most**{metrics["point_difference_m"]["maximum"]:.3g}m**. Ring phase differed by at most**{metrics["ring_phase_difference_deg"]["maximum"]:.3g}°**. Regenerated512-cell goal-code activity differed by at most**{metrics["regenerated_goal_code_max_difference"]["maximum"]:.3g}**.','',
        '**Meaning:** the useful arrival cue can also be reconstructed from ordinary spatial history. Predictive cells supply it directly in the saved goal message. This control shows information availability; it does not claim that the original Grid LSTM actually performs the explicit inverse and displacement calculation.','',
        'The current ArenaEnv implementation updates movement direction from the whole decision-step displacement, despite action_repeat=4. This is why the observed-position history matches its direction convention. The last-moving threshold is0.05m/s×4/60s. The previous30observations are an explicit finite history window; every firstpickup and any unavailable direction are retained in results.json.','',
        'No new behavioral cohort was needed to compare identical ring outputs. The actor was unchanged, and true positions, directions, and goals were used only for diagnostics.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pickups=len(records),available=len(valid),metrics=metrics),indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/original_rl_goal_ring/history_control');ap.add_argument('--device',default='mps')
    ap.add_argument('--episodes',type=int,default=64);ap.add_argument('--n-envs',type=int,default=32);ap.add_argument('--eval-seed',type=int,default=63001);ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    protocol=dict(actor='Original frozen Gridseed1',episodes=args.episodes,eval_seed=args.eval_seed,history_observations=30,
        cue='Only observed343ordinary map rates at previouspositions and pickup; last decoded displacement>0.05m/s×4/60s; no true direction substituted',
        comparison='Independently invert169PGC goalrates; compare reconstructed aheadpoint,2Dringphase and full512regeneratedgoalcode',
        inclusion='All actualfirstpickups; unavailablehistory retained asmissing ratherthanexcluded orfilled',
        decision_thresholds=dict(point_difference_m=1e-4,phase_difference_deg=.01,goal_code_max_difference=.001),
        source_sha256={p:sha(ROOT/p)for p in ['scripts/original_goal_ring_history_control.py','scripts/original_rl_goal_ring.py','scripts/rl_goal_memory.py','code/env.py','code/codes.py','code/nets.py']})
    p=args.out/'protocol.json'
    if not p.exists():save_json(p,protocol)
    else:assert json.loads(p.read_text())==protocol
    if args.freeze_only:return
    cfg,records,episodes,collection=collect(args);analyze(cfg,records,episodes,collection,args)

if __name__=='__main__':main()
