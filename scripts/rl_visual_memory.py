"""Reward-only visual-memory continuation of the original square-arena LSTM.

Only observation availability changes: after the first reward, native RGB is
intermittently replaced by black frames. No ring, angle target, goal-coordinate
input, or external controller is added. Goal/current codes remain native during
training. Original checkpoint files and source modules are never changed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import CodeSource, build_env, build_policy, build_population, extra_from_obs
from lstm_code_swap import MatchedArena, verify_codes
from rl_goal_memory import atomic_json, atomic_torch, episode_summary, replay, reset_state, state_digest

CONDITIONS = ('gc100', 'pgc33')
VARIANTS = ('blind', 'blind_reset', 'seen', 'seen_reset', 'blind_current_grid', 'seen_current_grid')
GAPS = (4, 8, 15, 30, 60)
SOURCE_FILES = ('scripts/rl_visual_memory.py', 'scripts/rl_goal_memory.py',
                'scripts/lstm_code_swap.py', 'code/__init__.py', 'code/common.py',
                'code/env.py', 'code/codes.py', 'code/nets.py', 'code/train.py',
                'tests/test_rl_visual_memory.py', 'tests/test_rl_goal_memory.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VisualClock:
    """Observed reward starts15 visible updates, then random blind gaps.

    Duration RNG is independent of environment/action RNG. A base-slot draw on
    every decision is repeated across evaluation branches, even if their later
    reward/phase histories diverge. Reward renews the visible period; episode end
    clears it. Curriculum changes affect newly sampled gaps only.
    """
    def __init__(self, n, device, seed, groups=1, visible_steps=15):
        self.n, self.groups, self.visible_steps = n, groups, visible_steps
        self.known = torch.zeros(n*groups, dtype=torch.bool, device=device)
        self.blind = torch.zeros_like(self.known)
        self.remaining = torch.full((n*groups,), visible_steps, dtype=torch.long, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def advance(self, hit, done, max_gap):
        if max_gap < 1:
            raise ValueError('Positive maximum blackout length required')
        low = (int(max_gap)+1)//2
        durations = torch.randint(low, int(max_gap)+1, (self.n,), generator=self.generator,
                                  device=self.remaining.device).repeat(self.groups)
        active = self.known & ~hit & ~done
        self.remaining = self.remaining-active.long()
        boundary = active & (self.remaining == 0)
        entering = boundary & ~self.blind
        leaving = boundary & self.blind
        self.blind = (self.blind | entering) & ~leaving
        self.remaining = torch.where(entering, durations, self.remaining)
        self.remaining = torch.where(leaving, torch.full_like(self.remaining, self.visible_steps), self.remaining)
        self.known = (self.known | hit) & ~done
        self.blind &= ~(hit | done)
        self.remaining = torch.where(hit | done, torch.full_like(self.remaining, self.visible_steps), self.remaining)
        return entering

    def mask(self):
        return self.known & self.blind


class NativeGoal:
    def __init__(self, n, n_code, device):
        self.saved = torch.zeros(n, n_code, device=device)

    def advance(self, hit, done, hit_code=None):
        if bool(hit.any()):
            if hit_code is None:
                raise ValueError('Actual-hit code required')
            self.saved = torch.where(hit[:, None], hit_code, self.saved)
        self.saved = self.saved*(~done).unsqueeze(1)


def masked_rgb(rgb, clock, blind_group=None):
    mask = clock.mask()
    if blind_group is not None:
        mask = mask & blind_group
    return rgb*(~mask)[:, None, None, None]


def require_time(deadline, reserve=30):
    if time.monotonic()+reserve >= deadline:
        raise TimeoutError('Fixed pilot time limit reached; partial files retained')


@torch.inference_mode()
def evaluate(net, cfg, *, device, seed, episodes=64, n_envs=32, max_gap=60,
             deadline=float('inf')):
    if episodes % n_envs:
        raise ValueError('Complete evaluation episode batches required')
    net.eval()
    n, groups = n_envs, len(VARIANTS)
    env = MatchedArena(cfg, n, groups, seed, device)
    obs = env.reset_all()
    for key in ('pos', 'hd', 'dir', 'rgb'):
        assert all(torch.equal(obs[key][:n], obs[key][j*n:(j+1)*n]) for j in range(1, groups))
    pop = build_population(cfg, device)
    source = CodeSource(cfg, pop, device)
    clock = VisualClock(n, device, seed+2, groups=groups)
    goal = NativeGoal(n*groups, cfg['n_cells'], device)
    state = net.zero_state(n*groups, device)
    pa = torch.full((n*groups,), -1, dtype=torch.long, device=device)
    pr = torch.zeros(n*groups, device=device)
    blind_group = torch.tensor([v.startswith('blind') for v in VARIANTS], device=device).repeat_interleave(n)
    reset_group = torch.tensor([v.endswith('_reset') for v in VARIANTS], device=device).repeat_interleave(n)
    grid_group = torch.tensor([v.endswith('_current_grid') for v in VARIANTS], device=device).repeat_interleave(n)
    action_rng = torch.Generator(device=device).manual_seed(seed+1)
    rows = {v: [] for v in VARIANTS}
    reset_count = torch.zeros(groups, dtype=torch.long, device=device)
    blind_count = torch.zeros(groups, dtype=torch.long, device=device)
    known_count = torch.zeros(groups, dtype=torch.long, device=device)
    squared_error = torch.zeros(groups, device=device)
    movement_count = torch.zeros(groups, dtype=torch.long, device=device)
    for step in range(cfg['episode_len']*(episodes//n)):
        if step % 100 == 0:
            require_time(deadline)
        scheduled = clock.mask()
        rgb = masked_rgb(obs['rgb'], clock, blind_group)
        current = source(obs['pos'], obs['dir'], obs['vel'])
        ordinary = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0.)
        swap = scheduled & grid_group
        current = torch.where(swap[:, None], ordinary, current)
        logits, _, state = net.step(rgb, current, goal.saved, pa, pr, state, extra_from_obs(cfg, obs))
        uniform = torch.rand(n, generator=action_rng, device=device).repeat(groups)
        action = (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        blind_count += (scheduled & blind_group).view(groups,n).sum(1)
        known_count += clock.known.view(groups,n).sum(1)
        # Diagnostic only: heading and last movement direction are distinct variables.
        travel_angle = torch.atan2(obs['dir'][:,1], obs['dir'][:,0])
        difference = torch.atan2(torch.sin(obs['hd']-travel_angle), torch.cos(obs['hd']-travel_angle))
        moving = obs['vel'].norm(dim=1) > env.v_min
        squared_error += (difference.square()*moving).view(groups,n).sum(1)
        movement_count += moving.view(groups,n).sum(1)
        obs, raw, done, info = env.step(action)
        hit = info['hit']
        hit_code = source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']) if bool(hit.any()) else None
        goal.advance(hit, done, hit_code)
        entering = clock.advance(hit, done, max_gap)
        resets = entering & reset_group
        reset_count += resets.view(groups,n).sum(1)
        state = reset_state(state, done | resets)
        reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
        pa = torch.where(done, torch.full_like(action,-1), action)
        pr = reward*(~done)
        stats = info['ep_stats']
        if stats is not None:
            for k,index in enumerate(stats['env'].cpu().tolist()):
                group,slot=divmod(index,n)
                rows[VARIANTS[group]].append(dict(slot=slot,episode=step//cfg['episode_len'],
                    score=float(stats['ret'][k]),n_goals=int(stats['n_goals'][k]),
                    first_goal_step=int(stats['first_goal_t'][k]),path_eff=float(stats['path_eff'][k]),
                    path_eff_n=int(stats['path_eff_n'][k])))
    reference=[r['first_goal_step'] for r in rows['blind']]
    assert len(reference)==episodes
    assert all([r['first_goal_step'] for r in rr]==reference for rr in rows.values())
    if not bool(pop.is_pgc.any()):
        assert rows['blind']==rows['blind_current_grid']
        assert rows['seen']==rows['seen_current_grid']
    metrics={v:episode_summary(rr,cfg['episode_len']) for v,rr in rows.items()}
    score={v:d['mean_score'] for v,d in metrics.items()}
    return dict(seed=seed,episodes_per_variant=episodes,max_blind_gap=max_gap,
        min_blind_gap=(max_gap+1)//2,visible_updates=15,variants=metrics,
        all_first_discoveries_identical=True,
        grid_replacement_is_exact_noop=not bool(pop.is_pgc.any()),
        state_reset_counts=dict(zip(VARIANTS,reset_count.cpu().tolist())),
        blind_fraction_of_post_discovery_updates=dict(zip(VARIANTS,(blind_count/known_count.clamp_min(1)).cpu().tolist())),
        body_heading_vs_travel_direction_rms_deg=dict(zip(VARIANTS,torch.rad2deg((squared_error/movement_count.clamp_min(1)).sqrt()).cpu().tolist())),
        descriptive_contrasts=dict(blind_reset_cost=score['blind']-score['blind_reset'],
            seen_reset_cost=score['seen']-score['seen_reset'],
            additional_reset_cost_in_darkness=(score['blind']-score['blind_reset'])-(score['seen']-score['seen_reset']),
            blind_current_prediction_benefit=score['blind']-score['blind_current_grid'],
            seen_current_prediction_benefit=score['seen']-score['seen_current_grid']))


def collect_rollout(net, env, source, cfg, obs, state, goal, clock, pa, pr, length, max_gap):
    state0=tuple(x.detach() for x in state)
    roll={k:[] for k in ('rgb','code','goal','pa','pr','extra','a','r','d','actor_logits')}
    episodes=[]
    with torch.no_grad():
        for _ in range(length):
            rgb=masked_rgb(obs['rgb'],clock)
            code=source(obs['pos'],obs['dir'],obs['vel'])
            extra=extra_from_obs(cfg,obs)
            logits,_,state=net.step(rgb,code,goal.saved,pa,pr,state,extra)
            action=torch.distributions.Categorical(logits=logits).sample()
            for key,value in (('rgb',rgb),('code',code),('goal',goal.saved),('pa',pa),('pr',pr),
                              ('extra',extra),('a',action),('actor_logits',logits)):
                roll[key].append(value)
            obs,raw,done,info=env.step(action)
            reward=raw.clamp(-cfg['reward_clip'],cfg['reward_clip']) if cfg['reward_clip']>0 else raw
            roll['r'].append(reward);roll['d'].append(done.float())
            hit=info['hit']
            hit_code=source.goal(info['hit_pos'],info['hit_dir'],info['hit_vel']) if bool(hit.any()) else None
            goal.advance(hit,done,hit_code);clock.advance(hit,done,max_gap)
            state=reset_state(state,done)
            pa=torch.where(done,torch.full_like(action,-1),action);pr=reward*(~done)
            if info['ep_stats'] is not None:
                episodes.extend(info['ep_stats']['ret'].cpu().tolist())
        bootstrap_rgb=masked_rgb(obs['rgb'],clock)
        _,bootstrap,_=net.step(bootstrap_rgb,source(obs['pos'],obs['dir'],obs['vel']),goal.saved,
                              pa,pr,tuple(x.detach() for x in state),extra_from_obs(cfg,obs))
    return roll,state0,bootstrap,bootstrap_rgb,obs,state,pa,pr,episodes


def protocol():
    assets={}
    for condition in CONDITIONS:
        for ext in ('.json','.pt'):
            name=f'models/{condition}/{condition}_lstm_s1_1000M{ext}'
            assets[name]=sha(ROOT/name)
    return dict(version='visual_memory_pilot_v1',conditions=list(CONDITIONS),model_seed=1,
        training_seed=96201,evaluation_seed=98201,episodes=64,n_envs_eval=32,
        original_frames=1_000_000_000,requested_learning_frames_per_stage=5_000_000,
        actual_learning_frames_per_stage=5_004_800,total_learning_frames_per_model=25_024_000,
        warmup_updates=200,warmup_extra_exposure_frames=2_560_000,
        gaps=list(GAPS),minimum_gap_rule='ceil(max_gap/2)',visible_updates=15,
        variants=list(VARIANTS),mask='RGB exactly zero only after first observed reward; native image otherwise',
        schedule='Every actual reward renews15visibleupdates; episodeendclearsclock; currentgapfinishesafterstagechange',
        unchanged='Originaluniformgoalprior,squarephysics,goalradius,rewards,respawns,512nativecurrent+savedgoalcodes,sixactions,LSTM256,CNN,A2Chyperparameters100stepBPTT',
        primary_pilot_criteria='Reportallfixedstages; usefulmemoryrequiresmaintainedblindnavigationandpositiveblindresetcostbeyondseenresetcost. Furtherfull-stategeometryandcausalrepairtestsrequiredforringmechanism. Noadaptivepositiveselection.',
        descriptive_health_references=dict(blind_over_seen_return_rate=.70,additional_reset_cost_score=5.),
        interpretation='Headingandtraveldirectionareseparate;PGCencodeslastmovementdirectionwhichisstaleduringpureturns. Thisisnewvisualavailabilitytrainingdistribution. OriginalGrid/PGC1Btraininghistoriesdiffer.',
        budget='Two2hourboundedT4workersmaximum; originalcumulative500USD; prior44.1913USD; totalpilotreservation5.625USD',
        source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},original_asset_sha256=assets)


def train(net,cfg,args,design,deadline):
    torch.manual_seed(96201);np.random.seed(96201)
    net.train()
    env=build_env(cfg,cfg['n_envs'],seed=96201*1000+1,device=args.device,stagger=True)
    pop=build_population(cfg,args.device);source=CodeSource(cfg,pop,args.device)
    n,length=cfg['n_envs'],cfg['rollout']
    assert (n,length,cfg['action_repeat'])==(32,100,4)
    opt=torch.optim.RMSprop(net.parameters(),lr=0.,alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    obs,state=env.reset_all(),net.zero_state(n,args.device)
    goal=NativeGoal(n,cfg['n_cells'],args.device);clock=VisualClock(n,args.device,96203)
    pa=torch.full((n,),-1,dtype=torch.long,device=args.device);pr=torch.zeros(n,device=args.device)
    initial_hash=state_digest(net);learning_frames=0;warmup=200
    per_stage=math.ceil(5_000_000/(n*length*cfg['action_repeat']))
    started=time.monotonic();recent=[];stage_results=[]
    def save(name,completed=False):
        atomic_torch(args.out/name,dict(policy=net.state_dict(),optimizer=opt.state_dict(),
            original_frames=1_000_000_000,extra_frames=learning_frames,completed=completed,config=cfg,
            provenance=design,cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(args.device) if str(args.device).startswith('cuda') else None,
            warmup_updates=warmup,stage_evaluations_completed=len(stage_results)))
    with (args.out/'metrics.jsonl').open('w') as log:
        for update in range(warmup+per_stage*len(GAPS)):
            require_time(deadline,60)
            warm=update<warmup
            stage=0 if warm else (update-warmup)//per_stage
            gap=GAPS[stage]
            roll,state0,bootstrap,_,obs,state,pa,pr,returns_episode=collect_rollout(
                net,env,source,cfg,obs,state,goal,clock,pa,pr,length,gap)
            future=bootstrap;targets=[]
            for t in reversed(range(length)):
                future=roll['r'][t]+cfg['gamma']*(1-roll['d'][t])*future
                targets.append(future)
            targets=torch.stack(list(reversed(targets)))
            logits,values=replay(net,roll,state0)
            dist=torch.distributions.Categorical(logits=logits)
            policy_loss=-(dist.log_prob(torch.stack(roll['a']))*(targets-values).detach()).sum(0).mean()
            value_loss=.5*(targets-values).square().sum(0).mean()
            entropy=dist.entropy().sum(0).mean()
            loss=policy_loss+cfg['value_coef']*value_loss-cfg['entropy']*entropy
            if not torch.isfinite(loss):raise FloatingPointError('NonfiniteA2Closs')
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['grad_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinitegradient')
            opt.step();state=tuple(x.detach() for x in state)
            if warm and update+1==warmup:
                assert state_digest(net)==initial_hash
                save('after_warmup.pt')
                for group in opt.param_groups:group['lr']=cfg['lr']
            if not warm:
                learning_frames+=n*length*cfg['action_repeat'];recent.extend(returns_episode);recent=recent[-256:]
            if (update+1)%20==0:
                item=dict(total_updates=update+1,warmup=warm,curriculum_stage=stage,max_blind_gap=gap,
                    extra_frames=learning_frames,all_exposure_frames=(update+1)*n*length*cfg['action_repeat'],
                    seconds=time.monotonic()-started,loss=float(loss),entropy=float(entropy/length),
                    grad_norm=float(norm),mean_recent_score=float(np.mean(recent)) if recent else None)
                log.write(json.dumps(item)+'\n');log.flush();print(json.dumps(item),flush=True)
            if not warm and (update-warmup+1)%100==0:save('latest.pt')
            if not warm and (update-warmup+1)%per_stage==0:
                save(f'stage{stage+1}_gap{gap}.pt',completed=stage==len(GAPS)-1)
                before=state_digest(net)
                evaluation=evaluate(net,cfg,device=args.device,seed=98201,episodes=64,n_envs=32,max_gap=60,deadline=deadline)
                assert state_digest(net)==before
                atomic_json(args.out/f'stage{stage+1}_evaluation.json',evaluation)
                stage_results.append(dict(stage=stage+1,max_training_gap=gap,extra_frames=learning_frames,
                    evaluation=evaluation))
                atomic_json(args.out/'progress.json',dict(stages=stage_results))
                net.train()
    save('final.pt',completed=True)
    atomic_json(args.out/'final_evaluation.json',stage_results[-1]['evaluation'])
    return dict(completed=True,extra_frames=learning_frames,warmup_exposure_frames=2_560_000,
        warmup_policy_hash_unchanged=True,initial_policy_hash=initial_hash,final_policy_hash=state_digest(net),
        seconds=time.monotonic()-started,stages=stage_results)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition',choices=CONDITIONS)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--max-seconds',type=int,default=6600)
    ap.add_argument('--protocol',type=Path,default=ROOT/'docs/research/rl_visual_memory_pilot_protocol.json')
    ap.add_argument('--freeze-only',action='store_true')
    args=ap.parse_args();torch.set_num_threads(2)
    design=protocol()
    if args.freeze_only:
        if args.protocol.exists():raise FileExistsError('Do notoverwritefrozenprotocol')
        design['frozen_at']=dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(args.protocol,design);print(sha(args.protocol));return
    if not args.condition or args.out is None:ap.error('--conditionand--outrequired')
    frozen=json.loads(args.protocol.read_text());frozen.pop('frozen_at')
    if frozen!=design:raise AssertionError('Frozensources/assets/protocolchanged')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'protocol.json').exists():raise FileExistsError('Pilotoutputalreadyexists')
    design['runtime']=dict(torch_version=str(torch.__version__),device=args.device,condition=args.condition,
                           local_protocol_sha256=sha(args.protocol))
    atomic_json(args.out/'protocol.json',design)
    deadline=time.monotonic()+args.max_seconds
    cfgpath=ROOT/f'models/{args.condition}/{args.condition}_lstm_s1_1000M.json'
    cfg=json.loads(cfgpath.read_text());checkpoint=torch.load(cfgpath.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert checkpoint['frames']==1_000_000_000
    net=build_policy(cfg,args.device);net.load_state_dict(checkpoint['policy'],strict=True)
    atomic_json(args.out/'original_config.json',cfg)
    try:
        assert verify_codes(1)['max_absolute_code_difference']==0
        initial=state_digest(net)
        baseline=evaluate(net,cfg,device=args.device,seed=98201,episodes=64,n_envs=32,max_gap=60,deadline=deadline)
        assert state_digest(net)==initial
        atomic_json(args.out/'baseline.json',baseline)
        training=train(net,cfg,args,design,deadline)
        atomic_json(args.out/'training.json',training)
        for name,digest in design['original_asset_sha256'].items():assert sha(ROOT/name)==digest
        atomic_json(args.out/'summary.json',dict(completed=True,condition=args.condition,
            baseline=baseline,training=training,final_checkpoint_sha256=sha(args.out/'final.pt')))
    except Exception as error:
        atomic_json(args.out/'incomplete.json',dict(error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':main()
