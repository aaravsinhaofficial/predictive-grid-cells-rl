"""Bounded, separately frozen reward-only blind-reorientation pilot.

Implementation and tests may be reviewed without launching. --freeze-only
creates the immutable source/asset manifest; training requires that manifest.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import CodeSource, build_policy, build_population
from rl_goal_memory import atomic_json, atomic_torch, state_digest
from rl_visual_memory import sha, require_time
import rl_blind_reorientation as task

CONDITIONS = ('gc100', 'pgc33')
DELAYS = (4, 15, 30, 60)
UPDATES_PER_STAGE, WARMUP = 489, 200
FRAMES_PER_UPDATE = 32 * 160 * 4
STAGE_FRAMES = UPDATES_PER_STAGE * FRAMES_PER_UPDATE
TOTAL_FRAMES = len(DELAYS) * STAGE_FRAMES
SOURCE_FILES = (
    'scripts/rl_blind_reorientation.py', 'scripts/rl_blind_reorientation_train.py',
    'tests/test_rl_blind_reorientation.py', 'scripts/rl_visual_memory.py',
    'scripts/rl_visual_memory_noise.py', 'scripts/rl_visual_memory_extend.py',
    'scripts/rl_goal_memory.py', 'scripts/lstm_code_swap.py',
    'code/common.py', 'code/env.py', 'code/codes.py', 'code/nets.py')


def protocol():
    return dict(version='blind_reorientation_reward_only_v1', conditions=list(CONDITIONS),
        model_seed=1, original_frames=1_000_000_000, parent='Original1B checkpoint, independent of prior25M/75M arms',
        objective='Original navigation reward only, all original network weights trainable; no angle/ring target, decoder loss or hand-written action controller during free navigation',
        unchanged='Square geometry, uniform goals, original maps/RGB renderer, goal radius/reward, six motor actions, native saved-goal code, CNN/LSTM256/actor/critic architecture',
        timing=dict(visible=15, independent_labelled_turn_blocks=2, each_turn_min=5, each_turn_max=10,
            forward_probe=4, endpoint_glimpse='First input of delay block, not an extra update',
            delay_action='Constant left, action0; previous action always records actually applied action',
            delay_maxima=list(DELAYS), delay_minimum_rule='ceil(stage_max/2)',
            free_blind_updates=30, maximum_event_updates=129,
            brakes='Zero vf/vs/vr at first blind turn, immediately before forward probe, and after probe before first delay action',
            interrupts='Actual reward interrupts any event, refreshes native saved goal and returns to visible waiting; done clears saved goal/known state',
            admission='After first reward, at most1 event per160-update BPTT segment and8/90s episode; require129 updates left in both segment and episode; ordinary native visible navigation while waiting; recurrent state persists across segments',
            schedule_rng='Private per-slot per-episode8-event banks, event-index paired across evaluation branches; two independent turn labels/durations and one delay quantile'),
        inputs=dict(blind_rgb='All zero through original CNN',
            blind_current='Ordinary position-only512-slot snapshot held from first blind turn, updated at actual endpoint of fourth forward action; one native endpoint pulse during first delay input',
            native_goal='Continuous native512-slot snapshot of actual last goal pickup, including predictive component',
            preserved='Actual previous action and clipped reward; physical motor noise, walls and goal respawn',
            no_extra_input=True),
        learning=dict(n_envs=32, bptt=160, original_bptt=100, all_summed_losses_scale=100/160,
            credit_window_reason='Complete worst-case129-update event stays within one differentiable recurrent unroll',
            forced_actions='Mask policy-gradient AND entropy on forced turns/probe/delay; train critic normally; retain recurrent gradients through forced steps',
            warmup_updates=WARMUP, warmup_learning_rate=0., warmup_exposure_frames=WARMUP*FRAMES_PER_UPDATE,
            optimizer='New original RMSProp,200zero-LR accumulator warmup, then original learning rate; no resumed optimizer',
            updates_per_stage=UPDATES_PER_STAGE, learning_frames_per_stage=STAGE_FRAMES,
            total_learning_frames=TOTAL_FRAMES, action_repeat=4,
            noise=dict(sigma_h=task.SIGMA, sigma_c=task.SIGMA, seed=task.NOISE_SEED,
                timing='Independent additive h/c Gaussian after LSTM before actor/critic only on actual black-input updates',
                replay='Store and replay identical eps through BPTT; separate bootstrap draw; noise RNG saved',
                tuning='Fixed0.02 absolute units, no outcome-based tuning')),
        seeds=dict(training=task.TRAIN_SEED, environment=task.TRAIN_SEED*1000+1,
                   clock=task.CLOCK_SEED, evaluation=task.EVAL_SEED),
        evaluation=dict(episodes=64, n_envs=32, max_delay=60, clean=True,
            endpoints='Before warmup and after every fixed stage; retain all four stages and final',
            variants=list(task.VARIANTS),
            continuous_vision='Restore RGB only; preserve forced actions, held ordinary snapshots and pulse/reset intervention',
            cue_control='Replace only169 predictive current slots by ordinary values, or rotate actual endpoint travel direction/velocity by+60/-60degrees; saved goal unchanged; Grid controls exact identities',
            resets='Reset after processing endpoint glimpse, before next input; or before every scheduled blind input. Matching resets in continuously visible controls',
            metrics='Whole-episode score and per-event free-window hits/progress/path/updates; forced hits/walls; admission/wait/cap/interruption/full-window coverage counts; actual probe speed/displacement/travel-versus-heading/wall contacts and all three pre/post brakes, with no outcome-based exclusion',
            pairing='Shared slot/episode/event plans and physics/action draws; actual outcome-driven later window onset may differ',
            reset_scope='h/c carryover only; actual previous actions/rewards, external saved goal and spatial snapshot, and physical history remain'),
        interpretation='Prospective same-agent reward-trained memory mechanism under a newly necessary blind reorientation task; not retrospective proof of a ring in original1B agents. Positive behavioral memory and PGC effects motivate fixed full-state geometry/causal stabilization tests.',
        selection='Prespecified fixed frame budgets, no stage selection or adaptive stopping for positive results',
        budget=dict(workers=2,accelerator='T4',worker_guard='DELETE after2hours',
            process_max_seconds=6600,reservation_usd=5.625,cumulative_user_budget_usd=500.,
            launch='Separate explicit root authorization after source tests, independent review and frozen protocol'),
        source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},
        original_asset_sha256={f'models/{c}/{c}_lstm_s1_1000M{suffix}':
            sha(ROOT/f'models/{c}/{c}_lstm_s1_1000M{suffix}') for c in CONDITIONS for suffix in ('.json','.pt')})


def train(net, cfg, args, design, deadline):
    assert (cfg['n_envs'], cfg['rollout'], cfg['action_repeat']) == (32, 160, 4)
    assert cfg['extra_input'] == 'none' and cfg['code_noise'] == 0.
    torch.manual_seed(task.TRAIN_SEED); np.random.seed(task.TRAIN_SEED)
    runtime = task.fresh_runtime(net, cfg, args.device)
    source = CodeSource(cfg, build_population(cfg, args.device), args.device)
    opt = torch.optim.RMSprop(net.parameters(), lr=0., alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    initial_hash = state_digest(net); frames = 0; stages = []; recent = []
    started = time.monotonic(); net.train()

    def save(name, updates, completed=False):
        atomic_torch(args.out/name, dict(policy=net.state_dict(), optimizer=opt.state_dict(), config=cfg,
            provenance=design, completed=completed, original_frames=1_000_000_000,
            extra_frames=frames, total_updates=updates, warmup_updates=WARMUP,
            runtime_snapshot=task.capture_runtime(runtime,args.device),
            stage_evaluations_completed=len(stages), recent_scores=recent))

    save('incoming.pt', 0)
    with (args.out/'metrics.jsonl').open('w') as log:
        for index in range(WARMUP+len(DELAYS)*UPDATES_PER_STAGE):
            require_time(deadline,60)
            warm = index < WARMUP
            stage = 0 if warm else (index-WARMUP)//UPDATES_PER_STAGE
            metrics, episodes = task.update(net,cfg,opt,source,runtime,DELAYS[stage])
            if not warm:
                frames += FRAMES_PER_UPDATE; recent.extend(episodes); recent = recent[-256:]
            if warm and index+1 == WARMUP:
                assert state_digest(net) == initial_hash
                for group in opt.param_groups:
                    group['lr'] = cfg['lr']
                save('after_warmup.pt',index+1)
            if (index+1)%20 == 0:
                item=dict(update=index+1,warmup=warm,stage=stage+1,max_delay=DELAYS[stage],
                    learning_frames=frames,total_exposure_frames=(index+1)*FRAMES_PER_UPDATE,
                    seconds=time.monotonic()-started,mean_recent_score=float(np.mean(recent)) if recent else None,
                    **metrics,schedule=runtime['clock'].summary(),probe_audit=runtime['inputs'].summary())
                log.write(json.dumps(item)+'\n');log.flush();print(json.dumps(item),flush=True)
            if not warm and (index-WARMUP+1)%100 == 0:
                save('latest.pt',index+1)
            if not warm and (index-WARMUP+1)%UPDATES_PER_STAGE == 0:
                save(f'stage{stage+1}_delay{DELAYS[stage]}.pt',index+1,completed=stage==len(DELAYS)-1)
                before = state_digest(net)
                evaluation = task.evaluate(net,cfg,device=args.device,deadline=deadline)
                assert state_digest(net) == before
                atomic_json(args.out/f'stage{stage+1}_evaluation.json',evaluation)
                stages.append(dict(stage=stage+1,max_training_delay=DELAYS[stage],learning_frames=frames,
                                   evaluation=evaluation))
                atomic_json(args.out/'progress.json',dict(stages=stages))
                net.train()
    assert frames == TOTAL_FRAMES
    save('final.pt',WARMUP+len(DELAYS)*UPDATES_PER_STAGE,completed=True)
    atomic_json(args.out/'final_evaluation.json',stages[-1]['evaluation'])
    return dict(completed=True,learning_frames=frames,warmup_exposure_frames=WARMUP*FRAMES_PER_UPDATE,
        initial_policy_hash=initial_hash,final_policy_hash=state_digest(net),warmup_policy_unchanged=True,
        seconds=time.monotonic()-started,stages=stages,schedule=runtime['clock'].summary(),
        noise_replay_max_error=runtime['noise'].max_replay_error,noise_replayed_updates=runtime['noise'].replays,
        probe_audit=runtime['inputs'].summary())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition',choices=CONDITIONS);ap.add_argument('--out',type=Path)
    ap.add_argument('--device',default='cuda');ap.add_argument('--max-seconds',type=int,default=6600)
    ap.add_argument('--protocol',type=Path,default=ROOT/'docs/research/rl_blind_reorientation_protocol.json')
    ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
    design=protocol()
    if args.freeze_only:
        if args.protocol.exists():raise FileExistsError('Do not overwrite frozen reorientation protocol')
        design['frozen_at']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(args.protocol,design)
        print(sha(args.protocol));return
    if args.condition is None or args.out is None:ap.error('--condition and --out required')
    frozen=json.loads(args.protocol.read_text());frozen.pop('frozen_at')
    assert frozen==design,'Frozen reorientation sources/assets/protocol changed'
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'protocol.json').exists():raise FileExistsError('Reorientation output already exists')
    atomic_json(args.out/'protocol.json',design)
    cfgpath=ROOT/f'models/{args.condition}/{args.condition}_lstm_s1_1000M.json'
    original=json.loads(cfgpath.read_text());cfg=dict(original,rollout=160)
    checkpoint=torch.load(cfgpath.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert checkpoint['frames']==1_000_000_000
    net=build_policy(cfg,args.device);net.load_state_dict(checkpoint['policy'],strict=True)
    atomic_json(args.out/'original_config.json',original);atomic_json(args.out/'training_config.json',cfg)
    deadline=time.monotonic()+args.max_seconds
    try:
        initial=state_digest(net)
        baseline=task.evaluate(net,cfg,device=args.device,deadline=deadline)
        assert state_digest(net)==initial
        atomic_json(args.out/'baseline.json',baseline)
        result=train(net,cfg,args,design,deadline);atomic_json(args.out/'training.json',result)
        for name,digest in design['original_asset_sha256'].items():assert sha(ROOT/name)==digest
        atomic_json(args.out/'summary.json',dict(completed=True,condition=args.condition,training=result,
            baseline=baseline,final_checkpoint_sha256=sha(args.out/'final.pt')))
    except Exception as error:
        atomic_json(args.out/'incomplete.json',dict(error_type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':main()
