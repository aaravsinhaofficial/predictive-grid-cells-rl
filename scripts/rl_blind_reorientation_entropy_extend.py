"""Draft matched entropy continuation of the completed reorientation pilot.

No launch or freeze is implied by this file. --write-draft records a reviewable
proposal; production requires a separately frozen manifest and both final parents.
The original task's collect/update/evaluate/capture/restore functions are reused.
"""
from __future__ import annotations
import argparse
import copy
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
from rl_visual_memory_extend import tree_equal
import rl_blind_reorientation as task
import rl_blind_reorientation_train as parent_training

ARMS = {'original': 8e-5, 'explore': 0.01}
WORKERS = {'gc100': 'ring-20261105-reorient-grid-01',
           'pgc33': 'ring-20261105-reorient-pgc-02'}
PARENT_FRAMES = 40_058_880
PARENT_UPDATES = 200 + 4*489
BLOCKS, BLOCK_UPDATES = 5, 489
FRAMES_PER_UPDATE = 32*160*4
BLOCK_FRAMES = BLOCK_UPDATES*FRAMES_PER_UPDATE
EXTENSION_FRAMES = BLOCKS*BLOCK_FRAMES
TOTAL_EXTRA_FRAMES = PARENT_FRAMES + EXTENSION_FRAMES
PARENT_PROTOCOL = ROOT/'docs/research/rl_blind_reorientation_protocol.json'
PROTOCOL = ROOT/'docs/research/rl_blind_reorientation_entropy_protocol.json'
DRAFT = ROOT/'docs/research/rl_blind_reorientation_entropy_draft.json'
SOURCE_FILES = (*parent_training.SOURCE_FILES,
    'scripts/rl_blind_reorientation_entropy_extend.py',
    'tests/test_rl_blind_reorientation_entropy_extend.py')


def parent_folder(condition):
    return ROOT/'results/ring_cloud_artifacts'/WORKERS[condition]


def parent_design():
    result = json.loads(PARENT_PROTOCOL.read_text()); result.pop('frozen_at')
    return result


def validate_parent(parent, condition):
    assert parent['completed'] is True
    assert parent['original_frames'] == 1_000_000_000
    assert parent['extra_frames'] == PARENT_FRAMES
    assert parent['total_updates'] == PARENT_UPDATES
    assert parent['warmup_updates'] == 200
    assert parent['stage_evaluations_completed'] == 4
    assert parent['provenance'] == parent_design()
    original = json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    assert parent['config'] == dict(original, rollout=160)
    assert parent['config']['entropy'] == ARMS['original']
    assert parent['config']['cond'] == condition and parent['config']['seed'] == 1
    assert {float(s['step']) for s in parent['optimizer']['state'].values()} == {float(PARENT_UPDATES)}
    assert parent['runtime_snapshot']['noise_replays'] == PARENT_UPDATES
    assert parent['runtime_snapshot']['noise_max_replay_error'] < 3e-4
    assert not bool((parent['runtime_snapshot']['clock']['phase'] != task.READY).any())


def protocol(require_parents=False):
    original = parent_design()
    # No frozen source, original model or original protocol may silently drift.
    assert parent_training.protocol() == original
    parents = {}
    for condition in WORKERS:
        folder = parent_folder(condition)
        assets = {name: (sha(folder/name) if (folder/name).exists() else None)
                  for name in ('final.pt', 'final_evaluation.json')}
        if require_parents:
            assert all(assets.values()), f'Completed parent assets pending: {condition}'
            validate_parent(torch.load(folder/'final.pt', map_location='cpu', weights_only=True), condition)
        parents[condition] = dict(folder=str(folder.relative_to(ROOT)), sha256=assets)
    return dict(version='blind_reorientation_matched_entropy_continuation_v1',
        status='DRAFT until separately reviewed and frozen; no launch authorization implied',
        conditions=list(WORKERS), arms=ARMS, workers=4, original_training_seed=1,
        question='Does stronger entropy regularization improve exploration and free blind rewards in the same task?',
        factor='Only config.entropy: original8e-5 versus0.01. It acts on all policy-controlled updates (visible/waiting and free), with the existing forced-action entropy/policy masks unchanged.',
        unchanged='Original task, input timing, goal/current codes, reward, architecture, all trainable weights, learning rate/RMSProp,160-update BPTT,100/160 loss scaling, sigma.02 h/c noise and exact stored-noise replay; no ring/angle target or reward reshaping',
        parent_protocol_sha256=sha(PARENT_PROTOCOL), parent_assets=parents,
        resume='Exact completed40,058,880-learning-frame final.pt only, preserving policy, full RMSProp, all captured simulator/observation/h/c/goal/action/reward/clock/plan/input-audit/private RNG tensors and recent scores. No new warmup, seed substitution, native-state reset or curriculum restart.',
        parent_global_rng_scope='Parent snapshots contain all three task-used private RNG streams and deterministic per-slot clock plans. They do not save unused global torch/NumPy RNG state; no substitute is claimed for an absent parent field.',
        learning=dict(blocks=BLOCKS, updates_per_block=BLOCK_UPDATES,
            n_envs=32, bptt=160, action_repeat=4, frames_per_block=BLOCK_FRAMES,
            extension_learning_frames=EXTENSION_FRAMES, parent_learning_frames=PARENT_FRAMES,
            total_learning_frames_beyond_1B=TOTAL_EXTRA_FRAMES,
            extension_warmup_updates=0, parent_warmup_updates=200,
            parent_warmup_exposure_frames=4_096_000,
            delay='Unchanged final-stage uniform integer30–60, constantLEFT delay, throughout all five blocks',
            pairing='Within each population both entropy arms start from identical policy/optimizer/runtime/private RNG states. Streams can diverge after outcomes differ; no claim of identical later trajectories.'),
        evaluation=dict(variants=list(task.VARIANTS), episodes=64, n_envs=32, seed=task.EVAL_SEED,
            clean=True, max_delay=60, endpoints='Parent baseline and every one of five fixed blocks; all retained; no checkpoint selection',
            baseline='Same CUDA backend; all12 complete raw episode, free-event and window rows plus schedules/probe counters must equal parent final reference exactly before training',
            invariance='Every evaluation must leave policy, optimizer and the training runtime/private RNG snapshot unchanged',
            reporting='Whole-episode score; free-window hits/progress per original episode including zero-event episodes; opportunity/phase/forced-reward counts; paired slot-cluster uncertainty, no trained-seed significance'),
        interpretation='Continued original-seed exploration pilot, not entropy-optimum estimation and not ring-target training. Existing40M endpoints and their fixed mechanistic assays remain separate and unchanged.',
        selection='Fixed five blocks per arm; all baselines/stages and incomplete attempts retained; no outcome-based stopping or stage selection',
        budget=dict(workers=4, accelerator='T4', worker_guard_seconds=7200,
            process_max_seconds=6600, planned_reservation_usd=11.25,
            original_user_total_authorization_usd=500.,
            launch='Separate root approval after tests/review and exact final parent hashes; draft never authorizes launch'),
        source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},
        original_asset_sha256=original['original_asset_sha256'])


def continuation_config(parent_config, arm):
    assert arm in ARMS and parent_config['entropy'] == ARMS['original']
    result = copy.deepcopy(parent_config); result['entropy'] = ARMS[arm]
    assert {k:v for k,v in result.items() if k != 'entropy'} == {k:v for k,v in parent_config.items() if k != 'entropy'}
    return result


def restore_training(net, cfg, parent, device):
    """Use the frozen task's full snapshot machinery; assert byte-exact roundtrip."""
    net.load_state_dict(parent['policy'], strict=True)
    assert tree_equal(net.state_dict(), parent['policy'])
    opt = torch.optim.RMSprop(net.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    opt.load_state_dict(parent['optimizer'])
    assert tree_equal(opt.state_dict(), parent['optimizer'])
    assert all(g['lr'] == cfg['lr'] for g in opt.param_groups)
    runtime = task.fresh_runtime(net, cfg, device)
    task.restore_runtime(runtime, parent['runtime_snapshot'], device)
    assert tree_equal(task.capture_runtime(runtime, device), parent['runtime_snapshot'])
    net.train()
    return opt, runtime


def assert_baseline_equal(actual, expected):
    assert tuple(actual['variants']) == task.VARIANTS == tuple(expected['variants'])
    for variant in task.VARIANTS:
        assert actual['variants'][variant]['rows'] == expected['variants'][variant]['rows'], variant
        assert actual['free_window'][variant]['rows'] == expected['free_window'][variant]['rows'], variant
        assert actual['window_rows'][variant] == expected['window_rows'][variant], variant
    assert tree_equal(actual, expected), 'Baseline metadata/counters differ from exact parent reference'


def checked_evaluate(net, cfg, opt, runtime, device, deadline):
    before_policy = state_digest(net)
    before_optimizer = copy.deepcopy(opt.state_dict())
    before_runtime = task.capture_runtime(runtime, device)
    result = task.evaluate(net, cfg, device=device, deadline=deadline)
    assert state_digest(net) == before_policy
    assert tree_equal(before_optimizer, opt.state_dict()), 'Evaluation changed optimizer'
    assert tree_equal(before_runtime, task.capture_runtime(runtime, device)), 'Evaluation changed training runtime/RNG'
    net.train()
    return result


def train(net, cfg, opt, runtime, parent, args, design, deadline):
    assert (cfg['n_envs'], cfg['rollout'], cfg['action_repeat']) == (32,160,4)
    source = CodeSource(cfg, build_population(cfg, args.device), args.device)
    frames = 0; updates = 0; stages = []; recent = list(parent['recent_scores'])
    started = time.monotonic(); net.train()

    def save(name, completed=False):
        atomic_torch(args.out/name, dict(policy=net.state_dict(), optimizer=opt.state_dict(), config=cfg,
            provenance=design, completed=completed, original_frames=1_000_000_000,
            parent_extra_frames=PARENT_FRAMES, extension_frames=frames,
            extra_frames=PARENT_FRAMES+frames, extension_updates=updates,
            total_updates=PARENT_UPDATES+updates, warmup_updates=200, extension_warmup_updates=0,
            runtime_snapshot=task.capture_runtime(runtime,args.device),
            stage_evaluations_completed=len(stages), recent_scores=recent))

    save('incoming.pt')
    with (args.out/'metrics.jsonl').open('w') as log:
        for index in range(BLOCKS*BLOCK_UPDATES):
            require_time(deadline,60)
            metrics, episodes = task.update(net,cfg,opt,source,runtime,60)
            frames += FRAMES_PER_UPDATE; updates += 1
            recent.extend(episodes); recent = recent[-256:]
            if updates%20 == 0:
                item = dict(extension_updates=updates,total_updates=PARENT_UPDATES+updates,
                    block=index//BLOCK_UPDATES+1, max_delay=60, entropy_coefficient=cfg['entropy'],
                    extension_frames=frames, extra_frames=PARENT_FRAMES+frames,
                    total_exposure_frames=4_096_000+PARENT_FRAMES+frames,
                    seconds=time.monotonic()-started,mean_recent_score=float(np.mean(recent)) if recent else None,
                    **metrics,schedule=runtime['clock'].summary(),probe_audit=runtime['inputs'].summary())
                log.write(json.dumps(item)+'\n'); log.flush(); print(json.dumps(item),flush=True)
            if updates%100 == 0: save('latest.pt')
            if updates%BLOCK_UPDATES == 0:
                block=updates//BLOCK_UPDATES; save(f'block{block}.pt')
                evaluation=checked_evaluate(net,cfg,opt,runtime,args.device,deadline)
                atomic_json(args.out/f'block{block}_evaluation.json',evaluation)
                stages.append(dict(block=block,extension_frames=frames,extra_frames=PARENT_FRAMES+frames,
                    evaluation=evaluation))
                atomic_json(args.out/'progress.json',dict(stages=stages))
    assert frames == EXTENSION_FRAMES and updates == BLOCKS*BLOCK_UPDATES
    assert {float(s['step']) for s in opt.state_dict()['state'].values()} == {float(PARENT_UPDATES+updates)}
    assert runtime['noise'].replays == PARENT_UPDATES+updates
    save('final.pt',completed=True)
    atomic_json(args.out/'final_evaluation.json',stages[-1]['evaluation'])
    return dict(completed=True,extension_frames=frames,extra_frames=TOTAL_EXTRA_FRAMES,
        extension_updates=updates,total_updates=PARENT_UPDATES+updates,
        extension_warmup_updates=0,seconds=time.monotonic()-started,stages=stages,
        optimizer_restored_exactly=True,runtime_and_private_rng_restored_exactly=True,
        no_boundary_hidden_or_episode_reset=True,
        noise_replay_max_error=runtime['noise'].max_replay_error,noise_replayed_updates=runtime['noise'].replays,
        schedule=runtime['clock'].summary(),probe_audit=runtime['inputs'].summary())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition',choices=tuple(WORKERS));ap.add_argument('--arm',choices=tuple(ARMS))
    ap.add_argument('--out',type=Path);ap.add_argument('--device',default='cuda')
    ap.add_argument('--max-seconds',type=int,default=6600);ap.add_argument('--protocol',type=Path,default=PROTOCOL)
    mode=ap.add_mutually_exclusive_group();mode.add_argument('--write-draft',action='store_true');mode.add_argument('--freeze-only',action='store_true')
    args=ap.parse_args();torch.set_num_threads(2)
    if args.write_draft:
        atomic_json(DRAFT,protocol(require_parents=False));print(str(DRAFT));return
    design=protocol(require_parents=True)
    if args.freeze_only:
        if args.protocol.exists():raise FileExistsError('Do not overwrite frozen entropy protocol')
        design['status']='Frozen matched continuation; launch still requires separate authorization'
        design['frozen_at']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(args.protocol,design)
        print(sha(args.protocol));return
    if not args.condition or not args.arm or args.out is None:ap.error('--condition, --arm and --out required')
    assert args.device in ('cuda','cuda:0'),'Exact parent CUDA baseline/RNG restoration is required'
    assert 0 < args.max_seconds <= 6600
    frozen=json.loads(args.protocol.read_text());assert frozen.pop('frozen_at')
    design['status']='Frozen matched continuation; launch still requires separate authorization'
    assert frozen==design,'Frozen source, task, parent or asset manifest changed'
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):raise FileExistsError('Use an empty output folder')
    design=copy.deepcopy(design);design['runtime']=dict(condition=args.condition,entropy_arm=args.arm,
        entropy_coefficient=ARMS[args.arm],device=args.device,torch_version=str(torch.__version__),
        protocol_sha256=sha(args.protocol))
    atomic_json(args.out/'protocol.json',design)
    parent_path=parent_folder(args.condition)/'final.pt'
    parent=torch.load(parent_path,map_location='cpu',weights_only=True);validate_parent(parent,args.condition)
    cfg=continuation_config(parent['config'],args.arm)
    net=build_policy(cfg,args.device);opt,runtime=restore_training(net,cfg,parent,args.device)
    atomic_json(args.out/'training_config.json',cfg)
    atomic_json(args.out/'resume_audit.json',dict(parent_checkpoint_sha256=sha(parent_path),
        parent_policy_hash=state_digest(net),optimizer_exact=True,runtime_private_rng_exact=True,
        changed_config_fields=[] if args.arm=='original' else ['entropy'],
        parent_updates=PARENT_UPDATES,extension_warmup_updates=0))
    deadline=time.monotonic()+args.max_seconds
    try:
        baseline=checked_evaluate(net,cfg,opt,runtime,args.device,deadline)
        expected=json.loads((parent_folder(args.condition)/'final_evaluation.json').read_text())
        atomic_json(args.out/'baseline.json',baseline)
        assert_baseline_equal(baseline,expected)
        atomic_json(args.out/'baseline_verification.json',dict(all12_complete_raw_episode_event_window_rows_exact=True,
            all_metadata_counters_exact=True,parent_reference_sha256=sha(parent_folder(args.condition)/'final_evaluation.json')))
        result=train(net,cfg,opt,runtime,parent,args,design,deadline)
        atomic_json(args.out/'training.json',result)
        assert sha(parent_path)==design['parent_assets'][args.condition]['sha256']['final.pt']
        for name,digest in design['original_asset_sha256'].items():assert sha(ROOT/name)==digest
        atomic_json(args.out/'summary.json',dict(completed=True,condition=args.condition,arm=args.arm,
            training=result,baseline_matches_parent_exactly=True,final_checkpoint_sha256=sha(args.out/'final.pt')))
    except Exception as error:
        atomic_json(args.out/'incomplete.json',dict(error_type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':main()
