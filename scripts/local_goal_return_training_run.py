"""Bounded reward-only local-return adaptation, separately frozen before use.

Fixed trial/update endpoints, explicit common native head reset, exact private
RNG/optimizer resume at complete-trial boundaries. No automatic extensions.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from code.common import build_policy
from rl_goal_memory import atomic_json, atomic_torch, state_digest
from rl_visual_memory_extend import tree_equal
import local_goal_return_training as task
from local_goal_return_training_evaluate import (BEHAVIOR, BEHAVIOR_HASH, evaluate_bank,
                                               initial_policy, load_bank, sha)

PROTOCOL = ROOT/'docs/research/local_goal_return_training_protocol.json'
BATCHES_PER_STAGE, TRIALS_PER_BATCH, TRAIN_SEED = 1024, 32, 294801
MAX_SECONDS, PROVIDER_MAX_SECONDS = 12600, 14400
BASELINE_ROOT = ROOT/'results/local_goal_return_baseline'
SOURCES = ('scripts/local_goal_return_training.py', 'scripts/local_goal_return_training_run.py',
    'scripts/local_goal_return_training_evaluate.py', 'scripts/local_goal_return_training_certificate.py',
    'tests/test_local_goal_return_training.py', 'tests/test_local_goal_return_audit.py',
    'tests/test_local_goal_return_resume.py', 'tests/test_local_goal_return_training_certificate.py',
    'scripts/certify_local_goal_return.py', 'scripts/rl_goal_memory.py',
    'scripts/rl_visual_memory.py', 'scripts/rl_visual_memory_extend.py', 'scripts/lstm_code_swap.py',
    'code/__init__.py', 'code/common.py', 'code/env.py', 'code/codes.py', 'code/nets.py')


def baseline_assets():
    assets = {}; heads = []
    for condition in ('gc100','pgc33'):
        folder = BASELINE_ROOT/condition
        top = json.loads((folder/'summary.json').read_text())
        near = json.loads((folder/'near_baseline/summary.json').read_text())
        assert top['complete'] and top['bank_gates']['near_baseline']['near_baseline_pass']
        assert near['complete'] and near['all_physical_checks'] and near['gates']['near_baseline_pass']
        assert (near['condition'], near['model_seed']) == (condition, 1)
        assert near['variants']['blind_native']['trials'] == 256
        assert near['variants']['blind_native']['successes'] >= 13
        assert set(near['controls']) == {'uniform','planner'}
        assert all(row['trials']==256 and row['invalid_trials']==0 for row in near['controls'].values())
        assert near['source_sha256'] == sha(ROOT/'scripts/local_goal_return_training_evaluate.py')
        assert near['core_sha256'] == sha(Path(task.__file__))
        assert near['behavior_protocol_sha256'] == BEHAVIOR_HASH
        assert top['initialized_policy_sha256'] == sha(folder/'initialized_policy.pt')
        for bank_name in ('near_baseline','final_hard'):
            result=json.loads((folder/bank_name/'summary.json').read_text())
            assert result['complete'] and result['all_physical_checks']
            assert set(result['variants']) == set(task.VARIANTS)
            assert all(v['trials']==256 and v['invalid_trials']==0 for v in result['variants'].values())
            assert set(result['controls']) == {'uniform','planner'}
            assert all(v['trials']==256 and v['invalid_trials']==0 for v in result['controls'].values())
            assert result['source_sha256']==sha(ROOT/'scripts/local_goal_return_training_evaluate.py')
            assert result['core_sha256']==sha(Path(task.__file__))
            assert result['behavior_protocol_sha256']==BEHAVIOR_HASH
            sealed=json.loads(BEHAVIOR.read_text())['banks'][bank_name]
            assert result['bank']==bank_name and result['bank_sha256']==sealed['sha256']
            assert result['checkpoint_policy_hash']==top['initialization']['full_initialized_policy_hash']
        heads.append(top['initialization']['new_head_hashes'])
        for name in ('summary.json','execution.json','initialization.json','initialized_policy.pt',
                     'near_baseline/summary.json','final_hard/summary.json'):
            assets[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    assert heads[0] == heads[1], 'Matched populations must receive identical new heads'
    return assets


def assert_training_config(cfg):
    expected = dict(n_envs=32, rollout=160, nh=256, n_cells=512, action_repeat=4,
        lr=3e-4, rms_alpha=.99, rms_eps=1e-5, gamma=.99, entropy=.01,
        value_coef=.5, grad_clip=40., reward_clip=1., extra_input='none', code_noise=0.)
    assert all(cfg[key] == value for key,value in expected.items()), 'Native hyperparameters differ from frozen design'


def load_baseline_policy(condition, device):
    path = BASELINE_ROOT/condition/'initialized_policy.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['no_optimization'] and saved['behavior_protocol_sha256'] == BEHAVIOR_HASH
    original_path = ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    assert original['frames']==1_000_000_000
    cfg = saved['config']; assert_training_config(cfg)
    expected_cfg = json.loads(original_path.with_suffix('.json').read_text())
    expected_cfg.update(n_envs=32, rollout=160, entropy=.01); assert cfg == expected_cfg
    initialization = saved['initialization']; assert initialization['parent_checkpoint_sha256'] == sha(original_path)
    for key,value in saved['policy'].items():
        if key.startswith(('pi.', 'v.')):
            assert task.tensor_hash(value) == initialization['new_head_hashes'][key]
        else: assert torch.equal(value, original['policy'][key]), key
    with torch.random.fork_rng(devices=[]): net = build_policy(cfg, 'cpu')
    net.load_state_dict(saved['policy']); assert state_digest(net) == initialization['full_initialized_policy_hash']
    return net.to(device), cfg, initialization


def protocol():
    behavior = json.loads(BEHAVIOR.read_text()); assert sha(BEHAVIOR) == BEHAVIOR_HASH
    for name in behavior['banks']: load_bank(name)
    return dict(version='standalone_local_return_reward_only_v1',
        original_model_seeds=[1], conditions=['gc100', 'pgc33'],
        scope='New episodic reward-only task adaptation. Retained original1B CNN/LSTM spatial weights; action/value heads explicitly reset. No claim of an attractor in the incoming policies.',
        architecture='Unchanged native RGB CNN embedding256, input1287, LSTM256, six15Hz actions, linear pi/v. All network parameters trainable; no ring/angle/decoder loss.',
        initialization=dict(head_seed_formula='291800+original_model_seed', native_actor_orthogonal_gain=.01,
            native_value_orthogonal_gain=1., biases_zero=True, reset_fields=['pi.weight','pi.bias','v.weight','v.bias'],
            exact_original_nonhead_bytes=True, common_heads_within_population_pair=True),
        goal='Actual native +10 setup pickup at goal−.02*independent arrival direction, through CodeSource.goal and NativeGoal.advance; exclude setup reward/pose/history from learner; fixed saved snapshot throughout the new trial.',
        task=dict(central_endpoints=[-.30,.30], stages=[asdict(s) for s in task.STAGES],
            independently_uniform='Endpoint x/y, local goal bearing, desired endpoint heading, setup arrival direction, four landmarks and four stripe phases; every training slot is a fresh context.',
            preparation='15visible forcedLEFT; two independent left/right blocks5–10updates; actual4forward probe; one native endpoint glimpse as first delay input; LEFT delay uniformly ceil(max/2)..max; motors braked at blind entry, beforeprobe,afterprobe.',
            physical_guards='Actual native preparatory path, rotated wall and goal clearance strictly>.01m; record/stop any failure without retry; endpoint roundoff correction must<1e−5m.',
            free_updates=30, termination='First actual FREE pickup or timeout is a true terminal. No future visible reward and no respawn exposure. h/c=0, pa=-1/pr=0 before each new trial.',
            observations='Native current code while visible; ordinary snapshot held through blindturns, updated at actual probe endpoint; native current pulse once; ordinary endpoint snapshot held duringremainingdelay/FREE. Saved native goal unmodified; actuallastaction, zero previousreward until terminal. No extraheading/goalcoordinate input.',
            noise='Preparation and images noiseless as physicalcertificate. NativeFREE body motor noise sigma_v.03/sigma_r.15,tau.25. Raw independent Gaussianh/c noise sigma.02 afterLSTMbeforeheads onlyblackactiveinputs, replay exactsavednoise.',
            resets='One complete trial/unroll; padding never carries state or losses. No clocks/resetlabels supplied to policy.'),
        learning=dict(optimizer='Fresh native RMSProp; no optimizer warmup or resumed source-task optimizer',
            original_learning_rate=3e-4, original_rms_alpha=.99, original_rms_eps=1e-5,
            gamma=.99, entropy=.01, value_coef=.5, grad_clip=40., reward_clip=1.,
            reward='Only native actualFREE+10 pickup; learner clips to1 exactly as originalconfig. No distance,setup,forced or futurevisible reward.',
            losses='Actor andentropy onlyFREE; critic allactiveforced/FREE. Sum over160 slots,mean across32trials,multiply100/160 fixed atallstages; maskallpadding. Complete MonteCarlo discountedreturns,terminalandtimeoutbootstrap identicallyzero.',
            action_sampling='Ordinary six-action categorical eachupdate; no temperature, actionholds, actionprior or planner intraining',
            stages=4, batches_per_stage=BATCHES_PER_STAGE, trials_per_batch=TRIALS_PER_BATCH,
            total_optimizer_updates=4*BATCHES_PER_STAGE, total_trials=4*BATCHES_PER_STAGE*TRIALS_PER_BATCH,
            counting='Report activeforced/FREEphysicalupdates×4 separately; setup,padding,planningphysics nevercalled experiencedlearningframes. Same attemptedtrial/optimizercount acrosspopulations.'),
        randomness=dict(training_seed_formula='294800+original_model_seed',
            private_streams='CPUplan,motor,action,state generators; entire160update tapes drawn beforeoutcomes; keyedbyfixedbatchorder, neverprevioushitcount. Layouts explicit; setup usesseparatefixednativeenv generator.',
            resume='Save exactpolicy,RMSProp,privategeneratorstates,counters,CPU/CUDAglobalRNG at complete-trial boundaries. No in-flighttrial state is lost because none persists.'),
        evaluation=dict(behavior_protocol_sha256=BEHAVIOR_HASH, sealed_banks=behavior['banks'],
            endpoints='Reset-headbaseline and eachof4fixedstageends; full256hardtrials andall12variants at eachstage. Final is prescribedlastupdate, no bestcheckpointselection.',
            near_baseline_gate=behavior['baseline_gate'], final_behavior_gate=behavior['final_behavior_gate'],
            nativecontrols='Correct/ordinary/±60currentcue; postcue/everyblindstateerase; matchedvisiblecontrols; originalsavedgoal retained. Gridcurrentcuecontrols exactshams.',
            raw='Everytrial/context/variant, setup/forced/FREE counters, actualfirstFREEstate/input/pose/bodymotors,noise/actiontapesbysealedreference,actions/positions,source/provenance hashes; allfailures retained.',
            clean_h_c_noise=0., scientific_inference='Pilot one originaltrainingseedpair.256trials nested in16contexts; descriptivefrozen gates,nottrained-seed significance. Any50percentcertificate is conditionaltapes/observationrule.'),
        passed_baseline_assets=baseline_assets(),
        incoming_policy='Load exact stored reset-head baseline initialized_policy.pt bytes; never rerun QR/head initialization on another backend/version.',
        source_sha256={name:sha(ROOT/name) for name in SOURCES},
        original_assets={f'models/{condition}/{condition}_lstm_s1_1000M{suffix}':
            sha(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M{suffix}')
            for condition in ('gc100','pgc33') for suffix in ('.json','.pt')},
        infrastructure=dict(process_max_seconds=MAX_SECONDS, provider_delete_seconds=PROVIDER_MAX_SECONDS,
            planned_workers=2, accelerator='nvidia-tesla-t4', machine_type='n1-standard-8',
            planned_zone='us-central1-a', maximum_pair_reservation_usd=10.625,
            early_cleanup='Collect and verify all fixed endpoints, then delete instances/disks promptly; no automatic extension or restart.'),
        launch='No launch implied by implementation/freeze. Separate root scientific/infrastructure authorization within original cumulative500USD.')


def global_rng_snapshot():
    return dict(cpu=torch.random.get_rng_state().clone(),
        cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_global_rng(saved):
    torch.random.set_rng_state(saved['cpu'].cpu())
    if saved['cuda']:
        assert torch.cuda.is_available() and len(saved['cuda']) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all(saved['cuda'])


def fresh_counters():
    return dict(updates=0, trials=0, active_physics_updates=0, forced_updates=0, free_updates=0,
        padded_slots=0, setup_pickups=0, free_hits=0, completed_stages=0, evaluation_records=[])


def snapshot(net, optimizer, streams, cfg, initialization, counters, provenance):
    assert streams.batches == counters['updates'], 'Only completed trial batches are resumable'
    return dict(version='local_return_exact_boundary_resume_v1', policy=net.state_dict(),
        optimizer=optimizer.state_dict(), streams=streams.snapshot(), global_rng=global_rng_snapshot(),
        config=cfg, initialization=initialization, counters=dict(counters), provenance=provenance,
        no_inflight_trial=True, boundary_state='Everytrialterminated; nexttrialh/czero,pa−1/pr0',
        completed=counters['updates']==4*BATCHES_PER_STAGE and counters['completed_stages']==4,
        active_physics_frames=counters['active_physics_updates']*cfg['action_repeat'])


def restore(net, optimizer, streams, saved, cfg, provenance):
    assert saved['version'] == 'local_return_exact_boundary_resume_v1' and saved['no_inflight_trial']
    assert saved['config'] == cfg and saved['provenance'] == provenance
    net.load_state_dict(saved['policy']); optimizer.load_state_dict(saved['optimizer'])
    streams.restore(saved['streams']); restore_global_rng(saved['global_rng'])
    assert streams.batches == saved['counters']['updates']
    assert streams.contexts == saved['counters']['trials']
    pending_stage(saved['counters'])
    return dict(saved['counters'])


def pending_stage(counters):
    """Only a just-completed boundary may have one unevaluated stage."""
    updates, completed = counters['updates'], counters['completed_stages']
    boundary = updates // BATCHES_PER_STAGE
    assert 0 <= completed <= boundary <= 4
    if updates % BATCHES_PER_STAGE:
        assert completed == boundary, 'Cannot resume after skipping an earlier prescribed evaluation'
    else:
        assert boundary-completed in (0,1), 'Multiple missing boundary evaluations cannot be reconstructed'
    return boundary if completed < boundary else None


def inherit_resume_artifacts(checkpoint, out, counters):
    """Keep completed endpoint evidence and the precise parent checkpoint."""
    parent = checkpoint.parent; records = []
    evidence=out/'resume_parent_evidence'; evidence.mkdir()
    evidence_sources=[(checkpoint,evidence/'checkpoint.pt')]
    assert (parent/'metrics.jsonl').exists(), 'Preserve the complete raw parent-attempt learning log'
    for name in ('metrics.jsonl','protocol.json','initialization.json','training_config.json',
                 'incomplete.json','summary.json','resume_lineage.json'):
        if (parent/name).exists(): evidence_sources.append((parent/name,evidence/name))
    for source,target in evidence_sources:
        shutil.copy2(source,target); assert sha(source)==sha(target)
        records.append(dict(source=str(source),copied=str(target.relative_to(out)),sha256=sha(source)))
    if (parent/'resume_parent_evidence').exists():
        ancestor=parent/'resume_parent_evidence'; target=evidence/'ancestor_evidence'
        shutil.copytree(ancestor,target)
        for p in sorted(q for q in ancestor.rglob('*') if q.is_file()):
            copied=target/p.relative_to(ancestor); assert sha(copied)==sha(p)
            records.append(dict(source=str(p),copied=str(copied.relative_to(out)),sha256=sha(p)))
    assert len(counters['evaluation_records'])==counters['completed_stages']
    for stage in range(1, counters['completed_stages']+1):
        record=counters['evaluation_records'][stage-1]; assert record['stage']==stage
        assert sha(parent/f'stage{stage}_before_evaluation.pt')==record['checkpoint_before_evaluation_sha256']
        assert sha(parent/f'stage{stage}_evaluation/summary.json')==record['evaluation_sha256']
        for name in (f'stage{stage}.pt', f'stage{stage}_before_evaluation.pt',
                     f'stage{stage}_evaluation', f'stage{stage}_evaluation_integrity.json'):
            source = parent/name; assert source.exists(), f'Missing required resumed-stage artifact: {source}'
            target = out/name
            if source.is_dir(): shutil.copytree(source,target)
            else: shutil.copy2(source,target)
            sources = sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
            for p in sources:
                copied = out/p.relative_to(parent)
                assert sha(copied) == sha(p)
                records.append(dict(source=str(p),copied=str(copied.relative_to(out)),sha256=sha(p)))
    pending = pending_stage(counters)
    if pending and (parent/f'stage{pending}_evaluation').exists():
        old = parent/f'stage{pending}_evaluation'; preserved = out/'parent_incomplete_evaluation'
        shutil.copytree(old,preserved)
        for p in sorted(q for q in old.rglob('*') if q.is_file()):
            assert sha(preserved/p.relative_to(old))==sha(p)
            records.append(dict(source=str(p),copied=str((preserved/p.relative_to(old)).relative_to(out)),sha256=sha(p)))
    lineage = dict(resume_checkpoint=str(checkpoint.resolve()),resume_checkpoint_sha256=sha(checkpoint),
        restored_updates=counters['updates'],completed_stages=counters['completed_stages'],pending_stage=pending,
        copied_artifacts=records,parent_lineage=None,
        parent_metrics_scope='Raw entire parent attempt is preserved, including any updates after the restored cutoff. Rows after restored_updates are rolled-back attempt evidence, never added to accepted-run counts. Nested ancestor evidence is preserved without double counting.')
    if (parent/'resume_lineage.json').exists(): lineage['parent_lineage']=json.loads((parent/'resume_lineage.json').read_text())
    atomic_json(out/'resume_lineage.json',lineage)


def checked_evaluation(net,cfg,optimizer,streams,out,device):
    policy_before=state_digest(net); optimizer_before=copy.deepcopy(optimizer.state_dict())
    streams_before=streams.snapshot(); rng_before=global_rng_snapshot()
    result=evaluate_bank(net,cfg,'final_hard',out,device=device)
    assert state_digest(net)==policy_before
    assert tree_equal(optimizer_before,optimizer.state_dict()), 'Evaluation mutated RMSProp'
    assert tree_equal(streams_before,streams.snapshot()), 'Evaluation consumed training private streams'
    assert tree_equal(rng_before,global_rng_snapshot()), 'Evaluation consumed global learning RNG'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=('gc100','pgc33')); parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--out',type=Path)
    parser.add_argument('--resume',type=Path); parser.add_argument('--max-seconds',type=int,default=MAX_SECONDS)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL); parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(2); design=protocol()
    if args.freeze_only:
        assert not args.protocol.exists(), 'Never overwrite a frozen protocol'
        atomic_json(args.protocol, dict(design, frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(sha(args.protocol)); return
    assert args.condition and args.out and args.seed==1 and 0<args.max_seconds<=MAX_SECONDS
    frozen=json.loads(args.protocol.read_text()); assert frozen.pop('frozen_utc')
    assert frozen==design, 'Training sources/assets/banks changed after freeze'
    args.out.mkdir(parents=True,exist_ok=True); assert not any(args.out.iterdir()), 'Use an empty experiment child folder'
    net,cfg,initialization=load_baseline_policy(args.condition,args.device); net.train()
    source=task.make_source(cfg,args.device)
    optimizer=torch.optim.RMSprop(net.parameters(),lr=cfg['lr'],alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    streams=task.TrialStreams(TRAIN_SEED+args.seed-1); counters=fresh_counters()
    provenance=dict(design=design,protocol_sha256=sha(args.protocol),condition=args.condition,
        model_seed=args.seed,device=args.device,torch_version=str(torch.__version__))
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        counters=restore(net,optimizer,streams,saved,cfg,provenance); initialization=saved['initialization']
        inherit_resume_artifacts(args.resume,args.out,counters)
    atomic_json(args.out/'protocol.json',provenance); atomic_json(args.out/'initialization.json',initialization)
    atomic_json(args.out/'training_config.json',cfg)
    save=lambda name:atomic_torch(args.out/name,snapshot(net,optimizer,streams,cfg,initialization,counters,provenance))
    save('incoming.pt'); started=time.monotonic(); deadline=started+args.max_seconds

    def evaluate_stage(stage):
        assert counters['updates']==stage*BATCHES_PER_STAGE and counters['completed_stages']==stage-1
        save(f'stage{stage}_before_evaluation.pt')
        folder=args.out/f'stage{stage}_evaluation'
        evaluation=checked_evaluation(net,cfg,optimizer,streams,folder,args.device)
        assert evaluation['complete']
        record=dict(stage=stage,checkpoint_before_evaluation_sha256=sha(args.out/f'stage{stage}_before_evaluation.pt'),
            evaluation_path=str(folder/'summary.json'),evaluation_sha256=sha(folder/'summary.json'),
            model_policy_unchanged=True,optimizer_unchanged=True,private_streams_unchanged=True,global_rng_unchanged=True)
        atomic_json(args.out/f'stage{stage}_evaluation_integrity.json',record)
        counters['evaluation_records'].append(record); counters['completed_stages']=stage
        save(f'stage{stage}.pt')

    try:
        pending=pending_stage(counters)
        if pending is not None: evaluate_stage(pending)
        with (args.out/'metrics.jsonl').open('w') as log:
            for index in range(counters['updates'],4*BATCHES_PER_STAGE):
                if time.monotonic()>deadline-180: raise TimeoutError('Preserve incomplete fixed schedule; no adaptive extension')
                stage=index//BATCHES_PER_STAGE; before_streams=streams.snapshot()
                plan=streams.draw(TRIALS_PER_BATCH,task.STAGES[stage],cfg)
                try: metrics,rows=task.update(net,cfg,optimizer,source,plan,args.device)
                except Exception:
                    atomic_torch(args.out/'failed_plan.pt',plan)
                    atomic_torch(args.out/'failed_predraw_streams.pt',before_streams)
                    raise
                counters['updates']+=1; counters['trials']+=TRIALS_PER_BATCH
                for key in ('active_physics_updates','forced_updates','free_updates','padded_slots','setup_pickups','free_hits'):
                    counters[key]+=metrics[key]
                item=dict(update=counters['updates'],stage=stage+1,seconds=time.monotonic()-started,
                    active_physics_frames=counters['active_physics_updates']*4,**metrics)
                log.write(json.dumps(item)+'\n');log.flush()
                if counters['updates']%20==0:print(json.dumps(item),flush=True)
                if counters['updates']%128==0:save('latest.pt')
                if counters['updates']%BATCHES_PER_STAGE==0:
                    evaluate_stage(stage+1)
            assert counters['updates']==4096 and counters['trials']==131072 and counters['completed_stages']==4
            save('final.pt')
            atomic_json(args.out/'summary.json',dict(completed=True,counters=counters,
                active_physics_frames=counters['active_physics_updates']*4,final_checkpoint_sha256=sha(args.out/'final.pt'),
                final_evaluation='stage4_evaluation/summary.json',seconds=time.monotonic()-started,
                protocol_sha256=sha(args.protocol),initialization=initialization))
    except Exception as error:
        atomic_json(args.out/'incomplete.json',dict(error_type=type(error).__name__,error=str(error),counters=counters))
        raise


if __name__=='__main__':main()
