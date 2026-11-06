"""Draft bounded native goal-memory qualification/training runner.

No study action occurs on import. A future frozen protocol and exact sealed
banks are mandatory before qualification; both original policies must qualify
before this runner can learn. No automatic extension or checkpoint selection.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import goal_cue_memory as task
import goal_cue_memory_bank as banks
import goal_cue_memory_evaluate as evaluator
import local_return_feedback as initializer
from local_return_feedback_evaluate import global_rng_snapshot,restore_global_rng,preserved_rng
from rl_goal_memory import atomic_json,atomic_torch,state_digest
from rl_visual_memory_extend import tree_equal

PROTOCOL=ROOT/'docs/research/goal_cue_memory_training_protocol.json'
INITIALIZATION=ROOT/'docs/research/local_return_feedback_initialization.json'
INITIALIZATION_SHA='471ab69f6167966c2bf1efb5a05e27d5fbc290308c13851dab4bc966b1c8eb82'
PARENT_SOURCE_PROTOCOL=ROOT/'docs/research/local_return_feedback_portable_protocol.json'
PARENT_SOURCE_PROTOCOL_SHA='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
UPDATES_PER_BLOCK,TRIALS_PER_BATCH,TRAIN_SEED=1024,32,694801
DELAY_CAPS=(0,15,45)
TOTAL_UPDATES=UPDATES_PER_BLOCK*len(DELAY_CAPS)
MAX_SECONDS,PROVIDER_SECONDS=19800,21600
VERSION='goal_cue_memory_exact_resume_v1'
COUNTERS=('active_recurrent_updates','active_physics_updates','free_updates','wait_updates',
          'padded_slots','setup_pickups','free_hits')
ENDPOINTS={'baseline':0,'block1':1024,'block2':2048,'block3':3072}
NEW_SOURCES=('scripts/goal_cue_memory.py','scripts/goal_cue_memory_bank.py',
    'scripts/goal_cue_memory_evaluate.py','scripts/goal_cue_memory_run.py','scripts/audit_goal_cue_memory.py',
    'tests/test_goal_cue_memory.py','tests/test_goal_cue_memory_independent.py',
    'tests/test_goal_cue_memory_bank.py','tests/test_goal_cue_memory_evaluate.py',
    'tests/test_goal_cue_memory_run.py','tests/test_audit_goal_cue_memory.py')
sha=banks.sha


def assert_config(cfg):
    for key,expected in dict(n_envs=32,rollout=160,nh=256,n_cells=512,action_repeat=4,
        entropy=8e-5,lr=3e-4,rms_alpha=.99,rms_eps=1e-5,gamma=.99,value_coef=.5,
        grad_clip=40.,reward_clip=1.,policy='lstm',conv_arch='si',encoder_norm='layernorm',
        extra_input='none',code_noise=0.,goal_radius=.3,pgc_shift=.2,img_noise=0.).items():
        assert cfg[key]==expected,(key,cfg[key])
    assert cfg['cond'] in ('gc100','pgc33') and cfg['seed']==1


def initial_assets():
    assert sha(INITIALIZATION)==INITIALIZATION_SHA
    manifest=json.loads(INITIALIZATION.read_text())
    assert manifest['complete'] and manifest['no_forward_pass'] and manifest['no_optimization']
    assets={str(INITIALIZATION.relative_to(ROOT)):INITIALIZATION_SHA}
    for condition in ('gc100','pgc33'):
        row=next(r for r in manifest['policies'] if r['condition']==condition and r['actor_mode']=='retained_actor')
        relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(ROOT/relative)==row['sha256'] and (ROOT/relative).stat().st_size==row['bytes']
        assets[str(relative)]=row['sha256']
        for name in (f'models/{condition}/{condition}_lstm_s1_1000M.pt',
                     f'models/{condition}/{condition}_lstm_s1_1000M.json',
                     f'results/local_goal_return_baseline/{condition}/initialized_policy.pt'):
            assets[name]=sha(ROOT/name)
    assets[initializer.PARENT_PROTOCOL]=sha(ROOT/initializer.PARENT_PROTOCOL)
    assert assets[initializer.PARENT_PROTOCOL]==initializer.PARENT_PROTOCOL_SHA256
    return assets


def load_initial(condition,device='cpu'):
    """Use exact archived original actor/features and the shared fresh critic."""
    initial_assets(); manifest=json.loads(INITIALIZATION.read_text())
    row=next(r for r in manifest['policies'] if r['condition']==condition and r['actor_mode']=='retained_actor')
    saved=torch.load(ROOT/row['path'],map_location='cpu',weights_only=True)
    with preserved_rng():
        net,cfg,metadata=initializer.initialize(condition,'retained_actor',device='cpu')
    for key in ('parent_checkpoint_path','archived_heads_path'):
        metadata[key]=str(Path(metadata[key]).resolve().relative_to(ROOT))
    assert saved['config']==cfg and saved['initialization']==metadata
    assert saved['no_forward_pass'] and saved['no_optimization']
    assert tree_equal(saved['policy'],net.state_dict())
    assert state_digest(net)==row['full_initial_policy_hash']
    cfg=dict(cfg,img_noise=0.)
    assert_config(cfg); initializer.configure_trainable(net)
    initialization=dict(condition=condition,original_seed=1,actor='retained_original_1B',
        shared_fresh_critic=True,all_parameters_trainable=True,no_optimization=True,
        incoming_asset=row['path'],incoming_asset_sha256=row['sha256'],
        original_initialization_manifest_sha256=INITIALIZATION_SHA,
        full_initial_policy_hash=state_digest(net),original=metadata,
        task_setting_change=dict(img_noise=0.,stationary_cue_delay=True,first_return_trial=True))
    return net.to(device),cfg,initialization


def protocol(manifest=banks.BEHAVIOR):
    """Prospective source/assets only; never reads qualification outcomes."""
    assert sha(PARENT_SOURCE_PROTOCOL)==PARENT_SOURCE_PROTOCOL_SHA
    sources=set(json.loads(PARENT_SOURCE_PROTOCOL.read_text())['source_sha256'])|set(NEW_SOURCES)|{'code/__init__.py'}
    assets=initial_assets();bank_records={}
    for name in banks.BANKS:
        _,record,_=banks.load_bank(manifest,name);bank_records[name]=record
        assets[record['path']]=record['sha256']
    relative=str(Path(manifest).resolve().relative_to(ROOT));assets[relative]=sha(manifest)
    assets[str(PARENT_SOURCE_PROTOCOL.relative_to(ROOT))]=PARENT_SOURCE_PROTOCOL_SHA
    return dict(version='native_goal_cue_memory_training_v1',original_seed=1,conditions=['gc100','pgc33'],
        initialization='Exact original1B actor/CNN/LSTM; identical archived fresh critic; fresh native RMSProp; all parameters trainable.',
        qualification=dict(bank='qualification',variants=list(evaluator.QUALIFICATION_VARIANTS),successes=192,trials=256,
            both_populations_required=True,no_qualification_training=True,delay_steps=45),
        training=dict(updates_per_block=1024,blocks=3,total_updates=3072,trials_per_batch=32,
            trials_per_model=98304,training_seed=TRAIN_SEED,delay_caps=list(DELAY_CAPS),
            delay_sampling='One private torch.randint(0,cap+1,(1,)) per batch, including cap0; generator checkpointed; both populations consume identical draws.',
            full_trial_bptt=160,cue_steps=15,free_steps=90,actor_entropy_only_free=True,critic_all_active=True,
            gamma=.99,entropy=8e-5,lr=3e-4,rms_alpha=.99,rms_eps=1e-5,value_coef=.5,grad_clip=40.,reward_clip=1.,
            state_noise=0.,img_noise=0.,free_motor_noise_v=.03,free_motor_noise_r=.15,
            loss_scaling='Fixed100/160 times mean per-trial summed policy/value/entropy losses; padding fully masked.'),
        evaluation=dict(manifest_path=relative,manifest_sha256=sha(manifest),banks=bank_records,
            baseline_variants=list(evaluator.FULL_VARIANTS),interim_variants=list(evaluator.INTERIM_VARIANTS),
            final_variants=list(evaluator.FULL_VARIANTS),all_on_memory_bank=True,
            descriptive_memory_screen=dict(successes=167,erasure_loss=39,trials=256),
            no_50_percent_certificate=True,donor_before_first_free_input=True,no_adaptive_endpoint_selection=True),
        limits=dict(process_seconds=MAX_SECONDS,provider_delete_seconds=PROVIDER_SECONDS,
            new_pilot_usd_ceiling=20,original_user_budget_usd=500,no_automatic_extension=True),
        source_sha256={name:sha(ROOT/name) for name in sorted(sources)},input_assets=assets)


def fresh_counters():
    return dict(updates=0,trials=0,**{key:0 for key in COUNTERS},evaluation_records={},
                delay_histograms={f'block{i+1}':[0]*46 for i in range(3)})


def pending_evaluation(counters):
    updates=counters['updates'];records=counters['evaluation_records']
    assert 0<=updates<=TOTAL_UPDATES and set(records)<=set(ENDPOINTS)
    for name,boundary in ENDPOINTS.items():
        if name in records: assert updates>=boundary
        elif updates>boundary: raise AssertionError(f'Missing prescribed evaluation {name} before later learning')
        elif updates==boundary: return name
    return None


def validate_counters(counters,cfg):
    pending_evaluation(counters)
    updates=counters['updates'];n=cfg['n_envs'];trials=counters['trials']
    assert trials==updates*n and all(isinstance(counters[k],int) and counters[k]>=0 for k in COUNTERS)
    hist=counters['delay_histograms'];assert set(hist)=={'block1','block2','block3'}
    wait=0
    for index,cap in enumerate(DELAY_CAPS):
        values=hist[f'block{index+1}'];assert len(values)==46
        assert all(isinstance(v,int) and v>=0 for v in values)
        assert sum(values)==min(max(updates-index*UPDATES_PER_BLOCK,0),UPDATES_PER_BLOCK)
        assert not any(values[cap+1:])
        wait+=sum(count*(task.CUE_STEPS+delay)*n for delay,count in enumerate(values))
    assert counters['wait_updates']==wait
    assert counters['active_physics_updates']==counters['free_updates']
    assert trials<=counters['free_updates']<=task.FREE_STEPS*trials
    assert counters['active_recurrent_updates']==counters['wait_updates']+counters['free_updates']
    assert counters['padded_slots']==task.MAX_STEPS*trials-counters['active_recurrent_updates']
    assert counters['setup_pickups']==trials and counters['free_hits']<=trials


def snapshot(net,opt,streams,cfg,initialization,counters,provenance):
    assert streams.batches==counters['updates'] and streams.contexts==counters['trials']
    assert streams.delay_draws==counters['updates']
    assert counters['trials']==counters['updates']*cfg['n_envs'];validate_counters(counters,cfg)
    return dict(version=VERSION,policy=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        streams=streams.snapshot(),global_rng=global_rng_snapshot(),config=copy.deepcopy(cfg),
        initialization=copy.deepcopy(initialization),counters=copy.deepcopy(counters),provenance=copy.deepcopy(provenance),
        no_inflight_trial=True,boundary_state='All trials terminal; next trial starts center,h/c0,motors0,pa=-1,pr0',
        completed=counters['updates']==TOTAL_UPDATES and 'block3' in counters['evaluation_records'],
        active_physics_frames=counters['active_physics_updates']*cfg['action_repeat'])


def restore(net,opt,streams,saved,cfg,provenance):
    assert saved['version']==VERSION and saved['no_inflight_trial']
    assert saved['config']==cfg and saved['provenance']==provenance
    counters=copy.deepcopy(saved['counters']);validate_counters(counters,cfg)
    assert saved['streams']['batches']==counters['updates']
    assert saved['streams']['delay_draws']==counters['updates']
    assert saved['streams']['contexts']==counters['trials']==counters['updates']*cfg['n_envs']
    assert saved['active_physics_frames']==counters['active_physics_updates']*cfg['action_repeat']
    net.load_state_dict(saved['policy'],strict=True);initializer.configure_trainable(net)
    opt.load_state_dict(saved['optimizer']);streams.restore(saved['streams']);restore_global_rng(saved['global_rng'])
    return counters


def update(net,cfg,opt,source,plan,*,device='cpu'):
    expected={id(p) for p in net.parameters()}
    supplied=[p for group in opt.param_groups for p in group['params']]
    assert len(supplied)==len(expected) and {id(p)for p in supplied}==expected
    assert all(p.requires_grad for p in supplied)
    roll,rows,audit,_=task.collect(net,cfg,source,plan,device=device,cue_mode='brief')
    targets=task.returns(roll,cfg['gamma']);logits,values=task.replay(net,roll)
    logit_error=float((logits.detach()-torch.stack(roll['logits'])).abs().max())
    value_error=float((values.detach()-torch.stack(roll['values'])).abs().max())
    assert logit_error<3e-4 and value_error<3e-4
    loss,metrics=task.losses(logits,values,targets,roll,cfg);assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['grad_clip']);assert torch.isfinite(norm)
    opt.step()
    return dict(loss=float(loss.detach()),grad_norm=float(norm),replay_error=logit_error,
                value_replay_error=value_error,delay_steps=plan['delay_steps'],**metrics,**audit),rows


def advance_counters(counters,metrics,cfg):
    assert metrics['completed_trials']==cfg['n_envs']
    assert metrics['setup_learning_reward']==metrics['forced_rewards']==metrics['forced_actions']==0
    block=counters['updates']//UPDATES_PER_BLOCK
    assert 0<=metrics['delay_steps']<=DELAY_CAPS[block]
    counters['delay_histograms'][f'block{block+1}'][metrics['delay_steps']]+=1
    counters['updates']+=1;counters['trials']+=cfg['n_envs']
    for key in COUNTERS:counters[key]+=metrics[key]


def checked_evaluation(net,cfg,opt,streams,out,*,device,variants,manifest=banks.BEHAVIOR):
    before=dict(policy=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        streams=streams.snapshot(),rng=global_rng_snapshot(),training=net.training)
    result=evaluator.evaluate_bank(net,cfg,out,device=device,manifest=manifest,bank_name='memory',variants=variants)
    assert tree_equal(before['policy'],net.state_dict()) and tree_equal(before['optimizer'],opt.state_dict())
    assert tree_equal(before['streams'],streams.snapshot()) and tree_equal(before['rng'],global_rng_snapshot())
    assert net.training==before['training'] and result['complete']
    assert all(result['variants'][name]['all_physical_checks'] for name in variants)
    return result


def qualification_gate(folder,protocol_sha256):
    """Authenticated pair prerequisite, evaluated before any learning occurs."""
    folder=Path(folder);rows=[]
    for condition in ('gc100','pgc33'):
        child=folder/condition;ap=folder/'audits'/f'{condition}.json'
        audit=json.loads(ap.read_text())
        assert audit['phase']=='qualification' and audit['condition']==condition
        assert audit['protocol_sha256']==protocol_sha256
        assert audit['passed'] and audit['complete'] and not audit['failures'] and not audit['pending']
        assert audit['auditor_sha256']==sha(ROOT/'scripts/audit_goal_cue_memory.py')
        assert {'incoming.pt','qualification_after.pt','protocol.json','qualification_evaluation/summary.json'} <= set(audit['artifacts'])
        for name,proof in audit['artifacts'].items():
            rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
            assert sha(child/rel)==proof['sha256'] and (child/rel).stat().st_size==proof['bytes']
        report=json.loads((child/'qualification_evaluation/summary.json').read_text())
        assert report['condition']==condition and report['variant_order']==list(evaluator.QUALIFICATION_VARIANTS)
        initial=next(r for r in json.loads(INITIALIZATION.read_text())['policies']
                     if r['condition']==condition and r['actor_mode']=='retained_actor')
        assert report['checkpoint_policy_hash']==initial['full_initial_policy_hash']
        result=report['variants']['continuous'];totals=result['totals']
        assert result['all_physical_checks'] and totals['valid_trials']==totals['trials']==256
        assert totals['invalid_trials']==0 and totals['successes']>=192,'Both frozen original policies must qualify; no competence-training fallback'
        rows.append(dict(condition=condition,successes=totals['successes'],trials=256,
            audit_path=str(ap.resolve()),audit_relative_path=str(ap.relative_to(folder)),
            summary_relative_path=f'{condition}/qualification_evaluation/summary.json',
            audit_sha256=sha(ap),summary_sha256=sha(child/'qualification_evaluation/summary.json')))
    return dict(passed=True,both_originals_qualified=True,protocol_sha256=protocol_sha256,
                qualification_root_at_execution=str(folder.resolve()),rows=rows)


def inherit_resume_artifacts(checkpoint,out,counters):
    """Keep exact restart bytes, accepted earlier endpoints and rolled-back logs."""
    checkpoint,out=Path(checkpoint),Path(out);parent=checkpoint.parent
    evidence=out/'resume_parent_evidence';evidence.mkdir();copied=[]
    def copy_item(source,target):
        if source.is_dir():shutil.copytree(source,target)
        else:shutil.copy2(source,target)
        files=sorted(p for p in source.rglob('*')if p.is_file())if source.is_dir()else[source]
        for path in files:
            dest=target/path.relative_to(source)if source.is_dir()else target
            assert sha(path)==sha(dest)
            copied.append(dict(source=str(path.resolve()),copied=str(dest.relative_to(out)),sha256=sha(path)))
    assert(parent/'metrics.jsonl').exists();copy_item(checkpoint,evidence/'checkpoint.pt')
    if counters['updates']==TOTAL_UPDATES:
        # Keep latest's role as the final pre-evaluation training boundary.
        # The supplied restart bytes are always preserved separately above,
        # including a restart from an already completed final checkpoint.
        latest_source=parent/'block3_before_evaluation.pt' if 'block3' in counters['evaluation_records'] else checkpoint
        copy_item(latest_source,out/'latest.pt')
    for name in ('incoming.pt','metrics.jsonl','protocol.json','initialization.json','training_config.json',
                 'qualification_gate.json','incomplete.json','summary.json','resume_lineage.json',
                 'failed_plan.pt','failed_predraw_streams.pt'):
        if(parent/name).exists():copy_item(parent/name,evidence/name)
    if(parent/'resume_parent_evidence').exists():copy_item(parent/'resume_parent_evidence',evidence/'ancestor_evidence')
    for name,record in counters['evaluation_records'].items():
        assert sha(parent/f'{name}_before_evaluation.pt')==record['checkpoint_before_sha256']
        assert sha(parent/f'{name}_evaluation/summary.json')==record['summary_sha256']
        for relative in (f'{name}_before_evaluation.pt',f'{name}.pt',f'{name}_evaluation',f'{name}_evaluation_integrity.json'):
            copy_item(parent/relative,out/relative)
    pending=pending_evaluation(counters)
    if pending and(parent/f'{pending}_evaluation').exists():copy_item(parent/f'{pending}_evaluation',out/f'parent_incomplete_{pending}_evaluation')
    atomic_json(out/'resume_lineage.json',dict(resume_checkpoint=str(checkpoint.resolve()),resume_checkpoint_sha256=sha(checkpoint),
        restored_updates=counters['updates'],pending_evaluation=pending,copied_artifacts=copied,
        parent_lineage=json.loads((parent/'resume_lineage.json').read_text())if(parent/'resume_lineage.json').exists()else None,
        parent_metrics_scope='Raw attempt rows retained; only updates at/below restored cutoff accepted, later rows rolled back.'))


def qualification_snapshot(net,cfg,initialization,provenance):
    return dict(version='goal_cue_memory_qualification_v1',phase='qualification',
        policy=copy.deepcopy(net.state_dict()),config=copy.deepcopy(cfg),
        initialization=copy.deepcopy(initialization),provenance=copy.deepcopy(provenance),
        global_rng=global_rng_snapshot(),no_optimization=True)


def run_qualification(net,cfg,initialization,provenance,out,*,device,manifest):
    before=qualification_snapshot(net,cfg,initialization,provenance)
    atomic_torch(out/'incoming.pt',before)
    result=evaluator.evaluate_bank(net,cfg,out/'qualification_evaluation',device=device,
        manifest=manifest,bank_name='qualification',variants=evaluator.QUALIFICATION_VARIANTS)
    after=qualification_snapshot(net,cfg,initialization,provenance)
    assert tree_equal(before,after),'Qualification must preserve the entire original policy and RNG'
    atomic_torch(out/'qualification_after.pt',after)
    totals=result['variants']['continuous']['totals']
    valid=result['complete'] and result['variants']['continuous']['all_physical_checks']
    qualified=valid and totals['valid_trials']==totals['trials']==256 and totals['invalid_trials']==0 and totals['successes']>=192
    integrity=dict(policy_and_global_rng_unchanged=True,incoming_sha256=sha(out/'incoming.pt'),
        after_sha256=sha(out/'qualification_after.pt'),summary_sha256=sha(out/'qualification_evaluation/summary.json'))
    atomic_json(out/'qualification_evaluation_integrity.json',integrity)
    summary=dict(phase='qualification',complete=True,condition=cfg['cond'],qualified=qualified,
        no_optimization=True,protocol_sha256=provenance['protocol_sha256'],totals=totals,
        policy_hash=state_digest(net),evaluation_summary_sha256=integrity['summary_sha256'])
    atomic_json(out/'summary.json',summary)
    return summary


def run_training(net,cfg,initialization,provenance,out,*,device,manifest,resume=None,max_seconds=MAX_SECONDS):
    opt=initializer.optimizer(net,cfg);streams=task.TrialStreams(TRAIN_SEED)
    source=task.make_source(cfg,device);counters=fresh_counters();started=time.monotonic()
    if resume is not None:
        saved=torch.load(resume,map_location=device,weights_only=True)
        assert saved['initialization']==initialization
        counters=restore(net,opt,streams,saved,cfg,provenance)
        inherit_resume_artifacts(resume,out,counters)
    atomic_torch(out/'incoming.pt',snapshot(net,opt,streams,cfg,initialization,counters,provenance))
    (out/'metrics.jsonl').touch(exist_ok=False)
    failed_plan=predraw=None
    try:
        while counters['updates']<TOTAL_UPDATES or pending_evaluation(counters) is not None:
            if time.monotonic()-started>=max_seconds:
                atomic_torch(out/'latest.pt',snapshot(net,opt,streams,cfg,initialization,counters,provenance))
                atomic_json(out/'incomplete.json',dict(reason='process_deadline',counters=counters,
                    seconds=time.monotonic()-started,no_automatic_extension=True))
                return dict(completed=False,reason='process_deadline')
            pending=pending_evaluation(counters)
            if pending is not None:
                before_path=out/f'{pending}_before_evaluation.pt'
                atomic_torch(before_path,snapshot(net,opt,streams,cfg,initialization,counters,provenance))
                variants=evaluator.FULL_VARIANTS if pending in ('baseline','block3') else evaluator.INTERIM_VARIANTS
                result=checked_evaluation(net,cfg,opt,streams,out/f'{pending}_evaluation',
                    device=device,variants=variants,manifest=manifest)
                record=dict(checkpoint_before_sha256=sha(before_path),summary_sha256=sha(out/f'{pending}_evaluation/summary.json'),
                    policy_optimizer_private_global_rng_unchanged=True,variants=list(variants),bank_name='memory',
                    updates=counters['updates'])
                counters['evaluation_records'][pending]=record
                atomic_json(out/f'{pending}_evaluation_integrity.json',record)
                atomic_torch(out/f'{pending}.pt',snapshot(net,opt,streams,cfg,initialization,counters,provenance))
                continue
            block=counters['updates']//UPDATES_PER_BLOCK
            predraw=streams.snapshot()
            failed_plan=streams.draw_training(cfg['n_envs'],cfg,delay_cap=DELAY_CAPS[block])
            metrics,_=update(net,cfg,opt,source,failed_plan,device=device)
            advance_counters(counters,metrics,cfg)
            entry=dict(update=counters['updates'],block=block+1,delay_cap=DELAY_CAPS[block],
                seconds=time.monotonic()-started,**metrics)
            with(out/'metrics.jsonl').open('a')as stream:stream.write(json.dumps(entry)+'\n')
            if counters['updates']%128==0:
                atomic_torch(out/'latest.pt',snapshot(net,opt,streams,cfg,initialization,counters,provenance))
            if counters['updates']%32==0:print(json.dumps(entry),flush=True)
            failed_plan=predraw=None
        assert set(counters['evaluation_records'])==set(ENDPOINTS)
        for name in ENDPOINTS:
            assert(out/f'{name}.pt').exists() and(out/f'{name}_evaluation/summary.json').exists()
        shutil.copy2(out/'block3.pt',out/'final.pt')
        summary=dict(completed=True,condition=cfg['cond'],protocol_sha256=provenance['protocol_sha256'],
            counters=counters,seconds=time.monotonic()-started,final_checkpoint_sha256=sha(out/'final.pt'),
            active_physics_frames=counters['active_physics_updates']*cfg['action_repeat'],
            active_recurrent_updates=counters['active_recurrent_updates'],
            all_fixed_endpoints_preserved=True,no_endpoint_selection=True)
        atomic_json(out/'summary.json',summary);return summary
    except Exception as error:
        if failed_plan is not None:atomic_torch(out/'failed_plan.pt',failed_plan)
        if predraw is not None:atomic_torch(out/'failed_predraw_streams.pt',predraw)
        atomic_json(out/'incomplete.json',dict(reason='exception',error_type=type(error).__name__,error=str(error),
            counters=counters,seconds=time.monotonic()-started,last_saved_boundary_only_is_resumable=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('qualification','training'))
    parser.add_argument('--condition',choices=('gc100','pgc33'))
    parser.add_argument('--out',type=Path);parser.add_argument('--device',default='cuda')
    parser.add_argument('--manifest',type=Path,default=banks.BEHAVIOR)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--qualification-root',type=Path)
    parser.add_argument('--resume',type=Path);parser.add_argument('--max-seconds',type=int,default=MAX_SECONDS)
    parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);design=protocol(args.manifest)
    if args.freeze_only:
        assert not args.protocol.exists(),'Never overwrite a frozen prospective protocol'
        atomic_json(args.protocol,dict(design,frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(json.dumps(dict(protocol=str(args.protocol),sha256=sha(args.protocol))));return
    frozen=json.loads(args.protocol.read_text());assert frozen.pop('frozen_utc')
    assert frozen==design,'Source, settings or assets changed after prospective freeze'
    assert args.phase and args.condition and args.out
    assert 0<args.max_seconds<=MAX_SECONDS
    assert not args.out.exists() or not any(args.out.iterdir()),'Use an empty child output; preserve earlier attempts'
    gate=None
    if args.phase=='training':
        assert args.qualification_root is not None
        gate=qualification_gate(args.qualification_root,sha(args.protocol))
    else:assert args.resume is None
    net,cfg,initialization=load_initial(args.condition,args.device)
    torch.manual_seed(TRAIN_SEED)
    provenance=dict(phase=args.phase,condition=args.condition,original_seed=1,protocol_sha256=sha(args.protocol),
        design=design,device=str(args.device),torch_version=str(torch.__version__),threads=torch.get_num_threads(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        qualification_gate=gate)
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/'protocol.json',provenance)
    atomic_json(args.out/'initialization.json',initialization)
    atomic_json(args.out/'training_config.json',cfg)
    if gate is not None:atomic_json(args.out/'qualification_gate.json',gate)
    if args.phase=='qualification':
        result=run_qualification(net,cfg,initialization,provenance,args.out,device=args.device,manifest=args.manifest)
    else:
        result=run_training(net,cfg,initialization,provenance,args.out,device=args.device,manifest=args.manifest,
                            resume=args.resume,max_seconds=args.max_seconds)
    print(json.dumps(result),flush=True)
    if args.phase=='training' and not result['completed']:raise SystemExit(3)


if __name__=='__main__':main()
