"""Bounded runner; execution requires the separately reviewed frozen plan."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import native_turn_distillation as train
import evaluate_native_turn_distillation as evaluate
import native_turn_original_memory as fixed
from rl_goal_memory import atomic_json,atomic_torch,state_digest

SOURCES=('scripts/native_turn_distillation.py','scripts/evaluate_native_turn_distillation.py',
    'scripts/run_native_turn_distillation.py','tests/test_native_turn_distillation.py')
ACQUISITION_SECONDS=1800
EVALUATION_SECONDS=600


def validate(design):
    assert design['updates']==180 and design['acquisition_seconds']==1800 and design['evaluation_seconds']==600
    assert design['arms']==list(train.ARMS)and design['suffix_ranges']==[list(v)for v in train.SUFFIX_RANGES]
    for key in ('source_sha256','asset_sha256'):
        for name,digest in design[key].items():assert train.dist.sha(ROOT/name)==digest,name


def freeze():
    assert not train.PROTOCOL.exists()
    prior=json.loads(fixed.PROTOCOL.read_text());fixed.validate_frozen(prior)
    old=json.loads(train.dist.PROTOCOL.read_text())
    reorientation=json.loads((ROOT/'docs/research/rl_blind_reorientation_protocol.json').read_text())
    sources={**prior['source_sha256'],**old['source_sha256'],**reorientation['source_sha256'],
        **{name:train.dist.sha(ROOT/name)for name in SOURCES}}
    assets={**prior['asset_sha256'],str(fixed.PROTOCOL.relative_to(ROOT)):train.dist.sha(fixed.PROTOCOL),
        str(train.CALIBRATION.relative_to(ROOT)):train.dist.sha(train.CALIBRATION)}
    for condition in ('gc100','pgc33'):
        p=fixed.BASE/condition/'summary.json';result=json.loads(p.read_text())
        assert result['complete']and result['protocol_sha256']==train.dist.sha(fixed.PROTOCOL)
        assets[str(p.relative_to(ROOT))]=train.dist.sha(p)
    protocol=dict(version='native_turn_recurrent_distillation_v1',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        conditions=['gc100','pgc33'],arms=list(train.ARMS),model_seed=1,train_seed=train.TRAIN_SEED,eval_seed=train.EVAL_SEED,
        n_envs=train.N,unroll=train.UNROLL,updates=train.UPDATES,block_updates=train.BLOCK_UPDATES,
        extra_frames=train.UPDATES*train.N*train.UNROLL*4,suffix_ranges=[list(v)for v in train.SUFFIX_RANGES],
        teacher_mixture=list(train.BETAS),choice_updates=3,max_whole_event_updates=57,max_events_per_episode=8,
        cue='One actual forward-probe endpoint current code, native/ordinary fair per-event coin; then held ordinary endpoint code. Goal stays native.',
        loss='0.5 mean teacher-forward-KL on free choices plus0.5 mean on other policy-controlled decisions. If no choices, control mean alone. Forced decisions have zero direct loss and retain BPTT.',
        optimizer=dict(name='Adam',lr=1e-4,grad_clip=1.,trainable='Only four existing LSTM tensors'),
        initialization='Exact original1B checkpoint as teacher/student including original CNN,actor,critic; fixed teacher/CNN/actor/critic/code.',
        noise=dict(arm='noise',distribution='Uniform attempted impulses with each block standard deviation0.1 times its original native one-update deltaRMS',
            scale_by_condition={c:dict(zip(('h','c'),train.fixed_sigmas(c)))for c in ('gc100','pgc33')},
            calibration='All archived post-discovery native census updates; unsupervised existing-state scale calibration, not held-out efficacy evidence',
            timing='Incoming h/c before native-or-ordinary cue input and before second ordinary post-cue input',
            bounds='Clip attempted h to[-1,1]; c additive. Realized h is not necessarily zero-mean. Retain attempted/realized impulses and clipping counts.',
            matching='Independent private per-slot/episode/event coins+tapes. Clean arm draws same attempted arrays but does not apply them. No complete erasure training.'),
        live_variants=list(evaluate.LIVE_VARIANTS),live_episodes_per_variant=16,
        heldout='All fixed original preflight paired histories, own-checkpoint native prefixes, all8 interventions; six modified-event live branches. Evaluation-only.',
        geometry='All four fixed finals plus originals, no behavioral gate. Controlled actual native-turn history family with byte-identical input banks, own-checkpoint prefix states; not asserted to be on-policy training trajectories.',
        acquisition_seconds=ACQUISITION_SECONDS,evaluation_seconds=EVALUATION_SECONDS,
        cap_scope='1800 monotonic seconds for acquisition only;600s separately for final read-only evaluation. Shared-original evaluation and geometry separately timed. No optimization extension.',
        endpoints=['initial.pt','stage1.pt','stage2.pt','stage3.pt','final.pt'],no_cloud=True,no_automatic_retry=True,
        source_sha256=sources,asset_sha256=assets)
    validate(protocol);atomic_json(train.PROTOCOL,protocol);print(train.dist.sha(train.PROTOCOL))


@torch.no_grad()
def baseline(condition):
    design=json.loads(train.PROTOCOL.read_text());validate(design)
    folder=train.BASE/condition/'original_baseline';assert not folder.exists()or not any(folder.iterdir());folder.mkdir(parents=True,exist_ok=True)
    torch.random.default_generator.manual_seed(train.TRAIN_SEED)
    teacher,student,cfg=train.dist.load_models(condition);digest=state_digest(student);rng=torch.get_rng_state().clone()
    start=time.monotonic();result=evaluate.live(student,teacher,cfg,start+EVALUATION_SECONDS)
    assert state_digest(student)==state_digest(teacher)==digest and torch.equal(rng,torch.get_rng_state())
    atomic_json(folder/'live.json',result);validate(design)
    p=fixed.BASE/condition/'summary.json'
    atomic_json(folder/'summary.json',dict(complete=True,condition=condition,policy_hash=digest,
        protocol_sha256=train.dist.sha(train.PROTOCOL),source_unchanged=True,policy_unchanged=True,
        shared_across_arms=True,no_optimization=True,seconds=time.monotonic()-start,
        paired_original_summary_path=str(p.relative_to(ROOT)),paired_original_summary_sha256=train.dist.sha(p),
        live_sha256=train.dist.sha(folder/'live.json')))


def baseline_record(condition,teacher):
    folder=train.BASE/condition/'original_baseline';p=folder/'summary.json';r=json.loads(p.read_text())
    assert r['complete']and r['condition']==condition and r['policy_hash']==state_digest(teacher)
    assert r['protocol_sha256']==train.dist.sha(train.PROTOCOL)and r['live_sha256']==train.dist.sha(folder/'live.json')
    live=json.loads((folder/'live.json').read_text());assert live['complete']and live['episodes_per_variant']==16
    assert set(live['variants'])==set(evaluate.LIVE_VARIANTS)
    return dict(path=str(p.relative_to(ROOT)),sha256=train.dist.sha(p),**r)


def run(condition,arm):
    design=json.loads(train.PROTOCOL.read_text());validate(design)
    out=train.BASE/condition/arm;assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    torch.random.default_generator.manual_seed(train.TRAIN_SEED)
    teacher,student,cfg=train.dist.load_models(condition);baseline_ref=baseline_record(condition,teacher)
    initial_hash=state_digest(teacher);fixed_hash=train.dist.frozen_digest(student)
    assert state_digest(student)==initial_hash
    opt=torch.optim.Adam(student.lstm.parameters(),lr=1e-4)
    started=time.monotonic();start_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    env,pop,runtime=train.make_runtime(teacher,cfg,arm,started+ACQUISITION_SECONDS)
    protocol_hash=train.dist.sha(train.PROTOCOL);updates=0;saved=False;changed=False
    atomic_json(out/'protocol.json',design);atomic_json(out/'baseline_reference.json',baseline_ref)
    atomic_torch(out/'initial.pt',train.snapshot(student,teacher,cfg,opt,env,runtime,0,protocol_hash))
    try:
        with(out/'metrics.jsonl').open('w')as log:
            for index in range(train.UPDATES):
                stage=index//train.BLOCK_UPDATES
                metrics,probe=train.train_unroll(student,teacher,pop,cfg,env,runtime,opt,stage)
                updates=index+1;changed|=metrics['parameter_update_l2']>0
                row=dict(update=updates,stage=stage+1,arm=arm,condition=condition,
                    extra_frames=updates*train.N*train.UNROLL*cfg['action_repeat'],seconds=time.monotonic()-started,**metrics)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if not saved and metrics['choice_count']:
                    atomic_torch(out/'first_choice_unroll.pt',dict(update=updates,**probe));saved=True
                if updates%10==0:print(json.dumps(row),flush=True)
                if updates%train.BLOCK_UPDATES==0:
                    assert train.dist.frozen_digest(student)==fixed_hash and state_digest(teacher)==initial_hash
                    atomic_torch(out/f'stage{stage+1}.pt',train.snapshot(student,teacher,cfg,opt,env,runtime,updates,protocol_hash))
            assert updates==180 and saved and changed
            train.dist.check_time(runtime['deadline'])
            acquisition_seconds=time.monotonic()-started
        assert train.dist.frozen_digest(student)==fixed_hash and state_digest(teacher)==initial_hash
        atomic_torch(out/'final.pt',train.snapshot(student,teacher,cfg,opt,env,runtime,updates,protocol_hash))
        atomic_json(out/'events.json',runtime['event_rows'])
        evaluation_started=time.monotonic();deadline=evaluation_started+EVALUATION_SECONDS
        rng=torch.get_rng_state().clone();final_hash=state_digest(student)
        evaluate.paired(student,condition,out/'final_paired',deadline)
        atomic_json(out/'final_live.json',evaluate.live(student,teacher,cfg,deadline))
        train.dist.check_time(deadline)
        assert torch.equal(rng,torch.get_rng_state())and state_digest(student)==final_hash
        assert state_digest(teacher)==initial_hash and train.dist.frozen_digest(student)==fixed_hash
        validate(design)
        atomic_json(out/'summary.json',dict(complete=True,condition=condition,arm=arm,updates=updates,
            extra_frames=updates*train.N*train.UNROLL*4,policy_decisions=updates*train.N*train.UNROLL,
            protocol_sha256=protocol_hash,initial_policy_hash=initial_hash,final_policy_hash=final_hash,
            original_checkpoint_sha256=train.dist.sha(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'),
            original_config_sha256=train.dist.sha(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json'),
            fixed_tensors_unchanged=True,teacher_unchanged=True,frozen_tensor_hash=fixed_hash,
            start_utc=start_utc,end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            acquisition_monotonic_seconds=acquisition_seconds,evaluation_monotonic_seconds=time.monotonic()-evaluation_started,
            acquisition_cap_seconds=ACQUISITION_SECONDS,evaluation_cap_seconds=EVALUATION_SECONDS,
            no_cloud=True,cost_usd=0,artifacts={str(p.relative_to(out)):dict(sha256=train.dist.sha(p),bytes=p.stat().st_size)
                for p in out.rglob('*')if p.is_file()}))
    except Exception as error:
        atomic_json(out/'incomplete.json',dict(complete=False,condition=condition,arm=arm,updates=updates,
            seconds=time.monotonic()-started,error_type=type(error).__name__,error=str(error),no_retry=True));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--baseline',action='store_true')
    p.add_argument('--condition',choices=('gc100','pgc33'));p.add_argument('--arm',choices=train.ARMS)
    a=p.parse_args();torch.set_num_threads(2)
    if a.freeze:freeze();return
    assert a.condition
    if a.baseline:baseline(a.condition);return
    assert a.arm;run(a.condition,a.arm)
if __name__=='__main__':main()
