"""Isolated original-checkpoint smoke; all optimized copies are discarded.

Private synthetic known-goal starting context, never a study/evaluation bank.
Verifies actual Adam change, stored-input/noise replay and serialized resume.
"""
from pathlib import Path
import datetime as dt
import json
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import native_turn_distillation as study
from rl_goal_memory import atomic_json,atomic_torch,state_digest


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,type(a))and len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b


def one(condition):
    original=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt';source_sha=study.dist.sha(original)
    teacher,student,cfg=study.dist.load_models(condition)
    teacher_hash=state_digest(teacher);fixed=study.dist.frozen_digest(student)
    env,pop,runtime=study.make_runtime(teacher,cfg,'noise',float('inf'),seed=989201)
    # Explicit unit fixture: establish a synthetic known goal, central pose and
    # zero motors. This is not an actual study trial or claimed pickup history.
    env.pos.zero_();env.goal[:]=torch.tensor([.9,0.]);env.hd[:]=.1;env.mdir[:]=torch.tensor([1.,0.])
    study.event.brake(env,torch.ones(study.N,dtype=torch.bool));env.n_goals[:]=1
    runtime['goal'].saved=pop.activity(env.goal,env.mdir,torch.zeros_like(env.pos))
    runtime['clock'].known[:]=True;runtime['obs']=env.observe()
    opt=torch.optim.Adam(student.lstm.parameters(),lr=1e-4)
    first,probe=study.train_unroll(student,teacher,pop,cfg,env,runtime,opt,0)
    assert first['parameter_update_l2']>0 and first['grad_norm_before_clip']>0
    assert first['choice_count']==48 and first['replay_max_error']==first['teacher_replay_max_error']==0
    snapshot=study.snapshot(student,teacher,cfg,opt,env,runtime,1,'isolated_private_smoke_not_study')
    with tempfile.TemporaryDirectory(prefix='native-turn-discarded-smoke-')as tmp:
        path=Path(tmp)/'after_one_update.pt';atomic_torch(path,snapshot)
        checkpoint_sha=study.dist.sha(path);checkpoint_bytes=path.stat().st_size
        restored=torch.load(path,map_location='cpu',weights_only=True);assert equal(snapshot,restored)
    direct,probe_a=study.train_unroll(student,teacher,pop,cfg,env,runtime,opt,1)
    state_a=study.snapshot(student,teacher,cfg,opt,env,runtime,2,'isolated_private_smoke_not_study')
    resumed=study.dist.trainable_student(teacher);opt2=torch.optim.Adam(resumed.lstm.parameters(),lr=1e-4)
    env2,pop2,runtime2=study.restore(restored,resumed,teacher,cfg,opt2,float('inf'))
    replayed,probe_b=study.train_unroll(resumed,teacher,pop2,cfg,env2,runtime2,opt2,1)
    state_b=study.snapshot(resumed,teacher,cfg,opt2,env2,runtime2,2,'isolated_private_smoke_not_study')
    assert direct==replayed and equal(probe_a,probe_b)and equal(state_a,state_b)
    assert study.dist.frozen_digest(student)==study.dist.frozen_digest(resumed)==fixed
    assert state_digest(teacher)==teacher_hash and study.dist.sha(original)==source_sha
    return dict(complete=True,condition=condition,original_checkpoint_sha256=source_sha,original_policy_hash=teacher_hash,
        first_update=first,second_update=direct,exact_serialized_resume=True,
        exact_policy_optimizer_runtime_global_private_rng=True,exact_all_inputs_actions_noise_and_targets=True,
        frozen_cnn_actor_critic_unchanged=True,teacher_unchanged=True,original_file_unchanged=True,
        temporary_checkpoint_sha256=checkpoint_sha,temporary_checkpoint_bytes=checkpoint_bytes,temporary_checkpoint_deleted=True,
        optimized_copy_discarded=True,study_endpoint_created=False)


def main():
    torch.set_num_threads(2);torch.random.default_generator.manual_seed(989001)
    names=('scripts/native_turn_distillation.py','scripts/evaluate_native_turn_distillation.py',
        'scripts/run_native_turn_distillation.py','scripts/validate_native_turn_distillation.py','tests/test_native_turn_distillation.py')
    sources={name:study.dist.sha(ROOT/name)for name in names};start=time.monotonic()
    results={}
    for c in ('gc100','pgc33'):
        results[c]=one(c);print(json.dumps(dict(condition=c,complete=True,seconds=time.monotonic()-start)),flush=True)
    assert all(study.dist.sha(ROOT/n)==h for n,h in sources.items())
    folder=ROOT/'results/native_turn_distillation_design';folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'native_validation.json',dict(complete=True,recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        seconds=time.monotonic()-start,source_sha256=sources,models=results,
        scope='Two original full CNN/LSTM checkpoints, isolated private fixture seed989201. Two Adam updates plus exact replayed second update per population, all copies discarded. No study/holdout bank, endpoint, cloud, or study optimization.'))
if __name__=='__main__':main()
