"""Frozen, read-only short rollouts from saved reorientation runtime states.

CUDA generator blobs cannot be restored into CPU generators. Preserve the exact
saved non-RNG runtime and original generator blobs; substitute declared private
CPU streams. This is a fresh rollout from a saved state, not CUDA future replay.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population,CodeSource
from rl_goal_memory import state_digest
import rl_blind_reorientation as task

OUT=ROOT/'results/rl_blind_reorientation_review/entropy_diagnostic'
WORKERS={'gc100':'ring-20261105-reorient-grid-01','pgc33':'ring-20261105-reorient-pgc-02'}
DELAYS=(4,15,30,60)
SEEDS={'environment_rng':199201,'action_rng':199202,'noise_rng':199203}
ACTION_NAMES=('left','right','forward','backward','strafe_left','strafe_right')
SOURCES=('scripts/lstm_reorientation_entropy_diagnostic.py','scripts/rl_blind_reorientation.py',
    'scripts/rl_visual_memory_noise.py','scripts/rl_goal_memory.py','code/common.py','code/codes.py','code/env.py','code/nets.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def cpu_snapshot(saved):
    cloned=copy.deepcopy(saved)
    for key,seed in SEEDS.items():cloned[key]=torch.Generator(device='cpu').manual_seed(seed).get_state()
    assert tree_digest({k:v for k,v in saved.items() if k not in SEEDS})==tree_digest({k:v for k,v in cloned.items() if k not in SEEDS})
    return cloned


def distribution_summary(logits,actions,mask):
    ll=logits[mask].double();aa=actions[mask];n=len(aa)
    if not n:return dict(updates=0,entropy_nats=None,mean_action_probability=None,actual_action_counts=[0]*6,
        turn_probability=None,translation_probability=None,retained_zero_exposure=True)
    logp=ll.log_softmax(-1);p=logp.exp();entropy=-(p*logp).sum(-1)
    return dict(updates=n,entropy_nats=dict(mean=float(entropy.mean()),median=float(entropy.median()),
        min=float(entropy.min()),max=float(entropy.max())),
        maximum_entropy_nats=float(np.log(6)),effective_actions_from_mean_entropy=float(entropy.mean().exp()),
        mean_action_probability=p.mean(0).tolist(),actual_action_counts=torch.bincount(aa,minlength=6).tolist(),
        turn_probability=float(p[:,:2].sum(-1).mean()),translation_probability=float(p[:,2:].sum(-1).mean()),
        maximum_action_probability_mean=float(p.max(-1).values.mean()),
        probability_above99percent_fraction=float((p.max(-1).values>=.99).double().mean()),
        slots_with_exposure=int(mask.any(0).sum()))


def summarize_rollout(roll,before,after):
    logits=torch.stack(roll['actor_logits']);actions=torch.stack(roll['a']);phase=torch.stack(roll['phase'])
    rewards=torch.stack(roll['r']);policy=torch.stack(roll['policy_mask']).bool()
    assert phase.shape==(160,32) and logits.shape==(160,32,6)
    perphase={name:distribution_summary(logits,actions,phase==i) for i,name in enumerate(task.PHASE_NAMES)}
    free=phase==task.FREE
    hits=int((free & (rewards>0)).sum())
    counter_hits=int((after['input_audit']['free_goal_hits']-before['input_audit']['free_goal_hits']).sum())
    assert hits==counter_hits
    delta=(after['clock']['phase_counts']-before['clock']['phase_counts']).sum(0)
    assert delta.tolist()==[perphase[p]['updates'] for p in task.PHASE_NAMES]
    entropy=torch.distributions.Categorical(logits=logits).entropy()
    return dict(phases=perphase,all_policy_controlled=distribution_summary(logits,actions,policy),
        free_goal_hits=hits,free_goals_per_100_free_updates=100*hits/int(free.sum()) if free.any() else None,
        free_slots=int(free.any(0).sum()),total_slots=32,free_update_fraction=float(free.float().mean()),
        logged_entropy_equivalent=float((entropy*policy).mean()),
        logged_policy_fraction_equivalent=float(policy.float().mean()),
        schedule_delta={k:int((after['clock_stats'][k]-before['clock_stats'][k]).sum()) for k in before['clock_stats']})


def checkpoint_specs():
    return [('incoming','incoming.pt',4,0,0),('after_warmup','after_warmup.pt',4,0,200)]+[
        (f'stage{s}',f'stage{s}_delay{delay}.pt',delay,s*10_014_720,200+s*489) for s,delay in enumerate(DELAYS,1)]


def diagnose(condition,path,delay,frames,updates,out):
    checksum=sha(path);result_path=out/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['checkpoint_sha256']==checksum and result['source_sha256']==sha(__file__)
        return result
    data=torch.load(path,map_location='cpu',weights_only=True);cfg=data['config']
    assert data['extra_frames']==frames and data['total_updates']==updates and data['original_frames']==1_000_000_000
    assert cfg['cond']==condition and cfg['entropy']==8e-5 and (cfg['n_envs'],cfg['rollout'])==(32,160)
    expected=json.loads((ROOT/'docs/research/rl_blind_reorientation_protocol.json').read_text());expected.pop('frozen_at')
    assert data['provenance']==expected
    saved=data['runtime_snapshot'];saved_digest=tree_digest(saved);private=cpu_snapshot(saved)
    old_numpy=np.random.get_state();old_torch=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        net=build_policy(cfg,'cpu').train().requires_grad_(False);net.load_state_dict(data['policy'])
        weights_before=state_digest(net)
        if frames==0:
            original=torch.load(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v,original['policy'][k]) for k,v in net.state_dict().items())
            del original
        runtime=task.fresh_runtime(net,cfg,'cpu')
        task.restore_runtime(runtime,private,'cpu')
        before=task.capture_runtime(runtime,'cpu')
        assert tree_digest(before)==tree_digest(private),'Copied non-RNG runtime or chosen private RNG was not restored exactly'
        source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
        roll,_,_,_=task.collect(net,cfg,source,runtime,delay)
        after=task.capture_runtime(runtime,'cpu')
        summary=summarize_rollout(roll,before,after)
        assert state_digest(net)==weights_before
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'short_rollout.npz',actor_logits=torch.stack(roll['actor_logits']).numpy(),
            actual_actions=torch.stack(roll['a']).numpy(),phase=torch.stack(roll['phase']).numpy(),
            clipped_rewards=torch.stack(roll['r']).numpy(),done=torch.stack(roll['d']).numpy(),
            policy_mask=torch.stack(roll['policy_mask']).numpy(),
            **{f'original_cuda_{k}':saved[k].numpy() for k in SEEDS},
            **{f'initial_cpu_{k}':private[k].numpy() for k in SEEDS})
    assert torch.equal(torch.get_rng_state(),old_torch)
    assert tree_digest(np.random.get_state())==tree_digest(old_numpy)
    assert tree_digest(saved)==saved_digest and sha(path)==checksum
    result=dict(condition=condition,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=checksum,
        source_sha256=sha(__file__),max_training_delay=delay,extra_frames=frames,total_training_updates=updates,
        entropy_coefficient=cfg['entropy'],checkpoint_and_policy_unchanged=True,optimizer_constructed=False,
        saved_runtime_digest=saved_digest,private_runtime_start_digest=tree_digest(private),
        exact_non_rng_runtime_restoration=True,private_cpu_seeds=SEEDS,
        original_cuda_rng_hex={k:bytes(saved[k].tolist()).hex() for k in SEEDS},
        interpretation='Fresh160-update CPU rollout from an exact saved non-RNG runtime, with declared private CPU RNG substitution; not CUDA future replay; one nested diagnostic sample per fixed checkpoint',
        recurrent_noise='Unchanged task.collect: sigma.02 on blind updates, sampled from private CPU noise stream',**summary)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(condition=condition,checkpoint=path.name,free_updates=result['phases']['free_blind']['updates'])),flush=True)
    return result


def logs():
    result={}
    for condition,worker in WORKERS.items():
        path=ROOT/'results/ring_cloud_artifacts'/worker/'metrics.jsonl'
        raw=path.read_bytes();lines=raw.splitlines(keepends=True)
        complete=[line for line in lines if line.endswith(b'\n')]
        rows=[json.loads(line) for line in complete]
        selected=[{k:r[k] for k in ('update','warmup','stage','max_delay','learning_frames','entropy','free_policy_fraction')} for r in rows]
        for r in selected:r['entropy_conditional_on_policy_controlled']=r['entropy']/r['free_policy_fraction'] if r['free_policy_fraction'] else None
        groups={}
        for label,predicate in [('warmup',lambda r:r['warmup'])]+[(f'stage{s}',lambda r,s=s:not r['warmup'] and r['stage']==s) for s in range(1,5)]:
            rr=[r for r in selected if predicate(r)]
            if not rr:continue
            values=np.array([r['entropy'] for r in rr]);conditional=np.array([r['entropy_conditional_on_policy_controlled'] for r in rr])
            groups[label]=dict(logged_rollouts=len(rr),first_update=rr[0]['update'],last_update=rr[-1]['update'],
                masked_per_all_step_entropy_nats=dict(mean=float(values.mean()),min=float(values.min()),max=float(values.max())),
                policy_controlled_entropy_nats_mean=float(conditional.mean()))
        result[condition]=dict(path=str(path.relative_to(ROOT)),captured_file_sha256=hashlib.sha256(raw).hexdigest(),
            incomplete_final_lines_retained_as_count=len(lines)-len(complete),groups=groups,rows=selected)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-only',action='store_true');p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='saved_state_free_action_entropy_diagnostic_v1',frozen_before_first_checkpoint_rollout=True,
        scope='No optimizer, no learning, no VM; one160-update rollout per every fixed incoming/after-warmup/stage checkpoint in both originalseed1 populations; latest checkpoints excluded',
        checkpoint_stage_delays=[list(spec) for spec in checkpoint_specs()],private_cpu_seeds=SEEDS,
        rng='Preserve all original CUDA RNG blobs; restore all other runtime fields byte-exact, substitute private CPU streams because CUDA16byte generator state is incompatible with CPU5056byte state',
        metrics='FREE entropy in nats, six actor probabilities/actual action counts, turn versus translation, free reward and phase/event exposure; all zero-FREE cases retained; per-phase and all-policy-controlled comparisons',
        logs='All complete saved log rows; entropy field is policy-masked and divided by160, not FREE-specific; conditional policy entropy recovered by dividing policy-controlled fraction',
        replication='Single saved-state future rollout per checkpoint, one original trained-seed pair; sampled steps are not independent model replications',
        source_sha256={s:sha(ROOT/s) for s in SOURCES})
    manifest=OUT/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==protocol,'Frozen diagnostic changed'
    else:manifest.write_text(json.dumps(protocol,indent=2)+'\n')
    if args.freeze_only:return
    rows=[];missing={}
    for condition,worker in WORKERS.items():
        missing[condition]=[]
        for name,filename,delay,frames,updates in checkpoint_specs():
            checkpoint=ROOT/'results/ring_cloud_artifacts'/worker/filename
            if not checkpoint.exists():missing[condition].append(name);continue
            rows.append(dict(endpoint=name,**diagnose(condition,checkpoint,delay,frames,updates,OUT/condition/name)))
    if args.require_complete:assert not any(missing.values()),missing
    summary=dict(protocol_sha256=sha(manifest),action_names=ACTION_NAMES,missing=missing,checkpoints=rows,training_logs=logs())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
