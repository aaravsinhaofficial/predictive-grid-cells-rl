"""Fixed held-out evaluations; never optimization targets."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import native_turn_distillation as train
import native_turn_original_memory as fixed
from rl_goal_memory import atomic_json,atomic_torch,state_digest,reset_state,episode_summary

LIVE_VARIANTS=('seen_native','seen_reset_every_choice','blind_native','blind_ordinary',
    'blind_reset_every_choice','blind_ordinary_reset_every_choice')


def live_controls(native,x,phase,n):
    answer=x.clone()
    seen=torch.tensor([v.startswith('seen')for v in LIVE_VARIANTS]).repeat_interleave(n)
    answer[seen,:256]=native[seen,:256]
    reset=torch.tensor(['reset'in v for v in LIVE_VARIANTS]).repeat_interleave(n)&(phase==train.event.FREE)
    assert torch.equal(answer[:,256:],x[:,256:])
    return answer,reset


@torch.no_grad()
def paired(student,condition,out,deadline):
    """All fixed input histories; student's own full prefix state, no cache reuse."""
    design=json.loads(fixed.PROTOCOL.read_text());fixed.validate_frozen(design)
    parent=json.loads(fixed.parent.PROTOCOL.read_text());bankrec=parent['banks'][condition]
    bank=torch.load(ROOT/bankrec['path'],map_location='cpu',weights_only=True)
    assert fixed.parent.tensor_inventory(bank)==bankrec['tensors']
    data=torch.load(ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt',map_location='cpu',weights_only=True)
    states,prefix=fixed.replay_own_prefixes(student,bank,data,False);del data
    out=Path(out);assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'prefix_replay.json',prefix)
    pop=train.build_population(bank['config'],'cpu');before=state_digest(student);rng=torch.get_rng_state().clone()
    rows=[];pairs=[];inputs=[];targets=[];refs=[];predictions={v:[]for v in fixed.VARIANTS}
    for context in bank['contexts']:
        for kind in fixed.parent.KINDS:
            train.dist.check_time(deadline)
            name=f"context{context['context_id']:02d}_{kind}";folder=fixed.parent.BASE/condition
            raw=torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=True)
            reference=json.loads((folder/f'{name}.json').read_text())
            saved,result=fixed.evaluate_context(student,pop,raw,reference,context['context_id'],kind,states[context['context_id']])
            result.update(source_raw_sha256=fixed.parent.sha(folder/f'{name}.pt'),source_reference_sha256=fixed.parent.sha(folder/f'{name}.json'))
            atomic_torch(out/f'{name}.pt',saved);atomic_json(out/f'{name}.json',result)
            rows.extend(result['rows']);pairs.extend(result['pairs'])
            inputs.append(raw['student_native_x'][-1]);targets.append(raw['teacher_first_choice_probabilities'])
            refs.extend(dict(context=context['context_id'],kind=kind,case=i,pair=i//2,member=i%2)for i in range(16))
            for v in fixed.VARIANTS:predictions[v].append(saved['variants'][v]['first_choice_probabilities'])
    inputs,targets=torch.cat(inputs),torch.cat(targets);predictions={k:torch.cat(v)for k,v in predictions.items()}
    groups={'all_modes':fixed.grouped_comparisons(inputs,targets,refs,predictions)}
    for kind in fixed.parent.KINDS:
        ix=[i for i,r in enumerate(refs)if r['kind']==kind]
        groups[kind]=fixed.grouped_comparisons(inputs[ix],targets[ix],[refs[i]for i in ix],{v:p[ix]for v,p in predictions.items()})
    assert len(rows)==4096 and len(pairs)==2048
    assert state_digest(student)==before and torch.equal(rng,torch.get_rng_state())
    result=dict(complete=True,condition=condition,policy_hash=before,variants=list(fixed.VARIANTS),
        original_memory_protocol_sha256=fixed.parent.sha(fixed.PROTOCOL),own_prefix_recomputed=True,
        rows=rows,pairs=pairs,exact_input_groups=groups,policy_unchanged=True,global_cpu_rng_preserved=True,
        artifacts={p.name:dict(sha256=fixed.parent.sha(p),bytes=p.stat().st_size)for p in out.iterdir()if p.is_file()})
    atomic_json(out/'summary.json',result);return result


@torch.inference_mode()
def live(student,teacher,cfg,deadline):
    """Shared event rules and native noise; erasure clock can diverge after actions."""
    n=train.N;groups=len(LIVE_VARIANTS);total=n*groups
    env=train.MatchedArena(cfg,n,groups,train.EVAL_SEED,'cpu');obs=env.reset_all();pop=train.build_population(cfg,'cpu')
    clock=train.TurnClock(n,'cpu',train.EVAL_SEED+2,groups=groups)
    inputs=train.event.EventInputs(total,512,'cpu',variants=LIVE_VARIANTS,n_base=n)
    inputs.ordinary=torch.tensor(['ordinary'in v for v in LIVE_VARIANTS]).repeat_interleave(n)
    goal=train.NativeGoal(total,512,'cpu');pa=torch.full((total,),-1,dtype=torch.long);pr=torch.zeros(total)
    state=student.zero_state(total,'cpu');teacher_state=teacher.zero_state(total,'cpu')
    rng=torch.Generator().manual_seed(train.EVAL_SEED+1)
    runtime=dict(clock=clock,inputs=inputs,source=train.CodeSource(cfg,pop,'cpu'),
        black=teacher.embed(torch.zeros(n,3,84,84)).repeat(groups,1),
        choice_brake_count=torch.zeros(total,dtype=torch.long),choice_brake_before=torch.zeros(total,3),
        choice_brake_seen=torch.zeros(total,dtype=torch.bool))
    rows={v:[]for v in LIVE_VARIANTS};sums=torch.zeros(groups,3,dtype=torch.float64);counts=torch.zeros(groups,3,dtype=torch.long)
    rewards=torch.zeros(groups,len(train.event.PHASE_NAMES),dtype=torch.float64)
    erasures=torch.zeros(groups,dtype=torch.long)
    for t in range(cfg['episode_len']):
        if t%30==0:train.dist.check_time(deadline)
        if t%train.UNROLL==0:clock.begin_segment()
        admit=clock.prepare(train.UNROLL-t%train.UNROLL,cfg['episode_len']-env.t,train.SUFFIX_RANGES[-1])
        runtime['choice_brake_seen'][admit]=False;runtime['choice_brake_before'][admit]=0
        native,x,blind,glimpse,first=train.make_inputs(teacher,pop,cfg,env,runtime,obs,goal,pa,pr)
        # Visibility alone differs: current-code holding/cue schedules are shared.
        x,reset=live_controls(native,x,clock.phase,n)
        state=reset_state(state,reset);erasures+=reset.view(groups,n).sum(1)
        logits,state=train.dist.chunk_forward(student,x,state,n)
        target,teacher_state=train.dist.chunk_forward(teacher,native,teacher_state,n)
        loss=train.dist.kl(target,logits).reshape(groups,n).double()
        masks=(clock.phase==train.event.FREE,~clock.forced()&(clock.phase!=train.event.FREE),clock.forced())
        for j,mask in enumerate(masks):
            sums[:,j]+=(loss*mask.reshape(groups,n)).sum(1);counts[:,j]+=mask.reshape(groups,n).sum(1)
        uniforms=torch.rand(n,generator=rng)
        sampled=torch.cat([train.dist.sample(logits[i*n:(i+1)*n],uniforms)for i in range(groups)])
        action=clock.action(sampled);phase=clock.phase.clone()
        obs,reward,done,info=env.step(action)
        for j in range(len(train.event.PHASE_NAMES)):
            rewards[:,j]+=(reward*(phase==j)).reshape(groups,n).double().sum(1)
        inputs.transition(clock,info);train.dist.advance_goal(pop,goal,info,done,n);clock.advance(info['hit'],done)
        state=reset_state(state,done);teacher_state=reset_state(teacher_state,done)
        pa=torch.where(done,torch.full_like(action,-1),action);pr=reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])*(~done)
        stats=info['ep_stats']
        if stats is not None:
            for j,index in enumerate(stats['env'].tolist()):
                group,slot=divmod(index,n)
                rows[LIVE_VARIANTS[group]].append(dict(slot=slot,score=float(stats['ret'][j]),n_goals=int(stats['n_goals'][j]),
                    first_goal_step=int(stats['first_goal_t'][j]),path_eff=float(stats['path_eff'][j]),path_eff_n=int(stats['path_eff_n'][j])))
    assert all(len(v)==n and len({r['slot']for r in v})==n for v in rows.values())
    assert all([r['first_goal_step']for r in v]==[r['first_goal_step']for r in rows['seen_native']]for v in rows.values())
    if cfg['cond']=='gc100':
        assert rows['blind_native']==rows['blind_ordinary']
        assert rows['blind_reset_every_choice']==rows['blind_ordinary_reset_every_choice']
    return dict(complete=True,seed=train.EVAL_SEED,episodes_per_variant=n,variants={v:episode_summary(rr,cfg['episode_len'])for v,rr in rows.items()},
        exposure={v:dict(zip(('choice','other_control','forced'),counts[i].tolist()))for i,v in enumerate(LIVE_VARIANTS)},
        teacher_kl_sums={v:dict(zip(('choice','other_control','forced'),sums[i].tolist()))for i,v in enumerate(LIVE_VARIANTS)},
        mean_teacher_kl={v:{name:float(sums[i,j]/counts[i,j])if counts[i,j]>0 else None for j,name in enumerate(('choice','other_control','forced'))}for i,v in enumerate(LIVE_VARIANTS)},
        choice_erasure_counts=dict(zip(LIVE_VARIANTS,erasures.tolist())),reward_by_phase=dict(zip(LIVE_VARIANTS,rewards.tolist())),
        event_clock=clock.summary(),input_audit=inputs.summary(groups),
        grid_native_ordinary_sham_exact=cfg['cond']=='gc100',evaluation_state_noise=0,
        scope='Six event-rule-matched branches share held-code/cue scheduling, native noise/action uniforms and forced turns/brakes. Only declared RGB visibility, cue and choice erasure differ. Hit-dependent event times can diverge after behavior differs. This is modified-task control, not original native-performance evaluation. No evaluation optimization.')
