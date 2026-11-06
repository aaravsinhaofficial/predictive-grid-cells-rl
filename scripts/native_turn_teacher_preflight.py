"""Draft teacher/input-information audit. No optimizer or training code.

The physical bank and descriptive definition must be reviewed and sealed before
--run. Physical integrity is required; there is no scientific pass/fail gate.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from code.common import build_env,build_policy,build_population
from code.env import ACTIONS,wrap_pi
from rl_goal_memory import atomic_json,atomic_torch,state_digest

BASE=ROOT/'results/native_turn_teacher_preflight'
PROTOCOL=ROOT/'docs/research/native_turn_teacher_preflight_protocol.json'
CONDITIONS=('gc100','pgc33')
CONTEXTS,PAIRS=16,8
VISIBLE,PREFIX,PROBE=15,14,4
SUFFIX=(0,)*8
CONTINUATION=30
ACTION_SEED=983901
KINDS=('turn_only','probe_glimpse')
BOOKKEEPING=('t','n_goals','first_goal_t','ep_return','travelled','path_eff_sum','path_eff_n')
SOURCES=('scripts/native_turn_teacher_preflight.py','tests/test_native_turn_teacher_preflight.py',
    'scripts/rl_goal_memory.py','scripts/lstm_code_swap.py','code/__init__.py',
    'code/common.py','code/codes.py','code/env.py','code/nets.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_inventory(value,prefix=''):
    answer={}
    if isinstance(value,torch.Tensor):return {prefix:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value))}
    if isinstance(value,dict):
        for k,v in value.items():answer.update(tensor_inventory(v,f'{prefix}.{k}'if prefix else str(k)))
    elif isinstance(value,list):
        for k,v in enumerate(value):answer.update(tensor_inventory(v,f'{prefix}.{k}'))
    return answer


def paired_equal(value):
    assert len(value)%2==0
    return torch.equal(value[::2],value[1::2])


def brake(env):
    before=torch.stack((env.vf,env.vs,env.vr),1).clone()
    for k in ('vf','vs','vr'):getattr(env,k).zero_()
    assert torch.count_nonzero(torch.stack((env.vf,env.vs,env.vr),1))==0
    return before


def native_motion(cfg,actions,column):
    """Exact scalar motor recurrence; checked against native ArenaEnv in tests."""
    step=cfg['action_repeat']/60.;k=1-math.exp(-step/.25)
    targets=((0,0,1),(0,0,-1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0))
    maximum=cfg['rot_max']if column==2 else cfg['v_max']
    speed=0.;displacement=0.
    for action in actions:
        speed+=(targets[action][column]*maximum-speed)*k
        displacement+=speed*step
    return displacement


def select_contexts(data,cfg):
    """One first eligible context per episode, then first16 episode indices.

No neural state, action, logit or later success participates in selection.
The full32-episode inventory and all static eligibility counts are retained.
"""
    pre=data['pre'];T,n=pre['n_goals'].shape
    assert n==32 and T==1350
    geometry=(pre['n_goals']>0)&((pre['pos']-pre['goal']).norm(dim=2)>cfg['goal_radius']+.10)
    geometry&=pre['pos'].abs().amax(2)<cfg['arena_half']-.08-.10
    required=(VISIBLE-1)+PREFIX+PROBE+len(SUFFIX)+CONTINUATION
    time_ok=cfg['episode_len']-pre['t']>=required
    eligible=geometry&time_ok
    eligible[:VISIBLE]=False
    inventory=[];selected=[]
    action_tape=torch.rand(CONTEXTS,CONTINUATION,PAIRS,generator=torch.Generator().manual_seed(ACTION_SEED))
    for slot in range(n):
        indices=eligible[:,slot].nonzero().flatten()
        first=int(indices[0])if len(indices)else None
        row=dict(slot=slot,known_updates=int((pre['n_goals'][:,slot]>0).sum()),geometry_eligible_updates=int(geometry[:,slot].sum()),
            eligible_updates=int(len(indices)),late_geometry_eligible_updates=int((geometry[:,slot]&~time_ok[:,slot]).sum()),
            required_remaining_physics_updates=required,first_eligible_step=first,selected=False)
        if first is not None and len(selected)<CONTEXTS:
            row['selected']=True;row['context_id']=len(selected)
            x=data['x'][first,slot].clone();assert bool(x[768:1280].abs().sum()>0)
            selected.append(dict(context_id=len(selected),slot=slot,step=first,
                endpoint=pre['pos'][first,slot].clone(),goal_center=pre['goal'][first,slot].clone(),
                direction=pre['mdir'][first,slot].clone(),heading=pre['hd'][first,slot].clone(),
                native_x=x,initial_hc=data['state_before'][first,slot].clone(),
                preceding_native_x=data['x'][:first,slot].clone(),
                preceding_done=data['done'][:first,slot].clone(),
                native_bookkeeping={k:pre[k][first,slot].clone()for k in BOOKKEEPING},
                continuation_uniform=action_tape[len(selected)].clone(),
                land_pos=data['world']['land_pos'][slot].clone(),stripe_phase=data['world']['stripe_phase'][slot].clone()))
        inventory.append(row)
    return dict(contexts=selected,inventory=inventory,complete=len(selected)==CONTEXTS,
        selection='First geometry-eligible post-discovery observation per episode, first16 eligible episode indices. No policy output or neural-geometry selection.')


def make_env(cfg,context,n):
    env=build_env(dict(cfg,img_noise=0.),n,983201,'cpu',stagger=False)
    env.reset_all();env.noise_v=env.noise_r=0.
    env.goal[:]=context['goal_center'];env.land_pos[:]=context['land_pos'];env.stripe_phase[:]=context['stripe_phase']
    env.pos[:]=context['endpoint'];env.spawn[:]=env.pos;env.mdir[:]=context['direction'];brake(env)
    for k in BOOKKEEPING:getattr(env,k)[:]=context['native_bookkeeping'][k]
    return env


def clean_step(env,actions):
    obs,reward,done,info=env.step(actions)
    assert not bool(done.any()or info['hit'].any()or info['hit_wall'].any()or reward.any()),'Forced path contacted goal/wall or ended'
    return obs


@torch.no_grad()
def construct_history(net,pop,cfg,context,kind):
    """16 genuine forced histories, eight matched-input antipodal pairs.

Inputs are recorded after native transitions, with true preceding actions.
The final recorded observation is the first free-decision input. The final
brake changes motor state only, before that input; the teacher always sees RGB.
"""
    assert kind in KINDS
    n=2*PAIRS;env=make_env(cfg,context,n)
    bearing=torch.atan2(context['goal_center'][1]-context['endpoint'][1],context['goal_center'][0]-context['endpoint'][0])
    first=bearing+torch.arange(PAIRS,dtype=torch.float32)*(math.pi/PAIRS)
    target_heading=torch.stack((first,first+math.pi),1).reshape(-1)
    prefix_actions=torch.tensor([0,1]).repeat(PAIRS)
    visible_delta=native_motion(cfg,[0]*(VISIBLE-1),2)
    prefix_delta=native_motion(cfg,[0]*PREFIX,2)*torch.where(prefix_actions==0,1.,-1.)
    suffix_delta=native_motion(cfg,SUFFIX,2)
    env.hd[:]=wrap_pi(target_heading-visible_delta-prefix_delta-suffix_delta)
    probe_heading=target_heading-suffix_delta
    distance=native_motion(cfg,[2]*PROBE,0)if kind=='probe_glimpse'else 0.
    env.pos[:]=context['endpoint']-distance*torch.stack((probe_heading.cos(),probe_heading.sin()),1)
    env.spawn[:]=env.pos;start_pos=env.pos.clone();brakes={'start':brake(env)}
    goal=context['native_x'][768:1280].expand(n,-1).clone()
    pa0=context['native_x'][1280:1286];pa0=int(pa0.argmax())if pa0.sum()else-1
    pr0=float(context['native_x'][-1]);black=net.embed(torch.zeros(n,3,84,84))
    student=[];ordinary=[];teacher=[];physical=[];labels=[];executed=[];rgb_hashes=[]
    obs=env.observe();held=None;cue_index=None;cue_error=0.;probe_start=None
    def append(obs,pa,pr,label,visible,cue=False):
        native=pop.activity(obs['pos'],obs['dir'],obs['vel'])
        position=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
        emb=net.embed(obs['rgb']);pa=torch.as_tensor(pa,dtype=torch.long).expand(n);pr=torch.as_tensor(pr,dtype=torch.float32).expand(n)
        tx=net._concat(emb,native,goal,pa,pr,None)
        sx=net._concat(emb if visible else black,native if visible or cue else held,goal,pa,pr,None)
        ox=sx.clone()
        if cue:ox[:,256:768]=position
        teacher.append(tx);student.append(sx);ordinary.append(ox);labels.append(label);executed.append(pa.clone())
        physical.append({**{k:getattr(env,k).clone()for k in ('pos','hd','mdir','vf','vs','vr','goal','t')},'obs_vel':obs['vel'].clone()})
        rgb_hashes.append(tensor_sha(obs['rgb']))
    for j in range(VISIBLE):
        append(obs,pa0 if j==0 else 0,pr0 if j==0 else 0.,f'visible_{j}',True)
        if j<VISIBLE-1:obs=clean_step(env,torch.zeros(n,dtype=torch.long))
    brakes['blind_entry']=brake(env)
    held=pop.activity(env.pos,env.mdir,torch.zeros_like(env.pos),shift_scale=0.)
    for j in range(PREFIX):
        obs=clean_step(env,prefix_actions);append(obs,prefix_actions,0.,f'turn_{j}',False)
    brakes['pre_probe_or_suffix']=brake(env)
    if kind=='probe_glimpse':
        probe_start=env.pos.clone()
        for j in range(PROBE):
            obs=clean_step(env,torch.full((n,),2,dtype=torch.long))
            cue=j==PROBE-1
            if cue:
                cue_error=float((env.pos-context['endpoint']).norm(dim=1).max());assert cue_error<1e-5
                before_snap=env.pos.clone();env.pos[:]=context['endpoint']
                obs=env.observe(obs['vel'])
                held=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
                cue_index=len(student);probe_obs={k:obs[k].clone()for k in ('pos','dir','vel','hd')}
            append(obs,2,0.,f'probe_{j}',False,cue)
    assert torch.equal(env.pos,context['endpoint'].expand(n,-1))
    brakes['suffix_entry']=brake(env)
    for j,action in enumerate(SUFFIX):
        obs=clean_step(env,torch.full((n,),action,dtype=torch.long))
        if j==len(SUFFIX)-1:brakes['first_choice']=brake(env)
        append(obs,action,0.,f'suffix_{j}',False)
    sx,ox,tx=map(torch.stack,(student,ordinary,teacher));physical={k:torch.stack([v[k]for v in physical])for k in physical[0]}
    assert sx.shape[-1]==1287 and torch.equal(sx[:,:,768:1280],goal[None].expand(len(sx),-1,-1))
    assert torch.equal(sx[:VISIBLE],tx[:VISIBLE])and torch.equal(sx[:,:,768:],tx[:,:,768:])
    assert torch.equal(sx[VISIBLE:,:,:256],black[None].expand(len(sx)-VISIBLE,-1,-1))
    assert all(paired_equal(v)for v in sx[-3:])
    assert paired_equal(env.pos)and paired_equal(torch.stack((env.vf,env.vs,env.vr),1))
    assert torch.count_nonzero(torch.stack((env.vf,env.vs,env.vr),1))==0
    separation=wrap_pi(env.hd[1::2]-env.hd[::2]).abs()
    heading_error=wrap_pi(env.hd-target_heading).abs();assert float(heading_error.max())<2e-5
    assert float((separation-math.pi).abs().max())<2e-5
    outside=torch.ones(1287,dtype=torch.bool);outside[256+torch.as_tensor(pop.pgc_idx,dtype=torch.long)]=False
    assert torch.equal(sx[:,:,outside],ox[:,:,outside])
    if cue_index is not None:
        mask=torch.ones(len(sx),dtype=torch.bool);mask[cue_index]=False;assert torch.equal(sx[mask],ox[mask])
    if cfg['cond']=='gc100':assert torch.equal(sx,ox)
    audit=dict(full_first_choice_input_pair_identity=True,last_three_inputs_pair_identity=True,
        same_position_goal_and_zero_motors=True,native_goal_unchanged=True,only_predictive_cue_slots_changed=True,
        first_choice_input_hashes=[tensor_sha(row)for row in sx[-1]],
        native_episode_len=cfg['episode_len'],source_episode_time=int(context['native_bookkeeping']['t']),
        first_choice_episode_time=env.t.tolist(),
        actual_heading_rad=env.hd.tolist(),heading_target_error_max_rad=float(heading_error.max()),
        antipodal_error_max_rad=float((separation-math.pi).abs().max()),
        cue_index=cue_index,first_choice_index=len(sx)-1,endpoint_rounding_correction_max_m=cue_error,
        image_noise=0.,motor_noise=0.,noise_scope='Deterministic information audit only, not proposed noisy training',
        initial_positions=start_pos.tolist(),brakes={k:v.tolist()for k,v in brakes.items()},grid_sham_exact=cfg['cond']=='gc100')
    audit['timeline']=dict(visible_observations=VISIBLE,visible_left_physics_updates=VISIBLE-1,
        opposite_prefix_physics_updates=PREFIX,probe_forward_updates=PROBE if cue_index is not None else 0,
        cue='Post-fourth-forward observation, before first suffix action'if cue_index is not None else None,
        common_suffix_actions=list(SUFFIX),ordinary_postcue_observations_including_choice=len(SUFFIX)if cue_index is not None else None,
        first_choice='Last recorded observation, after all suffix actions and final brake',
        initial_heading='Prescribed final phase minus exact native visible/prefix/suffix motor integration; not an assertion that14 opposite turns alone are antipodal',
        initial_state='Saved original state before the selected native context; pose/heading and velocities are then explicitly intervened on. Earlier native x is archived as provenance, not re-enacted physical motion.')
    if cue_index is not None:
        angle=torch.atan2(probe_obs['dir'][:,1],probe_obs['dir'][:,0])
        audit.update(probe_displacement_m=(before_snap-probe_start).norm(dim=1).tolist(),
            probe_speed_m_s=probe_obs['vel'].norm(dim=1).tolist(),
            probe_travel_minus_body_rad=wrap_pi(angle-probe_obs['hd']).tolist(),
            first_choice_travel_minus_body_rad=wrap_pi(torch.atan2(env.mdir[:,1],env.mdir[:,0])-env.hd).tolist())
    raw=dict(student_native_x=sx,student_ordinary_x=ox,teacher_x=tx,physical=physical,
        previous_actions=torch.stack(executed),labels=labels,initial_hc=context['initial_hc'].expand(n,-1).clone(),goal=goal,
        full_native_rgb_sha256_by_observation=rgb_hashes,
        target_heading=target_heading,context_reference={k:context[k]for k in ('context_id','slot','step')})
    return raw,env,audit


@torch.no_grad()
def action_progress(cfg,context,terminal):
    n=terminal.n;env=make_env(cfg,context,n*6)
    for k in ('pos','hd','mdir','vf','vs','vr','goal','t'):
        getattr(env,k)[:]=getattr(terminal,k).repeat_interleave(6,0)
    env.spawn[:]=env.pos;before=env.pos.clone();unit=(env.goal-before);unit/=unit.norm(dim=1,keepdim=True)
    before_error=wrap_pi(torch.atan2(unit[:,1],unit[:,0])-env.hd).abs()
    actions=torch.arange(6).repeat(n);clean_step(env,actions)
    end_bearing=torch.atan2(env.goal[:,1]-env.pos[:,1],env.goal[:,0]-env.pos[:,0])
    alignment=before_error-wrap_pi(end_bearing-env.hd).abs()
    return dict(progress_m=((env.pos-before)*unit).sum(1).reshape(n,6),
        body_goal_angle_error_reduction_rad=alignment.reshape(n,6),
        endpoint=env.pos.reshape(n,6,2).clone(),heading=env.hd.reshape(n,6).clone())


def pair_metrics(probabilities,progress,turn_alignment=None):
    assert probabilities.shape==progress.shape and probabilities.shape[1]==6
    records=[]
    for i in range(len(progress)//2):
        p=probabilities[2*i:2*i+2].double();p=p/p.sum(1,keepdim=True)
        q=progress[2*i:2*i+2].double()
        expected=(p*q).sum(1);m=p.mean(0);tiny=torch.finfo(torch.float64).tiny
        js=.5*(p*(p.clamp_min(tiny).log()-m.clamp_min(tiny).log())).sum()
        ceiling=float(q.mean(0).max());oracle=float(q.max(1).values.mean())
        records.append(dict(pair=i,teacher_probabilities=p.tolist(),action_progress_m=q.tolist(),
            teacher_action_argmax=p.argmax(1).tolist(),teacher_total_variation=float(.5*(p[0]-p[1]).abs().sum()),
            teacher_jensen_shannon=float(js),teacher_expected_progress_m=expected.tolist(),
            teacher_mean_expected_progress_m=float(expected.mean()),best_common_action_ceiling_m=ceiling,
            phase_specific_action_ceiling_m=oracle,teacher_advantage_over_common_m=float(expected.mean())-ceiling,
            phase_specific_ceiling_gap_m=oracle-ceiling))
        if turn_alignment is not None:
            angles=turn_alignment[2*i:2*i+2,:2].double();mass=p[:,:2].sum(1)
            weighted=(p[:,:2]*angles).sum(1)
            records[-1].update(turn_action_probabilities=p[:,:2].tolist(),
                native_turn_body_goal_alignment_rad=angles.tolist(),
                teacher_turn_weighted_alignment_rad=weighted.tolist(),
                teacher_alignment_conditional_on_turn_rad=[float(a/b)if b>0 else None for a,b in zip(weighted,mass)],
                turn_alignment_scope='Descriptive body-facing alignment, not a utility criterion: native strafing can be useful without facing the goal.')
    return records


@torch.no_grad()
def verify_native_prefixes(net,bank,data):
    """Re-enact archived32-slot recurrent inputs before pose intervention.

    Original batch shape and terminal resets are retained so the saved incoming
    h/c must agree exactly, rather than inheriting an unaudited cached state.
    """
    contexts=bank['contexts'];by_step={};last=max(c['step']for c in contexts)
    for context in contexts:
        slot,t=context['slot'],context['step']
        assert torch.equal(context['preceding_native_x'],data['x'][:t,slot])
        assert torch.equal(context['preceding_done'],data['done'][:t,slot])
        assert torch.equal(context['native_x'],data['x'][t,slot])
        assert torch.equal(context['initial_hc'],data['state_before'][t,slot])
        by_step.setdefault(t,[]).append(context)
    state=net.zero_state(32,'cpu');records=[]
    for t in range(last+1):
        for context in by_step.get(t,[]):
            actual=torch.cat(state,1)[context['slot']]
            error=float((actual-context['initial_hc']).abs().max());assert error==0.,(context['context_id'],error)
            records.append(dict(context=context['context_id'],slot=context['slot'],step=t,incoming_hc_max_error=error))
        if t<last:
            state=net.lstm(data['x'][t],state)
            state=tuple(v*(~data['done'][t])[:,None]for v in state)
    return dict(complete=True,original_batch_shape=32,replayed_updates=last,contexts=records,
        all_incoming_hc_exact=True,scope='Original archived recurrent-input prefix only. The subsequent controlled pose/heading and motor intervention is explicit and separate.')


def identical_input_information(inputs,probabilities,case_refs):
    """Generalized JS = minimum mean forward KL for each common-input group.

    Every case is retained, including singleton groups. Statistical arithmetic
    normalizes the archived float32 probabilities in float64; no model changes.
    """
    assert inputs.ndim==2 and inputs.shape[1]==1287 and len(inputs)==len(probabilities)==len(case_refs)
    p=probabilities.double();assert bool(torch.isfinite(p).all()and(p>=0).all())
    p=p/p.sum(1,keepdim=True);groups={}
    for i,x in enumerate(inputs):groups.setdefault(tensor_sha(x),[]).append(i)
    result=[];tiny=torch.finfo(torch.float64).tiny
    for digest,indices in groups.items():
        assert all(torch.equal(inputs[indices[0]],inputs[i])for i in indices)
        rows=p[indices];mean=rows.mean(0)
        kl=(rows*(rows.clamp_min(tiny).log()-mean.clamp_min(tiny).log())).sum(1)
        entropy=-(mean*mean.clamp_min(tiny).log()).sum()
        row_entropy=-(rows*rows.clamp_min(tiny).log()).sum(1).mean()
        js=float(kl.mean());assert abs(js-float(entropy-row_entropy))<1e-12
        result.append(dict(input_sha256=digest,count=len(indices),cases=[case_refs[i]for i in indices],
            optimal_common_action_distribution=mean.tolist(),generalized_js_nats=js,
            minimum_mean_forward_kl_nats=js,per_case_kl_to_common_nats=kl.tolist()))
    return dict(cases=len(inputs),groups=len(groups),records=result,
        case_weighted_minimum_forward_kl_nats=sum(v['count']*v['generalized_js_nats']for v in result)/len(inputs),
        probability_arithmetic='Archived float32 softmax probabilities normalized in float64 for information metrics',
        interpretation='For a memoryless stochastic actor given these byte-identical inputs, the mixture is the optimal common action distribution. JS quantifies teacher information that current inputs alone cannot reproduce; it is not a navigation or ring criterion.')


@torch.no_grad()
def continue_teacher(net,pop,cfg,context,env,state,first_x,first_logits):
    """30 actual full-vision native decisions; score only up to first event.

    Native respawns still occur inside ArenaEnv. The scoring position uses
    info.hit_pos before teleport and is then held; inactive rows are padding.
    Each paired row shares the exact action-uniform tape, not selected actions.
    """
    n=2*PAIRS;assert env.n==n
    tape=context['continuation_uniform'];assert tape.shape==(CONTINUATION,PAIRS)
    active=torch.ones(n,dtype=torch.bool);start=env.pos.clone();goal_center=env.goal.clone()
    endpoint=start.clone();path=torch.zeros(n);hits=torch.zeros(n,dtype=torch.bool)
    first_hit=torch.full((n,),-1,dtype=torch.long);first_done=torch.full_like(first_hit,-1)
    x=first_x;logits=first_logits;goal=context['native_x'][768:1280].expand(n,-1)
    raw={k:[]for k in ('x','logits','state_after_input','uniform','action','active_before','hit','done','position_preteleport','scoring_position','heading_before_action','reward')}
    for t in range(CONTINUATION):
        uniforms=tape[t].repeat_interleave(2);action=(uniforms[:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        before=env.pos.clone();heading_before=env.hd.clone();before_active=active.clone()
        obs,reward,done,info=env.step(action)
        hit=info['hit']&before_active;terminal=done&before_active
        physical=info['hit_pos'].clone()
        path+=torch.where(before_active,(physical-before).norm(dim=1),0.)
        endpoint=torch.where(before_active[:,None],physical,endpoint)
        first_hit=torch.where(hit,torch.full_like(first_hit,t),first_hit)
        first_done=torch.where(terminal,torch.full_like(first_done,t),first_done)
        hits|=hit;active=before_active&~hit&~terminal
        for k,value in (('x',x),('logits',logits),('state_after_input',torch.cat(state,1)),
                ('uniform',uniforms),('action',action),('active_before',before_active),('hit',hit),('done',terminal),
                ('position_preteleport',physical),('scoring_position',endpoint),('heading_before_action',heading_before),('reward',reward)):
            raw[k].append(value.clone())
        if t+1<CONTINUATION:
            code=pop.activity(obs['pos'],obs['dir'],obs['vel']);x=net._concat(net.embed(obs['rgb']),code,goal,action,reward,None)
            state=tuple(v*active[:,None]for v in state);state=net.lstm(x,state);logits=net.pi(state[0])
    progress=(start-goal_center).norm(dim=1)-(endpoint-goal_center).norm(dim=1)
    raw={k:torch.stack(v)for k,v in raw.items()};raw.update(start_position=start,original_goal_center=goal_center)
    rows=[dict(case=i,pair=i//2,member=i%2,first_hit_step=int(first_hit[i]),first_done_step=int(first_done[i]),
        hit=bool(hits[i]),active_updates=int(raw['active_before'][:,i].sum()),path_m=float(path[i]),
        goal_distance_progress_m=float(progress[i]),endpoint_before_teleport=endpoint[i].tolist(),
        initial_goal_distance_m=float((start[i]-goal_center[i]).norm()))for i in range(n)]
    return raw,dict(steps=CONTINUATION,shared_uniforms_within_pair=True,first_hit_censored=True,
        preteleport_endpoint_accounting=True,image_noise=0.,motor_noise=0.,rows=rows,
        interpretation='Controlled noise-free native full-vision teacher utility after actual forced history; not an oracle ceiling. Native post-terminal simulator rows are padding and contribute no further progress or hits.')


@torch.no_grad()
def evaluate_context(net,pop,cfg,context,kind):
    raw,env,audit=construct_history(net,pop,cfg,context,kind)
    state=tuple(v.clone()for v in raw['initial_hc'].chunk(2,1));states=[]
    for x in raw['teacher_x']:
        state=net.lstm(x,state);states.append(torch.cat(state,1))
    logits=net.pi(state[0]);prob=logits.softmax(1)
    erased=net.lstm(raw['student_native_x'][-1],net.zero_state(2*PAIRS,'cpu'))
    erased_logits=net.pi(erased[0]);assert paired_equal(erased_logits)
    physics=action_progress(cfg,context,env);progress=physics['progress_m']
    rows=pair_metrics(prob,progress,physics['body_goal_angle_error_reduction_rad'])
    for i,row in enumerate(rows):
        row.update(context=context['context_id'],kind=kind,source_slot=context['slot'],source_step=context['step'])
        erased_mean=float((erased_logits.softmax(1)[2*i:2*i+2]*progress[2*i:2*i+2]).sum(1).mean())
        assert erased_mean<=row['best_common_action_ceiling_m']+1e-8
        row['erased_student_mean_progress_m']=erased_mean
    continuation,utility=continue_teacher(net,pop,cfg,context,env,state,raw['teacher_x'][-1],logits)
    raw.update(teacher_states_after=torch.stack(states),teacher_first_choice_logits=logits,
        teacher_first_choice_probabilities=prob,erased_student_logits=erased_logits,first_update_physics=physics,
        continuation=continuation)
    return raw,dict(complete=True,audit=audit,pairs=rows,teacher_continuation=utility)


def freeze():
    """Only run after independent review and explicit root freeze approval."""
    assert not PROTOCOL.exists()and not(BASE/'banks').exists()
    BASE.mkdir(parents=True,exist_ok=True)
    assets={};banks={}
    for condition in CONDITIONS:
        stem=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M'
        cfg=json.loads(stem.with_suffix('.json').read_text())
        assert cfg['episode_len']==1350 and cfg['nh']==256 and cfg['extra_input']=='none'and cfg['code_noise']==0
        trace=ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt'
        summary=trace.with_name('summary.json');assert sha(trace)==json.loads(summary.read_text())['trace_sha256']
        collected=trace.with_name('protocol.json');provenance=json.loads(collected.read_text())
        assert provenance['checkpoint_sha256']==sha(stem.with_suffix('.pt'))and provenance['config_sha256']==sha(stem.with_suffix('.json'))
        assert provenance['condition']==condition and provenance['seed']==1
        data=torch.load(trace,map_location='cpu',weights_only=True)
        assert data['condition']==condition and data['seed']==1 and data['config']==cfg
        assert data['source_manifest_sha256']==provenance['source_manifest_sha256']
        bank=select_contexts(data,cfg)
        if not bank['complete']:
            atomic_json(BASE/f'{condition}_incomplete_inventory.json',bank['inventory'])
            raise AssertionError('Fewer than16 geometry-eligible episodes; inventory retained, no model evaluated')
        bank.update(condition=condition,trace_sha256=sha(trace),checkpoint_sha256=sha(stem.with_suffix('.pt')),config=cfg,
            census_collection_protocol_sha256=sha(collected),census_collection_provenance=provenance)
        path=BASE/'banks'/f'{condition}.pt';path.parent.mkdir(parents=True,exist_ok=True);atomic_torch(path,bank)
        banks[condition]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,tensors=tensor_inventory(bank))
        for p in (stem.with_suffix('.pt'),stem.with_suffix('.json'),trace,summary,collected):assets[str(p.relative_to(ROOT))]=sha(p)
    protocol=dict(version='native_turn_teacher_preflight_v1',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        no_optimization=True,no_cloud=True,conditions=list(CONDITIONS),contexts=CONTEXTS,pairs_per_context=PAIRS,
        pairs_per_kind_per_model=CONTEXTS*PAIRS,kinds=list(KINDS),visible_observations=VISIBLE,blind_turn_updates=PREFIX,
        probe_updates=PROBE,suffix=list(SUFFIX),teacher_continuation_updates=CONTINUATION,action_seed=ACTION_SEED,
        qualification=None,qualification_scope='Descriptive-only information and teacher utility audit. No scientific pass/fail threshold or optimization permission.',
        banks=banks,source_sha256={name:sha(ROOT/name)for name in SOURCES},asset_sha256=assets,
        metric_scope='First native update progress is reported together with teacher distribution distances. It can penalize useful turning; no threshold is supplied by code.')
    atomic_json(PROTOCOL,protocol);print(sha(PROTOCOL))


def run(condition,out):
    assert PROTOCOL.exists(),'Reviewed frozen protocol required'
    design=json.loads(PROTOCOL.read_text());assert design['no_optimization']and design['no_cloud']
    for key in ('source_sha256','asset_sha256'):
        for name,digest in design[key].items():assert sha(ROOT/name)==digest,name
    record=design['banks'][condition];path=ROOT/record['path'];assert sha(path)==record['sha256']and path.stat().st_size==record['bytes']
    bank=torch.load(path,map_location='cpu',weights_only=True);assert tensor_inventory(bank)==record['tensors']
    cfg=bank['config'];ckpt=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
    saved=torch.load(ckpt,map_location='cpu',weights_only=True);assert saved['frames']==1_000_000_000
    rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):net=build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(saved['policy'],strict=True)
    digest=state_digest(net);pop=build_population(cfg,'cpu');out=Path(out);assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    assert digest==bank['census_collection_provenance']['policy_hash']
    atomic_json(out/'inventory.json',dict(condition=condition,inventory=bank['inventory'],bank_sha256=record['sha256']))
    trace_path=ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt'
    data=torch.load(trace_path,map_location='cpu',weights_only=True)
    assert torch.equal(net.pi.weight,data['actor_weight'])and torch.equal(net.pi.bias,data['actor_bias'])
    try:
        prefixes=verify_native_prefixes(net,bank,data)
    except Exception as error:
        atomic_json(out/'incomplete.json',dict(complete=False,phase='original_prefix_verification',error_type=type(error).__name__,error=str(error),no_retry=True));raise
    atomic_json(out/'native_prefix_verification.json',prefixes);del data
    rows=[];artifacts={};common_inputs=[];common_targets=[];case_refs=[];continuations=[]
    for context in bank['contexts']:
        for kind in KINDS:
            stem=out/f"context{context['context_id']:02d}_{kind}"
            try:
                raw,result=evaluate_context(net,pop,cfg,context,kind)
                atomic_torch(stem.with_suffix('.pt'),raw);atomic_json(stem.with_suffix('.json'),result);rows.extend(result['pairs'])
                common_inputs.append(raw['student_native_x'][-1]);common_targets.append(raw['teacher_first_choice_probabilities'])
                case_refs.extend(dict(context=context['context_id'],kind=kind,case=i,pair=i//2,member=i%2)for i in range(2*PAIRS))
                continuations.append(dict(context=context['context_id'],kind=kind,**result['teacher_continuation']))
            except Exception as error:
                atomic_torch(stem.with_name(stem.name+'_failed_context.pt'),context)
                atomic_json(out/'incomplete.json',dict(complete=False,context=context['context_id'],kind=kind,error_type=type(error).__name__,error=str(error),no_retry=True));raise
    assert len(rows)==CONTEXTS*PAIRS*len(KINDS)and state_digest(net)==digest
    assert torch.equal(rng,torch.get_rng_state())
    common_inputs=torch.cat(common_inputs);common_targets=torch.cat(common_targets)
    grouped={'all_modes':identical_input_information(common_inputs,common_targets,case_refs)}
    for kind in KINDS:
        indices=[i for i,v in enumerate(case_refs)if v['kind']==kind]
        grouped[kind]=identical_input_information(common_inputs[indices],common_targets[indices],[case_refs[i]for i in indices])
    for key in ('source_sha256','asset_sha256'):
        for name,d in design[key].items():assert sha(ROOT/name)==d,name
    for p in out.iterdir():
        if p.is_file():artifacts[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    result=dict(complete=True,condition=condition,protocol_sha256=sha(PROTOCOL),checkpoint_sha256=sha(ckpt),policy_hash=digest,
        pairs=rows,pair_counts={kind:sum(r['kind']==kind for r in rows)for kind in KINDS},
        qualification=None,qualification_status='descriptive_only_no_scientific_gate',
        exact_input_group_information=grouped,teacher_continuations=continuations,
        global_cpu_rng_preserved=True,no_optimization=True,no_cloud=True,artifacts=artifacts)
    atomic_json(out/'summary.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--out',type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    if args.freeze:freeze();return
    assert args.condition and args.out;run(args.condition,args.out)
if __name__=='__main__':main()
