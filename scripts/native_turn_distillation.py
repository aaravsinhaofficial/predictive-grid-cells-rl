"""Recurrent policy distillation on the existing native blind-turn event.

The separate runner requires the reviewed frozen protocol before execution.
The only learning target is the original full-vision policy on actual history.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from code.common import CodeSource,build_population
from lstm_code_swap import MatchedArena
from rl_visual_memory import NativeGoal
from rl_goal_memory import reset_state,state_digest
import native_policy_distillation as dist
import rl_blind_reorientation as event

BASE=ROOT/'results/native_turn_distillation'
PROTOCOL=ROOT/'docs/research/native_turn_distillation_protocol.json'
CALIBRATION=ROOT/'results/native_turn_distillation_design/original_delta_rms.json'
N,UNROLL,UPDATES=16,90,180
BLOCK_UPDATES=60
SUFFIX_RANGES=((4,4),(4,8),(4,15))
BETAS=(.5,.25,0.)
CHOICE_STEPS=3
MAX_EVENT=15+2*10+4+15+CHOICE_STEPS
TRAIN_SEED,EVAL_SEED=984201,984901
ARMS=('clean','noise')


class TurnClock(event.ReorientationClock):
    """Same event plans/interruptions; fixed57-step fit and three free choices."""
    def prepare(self,segment_remaining,episode_remaining,suffix_range):
        low,high=suffix_range;assert 4<=low<=high<=15
        ready=self.known&(self.phase==event.READY);cap=self.event_count>=event.EVENT_CAP
        segment_bad=self.used_segment|(segment_remaining<MAX_EVENT)
        episode_bad=episode_remaining<MAX_EVENT
        self.stats['wait_cap']+=(ready&cap).long()
        self.stats['wait_segment']+=(ready&~cap&segment_bad).long()
        self.stats['wait_episode']+=(ready&~cap&~segment_bad&episode_bad).long()
        admit=ready&~cap&~segment_bad&~episode_bad
        if admit.any():
            idx=admit.nonzero().flatten();draws=self.plans[idx.cpu(),self.event_count[idx].cpu()].float()
            self.turn_a[idx]=(draws[:,0]>=.5).long();self.turn_b[idx]=(draws[:,2]>=.5).long()
            self.duration_a[idx]=5+(draws[:,1]*6).long().clamp_max(5)
            self.duration_b[idx]=5+(draws[:,3]*6).long().clamp_max(5)
            self.duration_delay[idx]=low+(draws[:,4]*(high-low+1)).long().clamp_max(high-low)
            self.phase[idx]=event.VISIBLE;self.remaining[idx]=15;self.age[idx]=0;self.elapsed[idx]=0
            self.event_count[idx]+=1;self.stats['admitted'][idx]+=1;self.used_segment[idx]=True
        return admit

    def advance(self,hit,done):
        old=self.phase.clone();super().advance(hit,done)
        entering=(old==event.DELAY)&(self.phase==event.FREE)&~hit&~done
        self.remaining[entering]=CHOICE_STEPS


class EventTapes:
    """Independent per-slot/episode/event cue coin and two unit-variance tapes.

    Indexed private seeds prevent previous hit timing from consuming another
    event's noise. Both arms draw/store identical attempted tapes; clean ignores
    them. Episode, seed, coins and complete tapes are included in checkpoints.
    """
    def __init__(self,n,hidden,seed):
        self.n,self.hidden,self.seed=n,hidden,int(seed)
        self.episode=torch.zeros(n,dtype=torch.long)
        self.ordinary=torch.zeros(n,event.EVENT_CAP,dtype=torch.bool)
        self.unit=torch.zeros(n,event.EVENT_CAP,2,2,hidden)
        self.draw(torch.ones(n,dtype=torch.bool))

    def draw(self,mask):
        for slot in mask.nonzero().flatten().tolist():
            key=self.seed+slot*7919+int(self.episode[slot])*104729
            cue_rng=torch.Generator().manual_seed(key)
            noise_rng=torch.Generator().manual_seed(key+1_000_003)
            self.ordinary[slot]=torch.rand(event.EVENT_CAP,generator=cue_rng)<.5
            self.unit[slot]=(2*torch.rand(event.EVENT_CAP,2,2,self.hidden,generator=noise_rng)-1)*math.sqrt(3.)

    def advance(self,done):
        self.episode+=done.long()
        if done.any():self.draw(done)

    def current(self,clock):
        indices=(clock.event_count-1).clamp_min(0);rows=torch.arange(self.n)
        return self.ordinary[rows,indices],self.unit[rows,indices]


def fixed_sigmas(condition):
    record=json.loads(CALIBRATION.read_text())['models'][condition]
    return tuple(.10*record['blocks'][block]['one_update_delta_rms']for block in ('h','c'))


def inject(state,eps,mask):
    """Incoming impulse, no detach; only h is clipped to its physical range."""
    h,c=state;eh,ec=eps
    after_h=torch.where(mask[:,None],(h+eh).clamp(-1.,1.),h)
    after_c=c+ec*mask[:,None]
    return after_h,after_c


def weighted_kl(losses,choice,forced):
    assert losses.shape==choice.shape==forced.shape and not(choice&forced).any()
    control=~choice&~forced
    assert control.any(),'The fixed event-fit rule must leave native control decisions'
    control_mean=losses[control].mean()
    choice_mean=losses[choice].mean()if choice.any()else None
    total=.5*(choice_mean+control_mean)if choice_mean is not None else control_mean
    return total,dict(choice_count=int(choice.sum()),control_count=int(control.sum()),
        forced_count=int(forced.sum()),no_choice_fallback=choice_mean is None,
        choice_kl=float(choice_mean.detach())if choice_mean is not None else None,
        control_kl=float(control_mean.detach()))


def make_runtime(teacher,cfg,arm,deadline,seed=TRAIN_SEED):
    assert arm in ARMS
    n=N;env=MatchedArena(cfg,n,1,seed,'cpu');obs=env.reset_all();pop=build_population(cfg,'cpu')
    runtime=dict(obs=obs,student_state=teacher.zero_state(n,'cpu'),teacher_state=teacher.zero_state(n,'cpu'),
        goal=NativeGoal(n,512,'cpu'),clock=TurnClock(n,'cpu',seed+2),inputs=event.EventInputs(n,512,'cpu'),
        tapes=EventTapes(n,teacher.lstm.hidden_size,seed+4),source=CodeSource(cfg,pop,'cpu'),
        pa=torch.full((n,),-1,dtype=torch.long),pr=torch.zeros(n),
        action_rng=torch.Generator().manual_seed(seed+1),mixture_rng=torch.Generator().manual_seed(seed+3),
        black=teacher.embed(torch.zeros(n,3,84,84)).detach(),deadline=deadline,arm=arm,
        sigma=fixed_sigmas(cfg['cond']),choice_brake_count=torch.zeros(n,dtype=torch.long),
        choice_brake_before=torch.zeros(n,3),choice_brake_seen=torch.zeros(n,dtype=torch.bool),event_rows=[])
    return env,pop,runtime


@torch.no_grad()
def make_inputs(teacher,pop,cfg,env,runtime,obs,goal,pa,pr):
    clock,inputs=runtime['clock'],runtime['inputs']
    native,_=dist.feature_inputs(teacher,pop,obs,goal.saved,pa,pr,N)
    _,current,blind,glimpse=inputs.prepare(env,obs,clock,runtime['source'])
    first_choice=(clock.phase==event.FREE)&(clock.age==0)
    before=torch.stack((env.vf,env.vs,env.vr),1)
    runtime['choice_brake_before']=torch.where(first_choice[:,None],before,runtime['choice_brake_before'])
    runtime['choice_brake_count']+=first_choice.long();event.brake(env,first_choice)
    runtime['choice_brake_seen']|=first_choice
    assert not torch.stack((env.vf,env.vs,env.vr),1)[first_choice].any()
    x=native.clone();x[:,:256]=torch.where(blind[:,None],runtime['black'],x[:,:256]);x[:,256:768]=current
    assert torch.equal(x[:,768:],native[:,768:])
    return native,x,blind,glimpse,first_choice


@torch.no_grad()
def replay(student,probe):
    state=tuple(v.clone()for v in probe['state0']);logits=[]
    for x,done,eps,mask in zip(probe['student_x'],probe['done'],probe['attempted_noise'],probe['noise_applied']):
        state=inject(state,tuple(eps.unbind(1)),mask)
        state=student.lstm(x,state);logits.append(student.pi(state[0]));state=reset_state(state,done)
    return torch.stack(logits),state


def train_unroll(student,teacher,pop,cfg,env,runtime,opt,stage,length=UNROLL):
    """Whole-event training primitive; study execution is guarded by the runner."""
    assert stage in (0,1,2)
    clock,inputs,tapes=runtime['clock'],runtime['inputs'],runtime['tapes'];clock.begin_segment()
    obs=runtime['obs'];goal=runtime['goal'];pa=runtime['pa'];pr=runtime['pr']
    state=tuple(v.detach()for v in runtime['student_state']);teacher_state=runtime['teacher_state']
    state0=tuple(v.clone()for v in state);teacher_state0=tuple(v.clone()for v in teacher_state)
    keys=('student_x','teacher_x','teacher_logits','student_logits','done','phase','choice','forced','blind',
        'glimpse','ordinary_cue','attempted_noise','realized_noise','h_clip','noise_scheduled','noise_applied','action','sampled_action','teacher_action',
        'reward','hit','hit_pos','wall_hit','position_before','heading_before','motors_before','choice_brake_before')
    stored={k:[]for k in keys};losses=[];episodes=[]
    for t in range(length):
        dist.check_time(runtime['deadline'])
        admit=clock.prepare(length-t,cfg['episode_len']-env.t,SUFFIX_RANGES[stage])
        runtime['choice_brake_before'][admit]=0.;runtime['choice_brake_seen'][admit]=False
        ordinary,unit=tapes.current(clock);inputs.ordinary=torch.where(admit,ordinary,inputs.ordinary)
        native,x,blind,glimpse,first_choice=make_inputs(teacher,pop,cfg,env,runtime,obs,goal,pa,pr)
        post=(clock.phase==event.DELAY)&(clock.age==2);scheduled=glimpse|post
        pulse_index=post.long();attempted=unit[torch.arange(N),pulse_index].clone()
        attempted[:,0]*=runtime['sigma'][0];attempted[:,1]*=runtime['sigma'][1]
        applied=scheduled&(runtime['arm']=='noise')
        before_state=tuple(v.detach().clone()for v in state)
        clipped=((before_state[0]+attempted[:,0]).abs()>1)&applied[:,None]
        state=inject(state,tuple(attempted.unbind(1)),applied)
        realized=torch.stack([v.detach()-b for v,b in zip(state,before_state)],1)
        with torch.no_grad():teacher_logits,teacher_state=dist.chunk_forward(teacher,native,teacher_state,N)
        student_logits,state=dist.chunk_forward(student,x,state,N)
        loss=dist.kl(teacher_logits,student_logits);losses.append(loss)
        forced=clock.forced();choice=clock.phase==event.FREE
        with torch.no_grad():
            mixture=torch.rand(N,generator=runtime['mixture_rng'])<BETAS[stage]
            sampled=dist.sample(torch.where(mixture[:,None],teacher_logits,student_logits.detach()),
                torch.rand(N,generator=runtime['action_rng']))
            action=clock.action(sampled)
        phase=clock.phase.clone();position=env.pos.clone();heading=env.hd.clone();motors=torch.stack((env.vf,env.vs,env.vr),1)
        obs,reward,done,info=env.step(action)
        values=dict(student_x=x,teacher_x=native,teacher_logits=teacher_logits,student_logits=student_logits.detach(),done=done,
            phase=phase,choice=choice,forced=forced,blind=blind,glimpse=glimpse,ordinary_cue=glimpse&inputs.ordinary,
            attempted_noise=attempted,realized_noise=realized,h_clip=clipped,noise_scheduled=scheduled,noise_applied=applied,
            action=action,sampled_action=sampled,teacher_action=mixture&~forced,reward=reward,hit=info['hit'],hit_pos=info['hit_pos'],
            wall_hit=info['hit_wall'],position_before=position,heading_before=heading,motors_before=motors,
            choice_brake_before=runtime['choice_brake_before'].clone())
        for key in keys:stored[key].append(values[key].detach().clone())
        inputs.transition(clock,info)
        # Preserve telemetry when each actual event completes or is interrupted.
        ending=(clock.phase!=event.READY)&(info['hit']|done|((clock.phase==event.FREE)&(clock.remaining==1)))
        for i in ending.nonzero().flatten().tolist():
            runtime['event_rows'].append(dict(slot=i,episode=int(clock.episode[i]),event=int(clock.event_count[i]),
                ordinary_cue=bool(inputs.ordinary[i]),last_phase=int(phase[i]),interrupted_hit=bool(info['hit'][i]),
                interrupted_done=bool(done[i]),choice_brake_delivered=bool(runtime['choice_brake_seen'][i]),
                choice_brake_before=runtime['choice_brake_before'][i].tolist(),
                **inputs.event_audit(i)))
        dist.advance_goal(pop,goal,info,done,N)
        clock.advance(info['hit'],done);tapes.advance(done)
        assert torch.equal(tapes.episode,clock.episode)
        state=reset_state(state,done);teacher_state=reset_state(teacher_state,done)
        pa=torch.where(done,torch.full_like(action,-1),action)
        pr=reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])*(~done)
        if info['ep_stats']is not None:episodes.extend(info['ep_stats']['ret'].tolist())
    assert not(clock.phase!=event.READY).any(),'No event may cross a BPTT boundary'
    probe={k:torch.stack(v)for k,v in stored.items()};probe.update(state0=state0,teacher_state0=teacher_state0)
    all_losses=torch.stack(losses);loss,accounting=weighted_kl(all_losses,probe['choice'],probe['forced'])
    assert torch.isfinite(loss)
    replayed,replayed_state=replay(student,probe)
    replay_error=float((replayed-probe['student_logits']).abs().max());assert replay_error==0.
    assert all(torch.equal(a,b.detach())for a,b in zip(replayed_state,state))
    with torch.no_grad():
        ts=teacher_state0;tl=[]
        for xx,dd in zip(probe['teacher_x'],probe['done']):
            ts=teacher.lstm(xx,ts);tl.append(teacher.pi(ts[0]));ts=reset_state(ts,dd)
        teacher_replay_error=float((torch.stack(tl)-probe['teacher_logits']).abs().max());assert teacher_replay_error==0.
        assert all(torch.equal(a,b)for a,b in zip(ts,teacher_state))
    before=[p.detach().clone()for p in student.lstm.parameters()]
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in student.lstm.parameters())
    gradient=float(torch.nn.utils.clip_grad_norm_(student.lstm.parameters(),1.));assert math.isfinite(gradient)
    opt.step();change=sum(float((p.detach()-b).square().sum())for p,b in zip(student.lstm.parameters(),before))**.5
    assert all(torch.isfinite(p).all()for p in student.lstm.parameters())
    runtime.update(obs=obs,student_state=tuple(v.detach()for v in state),teacher_state=teacher_state,pa=pa,pr=pr)
    counts={k:int(probe[k].sum())for k in ('blind','glimpse','ordinary_cue','noise_scheduled','noise_applied','teacher_action','hit','wall_hit')}
    metrics=dict(loss=float(loss.detach()),**accounting,counts=counts,
        grad_norm_before_clip=gradient,parameter_update_l2=change,replay_max_error=replay_error,
        teacher_replay_max_error=teacher_replay_error,completed_episode_scores=episodes,
        phase_counts=[int((probe['phase']==i).sum())for i in range(len(event.PHASE_NAMES))],
        reward_by_phase=[float(probe['reward'][probe['phase']==i].sum())for i in range(len(event.PHASE_NAMES))],
        clipped_hidden_coordinates=int(probe['h_clip'].sum()),
        noise_realized_rms=[float(probe['realized_noise'][:,:,i][probe['noise_applied']].square().mean().sqrt())
            if probe['noise_applied'].any()else 0. for i in (0,1)],
        noise_realized_mean=[float(probe['realized_noise'][:,:,i][probe['noise_applied']].mean())
            if probe['noise_applied'].any()else 0. for i in (0,1)])
    return metrics,probe


def snapshot(student,teacher,cfg,opt,env,runtime,updates,protocol_sha256):
    return dict(policy=copy.deepcopy(student.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),config=cfg,
        teacher_policy_hash=state_digest(teacher),frozen_tensor_hash=dist.frozen_digest(student),
        updates=updates,original_frames=1_000_000_000,extra_frames=updates*N*UNROLL*cfg['action_repeat'],
        protocol_sha256=protocol_sha256,arm=runtime['arm'],cpu_rng=torch.get_rng_state().clone(),
        environment_rng=env.gen.get_state().clone(),
        environment_tensors={k:v.clone()for k,v in vars(env).items()if isinstance(v,torch.Tensor)},
        environment_scalars={k:v for k,v in vars(env).items()if isinstance(v,(int,float,bool,str))},
        clock=copy.deepcopy(vars(runtime['clock'])),inputs=copy.deepcopy(vars(runtime['inputs'])),
        event_tapes=copy.deepcopy(vars(runtime['tapes'])),
        runtime=copy.deepcopy({k:runtime[k]for k in ('student_state','teacher_state','obs','pa','pr','black','sigma',
            'choice_brake_count','choice_brake_before','choice_brake_seen','event_rows')}),goal=runtime['goal'].saved.clone(),
        action_rng=runtime['action_rng'].get_state().clone(),mixture_rng=runtime['mixture_rng'].get_state().clone())


def restore(snapshot,student,teacher,cfg,opt,deadline):
    assert snapshot['config']==cfg and snapshot['teacher_policy_hash']==state_digest(teacher)
    student.load_state_dict(snapshot['policy']);opt.load_state_dict(snapshot['optimizer'])
    assert dist.frozen_digest(student)==snapshot['frozen_tensor_hash']
    env,pop,runtime=make_runtime(teacher,cfg,snapshot['arm'],deadline)
    # Some native render grids are expanded views; replace with owned tensors.
    for k,v in snapshot['environment_tensors'].items():setattr(env,k,v.clone())
    for k,v in snapshot['environment_scalars'].items():setattr(env,k,v)
    env.gen.set_state(snapshot['environment_rng'].clone())
    for name,key in (('clock','clock'),('inputs','inputs'),('tapes','event_tapes')):
        vars(runtime[name]).update(copy.deepcopy(snapshot[key]))
    runtime.update(copy.deepcopy(snapshot['runtime']));runtime['goal'].saved.copy_(snapshot['goal'])
    runtime['action_rng'].set_state(snapshot['action_rng'].clone());runtime['mixture_rng'].set_state(snapshot['mixture_rng'].clone())
    torch.set_rng_state(snapshot['cpu_rng'].clone())
    return env,pop,runtime
