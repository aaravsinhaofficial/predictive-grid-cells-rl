"""Synthetic/private fixtures only. No study policy or optimization run."""
import copy
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_turn_distillation as study
from test_native_turn_teacher_preflight import fixture


def test_exact_event_duration_and_whole_unroll_eligibility():
    clock=study.TurnClock(2,'cpu',1234);clock.known[:]=True;clock.plans[:]=.999999
    assert not clock.prepare(56,torch.tensor([1350,1350]),(4,15)).any()
    admitted=clock.prepare(90,torch.tensor([56,57]),(4,15));assert admitted.tolist()==[False,True]
    phases=[];ages=[]
    for _ in range(57):
        phases.append(int(clock.phase[1]));ages.append(int(clock.age[1]))
        clock.advance(torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    assert [phases.count(i)for i in range(7)]==[0,15,10,10,4,15,3]
    assert (clock.phase==study.event.READY).all()
    assert ages[phases.index(study.event.FREE)]==0
    assert clock.stats['completed'].tolist()==[0,1]
    clock.begin_segment();assert not clock.used_segment.any()


def test_event_tapes_are_private_indexed_and_clean_noise_matched():
    rng=torch.get_rng_state().clone();a=study.EventTapes(2,8,743201);b=study.EventTapes(2,8,743201)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(a.unit,b.unit)and torch.equal(a.ordinary,b.ordinary)
    before=a.unit.clone();a.advance(torch.tensor([True,False]));b.advance(torch.tensor([True,False]))
    assert torch.equal(a.unit,b.unit)and torch.equal(a.unit[1],before[1])
    assert not torch.equal(a.unit[0],before[0])
    assert a.episode.tolist()==[1,0]


def test_bounded_incoming_noise_preserves_gradient_and_clean_identity():
    h=torch.tensor([[.99,-.99],[.2,-.3]],requires_grad=True);c=torch.zeros_like(h,requires_grad=True)
    eps=(torch.full_like(h,.1),torch.full_like(c,.3))
    a,b=study.inject((h,c),eps,torch.tensor([True,False]))
    assert a[0,0]==1 and torch.equal(a[1],h[1])and torch.equal(b[1],c[1])
    assert torch.equal(b[0],torch.full((2,),.3))
    (a.sum()+b.sum()).backward()
    assert h.grad.tolist()==[[0.,1.],[1.,1.]]and torch.equal(c.grad,torch.ones_like(c))
    clean=study.inject((h,c),eps,torch.zeros(2,dtype=torch.bool))
    assert torch.equal(clean[0],h)and torch.equal(clean[1],c)


def test_weighted_choice_control_loss_and_forced_zero_direct_gradient():
    losses=torch.tensor([[1.,2.,3.,4.,5.,6.]],requires_grad=True)
    choice=torch.tensor([[True,False,False,True,False,False]])
    forced=torch.tensor([[False,True,False,False,True,False]])
    value,counts=study.weighted_kl(losses,choice,forced);value.backward()
    assert value==3.5 and losses.grad.tolist()==[[.25,0.,.25,.25,0.,.25]]
    assert counts['choice_count']==2 and counts['control_count']==2 and counts['forced_count']==2
    fallback,counts=study.weighted_kl(losses,torch.zeros_like(choice),forced)
    assert fallback==3.5 and counts['no_choice_fallback']


def test_exact_noise_replay_and_gradient_reaches_forced_cue_input():
    _,teacher,_,_=fixture();student=study.dist.trainable_student(teacher)
    gen=torch.Generator().manual_seed(743901);T,n=10,16
    x=(torch.randn(T,n,1287,generator=gen)*.05).requires_grad_(True)
    eps=torch.randn(T,n,2,8,generator=gen)*.01
    mask=torch.zeros(T,n,dtype=torch.bool);mask[2]=True;mask[4]=True
    done=torch.zeros(T,n,dtype=torch.bool);done[-1,0]=True
    state0=student.zero_state(n,'cpu');state=state0;logits=[]
    for t in range(T):
        state=study.inject(state,tuple(eps[t].unbind(1)),mask[t]);state=student.lstm(x[t],state)
        logits.append(student.pi(state[0]));state=study.reset_state(state,done[t])
    torch.stack(logits)[-1].square().sum().backward()
    assert x.grad[2].abs().sum()>0  # No direct loss at cue; later choice supplies gradient.
    replayed,replayed_state=study.replay(student,dict(student_x=x.detach(),state0=state0,done=done,
        attempted_noise=eps,noise_applied=mask))
    assert torch.equal(replayed,torch.stack(logits).detach())
    assert all(torch.equal(a,b.detach())for a,b in zip(replayed_state,state))


class NoUpdate:
    def __init__(self,parameters):self.parameters=list(parameters);self.steps=0
    def zero_grad(self,set_to_none=True):
        for p in self.parameters:p.grad=None
    def step(self):self.steps+=1  # Explicitly no parameter update in this fixture.


def private_runtime(arm):
    cfg,teacher,_,_=fixture();student=study.dist.trainable_student(teacher)
    env,pop,r=study.make_runtime(teacher,cfg,arm,float('inf'),seed=744201)
    env.pos.zero_();env.goal[:]=torch.tensor([.9,0.]);env.hd[:]=.1;env.mdir[:]=torch.tensor([1.,0.])
    study.event.brake(env,torch.ones(study.N,dtype=torch.bool));env.n_goals[:]=1
    r['goal'].saved=pop.activity(env.goal,env.mdir,torch.zeros_like(env.pos));r['clock'].known[:]=True
    r['obs']=env.observe();return cfg,teacher,student,env,pop,r


def test_native_event_noise_masks_brakes_and_full_replay_without_optimization():
    cfg,teacher,student,env,pop,r=private_runtime('noise')
    before=copy.deepcopy(student.state_dict());opt=NoUpdate(student.lstm.parameters())
    metrics,probe=study.train_unroll(student,teacher,pop,cfg,env,r,opt,stage=0)
    assert opt.steps==1 and all(torch.equal(before[k],v)for k,v in student.state_dict().items())
    assert metrics['parameter_update_l2']==0 and metrics['grad_norm_before_clip']>0
    assert metrics['choice_count']==3*study.N and metrics['replay_max_error']==metrics['teacher_replay_max_error']==0
    assert int(probe['noise_applied'].sum())==2*study.N
    assert torch.equal(probe['student_logits'][:15],probe['teacher_logits'][:15])
    for slot in range(study.N):
        cue=int(probe['glimpse'][:,slot].nonzero()[0]);pulses=probe['noise_applied'][:,slot].nonzero().flatten().tolist()
        assert pulses==[cue,cue+2]
        first=int(probe['choice'][:,slot].nonzero()[0]);assert first-cue==4
        assert torch.count_nonzero(probe['motors_before'][first,slot])==0
        assert torch.equal(probe['student_x'][cue+1:first+3,slot,256:768],
            probe['student_x'][cue+1,slot,256:768].expand(first+2-cue,-1))
    assert (r['clock'].phase==study.event.READY).all()


def test_native_terminal_clears_both_histories_and_goal_action_reward():
    cfg,teacher,student,env,pop,r=private_runtime('clean');env.t[:]=1349;r['obs']=env.observe()
    metrics,probe=study.train_unroll(student,teacher,pop,cfg,env,r,NoUpdate(student.lstm.parameters()),stage=0,length=2)
    assert probe['done'][0].all()and not probe['phase'].any()
    assert torch.count_nonzero(probe['student_x'][1,:,768:])==0
    assert torch.equal(probe['student_x'],probe['teacher_x'])
    assert torch.equal(probe['student_logits'],probe['teacher_logits'])
    assert not probe['noise_scheduled'].any()and metrics['no_choice_fallback']


def test_visible_reference_keeps_identical_held_code_and_reset_schedule():
    import evaluate_native_turn_distillation as ev
    native=torch.ones(12,1287);held=torch.zeros_like(native);held[:,768:]=native[:,768:]
    phase=torch.full((12,),study.event.FREE);phase[1::2]=study.event.DELAY
    x,reset=ev.live_controls(native,held,phase,2)
    assert torch.equal(x[:4,:256],native[:4,:256])and torch.equal(x[4:,:256],held[4:,:256])
    assert torch.equal(x[:,256:],held[:,256:])
    assert reset.tolist()==[False,False,True,False,False,False,False,False,True,False,True,False]


def test_snapshot_restores_private_tapes_physics_history_and_optimizer_evidence():
    cfg,teacher,student,env,pop,r=private_runtime('noise')
    study.train_unroll(student,teacher,pop,cfg,env,r,NoUpdate(student.lstm.parameters()),stage=0)
    opt=torch.optim.Adam(student.lstm.parameters(),lr=1e-4)
    # Synthetic nonempty optimizer evidence, without taking an optimization step.
    for p in student.lstm.parameters():
        opt.state[p]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,.02),exp_avg_sq=torch.full_like(p,.03))
    snap=study.snapshot(student,teacher,cfg,opt,env,r,1,'synthetic_protocol')
    metrics_a,probe_a=study.train_unroll(student,teacher,pop,cfg,env,r,NoUpdate(student.lstm.parameters()),stage=1)
    resumed=study.dist.trainable_student(teacher);opt2=torch.optim.Adam(resumed.lstm.parameters(),lr=1e-4)
    env2,pop2,r2=study.restore(snap,resumed,teacher,cfg,opt2,float('inf'))
    assert all(v['step']==7 and torch.all(v['exp_avg']==.02)and torch.all(v['exp_avg_sq']==.03)for v in opt2.state.values())
    metrics_b,probe_b=study.train_unroll(resumed,teacher,pop2,cfg,env2,r2,NoUpdate(resumed.lstm.parameters()),stage=1)
    assert metrics_a==metrics_b
    for key in probe_a:
        a,b=probe_a[key],probe_b[key]
        if isinstance(a,tuple):assert all(torch.equal(x,y)for x,y in zip(a,b))
        else:assert torch.equal(a,b),key
    assert torch.equal(r['tapes'].unit,r2['tapes'].unit)
    assert torch.equal(env.gen.get_state(),env2.gen.get_state())
