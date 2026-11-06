"""Independent timing, physics, loss, cue and exact replay invariants."""
import copy
import math
import numpy as np
import torch
from code.common import CodeSource, build_population, build_env
from scripts import rl_blind_reorientation as task
from scripts import rl_blind_reorientation_train as training
from scripts.rl_visual_memory_extend import tree_equal
from tests.test_rl_visual_memory_noise import HookablePolicy
from tests.test_rl_goal_memory import config


def cfg_for_test():
    cfg=config();cfg.update(rollout=160,episode_len=600)
    return cfg


def test_exact_maximum_timeline_glimpse_is_first_delay_input_and_forced_pa():
    clock=task.ReorientationClock(1,'cpu',7)
    clock.known[:]=True
    clock.plans.fill_(np.nextafter(1.,0.))  # rounding-to-float32 still must respect10/10/60
    clock.begin_segment();no=torch.tensor([False]);phases=[];glimpses=[];actions=[];forced=[]
    for t in range(160):
        clock.prepare(160-t,torch.tensor([600-t]),60)
        phases.append(int(clock.phase));glimpses.append(bool(clock.glimpse()))
        actions.append(int(clock.action(torch.tensor([4]))));forced.append(bool(clock.forced()))
        clock.advance(no,no)
    expected=([task.VISIBLE]*15+[task.TURN_A]*10+[task.TURN_B]*10+
              [task.PROBE]*4+[task.DELAY]*60+[task.FREE]*30+[task.READY]*31)
    assert phases==expected
    assert [i for i,g in enumerate(glimpses) if g]==[39]
    assert actions[15:35]==[1]*20 and actions[35:39]==[2]*4
    assert actions[39:99]==[0]*60 and actions[99:129]==[4]*30
    assert forced==[task.TURN_A<=p<=task.DELAY for p in expected]
    assert clock.stats['admitted'].item()==clock.stats['completed'].item()==1
    assert clock.stats['coverage_violations'].item()==0
    clock.begin_segment()  # no event crosses a credit-window boundary


def test_schedule_private_rng_per_event_pairing_caps_and_interrupts():
    torch_before=torch.get_rng_state().clone();np_before=np.random.get_state()
    clock=task.ReorientationClock(3,'cpu',8,groups=2)
    assert torch.equal(clock.plans[:3],clock.plans[3:])
    assert not torch.equal(clock.plans[0],clock.plans[1])
    clock.known[:]=True;clock.begin_segment()
    clock.prepare(128,torch.full((6,),600),60)
    assert not clock.stats['admitted'].any() and clock.stats['wait_segment'].sum()==6
    clock.prepare(160,torch.full((6,),128),60)
    assert not clock.stats['admitted'].any() and clock.stats['wait_episode'].sum()==6
    clock.event_count[:]=task.EVENT_CAP
    clock.prepare(160,torch.full((6,),600),60)
    assert clock.stats['wait_cap'].sum()==6
    clock.event_count.zero_();clock.prepare(160,torch.full((6,),600),15)
    assert clock.duration_delay.min()>=8 and clock.duration_delay.max()<=15
    assert torch.equal(clock.duration_delay[:3],clock.duration_delay[3:])
    hit=torch.tensor([True,True,False,False,False,False]);done=torch.tensor([False,True,False,False,False,False])
    clock.advance(hit,done)
    assert clock.phase[:2].tolist()==[task.READY,task.READY]
    assert clock.known[:2].tolist()==[True,False]
    assert clock.event_count[:2].tolist()==[1,0]
    assert clock.used_segment[:2].tolist()==[True,True]
    assert clock.stats['interrupted_reward'][0]==1 and clock.stats['interrupted_done'][1]==1
    assert torch.equal(torch.get_rng_state(),torch_before)
    np_after=np.random.get_state()
    assert np_before[0]==np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]


def test_forced_policy_and_entropy_gradients_are_zero_but_critic_is_trained():
    cfg=cfg_for_test();torch.manual_seed(9)
    logits=torch.randn(4,2,6,requires_grad=True);values=torch.randn(4,2,requires_grad=True)
    mask=torch.tensor([[1.,1.],[0.,0.],[0.,1.],[1.,1.]])
    targets=values.detach()+1;actions=torch.zeros(4,2,dtype=torch.long)
    total,policy,value,entropy=task.losses(logits,values,targets,actions,mask,cfg)
    assert torch.equal(total,(100/160)*(policy+cfg['value_coef']*value-cfg['entropy']*entropy))
    total.backward()
    assert not logits.grad[mask==0].any() and logits.grad[mask==1].abs().sum()>0
    assert values.grad[mask==0].abs().min()>0
    # Entropy alone must not update forced actions, independently of advantage.
    logits2=logits.detach().clone().requires_grad_(True)
    _,_,_,entropy2=task.losses(logits2,values.detach(),targets,actions,mask,cfg)
    entropy2.backward()
    assert not logits2.grad[mask==0].any()


def test_endpoint_pulse_brakes_only_predictive_slots_and_snapshot_hold():
    cfg=cfg_for_test();n=len(task.VARIANTS);cfg['n_envs']=n
    env=build_env(cfg,n,13,'cpu',stagger=False);obs=env.reset_all()
    for key in obs:obs[key][:]=obs[key][0].clone()
    obs['rgb'].fill_(1.);obs['pos'][:]=torch.tensor([.2,-.3])
    obs['dir'][:]=torch.tensor([.6,.8]);obs['vel'][:]=torch.tensor([.06,.08])
    env.vf.fill_(.2);env.vs.fill_(.1);env.vr.fill_(.5)
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    clock=task.ReorientationClock(1,'cpu',14,groups=n)
    clock.phase[:]=task.DELAY;clock.age.zero_()
    inputs=task.EventInputs(n,512,'cpu',task.VARIANTS,1)
    goal_before=source.goal(obs['pos'],obs['dir'],obs['vel']).clone()
    rgb,code,blind,glimpse=inputs.prepare(env,obs,clock,source)
    ordinary=source.pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
    native=source(obs['pos'],obs['dir'],obs['vel'])
    assert torch.equal(code[0],native[0]) and torch.equal(code[1],ordinary[1])
    gc=~source.pop.is_pgc
    assert torch.equal(code[:,gc],native[:,gc])
    assert torch.equal(code[2],source(obs['pos'],task.rotate(obs['dir'],torch.full((n,),math.pi/3)),
                                     task.rotate(obs['vel'],torch.full((n,),math.pi/3)))[2])
    assert torch.equal(goal_before,source.goal(obs['pos'],obs['dir'],obs['vel']))
    assert not env.vf.any() and not env.vs.any() and not env.vr.any()
    assert not rgb[:6].any() and rgb[6:].min()==1
    assert blind.all() and glimpse.all()
    clock.age[:]=1;obs['pos']+=.2;obs['dir']=-obs['dir']
    _,held,_,_=inputs.prepare(env,obs,clock,source)
    assert torch.equal(held,ordinary),'Later blind movement cannot refresh held position or predictive cue'


def test_actual_forward_probe_endpoint_is_native_not_heading_substitution():
    cfg=cfg_for_test();cfg['img_noise']=0.
    env=build_env(cfg,2,25,'cpu',stagger=False);obs=env.reset_all()
    env.pos[:]=torch.tensor([0.,0.]);env.goal[:]=torch.tensor([.9,.9]);env.hd[:]=.7
    env.noise_v=0.;env.noise_r=0.;task.brake(env,torch.ones(2,dtype=torch.bool))
    for _ in range(4):obs,_,_,_=env.step(torch.full((2,),2))
    endpoint=obs['pos'].clone();velocity=obs['vel'].clone();direction=obs['dir'].clone()
    assert .04<float(endpoint[0].norm())<.05
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    clock=task.ReorientationClock(2,'cpu',26);clock.phase[:]=task.DELAY
    inputs=task.EventInputs(2,512,'cpu')
    _,pulse,_,_=inputs.prepare(env,obs,clock,source)
    assert torch.equal(pulse,source(endpoint,direction,velocity))
    assert torch.equal(env.pos,endpoint) and torch.equal(obs['vel'],velocity)
    assert not env.vf.any() and not env.vr.any()


def test_noisy_rollout_replay_actual_actions_and_exact_snapshot_resume():
    torch.set_num_threads(2);torch.manual_seed(29)
    cfg=cfg_for_test();net=HookablePolicy()
    runtime=task.fresh_runtime(net,cfg,'cpu',stagger=False)
    runtime['clock'].known[:]=True
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    roll,state0,_,_=task.collect(net,cfg,source,runtime,4)
    assert any(bool(b.any()) for b in roll['blind'])
    for t in range(1,160):
        assert torch.equal(roll['pa'][t],torch.where(roll['d'][t-1].bool(),
            torch.full_like(roll['a'][t-1],-1),roll['a'][t-1]))
    for phase,mask,eps in zip(roll['phase'],roll['policy_mask'],roll['recurrent_noise']):
        forced=(phase>=task.TURN_A)&(phase<=task.DELAY)
        assert torch.equal(mask,(~forced).float())
        for e in eps:assert not e[phase<task.TURN_A].any()
    logits,_=task.replay(net,roll,state0)
    torch.testing.assert_close(logits,torch.stack(roll['actor_logits']),atol=2e-7,rtol=2e-6)
    assert all(not e.any() for e in roll['bootstrap_recurrent_noise'])
    opt=torch.optim.RMSprop(net.parameters(),lr=cfg['lr'],alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    task.update(net,cfg,opt,source,runtime,4)
    checkpoint=dict(policy=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                    snapshot=task.capture_runtime(runtime,'cpu'))
    expected=task.update(net,cfg,opt,source,runtime,15)
    expected_runtime=task.capture_runtime(runtime,'cpu')
    restored=HookablePolicy();restored.load_state_dict(checkpoint['policy'])
    restored_opt=torch.optim.RMSprop(restored.parameters(),lr=cfg['lr'],alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    restored_opt.load_state_dict(checkpoint['optimizer'])
    restored_runtime=task.fresh_runtime(restored,cfg,'cpu',stagger=False)
    task.restore_runtime(restored_runtime,checkpoint['snapshot'],'cpu')
    actual=task.update(restored,cfg,restored_opt,source,restored_runtime,15)
    assert expected==actual
    assert tree_equal(expected_runtime,task.capture_runtime(restored_runtime,'cpu'))
    assert tree_equal(net.state_dict(),restored.state_dict()) and tree_equal(opt.state_dict(),restored_opt.state_dict())


def test_evaluation_grid_shams_first_discovery_and_free_window_accounting(monkeypatch):
    parent=task.MatchedArena
    class FirstHitArena(parent):
        def reset_all(self):
            super().reset_all();self.pos[:]=self.goal
            return self.observe()
    monkeypatch.setattr(task,'MatchedArena',FirstHitArena)
    cfg=cfg_for_test();cfg.update(cond='gc100',pgc_frac=0.,episode_len=180)
    result=task.evaluate(HookablePolicy(),cfg,device='cpu',episodes=2,n_envs=2,max_delay=4)
    assert result['all_first_discoveries_identical'] and result['grid_pulse_controls_exact_noop']
    assert sum(result['schedule']['admitted'])>0 and sum(result['schedule']['coverage_violations'])==0
    for variant,item in result['free_window'].items():
        assert item['free_updates']==sum(r['free_updates'] for r in item['rows'])
        assert item['free_hits']==sum(r['goal_hit'] for r in item['rows'])
        assert abs(item['free_progress_m']-sum(r['progress_m'] for r in item['rows']))<1e-5
    assert result['free_window']['blind_reset_after_glimpse']['resets']>0
    assert result['free_window']['blind_reset_every_blind']['resets']>result['free_window']['blind_reset_after_glimpse']['resets']
    assert result['free_window']['blind_native']['resets']==0


def test_reset_after_glimpse_occurs_after_cue_update_before_next_input(monkeypatch):
    parent_env=task.MatchedArena;parent_clock=task.ReorientationClock;holder={}
    class FirstHitArena(parent_env):
        def reset_all(self):
            super().reset_all();self.pos[:]=self.goal
            return self.observe()
    class Clock(parent_clock):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);holder['clock']=self
    class TracePolicy(HookablePolicy):
        def __init__(self):
            super().__init__();self.inputs=[]
        def step(self,rgb,code,goal,pa,pr,state,extra=None):
            self.inputs.append(dict(state=tuple(s.clone() for s in state),
                phase=holder['clock'].phase.clone(),glimpse=holder['clock'].glimpse().clone()))
            return super().step(rgb,code,goal,pa,pr,state,extra)
    monkeypatch.setattr(task,'MatchedArena',FirstHitArena);monkeypatch.setattr(task,'ReorientationClock',Clock)
    cfg=cfg_for_test();cfg.update(cond='gc100',pgc_frac=0.,episode_len=180)
    net=TracePolicy();task.evaluate(net,cfg,device='cpu',episodes=1,n_envs=1,max_delay=4)
    for vision in ('blind','seen'):
        reset_after=task.VARIANTS.index(f'{vision}_reset_after_glimpse')
        every=task.VARIANTS.index(f'{vision}_reset_every_blind')
        cue_times=[i for i,row in enumerate(net.inputs) if row['glimpse'][reset_after]]
        assert cue_times
        for i in cue_times:
            assert all(s[reset_after].abs().sum()>0 for s in net.inputs[i]['state'])
            assert all(not s[reset_after].any() for s in net.inputs[i+1]['state'])
        for row in net.inputs:
            if row['phase'][every]>=task.TURN_A:
                assert all(not s[every].any() for s in row['state'])


def test_prespecified_frame_accounting_and_loss_window():
    assert task.MAX_WINDOW==129 and task.ROLLOUT==160
    assert training.STAGE_FRAMES==10_014_720 and training.TOTAL_FRAMES==40_058_880
    assert training.WARMUP*training.FRAMES_PER_UPDATE==4_096_000
