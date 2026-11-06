"""Timing, sensory, replay, and matched-environment invariants for vision dropout."""
import torch
from code.common import CodeSource, build_population, build_env
from code.env import ArenaEnv
from scripts import rl_visual_memory as study
from tests.test_rl_goal_memory import TinyPolicy, OneHitEnv, config


def test_clock_is_visible_before_reward_and_for_exactly15_updates_after():
    clock=study.VisualClock(1,'cpu',12)
    no=torch.tensor([False]);yes=~no
    for _ in range(35):
        assert not clock.mask().any()
        assert not clock.advance(no,no,1).any()
    clock.advance(yes,no,1)
    for remaining in range(15,0,-1):
        assert clock.remaining.item()==remaining
        assert not clock.mask().any()
        enter=clock.advance(no,no,1)
        assert enter.item()==(remaining==1)
    assert clock.mask().item() and clock.remaining.item()==1
    assert not clock.advance(no,no,1).item()
    assert not clock.mask().any() and clock.remaining.item()==15


def test_reward_refresh_episode_end_and_stage_boundary():
    clock=study.VisualClock(2,'cpu',13)
    clock.known[:]=True;clock.remaining[:]=1
    entering=clock.advance(torch.tensor([True,True]),torch.tensor([False,True]),60)
    assert not entering.any() and not clock.mask().any()
    assert clock.known.tolist()==[True,False]
    assert clock.remaining.tolist()==[15,15]
    clock.known[:]=True;clock.blind[:]=True;clock.remaining[:]=3
    clock.advance(torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),60)
    assert clock.remaining.tolist()==[2,2]  # a new stage does not extend an existing gap


def test_duration_bounds_coupled_groups_and_independent_rng():
    before=torch.get_rng_state().clone()
    clock=study.VisualClock(32,'cpu',14,groups=3)
    no=torch.zeros(96,dtype=torch.bool)
    clock.advance(~no,no,4)
    for _ in range(15):clock.advance(no,no,4)
    assert clock.remaining.min()>=2 and clock.remaining.max()<=4
    assert torch.equal(clock.remaining[:32],clock.remaining[32:64])
    assert torch.equal(clock.remaining[:32],clock.remaining[64:])
    assert torch.equal(before,torch.get_rng_state())


def test_rgb_only_mask_leaves_original_observation_unchanged():
    clock=study.VisualClock(2,'cpu',15)
    clock.known[:]=True;clock.blind[:]=True
    original=torch.arange(24.).reshape(2,3,2,2)+1
    copy=original.clone()
    result=study.masked_rgb(original,clock,torch.tensor([True,False]))
    assert not result[0].any() and torch.equal(result[1],original[1])
    assert torch.equal(original,copy)


def test_original_environment_and_matched_physics_noise_are_preserved():
    cfg=config()
    ordinary=build_env(cfg,2,71,'cpu',stagger=False)
    assert type(ordinary) is ArenaEnv
    env=study.MatchedArena(cfg,2,3,71,'cpu')
    obs=env.reset_all()
    for step in range(20):
        for value in (env.goal,env.pos,env.hd,obs['rgb']):
            assert all(torch.equal(value[:2],value[j*2:(j+1)*2]) for j in range(1,3))
        obs,_,_,_=env.step(torch.tensor([step%6,(step+2)%6]).repeat(3))
    assert study.MatchedArena.step is ArenaEnv.step
    assert study.MatchedArena.render is ArenaEnv.render


def test_actor_replay_bootstrap_mask_and_native_goal_are_exact():
    class WhiteEnv(OneHitEnv):
        def observe(self):
            out=super().observe();out['rgb'].fill_(1.)
            return out
    torch.manual_seed(16)
    cfg,env,net=config(),WhiteEnv(),TinyPolicy()
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    goal=study.NativeGoal(2,512,'cpu');clock=study.VisualClock(2,'cpu',17)
    batch=study.collect_rollout(net,env,source,cfg,env.reset_all(),net.zero_state(2,'cpu'),
        goal,clock,torch.full((2,),-1),torch.zeros(2),16,1)
    roll,state0,_,boot_rgb=batch[:4]
    assert all(torch.equal(rgb,torch.ones_like(rgb)) for rgb in roll['rgb'])
    assert not boot_rgb.any()  # first blank input is update16 after reward at0
    hitcode=source.goal(torch.tensor([[.4,.1]]).repeat(2,1),torch.tensor([[1.,0.]]).repeat(2,1),
                        torch.tensor([[.35,0.]]).repeat(2,1))
    assert not roll['goal'][0].any()
    assert all(torch.equal(g,hitcode) for g in roll['goal'][1:])
    assert torch.equal(goal.saved,hitcode)  # no goal flash: snapshot remains available
    logits,_=study.replay(net,roll,state0)
    torch.testing.assert_close(logits,torch.stack(roll['actor_logits']),atol=2e-7,rtol=2e-6)


def test_full_evaluation_reset_clock_and_grid_sham(monkeypatch):
    parent=study.MatchedArena
    class PickupArena(parent):
        def reset_all(self):
            super().reset_all();self.pos[:]=self.goal
            return self.observe()
    monkeypatch.setattr(study,'MatchedArena',PickupArena)
    cfg=config();cfg.update(cond='gc100',pgc_frac=0.,episode_len=40)
    result=study.evaluate(TinyPolicy(),cfg,device='cpu',seed=72,episodes=2,n_envs=2,max_gap=3)
    assert result['all_first_discoveries_identical']
    assert result['grid_replacement_is_exact_noop']
    assert result['state_reset_counts']['blind_reset']>=2
    assert result['state_reset_counts']['seen_reset']>=2
    assert result['state_reset_counts']['blind']==0
    assert result['blind_fraction_of_post_discovery_updates']['blind']>0
    assert result['blind_fraction_of_post_discovery_updates']['seen']==0
    assert result['variants']['blind']['rows']==result['variants']['blind_current_grid']['rows']
    assert result['variants']['seen']['rows']==result['variants']['seen_current_grid']['rows']
