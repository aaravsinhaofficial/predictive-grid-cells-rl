"""Independent synthetic native-physics checks; no policy weights or study bank."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from code.env import ArenaEnv
import goal_cue_memory as memory


def test_mixed_selected_rows_match_all_six_native_actions_with_noise_inertia_and_wall():
    cfg=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    cfg=dict(cfg,vision='none')
    source=memory.make_source(cfg,'cpu')
    plan=memory.TrialStreams(782303).draw(8,cfg,delay_steps=15)
    env=memory.prepare_batch(cfg,source,plan)['env']
    env.goal.fill_(5.)  # Physical-equation fixture excludes pickups, not a task goal.
    env.pos[:]=torch.tensor([[1.1699,0],[.2,.1],[.1,.2],[.3,-.2],[-.3,.2],[.2,-.1],[0,.3],[-.2,-.2]])
    env.hd[:]=torch.tensor([0.,.1,.4,-.6,.9,1.2,-1.3,2.1])
    env.vf[:]=torch.tensor([.15,-.1,.12,.13,.14,-.16,.17,.18])
    env.vs[:]=torch.tensor([.03,.04,-.05,.06,.07,.08,-.09,.10])
    env.vr[:]=torch.tensor([.11,.12,.13,-.14,.15,.16,-.17,.18])
    active=torch.tensor([True,False,True,True,False,True,True,True])
    indices=active.nonzero().flatten()
    actions=torch.tensor([2,-1,5,0,-1,1,3,4])
    tape=torch.tensor([[.1,.2,-.3,.4,.5,-.6,.7,.8],[-.8,.7,.6,-.5,.4,.3,-.2,.1],[.7,-.6,.5,.4,-.3,.2,.1,-.8]])
    before={k:v.clone()for k,v in memory.physical_snapshot(env).items()if isinstance(v,torch.Tensor)}
    native=ArenaEnv(len(indices),device='cpu',vision='none',img_noise=0.,seed=912,
        half=cfg['arena_half'],episode_len=cfg['episode_len'],action_repeat=cfg['action_repeat'],
        v_max=cfg['v_max'],rot_max=cfg['rot_max'],goal_radius=cfg['goal_radius'],stagger=False)
    for field in memory.STATE_FIELDS:setattr(native,field,getattr(env,field)[indices].clone())
    draws=iter(tape[:,indices]);calls=[]
    def native_noise(*shape):
        assert shape==(len(indices),);calls.append(shape);return next(draws)
    native._randn=native_noise
    _,expected_reward,expected_done,expected_info=native.step(actions[indices])
    _,reward,done,info=memory.step_free(env,actions,active,tape)
    assert len(calls)==3 and bool(expected_info['hit_wall'].any())
    assert torch.equal(reward[indices],expected_reward)and torch.equal(done[indices],expected_done)
    for key in ('hit','hit_wall','hit_pos','hit_dir','hit_vel'):
        assert torch.equal(info[key][indices],expected_info[key]),key
    after=memory.physical_snapshot(env)
    for field in memory.STATE_FIELDS:
        assert torch.equal(getattr(env,field)[indices],getattr(native,field)),field
        if field!='goal':assert torch.equal(getattr(env,field)[~active],before[field][~active]),field
    assert torch.equal(after['environment_rng'],before['environment_rng'])
    assert env.n==8 and env.t.tolist()==active.long().tolist()
