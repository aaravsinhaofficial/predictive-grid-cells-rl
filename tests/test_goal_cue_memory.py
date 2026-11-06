"""Synthetic networks and physical traces only; no trained model or study bank."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import pytest
import torch
from torch import nn

from code.env import ArenaEnv
from code.nets import PolicyNet
import goal_cue_memory as memory


class TinyPolicy(nn.Module):
    """Native 1287-wide concatenation with a deliberately small unit-test RNN."""
    def __init__(self):
        super().__init__(); self.nh = 8; self.n_actions = 6; self.extra_in = 0
        self.encoder = nn.Linear(3, 256); self.lstm = nn.LSTMCell(1287, self.nh)
        self.pi = nn.Linear(self.nh, 6); self.v = nn.Linear(self.nh, 1)

    def embed(self, rgb): return self.encoder(rgb.mean((2, 3)))
    zero_state = PolicyNet.zero_state
    _concat = PolicyNet._concat


def config(condition='pgc33'):
    cfg = json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    return dict(cfg, nh=8, n_envs=2, rollout=160)


def fixture(seed=7511, delay=3, angles=2, condition='pgc33'):
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed); net = TinyPolicy()
    cfg = config(condition); source = memory.make_source(cfg, 'cpu')
    plan = memory.TrialStreams(seed+1).draw(1, cfg, delay_steps=delay, angles_per_context=angles)
    return net, cfg, source, plan


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


def test_private_streams_restore_and_goal_draws_do_not_control_nuisance_tapes():
    cfg = config(); before = torch.get_rng_state().clone()
    streams = memory.TrialStreams(7601)
    plan = streams.draw(3, cfg, delay_steps=45, angles_per_context=4)
    saved = streams.snapshot(); restored = memory.TrialStreams(7601); restored.restore(saved)
    assert equal(streams.draw(2, cfg), restored.draw(2, cfg))
    assert torch.equal(torch.get_rng_state(), before)
    altered = memory.TrialStreams(7601); altered.generators['goal'].manual_seed(999)
    peer = altered.draw(3, cfg, delay_steps=45, angles_per_context=4)
    assert not torch.equal(plan['goal'], peer['goal'])
    for key in set(plan)-{'goal', 'goal_bearing', 'pickup_angle'}: assert equal(plan[key], peer[key]), key
    assert torch.equal(plan['pickup_angle'],plan['goal_bearing']+plan['pickup_offset'])
    assert torch.equal(peer['pickup_angle'],peer['goal_bearing']+peer['pickup_offset'])
    assert torch.equal(plan['delay'], torch.full((12,), 45))
    for start in (0,4,8):
        for key in ('pickup_offset','land_pos','stripe_phase','start','heading'):
            assert torch.equal(plan[key][start:start+4], plan[key][start:start+1].expand_as(plan[key][start:start+4]))
    corrupt = copy.deepcopy(plan); corrupt['goal'][0,0] += .01
    with pytest.raises(AssertionError, match='declared bearing'): memory.validate_plan(cfg, corrupt)


def test_training_delay_is_one_private_shared_draw_even_at_cap_zero():
    cfg=config();stream=memory.TrialStreams(7611);before=stream.snapshot()
    zero=stream.draw_training(3,cfg,delay_cap=0)
    assert zero['delay_steps']==0 and zero['delay_was_sampled'] and zero['delay_cap']==0
    after=stream.snapshot();assert after['delay_draws']==1
    assert not torch.equal(before['generators']['delay'],after['generators']['delay'])
    peer=memory.TrialStreams(7611);peer.restore(after)
    for cap in (15,45,15):
        a=stream.draw_training(3,cfg,delay_cap=cap);b=peer.draw_training(3,cfg,delay_cap=cap)
        assert equal(a,b) and 0<=a['delay_steps']<=cap
        assert torch.equal(a['delay'],torch.full((3,),a['delay_steps']))
    assert equal(stream.snapshot(),peer.snapshot()) and stream.delay_draws==4


def test_geometry_tolerance_preserves_literal_saved_values_without_regeneration():
    _,cfg,source,plan=fixture(7602)
    plan['goal'][0,0]+=1e-7
    preserved=plan['goal'].clone()
    memory.validate_plan(cfg,plan)
    batch=memory.prepare_batch(cfg,source,plan)
    assert torch.equal(plan['goal'],preserved) and torch.equal(batch['env'].goal,preserved)
    corrupt=copy.deepcopy(plan);corrupt['goal'][0,0]+=1e-4
    with pytest.raises(AssertionError,match='declared bearing'):memory.validate_plan(cfg,corrupt)


def test_setup_is_a_real_native_boundary_crossing_and_only_goal_code_transfers():
    _, cfg, source, plan = fixture(angles=8)
    global_rng = torch.get_rng_state().clone(); batch = memory.prepare_batch(cfg, source, plan)
    assert torch.equal(global_rng, torch.get_rng_state())
    trace = batch['setup']; n = len(plan['goal'])
    assert bool((trace['distance_before'] > .3).all())
    assert bool(((trace['hit_position']-plan['goal']).norm(dim=1) < .3).all())
    assert bool((trace['hit_velocity'].norm(dim=1) > .05).all())
    assert bool((trace['native_reward'] == 10).all()) and trace['learning_reward'] == 0
    native = ArenaEnv(n, vision='none', device='cpu', seed=900, img_noise=0,
                      noise_v=0, noise_r=0, stagger=False)
    native.goal[:] = plan['goal']; native.pos[:] = trace['position_before']
    native.hd[:] = plan['pickup_angle']; native.mdir[:] = torch.stack((native.hd.cos(),native.hd.sin()),1)
    _, reward, done, info = native.step(torch.full((n,),2))
    assert torch.equal(reward,trace['native_reward']) and not bool(done.any())
    for native_key, saved_key in [('hit_pos','hit_position'),('hit_dir','hit_direction'),('hit_vel','hit_velocity')]:
        assert torch.equal(info[native_key],trace[saved_key])
    expected = source.goal(info['hit_pos'],info['hit_dir'],info['hit_vel'])
    assert torch.equal(expected,batch['saved_goal'])
    assert not torch.equal(batch['saved_goal'][0],batch['saved_goal'][1])
    for key,value in batch['initial_physical_state'].items():
        if isinstance(value,torch.Tensor) and key != 'environment_rng':
            assert torch.equal(value,value[:1].expand_as(value)),key
    env=batch['env']; assert bool((env.pos==0).all() and (env.hd==0).all() and (env.n_goals==0).all())
    assert bool((env.ep_return==0).all() and (env.first_goal_t==-1).all() and (env.t==0).all())
    current=source(batch['obs']['pos'],batch['obs']['dir'],batch['obs']['vel'])
    assert torch.equal(current,current[:1].expand_as(current))
    assert torch.equal(batch['obs']['rgb'],batch['obs']['rgb'][:1].expand_as(batch['obs']['rgb']))


def test_same_physics_and_ordinary_goal_sham_across_populations():
    _,cfg,pgc,plan=fixture(angles=4)
    a=memory.prepare_batch(cfg,pgc,plan)
    gc_cfg=config('gc100'); gc=memory.make_source(gc_cfg,'cpu'); b=memory.prepare_batch(gc_cfg,gc,plan)
    assert equal(a['setup'],b['setup']) and equal(a['initial_physical_state'],b['initial_physical_state'])
    assert equal(a['obs'],b['obs'])
    assert torch.equal(b['saved_goal'],b['ordinary_goal'])
    assert torch.equal(a['ordinary_goal'],b['ordinary_goal'])


def test_native_step_only_advances_selected_free_rows_and_never_respawns():
    _,cfg,source,plan=fixture()
    batch=memory.prepare_batch(cfg,source,plan);env=batch['env']
    # An explicit physical unit fixture places one row immediately outside a
    # goal; the other remains waiting. This is not a proposed study start.
    env.pos[0]=env.goal[0]-torch.tensor([.302,0.]);env.hd[0]=0;env.mdir[0]=torch.tensor([1.,0.])
    before=memory.physical_snapshot(env)
    active=torch.tensor([True,False]);actions=torch.tensor([2,-1]);motor=torch.zeros(3,2)
    _,reward,done,info=memory.step_free(env,actions,active,motor)
    assert reward.tolist()==[10.,0.] and not bool(done.any())
    assert info['hit'].tolist()==[True,False] and env.t.tolist()==[1,0]
    assert torch.equal(env.pos[0],info['hit_pos'][0])
    after=memory.physical_snapshot(env)
    for key,value in before.items():
        if isinstance(value,torch.Tensor) and key!='environment_rng': assert torch.equal(value[1],after[key][1]),key
    stopped=copy.deepcopy(after)
    for _ in range(2):
        _,r,_,i=memory.step_free(env,torch.tensor([-1,-1]),torch.tensor([False,False]),motor)
        assert not bool(r.any() or i['hit'].any())
    assert equal(stopped,memory.physical_snapshot(env))


@pytest.mark.parametrize('delay',[0,15,45])
def test_clock_fits_160_and_wait_is_not_a_policy_or_physics_action(delay):
    net,cfg,source,plan=fixture(7522,delay=delay)
    roll,rows,audit,evidence=memory.collect(net,cfg,source,plan,cue_mode='brief')
    free_start=15+delay
    assert audit['wait_updates']==2*free_start and audit['forced_actions']==0
    assert audit['active_recurrent_updates']+audit['padded_slots']==320
    assert audit['active_physics_updates']==sum(r['free_steps']for r in rows)
    assert audit['active_physics_updates']+audit['wait_updates']==audit['active_recurrent_updates']
    assert audit['setup_learning_reward']==audit['forced_rewards']==0
    for key in ('pa','action'):
        assert bool((torch.stack(roll[key])[:free_start]==-1).all())
    for key in ('pr','reward','raw_reward','physics_active','policy_mask'):
        assert not bool(torch.stack(roll[key])[:free_start].any())
    for key in ('rgb','code','observation_position','observation_heading','observation_direction','observation_velocity','motors_before_action'):
        values=torch.stack(roll[key])[:free_start]
        assert torch.equal(values,values[:1].expand_as(values)),key
        assert torch.equal(values[:,0],values[:,1]),key
    supplied=torch.stack(roll['goal_input'])
    assert torch.equal(supplied[:15],roll['saved_goal'][None].expand_as(supplied[:15]))
    assert not bool(supplied[15:].any())
    assert bool((torch.stack(roll['terminal']).sum(0)==1).all())
    assert torch.equal(evidence['final_physical_state']['t'],torch.tensor([r['free_steps']for r in rows]))


def test_recorded_goal_inputs_replay_exactly_and_external_cache_is_not_read():
    net,cfg,source,plan=fixture(7531)
    roll,_,_,_=memory.collect(net,cfg,source,plan,cue_mode='brief')
    logits,values=memory.replay(net,roll)
    assert torch.equal(logits.detach(),torch.stack(roll['logits']))
    assert torch.equal(values.detach(),torch.stack(roll['values']))
    roll['saved_goal'].fill_(1e8);roll['physical_goal'].fill_(1e8)
    other_logits,other_values=memory.replay(net,roll)
    assert torch.equal(logits,other_logits) and torch.equal(values,other_values)
    roll['goal_input'][14]=roll['goal_input'][14].clone().requires_grad_(True)
    logits,_=memory.replay(net,roll)
    gradient=torch.autograd.grad(logits[18,0,2],roll['goal_input'][14])[0]
    assert float(gradient[0].abs().sum())>0


def test_postcue_erasure_happens_before_first_zero_input_and_removes_goal_history():
    net,cfg,source,plan=fixture(7540,delay=3)
    roll,_,audit,ev=memory.collect(net,cfg,source,plan,cue_mode='brief',erase_after_cue=True)
    reset=torch.stack(roll['reset_before']);assert reset.sum()==2 and bool(reset[15].all())
    assert not torch.equal(ev['cue_end_state'][0],ev['cue_end_state'][1])
    for key in ('input','state','state_before_input','probabilities','logits','value'):
        assert torch.equal(ev['first_free'][0][key],ev['first_free'][1][key]),key
    assert audit['erased_states']==2
    logits,_=memory.replay(net,roll);assert torch.equal(logits.detach(),torch.stack(roll['logits']))
    roll['goal_input'][14]=roll['goal_input'][14].clone().requires_grad_(True)
    logits,_=memory.replay(net,roll)
    gradient=torch.autograd.grad(logits[18,0,2],roll['goal_input'][14])[0]
    assert not bool(gradient.any())


def test_first_free_records_incoming_and_postinput_states_without_changing_timing():
    net,cfg,source,plan=fixture(7541,delay=0)
    roll,_,_,evidence=memory.collect(net,cfg,source,plan,cue_mode='brief',erase_after_cue=True)
    recorded=[evidence['first_free'][i] for i in range(len(plan['goal']))]
    inputs=torch.stack([r['input'] for r in recorded])
    incoming=torch.stack([r['state_before_input'] for r in recorded])
    h,c=incoming.split(cfg['nh'],dim=1)
    with torch.no_grad():post=torch.cat(net.lstm(inputs,(h,c)),1)
    for i,saved in evidence['first_free'].items():
        assert not bool(saved['state_before_input'].any())
        assert torch.equal(saved['state'],post[i])
        assert saved['state_timing'].startswith('After processing')
        assert saved['state_before_input_timing'].startswith('After any scheduled')


def test_padding_and_wait_masks_preserve_full_bptt_without_fictitious_action_loss():
    net,cfg,source,plan=fixture(7550)
    roll,_,_,_=memory.collect(net,cfg,source,plan,cue_mode='brief')
    roll['bootstrap'].fill_(1e8)
    targets=memory.returns(roll,cfg['gamma']);assert float(targets.max())<=1.
    logits,values=memory.replay(net,roll);logits.retain_grad();values.retain_grad()
    loss,_=memory.losses(logits,values,targets,roll,cfg);loss.backward()
    active=torch.stack(roll['active']);policy=torch.stack(roll['policy_mask'])
    assert not bool(logits.grad[~policy].any()) and not bool(values.grad[~active].any())
    assert float(values.grad[active&~policy].abs().sum())>0
    for component in (net.encoder,net.lstm,net.pi,net.v):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())


def test_no_cue_cannot_read_goal_cache_and_continuous_goal_remains_available():
    net,cfg,source,plan=fixture(7560,delay=0,condition='gc100')
    no,_,_,evidence=memory.collect(net,cfg,source,plan,cue_mode='no_cue')
    assert not bool(torch.stack(no['goal_input']).any())
    assert torch.equal(evidence['first_free'][0]['state'],evidence['first_free'][1]['state'])
    normal,rows,_,_=memory.collect(net,cfg,source,plan,cue_mode='continuous')
    ordinary,peer,_,_=memory.collect(net,cfg,source,plan,cue_mode='continuous',goal_kind='ordinary')
    assert rows==peer and equal(normal,ordinary)
    for supplied,active in zip(normal['goal_input'],normal['active']):
        assert torch.equal(supplied[active],normal['saved_goal'][active])
        assert not bool(supplied[~active].any())


def test_first_free_hit_ends_collector_and_preserves_preteleport_position():
    net,cfg,source,plan=fixture(7570,delay=0,angles=1)
    # Synthetic deterministic straight return, never a sampled study outcome.
    plan['goal_bearing'].zero_();plan['goal'][:]=torch.tensor([.75,0.]);plan['motor'].zero_()
    plan['pickup_angle']=plan['goal_bearing']+plan['pickup_offset']
    with torch.no_grad():
        net.pi.weight.zero_();net.pi.bias.fill_(-100);net.pi.bias[2]=100
    roll,rows,audit,evidence=memory.collect(net,cfg,source,plan)
    assert rows[0]['hit'] and 1<rows[0]['free_steps']<90
    terminal=torch.stack(roll['terminal'])[:,0].nonzero().flatten()
    assert len(terminal)==1;end=int(terminal[0])
    assert float(roll['raw_reward'][end][0])==10 and float(roll['reward'][end][0])==1
    assert not bool(torch.stack(roll['active'])[end+1:].any())
    assert not bool(torch.stack(roll['reward'])[end+1:].any())
    positions=torch.stack(roll['position'])[:,0]
    assert torch.equal(positions[end:],positions[end].expand_as(positions[end:]))
    assert float((positions[end]-plan['goal'][0]).norm())<.3
    expected_path=float((positions[1:]-positions[:-1]).norm(dim=1).sum())
    assert abs(rows[0]['free_path_m']-expected_path)<1e-7
    assert audit['free_hits']==1 and evidence['final_physical_state']['n_goals'].tolist()==[1]


def test_actual_donor_and_wrong_goal_overrides_are_recorded_replayed_and_leave_target_fixed():
    net,cfg,source,plan=fixture(7580,delay=3,angles=4)
    original,rows,_,ev=memory.collect(net,cfg,source,plan,cue_mode='brief')
    incoming=torch.stack([ev['first_free'][i]['state_before_input'] for i in range(4)])
    sham,sham_rows,_,sham_ev=memory.collect(net,cfg,source,plan,cue_mode='brief',first_free_state_override=incoming)
    assert rows==sham_rows
    for key in ('logits','values','action','position','reward','goal_input'):
        assert equal(original[key],sham[key]),key
    donor=incoming.roll(-1,0)
    edited,_,audit,edited_ev=memory.collect(net,cfg,source,plan,cue_mode='brief',first_free_state_override=donor)
    assert audit['first_free_state_override_used'] and torch.stack(edited['state_override_mask']).sum()==4
    assert bool(torch.stack(edited['state_override_mask'])[18].all())
    assert torch.equal(edited['physical_goal'],original['physical_goal'])
    for i in range(4):
        assert torch.equal(edited_ev['first_free'][i]['state_before_input'],donor[i])
        assert torch.equal(edited_ev['first_free'][i]['state_before_intervention'],incoming[i])
    logits,values=memory.replay(net,edited)
    assert torch.equal(logits.detach(),torch.stack(edited['logits']))
    assert torch.equal(values.detach(),torch.stack(edited['values']))
    wrong=original['native_saved_goal'].roll(-1,0)
    switched,_,audit,_=memory.collect(net,cfg,source,plan,cue_mode='brief',goal_override=wrong)
    assert audit['goal_override_used'] and torch.equal(switched['saved_goal'],wrong)
    assert torch.equal(switched['physical_goal'],original['physical_goal'])
    assert torch.equal(switched['goal_input'][0],wrong) and not bool(torch.stack(switched['goal_input'])[15:].any())
    logits,values=memory.replay(net,switched)
    assert torch.equal(logits.detach(),torch.stack(switched['logits']))
