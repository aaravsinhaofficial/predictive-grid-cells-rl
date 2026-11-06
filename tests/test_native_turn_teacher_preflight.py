"""Synthetic/private physical fixtures only; no trained-policy outcome."""
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_turn_teacher_preflight as audit


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lstm=torch.nn.LSTMCell(1287,8);self.pi=torch.nn.Linear(8,6)
    def embed(self,rgb):
        means=rgb.mean((2,3));return means.repeat(1,86)[:,:256]
    def _concat(self,emb,code,goal,pa,pr,extra):
        onehot=torch.nn.functional.one_hot(pa.clamp_min(0),6).float()*(pa>=0)[:,None]
        return torch.cat((emb,code,goal,onehot,pr[:,None]),1)
    def zero_state(self,n,device):return torch.zeros(n,8,device=device),torch.zeros(n,8,device=device)


def fixture(condition='pgc33'):
    cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(637901);net=TinyNet().eval().requires_grad_(False)
    pop=audit.build_population(cfg,'cpu')
    goal=torch.tensor([[.75,.1]]);direction=torch.tensor([[1.,0.]])
    x=torch.zeros(1287);x[768:1280]=pop.activity(goal,direction,torch.zeros(1,2))[0]
    ctx=dict(context_id=0,slot=0,step=15,endpoint=torch.tensor([.02,-.04]),goal_center=goal[0],
        direction=direction[0],heading=torch.tensor(.4),native_x=x,initial_hc=torch.zeros(16),
        land_pos=torch.tensor([-.6,.3,.8,-.2]),stripe_phase=torch.tensor([.0,.25,.5,.75]),
        continuation_uniform=torch.rand(30,8,generator=torch.Generator().manual_seed(638901)))
    ctx['native_bookkeeping']={k:torch.tensor(0.)for k in audit.BOOKKEEPING}
    ctx['native_bookkeeping'].update(t=torch.tensor(15),n_goals=torch.tensor(1),first_goal_t=torch.tensor(10))
    return cfg,net,pop,ctx


def test_motor_formula_matches_native_inertia_and_strafe_signs():
    cfg,_,_,ctx=fixture()
    for action,column in ((0,2),(1,2),(2,0),(3,0),(4,1),(5,1)):
        env=audit.make_env(cfg,ctx,1);env.pos.zero_();env.goal.fill_(10.);env.hd.zero_();audit.brake(env)
        for _ in range(4):env.step(torch.tensor([action]))
        measured=float(env.hd[0])if column==2 else float(env.pos[0,column])
        assert abs(measured-audit.native_motion(cfg,[action]*4,column))<2e-7


def test_true_antipodal_alias_and_postcue_staleness_all_modes_populations():
    for condition in ('gc100','pgc33'):
        cfg,net,pop,ctx=fixture(condition)
        for kind in audit.KINDS:
            raw,env,checks=audit.construct_history(net,pop,cfg,ctx,kind)
            assert checks['antipodal_error_max_rad']<2e-5
            assert audit.paired_equal(raw['student_native_x'][-1])
            assert all(audit.paired_equal(v)for v in raw['student_native_x'][-4:])
            assert not audit.paired_equal(raw['teacher_x'][-1,:,:256])
            assert all(torch.count_nonzero(getattr(env,k))==0 for k in ('vf','vs','vr'))
            assert checks['timeline']['common_suffix_actions']==[0]*8
            if kind=='probe_glimpse':
                cue=checks['cue_index'];assert len(raw['student_native_x'])-1-cue==8
                assert min(checks['probe_displacement_m'])>.04
                assert max(abs(v)for v in checks['probe_travel_minus_body_rad'])<2e-5
                assert min(abs(v)for v in checks['first_choice_travel_minus_body_rad'])>.5
                if condition=='pgc33':assert not torch.equal(raw['student_native_x'][cue],raw['student_ordinary_x'][cue])
            if condition=='gc100':assert torch.equal(raw['student_native_x'],raw['student_ordinary_x'])


def test_js_lower_bound_and_useful_turn_has_zero_immediate_progress():
    x=torch.zeros(4,1287);x[3,0]=1
    p=torch.zeros(4,6);p[0,0]=1;p[1,1]=1;p[2,0]=1;p[3,2]=1
    result=audit.identical_input_information(x,p,[{'case':i}for i in range(4)])
    assert sorted(v['count']for v in result['records'])==[1,3]
    group=[v for v in result['records']if v['count']==3][0]
    expected=-(2/3)*math.log(2/3)-(1/3)*math.log(1/3)
    assert abs(group['generalized_js_nats']-expected)<1e-12
    q=torch.tensor([[0.,0.,0.,0.,.005,-.005],[0.,0.,0.,0.,-.005,.005]])
    angle=torch.zeros(2,6);angle[0,0]=angle[1,1]=.03;angle[0,1]=angle[1,0]=-.03
    row=audit.pair_metrics(p[:2],q,angle)[0]
    assert row['teacher_mean_expected_progress_m']==0
    assert row['teacher_total_variation']==1 and abs(row['teacher_jensen_shannon']-math.log(2))<1e-12
    assert min(row['teacher_turn_weighted_alignment_rad'])>0


def test_continuation_uses_existing_first_input_once_and_stops_scoring_before_teleport():
    cfg,net,pop,ctx=fixture();raw,env,checks=audit.construct_history(net,pop,cfg,ctx,'probe_glimpse')
    direction=torch.stack((env.hd.cos(),env.hd.sin()),1)
    env.goal[:]=env.pos+.301*direction
    logits=torch.full((16,6),-80.);logits[:,2]=80.
    state=net.zero_state(16,'cpu');x=raw['teacher_x'][-1]
    trajectory,result=audit.continue_teacher(net,pop,cfg,ctx,env,state,x,logits)
    assert torch.equal(trajectory['x'][0],x)and torch.equal(trajectory['logits'][0],logits)
    assert torch.equal(trajectory['state_after_input'][0],torch.cat(state,1))
    assert torch.equal(trajectory['uniform'][:,::2],trajectory['uniform'][:,1::2])
    assert all(r['hit']and r['first_hit_step']==0 and r['active_updates']==1 for r in result['rows'])
    assert all(0<r['path_m']<.01 and 0<r['goal_distance_progress_m']<.01 for r in result['rows'])
    assert torch.equal(trajectory['scoring_position'],trajectory['scoring_position'][:1].expand_as(trajectory['scoring_position']))
    assert not torch.equal(trajectory['position_preteleport'][1],trajectory['scoring_position'][1])
    assert not trajectory['active_before'][1:].any()and not trajectory['hit'][1:].any()


def test_context_inventory_uses_distinct_episodes_geometry_and_private_tape_only():
    cfg,_,_,_=fixture();T,n=1350,32
    x=torch.zeros(1287);x[768:1280]=1
    pos=torch.zeros(T,n,2);goal=torch.zeros_like(pos);goal[:,:,0]=.8
    known=torch.zeros(T,n,dtype=torch.long);known[20:,1:]=1
    pos[20:30,1,0]=cfg['arena_half']
    state=torch.zeros(1,1,512).expand(T,n,-1)
    times=torch.arange(T)[:,None].expand(T,n).clone();times[:,2]=1340
    pre={k:torch.zeros(T,n)for k in audit.BOOKKEEPING}
    pre.update(n_goals=known,t=times,pos=pos,goal=goal,mdir=torch.tensor([1.,0.]).expand(T,n,-1),hd=torch.zeros(T,n))
    data=dict(pre=pre,
        x=x.expand(T,n,-1),state_before=state,done=torch.zeros(T,n,dtype=torch.bool),
        world=dict(land_pos=torch.zeros(n,4),stripe_phase=torch.zeros(n,4)))
    rng=torch.get_rng_state().clone();bank=audit.select_contexts(data,cfg)
    assert torch.equal(rng,torch.get_rng_state())and bank['complete']
    assert [c['slot']for c in bank['contexts']]==[1]+list(range(3,18))
    assert bank['contexts'][0]['step']==30 and bank['contexts'][1]['step']==20
    assert len(bank['inventory'])==32 and bank['inventory'][0]['first_eligible_step']is None
    assert sum(r['selected']for r in bank['inventory'])==16
    again=audit.select_contexts(data,cfg)
    assert all(torch.equal(a['continuation_uniform'],b['continuation_uniform'])for a,b in zip(bank['contexts'],again['contexts']))


def test_exact_native_prefix_replay_preserves_batch_and_terminal_resets():
    _,net,_,_=fixture();gen=torch.Generator().manual_seed(639901)
    x=torch.randn(10,32,1287,generator=gen)*.05;done=torch.zeros(10,32,dtype=torch.bool);done[3,2]=True
    state=net.zero_state(32,'cpu');before=[]
    with torch.no_grad():
        for t in range(10):
            before.append(torch.cat(state,1).clone());state=net.lstm(x[t],state)
            state=tuple(v*(~done[t])[:,None]for v in state)
    data=dict(x=x,done=done,state_before=torch.stack(before));contexts=[]
    for i,(slot,t)in enumerate(((1,3),(2,8))):
        contexts.append(dict(context_id=i,slot=slot,step=t,preceding_native_x=x[:t,slot].clone(),
            preceding_done=done[:t,slot].clone(),native_x=x[t,slot].clone(),initial_hc=data['state_before'][t,slot].clone()))
    bank=dict(contexts=contexts);assert audit.verify_native_prefixes(net,bank,data)['all_incoming_hc_exact']
    contexts[1]['initial_hc'][0]+=.001
    import pytest
    with pytest.raises(AssertionError):audit.verify_native_prefixes(net,bank,data)


def test_native_episode_time_is_preserved_through_controlled_history():
    cfg,net,pop,ctx=fixture();ctx['native_bookkeeping']['t']=torch.tensor(1280)
    raw,env,checks=audit.construct_history(net,pop,cfg,ctx,'probe_glimpse')
    assert env.episode_len==1350 and (env.t==1320).all()
    assert checks['source_episode_time']==1280 and min(checks['first_choice_episode_time'])==1320
