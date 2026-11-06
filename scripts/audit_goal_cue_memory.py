"""Independent saved-evidence audit for native goal-cue memory.

No new trial, environment step, action, optimization or model selection. Native
equations are evaluated at recorded observations. The entire stationary prefix
is independently reconstructed from zero using one saved clean RGB embedding,
then the first FREE transition is checked. FREE CNN/LSTM histories are not rerun.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
import torch.nn.functional as F
from code.env import _TARGETS
import goal_cue_memory as task
import goal_cue_memory_bank as banks
from audit_local_return_feedback import (checked_path, load_raw, tensor_schema,
    render_saved, verify_rgb, RGB_ATOL, CODE_ATOL, GEOMETRY_MARGIN)
from audit_local_goal_return_training import finite, audit_rng
from audit_local_goal_return_baseline import sha, read, same_tree as equal
from rl_goal_memory import atomic_json

NEURAL_ATOL, NEURAL_RTOL, PHYSICS_ATOL = 3e-4, 2e-6, 2e-5
VARIANTS = ('brief_native','continuous','brief_ordinary','no_cue','brief_reset',
    'donor_plus90','donor_minus90','continuous_reset','wrong_plus90','wrong_minus90','self_swap')
DONORS = ('donor_plus90','donor_minus90','self_swap')
WRONG = ('wrong_plus90','wrong_minus90')
SCOPE = dict(stationary_replay='Independent functional native CNN and LSTM from zero through every stationary input and the first FREE input; exact recorded cue/reset/override timing.',
    free_replay='Every saved physical transition, current code, rendered frame, sampled action CDF, and row statistic; full FREE CNN/LSTM histories are not independently rerun.',
    numeric_guard=dict(neural_atol=NEURAL_ATOL,neural_rtol=NEURAL_RTOL,physics_metres_radians_velocity_atol=PHYSICS_ATOL,rgb_atol=RGB_ATOL,code_atol=CODE_ATOL),
    rendering='Each frame has at least50% independently reconstructed pixels; explicit fixed renderer-boundary exclusions are reported separately. All pixels retain exact archived bytes and tensor hashes.',
    renderer_boundary_margin=GEOMETRY_MARGIN,
    full_history_replay_performed=False,no_environment_or_policy_rollout=True,
    qualification_is_distinct_from_integrity=True,common_information_50_percent_certificate=False)


def near(actual, expected, label, *, atol=NEURAL_ATOL, rtol=NEURAL_RTOL):
    assert actual.shape == expected.shape, label
    assert torch.allclose(actual.double(),expected.double(),atol=atol,rtol=rtol), label
    return float((actual.double()-expected.double()).abs().max()) if actual.numel() else 0.


def policy_digest(policy):
    digest=hashlib.sha256()
    for key,value in sorted(policy.items()):
        digest.update(key.encode());digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def embed_saved(rgb, policy, cfg):
    """Functional native encoder; no network initialization or RNG consumption."""
    x=rgb*2-1
    if cfg['conv_arch']=='si': settings=((0,2,2),(2,2,2),(4,2,2),(6,2,2))
    else:
        assert cfg['conv_arch']=='dqn';settings=((0,4,0),(2,2,0))
    for index,stride,padding in settings:
        key=f'encoder.conv.{index}'
        x=F.relu(F.conv2d(x,policy[key+'.weight'],policy[key+'.bias'],stride=stride,padding=padding))
    x=F.linear(x.flatten(1),policy['encoder.fc.weight'],policy['encoder.fc.bias'])
    if cfg.get('encoder_norm','none')=='layernorm':
        x=F.layer_norm(x,(256,),policy['encoder.norm.weight'],policy['encoder.norm.bias'],1e-5)
    else: assert cfg.get('encoder_norm','none')=='none'
    return F.relu(x)


def cell_saved(x,state,policy):
    h,c=state.chunk(2,dim=1)
    gates=F.linear(x,policy['lstm.weight_ih'],policy['lstm.bias_ih'])+F.linear(h,policy['lstm.weight_hh'],policy['lstm.bias_hh'])
    i,f,g,o=gates.chunk(4,dim=1)
    c=f.sigmoid()*c+i.sigmoid()*g.tanh()
    return torch.cat((o.sigmoid()*c.tanh(),c),1)


def variant_configuration(name,n):
    assert name in VARIANTS
    spec=dict(cue_mode='continuous' if name.startswith('continuous') else 'no_cue' if name=='no_cue' else 'brief',
        goal_kind='ordinary' if name=='brief_ordinary' else 'native',erase_after_cue=name in ('brief_reset','continuous_reset'))
    ids=None
    if name in DONORS+WRONG:
        assert n%4==0
        ids=(torch.arange(n)+(0 if name=='self_swap' else n//4*(1 if 'plus' in name else -1)))%n
    return spec,ids


def verify_masks(plan,raw,name):
    """Rebuild clock, alive, terminal and intervention masks independently."""
    n=len(plan['goal']);start=15+int(plan['delay_steps']);end=start+90
    spec,ids=variant_configuration(name,n)
    assert raw['active'].shape==(160,n) and raw['active'].dtype==torch.bool
    assert raw['action'].shape==(160,n) and raw['action'].dtype==torch.int64
    assert not raw['noise'].any() and not raw['reset_after'].any() and not raw['bootstrap'].any()
    assert not raw['pr'].any() and torch.equal(raw['physical_goal'],plan['goal'])
    alive=torch.ones(n,dtype=torch.bool)
    for t in range(160):
        active=alive & (t<end);free=active & (t>=start)
        assert torch.equal(raw['active'][t],active), 'Active/first-hit/timeout mask'
        assert torch.equal(raw['physics_active'][t],free) and torch.equal(raw['policy_mask'][t],free), 'Wait is not a physical/action update'
        phase=0 if t<15 else 1 if t<start else 2 if t<end else 3
        assert bool((raw['phase'][t]==phase).all()), 'Declared phase clock'
        assert bool((raw['action'][t,~free]==-1).all()) and bool(((raw['action'][t,free]>=0)&(raw['action'][t,free]<6)).all())
        reset=active & (spec['erase_after_cue'] and t==15)
        override=free & (name in DONORS and t==start)
        assert torch.equal(raw['reset_before'][t],reset), 'Erasure must occur before input15 only'
        assert torch.equal(raw['state_override_mask'][t],override), 'Donor substitution must occur before first FREE only'
        expected_pa=torch.full((n,),-1,dtype=torch.long)
        if t>start: expected_pa=torch.where(active,raw['action'][t-1],expected_pa)
        assert torch.equal(raw['pa'][t],expected_pa), 'Previous action leaks wait/terminal history'
        hit=free&((raw['position'][t]-plan['goal']).norm(dim=1)<.3)
        terminal=free&(hit|(t==end-1))
        assert torch.equal(raw['terminal'][t],terminal), 'First-hit/timeout termination'
        assert torch.equal(raw['raw_reward'][t],hit.float()*10), 'Only actual FREE pickup earns native reward'
        alive &= ~terminal
    assert not alive.any() and bool((raw['terminal'].sum(0)==1).all())
    assert int(raw['state_override_mask'].sum())==(n if name in DONORS else 0)
    assert torch.equal(raw['reward'],raw['raw_reward'].clamp(-1,1))
    return dict(start=start,wait_updates=start*n,free_updates=int(raw['physics_active'].sum()),trials=n)


def verify_physics(plan,cfg,raw,rows):
    """Native equations on saved pre-action states, never creating an arena."""
    n=len(plan['goal']);free=raw['physics_active'];active=raw['active'];start=15+int(plan['delay_steps'])
    dt_step=cfg['action_repeat']/60;k=1-math.exp(-dt_step/.25)
    target=torch.tensor(_TARGETS,dtype=torch.float64)*torch.tensor([cfg['v_max'],cfg['v_max'],cfg['rot_max']],dtype=torch.float64)
    p=plan['start'].double();hd=plan['heading'].double();motors=torch.zeros(n,3,dtype=torch.float64)
    direction=torch.stack((hd.cos(),hd.sin()),1);velocity=torch.zeros_like(p)
    maxima=dict(position=0.,heading=0.,motors=0.,velocity=0.,direction=0.)
    wall_count=0
    for t in range(160):
        # Compare to archived previous state, then advance using its float32
        # values to avoid treating 90-step cross-device accumulation as error.
        for key,expected,label in [('observation_position',p,'position'),('motors_before_action',motors,'motors'),
                ('observation_velocity',velocity,'velocity'),('observation_direction',direction,'direction')]:
            maxima[label]=max(maxima[label],near(raw[key][t],expected,label,atol=PHYSICS_ATOL,rtol=0))
        angle=((raw['observation_heading'][t].double()-hd+math.pi)%(2*math.pi)-math.pi).abs()
        maxima['heading']=max(maxima['heading'],float(angle.max()));assert maxima['heading']<=PHYSICS_ATOL
        p=raw['observation_position'][t].double();hd=raw['observation_heading'][t].double();motors=raw['motors_before_action'][t].double()
        direction=raw['observation_direction'][t].double();use=free[t]
        next_p=p.clone();next_hd=hd.clone();next_m=motors.clone();velocity=torch.zeros_like(p)
        if use.any():
            actions=raw['action'][t,use]
            targets=target[actions]+plan['motor'][t,:,use].T.double()*torch.tensor([.03,.03,.15],dtype=torch.float64)
            next_m[use]=motors[use]+k*(targets-motors[use])
            next_hd[use]=(hd[use]+next_m[use,2]*dt_step+math.pi)%(2*math.pi)-math.pi
            cs,sn=next_hd[use].cos(),next_hd[use].sin()
            proposed=p[use]+torch.stack((next_m[use,0]*cs-next_m[use,1]*sn,next_m[use,0]*sn+next_m[use,1]*cs),1)*dt_step
            actual=proposed.clamp(-cfg['arena_half']+.08,cfg['arena_half']-.08)
            walls=(actual!=proposed).any(1);wall_count+=int(walls.sum())
            next_m[use,:2]*=torch.where(walls,.2,1.).double()[:,None];next_p[use]=actual
        maxima['position']=max(maxima['position'],near(raw['position'][t],next_p,'Post-action physical endpoint',atol=PHYSICS_ATOL,rtol=0))
        displacement=raw['position'][t].double()-p;distance=displacement.norm(dim=1)
        moving=use&(distance>.05*dt_step)
        direction=torch.where(moving[:,None],displacement/distance.clamp_min(1e-9)[:,None],direction)
        velocity[use]=displacement[use]/dt_step
        hit=raw['raw_reward'][t]>0;velocity[hit]=0
        p=raw['position'][t].double();hd=next_hd;motors=next_m
    for i,row in enumerate(rows):
        assert row['context']==int(plan['context'][i]) and row['angle_index']==int(plan['angle_index'][i])
        use=free[:,i];count=int(use.sum());hits=raw['raw_reward'][:,i]>0
        path=(raw['position'][:,i].double()-raw['observation_position'][:,i].double()).norm(dim=1)[use].sum()
        initial=(plan['start'][i].double()-plan['goal'][i].double()).norm()
        final=(raw['position'][use,i][-1].double()-plan['goal'][i].double()).norm()
        assert row['hit']==bool(hits.any()) and row['free_steps']==count
        assert row['hit_step']==(int(hits.nonzero()[0])+1-start if hits.any() else -1)
        for key,value in dict(free_path_m=path,initial_distance_m=initial,final_distance_m=final,progress_m=initial-final).items():
            assert math.isclose(row[key],float(value),rel_tol=3e-6,abs_tol=PHYSICS_ATOL),key
        assert row['action_counts']==torch.bincount(raw['action'][use,i],minlength=6).tolist()
    final=raw['evidence']['final_physical_state'];initial=raw['evidence']['initial_physical_state']
    assert torch.equal(final['t'],free.sum(0)) and not initial['t'].any()
    assert torch.equal(final['n_goals'],(raw['raw_reward']>0).sum(0))
    assert torch.equal(final['ep_return'],raw['raw_reward'].sum(0))
    near(final['pos'],p,'Final physical position',atol=PHYSICS_ATOL,rtol=0)
    assert equal(initial['environment_rng'],final['environment_rng']) and initial['episodes_done']==final['episodes_done']==0
    return dict(maximum_errors=maxima,free_wall_hits=wall_count)


def verify_inputs(plan,cfg,raw,name,native=None):
    n=len(plan['goal']);start=15+int(plan['delay_steps']);spec,ids=variant_configuration(name,n)
    assert cfg['vision']=='raycast' and cfg['reward_clip']==1
    for key,shape in dict(rgb=(160,n,3,84,84),code=(160,n,512),goal_input=(160,n,512),
            logits=(160,n,6),values=(160,n),noise=(160,2,n,cfg['nh'])).items():
        assert raw[key].shape==shape and raw[key].dtype==torch.float32,key
    initial=raw['evidence']['initial_physical_state'];setup=raw['evidence']['setup']
    for key in ('pos','spawn','hd','vf','vs','vr','t','partial','ep_idx','draw_idx','travelled','ep_return','n_goals','path_eff_sum','path_eff_n'):
        assert not initial[key].any(),f'Uncleared setup history: {key}'
    assert bool((initial['first_goal_t']==-1).all())
    assert torch.equal(initial['mdir'],torch.tensor([1.,0.]).expand(n,2))
    assert torch.equal(initial['land_pos'],plan['land_pos']) and torch.equal(initial['stripe_phase'],plan['stripe_phase'])
    angle=plan['pickup_angle'].double();direction=torch.stack((angle.cos(),angle.sin()),1)
    dt_step=cfg['action_repeat']/60;speed=cfg['v_max']*(1-math.exp(-dt_step/.25))
    before=plan['goal'].double()-.302*direction;after=before+direction*speed*dt_step
    near(setup['position_before'],before,'Actual outside pickup start',atol=PHYSICS_ATOL,rtol=0)
    near(setup['hit_position'],after,'Actual pickup crossing',atol=PHYSICS_ATOL,rtol=0)
    assert bool((setup['distance_before']>.3).all()) and bool(((setup['hit_position']-plan['goal']).norm(dim=1)<.3).all())
    near(setup['hit_velocity'],direction*speed,'Pickup velocity',atol=PHYSICS_ATOL,rtol=0)
    near(setup['hit_direction'],direction,'Pickup direction',atol=PHYSICS_ATOL,rtol=0)
    assert bool((setup['native_reward']==10).all() and setup['native_hit'].all() and not setup['native_wall'].any())
    assert setup['learning_reward']==0 and setup['private_rng_unchanged'] and setup['no_respawn'] and not setup['transferred_reward_history']
    source=task.make_source(cfg,'cpu')
    native_goal=source.goal(setup['hit_position'],setup['hit_direction'],setup['hit_velocity'])
    ordinary_goal=source.pop.activity(setup['hit_position'],setup['hit_direction'],setup['hit_velocity'],shift_scale=0.)
    near(raw['native_saved_goal'],native_goal,'Native boundary goal code',atol=CODE_ATOL,rtol=0)
    near(raw['ordinary_saved_goal'],ordinary_goal,'Ordinary boundary goal code',atol=CODE_ATOL,rtol=0)
    pgc=source.pop.is_pgc.cpu()
    assert torch.equal(raw['native_saved_goal'][:,~pgc],raw['ordinary_saved_goal'][:,~pgc]), 'Ordinary goal replacement modified ordinary-cell slots'
    goal=raw['ordinary_saved_goal'] if spec['goal_kind']=='ordinary' else raw['native_saved_goal']
    if name in WRONG:
        assert native is not None;goal=native['saved_goal'][ids]
    assert torch.equal(raw['saved_goal'],goal),'Wrong saved goal or unrecorded cue mutation'
    if name in DONORS:
        assert native is not None
        expected=torch.stack([native['evidence']['first_free'][int(i)]['state_before_input'] for i in ids])
        assert torch.equal(raw['first_free_state_override'],expected),'Donor is not genuine same-context incoming memory'
    else: assert not raw['first_free_state_override'].any()
    if cfg['cond']=='gc100': assert torch.equal(raw['native_saved_goal'],raw['ordinary_saved_goal'])
    for t in range(160):
        active=raw['active'][t]
        supplied=goal*active[:,None] if spec['cue_mode']=='continuous' or (spec['cue_mode']=='brief' and t<15) else torch.zeros_like(goal)
        assert torch.equal(raw['goal_input'][t],supplied),'External goal cache leaked past cue/blank boundary'
        assert not raw['rgb'][t,~active].any() and not raw['code'][t,~active].any()
    # Every goal direction has byte-identical non-goal stationary inputs.
    for key in ('rgb','code','observation_position','observation_heading','observation_direction','observation_velocity','motors_before_action'):
        assert torch.equal(raw[key][:start+1],raw[key][0:1].expand_as(raw[key][:start+1])),f'Wait input changes: {key}'
        assert torch.equal(raw[key][0],raw[key][0:1,0].expand_as(raw[key][0])),f'Goal leaks into stationary input: {key}'
    mask=raw['active'];times,indices=mask.nonzero(as_tuple=True)
    totals=dict(pixels=0,excluded_boundary_pixels=0,boundary_mismatches_above_atol=0,nonboundary_max_abs_error=0.,boundary_max_abs_error=0.)
    max_code=0.
    for lo in range(0,len(times),128):
        t,i=times[lo:lo+128],indices[lo:lo+128]
        expected=source(raw['observation_position'][t,i],raw['observation_direction'][t,i],raw['observation_velocity'][t,i])
        max_code=max(max_code,near(raw['code'][t,i],expected,'Fresh current code',atol=CODE_ATOL,rtol=0))
        image,boundary=render_saved(raw['observation_position'][t,i],raw['observation_heading'][t,i],plan['land_pos'][i],plan['stripe_phase'][i],cfg['arena_half'])
        checks=verify_rgb(raw['rgb'][t,i],image,boundary)
        for key,value in checks.items(): totals[key]=max(totals[key],value) if key.endswith('error') else totals[key]+value
    return dict(**totals,current_code_max_abs_error=max_code,exact_full_goal_withdrawal=True)


@torch.inference_mode()
def verify_stationary_prefix(plan,cfg,raw,policy,*,embedding_function=None):
    """Reconstruct actual recorded prefix, including erasure and donor timing.

    Optional embedding_function is for synthetic-policy tests only. Production
    calls always use embed_saved and the exact archived native CNN parameters.
    """
    n=len(plan['goal']);start=15+int(plan['delay_steps']);nh=cfg['nh'];entries=raw['evidence']['first_free']
    assert set(entries)==set(range(n)) and start<=60
    state=torch.zeros(n,2*nh,dtype=raw['rgb'].dtype)
    embedding=(embedding_function or (lambda rgb:embed_saved(rgb,policy,cfg)))(raw['rgb'][0])
    maximum=dict(state=0.,logits=0.,value=0.,embedding=0.)
    for t in range(start+1):
        reset=raw['reset_before'][t];state=torch.where(reset[:,None],torch.zeros_like(state),state)
        if t==start:
            before=torch.stack([entries[i]['state_before_intervention'] for i in range(n)])
            maximum['state']=max(maximum['state'],near(state,before,'Stationary memory before donor substitution'))
        state=torch.where(raw['state_override_mask'][t,:,None],raw['first_free_state_override'],state)
        action=raw['pa'][t]
        onehot=F.one_hot(action.clamp_min(0),6).to(embedding.dtype)*(action>=0)[:,None]
        x=torch.cat((embedding,raw['code'][t],raw['goal_input'][t],onehot,raw['pr'][t,:,None]),1)
        if t==start:
            before=torch.stack([entries[i]['state_before_input'] for i in range(n)])
            maximum['state']=max(maximum['state'],near(state,before,'Incoming first FREE memory'))
            observed_x=torch.stack([entries[i]['input'] for i in range(n)])
            maximum['embedding']=near(embedding,observed_x[:,:256],'Native CNN embedding')
            assert torch.equal(x[:,256:],observed_x[:,256:]),'Recorded first FREE input fields'
        state=cell_saved(x,state,policy)
        logits=F.linear(state[:,:nh],policy['pi.weight'],policy['pi.bias'])
        value=F.linear(state[:,:nh],policy['v.weight'],policy['v.bias']).squeeze(1)
        maximum['logits']=max(maximum['logits'],near(logits,raw['logits'][t],'Stationary/first FREE actor'))
        maximum['value']=max(maximum['value'],near(value,raw['values'][t],'Stationary/first FREE critic'))
        if t==14:maximum['state']=max(maximum['state'],near(state,raw['evidence']['cue_end_state'],'Cue-end memory from zero'))
        if t==start:
            maximum['state']=max(maximum['state'],near(state,torch.stack([entries[i]['state'] for i in range(n)]),'First FREE state processed exactly once'))
    for i,entry in entries.items():
        assert torch.equal(entry['logits'],raw['logits'][start,i]) and torch.equal(entry['value'],raw['values'][start,i])
        assert torch.equal(entry['goal_input'],raw['goal_input'][start,i])
        for key,other in [('position','observation_position'),('heading','observation_heading'),('motors','motors_before_action')]:
            assert torch.equal(entry[key],raw[other][start,i]),key
        assert entry['state_was_replaced']==bool(raw['state_override_mask'][start,i])
        near(entry['probabilities'],raw['logits'][start,i].double().softmax(0),'First FREE probabilities',atol=2e-6,rtol=2e-6)
    return dict(stationary_updates=start,first_free_updates=1,maximum_errors=maximum,
        zero_initial_state_reconstructed=True,full_free_history_replayed=False,
        tolerance=dict(atol=NEURAL_ATOL,rtol=NEURAL_RTOL))


def verify_action_statistics(plan,raw,rows):
    probabilities=raw['logits'].double().softmax(-1)
    entropy=-(probabilities*probabilities.clamp_min(1e-300).log()).sum(-1)
    for i,row in enumerate(rows):
        mask=raw['physics_active'][:,i];actions=raw['action'][mask,i];p=probabilities[mask,i]
        u=plan['action'][mask,i].double();cdf=p.cumsum(1);index=torch.arange(len(actions))
        lower=torch.where(actions>0,cdf[index,(actions-1).clamp_min(0)],torch.zeros_like(u))
        upper=cdf[index,actions]
        assert bool(((u>=lower-2e-6)&(u<=upper+2e-6)).all()),'Saved action differs from native policy CDF/tape'
        assert math.isclose(row['mean_free_entropy'],float(entropy[mask,i].mean()),rel_tol=3e-6,abs_tol=3e-6)
    return dict(all_free_actions_cdf_checked=True,all_free_entropies_checked=True)


def summarize_rows(rows,contexts,angles):
    expected={(c,a)for c in range(contexts)for a in range(angles)}
    assert len(rows)==contexts*angles and {(r['context'],r['angle_index'])for r in rows}==expected
    assert all(type(r['hit'])is bool for r in rows)
    invalid=sum(r.get('invalid',False)for r in rows)
    assert all(not r['hit']for r in rows if r.get('invalid',False))
    total=len(rows);hits=sum(r['hit']for r in rows)
    out=dict(trials=total,valid_trials=total-invalid,invalid_trials=invalid,successes=hits,success_fraction=hits/total,
        mean_path_m=sum(r.get('free_path_m',0.)for r in rows)/total,
        mean_progress_m=sum(r.get('progress_m',0.)for r in rows)/total,
        mean_free_entropy=sum(r.get('mean_free_entropy',0.)for r in rows)/total,
        free_updates=sum(r.get('free_steps',0)for r in rows),
        action_counts=[sum(r.get('action_counts',[0]*6)[a]for r in rows)for a in range(6)])
    directional=[r['donor_behavior']for r in rows if not r.get('invalid',False)and 'donor_behavior'in r]
    if directional:
        out['donor_behavior']=dict(valid_directional_trials=len(directional),prescribed_trials=total,
            mean_progress_to_donor_m=sum(r['progress_to_donor_m']for r in directional)/total,
            mean_progress_to_recipient_m=sum(r['progress_to_recipient_m']for r in directional)/total,
            mean_early_15_free_progress_to_donor_m=sum(r['early_15_free_progress_to_donor_m']for r in directional)/total,
            mean_early_15_free_progress_to_recipient_m=sum(r['early_15_free_progress_to_recipient_m']for r in directional)/total,
            early_terminal_position_held_trials=sum(r['early_terminal_position_held']for r in directional),
            full_path_recipient_hit_censored_trials=sum(r['full_path_recipient_hit_censored']for r in directional),
            early_horizon_convention='Position after 15 FREE updates; earlier terminal positions are held, not excluded.',
            full_path_censoring='Actual recipient hits end paths; donor-entry counts describe observation before that stop.',
            first_target_counts={key:sum(r['first_target_disk']==key for r in directional)for key in ('donor','recipient','same_step','neither')})
    return out


def expected_gates(variants):
    eligible=lambda name:name in variants and variants[name]['totals']['trials']==256 and variants[name]['totals']['valid_trials']==256 and variants[name]['all_physical_checks']
    control=eligible('continuous');memory=eligible('brief_native')and eligible('brief_reset')
    native=variants['brief_native']['totals']['successes']if 'brief_native'in variants else None
    cost=native-variants['brief_reset']['totals']['successes']if memory else None
    return dict(qualification=dict(eligible=control,success_threshold=192,passed=control and variants['continuous']['totals']['successes']>=192),
        useful_memory=dict(eligible=memory,native_success_threshold=167,reset_cost_threshold=39,native_successes=native,reset_cost=cost,passed=memory and native>=167 and cost>=39),
        common_information_certificate_applies=False)


def verify_donor_behavior(raw,plan,rows,indices):
    for i,row in enumerate(rows):
        measured=row['donor_behavior'];path=raw['position'][raw['physics_active'][:,i],i].double()
        donor=plan['goal'][indices[i]].double();recipient=plan['goal'][i].double()
        dd=(path-donor).norm(dim=1);rd=(path-recipient).norm(dim=1)
        # Disk membership is the declared strict float32 saved-data predicate;
        # distances/progress additionally receive an independent float64 check.
        dh=((path.float()-donor.float()).norm(dim=1)<.3).nonzero().flatten()
        rh=((path.float()-recipient.float()).norm(dim=1)<.3).nonzero().flatten()
        d=int(dh[0])+1 if len(dh)else None;r=int(rh[0])+1 if len(rh)else None
        first='neither'if d is None and r is None else 'donor'if r is None or(d is not None and d<r)else 'recipient'if d is None or r<d else 'same_step'
        assert measured['donor_index']==int(indices[i]) and measured['donor_goal_world']==plan['goal'][indices[i]].tolist()
        assert measured['recipient_goal_world']==plan['goal'][i].tolist()
        assert measured['first_donor_disk_entry_step']==d and measured['first_recipient_disk_entry_step']==r and measured['first_target_disk']==first
        for key,value in [('progress_to_donor_m',(plan['start'][i].double()-donor).norm()-dd[-1]),('progress_to_recipient_m',(plan['start'][i].double()-recipient).norm()-rd[-1])]:
            assert math.isclose(measured[key],float(value),rel_tol=3e-6,abs_tol=PHYSICS_ATOL),key
        assert measured['early_horizon_free_updates']==15 and measured['early_observed_free_updates']==min(15,len(path))
        assert measured['early_terminal_position_held']==(len(path)<15) and measured['full_path_recipient_hit_censored']==(r is not None)
        early=path[min(14,len(path)-1)]
        for key,target in [('donor',donor),('recipient',recipient)]:
            value=(plan['start'][i].double()-target).norm()-(early-target).norm()
            assert math.isclose(measured[f'early_15_free_progress_to_{key}_m'],float(value),rel_tol=3e-6,abs_tol=PHYSICS_ATOL)


def verify_branch(plan,cfg,raw,rows,name,policy,native=None,*,embedding_function=None):
    finite(raw);mask=verify_masks(plan,raw,name)
    physics=verify_physics(plan,cfg,raw,rows)
    inputs=verify_inputs(plan,cfg,raw,name,native)
    recurrent=verify_stationary_prefix(plan,cfg,raw,policy,embedding_function=embedding_function)
    actions=verify_action_statistics(plan,raw,rows)
    _,ids=variant_configuration(name,len(rows))
    if ids is not None:verify_donor_behavior(raw,plan,rows,ids)
    return dict(masks=mask,physics=physics,inputs=inputs,recurrent=recurrent,actions=actions)


def audit_evaluation(folder,checkpoint,plan,reference,variants,*,embedding_function=None):
    """All prescribed contexts/branches, including failures; no selection."""
    folder=Path(folder);report=read(folder/'summary.json');cfg=checkpoint['config'];policy=checkpoint['policy']
    assert report['version']=='native_goal_cue_memory_evaluation_v1' and report['complete']
    assert report['condition']==cfg['cond'] and report['checkpoint_policy_hash']==policy_digest(policy)
    assert report['collector_sha256']==sha(task.__file__) and report['bank_helper_sha256']==sha(banks.__file__)
    assert report['source_sha256']==sha(ROOT/'scripts/goal_cue_memory_evaluate.py')
    assert report['reference']==reference and report['no_optimization'] and not report['full_history_replay_performed']
    names=tuple(name for name in VARIANTS if name in variants)
    assert report['variant_order']==list(names) and set(report['variants'])==set(names)
    count,angles=plan['n_contexts'],plan['angles_per_context']
    assert (report['n_contexts'],report['angles_per_context'])==(count,angles) and len(report['contexts'])==count
    all_rows={name:[]for name in names};details=[];invalid=0
    for c,context in enumerate(report['contexts']):
        assert context['context']==c and set(context['variants'])==set(names)
        root=folder/f'context{c}';chunk=load_raw(root,context['input_plan'])
        assert equal(chunk,banks.context_plan(plan,c)),'Saved context is not exact sealed-bank slice'
        native=None;native_record=None
        for name in names:
            branch=root/name;record=read(branch/'result.json');assert record==context['variants'][name]
            input_ref=read(branch/'input.json');spec,ids=variant_configuration(name,angles)
            provenance=record['configuration'];assert input_ref['variant']==provenance
            for key,value in spec.items():assert provenance[key]==value
            assert provenance['variant']==name
            expected_ref=dict(reference,context_index=c,context_ids=chunk['context'].tolist(),input_plan_sha256=context['input_plan']['sha256'],variant=provenance)
            assert input_ref==expected_ref
            if ids is not None:
                assert provenance['donor_indices']==ids.tolist() and provenance['source_variant']=='brief_native' and provenance['recipient_physical_goal_unchanged']
                assert provenance['donor_angle_degrees']==(0 if name=='self_swap'else 90 if 'plus'in name else -90)
                if native is not None:
                    assert provenance['source_native_result_sha256']==sha(root/'brief_native/result.json')
                    assert provenance['source_native_raw_sha256']==native_record['raw']['sha256']
            rows=record['rows'];assert len(rows)==angles;all_rows[name].extend(rows)
            if not record['valid']:
                assert equal(load_raw(branch,record['failed_plan']),chunk)
                assert all(r.get('invalid')and not r['hit']for r in rows)
                invalid+=angles;details.append(dict(context=c,variant=name,valid=False,error=record['error']));continue
            assert not (branch/'failed_plan.pt.gz').exists(),'Unresolved failure evidence beside a valid result'
            raw=load_raw(branch,record['raw']);assert raw['plan_reference']==input_ref and raw['variant_provenance']==provenance
            checked=verify_branch(chunk,cfg,raw,rows,name,policy,native,embedding_function=embedding_function)
            actual=record['audit'];masks=checked['masks']
            for key,value in dict(controller='policy',cue_mode=spec['cue_mode'],goal_kind=spec['goal_kind'],
                    terminal_bootstrap_zero=True,task_owned_stationary_wait=True,img_noise=0.,state_noise_sigma=0.,
                    free_motor_noise_v=.03,free_motor_noise_r=.15,native_environment_rng_unchanged=True,
                    saved_goal_used_only_by_explicit_input=True,full_goal_vector_withdrawn=spec['cue_mode']!='continuous').items():
                assert actual[key]==value,key
            for key,value in dict(completed_trials=angles,setup_pickups=angles,setup_learning_reward=0.,forced_actions=0,forced_rewards=0.,
                    active_recurrent_updates=int(raw['active'].sum()),active_physics_updates=masks['free_updates'],free_updates=masks['free_updates'],wait_updates=masks['wait_updates'],
                    allocated_slots=160*angles,padded_slots=160*angles-int(raw['active'].sum()),free_hits=sum(r['hit']for r in rows),free_wall_hits=checked['physics']['free_wall_hits'],
                    erased_states=angles if spec['erase_after_cue']else 0,goal_override_used=name in WRONG,first_free_state_override_used=name in DONORS).items():
                assert actual[key]==value,key
            assert all(record['physical_checks'].values())
            if name=='self_swap' or(cfg['cond']=='gc100'and name=='brief_ordinary'):
                assert native is not None
                for key in ('position','action','reward','logits','values','goal_input'):
                    assert torch.equal(raw[key],native[key]),f'Grid/self sham {key}'
            if name=='brief_native':native=raw;native_record=record
            details.append(dict(context=c,variant=name,valid=True,checks=checked))
    for name,rows in all_rows.items():
        item=report['variants'][name];assert item['rows']==rows
        for key,value in summarize_rows(rows,count,angles).items():assert item['totals'][key]==value,key
        assert item['all_physical_checks']==all(ctx['variants'][name]['valid']for ctx in report['contexts'])
    gates=expected_gates(report['variants'])
    for name,value in gates.items():
        if isinstance(value,dict):
            for key,v in value.items():assert report['gates'][name][key]==v
        else:assert report['gates'][name]==value
    assert report['all_variants_valid']==(invalid==0)
    return dict(complete=True,valid=invalid==0,invalid_trials=invalid,contexts=details,gates=gates,
        variants={name:report['variants'][name]['totals']for name in names},scope=SCOPE)


COUNTERS=('active_recurrent_updates','active_physics_updates','free_updates','wait_updates',
    'padded_slots','setup_pickups','free_hits')
ENDPOINTS={'baseline':0,'block1':1024,'block2':2048,'block3':3072}


def audit_metric(row):
    finite(row);u=row['update'];block=(u-1)//1024+1;cap=(0,15,45)[block-1]
    assert 1<=u<=3072 and row['block']==block
    assert row['completed_trials']==row['setup_pickups']==32
    assert row['setup_learning_reward']==row['forced_rewards']==row['forced_actions']==0
    assert row['terminal_bootstrap_zero'] and row['controller']=='policy' and row['cue_mode']=='brief' and row['goal_kind']=='native'
    assert row['img_noise']==row['state_noise_sigma']==row['erased_states']==0
    assert not row['goal_override_used'] and not row['first_free_state_override_used']
    assert row['native_environment_rng_unchanged'] and row['full_goal_vector_withdrawn'] and row['task_owned_stationary_wait']
    assert row['free_motor_noise_v']==.03 and row['free_motor_noise_r']==.15
    assert row['allocated_slots']==5120 and 0<=row['delay_steps']<=cap
    assert row['wait_updates']==32*(15+row['delay_steps'])
    assert row['active_recurrent_updates']==row['wait_updates']+row['free_updates']
    assert row['active_physics_updates']==row['free_updates']
    assert row['active_recurrent_updates']+row['padded_slots']==5120
    assert 32<=row['free_updates']<=2880 and 0<=row['free_hits']<=32
    assert 0<=row['replay_error']<3e-4 and 0<=row['value_replay_error']<3e-4 and row['grad_norm']>=0
    assert 0<=row['entropy_per_free_decision']<=math.log(6)+2e-6
    assert math.isclose(row['entropy_sum_per_trial'],row['entropy_per_free_decision']*row['free_updates']/32,rel_tol=3e-6,abs_tol=3e-6)
    expected=(100/160)*(row['policy_loss']+.5*row['value_loss']-8e-5*row['entropy_sum_per_trial'])
    assert math.isclose(row['loss'],expected,rel_tol=2e-5,abs_tol=2e-5) and row['value_loss']>=0


def accepted_metrics(folder,cutoff=None,ancestor='resume_parent_evidence'):
    folder=Path(folder);prior=[];start=0;discarded=0
    if(folder/'resume_lineage.json').exists():
        lineage=read(folder/'resume_lineage.json');start=lineage['restored_updates']
        prior,meta=accepted_metrics(folder/ancestor,start,'ancestor_evidence');discarded+=meta['rolled_back_rows']
    rows=[];partial=False
    lines=(folder/'metrics.jsonl').read_bytes().splitlines()
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            assert index==len(lines)-1;partial=True;break
        audit_metric(row);assert row['update']==start+len(rows)+1;rows.append(row)
    if cutoff is not None:
        assert start<=cutoff<=start+len(rows);discarded+=sum(r['update']>cutoff for r in rows)
        rows=[r for r in rows if r['update']<=cutoff]
    rows=prior+rows;assert [r['update']for r in rows]==list(range(1,len(rows)+1))
    frames=0;delay_generator=torch.Generator().manual_seed(694806)
    for row in rows:
        frames+=row['active_physics_updates']*4
        if 'active_physics_frames'in row:assert row['active_physics_frames']==frames
        cap=(0,15,45)[(row['update']-1)//1024]
        assert row['delay_cap']==cap and row['delay_steps']==int(torch.randint(0,cap+1,(1,),generator=delay_generator)), 'Private delay draw differs'
    return rows,dict(accepted_updates=len(rows),rolled_back_rows=discarded,partial_last_row=partial,active_physics_frames=frames)


def verify_checkpoint_delays(rows,data):
    """Authenticate every logged delay, then its histogram and saved RNG state."""
    assert len(rows)==data['counters']['updates']
    counts={f'block{block}':[0]*46 for block in (1,2,3)}
    generator=torch.Generator().manual_seed(694806)
    for index,row in enumerate(rows):
        block=index//1024+1;cap=(0,15,45)[block-1]
        assert row['update']==index+1 and row['block']==block and row['delay_cap']==cap
        sampled=int(torch.randint(0,cap+1,(1,),generator=generator))
        assert row['delay_steps']==sampled, 'Checkpoint private delay draw differs from logged delay'
        counts[f'block{block}'][sampled]+=1
    assert data['counters']['delay_histograms']==counts
    assert torch.equal(data['streams']['generators']['delay'],generator.get_state())


def audit_checkpoint(data,initial,provenance):
    finite(data);cfg=initial['config'];c=data['counters'];u=c['updates'];records=c['evaluation_records']
    assert data['version']=='goal_cue_memory_exact_resume_v1' and data['no_inflight_trial']
    assert data['boundary_state']=='All trials terminal; next trial starts center,h/c0,motors0,pa=-1,pr0'
    assert data['config']==cfg and data['initialization']==initial['initialization'] and data['provenance']==provenance
    assert 0<=u<=3072 and c['trials']==u*32 and set(records)<=set(ENDPOINTS)
    for name,boundary in ENDPOINTS.items():
        if name in records:assert u>=boundary
        else:assert u<=boundary,'Required evaluation missing before subsequent learning'
    assert c['setup_pickups']==c['trials'] and c['active_physics_updates']==c['free_updates']
    assert c['active_recurrent_updates']==c['wait_updates']+c['free_updates']
    assert c['active_recurrent_updates']+c['padded_slots']==u*5120
    assert data['active_physics_frames']==c['active_physics_updates']*4
    assert data['completed']==(u==3072 and 'block3'in records)
    assert data['policy'].keys()==initial['policy'].keys()
    for key,value in data['policy'].items():assert value.shape==initial['policy'][key].shape and value.dtype==initial['policy'][key].dtype
    assert set(c['delay_histograms'])=={'block1','block2','block3'}
    for block,cap in enumerate((0,15,45),1):
        counts=c['delay_histograms'][f'block{block}'];assert len(counts)==46
        assert all(isinstance(v,int)and v>=0 for v in counts)and not any(counts[cap+1:])
        assert sum(counts)==min(1024,max(0,u-(block-1)*1024))
    expected_wait=sum(count*(15+delay)*32 for counts in c['delay_histograms'].values()for delay,count in enumerate(counts))
    assert c['wait_updates']==expected_wait and c['trials']<=c['free_updates']<=c['trials']*90 and 0<=c['free_hits']<=c['trials']
    optimizer=data['optimizer'];assert len(optimizer['param_groups'])==1;group=optimizer['param_groups'][0]
    for key,value in dict(lr=3e-4,alpha=.99,eps=1e-5,weight_decay=0,momentum=0,centered=False,
            capturable=False,maximize=False,differentiable=False,foreach=None).items():assert group[key]==value,key
    assert group['params']==list(range(len(data['policy'])))
    assert set(optimizer['state'])==(set(group['params'])if u else set())
    for index,record in optimizer['state'].items():
        value=list(data['policy'].values())[index]
        assert set(record)=={'step','square_avg'} and float(record['step'])==u
        assert record['square_avg'].shape==value.shape and record['square_avg'].dtype==value.dtype and bool((record['square_avg']>=0).all())
    stream=data['streams'];assert stream['seed']==694801 and stream['batches']==stream['delay_draws']==u and stream['contexts']==u*32
    assert set(stream['generators'])=={'goal','pickup','layout','motor','action','delay'}
    for value in stream['generators'].values():audit_rng(value)
    rng=data['global_rng'];assert set(rng)=={'cpu','cuda','mps'};audit_rng(rng['cpu'])
    assert len(rng['cuda'])==(1 if provenance['device'].startswith('cuda')else 0)
    for value in rng['cuda']:audit_rng(value,cuda=True)
    if rng['mps']is not None:audit_rng(rng['mps'],cuda=True)
    if not u:
        assert equal(data['policy'],initial['policy']) and equal(stream,task.TrialStreams(694801).snapshot())
        assert all(c[key]==0 for key in COUNTERS)
    return dict(updates=u,trials=c['trials'],completed=data['completed'],evaluation_records=list(records),
        active_physics_frames=data['active_physics_frames'],policy_hash=policy_digest(data['policy']))


def compare_boundary(before,after,endpoint):
    assert set(before)==set(after)
    for key in before:
        if key not in ('counters','completed'):assert equal(before[key],after[key]),f'Evaluation changed {key}'
    old,new=before['counters'],after['counters']
    assert endpoint not in old['evaluation_records'] and set(new['evaluation_records'])==set(old['evaluation_records'])|{endpoint}
    for key in old:
        if key!='evaluation_records':assert equal(old[key],new[key])
    for key in old['evaluation_records']:assert old['evaluation_records'][key]==new['evaluation_records'][key]


def audit_initial(initial,condition,root=ROOT):
    """Validate archived composition without creating a policy or optimizer."""
    import goal_cue_memory_run as runner
    root=Path(root);cfg=initial['config'];init=initial['initialization'];runner.assert_config(cfg)
    assert cfg['cond']==condition and init['condition']==condition and init['original_seed']==1
    assert init['actor']=='retained_original_1B' and init['shared_fresh_critic'] and init['all_parameters_trainable'] and init['no_optimization']
    manifest_path=root/'docs/research/local_return_feedback_initialization.json'
    assert sha(manifest_path)==runner.INITIALIZATION_SHA==init['original_initialization_manifest_sha256']
    manifest=read(manifest_path);entry=next(r for r in manifest['policies']if r['condition']==condition and r['actor_mode']=='retained_actor')
    asset=checked_path(root,entry['path']);assert sha(asset)==entry['sha256']==init['incoming_asset_sha256'] and asset.stat().st_size==entry['bytes']
    assert init['incoming_asset']==entry['path']
    saved=torch.load(asset,map_location='cpu',weights_only=True)
    assert saved['no_forward_pass'] and saved['no_optimization'] and init['original']==saved['initialization']
    original_meta=init['original'];original_path=checked_path(root,original_meta['parent_checkpoint_path']);critic_path=checked_path(root,original_meta['archived_heads_path'])
    assert sha(original_path)==original_meta['parent_checkpoint_sha256'] and sha(original_path.with_suffix('.json'))==original_meta['parent_config_sha256']
    assert sha(critic_path)==original_meta['archived_heads_sha256']
    original=torch.load(original_path,map_location='cpu',weights_only=True);critic=torch.load(critic_path,map_location='cpu',weights_only=True)
    assert original['frames']==1_000_000_000 and critic['no_optimization']
    assert saved['policy'].keys()==original['policy'].keys()==critic['policy'].keys()
    for key,value in saved['policy'].items():assert torch.equal(value,critic['policy'][key]if key.startswith('v.')else original['policy'][key]),key
    peer=next(r for r in manifest['policies']if r['condition']!=condition and r['actor_mode']=='retained_actor')
    peer_path=checked_path(root,peer['path']);assert sha(peer_path)==peer['sha256'] and peer_path.stat().st_size==peer['bytes']
    peer_policy=torch.load(peer_path,map_location='cpu',weights_only=True)['policy']
    for key in ('v.weight','v.bias'):assert torch.equal(saved['policy'][key],peer_policy[key]),'Both populations must use the same archived fresh critic'
    expected_cfg=dict(saved['config'],img_noise=0.);assert cfg==expected_cfg
    assert policy_digest(saved['policy'])==entry['full_initial_policy_hash']==init['full_initial_policy_hash']
    assert init['task_setting_change']==dict(img_noise=0.,stationary_cue_delay=True,first_return_trial=True)
    return dict(policy=saved['policy'],config=cfg,initialization=init),dict(original_actor_cnn_lstm_exact=True,archived_fresh_critic_exact=True,initial_policy_hash=policy_digest(saved['policy']))


def validate_protocol(path):
    design=read(path);assert design.pop('frozen_utc')
    assert design['version']=='native_goal_cue_memory_training_v1' and design['conditions']==['gc100','pgc33'] and design['original_seed']==1
    for name,expected in design['source_sha256'].items():assert sha(checked_path(ROOT,name))==expected,name
    for name,expected in design['input_assets'].items():assert sha(checked_path(ROOT,name))==expected,name
    q=design['qualification'];assert q['both_populations_required'] and q['no_qualification_training']
    assert (q['successes'],q['trials'],q['delay_steps'],q['variants'])==(192,256,45,['continuous'])
    t=design['training']
    for key,value in dict(updates_per_block=1024,blocks=3,total_updates=3072,trials_per_batch=32,trials_per_model=98304,
            training_seed=694801,delay_caps=[0,15,45],full_trial_bptt=160,cue_steps=15,free_steps=90,
            actor_entropy_only_free=True,critic_all_active=True,gamma=.99,entropy=8e-5,state_noise=0.,img_noise=0.).items():assert t[key]==value,key
    e=design['evaluation'];assert e['baseline_variants']==e['final_variants']==list(VARIANTS)
    assert e['interim_variants']==['brief_native','no_cue','brief_reset'] and e['all_on_memory_bank']
    assert e['no_50_percent_certificate'] and e['donor_before_first_free_input'] and e['no_adaptive_endpoint_selection']
    return design


def bank_reference(design,name):
    evaluation=design['evaluation'];manifest=checked_path(ROOT,evaluation['manifest_path'])
    assert sha(manifest)==evaluation['manifest_sha256']
    plan,record,_=banks.load_bank(manifest,name);assert record==evaluation['banks'][name]
    return plan,dict(manifest_path=evaluation['manifest_path'],manifest_sha256=evaluation['manifest_sha256'],bank_name=name,
        bank_path=record['path'],bank_sha256=record['sha256'])


def artifact_manifest(folder):
    return {str(path.relative_to(folder)):dict(sha256=sha(path),bytes=path.stat().st_size)
        for path in sorted(Path(folder).rglob('*'))if path.is_file()and not path.name.endswith('.tmp')}


def check_qualification_gate(gate,protocol_sha,qualification_root=None):
    assert gate['passed'] and gate['both_originals_qualified'] and gate['protocol_sha256']==protocol_sha
    assert len(gate['rows'])==2 and {r['condition']for r in gate['rows']}=={'gc100','pgc33'}
    for row in gate['rows']:
        condition=row['condition']
        assert row['audit_relative_path']==f'audits/{condition}.json'
        assert row['summary_relative_path']==f'{condition}/qualification_evaluation/summary.json'
        path=Path(row['audit_path'])if qualification_root is None else Path(qualification_root)/'audits'/f'{condition}.json'
        assert sha(path)==row['audit_sha256'];report=read(path)
        assert report['phase']=='qualification' and report['condition']==condition and report['protocol_sha256']==protocol_sha
        assert report['auditor_sha256']==sha(__file__) and report['passed'] and report['complete'] and not report['failures'] and not report['pending']
        child=path.parent.parent/condition
        for name,proof in report['artifacts'].items():
            artifact=checked_path(child,name);assert sha(artifact)==proof['sha256'] and artifact.stat().st_size==proof['bytes']
        summary_path=child/'qualification_evaluation/summary.json';assert sha(summary_path)==row['summary_sha256']
        summary=read(summary_path);assert summary['variant_order']==['continuous'] and summary['condition']==condition
        totals=summary['variants']['continuous']['totals']
        assert summary['variants']['continuous']['all_physical_checks'] and totals['valid_trials']==totals['trials']==row['trials']==256
        assert totals['successes']==row['successes']>=192 and not totals['invalid_trials']
    return dict(both_originals_qualified=True,exact_audits_and_raw_artifacts_verified=True)


def audit(folder,condition,phase,protocol,*,qualification_root=None):
    """Integrity may pass for a fully recorded negative qualification outcome."""
    folder=Path(folder);protocol=Path(protocol);checks={};failures=[];pending=[]
    result=dict(version='goal_cue_memory_independent_saved_audit_v1',phase=phase,condition=condition,
        folder=str(folder.resolve()),auditor_sha256=sha(__file__),protocol_sha256=sha(protocol),scope=SCOPE,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    rng=torch.get_rng_state().clone()
    try:
        assert phase in ('qualification','training') and condition in ('gc100','pgc33')
        design=validate_protocol(protocol);provenance=read(folder/'protocol.json')
        assert provenance['design']==design and provenance['protocol_sha256']==sha(protocol)
        assert provenance['phase']==phase and provenance['condition']==condition and provenance['original_seed']==1
        incoming=torch.load(folder/'incoming.pt',map_location='cpu',weights_only=True)
        initial,checks['initialization']=audit_initial(incoming,condition)
        assert read(folder/'initialization.json')==initial['initialization'] and read(folder/'training_config.json')==initial['config']
        assert incoming['provenance']==provenance
        if phase=='qualification':
            assert provenance['qualification_gate']is None
            assert incoming['version']=='goal_cue_memory_qualification_v1' and incoming['phase']=='qualification' and incoming['no_optimization']
            assert equal(incoming['policy'],initial['policy']) and 'optimizer'not in incoming
            after_path=folder/'qualification_after.pt';summary_path=folder/'summary.json';eval_path=folder/'qualification_evaluation/summary.json'
            required=(after_path,summary_path,eval_path,folder/'qualification_evaluation_integrity.json')
            pending.extend(str(path)for path in required if not path.exists())
            if not pending:
                after=torch.load(after_path,map_location='cpu',weights_only=True);assert equal(incoming,after),'Qualification changed original parameters or runtime'
                plan,reference=bank_reference(design,'qualification')
                checked=audit_evaluation(eval_path.parent,incoming,plan,reference,('continuous',));checks['evaluation']=checked
                integrity=read(folder/'qualification_evaluation_integrity.json')
                assert integrity==dict(policy_and_global_rng_unchanged=True,incoming_sha256=sha(folder/'incoming.pt'),after_sha256=sha(after_path),summary_sha256=sha(eval_path))
                summary=read(summary_path);totals=checked['variants']['continuous'];qualified=checked['gates']['qualification']['passed']
                assert summary==dict(phase='qualification',complete=True,condition=condition,qualified=qualified,no_optimization=True,
                    protocol_sha256=sha(protocol),totals=totals,policy_hash=policy_digest(initial['policy']),evaluation_summary_sha256=sha(eval_path))
                checks['qualified']=qualified
        else:
            gate=read(folder/'qualification_gate.json');assert gate==provenance['qualification_gate']
            checks['qualification_gate']=check_qualification_gate(gate,sha(protocol),qualification_root)
            rows,checks['metrics']=accepted_metrics(folder);checkpoints={}
            names=['incoming','latest','final']+[name+suffix for name in ENDPOINTS for suffix in ('_before_evaluation','')]
            for name in names:
                path=folder/f'{name}.pt'
                if not path.exists():pending.append(str(path));continue
                data=torch.load(path,map_location='cpu',weights_only=True);checkpoints[name]=data
                checks[name]=audit_checkpoint(data,initial,provenance);u=data['counters']['updates'];assert u<=len(rows)
                for key in COUNTERS:assert data['counters'][key]==sum(r[key]for r in rows[:u]),key
                verify_checkpoint_delays(rows[:u],data)
            if(folder/'resume_lineage.json').exists():
                lineage=read(folder/'resume_lineage.json')
                for item in lineage['copied_artifacts']:assert sha(checked_path(folder,item['copied']))==item['sha256']
                parent_path=folder/'resume_parent_evidence/checkpoint.pt';assert sha(parent_path)==lineage['resume_checkpoint_sha256']
                parent=torch.load(parent_path,map_location='cpu',weights_only=True)
                assert equal(parent,incoming) and parent['counters']['updates']==lineage['restored_updates']
            else:assert incoming['counters']['updates']==0 and not incoming['counters']['evaluation_records']
            plan,reference=bank_reference(design,'memory');checks['evaluations']={}
            for name,boundary in ENDPOINTS.items():
                child=folder/f'{name}_evaluation';before_name=name+'_before_evaluation'
                if name not in checkpoints or before_name not in checkpoints or not(child/'summary.json').exists():
                    pending.append(str(child/'summary.json'));continue
                before,after=checkpoints[before_name],checkpoints[name]
                assert before['counters']['updates']==after['counters']['updates']==boundary
                compare_boundary(before,after,name)
                variants=VARIANTS if name in ('baseline','block3')else ('brief_native','no_cue','brief_reset')
                record=read(folder/f'{name}_evaluation_integrity.json');assert record==after['counters']['evaluation_records'][name]
                assert record==dict(checkpoint_before_sha256=sha(folder/f'{before_name}.pt'),summary_sha256=sha(child/'summary.json'),
                    policy_optimizer_private_global_rng_unchanged=True,variants=list(variants),bank_name='memory',updates=boundary)
                checked=audit_evaluation(child,before,plan,reference,variants);checks['evaluations'][name]=checked
                assert checked['valid'],'Invalid completed training evaluation'
            if(folder/'summary.json').exists():
                summary=read(folder/'summary.json');final=checkpoints['final'];assert len(rows)==3072 and final['completed']
                assert equal(final,checkpoints['block3']) and sha(folder/'final.pt')==sha(folder/'block3.pt')
                assert equal(checkpoints['latest'],checkpoints['block3_before_evaluation'])
                assert summary['completed'] and summary['condition']==condition and summary['protocol_sha256']==sha(protocol)
                assert summary['counters']==final['counters'] and summary['final_checkpoint_sha256']==sha(folder/'final.pt')
                assert summary['active_physics_frames']==final['active_physics_frames'] and summary['active_recurrent_updates']==final['counters']['active_recurrent_updates']
                assert summary['all_fixed_endpoints_preserved'] and summary['no_endpoint_selection']
                assert set(checks['evaluations'])==set(ENDPOINTS)
            else:pending.append(str(folder/'summary.json'))
            for name in ('incomplete.json','failed_plan.pt','failed_predraw_streams.pt'):
                if(folder/name).exists():failures.append('Unresolved current-run failure evidence: '+name)
        assert torch.equal(rng,torch.get_rng_state()),'Saved-evidence audit consumed global RNG'
    except Exception as error:failures.append(f'{type(error).__name__}: {error}')
    result.update(checks=checks,failures=failures,pending=sorted(set(pending)),passed=not failures,
        complete=not failures and not pending,status='failed'if failures else 'pending'if pending else 'complete',artifacts=artifact_manifest(folder))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['qualification','training'],required=True)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--condition',choices=['gc100','pgc33'],required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/research/goal_cue_memory_training_protocol.json')
    parser.add_argument('--qualification-root',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    result=audit(args.folder,args.condition,args.phase,args.protocol,qualification_root=args.qualification_root)
    atomic_json(args.out,result)
    print(json.dumps({key:result[key]for key in ('phase','condition','passed','complete','status','failures','pending')}))
    if not result['passed']or(args.require_complete and not result['complete']):raise SystemExit(2)


if __name__=='__main__':main()
