"""Independent saved-artifact audit for the prospective feedback acquisition.

No environment step, policy forward pass, optimizer step, or new trial is run.
Native rendering and spatial-code equations are evaluated at SAVED observations.
All saved bytes remain hash checked, including pixels at discontinuous renderer
boundaries whose cross-device reconstruction is explicitly not claimed here.
"""
from __future__ import annotations
import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
from code.env import ArenaEnv, WALL_COLORS, _TARGETS
import audit_local_goal_return_baseline as base
from audit_local_goal_return_training import finite, audit_rng
import local_goal_return_training as task
import local_return_feedback as feedback
import local_return_feedback_evaluate as evaluator
import local_return_feedback_run as runner
from rl_goal_memory import atomic_json

sha, read, equal = base.sha, base.read, base.same_tree
COUNTERS = runner.COUNTERS
RGB_ATOL, CODE_ATOL, GEOMETRY_MARGIN = 1e-4, 5e-5, 5e-5
SCOPE = dict(no_policy_or_environment_rollout=True,
    physics='Independent native FREE recurrence from saved first-FREE pose, actions and exact tapes.',
    recurrent='First-FREE actor/value readout independently recomputed; every recorded training logit/value replay error checked. Full CNN/LSTM histories not independently rerun.',
    initial_state='Zero initial h/c follows the hash-locked collector and its tests; no pre-first-input state is archived. Initial pa/pr, no midtrial reset and zero state-noise arrays checked directly.',
    rgb='Pure native raycast at every active saved pose/layout. Every frame must have >=50% unambiguous pixels. Discontinuous boundary pixels remain exactly archived/hash checked but are excluded from cross-device numerical equality.',
    rgb_atol=RGB_ATOL, code_atol=CODE_ATOL,
    boundary_margins=dict(ray_wall_choice_metres=GEOMETRY_MARGIN,
        landmark_wall_coordinate_metres=GEOMETRY_MARGIN,
        stripe_fraction_of_period=GEOMETRY_MARGIN,
        projected_vertical_coordinate=GEOMETRY_MARGIN),
    summary='Success denominator always includes all 256 trials. Producer path/progress/entropy means also use all rows, assigning zero to invalid missing values; invalid counts are reported explicitly.',
    no_common_information_certificate=True)


def checked_path(root, relative):
    path = Path(relative)
    assert not path.is_absolute() and '..' not in path.parts
    return Path(root)/path


def tensor_schema(value, prefix=''):
    """Independent uncompressed tensor manifest, no conversion of dtype."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        return {prefix:dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))}
    result = {}
    if isinstance(value, dict):
        for key, child in value.items(): result.update(tensor_schema(child, f'{prefix}/{key}'))
    elif isinstance(value, (list, tuple)):
        for key, child in enumerate(value): result.update(tensor_schema(child, f'{prefix}/{key}'))
    return result


def load_raw(folder, record):
    path = checked_path(folder, record['file'])
    assert record['compression'] == 'gzip_level1_lossless'
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    with gzip.open(path, 'rb') as stream: value = torch.load(stream, map_location='cpu', weights_only=True)
    assert tensor_schema(value) == record['tensor_schema'], 'Uncompressed tensor manifest differs'
    finite(value)
    return value


def render_saved(position, heading, land_pos, stripe_phase, half):
    """Pure renderer call; creates no environment and consumes no RNG."""
    width = 84; q = (torch.arange(width, dtype=torch.float32)+.5)/width
    obj = SimpleNamespace(n=len(position), half=float(half), pos=position.float(), hd=heading.float(),
        land_pos=land_pos.float(), stripe_phase=stripe_phase.float(), _v=q*2-1,
        _col_ang=math.pi/4-math.pi/2*q, _wall_col=torch.tensor(WALL_COLORS),
        _ceil=torch.tensor([.12,.12,.16]), _floor=torch.tensor([.5,.5,.5]))
    image = ArenaEnv._render_raycast(obj).clamp(0,1).permute(0,3,1,2).contiguous()
    # Boundary locations are recomputed in float64. Each tolerance has fixed,
    # explicit units; no observed study error is used to choose this mask.
    angle = heading.double()[:,None]+obj._col_ang.double()[None]
    dx,dy = angle.cos(),angle.sin(); x,y=position.double()[:,:1],position.double()[:,1:]
    big=torch.full_like(dx,1e9)
    tx=torch.where(dx>1e-9,(half-x)/dx,torch.where(dx< -1e-9,(-half-x)/dx,big))
    ty=torch.where(dy>1e-9,(half-y)/dy,torch.where(dy< -1e-9,(-half-y)/dy,big))
    hit_x=tx<ty; t=torch.where(hit_x,tx,ty)
    wall=torch.where(hit_x,torch.where(dx>0,0,2),torch.where(dy>0,1,3))
    along=torch.where(hit_x,y+t*dy,x+t*dx)
    phase=torch.gather(stripe_phase.double(),1,wall)
    fraction=(((along+half)/(2*half))*6+phase)%1
    stripe_edge=torch.minimum(torch.minimum(fraction,1-fraction),(fraction-.15).abs())<=GEOMETRY_MARGIN
    landmark_edge=((along-torch.gather(land_pos.double(),1,wall)).abs()-.12).abs()<=GEOMETRY_MARGIN
    wall_edge=(tx-ty).abs()<=GEOMETRY_MARGIN
    depth=(t*obj._col_ang.double().cos()[None]).clamp_min(.05)
    height=(.45/depth).clamp(0,1)
    silhouette=(obj._v.double().abs()[None,:,None]-height[:,None]).abs()<=GEOMETRY_MARGIN
    ambiguous=silhouette|(wall_edge|stripe_edge|landmark_edge)[:,None,:]
    return image,ambiguous


def verify_rgb(saved, expected, ambiguous):
    assert saved.shape==expected.shape and saved.dtype==torch.float32
    assert ambiguous.shape==(len(saved),84,84)
    assert bool(((~ambiguous).float().mean((1,2))>=.5).all()), 'Insufficient independently checked pixels'
    error=(saved-expected).abs().amax(1)
    assert bool((error[~ambiguous]<=RGB_ATOL).all()), 'Fresh RGB differs away from renderer boundaries'
    return dict(pixels=error.numel(), excluded_boundary_pixels=int(ambiguous.sum()),
        boundary_mismatches_above_atol=int(((error>RGB_ATOL)&ambiguous).sum()),
        nonboundary_max_abs_error=float(error[~ambiguous].max()),
        boundary_max_abs_error=float(error[ambiguous].max()) if bool(ambiguous.any()) else 0.)


def verify_observation_history(plan,cfg,raw):
    """One-step native equations from recorded pre-action states, including brakes."""
    active=raw['active']; sched=task.schedule(plan,'cpu'); n=active.shape[1]
    assert raw['observation_position'].shape==(160,n,2)
    dt_step=cfg['action_repeat']/60; k=1-math.exp(-dt_step/.25)
    target=torch.tensor(_TARGETS,dtype=torch.float64)*torch.tensor([cfg['v_max'],cfg['v_max'],cfg['rot_max']])
    maximum=0.; maximum_heading=0.; maximum_velocity=0.; maximum_direction=0.
    for t in range(160):
        use=active[t]; free=raw['policy_mask'][t]
        if not bool(use.any()): continue
        motors=raw['motors_before_action'][t].double()
        brake=(t==15)|(sched['probe']==t)|(sched['glimpse']==t)
        assert not bool(raw['motors_before_action'][t][brake&use].any()), 'Brake missing'
        noise=plan['motor'][t].T.double()*torch.tensor([.03,.03,.15])*free[:,None]
        next_motors=motors+k*(target[raw['action'][t]]+noise-motors)
        hd=(raw['observation_heading'][t].double()+next_motors[:,2]*dt_step+math.pi)%(2*math.pi)-math.pi
        delta=torch.stack((next_motors[:,0]*hd.cos()-next_motors[:,1]*hd.sin(),
                           next_motors[:,0]*hd.sin()+next_motors[:,1]*hd.cos()),1)*dt_step
        old=raw['observation_position'][t].double()
        proposed=old+delta; new=proposed.clamp(-cfg['arena_half']+.08,cfg['arena_half']-.08)
        wall=(new!=proposed).any(1);next_motors[wall,:2]*=.2
        maximum=max(maximum,float((new[use]-raw['position'][t,use]).abs().max()))
        assert maximum<2e-5, 'Recorded post-action position disagrees with native equations'
        if t==159:continue
        alive=active[t+1]
        if not bool(alive.any()):continue
        displacement=raw['position'][t]-raw['observation_position'][t]
        assert torch.allclose(raw['observation_position'][t+1,alive],raw['position'][t,alive],atol=1e-5,rtol=0)
        err=((raw['observation_heading'][t+1].double()-hd+math.pi)%(2*math.pi)-math.pi).abs()
        maximum_heading=max(maximum_heading,float(err[alive].max()));assert maximum_heading<2e-5
        maximum_velocity=max(maximum_velocity,float((raw['observation_velocity'][t+1,alive]-displacement[alive]/dt_step).abs().max()))
        assert maximum_velocity<2e-5, 'Observation velocity is stale or post-brake'
        distance=displacement.norm(dim=1); moving=distance>.05*dt_step
        expected_dir=torch.where(moving[:,None],displacement/distance.clamp_min(1e-9)[:,None],raw['observation_direction'][t])
        maximum_direction=max(maximum_direction,float((expected_dir[alive]-raw['observation_direction'][t+1,alive]).abs().max()))
        assert maximum_direction<2e-5, 'Travel-direction observation is stale'
        next_brake=((t+1)==15)|(sched['probe']==t+1)|(sched['glimpse']==t+1)
        next_motors[next_brake]=0
        assert torch.allclose(next_motors[alive],raw['motors_before_action'][t+1,alive].double(),atol=2e-5,rtol=0)
    assert not bool(raw['observation_velocity'][0].any())
    assert not bool(raw['motors_before_action'][0].any())
    for i in range(n):
        cue=int(sched['glimpse'][i])
        assert torch.equal(raw['observation_position'][cue,i],plan['endpoint'][i]), 'Endpoint snap differs from plan'
        error=(float(raw['observation_heading'][cue,i])-float(plan['heading'][i])+math.pi)%(2*math.pi)-math.pi
        assert abs(error)<2e-5, 'Actual probe heading differs from prescribed heading'
    return dict(observation_position_max_error=maximum,observation_heading_max_error=maximum_heading,
        observation_velocity_max_error=maximum_velocity,observation_travel_direction_max_error=maximum_direction)


def verify_inputs(plan,cfg,raw):
    n=raw['active'].shape[1]
    for key,shape in dict(rgb=(160,n,3,84,84),code=(160,n,512),goal=(n,512),
            logits=(160,n,6),values=(160,n),noise=(160,2,n,cfg['nh'])).items():
        assert raw[key].shape==shape and raw[key].dtype==torch.float32,key
    assert not bool(raw['noise'].any()) and not bool(raw['reset_before'].any()) and not bool(raw['reset_after'].any())
    assert not bool(raw['bootstrap'].any()) and torch.equal(raw['goal_world'],plan['goal'])
    source=feedback.make_source(cfg,'cpu');active=raw['active'];max_code=0.
    totals=dict(pixels=0,excluded_boundary_pixels=0,boundary_mismatches_above_atol=0,
        nonboundary_max_abs_error=0.,boundary_max_abs_error=0.)
    for t in range(160):
        use=active[t]
        assert not bool(raw['rgb'][t,~use].any()), 'Padding images must be zero'
        if not bool(use.any()): continue
        expected=source(raw['observation_position'][t,use],raw['observation_direction'][t,use],raw['observation_velocity'][t,use])
        max_code=max(max_code,float((expected-raw['code'][t,use]).abs().max()))
        assert max_code<=CODE_ATOL, 'Current code is stale, ordinary-substituted, or rotated'
        image,boundary=render_saved(raw['observation_position'][t,use],raw['observation_heading'][t,use],
            plan['land_pos'][use],plan['stripe_phase'][use],cfg['arena_half'])
        report=verify_rgb(raw['rgb'][t,use],image,boundary)
        for key,value in report.items(): totals[key]=max(totals[key],value) if key.endswith('error') else totals[key]+value
    # Reconstruct the actual one-step pickup snapshot, not the goal center code.
    angle=plan['pickup_angle'];direction=torch.stack((angle.cos(),angle.sin()),1)
    dt_step=cfg['action_repeat']/60; speed=cfg['v_max']*(1-math.exp(-dt_step/.25))
    before=plan['goal']-.02*direction;pickup=before+direction*(speed*dt_step)
    velocity=(pickup-before)/dt_step
    travel=(pickup-before)/(pickup-before).norm(dim=1,keepdim=True).clamp_min(1e-9)
    goal=source.goal(pickup,travel,velocity)
    assert torch.allclose(goal,raw['goal'],atol=CODE_ATOL,rtol=0), 'Saved goal differs from actual pickup snapshot'
    totals.update(current_code_max_abs_error=max_code,goal_code_max_abs_error=float((goal-raw['goal']).abs().max()))
    return totals


def verify_actor(plan,cfg,raw,rows,policy):
    n=len(rows);nh=cfg['nh'];first=raw['first_free'];assert set(first)==set(range(n))
    sched=task.schedule(plan,'cpu'); maximum=0.
    for i,row in enumerate(rows):
        t=int(sched['free'][i]);f=first[i];h=f['state'][:nh].double()
        assert f['state'].shape==(2*nh,) and f['input'].shape==(1287,)
        expected=h@policy['pi.weight'].double().T+policy['pi.bias'].double()
        bound=2e-6+8*torch.finfo(torch.float32).eps*(h.abs()@policy['pi.weight'].double().abs().T+policy['pi.bias'].double().abs())
        error=(expected-f['logits'].double()).abs();assert bool((error<=bound).all())
        maximum=max(maximum,float(error.max()))
        value=h@policy['v.weight'].double().T+policy['v.bias'].double()
        vb=2e-6+8*torch.finfo(torch.float32).eps*(h.abs()@policy['v.weight'].double().abs().T+policy['v.bias'].double().abs())
        assert float((value-f['value'].double()).abs())<=float(vb)
        assert torch.equal(f['logits'],raw['logits'][t,i]) and torch.equal(f['value'],raw['values'][t,i])
        for key,rawkey in [('position','observation_position'),('heading','observation_heading'),('motors','motors_before_action')]:
            assert torch.equal(f[key],raw[rawkey][t,i])
        assert torch.equal(f['input'][256:768],raw['code'][t,i])
        assert torch.equal(f['input'][768:1280],raw['goal'][i])
        onehot=torch.nn.functional.one_hot(raw['pa'][t,i],6).float()
        assert torch.equal(f['input'][1280:1286],onehot) and f['input'][-1]==0
        logits=raw['logits'][:,i].double();prob=logits.softmax(1);free=raw['policy_mask'][:,i]
        entropy=-(prob*prob.clamp_min(1e-300).log()).sum(1)
        assert torch.allclose(f['probabilities'].double(),prob[t],atol=2e-6,rtol=2e-6)
        for field in ('probabilities','logits'): assert row['first_free_'+field]==f[field].tolist()
        for field in ('entropy','value'): assert row['first_free_'+field]==float(f[field])
        assert abs(float(entropy[t])-float(f['entropy']))<3e-6
        measurements=dict(mean_free_entropy=float(entropy[free].mean()),
            mean_max_action_probability=float(prob[free].amax(1).mean()),
            mean_value=float(raw['values'][free,i].double().mean()),max_abs_logit=float(logits[free].abs().max()))
        for key,value in measurements.items(): assert math.isclose(row[key],value,rel_tol=3e-6,abs_tol=3e-6),key
        for step in free.nonzero().flatten().tolist():
            action=int(raw['action'][step,i]);u=float(plan['action'][step,i]);p=prob[step]
            assert u>=float(p[:action].sum())-2e-6
            assert action==5 or u<=float(p[:action+1].sum())+2e-6
    return dict(first_free_actor_max_abs_error=maximum,all_free_action_cdf_and_row_statistics_checked=True)


def verify_reference(plan,cfg,raw,rows,controller):
    """Independent uniform/greedy rules, never evaluating a neural policy."""
    assert controller in ('uniform','planner')
    assert not ({'rgb','code','noise','logits'}&set(raw)), 'References must not retain neural-input histories'
    assert not bool(raw['values'].any()) and not bool(raw['bootstrap'].any())
    assert not bool(raw['reset_before'].any()) and not bool(raw['reset_after'].any())
    sched=task.schedule(plan,'cpu');dt_step=cfg['action_repeat']/60;k=1-math.exp(-dt_step/.25)
    targets=torch.tensor(_TARGETS,dtype=torch.float64)*torch.tensor([cfg['v_max'],cfg['v_max'],cfg['rot_max']])
    free=raw['policy_mask'];n=len(rows);assert set(raw['first_free'])==set(range(n))
    for i,row in enumerate(rows):
        first=raw['first_free'][i];start=int(sched['free'][i])
        assert not bool(first['state'].any()) and not bool(first['logits'].any()) and first['value']==0
        assert not bool(first['input'][:256].any()), 'Reference embedding must be parameterless zero'
        for key,rawkey in [('position','observation_position'),('heading','observation_heading'),('motors','motors_before_action')]:
            assert torch.equal(first[key],raw[rawkey][start,i])
        for field in ('probabilities','logits'):assert row['first_free_'+field]==first[field].tolist()
        for field in ('entropy','value'):assert row['first_free_'+field]==float(first[field])
        for t in free[:,i].nonzero().flatten().tolist():
            action=int(raw['action'][t,i])
            if controller=='uniform':
                probability=torch.full((6,),1/6,dtype=torch.float64)
                u=float(plan['action'][t,i]);assert u>=float(probability[:action].sum())-2e-6
                assert action==5 or u<=float(probability[:action+1].sum())+2e-6
            else:
                motors=raw['motors_before_action'][t,i].double()
                next_motors=motors[None]+k*(targets-motors[None])
                hd=raw['observation_heading'][t,i].double()+next_motors[:,2]*dt_step
                delta=torch.stack((next_motors[:,0]*hd.cos()-next_motors[:,1]*hd.sin(),
                    next_motors[:,0]*hd.sin()+next_motors[:,1]*hd.cos()),1)*dt_step
                pos=(raw['observation_position'][t,i].double()+delta).clamp(-cfg['arena_half']+.08,cfg['arena_half']-.08)
                distances=(pos-plan['goal'][i].double()).norm(dim=1)
                assert float(distances[action]-distances.min())<=2e-6, 'Planner differs from native expected-step rule'
                probability=torch.nn.functional.one_hot(torch.tensor(action),6).double()
            if t==start:assert torch.allclose(first['probabilities'].double(),probability,atol=2e-6,rtol=0)
        ent=math.log(6) if controller=='uniform' else 0.
        assert abs(row['mean_free_entropy']-ent)<3e-6 and abs(row['first_free_entropy']-ent)<3e-6
        assert row['mean_value']==row['max_abs_logit']==0
        assert abs(row['mean_max_action_probability']-(1/6 if controller=='uniform' else 1.))<3e-6
    return dict(parameterless_reference_first_free_verified=True,all_reference_actions_checked=True,
        input_history_scope='No RGB/code history is saved for physical references; source provenance only, not reconstructed.')


def audit_metric(row):
    finite(row);assert 1<=row['update']<=1024
    assert row['completed_trials']==row['setup_pickups']==32 and row['setup_raw_reward']==320
    assert row['setup_learning_reward']==row['forced_hits']==0
    assert row['terminal_bootstrap_zero'] and row['controller']=='policy'
    assert row['fresh_feedback_on_every_active_input'] and row['state_noise_sigma']==row['midtrial_resets']==0
    assert row['unique_endpoint_cue'] is False and row['allocated_slots']==5120
    assert row['active_physics_updates']==row['forced_updates']+row['free_updates']
    assert row['active_physics_updates']+row['padded_slots']==5120
    assert 32<=row['free_updates']<=960 and 0<=row['free_hits']<=32
    assert 32*31<=row['forced_updates']<=32*43
    assert row['clearance_min']>.01 and row['wall_clearance_min']>.01 and 0<=row['snap_max_m']<1e-5
    assert 0<=row['replay_error']<3e-4 and 0<=row['value_replay_error']<3e-4 and row['grad_norm']>=0
    assert 0<=row['entropy_per_free_decision']<=math.log(6)+2e-6
    assert math.isclose(row['entropy_sum_per_trial'],row['entropy_per_free_decision']*row['free_updates']/32,rel_tol=3e-6,abs_tol=3e-6)
    loss=(100/160)*(row['policy_loss']+.5*row['value_loss']-8e-5*row['entropy_sum_per_trial'])
    assert math.isclose(row['loss'],loss,rel_tol=2e-5,abs_tol=2e-5) and row['value_loss']>=0
    assert len(row['action_counts'])==6 and sum(row['action_counts'])==row['free_updates']


def accepted_metrics(folder,cutoff=None,ancestor='resume_parent_evidence'):
    folder=Path(folder);prior=[];start=0;discarded=0
    if (folder/'resume_lineage.json').exists():
        lineage=read(folder/'resume_lineage.json');start=lineage['restored_updates']
        prior,meta=accepted_metrics(folder/ancestor,start,'ancestor_evidence');discarded+=meta['rolled_back_rows']
    lines=(folder/'metrics.jsonl').read_bytes().splitlines();rows=[];partial=False
    for i,line in enumerate(lines):
        try: row=json.loads(line)
        except json.JSONDecodeError:
            assert i==len(lines)-1;partial=True;break
        audit_metric(row);assert row['update']==start+len(rows)+1;rows.append(row)
    if cutoff is not None:
        assert start<=cutoff<=start+len(rows);discarded+=sum(r['update']>cutoff for r in rows)
        rows=[r for r in rows if r['update']<=cutoff]
    rows=prior+rows;assert [r['update'] for r in rows]==list(range(1,len(rows)+1))
    active=0
    for row in rows: active+=row['active_physics_updates'];assert row['active_physics_frames']==active*4
    return rows,dict(accepted_updates=len(rows),rolled_back_rows=discarded,partial_last_row=partial)


def audit_initial(initial,condition,mode,root=ROOT):
    init=initial['initialization'];cfg=initial['config'];runner.assert_config(cfg)
    assert cfg['cond']==condition and init['condition']==condition and init['actor_mode']==mode
    original_path=checked_path(root,init['parent_checkpoint_path']);archive_path=checked_path(root,init['archived_heads_path'])
    assert sha(original_path)==init['parent_checkpoint_sha256']
    assert sha(original_path.with_suffix('.json'))==init['parent_config_sha256']
    assert sha(archive_path)==init['archived_heads_sha256']
    original=torch.load(original_path,map_location='cpu',weights_only=True)
    archive=torch.load(archive_path,map_location='cpu',weights_only=True)
    assert original['frames']==1_000_000_000 and archive['no_optimization']
    expected_cfg=read(original_path.with_suffix('.json'));expected_cfg.update(n_envs=32,rollout=160,entropy=8e-5)
    assert cfg==expected_cfg, 'Incoming configuration differs from original with prescribed overrides'
    assert init['parent_protocol_sha256']==feedback.PARENT_PROTOCOL_SHA256
    assert initial['no_forward_pass'] and initial['no_optimization']
    assert initial['collector_source_sha256']==sha(feedback.__file__)
    assert original['policy'].keys()==archive['policy'].keys()==initial['policy'].keys()
    for key,value in initial['policy'].items():
        head=key in feedback.HEADS
        expected=archive['policy'][key] if key.startswith('v.') or (mode=='reset_actor' and key.startswith('pi.')) else original['policy'][key]
        assert torch.equal(value,expected),f'Initial composition differs: {key}'
        if head:assert task.tensor_hash(value)==init['incoming_head_hashes'][key]
        else:
            assert torch.equal(archive['policy'][key],original['policy'][key])
            assert task.tensor_hash(value)==init['original_nonhead_hashes'][key]
    assert base.policy_hash(initial['policy'])==init['full_initial_policy_hash']
    assert init['all_parameters_trainable'] and init['critic_from_exact_untrained_archive'] and init['no_optimization']
    return dict(exact_original_nonheads=True,exact_actor_mode=mode,exact_archived_fresh_critic=True,
        full_policy_hash=base.policy_hash(initial['policy']))


def audit_checkpoint(data,initial,provenance):
    finite(data);cfg=initial['config'];c=data['counters'];u=c['updates'];records=c['evaluation_records']
    assert data['version']==runner.VERSION and data['no_inflight_trial'] is True
    assert data['boundary_state']=='Every trial terminal; next trial h/c zero, pa=-1, pr=0'
    assert data['config']==cfg and data['initialization']==initial['initialization'] and data['provenance']==provenance
    assert 0<=u<=1024 and c['trials']==u*32 and set(records)<={'baseline','final'}
    assert not u or 'baseline' in records
    assert 'final' not in records or u==1024
    assert c['setup_pickups']==c['trials'] and c['active_physics_updates']==c['forced_updates']+c['free_updates']
    assert c['active_physics_updates']+c['padded_slots']==u*5120
    assert data['active_physics_frames']==c['active_physics_updates']*4
    assert data['completed']==(u==1024 and 'final' in records)
    assert data['policy'].keys()==initial['policy'].keys()
    for key,value in data['policy'].items():assert value.shape==initial['policy'][key].shape and value.dtype==initial['policy'][key].dtype
    optimizer=data['optimizer'];assert len(optimizer['param_groups'])==1;group=optimizer['param_groups'][0]
    for key,value in dict(lr=3e-4,alpha=.99,eps=1e-5,weight_decay=0,momentum=0,centered=False,
            capturable=False,maximize=False,differentiable=False,foreach=None).items():assert group[key]==value,key
    assert group['params']==list(range(len(data['policy'])))
    assert set(optimizer['state'])==(set(group['params']) if u else set())
    for index,record in optimizer['state'].items():
        value=list(data['policy'].values())[index]
        assert set(record)=={'step','square_avg'} and float(record['step'])==u
        assert record['square_avg'].shape==value.shape and record['square_avg'].dtype==value.dtype
        assert bool((record['square_avg']>=0).all())
    stream=data['streams'];assert stream['seed']==494801 and stream['batches']==u and stream['contexts']==u*32
    assert set(stream['generators'])=={'plan','motor','action','state'}
    for state in stream['generators'].values():audit_rng(state)
    rng=data['global_rng'];assert set(rng)=={'cpu','cuda','mps'};audit_rng(rng['cpu'])
    assert len(rng['cuda'])==(1 if provenance['device'].startswith('cuda') else 0)
    for state in rng['cuda']:audit_rng(state,cuda=True)
    if rng['mps'] is not None:audit_rng(rng['mps'],cuda=True)
    if u==0:
        assert equal(data['policy'],initial['policy'])
        assert equal(stream,task.TrialStreams(494801).snapshot())
        assert all(c[k]==0 for k in COUNTERS)
    return dict(updates=u,trials=c['trials'],evaluation_records=list(records),completed=data['completed'],
        active_physics_frames=data['active_physics_frames'],policy_hash=base.policy_hash(data['policy']))


def compare_boundary(before,after,endpoint):
    assert set(before)==set(after)
    for key in before:
        if key not in ('counters','completed'):assert equal(before[key],after[key]),f'Evaluation changed {key}'
    old=before['counters'];new=after['counters']
    assert endpoint not in old['evaluation_records']
    assert set(new['evaluation_records'])==set(old['evaluation_records'])|{endpoint}
    for key in old:
        if key!='evaluation_records':assert equal(old[key],new[key])
    for key in old['evaluation_records']:assert old['evaluation_records'][key]==new['evaluation_records'][key]


def summarize_rows(rows):
    assert len(rows)==256 and {(r['context'],r['heading_index']) for r in rows}=={(c,h) for c in range(16) for h in range(16)}
    assert all(r['variant']=='feedback_native' and isinstance(r['hit'],bool) for r in rows)
    assert all(not r['hit'] for r in rows if r.get('invalid',False))
    mean=lambda key:sum(r.get(key,0.) for r in rows)/256
    invalid=sum(r.get('invalid',False) for r in rows)
    return dict(trials=256,valid_trials=256-invalid,
        descriptive_mean_convention='Means use all prescribed rows; invalid trials contribute zero. Invalid/valid counts are explicit; invalid trials remain failures.',
        successes=sum(r['hit'] for r in rows),invalid_trials=invalid,
        success_fraction=sum(r['hit'] for r in rows)/256,mean_path_m=mean('free_path_m'),mean_progress_m=mean('progress_m'),
        mean_free_entropy=mean('mean_free_entropy'),mean_initial_free_entropy=mean('first_free_entropy'),
        action_counts=[sum(r.get('action_counts',[0]*6)[i] for r in rows) for i in range(6)],free_updates=sum(r.get('free_steps',0) for r in rows))


def audit_evaluation(folder,checkpoint,plan,reference,controller='policy'):
    folder=Path(folder);report=read(folder/'summary.json');cfg=checkpoint['config'];contexts=[];rows=[]
    assert report['version']=='feedback_acquisition_evaluation_v2' and report['complete']
    assert report['condition']==cfg['cond'] and report['actor_mode']==checkpoint['provenance']['actor_mode']
    assert report['controller']==controller and report['checkpoint_policy_hash']==base.policy_hash(checkpoint['policy'])
    assert report['source_sha256']==sha(evaluator.__file__) and report['collector_sha256']==sha(feedback.__file__)
    assert report['reference']==reference and report['no_optimization'] and report['state_noise_sigma']==0
    assert report['common_information_certificate_applies'] is False and report['success_threshold']==192
    assert report['environment']==dict(device=checkpoint['provenance']['device'],torch_version=checkpoint['provenance']['torch_version'])
    assert len(report['contexts'])==16
    for index in range(16):
        child=folder/f'context{index}';r=read(child/'result.json');assert r==report['contexts'][index] and r['context']==index
        chunk=evaluator.context_plan(plan,index);expected=dict(reference,context_index=index,context_ids=chunk['context'].tolist())
        assert read(child/'input.json')==expected;checks={}
        if r['valid']:
            raw=load_raw(child,r['raw']);assert raw['plan_reference']==expected
            assert torch.equal(raw['goal_world'],chunk['goal']) and not bool(raw['bootstrap'].any())
            assert not (child/'failed_plan.pt.gz').exists(), 'Unresolved failure beside success'
            checks.update(base.verify_trajectory(chunk,cfg,raw,r['rows'],raw['first_free'],['feedback_native']))
            checks.update(verify_observation_history(chunk,cfg,raw))
            if controller=='policy':
                checks.update(verify_inputs(chunk,cfg,raw))
                checks.update(verify_actor(chunk,cfg,raw,r['rows'],checkpoint['policy']))
            else:checks.update(verify_reference(chunk,cfg,raw,r['rows'],controller))
            a=r['audit'];assert a['setup_pickups']==16 and a['setup_raw_reward']==160
            assert a['setup_learning_reward']==a['forced_hits']==0 and a['terminal_bootstrap_zero']
            assert a['fresh_feedback_on_every_active_input'] and a['state_noise_sigma']==a['midtrial_resets']==0
            assert a['unique_endpoint_cue'] is False and a['controller']==controller
            assert a['active_physics_updates']==int(raw['active'].sum())
            assert a['free_updates']==int(raw['policy_mask'].sum()) and a['free_hits']==sum(row['hit'] for row in r['rows'])
            assert a['active_physics_updates']==a['free_updates']+a['forced_updates']
            assert a['active_physics_updates']+a['padded_slots']==a['allocated_slots']==2560
            assert a['completed_trials']==16 and a['clearance_min']>.01 and a['wall_clearance_min']>.01 and a['snap_max_m']<1e-5
        else:
            assert equal(load_raw(child,r['failed_plan']),chunk)
            assert r['failure_category']=='collection' and r['error_type'] and 'error' in r
            assert len(r['rows'])==16 and all(row.get('invalid') and not row['hit'] for row in r['rows'])
        rows.extend(r['rows']);contexts.append(dict(context=index,valid=r['valid'],checks=checks))
    assert rows==report['rows'];totals=summarize_rows(rows);assert totals==report['totals']
    physical=all(r['valid'] for r in contexts);assert physical==report['all_physical_checks']
    assert report['criterion_pass']==(controller=='policy' and physical and totals['successes']>=192)
    return dict(complete=True,contexts=contexts,totals=totals,all_physical_checks=physical,criterion_pass=report['criterion_pass'])


def audit(folder,condition,mode,protocol=runner.PROTOCOL):
    folder=Path(folder);protocol=Path(protocol);failures=[];pending=[];checks={}
    report=dict(version='feedback_acquisition_independent_saved_audit_v2',condition=condition,actor_mode=mode,
        folder=str(folder.resolve()),auditor_sha256=sha(__file__),scope=SCOPE,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    try:
        frozen=read(protocol);design=dict(frozen);assert design.pop('frozen_utc')
        assert design['version']=='feedback_acquisition_reward_only_v2'
        for name,expected in design['source_sha256'].items():assert sha(checked_path(ROOT,name))==expected,name
        for name,expected in design['input_assets'].items():assert sha(checked_path(ROOT,name))==expected,name
        report['protocol_sha256']=sha(protocol)
        provenance=read(folder/'protocol.json');assert provenance['design']==design and provenance['protocol_sha256']==sha(protocol)
        assert provenance['condition']==condition and provenance['actor_mode']==mode and provenance['original_seed']==1
        manifest=read(runner.INITIALIZATION)
        entry=next(r for r in manifest['policies'] if r['condition']==condition and r['actor_mode']==mode)
        initial=torch.load(checked_path(ROOT,entry['path']),map_location='cpu',weights_only=True)
        assert sha(ROOT/entry['path'])==entry['sha256'] and (ROOT/entry['path']).stat().st_size==entry['bytes']
        checks['initialization']=audit_initial(initial,condition,mode)
        assert read(folder/'initialization.json')==initial['initialization'] and read(folder/'training_config.json')==initial['config']
        rows,checks['metrics']=accepted_metrics(folder)
        checkpoints={}
        for name in ('incoming','baseline_before_evaluation','baseline','latest','final_before_evaluation','final'):
            path=folder/f'{name}.pt'
            if not path.exists():pending.append(str(path));continue
            data=torch.load(path,map_location='cpu',weights_only=True)
            checkpoints[name]=data;checks[name]=audit_checkpoint(data,initial,provenance)
            u=data['counters']['updates'];assert u<=len(rows)
            for key in COUNTERS:assert data['counters'][key]==sum(r[key] for r in rows[:u]),key
        assert 'incoming' in checkpoints
        if 'latest' in checkpoints:assert checkpoints['latest']['counters']['updates']%128==0
        if (folder/'resume_lineage.json').exists():
            lineage=read(folder/'resume_lineage.json')
            for item in lineage['copied_artifacts']:assert sha(checked_path(folder,item['copied']))==item['sha256']
            parent=torch.load(folder/'resume_parent_evidence/checkpoint.pt',map_location='cpu',weights_only=True)
            assert sha(folder/'resume_parent_evidence/checkpoint.pt')==lineage['resume_checkpoint_sha256']
            assert equal(parent,checkpoints['incoming']) and parent['counters']['updates']==lineage['restored_updates']
        else:assert checkpoints['incoming']['counters']['updates']==0 and not checkpoints['incoming']['counters']['evaluation_records']
        bank_path=checked_path(ROOT,design['evaluation']['bank']['path'])
        plan=torch.load(bank_path,map_location='cpu',weights_only=True)
        assert tensor_schema(plan)==design['evaluation']['bank']['tensor_schema']
        reference=dict(manifest_path=design['evaluation']['manifest_path'],manifest_sha256=design['evaluation']['manifest_sha256'],
            bank_path=design['evaluation']['bank']['path'],bank_sha256=design['evaluation']['bank']['sha256'])
        checks['evaluations']={}
        for endpoint,u in [('baseline',0),('final',1024)]:
            child=folder/f'{endpoint}_evaluation'
            if endpoint not in checkpoints or endpoint+'_before_evaluation' not in checkpoints or not (child/'summary.json').exists():
                pending.append(str(child/'summary.json'));continue
            before=checkpoints[endpoint+'_before_evaluation'];after=checkpoints[endpoint]
            assert before['counters']['updates']==after['counters']['updates']==u
            compare_boundary(before,after,endpoint)
            record=read(folder/f'{endpoint}_evaluation_integrity.json')
            assert record==after['counters']['evaluation_records'][endpoint]
            assert record['endpoint']==endpoint and record['updates']==u and record['policy_optimizer_private_global_rng_unchanged']
            assert record['checkpoint_before_sha256']==sha(folder/f'{endpoint}_before_evaluation.pt')
            assert record['summary_sha256']==sha(child/'summary.json')
            result=audit_evaluation(child,before,plan,reference);checks['evaluations'][endpoint]=result
            assert record['successes']==result['totals']['successes'] and result['all_physical_checks']
        if (folder/'summary.json').exists():
            summary=read(folder/'summary.json');final=checkpoints['final']
            assert summary['completed'] and final['completed'] and len(rows)==1024
            assert summary['counters']==final['counters'] and summary['active_physics_frames']==final['active_physics_frames']
            assert summary['final_checkpoint_sha256']==sha(folder/'final.pt') and summary['protocol_sha256']==sha(protocol)
            assert summary['initialization']==initial['initialization'] and summary['final_evaluation']=='final_evaluation/summary.json'
            assert set(checks['evaluations'])=={'baseline','final'}
            assert checkpoints['latest']['counters']['updates']==1024
            assert equal(checkpoints['latest'],checkpoints['final_before_evaluation'])
        else:pending.append(str(folder/'summary.json'))
        for name in ('incomplete.json','failed_plan.pt','failed_predraw_streams.pt'):
            if (folder/name).exists():failures.append('Unresolved current-run failure evidence: '+name)
    except Exception as error:
        failures.append(f'{type(error).__name__}: {error}')
    report.update(checks=checks,failures=failures,pending=sorted(set(pending)),passed=not failures,
        complete=not failures and not pending,status='failed' if failures else ('pending' if pending else 'complete'))
    report['artifacts']={str(path.relative_to(folder)):dict(sha256=sha(path),bytes=path.stat().st_size)
        for path in sorted(folder.rglob('*')) if path.is_file() and not path.name.endswith('.tmp')}
    return report


def audit_references(folder,protocol=runner.PROTOCOL):
    """Once-only physical controls, separate from all four learning-arm audits."""
    folder=Path(folder);failures=[];checks={}
    try:
        design=read(protocol);assert design.pop('frozen_utc')
        for name,expected in design['source_sha256'].items():assert sha(checked_path(ROOT,name))==expected,name
        for name,expected in design['input_assets'].items():assert sha(checked_path(ROOT,name))==expected,name
        top=read(folder/'summary.json');assert top['complete'] and top['no_policy_or_optimization']
        assert top['reference_trials']==512 and set(top['results'])=={'uniform','planner'}
        assert top['manifest_sha256']==design['evaluation']['manifest_sha256']
        bank=design['evaluation']['bank'];plan=torch.load(checked_path(ROOT,bank['path']),map_location='cpu',weights_only=True)
        assert tensor_schema(plan)==bank['tensor_schema']
        reference=dict(manifest_path=design['evaluation']['manifest_path'],manifest_sha256=design['evaluation']['manifest_sha256'],
            bank_path=bank['path'],bank_sha256=bank['sha256'])
        cfg=read(ROOT/'models/gc100/gc100_lstm_s1_1000M.json');cfg.update(n_envs=32,rollout=160,entropy=8e-5)
        for controller in ('uniform','planner'):
            child=folder/controller;local=read(child/'summary.json')
            assert sha(child/'summary.json')==top['results'][controller]['summary_sha256']
            checkpoint=dict(config=cfg,policy={},provenance=dict(actor_mode=None,**local['environment']))
            checks[controller]=audit_evaluation(child,checkpoint,plan,reference,controller)
            assert checks[controller]['totals']==top['results'][controller]['totals']
    except Exception as error:failures.append(f'{type(error).__name__}: {error}')
    return dict(version='feedback_acquisition_reference_audit_v2',folder=str(folder.resolve()),auditor_sha256=sha(__file__),
        protocol_sha256=sha(protocol),scope=SCOPE,checks=checks,failures=failures,passed=not failures,complete=not failures,
        status='failed' if failures else 'complete',pending=[],
        artifacts={str(p.relative_to(folder)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(folder.rglob('*')) if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--condition',choices=['gc100','pgc33'])
    p.add_argument('--actor-mode',choices=feedback.ACTOR_MODES)
    p.add_argument('--references',action='store_true')
    p.add_argument('--protocol',type=Path,default=runner.PROTOCOL);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    if a.references:
        assert not a.condition and not a.actor_mode
        result=audit_references(a.folder,a.protocol)
    else:
        assert a.condition and a.actor_mode
        result=audit(a.folder,a.condition,a.actor_mode,a.protocol)
    atomic_json(a.out,result)
    print(json.dumps({k:result[k] for k in ('passed','complete','status','failures','pending')}))
    if not result['passed'] or (a.require_complete and not result['complete']):raise SystemExit(1)


if __name__=='__main__':main()
