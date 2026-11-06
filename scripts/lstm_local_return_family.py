"""Task-native local-return state families, collected without changing the task.

Read-only LSTMCell hooks capture the actual cue and first-FREE states. This
module computes no manifold, recovery, decoder, behavioral gate or selection.
It does not load a trained endpoint or execute a study automatically.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from torch import nn
import local_goal_return_training as task
import local_goal_return_training_evaluate as evaluator
from rl_goal_memory import state_digest

CUES=('native','ordinary','plus60','minus60')
VARIANTS=tuple('blind_'+name for name in CUES)
PROTOCOL=ROOT/'docs/research/local_goal_return_training_protocol.json'
PROTOCOL_HASH='0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
sha=evaluator.sha


def tensor_manifest(plan):
    return {key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=task.tensor_hash(value))
        for key,value in plan.items() if isinstance(value,torch.Tensor)}


def _densify_context(bank,context,phases=768,delay=60):
    """Pure transformation; also usable with explicitly synthetic test banks."""
    assert phases>=4 and delay==60 and 0<=context<bank['n_contexts']
    original=evaluator.context_plan(bank,context);n=original['headings'];result={}
    for key,value in original.items():
        if not isinstance(value,torch.Tensor):result[key]=value;continue
        axis=2 if key in ('motor','state') else (1 if key=='action' else 0)
        first=value.narrow(axis,0,1)
        if key not in ('heading','heading_index'):
            assert torch.equal(value,first.expand_as(value)),f'Context field is not shared: {key}'
        shape=list(value.shape);shape[axis]=phases
        result[key]=first.expand(*shape)
    formula=original['heading'][0]+torch.arange(phases,dtype=original['heading'].dtype)*(2*math.pi/phases)
    result['heading']=formula.clone()
    if phases%n==0:result['heading'][::phases//n]=original['heading']
    result['heading_index']=torch.arange(phases,dtype=original['heading_index'].dtype)
    result['delay']=torch.full((phases,),delay,dtype=original['delay'].dtype)
    result.update(n_contexts=1,headings=phases)
    metadata=dict(context=context,phases=phases,original_delay=int(original['delay'][0]),delay_override=delay,
        heading_rule='bank first heading + 2*pi*i/N; original16 heading bytes retained at matching anchors when N divisible16',
        sealed_heading_anchor_bytes_preserved=phases%n==0,
        maximum_heading_anchor_rounding_correction_rad=float((result['heading']-formula).abs().max()),
        transformed_fields=['heading','heading_index','delay','headings'],
        other_context_fields_and_fullclock_tapes_from_original_first_column=True,
        no_new_random_draws=True)
    return result,metadata


def dense_plan(cfg,context,phases=768,delay=60):
    """Return the sealed hard context, densified without selecting new trials."""
    assert context in (0,1) and cfg['nh']==256 and cfg['n_cells']==512
    assert cfg['cond'] in ('gc100','pgc33') and cfg['seed']==1
    assert sha(PROTOCOL)==PROTOCOL_HASH
    frozen=json.loads(PROTOCOL.read_text())
    for path,digest in frozen['source_sha256'].items():assert sha(ROOT/path)==digest,path
    reference=json.loads((ROOT/f"models/{cfg['cond']}/{cfg['cond']}_lstm_s1_1000M.json").read_text())
    reference.update(n_envs=32,rollout=160,entropy=.01);assert cfg==reference
    bank,record,_=evaluator.load_bank('final_hard')
    plan,metadata=_densify_context(bank,context,phases,delay)
    metadata.update(bank='final_hard',bank_path=record['path'],bank_sha256=record['sha256'],
        behavior_protocol_sha256=evaluator.BEHAVIOR_HASH,training_protocol_sha256=PROTOCOL_HASH,
        context_and_phase_selection='Prescribed contexts0/1 and all phases; no physical/neural outcome selection',
        transformed_plan_tensors=tensor_manifest(plan))
    return plan,metadata


def slice_plan(plan,indices):
    """Keep actual phase IDs and the matching complete160-update tapes."""
    indices=torch.as_tensor(indices,dtype=torch.long)
    assert indices.ndim==1 and indices.numel()>0 and len(indices.unique())==len(indices)
    assert int(indices.min())>=0 and int(indices.max())<len(plan['heading'])
    assert plan['n_contexts']==1
    result={}
    for key,value in plan.items():
        if isinstance(value,torch.Tensor):
            axis=2 if key in ('motor','state') else (1 if key=='action' else 0)
            result[key]=value.index_select(axis,indices).clone()
        else:result[key]=value
    result.update(n_contexts=1,headings=len(indices))
    return result


def left_movement(cfg):
    """Native noiseless LEFT displacement in angle from braked motor state.

    Velocity is evaluated in float32 as in the arena; angular increments are
    accumulated in float64. Actual saved headings independently check rounding.
    """
    velocity=torch.tensor(0.,dtype=torch.float32);k=1-math.exp(-(cfg['action_repeat']/60)/.25)
    result={'0':0.};angle=0.
    for step in range(1,61):
        velocity=velocity+(cfg['rot_max']-velocity)*k
        angle+=float(velocity*(cfg['action_repeat']/60));result[str(step)]=angle
    return result


def _wrap(angle):return torch.atan2(angle.sin(),angle.cos())


def _rng():
    return dict(cpu=torch.random.get_rng_state().clone(),
        cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def _same_rng(a,b):
    return torch.equal(a['cpu'],b['cpu']) and len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))


@torch.inference_mode()
def collect_family(net,cfg,plan,chunk_size=48,device='cpu',provenance=None):
    """Capture unchanged actual task.collect calls, preserving policy and RNG.

    Starts are POST-cue. At t60 the first FREE input has already been processed;
    the next action is pi(first_free_state[:nh]), without a further RNN update.
    The collector returns CPU tensors and never writes files.
    """
    assert isinstance(net.lstm,nn.LSTMCell) and net.lstm.input_size==1287
    assert net.lstm.hidden_size==cfg['nh'] and net.n_actions==6
    assert next(net.parameters()).dtype==torch.float32
    assert plan['n_contexts']==1 and bool((plan['delay']==60).all())
    n=len(plan['heading']);assert n>0 and chunk_size>0 and n%chunk_size==0
    assert plan['state'].shape==(160,2,n,cfg['nh'])
    assert len(plan['heading_index'].unique())==n
    schedule=task.schedule(plan,'cpu');cue=int(schedule['glimpse'][0]);free=int(schedule['free'][0])
    assert bool((schedule['glimpse']==cue).all() and (schedule['free']==free).all()) and free-cue==60
    assert bool((schedule['forced_action'][cue:free]==0).all()) and bool((schedule['forced_action'][cue-4:cue]==2).all())
    policy_before=state_digest(net);rng_before=_rng();mode_before=net.training
    sources={str(Path(p).relative_to(ROOT)):sha(p) for p in (__file__,task.__file__,evaluator.__file__,ROOT/'code/env.py',ROOT/'code/common.py',ROOT/'code/codes.py',ROOT/'code/nets.py')}
    source=task.make_source(cfg,device);pgc=source.pop.is_pgc.detach().cpu().bool()
    net.eval();chunks=[];dark=None;goal=None
    outputs={key:[] for key in ('pre_cue_state','theta','requested_heading','first_free_heading','first_free_position','first_free_motors','probe_velocity','endpoint_heading_backderived')}
    grouped={key:{name:[] for name in CUES} for key in ('cue_inputs','starts','after_first_common','first_free_states','first_free_inputs','first_free_logits','first_free_probabilities')}
    movement=left_movement(cfg)
    try:
        for offset in range(0,n,chunk_size):
            chunk=slice_plan(plan,torch.arange(offset,offset+chunk_size));count=chunk_size*4
            captures={};call=[0];common_checks=[0]
            def before(module,args):
                nonlocal dark,goal
                t=call[0];x,state=args;assert x.shape==(count,1287)
                if t==cue:
                    captures['pre_cue']=torch.cat(state,1).detach().cpu().clone()
                    captures['cue_input']=x.detach().cpu().clone()
                if cue<t<=free:
                    current=x.detach().cpu();reference=current[0]
                    assert torch.equal(current,reference[None].expand_as(current)), 'Delay drive is phase/cue dependent'
                    if dark is None:dark=reference.clone()
                    assert torch.equal(current,dark[None].expand_as(current)), 'Delay input changes with time/chunk'
                    common_checks[0]+=1
                if t==free:captures['free_input']=x.detach().cpu().clone()
            def after(module,args,result):
                t=call[0]
                if t in (cue,cue+1,free):captures[f'state_{t-cue}']=torch.cat(result,1).detach().cpu().clone()
                call[0]+=1
            handles=[net.lstm.register_forward_pre_hook(before),net.lstm.register_forward_hook(after)]
            try:
                roll,rows,audit,first=task.collect(net,cfg,source,chunk,variants=VARIANTS,sigma=0.,device=device,retain_inputs=False)
            finally:
                for handle in handles:handle.remove()
            assert call[0]==160 and common_checks[0]==60
            assert audit['setup_pickups']==count and audit['setup_raw_reward']==10*count
            assert audit['setup_learning_reward']==audit['forced_hits']==0
            assert audit['clearance_min']>.01 and audit['wall_clearance_min']>.01 and audit['snap_max_m']<1e-5
            assert not bool(torch.stack(roll['raw_reward'])[:free].any())
            local_goal=roll['goal'].detach().cpu();assert torch.equal(local_goal,local_goal[:1].expand_as(local_goal))
            if goal is None:goal=local_goal[0].clone()
            assert torch.equal(local_goal,goal[None].expand_as(local_goal))
            assert torch.equal(captures['pre_cue'],captures['pre_cue'][:chunk_size].repeat(4,1))
            unchanged=torch.ones(1287,dtype=torch.bool);unchanged[256:768]=~pgc
            ordinary=captures['cue_input'][chunk_size:2*chunk_size]
            assert torch.equal(ordinary,ordinary[:1].expand_as(ordinary))
            assert torch.equal(ordinary[:,256:768],dark[None,256:768].expand(chunk_size,-1))
            for group,name in enumerate(CUES):
                sl=slice(group*chunk_size,(group+1)*chunk_size)
                x=captures['cue_input'][sl]
                assert torch.equal(x[:,unchanged],ordinary[:,unchanged]), 'Cue modified something outside current PGC slots'
                assert torch.equal(x[:,768:1280],goal[None].expand(chunk_size,-1))
                assert bool((x[:,1280:1286]==torch.tensor([0.,0.,1.,0.,0.,0.])).all()) and bool((x[:,1286]==0).all())
                for key,capture in (('cue_inputs','cue_input'),('starts','state_0'),('after_first_common','state_1'),('first_free_states','state_60'),('first_free_inputs','free_input')):
                    grouped[key][name].append(captures[capture][sl])
                grouped['first_free_logits'][name].append(torch.stack([first[i]['logits'] for i in range(sl.start,sl.stop)]))
                grouped['first_free_probabilities'][name].append(torch.stack([first[i]['probabilities'] for i in range(sl.start,sl.stop)]))
                for i in range(sl.start,sl.stop):
                    assert torch.equal(first[i]['state'],captures['state_60'][i])
                    assert torch.equal(first[i]['input'],captures['free_input'][i])
                    for field in ('position','heading','motors'):assert torch.equal(first[i][field],first[i%chunk_size][field])
            if not bool(pgc.any()):
                for capture in captures.values():assert torch.equal(capture,capture[:chunk_size].repeat(4,1)), 'Grid cue sham changed captured state/input'
            position=torch.stack([first[i]['position'] for i in range(chunk_size)])
            heading=torch.stack([first[i]['heading'] for i in range(chunk_size)])
            motors=torch.stack([first[i]['motors'] for i in range(chunk_size)])
            assert torch.equal(position,chunk['endpoint']) and bool((motors[:,:2]==0).all())
            expected_rotation=cfg['rot_max']*(1-math.exp(-60*(cfg['action_repeat']/60)/.25))
            assert float((motors[:,2]-expected_rotation).abs().max())<2e-6
            last=roll['position'][cue-1].detach().cpu()[:chunk_size]
            previous=roll['position'][cue-2].detach().cpu()[:chunk_size]
            velocity=(last-previous)/(cfg['action_repeat']/60)
            assert bool((velocity.norm(dim=1)>0).all())
            theta=torch.atan2(velocity[:,1],velocity[:,0])
            backderived=_wrap(heading.double()-movement['60'])
            requested=chunk['heading']
            assert float(_wrap(theta-requested).abs().max())<1e-5
            assert float(_wrap(backderived-theta.double()).abs().max())<2e-5
            for key,value in dict(pre_cue_state=captures['pre_cue'][:chunk_size],theta=theta,
                requested_heading=requested,first_free_heading=heading,first_free_position=position,
                first_free_motors=motors,probe_velocity=velocity,endpoint_heading_backderived=backderived).items():outputs[key].append(value)
            chunks.append(dict(first_phase_id=int(chunk['heading_index'][0]),phase_ids=chunk['heading_index'].tolist(),
                audit=audit,hook_calls=call[0],common_delay_input_checks=common_checks[0],
                inputs_and_noise_tensors=tensor_manifest(chunk)))
    finally:
        net.train(mode_before)
    assert state_digest(net)==policy_before and _same_rng(rng_before,_rng()), 'Read-only collection changed policy/global RNG'
    assert sources=={path:sha(ROOT/path) for path in sources}, 'Collection source changed during run'
    flat={key:torch.cat(value,0) for key,value in outputs.items()}
    grouped={key:{name:torch.cat(values,0) for name,values in mapping.items()} for key,mapping in grouped.items()}
    assert torch.equal(dark[768:1280],goal)
    assert torch.equal(dark[1280:1286],torch.tensor([1.,0.,0.,0.,0.,0.])) and float(dark[1286])==0
    for collection in (flat,*grouped.values()):
        for value in collection.values():assert bool(torch.isfinite(value).all())
    delta=plan['goal'][0]-plan['endpoint'][0];goal_bearing=torch.atan2(delta[1],delta[0])
    return dict(version='local_return_actual_task_hooks_v1',theta=flat.pop('theta'),pre_cue_state=flat.pop('pre_cue_state'),
        **grouped,dark_input=dark,goal_snapshot=goal,phase_indices=plan['heading_index'].clone(),movement=movement,
        nativephysics=dict(**flat,goal_center=plan['goal'][0].clone(),endpoint=plan['endpoint'][0].clone(),
            goal_bearing_at_endpoint=goal_bearing,egocentric_goal_bearing_first_free=_wrap(goal_bearing-flat['first_free_heading']),
            theta_definition='Measured last actual forward-probe displacement; body endpoint angle separately back-derived from firstFREE heading',
            movement_definition='Float32 native braked LEFT velocity recurrence; summed unwrapped angle at input times0..60'),
        audit=dict(condition=cfg['cond'],phases=n,chunk_size=chunk_size,state_noise_sigma=0.,
            cue_clock=cue,first_free_clock=free,common_updates_after_postcue_start=60,
            first_free_state_already_processed_first_free_input=True,fullclock_tape_free_slice=[free,free+30],
            all_phase_variant_delay_inputs_exact=True,all_goal_snapshots_exact=True,
            changes_only_current_predictive_slots=True,grid_cue_identity_exact=not bool(pgc.any()),
            no_policy_or_global_rng_mutation=True,policy_sha256=policy_before,sources=sources,
            config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            torch_version=str(torch.__version__),device=str(device),provenance=provenance,
            chunks=chunks,scope='Raw task-derived family only. No geometry, decoder, candidate gate or outcome selection. Full unchanged collect also completes remaining FREE actions; incidental counters retained.'))
