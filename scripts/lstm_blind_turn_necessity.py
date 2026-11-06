"""A simulator/input audit of whether blind-turn histories require memory.

No training, angle decoder, navigation controller, or ring fitting. Actual
six-action physics supplies analysis-only one-decision progress ceilings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from code.env import ACTIONS
from lstm_heading_memory_probe import make_env,save,sha,wrap

VISIBLE_OBSERVATIONS=15
PREFIX_UPDATES=14
PROBE_UPDATES=4
SUFFIX=(0,1,0)


def brake(env):
    env.vf.zero_();env.vs.zero_();env.vr.zero_()


def native_distance(cfg,updates,target):
    dt=cfg['action_repeat']/60.;decay=math.exp(-dt/.25)
    return dt*target*sum(1-decay**k for k in range(1,updates+1))


def identical_bytes(rows):
    a=np.asarray(rows)
    return a[0].tobytes()==a[1].tobytes()


def ceilings(progress):
    p=np.asarray(progress,dtype=np.float64)
    assert p.shape==(2,6)
    return dict(common_action_progress_m=float(p.mean(0).max()),
                phase_aware_progress_m=float(p.max(1).mean()),
                phase_aware_advantage_m=float(p.max(1).mean()-p.mean(0).max()),
                best_common_action=int(p.mean(0).argmax()),
                best_phase_actions=p.argmax(1).tolist())


def require_clean_step(env,action):
    obs,reward,done,info=env.step(torch.as_tensor(action,dtype=torch.long))
    assert not bool(done.any() or reward.any() or info['hit'].any() or info['hit_wall'].any())
    return obs


@torch.inference_mode()
def history_pair(net,pop,cfg,window,kind):
    assert kind in ('turn_only','probe_glimpse')
    env=make_env(cfg,window,2,0)
    target=torch.tensor(window['history_pos'][0],dtype=torch.float32)
    goal_center=env.goal[0].clone()
    bearing=torch.atan2(goal_center[1]-target[1],goal_center[0]-target[0])
    env.hd[:]=bearing;brake(env)
    # Visible motion is the same leftward rotation in both branches. The blind
    # prefix executes left versus right. The probe reaches one common endpoint.
    visible_turn=native_distance(cfg,VISIBLE_OBSERVATIONS-1,cfg['rot_max'])
    blind_turn=native_distance(cfg,PREFIX_UPDATES,cfg['rot_max'])
    predicted_heading=bearing+visible_turn+torch.tensor([blind_turn,-blind_turn])
    if kind=='probe_glimpse':
        distance=native_distance(cfg,PROBE_UPDATES,cfg['v_max'])
        direction=torch.stack((predicted_heading.cos(),predicted_heading.sin()),1)
        env.pos[:]=target-distance*direction
    else:distance=0.;env.pos[:]=target
    env.spawn[:]=env.pos
    starts=env.pos.clone()
    assert float((env.pos-env.goal).norm(dim=1).min())>cfg['goal_radius']+.06
    assert float(env.pos.abs().max())<cfg['arena_half']-.08-.06
    goal=torch.tensor(window['history_x'][0,768:1280]).expand(2,-1).clone()
    pa0=window['history_x'][0,1280:1286];pa0=int(pa0.argmax()) if pa0.sum() else -1
    pr0=float(window['history_x'][0,-1])
    xs=[];labels=[];headings=[];positions=[];executed=[];motors=[]
    black=net.embed(torch.zeros(2,3,84,84))
    def append(obs,code,pa,pr,label,visible):
        emb=net.embed(obs['rgb']) if visible else black
        x=net._concat(emb,code,goal,torch.as_tensor(pa,dtype=torch.long),torch.as_tensor(pr,dtype=torch.float32),None)
        assert x.shape==(2,1287)
        xs.append(x.clone());labels.append(label);headings.append(env.hd.clone());positions.append(env.pos.clone())
        executed.append(torch.as_tensor(pa,dtype=torch.long));motors.append(torch.stack((env.vf,env.vs,env.vr),1).clone())
    obs=env.observe()
    for step in range(VISIBLE_OBSERVATIONS):
        current=pop.activity(obs['pos'],obs['dir'],obs['vel'])
        append(obs,current,[pa0,pa0] if step==0 else [0,0],[pr0,pr0] if step==0 else [0.,0.],f'visible_{step}',True)
        if step<VISIBLE_OBSERVATIONS-1:obs=require_clean_step(env,[0,0])
    brake(env)
    snapshot=pop.activity(env.pos,env.mdir,torch.zeros_like(env.pos),shift_scale=0.)
    for step in range(PREFIX_UPDATES):
        obs=require_clean_step(env,[0,1]);append(obs,snapshot,[0,1],[0.,0.],f'blind_turn_{step}',False)
    brake(env)
    cue_index=None;native_cue=None;ordinary_cue=None;endpoint_error=0.
    if kind=='probe_glimpse':
        for step in range(PROBE_UPDATES):
            obs=require_clean_step(env,[2,2])
            if step==PROBE_UPDATES-1:
                endpoint_error=float((env.pos-target).norm(dim=1).max())
                assert endpoint_error<1e-5,endpoint_error
                env.pos[:]=target # audited float32 rounding correction only
                obs=env.observe(obs['vel'])
                native_cue=pop.activity(obs['pos'],obs['dir'],obs['vel'])
                ordinary_cue=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
                cue_index=len(xs)
                append(obs,native_cue,[2,2],[0.,0.],'native_glimpse',False)
            else:append(obs,snapshot,[2,2],[0.,0.],f'blind_probe_{step}',False)
        snapshot=ordinary_cue.clone()
    assert identical_bytes(snapshot.numpy()),'Final held ordinary snapshot differs across branches'
    assert torch.equal(env.pos,target.expand_as(env.pos))
    pre_suffix_heading=env.hd.clone();pre_suffix_mdir=env.mdir.clone()
    brake(env);suffix_start=len(xs)
    for index,action in enumerate(SUFFIX):
        obs=require_clean_step(env,[action,action]);append(obs,snapshot,[action,action],[0.,0.],f'common_suffix_{index}',False)
    history=torch.stack(xs);final_motor=torch.stack((env.vf,env.vs,env.vr),1)
    audits=dict(final_input_byte_identical=identical_bytes(history[-1].numpy()),
                last_three_inputs_byte_identical=all(identical_bytes(x.numpy()) for x in history[-3:]),
                final_position_byte_identical=identical_bytes(env.pos.numpy()),
                final_motor_byte_identical=identical_bytes(final_motor.numpy()),
                saved_goal_unchanged=bool(torch.equal(history[:,:,768:1280],goal[None].expand(len(xs),-1,-1))),
                final_previous_actions=history[-1,:,1280:1286].argmax(1).tolist(),
                final_previous_rewards=history[-1,:,-1].tolist(),
                final_motor_state=final_motor.tolist(),
                final_input_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in history[-1]],
                final_heading_degrees=torch.rad2deg(env.hd).tolist(),
                heading_separation_degrees=float(abs(np.rad2deg(wrap(float(env.hd[0]-env.hd[1]))))),
                endpoint_snap_max_m=endpoint_error,
                probe_body_minus_mdir_deg=np.rad2deg(wrap((pre_suffix_heading-torch.atan2(pre_suffix_mdir[:,1],pre_suffix_mdir[:,0])).numpy())).tolist() if cue_index is not None else None,
                final_observation_index=len(xs)-1,suffix_start_index=suffix_start,cue_index=cue_index,
                nominal_probe_displacement_m=distance,initial_positions=starts.tolist(),endpoint=target.tolist(),
                native_physics_noise='Disabled in this deterministic information/alias audit; no trained policy or physics parameter changed on disk')
    assert all(audits[k] for k in ('final_input_byte_identical','last_three_inputs_byte_identical','final_position_byte_identical','final_motor_byte_identical','saved_goal_unchanged'))
    ordinary_history=history.clone()
    if cue_index is not None:
        ordinary_history[cue_index,:,256:768]=ordinary_cue
        outside=torch.ones(1287,dtype=torch.bool);outside[256+torch.tensor(pop.pgc_idx)]=False
        assert torch.equal(ordinary_history[:,:,outside],history[:,:,outside])
        assert identical_bytes(ordinary_history[cue_index].numpy())
        audits['glimpse_changed_current_slots']=int((history[cue_index,:,256:768]!=ordinary_cue).any(0).sum())
        audits['glimpse_only_predictive_slots_changed']=True
    return history,ordinary_history,env,audits,dict(labels=np.array(labels),headings=torch.stack(headings).numpy(),
        positions=torch.stack(positions).numpy(),previous_actions=torch.stack(executed).numpy(),motor_states=torch.stack(motors).numpy())


@torch.inference_mode()
def actual_action_progress(cfg,window,terminal):
    env=make_env(cfg,window,12,0)
    for key in ('pos','hd','mdir','vf','vs','vr'):
        value=getattr(terminal,key)
        setattr(env,key,value.repeat_interleave(6,0).clone())
    env.spawn[:]=env.pos;before=env.pos.clone()
    unit=(env.goal-before);unit=unit/unit.norm(dim=1,keepdim=True)
    require_clean_step(env,list(range(6))*2)
    return ((env.pos-before)*unit).sum(1).reshape(2,6).numpy()


@torch.inference_mode()
def replay_actor(net,history,initial,reset_before=None):
    state=tuple(x.clone() for x in initial)
    for t,x in enumerate(history):
        if t==reset_before:state=net.zero_state(2,'cpu')
        state=net.lstm(x,state)
    return net.pi(state[0]).softmax(1).numpy(),torch.cat(state,1).numpy()


def run(condition,seed,args):
    stem=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text());data=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert data['frames']==1_000_000_000
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy'])
    pop=build_population(cfg,'cpu')
    source=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'
    window=np.load(source)
    initial=tuple(torch.tensor(window[key][0]).expand(2,-1).clone() for key in ('history_h_before','history_c_before'))
    records=[]
    for kind in ('turn_only','probe_glimpse'):
        native,ordinary,env,audit,trace=history_pair(net,pop,cfg,window,kind)
        progress=actual_action_progress(cfg,window,env);bound=ceilings(progress)
        assert bound['phase_aware_advantage_m']>1e-4
        assert bound['best_phase_actions'][0]!=bound['best_phase_actions'][1]
        branches={};saved_states={}
        options=[('retained_native',native,None),('reset_at_final_decision',native,len(native)-1)]
        if audit['cue_index'] is not None:
            options.extend([('retained_ordinary',ordinary,None),('reset_before_native_glimpse',native,audit['cue_index']),
                ('reset_before_ordinary_glimpse',ordinary,audit['cue_index']),('erase_after_native_glimpse',native,audit['suffix_start_index'])])
        for name,history,reset in options:
            probability,state=replay_actor(net,history,initial,reset)
            expected=(probability*progress).sum(1)
            branches[name]=dict(expected_progress_m=expected.tolist(),mean_expected_progress_m=float(expected.mean()),
                probabilities=probability.tolist(),probability_rows_byte_identical=identical_bytes(probability),
                difference_from_common_action_ceiling_m=float(expected.mean()-bound['common_action_progress_m']),
                reset_before_observation=reset)
            saved_states[name]=state
            if name in ('reset_at_final_decision','reset_before_ordinary_glimpse','erase_after_native_glimpse'):
                assert identical_bytes(probability)
                assert expected.mean()<=bound['common_action_progress_m']+1e-8
        record=dict(condition=condition,seed=seed,kind=kind,checkpoint=str(stem.with_suffix('.pt')),
            checkpoint_sha256=sha(stem.with_suffix('.pt')),context_sha256=sha(source),source_sha256=sha(__file__),
            audits=audit,action_names=list(ACTIONS),progress_by_branch_action_m=progress.tolist(),ceilings=bound,
            original_actor_branches=branches,
            interpretation='The ceiling gap is an information-requirement audit, not a trained performance claim. Oracle actions are never supplied to the network, rewards, or controller. Original checkpoint responses are descriptive transfer tests; these models were not trained on this proposed task.')
        folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        save(folder/f'{kind}.json',record)
        np.savez_compressed(folder/f'{kind}.npz',native_history=native.numpy(),ordinary_history=ordinary.numpy(),
            initial_h=initial[0].numpy(),initial_c=initial[1].numpy(),action_progress=progress,
            **trace,**{'state_'+k:v for k,v in saved_states.items()})
        records.append(record)
        print(json.dumps(dict(model=f'{condition}_s{seed}',kind=kind,heading_gap=audit['heading_separation_degrees'],
            common=bound['common_action_progress_m'],phase_aware=bound['phase_aware_progress_m'],identical_inputs=True)),flush=True)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_blind_turn_necessity')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='blind_turn_information_necessity_v1',purpose='Design diagnostic only; no training, cloud launch, reward change, learned decoder, ring fitting or action teacher',
        models='Original Grid/PGC seeds1–2; context0; same independently saved native recurrent context and actual goal/room per model',
        timeline=dict(visible_observations=VISIBLE_OBSERVATIONS,visible_physics_updates=14,blind_prefix_updates=PREFIX_UPDATES,
            blind_prefix_pair_actions=[0,1],optional_probe_updates=PROBE_UPDATES,probe_action=2,common_suffix=list(SUFFIX)),
        control='Pure-turn history variant shares initial position. Probe/glimpse variant chooses actual starts to reach one common endpoint; numerical snap must be<1e-5m. Explicit vf/vs/vr brakes before each forced segment; common suffix starts at identical zero motor state. All final1287inputs and last3inputs must match byte-for-byte, with identical finalposition,motorstate,goal,pa,pr.',
        ceiling='Expected signed goalward displacement over one actual native update for all6actions; maximum mean across paired headings is best common-action ceiling, mean per-heading maximum is phase-aware ceiling. No ceiling action is executed by policy training or supplied as a label.',
        success='Input/motor/goal identities plus distinct best phase actions and phase-aware advantage>1e-4m. Original actor performance is reported separately and never required to pass the information audit.',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_blind_turn_necessity.py','scripts/lstm_heading_memory_probe.py','code/common.py','code/env.py','code/codes.py','code/nets.py']},
        input_sha256={f'{c}_s{s}':dict(checkpoint=sha(ROOT/f'models/{c}/{c}_lstm_s{s}_1000M.pt'),context=sha(ROOT/f'results/original_goal_bearing_manifold/pilot/{c}_s{s}/context0_states.npz')) for c in ('gc100','pgc33') for s in (1,2)})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen diagnostic changed'
    else:save(path,protocol)
    if args.freeze_only:return
    records=[]
    for c in ('gc100','pgc33'):
        for s in (1,2):records.extend(run(c,s,args))
    save(args.out/'summary.json',dict(records=records,all_information_audits_passed=True))


if __name__=='__main__':main()
