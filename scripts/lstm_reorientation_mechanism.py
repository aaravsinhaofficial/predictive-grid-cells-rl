"""Prospective task-derived geometry for the fixed blind-reorientation endpoint.

Fresh native event histories provide all preparatory actions and initial state.
Rotated actual-physics replays end at one location; the single endpoint cue is
followed by one common ordinary-code/LEFT delay. No angle/ring is trained here.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population,CodeSource,build_env
from code.env import ArenaEnv
from lstm_heading_memory_probe import sha,save,wrap
from lstm_blind_turn_necessity import native_distance
import rl_blind_reorientation as task
import lstm_continued_heading_memory as geometry

PHASES=768
COLLECTION_SEED=168201
VARIANTS=('native','ordinary','plus60','minus60')
PROTOCOL=ROOT/'docs/research/reorientation_mechanism_protocol.json'
SOURCES=('scripts/lstm_reorientation_mechanism.py','scripts/rl_blind_reorientation.py',
    'scripts/lstm_continued_heading_memory.py','scripts/lstm_heading_curve_flow.py',
    'scripts/lstm_endogenous_curve_recovery.py','scripts/lstm_endogenous_observable_flow.py',
    'scripts/lstm_heading_memory_probe.py','scripts/lstm_blind_turn_necessity.py',
    'code/common.py','code/env.py','code/nets.py','code/codes.py')


def motion_path(cfg,window):
    """Noiseless native motion, heading0; actual stored actions and brake phases."""
    env=ArenaEnv(1,half=cfg['arena_half'],episode_len=10000,action_repeat=cfg['action_repeat'],
        v_max=cfg['v_max'],rot_max=cfg['rot_max'],goal_radius=cfg['goal_radius'],
        vision='none',img_noise=0.,noise_v=0.,noise_r=0.,stagger=False,device='cpu')
    env.reset_all();env.pos.zero_();env.goal[:]=10.;env.hd.zero_();env.mdir[:]=torch.tensor([1.,0.])
    for key in ('vf','vs','vr'):getattr(env,key)[:]=float(window['initial_'+key])
    points=[env.pos[0].clone()];last=None
    for phase,action in zip(window['phases'],window['actions']):
        if phase!=last and phase in (task.TURN_A,task.PROBE):task.brake(env,torch.ones(1,dtype=torch.bool))
        _,reward,done,info=env.step(torch.tensor([int(action)]))
        assert not bool(info['hit_wall'].any() or done.any() or reward.any())
        points.append(env.pos[0].clone());last=int(phase)
    return torch.stack(points).numpy(),float(env.hd[0])


def physical_clearance(cfg,window):
    path,offset=motion_path(cfg,window);endpoint=np.asarray(window['endpoint'])
    radius=float(np.linalg.norm(path-path[-1],axis=1).max())
    wall=float(cfg['arena_half']-.08-np.abs(endpoint).max()-radius)
    goal=float(np.linalg.norm(endpoint-window['goal_center'])-cfg['goal_radius']-radius)
    return dict(eligible=wall>.01 and goal>.01,rotated_path_radius_m=radius,
                wall_margin_m=wall,goal_margin_m=goal,final_heading_offset_rad=offset)


def native_start(runtime,index):
    env=runtime['env'];result={}
    for key in ('pos','hd','mdir','vf','vs','vr','goal','land_pos','stripe_phase'):
        result['initial_'+key]=getattr(env,key)[index].detach().cpu().numpy().copy()
    result.update(h_before=runtime['state'][0][index].cpu().numpy().copy(),
        c_before=runtime['state'][1][index].cpu().numpy().copy(),
        goal_code=runtime['goal'].saved[index].cpu().numpy().copy(),
        previous_action=np.array(int(runtime['pa'][index])),previous_reward=np.array(float(runtime['pr'][index])),
        goal_center=env.goal[index].cpu().numpy().copy(),native_initial_step=np.array(int(env.t[index])))
    return result


@torch.inference_mode()
def collect_contexts(net,cfg,checkpoint,out):
    """Clean recurrent state; otherwise actual task policy/environment histories."""
    runtime=task.fresh_runtime(net,cfg,'cpu',seed=COLLECTION_SEED,stagger=False)
    env,clock,inputs=runtime['env'],runtime['clock'],runtime['inputs']
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu');buffers={};accepted=[];used=set();reasons={};coverage=[]
    for step in range(6000):
        if step%task.ROLLOUT==0:clock.begin_segment()
        clock.prepare(task.ROLLOUT-step%task.ROLLOUT,cfg['episode_len']-env.t,60)
        beginning=(clock.phase==task.VISIBLE)&(clock.age==0)
        for index in beginning.nonzero().flatten().tolist():
            buffers[index]=dict(start=native_start(runtime,index),phases=[],actions=[],x=[],h=[],c=[],wall=False)
        rgb,current,blind,glimpse=inputs.prepare(env,runtime['obs'],clock,source)
        x=net._concat(net.embed(rgb),current,runtime['goal'].saved,runtime['pa'],runtime['pr'],None)
        # Read-only native support observer; all encountered events, not selected
        # by outcome. The first2700 updates cover64 complete native episodes.
        observed=glimpse | ((clock.phase==task.FREE)&(clock.age==0))
        for index in observed.nonzero().flatten().tolist():
            bearing=torch.atan2(env.goal[index,1]-env.pos[index,1],env.goal[index,0]-env.pos[index,0])
            travel=torch.atan2(env.mdir[index,1],env.mdir[index,0])
            coverage.append(dict(step=step,slot=index,episode=int(clock.episode[index]),event=int(clock.event_count[index]),
                phase='endpoint' if bool(glimpse[index]) else 'free_start',body_heading=float(env.hd[index]),
                actual_travel_direction=float(travel),egocentric_goal_bearing=float(torch.atan2((bearing-env.hd[index]).sin(),(bearing-env.hd[index]).cos())),
                position=env.pos[index].tolist(),goal=env.goal[index].tolist()))
        for index in glimpse.nonzero().flatten().tolist():
            if index not in buffers or index in used or len(accepted)>=2:continue
            b=buffers[index];window=dict(b['start'],phases=np.array(b['phases']),actions=np.array(b['actions']),
                native_history_x=np.stack(b['x']),native_history_h_before=np.stack(b['h']),native_history_c_before=np.stack(b['c']),
                endpoint=env.pos[index].numpy().copy(),native_cue_input=x[index].numpy().copy(),
                native_pre_cue_h=runtime['state'][0][index].numpy().copy(),native_pre_cue_c=runtime['state'][1][index].numpy().copy(),
                source_slot=np.array(index),source_step=np.array(step),checkpoint_sha256=np.array(sha(checkpoint)))
            g=physical_clearance(cfg,window)
            reason='native_wall_contact' if b['wall'] else ('accepted' if g['eligible'] else 'all_rotation_physical_clearance')
            reasons[reason]=reasons.get(reason,0)+1
            if reason=='accepted':
                folder=out/'contexts';folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'context{len(accepted)}.npz',**window)
                accepted.append(window);used.add(index)
                print(json.dumps(dict(context=len(accepted)-1,slot=index,step=step,physical_clearance=g)),flush=True)
        for index,b in buffers.items():
            if int(clock.phase[index]) in (task.VISIBLE,task.TURN_A,task.TURN_B,task.PROBE):
                b['phases'].append(int(clock.phase[index]));b['x'].append(x[index].numpy().copy())
                b['h'].append(runtime['state'][0][index].numpy().copy());b['c'].append(runtime['state'][1][index].numpy().copy())
        h,c=net.lstm(x,runtime['state']);logits=net.pi(h)
        action=clock.action(task.sample_actions(logits,runtime['action_rng']))
        for index,b in buffers.items():
            if len(b['actions'])<len(b['phases']):b['actions'].append(int(action[index]))
        obs,raw,done,info=env.step(action);inputs.transition(clock,info)
        for index,b in buffers.items():b['wall']|=bool(info['hit_wall'][index])
        reward=raw.clamp(-cfg['reward_clip'],cfg['reward_clip']) if cfg['reward_clip']>0 else raw
        runtime['goal'].advance(info['hit'],done,source.goal(info['hit_pos'],info['hit_dir'],info['hit_vel']) if bool(info['hit'].any()) else None)
        clock.advance(info['hit'],done)
        for index in (info['hit']|done).nonzero().flatten().tolist():buffers.pop(index,None)
        runtime.update(obs=obs,state=task.reset_state((h,c),done),pa=torch.where(done,torch.full_like(action,-1),action),pr=reward*(~done))
        if len(accepted)==2 and step+1>=2700:
            save(out/'native_phase_coverage.json',dict(rows=coverage,observation_only=True,decisions=step+1,
                interpretation='Native support for body heading, actual movement direction and egocentric goal bearing; controlled360degree families are reported separately.'))
            save(out/'collection.json',dict(seed=COLLECTION_SEED,max_decisions=6000,decisions=step+1,contexts=2,reasons=reasons,
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),state_noise=False,native_environment_noise=True,
                selection='First two eligible endpoint events from distinct slots; no neural/performance selection',
                observer='Pure reads of native state; no new random draws, actions, state resets or environment mutations. At least2700updates, then extend only if fewer than2eligible contexts.'))
            return accepted
    save(out/'collection_incomplete.json',dict(contexts=len(accepted),reasons=reasons,max_decisions=6000))
    raise RuntimeError('Fewer than two prespecified physically eligible native contexts; retain incomplete collection, do not relax gates')


@torch.inference_mode()
def elicit(net,cfg,window,phases=PHASES):
    path,offset=motion_path(cfg,window);clearance=physical_clearance(cfg,window);assert clearance['eligible']
    pop=build_population(cfg,'cpu');source=CodeSource(cfg,pop,'cpu')
    env=build_env(dict(cfg,episode_len=10000,img_noise=0.),phases,176201,'cpu',stagger=False)
    env.reset_all();env.noise_v=env.noise_r=0.
    theta=torch.arange(phases,dtype=torch.float32)*(2*np.pi/phases);initial_heading=theta-offset
    endpoint=torch.tensor(window['endpoint']);d=torch.tensor(path[-1])
    rotation=torch.stack((initial_heading.cos()*d[0]-initial_heading.sin()*d[1],initial_heading.sin()*d[0]+initial_heading.cos()*d[1]),1)
    env.pos[:]=endpoint-rotation;env.spawn[:]=env.pos;env.hd[:]=initial_heading
    relative_dir=float(np.arctan2(window['initial_mdir'][1],window['initial_mdir'][0])-window['initial_hd'])
    env.mdir[:]=torch.stack(((initial_heading+relative_dir).cos(),(initial_heading+relative_dir).sin()),1)
    env.goal[:]=torch.tensor(window['goal_center']);env.land_pos[:]=torch.tensor(window['initial_land_pos']);env.stripe_phase[:]=torch.tensor(window['initial_stripe_phase'])
    for key in ('vf','vs','vr'):getattr(env,key)[:]=float(window['initial_'+key])
    state=tuple(torch.tensor(window[key]).expand(phases,-1).clone() for key in ('h_before','c_before'))
    goal=torch.tensor(window['goal_code']).expand(phases,-1)
    pa=torch.full((phases,),int(window['previous_action']),dtype=torch.long);pr=torch.full((phases,),float(window['previous_reward']))
    clock=task.ReorientationClock(phases,'cpu',176203);inputs=task.EventInputs(phases,512,'cpu')
    obs=env.observe();last=None;age=0;maxwall=0;maxhit=0
    for phase,action in zip(window['phases'],window['actions']):
        phase=int(phase);age=0 if phase!=last else age+1;clock.phase.fill_(phase);clock.age.fill_(age)
        rgb,code,_,_=inputs.prepare(env,obs,clock,source)
        x=net._concat(net.embed(rgb),code,goal,pa,pr,None);state=net.lstm(x,state)
        actions=torch.full((phases,),int(action),dtype=torch.long)
        obs,reward,done,info=env.step(actions);inputs.transition(clock,info)
        maxwall+=int(info['hit_wall'].sum());maxhit+=int(info['hit'].sum())
        assert not bool(done.any() or reward.any() or info['hit_wall'].any()),'A rotated physical replay was interrupted'
        pa=actions;pr=reward;last=phase
    endpoint_error=float((env.pos-endpoint).norm(dim=1).max());assert endpoint_error<1e-5,endpoint_error
    env.pos[:]=endpoint;obs=env.observe(obs['vel'])
    actual_theta=torch.atan2(obs['dir'][:,1],obs['dir'][:,0]).numpy().astype(np.float64)
    phase_error=float(np.abs(wrap(actual_theta-theta.numpy())).max());assert phase_error<1e-4
    body_error=float(np.abs(wrap(actual_theta-obs['hd'].numpy())).max());assert body_error<1e-4
    pre_cue=torch.cat(state,1).double();clock.phase.fill_(task.DELAY);clock.age.zero_()
    black,native,_,_=inputs.prepare(env,obs,clock,source)
    ordinary=inputs.held;assert torch.equal(ordinary,ordinary[:1].expand_as(ordinary))
    embedding=net.embed(black);cue_inputs={};starts={}
    for name,angle in [('native',0.),('ordinary',None),('plus60',np.pi/3),('minus60',-np.pi/3)]:
        code=ordinary if angle is None else source(obs['pos'],task.rotate(obs['dir'],torch.full((phases,),angle)),task.rotate(obs['vel'],torch.full((phases,),angle)))
        x=net._concat(embedding,code,goal,pa,pr,None)
        cue_inputs[name]=x.double();starts[name]=torch.cat(net.lstm(x,state),1).double()
    mask=torch.ones(1287,dtype=torch.bool);mask[256+torch.tensor(pop.pgc_idx)]=False
    assert all(torch.equal(x[:,mask],cue_inputs['native'][:,mask]) for x in cue_inputs.values())
    dark=net._concat(embedding[:1],ordinary[:1],goal[:1],torch.zeros(1,dtype=torch.long),torch.zeros(1),None).double()
    metadata=dict(phases=phases,physical_clearance=clearance,preparatory_updates=len(window['actions']),
        actions=window['actions'].tolist(),phase_names=[task.PHASE_NAMES[int(p)] for p in window['phases']],
        endpoint=endpoint.tolist(),endpoint_error_before_snap_m=endpoint_error,
        maximum_actual_travel_phase_error_rad=phase_error,maximum_travel_body_difference_rad=body_error,
        rotated_wall_hits=maxwall,rotated_goal_hits=maxhit,saved_goal_unchanged=True,
        current_cue_change_slots=int(pop.n_pgc),all_other_cue_inputs_unchanged=True,
        qualification='Native event action sequence replayed across controlled orientations; the actor did not independently choose the rotated actions. State before the visible event is its actual native state.')
    return pre_cue,starts,cue_inputs,dark,actual_theta,metadata


def analyze(net,data,cfg,window,checkpoint,index,out):
    before,starts,cues,dark,theta,metadata=elicit(net,cfg,window)
    net.double();scale=np.repeat([float(starts['native'][:,:256].square().mean().sqrt()),float(starts['native'][:,256:].square().mean().sqrt())],256).clip(1e-8)
    movement={str(t):native_distance(cfg,t,cfg['rot_max']) for t in range(61)}
    results={};states={}
    for name in VARIANTS:
        results[name],states[name]=geometry.analyze_branch(net,starts[name],dark,theta,movement,scale)
    save(out/f'context{index}.json',dict(condition=cfg['cond'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        context=index,physics=metadata,branches=results,source_sha256=sha(__file__),
        interpretation='Native/ordinary/wrong single cues share all subsequent autonomous input. Intrinsic attraction and acute phase repair are distinct outcomes; decoded angle alone is never a ring decision.'))
    fields=dict(times=np.array(geometry.TIMES),theta=theta,scale=scale,pre_cue_state=before.numpy(),
        ordinary_input=dark.numpy(),native_input=dark.numpy(),goal_center=window['goal_center'],endpoint=window['endpoint'],
        physical_turn_radians=np.array([movement[str(t)] for t in geometry.TIMES]),
        checkpoint_path=np.array(str(checkpoint)),checkpoint_sha256=np.array(sha(checkpoint)))
    for name in VARIANTS:
        fields[name+'_states']=np.stack([states[name][t] for t in geometry.TIMES]);fields[name+'_glimpse_inputs']=cues[name].numpy()
    np.savez_compressed(out/f'context{index}_states.npz',**fields)
    net.float()
    return {name:results[name]['passes_both_metric_candidate_screen'] for name in VARIANTS}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--out',type=Path,default=ROOT/'results/lstm_reorientation_mechanism')
    parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--smoke-original',choices=['gc100','pgc33']);args=parser.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    design=json.loads(PROTOCOL.read_text());manifest=dict(design_sha256=sha(PROTOCOL),source_sha256={p:sha(ROOT/p) for p in SOURCES})
    path=args.out/'implementation_protocol.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Frozen mechanistic source/design changed'
    else:save(path,manifest)
    if args.freeze_only:return
    if args.smoke_original:
        stem=ROOT/f'models/{args.smoke_original}/{args.smoke_original}_lstm_s1_1000M';data=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
        cfg=json.loads(stem.with_suffix('.json').read_text());cfg['rollout']=160;checkpoint=stem.with_suffix('.pt')
    else:
        assert args.checkpoint and args.checkpoint.name=='final.pt';checkpoint=args.checkpoint;data=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=data['config']
        assert data['completed'] and data['original_frames']==1_000_000_000 and data['extra_frames']==40_058_880
        assert data['warmup_updates']==200 and data['total_updates']==2156
        expected=json.loads((ROOT/'docs/research/rl_blind_reorientation_protocol.json').read_text());expected.pop('frozen_at')
        assert data['provenance']==expected and cfg['rollout']==160 and cfg['seed']==1
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy'])
    contexts=collect_contexts(net,cfg,checkpoint,args.out)
    if args.smoke_original:
        summaries=[]
        for w in contexts:
            before,starts,cues,dark,theta,meta=elicit(net,cfg,w,phases=96)
            summaries.append(dict(state_shape=list(before.shape),ordinary_shape=list(dark.shape),physics=meta))
        save(args.out/'smoke_shapes_only.json',dict(no_mechanistic_outcomes=True,summaries=summaries));return
    results=[analyze(net,data,cfg,w,checkpoint,i,args.out) for i,w in enumerate(contexts)]
    save(args.out/'summary.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),condition=cfg['cond'],contexts=results))


if __name__=='__main__':main()
