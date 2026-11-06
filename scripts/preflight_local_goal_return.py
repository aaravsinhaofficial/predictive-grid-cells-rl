"""Bounded, read-only local-goal physical preflight. No training or task freeze."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_env,build_policy,build_population,CodeSource
from lstm_endogenous_curve_recovery import sha,save
from lstm_reorientation_mechanism import motion_path,physical_clearance
from lstm_reorientation_causal_navigation import navigation_draws
from rl_visual_memory import NativeGoal
import rl_blind_reorientation as task

SEED=271801
NCONTEXTS=8
NHEADINGS=16
OUT=ROOT/'results/rl_blind_reorientation_review/local_goal_preflight'


def rows_identical(x):
    return bool(torch.equal(x,x[:1].expand_as(x)))


def make_contexts(cfg):
    rng=np.random.default_rng(SEED);contexts=[];attempts=[]
    for attempt in range(100):
        env=build_env(dict(cfg,episode_len=10000,img_noise=0.),1,SEED+attempt,'cpu',stagger=False)
        env.reset_all();env.noise_v=env.noise_r=0.
        goal=env.goal[0].clone();pickup_angle=float(rng.uniform(-math.pi,math.pi))
        direction=torch.tensor([math.cos(pickup_angle),math.sin(pickup_angle)])
        env.pos[0]=goal-.02*direction;env.hd.fill_(pickup_angle);env.mdir[0]=direction
        env.vf.zero_();env.vs.zero_();env.vr.zero_()
        _,reward,done,info=env.step(torch.tensor([2]))
        assert bool(info['hit'][0]) and float(reward[0])==10. and not bool(done[0])
        radius=float(rng.uniform(.45,.55));bearing=float(rng.uniform(-math.pi,math.pi))
        endpoint=goal.numpy()+radius*np.array([math.cos(bearing),math.sin(bearing)],dtype=np.float32)
        durations=rng.integers(5,11,size=2);turns=rng.integers(0,2,size=2);delay=int(rng.integers(30,61))
        phases=[task.VISIBLE]*15+[task.TURN_A]*int(durations[0])+[task.TURN_B]*int(durations[1])+[task.PROBE]*4
        actions=[0]*15+[int(turns[0])]*int(durations[0])+[int(turns[1])]*int(durations[1])+[2]*4
        context=dict(attempt=attempt,endpoint=endpoint,goal_center=goal.numpy(),
            initial_land_pos=env.land_pos[0].numpy().copy(),initial_stripe_phase=env.stripe_phase[0].numpy().copy(),
            initial_vf=0.,initial_vs=0.,initial_vr=0.,phases=np.array(phases),actions=np.array(actions),
            pickup_position=info['hit_pos'][0].numpy().copy(),pickup_direction=info['hit_dir'][0].numpy().copy(),
            pickup_velocity=info['hit_vel'][0].numpy().copy(),pickup_reward=float(reward[0]),
            radius_m=radius,goal_bearing_rad=bearing,heading_offset_rad=float(rng.uniform(-math.pi,math.pi)),delay=delay)
        clearance=physical_clearance(cfg,context)
        attempts.append(dict(attempt=attempt,accepted=clearance['eligible'],clearance=clearance,
            endpoint=endpoint.tolist(),goal_center=goal.tolist(),radius_m=radius,delay=delay))
        if clearance['eligible']:contexts.append(context)
        if len(contexts)==NCONTEXTS:break
    assert len(contexts)==NCONTEXTS
    return contexts,attempts


@torch.inference_mode()
def prepare(net,cfg,context):
    n=NHEADINGS;env=build_env(dict(cfg,episode_len=10000,img_noise=0.),n,SEED+1000,'cpu',stagger=False)
    env.reset_all();env.noise_v=env.noise_r=0.
    path,offset=motion_path(cfg,context)
    theta=torch.arange(n)*2*math.pi/n+context['heading_offset_rad'];initial=theta-offset
    d=torch.tensor(path[-1]);rot=torch.stack((initial.cos()*d[0]-initial.sin()*d[1],initial.sin()*d[0]+initial.cos()*d[1]),1)
    endpoint=torch.tensor(context['endpoint']);env.pos[:]=endpoint-rot;env.spawn[:]=env.pos
    env.goal[:]=torch.tensor(context['goal_center']);env.hd[:]=initial
    env.mdir[:]=torch.stack((initial.cos(),initial.sin()),1)
    env.land_pos[:]=torch.tensor(context['initial_land_pos']);env.stripe_phase[:]=torch.tensor(context['initial_stripe_phase'])
    env.vf.zero_();env.vs.zero_();env.vr.zero_()
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    pickup_code=source.goal(*[torch.tensor(context[k])[None] for k in ('pickup_position','pickup_direction','pickup_velocity')])
    saved=NativeGoal(1,512,'cpu');saved.advance(torch.tensor([True]),torch.tensor([False]),pickup_code)
    goal=saved.saved.expand(n,-1).clone();assert rows_identical(goal)
    states={'native':net.zero_state(n,'cpu'),'erase_after_glimpse':net.zero_state(n,'cpu')}
    pa=torch.full((n,),-1,dtype=torch.long);pr=torch.zeros(n);clock=task.ReorientationClock(n,'cpu',SEED+1002)
    inputs=task.EventInputs(n,512,'cpu');obs=env.observe();phase_before=None;age=0
    trace=[];snap=0.;failures=[];cue_inputs=None;first_free=None
    sequence=list(zip(context['phases'],context['actions']))+[(task.DELAY,0)]*context['delay']
    for step,(phase,action) in enumerate(sequence):
        phase=int(phase);age=age+1 if phase==phase_before else 0
        if phase==task.DELAY and age==0:
            snap=float((env.pos-endpoint).norm(dim=1).max())
            if snap>1e-5:failures.append(dict(step=step,reason='endpoint_rounding_exceeds_bound',error_m=snap));break
            env.pos[:]=endpoint;obs=env.observe(obs['vel'])
        clock.phase.fill_(phase);clock.age.fill_(age)
        rgb,code,_,glimpse=inputs.prepare(env,obs,clock,source)
        x=net._concat(net.embed(rgb),code,goal,pa,pr,None)
        for name,z in states.items():
            states[name]=net.lstm(x,z)
            if name=='erase_after_glimpse' and bool(glimpse.any()):states[name]=net.zero_state(n,'cpu')
        if bool(glimpse.any()):cue_inputs=x.clone()
        trace.append(dict(step=step,phase=task.PHASE_NAMES[phase],action=int(action)))
        obs,reward,done,info=env.step(torch.full((n,),int(action),dtype=torch.long))
        inputs.transition(clock,info)
        if bool(reward.any() or done.any() or info['hit_wall'].any()):
            failures.append(dict(step=step,reason='forced_phase_event',reward=reward.tolist(),
                goal_hit=info['hit'].tolist(),done=done.tolist(),wall=info['hit_wall'].tolist()));break
        pa.fill_(int(action));pr=reward;phase_before=phase
    if failures:return dict(failures=failures,trace=trace),None
    clock.phase.fill_(task.FREE);clock.age.zero_()
    rgb,code,_,_=inputs.prepare(env,obs,clock,source)
    first_free=net._concat(net.embed(rgb),code,goal,pa,pr,None)
    for name,z in states.items():states[name]=net.lstm(first_free,z)
    motors=torch.stack((env.vf,env.vs,env.vr),1)
    assert rows_identical(first_free) and rows_identical(env.pos) and rows_identical(motors)
    assert rows_identical(states['erase_after_glimpse'][0]) and rows_identical(states['erase_after_glimpse'][1])
    assert len(sequence)+30<=129<=160
    audit=dict(failures=[],context_steps=len(sequence)+30,within160=True,heading_variants=n,
        first_free_input_byte_identical=True,final_position_byte_identical=True,final_motor_byte_identical=True,
        saved_pickup_code_byte_identical=True,goal_snapshot_actual_hit_reward=context['pickup_reward'],
        endpoint_snap_max_m=snap,terminal_motors=motors[0].tolist(),
        terminal_heading_rad=env.hd.tolist(),travel_phase_rad=torch.atan2(env.mdir[:,1],env.mdir[:,0]).tolist(),
        forced_phase_rewards=0,forced_wall_hits=0,previous_action=0,previous_reward=0,
        actual_probe_cue_uses_native_direction=True,trace=trace)
    arrays=dict(first_free_input=first_free.numpy(),goal_snapshot=goal.numpy(),cue_inputs=cue_inputs.numpy(),
        terminal_pos=env.pos.numpy(),terminal_hd=env.hd.numpy(),terminal_motors=motors.numpy(),
        native_state=torch.cat(states['native'],1).numpy(),erased_state=torch.cat(states['erase_after_glimpse'],1).numpy())
    return audit,(env,states,first_free,arrays)


def predicted_next_distance(env):
    """Expected-noise one-step native mechanics for analysis-only action choice."""
    target=env._targets; k=1-math.exp(-env.dt/env.tau)
    vf=env.vf[:,None]+k*(target[None,:,0]*env.v_max-env.vf[:,None])
    vs=env.vs[:,None]+k*(target[None,:,1]*env.v_max-env.vs[:,None])
    vr=env.vr[:,None]+k*(target[None,:,2]*env.rot_max-env.vr[:,None])
    hd=env.hd[:,None]+vr*env.dt
    dx=torch.stack((vf*hd.cos()-vs*hd.sin(),vf*hd.sin()+vs*hd.cos()),2)*env.dt
    pos=(env.pos[:,None]+dx).clamp(-env.half+env.wall_margin,env.half-env.wall_margin)
    return (pos-env.goal[:,None]).norm(dim=2)


@torch.inference_mode()
def navigate(terminal,motor,uniform,mode,net=None,state=None,dark=None,noise=True):
    env=copy.deepcopy(terminal);env.vision='none';env.img=1;env.img_noise=0.
    env.noise_v=.03 if noise else 0.;env.noise_r=.15 if noise else 0.
    n=env.n;index=0
    def draw(*shape):
        nonlocal index
        assert shape==(n,);value=motor[index//3,index%3];index+=1;return value
    env._randn=draw
    active=torch.ones(n,dtype=torch.bool);hit_step=torch.full((n,),-1,dtype=torch.long)
    distance0=(env.pos-env.goal).norm(dim=1);distance=distance0.clone();path_length=torch.zeros(n)
    paths=[env.pos.clone()];actions=[];entropies=[]
    if state is not None:state=tuple(z.clone() for z in state)
    for t in range(30):
        if mode=='planner':action=predicted_next_distance(env).argmin(1)
        elif mode.startswith('constant'):action=torch.full((n,),int(mode[-1]),dtype=torch.long)
        else:
            probabilities=net.pi(state[0]).softmax(1)
            action=(uniform[t,:,None]>probabilities.cumsum(1)).sum(1).clamp_max(5)
            entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(1)
            entropies.append(torch.where(active,entropy,torch.full_like(entropy,float('nan'))))
        before=env.pos.clone();actions.append(torch.where(active,action,torch.full_like(action,-1)))
        _,reward,done,info=env.step(action);assert not bool(done.any())
        endpoint=info['hit_pos'];distance=torch.where(active,(endpoint-env.goal).norm(dim=1),distance)
        path_length+=active*(endpoint-before).norm(dim=1);paths.append(torch.where(active[:,None],endpoint,paths[-1]))
        hit=active&info['hit'];hit_step[hit]=t+1;active&=~hit
        if state is not None and t<29:
            x=dark.clone();x[:,1280:1286]=torch.nn.functional.one_hot(action,6);x[:,-1]=0
            new=net.lstm(x,state);state=tuple(torch.where(active[:,None],a,b) for a,b in zip(new,state))
    assert index==90
    actions=torch.stack(actions);entropy_values=torch.stack(entropies).numpy() if entropies else np.empty((0,n))
    summary=dict(mode=mode,noise=noise,windows=n,hits=int((hit_step>0).sum()),
        hit_steps=hit_step.tolist(),mean_path_m=float(path_length.mean()),mean_progress_m=float((distance0-distance).mean()),
        action_counts=[int((actions==a).sum()) for a in range(6)],
        mean_free_entropy=float(np.nanmean(entropy_values)) if entropies else None,
        first_free_entropy=float(np.nanmean(entropy_values[0])) if entropies else None)
    arrays=dict(actions=actions.numpy(),path=torch.stack(paths).numpy(),path_length=path_length.numpy(),
        progress=(distance0-distance).numpy(),hit_step=hit_step.numpy(),entropy=entropy_values)
    return summary,arrays


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    cfg=json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    contexts,attempts=make_contexts(cfg);save(args.out/'initialization_attempts.json',attempts)
    rows=[]
    for condition in ('gc100','pgc33'):
        stem=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M';cfg=json.loads(stem.with_suffix('.json').read_text())
        checkpoint=stem.with_suffix('.pt');digest=sha(checkpoint);data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy'])
        for i,context in enumerate(contexts):
            target=args.out/condition/f'context{i}';target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/'setup.npz',**context)
            audit,prepared=prepare(net,cfg,context)
            row=dict(condition=condition,context=i,checkpoint_sha256=digest,audit=audit)
            if prepared is not None:
                terminal,states,dark,arrays=prepared
                motor,uniform=navigation_draws(NHEADINGS,seed=SEED+2000+i)
                # Same private draws across branches; no change to native step equations.
                np.savez_compressed(target/'prepared.npz',**arrays,motor=motor.numpy(),uniform=uniform.numpy())
                result={};raw={}
                for name in ('native','erase_after_glimpse'):
                    result[name],raw[name]=navigate(terminal,motor,uniform,name,net,states[name],dark)
                if condition=='gc100':
                    for noise in (False,True):
                        for mode in ('planner',*(f'constant{a}' for a in range(6))):
                            key=mode+('_native_noise' if noise else '_noiseless')
                            result[key],raw[key]=navigate(terminal,motor,uniform,mode,noise=noise)
                    progress=(terminal.pos-terminal.goal).norm(dim=1)[:,None]-predicted_next_distance(terminal)
                    row['instantaneous_information_bound']=dict(best_common_action_progress_m=float(progress.mean(0).max()),
                        heading_aware_progress_m=float(progress.max(1).values.mean()),
                        qualification='Rigorous only for one decision in the noiseless native physics, not a30-step memoryless success ceiling')
                row['results']=result
                np.savez_compressed(target/'navigation.npz',**{f'{name}__{key}':value for name,a in raw.items() for key,value in a.items()})
            rows.append(row);save(target/'result.json',row)
            print(json.dumps(dict(condition=condition,context=i,failed=bool(audit['failures']),
                hits={name:r['hits'] for name,r in row.get('results',{}).items()})),flush=True)
        assert sha(checkpoint)==digest
    save(args.out/'summary.json',dict(version='bounded_local_goal_return_physical_preflight_v1',seed=SEED,
        contexts=NCONTEXTS,headings=NHEADINGS,rows=rows,source_sha256=sha(__file__),
        scope='Design diagnostic only; no training, no finaltask settings freeze. Original1B actors are transfer diagnostics, not newlytrainedresults. One genuine nativepickup snapshot is fixed across eachheadingfamily. Preparatory motor/image noise disabled for exactaliases; FREEphysics tested with bothzero andnativefixednoise forplanners/baselines; incomingactorsnativeFREEnoise. Sixconstantactionpolicies are baselines, not exhaustive memorylessceiling. Planner is feasiblegreedyone-step with truebodyheading, nofuture-noise knowledge, not optimaloracle.',
        initialization='Actualsetupgoalpickup generates+10 andNativeGoalcode; then explicitnearbytrialinitialposition/bodyheading,zeromotors,h/czero,pa−1/pr0. Setupreward is recordedoutside scoredtrial. Existingnativerespawn initializesmdirwithbodyheading; this is retained. Heldgoalcodefixedacrossphases. Firstfreeactionusesalreadyprocessedterminalinput. No freeonsetbrake. Everyphysicalrejection andforcedfailure retained.'))


if __name__=='__main__':main()
