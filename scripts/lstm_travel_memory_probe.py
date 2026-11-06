"""Coherent actual-travel histories, then common-input direction-memory release.

Fifteen visible observations contain14 native forward physics updates. Starts
are chosen so all trajectories end at one actual context position. Translation
is explicitly stopped before the shared blind-turn input. The remembered angle
is last travel direction, not body heading during the subsequent turn.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from lstm_heading_memory_probe import make_env,sha,save,wrap,fit_readout
from plot_lstm_heading_memory import phase_diagnostics
from lstm_endogenous_curve_recovery import advance
import lstm_continued_heading_memory as continued

TIMES=continued.TIMES
NATIVE_ELIGIBILITY=continued.eligible_window
PHASES=768


def forward_distance(cfg,updates=14):
    dt=cfg['action_repeat']/60.;a=math.exp(-dt/.25)
    return dt*cfg['v_max']*sum(1-a**k for k in range(1,updates+1))


def path_clearance(position,goal,cfg):
    distance=forward_distance(cfg)
    wall=cfg['arena_half']-.08-float(np.max(np.abs(position)))-distance
    target=float(np.linalg.norm(np.asarray(position)-goal))-cfg['goal_radius']-distance
    return dict(forward_distance_m=distance,wall_clearance_m=wall,goal_clearance_m=target,
                eligible=bool(wall>.01 and target>.01))


@torch.inference_mode()
def prepare_travel_family(net,pop,cfg,window,context,device):
    assert cfg['pgc_shift_mode']=='space' and cfg['extra_input']=='none'
    target=np.asarray(window['history_pos'][0],dtype=np.float32)
    geometry=path_clearance(target,window['history_goal_center'][0],cfg)
    assert geometry['eligible'],geometry
    env=make_env(cfg,window,PHASES,context)
    angle=torch.arange(PHASES,dtype=torch.float32)*(2*np.pi/PHASES)
    env.hd[:]=angle
    unit=torch.stack((angle.cos(),angle.sin()),1)
    env.pos[:]=torch.as_tensor(target)-geometry['forward_distance_m']*unit
    env.spawn[:]=env.pos
    env.vf.zero_();env.vs.zero_();env.vr.zero_()
    initial_positions=env.pos.clone()
    initial=np.r_[window['history_h_before'][0],window['history_c_before'][0]]
    z=torch.tensor(initial,device=device).expand(PHASES,-1)
    state=tuple(v.contiguous() for v in z.chunk(2,1))
    goal=torch.tensor(window['history_x'][0,768:1280],device=device).expand(PHASES,-1)
    old_pa=window['history_x'][0,1280:1286];pa0=int(old_pa.argmax()) if old_pa.sum() else -1
    pr0=float(window['history_x'][0,-1]);actions=torch.full((PHASES,),2,dtype=torch.long)
    obs=env.observe();endpoint_error=0.
    for step in range(15):
        if step==14:
            endpoint_error=float((env.pos-torch.as_tensor(target)).norm(dim=1).max())
            assert endpoint_error<1e-5,endpoint_error
            env.pos[:]=torch.as_tensor(target)  # bounded float32 endpoint snapping
            obs=env.observe(obs['vel'])
        code=pop.activity(obs['pos'],obs['dir'],obs['vel']).to(device)
        embedding=net.embed(obs['rgb'].to(device))
        pa=torch.full((PHASES,),pa0 if step==0 else 2,dtype=torch.long,device=device)
        pr=torch.full((PHASES,),pr0 if step==0 else 0.,device=device)
        state=net.lstm(net._concat(embedding,code,goal,pa,pr,None),state)
        if step<14:
            obs,reward,done,info=env.step(actions)
            assert not bool(done.any() or info['hit'].any() or info['hit_wall'].any() or reward.any())
    full=torch.cat(state,1).cpu().double()
    theta=torch.atan2(env.mdir[:,1],env.mdir[:,0]).numpy().astype(np.float64)
    uniform_error=float(np.max(np.abs(wrap(np.roll(theta,-1)-theta)-2*np.pi/PHASES)))
    assert uniform_error<1e-4,uniform_error
    previous_velocity=dict(forward=env.vf.numpy().tolist(),sideways=env.vs.numpy().tolist(),rotation=env.vr.numpy().tolist())
    direction=env.mdir.clone()
    env.vf.zero_();env.vs.zero_();env.vr.zero_()
    turn=context%2
    env.vision='none'
    for _ in range(61):
        _,reward,done,info=env.step(torch.full((PHASES,),turn,dtype=torch.long))
        assert not bool(done.any() or info['hit'].any() or reward.any())
        assert torch.equal(env.pos,torch.as_tensor(target).expand_as(env.pos))
        assert torch.equal(env.mdir,direction)
    native=pop.activity(env.pos,env.mdir,torch.zeros_like(env.pos)).to(device)
    ordinary=pop.activity(env.pos,env.mdir,torch.zeros_like(env.pos),shift_scale=0.).to(device)
    assert torch.equal(ordinary,ordinary[:1].expand_as(ordinary))
    black=net.embed(torch.zeros(1,3,84,84,device=device))
    native_x=net._concat(black.expand(PHASES,-1),native,goal,
                        torch.full((PHASES,),turn,dtype=torch.long,device=device),torch.zeros(PHASES,device=device),None).cpu().double()
    ordinary_x=net._concat(black,ordinary[:1],goal[:1],torch.tensor([turn],device=device),torch.zeros(1,device=device),None).cpu().double()
    mask=torch.zeros(1287,dtype=torch.bool);mask[256+torch.as_tensor(pop.pgc_idx)]=True
    assert torch.equal(native_x[:,~mask],ordinary_x.expand_as(native_x)[:,~mask])
    metadata=dict(phases=PHASES,visible_observations=15,native_forward_physics_updates=14,
        endpoint=target.tolist(),path_clearance=geometry,endpoint_error_before_snap_m=endpoint_error,
        endpoint_snap='After14 actual forward updates, numerical endpoint discrepancies below1e-5m are snapped to the exact shared target before the final visible observation.',
        travel_phase_uniformity_max_error_rad=uniform_error,velocity_before_stop=previous_velocity,
        velocity_after_stop=[0.,0.,0.],stop_intervention='Explicitly set vf,vs,vr to0 after the last visible update; then execute a native turn action. This is a controlled stationary-state assay, not a claim that native turn commands instantly brake.',
        initial_position_range=[initial_positions.min(0).values.tolist(),initial_positions.max(0).values.tolist()],
        turn_action=turn,last_movement_direction_held_exactly=True,position_held_exactly=True,
        unchanged_input_outside_current_pgc=True,current_predictive_slots=int(pop.n_pgc),
        native_first_previous_action=pa0,native_first_previous_reward=pr0,
        target_variable='Last actual travel direction, fixed during the stationary turn; body heading continues changing')
    return full,theta,ordinary_x,native_x,metadata


@torch.no_grad()
def native_cue_trajectory(net,start,native_input,theta,scale):
    drive=torch.nn.functional.linear(native_input,net.lstm.weight_ih,net.lstm.bias_ih)
    state=start.clone();states={};rows={}
    select=slice(None,None,4);target=theta[select]
    decoder=fit_readout((start.numpy()/scale)[select],target)
    for t in range(61):
        if t in TIMES:
            states[t]=state.numpy().copy()
            rows[str(t)]=dict(fixed_travel_phase=phase_diagnostics((states[t]/scale)[select],target,decoder),
                actor_probability_family_rms=float((net.pi(state[:,:256]).softmax(1)-net.pi(state[:,:256]).softmax(1).mean(0)).square().mean().sqrt()))
        if t<60:state=advance(net.lstm,drive,state)
    return rows,states


def analyze(model,window,context,args):
    checkpoint=Path(model['checkpoint']);data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=data['config'] if 'config' in data else json.loads(Path(model['config']).read_text())
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
    full,theta,ordinary,native,physics=prepare_travel_family(net,build_population(cfg,'cpu'),cfg,window,context,args.device)
    net.cpu().double()
    scale=torch.stack((full[:,:256].square().mean().sqrt(),full[:,256:].square().mean().sqrt())).clamp_min(1e-8).repeat_interleave(256).numpy()
    result,states=continued.analyze_branch(net,full,ordinary,theta,{str(t):0. for t in range(61)},scale)
    for row in result['trajectory'].values():
        row['fixed_travel_phase']=row.pop('fixed_initial_phase');row.pop('fixed_current_heading')
    cue_rows,cue_states=native_cue_trajectory(net,full,native,theta,scale)
    folder=args.out/f'{cfg["cond"]}_s{cfg["seed"]}';folder.mkdir(parents=True,exist_ok=True)
    record=dict(condition=cfg['cond'],seed=cfg['seed'],context=context,cohort=args.cohort,
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),extra_frames=data.get('extra_frames',0),
                physics=physics,ordinary_release=result,native_cue_input_driven=cue_rows,
                source_sha256=sha(__file__),input_condition='Only current predictive slots are released to ordinary grid; actual saved goal and all other inputs preserved',
                interpretation='The released full-state family remembers last actual travel direction under one common input. Native cue trajectories receive phase-specific sensory input and are labelled input-driven; they are not an autonomous-ring test.')
    save(folder/f'context{context}.json',record)
    np.savez_compressed(folder/f'context{context}_states.npz',times=np.array(TIMES),
        states=np.stack([states[t] for t in TIMES]),theta=theta,ordinary_input=ordinary.numpy(),native_inputs=native.numpy(),
        native_current_code=native[:,256:768].numpy(),scale=scale,rawscale=np.ones(512),
        native_cue_states=np.stack([cue_states[t] for t in TIMES]),
        checkpoint_path=np.array(str(checkpoint)),checkpoint_sha256=np.array(sha(checkpoint)))
    print(json.dumps(dict(condition=cfg['cond'],seed=cfg['seed'],context=context,
                          candidate=result['passes_both_metric_candidate_screen'],
                          endpoint_error=physics['endpoint_error_before_snap_m'])),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',choices=['original','continued'],default='original');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='mps');p.add_argument('--out',type=Path,default=ROOT/'results/lstm_travel_memory/original')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='coherent_travel_memory_v1',cohort=args.cohort,
        original_models='Grid/PGC original seeds1/2; preselected nativecontext0',
        continued_models='Fixed final25.024M continued pair only; first2 physically eligible native scheduled visible intervals from each continued actor, seed98201',
        elicitation='15 visible observations,14 actual noiseless native forward physics updates. Starts lie behind a common actual context endpoint by the analytically predicted forward distance. Native initial recurrent context and saved goal fixed. Every path must avoid walls/goal, using sufficient all-angle clearance>.01m; context choice depends only on physical geometry.',
        endpoint='Float32 endpoint discrepancies must be<1e-5m before snapping to one exact target; record velocity, then explicitly stop translation/rotation before initiating native stationary turn. Last actual movement directionθ is held; body heading changes.',
        release='Replace only current169 PGC slots by their ordinary-grid values. All phases now receive exactly one shared black-image/current/goal/action/reward input. Preserve native phase-specific PGC inputs separately for causal pulses.',
        analysis='Full512 states, raw and h/cRMS geometry; released1/2s reference curves, recovery through4s; fixed192angle analysis subset of768real phase histories;384uniform-arc tests; targetedradial versus tangent recovery; frozen continued candidate criteria. Native-cue family is input-driven and never called autonomous.',
        pulse_interface='NPZ times/states[time,phase,512]/theta/ordinary_input[1,1287]/native_inputs[phase,1287]/native_current_code[phase,512]/scale[512]/rawscale[512]/checkpoint_path/hash',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_travel_memory_probe.py','scripts/lstm_continued_heading_memory.py','scripts/lstm_heading_curve_flow.py','scripts/lstm_heading_memory_probe.py','scripts/lstm_endogenous_curve_recovery.py','scripts/rl_visual_memory.py','code/env.py','code/nets.py','code/codes.py']})
    frozen=args.out/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol,'Frozen travel protocol/source changed; choose a new output directory'
    else:save(frozen,protocol)
    if args.freeze_only:return
    results=[]
    if args.cohort=='original':
        for condition in ('gc100','pgc33'):
            for seed in (1,2):
                stem=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
                window=np.load(ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz')
                results.append(analyze(dict(checkpoint=str(stem.with_suffix('.pt')),config=str(stem.with_suffix('.json'))),window,0,args))
    else:
        if args.checkpoint is None:p.error('--checkpoint is required for continued models')
        data=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=data['config']
        assert data.get('completed') and data['extra_frames']==25_024_000
        assert args.out.resolve()!=(ROOT/'results/lstm_travel_memory/original').resolve()
        net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
        def eligible(window,slot,radius):
            ok,why=NATIVE_ELIGIBILITY(window,slot,radius)
            if not ok:return ok,why
            g=path_clearance(window[0]['pos'][slot],window[0]['goal_center'][slot],cfg)
            return (True,'accepted') if g['eligible'] else (False,'all_angle_forward_path_clearance')
        with patch.object(continued,'eligible_window',eligible):
            contexts=continued.collect_contexts(net,cfg,args.checkpoint,args)
        for i,w in enumerate(contexts):results.append(analyze(dict(checkpoint=str(args.checkpoint)),w,i,args))
    save(args.out/'summary.json',dict(results=[dict(condition=r['condition'],seed=r['seed'],context=r['context'],
        candidate=r['ordinary_release']['passes_both_metric_candidate_screen']) for r in results]))


if __name__=='__main__':main()
