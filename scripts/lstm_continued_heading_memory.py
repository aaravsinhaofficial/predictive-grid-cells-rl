"""Native continued-policy contexts and heading-memory tests within trained4s gaps.

The fixed final checkpoint pair is primary. Stage checkpoints are accepted only
with --smoke-only, which checks collection/shapes/input identity without reporting
mechanistic outcome metrics. No original-network recurrent states are reused.
"""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population,CodeSource
from lstm_code_swap import MatchedArena
from rl_visual_memory import VisualClock,NativeGoal,masked_rgb,reset_state
from lstm_heading_memory_probe import make_env,fit_readout,sha,save,wrap
from plot_lstm_heading_memory import phase_diagnostics
from lstm_heading_current_release import ordinary_current_input
import lstm_heading_curve_flow as flow
from lstm_endogenous_curve_recovery import advance

DESIGN=ROOT/'docs/research/continued_heading_memory_protocol.json'
TIMES=(0,1,5,15,30,45,60)
ANCHORS=(15,30)
RECOVERY=(1,15,30)
SOURCE_FILES=('scripts/lstm_continued_heading_memory.py','scripts/rl_visual_memory.py',
              'scripts/lstm_heading_memory_probe.py','scripts/lstm_heading_curve_flow.py',
              'scripts/lstm_heading_current_release.py','scripts/plot_lstm_heading_memory.py',
              'scripts/lstm_endogenous_curve_recovery.py','scripts/lstm_code_swap.py',
              'code/env.py','code/nets.py','code/codes.py','code/common.py')


def eligible_window(window,slot,radius):
    if len(window)!=15:return False,'short'
    if not all(bool(r['known'][slot]) and not bool(r['blind'][slot]) for r in window):return False,'visibility'
    if [int(r['remaining'][slot]) for r in window]!=list(range(15,0,-1)):return False,'schedule'
    if any(r['hit'][slot] or r['done'][slot] for r in window):return False,'hit_or_reset'
    goal=window[0]['x'][slot,768:1280]
    if not all(np.array_equal(r['x'][slot,768:1280],goal) for r in window):return False,'goal_changed'
    if np.linalg.norm(window[0]['pos'][slot]-window[0]['goal_center'][slot])<=radius+.01:return False,'near_goal'
    return True,'accepted'


@torch.inference_mode()
def collect_contexts(net,cfg,checkpoint,args):
    seed=98201;n=32
    env=MatchedArena(cfg,n,1,seed,'cpu');obs=env.reset_all()
    pop=build_population(cfg,'cpu');source=CodeSource(cfg,pop,'cpu')
    clock=VisualClock(n,'cpu',seed+2);goal=NativeGoal(n,cfg['n_cells'],'cpu')
    state=net.zero_state(n,args.device)
    pa=torch.full((n,),-1,dtype=torch.long);pr=torch.zeros(n)
    generator=torch.Generator().manual_seed(seed+1)
    records=deque(maxlen=15);contexts=[];used=set();rejections={};started=time.monotonic();last=started
    for step in range(6000):
        blind=clock.mask().clone();known=clock.known.clone();remaining=clock.remaining.clone()
        rgb=masked_rgb(obs['rgb'],clock)
        current=source(obs['pos'],obs['dir'],obs['vel'])
        embedding=net.embed(rgb.to(args.device))
        x=net._concat(embedding,current.to(args.device),goal.saved.to(args.device),pa.to(args.device),pr.to(args.device),None)
        before=state;state=net.lstm(x,state)
        probabilities=net.pi(state[0]).softmax(1).cpu()
        uniform=torch.rand(n,generator=generator)
        action=(uniform[:,None]>probabilities.cumsum(1)).sum(1).clamp_max(5)
        record={k:v.detach().cpu().numpy().copy() for k,v in dict(
            x=x,h_before=before[0],c_before=before[1],h_after=state[0],c_after=state[1],
            pos=obs['pos'],hd=obs['hd'],direction=obs['dir'],velocity=obs['vel'],
            goal_center=env.goal,landmark_pos=env.land_pos,stripe_phase=env.stripe_phase,
            known=known,blind=blind,remaining=remaining,action=action,episode=env.ep_idx).items()}
        obs,raw,done,info=env.step(action)
        record['hit']=info['hit'].numpy().copy();record['done']=done.numpy().copy()
        record['step']=np.full(n,step,dtype=np.int64);records.append(record)
        hit_code=source.goal(info['hit_pos'],info['hit_dir'],info['hit_vel']) if bool(info['hit'].any()) else None
        goal.advance(info['hit'],done,hit_code)
        entering=clock.advance(info['hit'],done,60)
        for slot in entering.nonzero().flatten().tolist():
            if slot in used:continue
            valid,reason=eligible_window(list(records),slot,cfg['goal_radius'])
            if not valid:rejections[reason]=rejections.get(reason,0)+1;continue
            window={f'history_{k}':np.stack([r[k][slot] for r in records]) for k in records[0]}
            # Original family helper expects one fixed room layout per context.
            window['history_landmark_pos']=window['history_landmark_pos'][0]
            window['history_stripe_phase']=window['history_stripe_phase'][0]
            window['history_slot']=np.array(slot)
            window['checkpoint_sha256']=np.array(sha(checkpoint))
            window['collector_step']=np.array(step)
            contexts.append(window);used.add(slot)
            print(f'{cfg["cond"]}: native continued context{len(contexts)-1}, slot{slot}, scheduled blackout at decision{step+1}',flush=True)
            if len(contexts)==2:break
        if len(contexts)==2:break
        state=reset_state(state,done.to(args.device))
        reward=raw.clamp(-cfg['reward_clip'],cfg['reward_clip']) if cfg['reward_clip']>0 else raw
        pa=torch.where(done,torch.full_like(action,-1),action);pr=reward*(~done)
        if time.monotonic()-last>20:
            print(f'{cfg["cond"]}: collecting native contexts, decision{step+1}, contexts{len(contexts)}',flush=True);last=time.monotonic()
    if len(contexts)!=2:raise RuntimeError(f'Only{len(contexts)} eligible continued-policy contexts in6000 decisions')
    folder=args.out/'contexts';folder.mkdir(parents=True,exist_ok=True)
    replay_errors=[]
    for i,window in enumerate(contexts):
        z=(torch.tensor(window['history_h_before'][0:1],device=args.device),torch.tensor(window['history_c_before'][0:1],device=args.device))
        error=0.
        for t in range(15):
            z=net.lstm(torch.tensor(window['history_x'][t:t+1],device=args.device),z)
            expected=np.r_[window['history_h_after'][t],window['history_c_after'][t]]
            error=max(error,float(np.max(np.abs(torch.cat(z,1).cpu().numpy()[0]-expected))))
        assert error<3e-4,error;replay_errors.append(error)
        np.savez_compressed(folder/f'context{i}.npz',**window)
    metadata=dict(checkpoint_sha256=sha(checkpoint),seed=seed,device=args.device,env_device='cpu',
        contexts=2,decisions=step+1,slots=sorted(used),rejected_intervals=rejections,
        replay_max_errors=replay_errors,seconds=time.monotonic()-started,
        schedule='Actual VisualClock with15 visible updates and integer-uniform30–60 blind updates after discovery; native goal/current codes; original resets only at episode endings',
        condition=cfg['cond'])
    save(args.out/'collection.json',metadata)
    return contexts


def geometry_gates(invariance,trajectory):
    gates={}
    for metric in ('raw','relative_rms'):
        q=invariance['30'][metric];g=q['geometry'];r=q['horizons']['30'];p=r['phase_transport']
        gates[metric]=dict(radius_retains5percent=trajectory['30'][metric]['radius_retention']>=.05,
            largest_edge_below10percent_radius=g['max_edge_over_radius']<=.1,
            nonlocal_separation_above5percent_radius=g['nonlocal_distance_over_radius']['p10']>=.05,
            two_second_fixed_curve_residual_p90_below5percent_radius=r['residual_to_fixed_curve_over_radius']['p90']<=.05,
            phase_coverage80percent=p['circular_coverage']>=.8,
            phase_map_degree_one=abs(p['winding_degree']-1)<.1,
            phase_order_preserved=p['backward_edge_fraction']<=.05,
            radial_gain_below_half=r['radial_normal_gain']['median']<=.5,
            radial_over_tangent_below_half=r['radial_over_tangent_gain']['median']<=.5)
    return gates


@torch.inference_mode()
def prepare_heading_family(net,pop,cfg,window,context,device):
    """15 actual visible updates, including the native initial previous reward."""
    n=768;env=make_env(cfg,window,n,context)
    position=env.pos.clone();direction=env.mdir.clone()
    action_id=context%2;actions=torch.full((n,),action_id,dtype=torch.long)
    code=pop.activity(env.pos,env.mdir,torch.zeros_like(env.pos)).to(device)
    goal=torch.tensor(window['history_x'][0,768:1280],device=device).expand(n,-1)
    initial=torch.tensor(np.r_[window['history_h_before'][0],window['history_c_before'][0]],device=device).expand(n,-1)
    state=tuple(x.contiguous() for x in initial.chunk(2,1))
    old_pa=window['history_x'][0,1280:1286]
    pa0=int(old_pa.argmax()) if old_pa.sum() else -1
    pr0=float(window['history_x'][0,-1])
    for t in range(15):
        theta=env.hd.numpy().copy().astype(np.float64)
        e=net.embed(env.render().to(device))
        pa=torch.full((n,),pa0 if t==0 else action_id,dtype=torch.long,device=device)
        pr=torch.full((n,),pr0 if t==0 else 0.,device=device)
        state=net.lstm(net._concat(e,code,goal,pa,pr,None),state)
        _,reward,done,info=env.step(actions)
        assert not bool(done.any() or info['hit'].any() or reward.any())
    full=torch.cat(state,1).cpu().double()
    dark=net.embed(torch.zeros(1,3,84,84,device=device))
    x=net._concat(dark,code[:1],goal[:1],torch.tensor([action_id],device=device),torch.zeros(1,device=device),None).cpu().double()
    movement={'0':0.};env.vision='none';offset_error=0.
    for t in range(1,61):
        angles=env.hd.numpy().copy().astype(np.float64)
        shift=float(np.angle(np.exp(1j*(angles-theta)).mean()))
        movement[str(t)]=shift;offset_error=max(offset_error,float(np.abs(wrap(angles-theta-shift)).max()))
        _,reward,done,info=env.step(actions)
        assert not bool(done.any() or info['hit'].any() or reward.any())
    assert torch.equal(env.pos,position) and torch.equal(env.mdir,direction)
    assert offset_error<1e-4,offset_error
    checks=dict(position=position[0].tolist(),fixed_last_movement_direction=direction[0].tolist(),
                native_first_previous_reward=pr0,native_first_previous_action=pa0,
                position_max_change=0.,movement_direction_max_change=0.,turn_action=action_id,
                common_turn_offset_max_error_rad=offset_error,visible_updates=15,
                initial_state_source='Native continued-policy visible-interval start, with matching native saved goal and prior action/reward')
    return full,x,theta,movement,checks


def analyze_branch(net,start,dark_x,theta,movement,scale):
    drive=torch.nn.functional.linear(dark_x,net.lstm.weight_ih,net.lstm.bias_ih)
    state=start.clone();trajectory={};snapshots={};metrics={'raw':np.ones(512),'relative_rms':scale}
    radii={name:flow.curve_geometry(start.numpy()/s)['radius'] for name,s in metrics.items()}
    # The protocol's192 analysis angles are a fixed subset of768 rendered
    # phases. Dense neighbors improve geometric resolution without adding tasks.
    select=slice(None,None,4);targets=theta[select]
    decoder=fit_readout((start.numpy()/scale)[select],targets)
    with torch.no_grad():
        for step in range(61):
            if step in TIMES:
                z=state.numpy().copy();snapshots[step]=z;row={}
                for name,s in metrics.items():
                    g=flow.curve_geometry(z/s);g['radius_retention']=g['radius']/radii[name];row[name]=g
                selected=(z/scale)[select]
                refit=fit_readout(selected,targets)
                row['fixed_initial_phase']=phase_diagnostics(selected,targets,decoder)
                row['fixed_current_heading']=phase_diagnostics(selected,targets,decoder,movement[str(step)])
                row['time_refit_information_only']=phase_diagnostics(selected,targets,refit)
                probabilities=net.pi(state[:,:256]).softmax(1)
                row['actor_probability_family_rms']=float((probabilities-probabilities.mean(0)).square().mean().sqrt())
                row['hidden_radius_retention']=float((state[:,:256]-state[:,:256].mean(0)).norm()/(start[:,:256]-start[:,:256].mean(0)).norm())
                row['cell_radius_retention']=float((state[:,256:]-state[:,256:].mean(0)).norm()/(start[:,256:]-start[:,256:].mean(0)).norm())
                trajectory[str(step)]=row
            if step<60:state=advance(net.lstm,drive,state)
    with patch.object(flow,'FLOW',RECOVERY):
        invariance={str(t):{name:flow.invariance_probe(net.lstm,drive,snapshots[t],s,384) for name,s in metrics.items()} for t in ANCHORS}
    gates=geometry_gates(invariance,trajectory)
    return dict(trajectory=trajectory,invariance=invariance,candidate_gates=gates,
                passes_both_metric_candidate_screen=all(all(v.values()) for v in gates.values())),snapshots


def analyze_context(data,cfg,window,context,args):
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
    pop=build_population(cfg,'cpu')
    full,x,theta,movement,physics=prepare_heading_family(net,pop,cfg,window,context,args.device)
    ordinary,change=ordinary_current_input(pop,x,physics['position'],physics['fixed_last_movement_direction'])
    assert len(full)==768 and full.shape[1]==512 and x.shape==(1,1287)
    assert change['saved_goal_max_change']==change['other_input_max_change']==0.
    if args.smoke_only:
        return dict(context=context,state_shape=list(full.shape),input_shape=list(x.shape),input_change=change,physics=physics,
                    smoke_only=True,no_mechanistic_metrics_evaluated=True)
    net.cpu().double()
    blocks=torch.stack((full[:,:256].square().mean().sqrt(),full[:,256:].square().mean().sqrt())).clamp_min(1e-8)
    scale=blocks.repeat_interleave(256).numpy();branches={};stored={}
    for name,inp in [('native_current',x),('ordinary_current',ordinary)]:
        if name=='ordinary_current' and torch.equal(x,ordinary):
            branches[name]=copy.deepcopy(branches['native_current']);branches[name]['exact_identity_with_native']=True
            stored[name]=stored['native_current'];continue
        branches[name],stored[name]=analyze_branch(net,full,inp,theta,movement,scale)
    result=dict(context=context,condition=cfg['cond'],physics=physics,input_change=change,
                primary_reference_seconds=2,primary_future_seconds=4,visible_preparation_seconds=1,
                analysis_angles=192,geometry_rendered_phases=768,branches=branches,
                saved_goal_preserved=True,source_sha256=sha(__file__))
    save(args.out/f'context{context}.json',result)
    np.savez_compressed(args.out/f'context{context}_states.npz',times=np.array(TIMES),theta=theta,scale=scale,
                        native_states=np.stack([stored['native_current'][t] for t in TIMES]),
                        ordinary_states=np.stack([stored['ordinary_current'][t] for t in TIMES]),
                        native_input=x.numpy(),ordinary_input=ordinary.numpy())
    print(json.dumps(dict(context=context,condition=cfg['cond'],candidates={k:v['passes_both_metric_candidate_screen'] for k,v in branches.items()})),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path,default=ROOT/'results/lstm_heading_memory/continued')
    p.add_argument('--device',default='mps');p.add_argument('--smoke-only',action='store_true');p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(design_path=str(DESIGN.relative_to(ROOT)),design_sha256=sha(DESIGN),
        source_sha256={name:sha(ROOT/name) for name in SOURCE_FILES},
        implementation='Native continued contexts from actual scheduled policy rollouts.192 fixed analysis/readout angles, with4fold deterministic rendering oversampling(768phases) for dense full-state geometry;384 uniform-arc probes. Primary2s reference plus2s recovery fits4s training gap. No original recurrent state reuse.',
        anchors=ANCHORS,recovery=RECOVERY,times=TIMES,smoke_only=args.smoke_only)
    frozen=args.out/'implementation_protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==json.loads(json.dumps(protocol)),'Implementation protocol/source changed; choose a new output directory'
    else:save(frozen,protocol)
    if args.freeze_only:return
    if args.checkpoint is None:p.error('--checkpoint is required for execution')
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=data['config']
    if not args.smoke_only:
        assert data.get('completed') and data['extra_frames']==25_024_000,'Primary assay requires the fixed final25.024M checkpoint'
    assert cfg['policy']=='lstm' and cfg['nh']==256 and cfg['extra_input']=='none'
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
    contexts=collect_contexts(net,cfg,args.checkpoint,args)
    results=[analyze_context(data,cfg,w,i,args) for i,w in enumerate(contexts)]
    save(args.out/('smoke_result.json' if args.smoke_only else 'summary.json'),
         dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),condition=cfg['cond'],
              extra_frames=data['extra_frames'],completed=data.get('completed',False),smoke_only=args.smoke_only,
              results=results if args.smoke_only else [{k:r[k] for k in ('context','condition','input_change')}|{'candidate_results':{name:b['passes_both_metric_candidate_screen'] for name,b in r['branches'].items()}} for r in results]))


if __name__=='__main__':main()
