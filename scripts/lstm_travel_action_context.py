"""Action-context sensitivity using the same physical travel paths and endpoints.

Each family uses one actual native translation action. Travel direction is theta;
body heading is adjusted to preserve the common world-space path. Preparation
starts from the same saved native recurrent context. All previously frozen
common-input release and full-state candidate criteria are reused unchanged.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from lstm_heading_memory_probe import make_env,sha,save,wrap
import lstm_travel_memory_probe as travel

PHASES=travel.PHASES
path_clearance=travel.path_clearance
ACTIONS={2:'forward',3:'backward',4:'strafe_left',5:'strafe_right'}
BODY_OFFSET={2:0.,3:np.pi,4:-np.pi/2,5:np.pi/2}


@torch.inference_mode()
def prepare_action_family(net,pop,cfg,window,context,device,action):
    assert action in ACTIONS
    assert cfg['pgc_shift_mode']=='space' and cfg['extra_input']=='none'
    target=np.asarray(window['history_pos'][0],dtype=np.float32)
    geometry=path_clearance(target,window['history_goal_center'][0],cfg)
    assert geometry['eligible'],geometry
    env=make_env(cfg,window,PHASES,context)
    angle=torch.arange(PHASES,dtype=torch.float32)*(2*np.pi/PHASES)
    env.hd[:]=angle+BODY_OFFSET[action]
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
    pr0=float(window['history_x'][0,-1]);actions=torch.full((PHASES,),action,dtype=torch.long)
    obs=env.observe();endpoint_error=0.
    for step in range(15):
        if step==14:
            endpoint_error=float((env.pos-torch.as_tensor(target)).norm(dim=1).max())
            assert endpoint_error<1e-5,endpoint_error
            env.pos[:]=torch.as_tensor(target)  # bounded float32 endpoint snapping
            obs=env.observe(obs['vel'])
        code=pop.activity(obs['pos'],obs['dir'],obs['vel']).to(device)
        embedding=net.embed(obs['rgb'].to(device))
        pa=torch.full((PHASES,),pa0 if step==0 else action,dtype=torch.long,device=device)
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
    metadata=dict(phases=PHASES,visible_observations=15,native_translation_physics_updates=14,
        translation_action=action,translation_action_name=ACTIONS[action],
        body_heading_offset_from_travel_radians=BODY_OFFSET[action],
        endpoint=target.tolist(),path_clearance=geometry,endpoint_error_before_snap_m=endpoint_error,
        endpoint_snap='After14 actual translation updates, numerical endpoint discrepancies below1e-5m are snapped to the exact shared target before the final visible observation.',
        travel_phase_uniformity_max_error_rad=uniform_error,velocity_before_stop=previous_velocity,
        velocity_after_stop=[0.,0.,0.],stop_intervention='Explicitly set vf,vs,vr to0 after the last visible update; then execute a native turn action. This is a controlled stationary-state assay, not a claim that native turn commands instantly brake.',
        initial_position_range=[initial_positions.min(0).values.tolist(),initial_positions.max(0).values.tolist()],
        turn_action=turn,last_movement_direction_held_exactly=True,position_held_exactly=True,
        unchanged_input_outside_current_pgc=True,current_predictive_slots=int(pop.n_pgc),
        native_first_previous_action=pa0,native_first_previous_reward=pr0,
        target_variable='Last actual travel direction, fixed during the stationary turn; body heading continues changing')
    return full,theta,ordinary_x,native_x,metadata


def run(args):
    rows=[]
    for condition in ('gc100','pgc33'):
        for seed in (1,2):
            stem=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
            window=np.load(ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz')
            for action in ACTIONS:
                folder=args.out/ACTIONS[action]
                a=SimpleNamespace(device=args.device,out=folder,cohort='original_action_context_pilot')
                with patch.object(travel,'prepare_travel_family',partial(prepare_action_family,action=action)):
                    record=travel.analyze(dict(checkpoint=str(stem.with_suffix('.pt')),config=str(stem.with_suffix('.json'))),window,0,a)
                record.update(action_context_source_sha256=sha(__file__),translation_action=action,
                    native_context_source=str(ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'))
                artifact=folder/f'{condition}_s{seed}/context0_states.npz'
                if action==2:
                    published=ROOT/f'results/lstm_travel_memory/original/{condition}_s{seed}/context0_states.npz'
                    with np.load(published) as old,np.load(artifact) as current:
                        deltas={key:float(np.max(np.abs(current[key]-old[key]))) for key in ('states','theta','ordinary_input','native_inputs','scale')}
                    record['published_forward_regression_max_abs']=deltas
                    assert max(deltas.values())<1e-4,deltas
                save(folder/f'{condition}_s{seed}/context0.json',record)
                rows.append(dict(condition=condition,seed=seed,translation_action=action,action_name=ACTIONS[action],
                    candidate=record['ordinary_release']['passes_both_metric_candidate_screen']))
    save(args.out/'summary.json',dict(contexts=len(rows),candidates=sum(r['candidate'] for r in rows),results=rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='mps');p.add_argument('--out',type=Path,default=ROOT/'results/lstm_travel_memory/action_context_original')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='travel_action_context_sensitivity_v1',conditions=['gc100','pgc33'],seeds=[1,2],contexts=[0],
        actions=ACTIONS,body_heading_offset_from_travel_radians=BODY_OFFSET,
        motivation='Native sampled original histories contain no forward actions for Grid; controlled forward-only histories may miss practiced backward or strafe contexts. This sensitivity covers all four native translation actions without selecting the best outcome.',
        control='Same actual world-space starts, paths, endpoints, saved-goal/room, and initial recurrent state for every action. Body heading and true previous-action input reflect the chosen native action. Fifteen visible observations contain14 actual deterministic translation updates; preserve native previous-action/reward only at first update.',
        release='Same bounded endpoint snap, explicit velocity stop, common stationary-turn input, and replacement of only current predictive slots by ordinary grid. No active direction cue in the autonomous screen.',
        analysis='Unchanged travel/continued full512 raw and h/cRMS metrics, within4s horizon, radial/tangent tests, decoder diagnostics, and every frozen candidate gate. Report all16 action/model contexts. Same-world paths do not imply identical visual histories because body heading differs by action.',
        interpretation='These are controlled native-physics action histories initialized from real policy histories, not claims that the actor chose each scripted sequence. Original forward outcomes remain primary and are regression-checked; this is a separately frozen sensitivity.',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_travel_action_context.py','scripts/lstm_travel_memory_probe.py','scripts/lstm_continued_heading_memory.py','scripts/lstm_heading_curve_flow.py','scripts/lstm_heading_memory_probe.py','scripts/lstm_endogenous_curve_recovery.py','code/env.py','code/nets.py','code/codes.py']},
        inputs={f'{c}_s{s}':dict(checkpoint_sha256=sha(ROOT/f'models/{c}/{c}_lstm_s{s}_1000M.pt'),context_sha256=sha(ROOT/f'results/original_goal_bearing_manifold/pilot/{c}_s{s}/context0_states.npz')) for c in ('gc100','pgc33') for s in (1,2)})
    # JSON object keys are strings after serialization.
    protocol=json.loads(json.dumps(protocol))
    frozen=args.out/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol,'Frozen action-context protocol changed'
    else:save(frozen,protocol)
    if not args.freeze_only:run(args)


if __name__=='__main__':main()
