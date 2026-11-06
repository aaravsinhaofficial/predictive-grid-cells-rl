"""Prospective physical-turn curve assay; no circular fit or optimization.

Original outcomes run only against a sealed protocol. Pure geometry helpers and
private synthetic controls can be exercised before sealing.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import native_turn_teacher_preflight as parent
import native_turn_original_memory as memory
import lstm_heading_curve_flow as flow
from lstm_endogenous_curve_recovery import advance,describe
from code.common import build_policy,build_population
from code.env import wrap_pi
from rl_goal_memory import atomic_json,atomic_torch,state_digest

BASE=ROOT/'results/native_turn_geometry'
PROTOCOL=ROOT/'docs/research/native_turn_geometry_protocol.json'
TRAINING_PROTOCOL=ROOT/'docs/research/native_turn_distillation_protocol.json'
PHASES=128
BRANCHES=('native','ordinary','pre_cue_reset_native','pre_cue_reset_ordinary')
LEFT_WINDOWS=(3,8,15,30,60)
CYCLE_EVOLUTION=(2,14,30,60)
NORMAL_SEED=984701
NORMAL_COUNT=16
ANCHORS=8


def same_rows(x):return torch.equal(x,x[:1].expand_as(x))


def endpoint_config(cfg):
    assert cfg['action_repeat']==4 and cfg['episode_len']==1350
    return cfg


@torch.no_grad()
def prepare_family(net,pop,cfg,context,kind,prefix_action,phases=PHASES):
    """One coherent 360-degree family with a SINGLE physical control tape.

    The reference is the post-cue input (probe) or post-prefix input (turn-only).
    Unlike the behavioral antipodal pairs, prefix action never varies by phase.
    Only CNN/input preparation occurs here; no LSTM state is calculated.
    """
    assert kind in parent.KINDS and prefix_action in (0,1)
    endpoint_config(cfg);n=phases
    required=(parent.VISIBLE-1)+parent.PREFIX+(parent.PROBE if kind=='probe_glimpse'else 0)+8+64
    assert cfg['episode_len']-int(context['native_bookkeeping']['t'])>=required
    env=parent.make_env(cfg,context,n)
    bearing=torch.atan2(*(context['goal_center']-context['endpoint']).flip(0))
    target=bearing+torch.arange(n,dtype=torch.float32)*(2*math.pi/n)
    visible_delta=parent.native_motion(cfg,[0]*(parent.VISIBLE-1),2)
    prefix_delta=parent.native_motion(cfg,[prefix_action]*parent.PREFIX,2)
    suffix_delta=parent.native_motion(cfg,parent.SUFFIX,2)
    env.hd[:]=wrap_pi(target-visible_delta-prefix_delta-suffix_delta)
    probe_heading=target-suffix_delta
    distance=parent.native_motion(cfg,[2]*parent.PROBE,0)if kind=='probe_glimpse'else 0.
    env.pos[:]=context['endpoint']-distance*torch.stack((probe_heading.cos(),probe_heading.sin()),1)
    env.spawn[:]=env.pos;parent.brake(env)
    goal=context['native_x'][768:1280].expand(n,-1).clone()
    pa=context['native_x'][1280:1286];pa0=int(pa.argmax())if pa.sum()else-1
    black=net.embed(torch.zeros(1,3,84,84)).expand(n,-1).clone()
    xs=[];os=[];physical=[];labels=[];held=None;cue_index=None
    def append(obs,action,reward,label,visible=False,cue=False):
        native=pop.activity(obs['pos'],obs['dir'],obs['vel'])
        ordinary=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
        emb=net.embed(obs['rgb'])if visible else black
        pa=torch.full((n,),action,dtype=torch.long);pr=torch.full((n,),reward,dtype=torch.float32)
        x=net._concat(emb,native if visible or cue else held,goal,pa,pr,None)
        other=x.clone()
        if cue:other[:,256:768]=ordinary
        xs.append(x);os.append(other);labels.append(label)
        physical.append({k:getattr(env,k).clone()for k in ('pos','hd','mdir','vf','vs','vr','t')})
    obs=env.observe()
    for j in range(parent.VISIBLE):
        append(obs,pa0 if j==0 else 0,float(context['native_x'][-1])if j==0 else 0.,f'visible_{j}',visible=True)
        if j<parent.VISIBLE-1:obs=parent.clean_step(env,torch.zeros(n,dtype=torch.long))
    parent.brake(env)
    held=pop.activity(env.pos,env.mdir,torch.zeros_like(env.pos),shift_scale=0.)
    for j in range(parent.PREFIX):
        obs=parent.clean_step(env,torch.full((n,),prefix_action,dtype=torch.long))
        append(obs,prefix_action,0.,f'turn_{j}')
    parent.brake(env);error=0.
    if kind=='probe_glimpse':
        for j in range(parent.PROBE):
            obs=parent.clean_step(env,torch.full((n,),2,dtype=torch.long));cue=j==parent.PROBE-1
            if cue:
                error=float((env.pos-context['endpoint']).norm(dim=1).max());assert error<1e-5
                env.pos[:]=context['endpoint'];obs=env.observe(obs['vel'])
                held=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
                cue_index=len(xs)
            append(obs,2,0.,f'probe_{j}',cue=cue)
    assert torch.equal(env.pos,context['endpoint'].expand(n,-1))
    parent.brake(env)
    # Subsequent recurrence sees exactly this common held code and true LEFT PA.
    left_x=net._concat(black,held,goal,torch.zeros(n,dtype=torch.long),torch.zeros(n),None)
    assert same_rows(left_x)
    origin_hd=env.hd.clone();origin_t=env.t.clone()
    left_geometry=[]
    for j in range(1,max(LEFT_WINDOWS)+1):
        obs=parent.clean_step(env,torch.zeros(n,dtype=torch.long))
        if j==8:parent.brake(env)
        if j in LEFT_WINDOWS:
            left_geometry.append(dict(updates=j,seconds=j*env.dt,heading=env.hd.tolist(),
                physical_heading_change_rad=wrap_pi(env.hd-origin_hd).tolist(),
                physical_position_error_m=float((env.pos-context['endpoint']).norm(dim=1).max()),
                episode_t=env.t.tolist(),after_choice_brake=j>=8))
        if j==8:
            choice={k:getattr(env,k).clone()for k in ('pos','hd','mdir','vf','vs','vr','t')}
    # Restore the actual eight-update first-choice endpoint and set the explicitly
    # prescribed two-step periodic motor state. No invented previous-action input.
    for k,v in choice.items():getattr(env,k)[:]=v
    k=1-math.exp(-env.dt/env.tau);speed=env.v_max*k/(2-k)
    env.vf.fill_(-speed);env.vs.zero_();env.vr.zero_()
    fb=[];fb_physical=[];cycle_start=env.pos.clone();hd0=env.hd.clone()
    for j in range(64):
        action=2 if j%2==0 else 3
        obs=parent.clean_step(env,torch.full((n,),action,dtype=torch.long))
        x=net._concat(black,held,goal,torch.full((n,),action,dtype=torch.long),torch.zeros(n),None)
        assert same_rows(x)
        if j<2:fb.append(x[0].clone())
        assert torch.equal(x,fb[j%2][None].expand_as(x))
        if j<3 or j%2==1:
            fb_physical.append(dict(update_after_choice=j+1,action=action,
                position_error_m=float((env.pos-cycle_start).norm(dim=1).max()),
                heading_error_rad=float(wrap_pi(env.hd-hd0).abs().max()),
                vf_error=float((env.vf-(speed if j%2==0 else-speed)).abs().max()),
                episode_t=env.t.tolist()))
        if j%2==1:
            assert float((env.pos-cycle_start).norm(dim=1).max())<1e-6
            assert float(wrap_pi(env.hd-hd0).abs().max())<1e-6
    x,o=torch.stack(xs),torch.stack(os)
    outside=torch.ones(1287,dtype=torch.bool);outside[256+torch.as_tensor(pop.pgc_idx,dtype=torch.long)]=False
    assert torch.equal(x[:,:,outside],o[:,:,outside])
    if cue_index is None or not len(pop.pgc_idx):assert torch.equal(x,o)
    if cue_index is not None:assert torch.equal(o[cue_index],o[cue_index,:1].expand_as(o[cue_index]))
    assert torch.equal(x[:,:,768:1280],goal[None].expand(len(x),-1,-1))
    # Three exact equality checks distinguish input closure from learned-state geometry.
    audit=dict(complete=True,phases=n,prefix_action=prefix_action,kind=kind,context=context['context_id'],
        cue_index=cue_index,reference_input_index=len(x)-1,labels=labels,
        full_left_input_common=True,full_forward_input_common=True,full_backward_input_common=True,
        left_input_sha256=parent.tensor_sha(left_x[0]),forward_input_sha256=parent.tensor_sha(fb[0]),backward_input_sha256=parent.tensor_sha(fb[1]),
        physical_family_target='Final body heading at the eight-update choice boundary',
        reference_body_heading=origin_hd.tolist(),target_choice_body_heading=target.tolist(),
        choice_heading_error_max_rad=float(wrap_pi(choice['hd']-target).abs().max()),
        reference_episode_t=origin_t.tolist(),endpoint_correction_max_m=error,
        original_episode_limit=cfg['episode_len'],left_physical=left_geometry,fb_physical=fb_physical,
        fb_initial_vf=-speed,fb_setup_updates=2,fb_B_reference_updates_after_choice=2,fb_F_reference_updates_after_choice=3,
        physical_input_intervention='Black RGB and held ordinary current code; native goal snapshot retained; actual previous actions and zero rewards.',
        grid_sham_exact=not len(pop.pgc_idx))
    assert audit['choice_heading_error_max_rad']<2e-5
    return dict(native_x=x,ordinary_x=o,left_x=left_x[0],fb_x=torch.stack(fb),
        physical={k:torch.stack([v[k]for v in physical])for k in physical[0]},
        target_choice_heading=target,choice_heading=choice['hd'],reference_heading=origin_hd,audit=audit)


def transition_from_inputs(net,inputs):
    """Callable with update index: observation AFTER the truthful given action."""
    inputs=torch.as_tensor(inputs,dtype=torch.float64)
    drive=torch.nn.functional.linear(inputs,net.lstm.weight_ih,net.lstm.bias_ih)
    return lambda z,step:advance(net.lstm,drive[step%len(drive)],z)


def observer(net,kind):
    if kind=='full_state':return lambda z:z
    if kind=='all_hidden':return lambda z:z[:,:z.shape[1]//2]
    assert kind=='all_actor'
    def output(z):
        v=net.pi(z[:,:z.shape[1]//2]);return v-v.mean(1,keepdim=True)
    return output


@torch.no_grad()
def trajectory(transition,starts,steps,needed=None):
    needed=set(range(steps+1))if needed is None else set(needed)
    state=torch.as_tensor(starts,dtype=torch.float64);out={}
    if 0 in needed:out[0]=state.numpy().copy()
    for k in range(steps):
        state=transition(state,k)
        if k+1 in needed:out[k+1]=state.numpy().copy()
    return out


def observation_arrays(states,output,offsets,horizon):
    first=np.concatenate([output(torch.from_numpy(states[t])).detach().numpy()for t in offsets],1)
    last=np.concatenate([output(torch.from_numpy(states[horizon+t])).detach().numpy()for t in offsets],1)
    return first,last


def geometry_screen(first,last,center=False):
    """Existing dense-curve criteria, without a learned/readout projection."""
    first=np.asarray(first,dtype=float);last=np.asarray(last,dtype=float)
    g=flow.curve_geometry(first);q=flow.curve_geometry(last);r=max(g['radius'],1e-30)
    reference=first-first.mean(0)if center else first
    evolved=last-last.mean(0)if center else last
    _,phase0,_,_=flow.nearest_closed_segments(reference,reference)
    distance,phase,_,_=flow.nearest_closed_segments(evolved,reference)
    transport=flow.phase_transport(phase,phase0)
    gates=dict(radius_RMS_at_least_1e_minus4=g['radius']/math.sqrt(first.shape[1])>=1e-4,
        max_edge_below_10percent_radius=g['max_edge_over_radius']<=.1,
        nonlocal_separation_above_5percent=g['nonlocal_distance_over_radius']['p10']>=.05,
        evolved_radius_retains_5percent=q['radius']/r>=.05,
        p90_invariance_residual_below_5percent=float(np.quantile(distance,.9))/r<=.05,
        phase_coverage_80percent=transport['circular_coverage']>=.8,
        phase_degree_one=abs(transport['winding_degree']-1)<.1,
        phase_order_preserved=transport['backward_edge_fraction']<=.05)
    gates={k:bool(v)for k,v in gates.items()}
    return dict(geometry=g,evolved_geometry=q,centered=center,gates=gates,necessary_geometry_pass=all(gates.values()),
        residual_to_reference_curve_over_radius=describe(distance/r),phase_transport=transport,
        radius_retention=q['radius']/r,common_mean_shift_over_radius=float(np.linalg.norm(last.mean(0)-first.mean(0)))/r)


def make_perturbations(curve,scale,anchors=ANCHORS,seed=NORMAL_SEED):
    """All actual phase samples retained; eight fixed physical phases perturbed."""
    curve=np.asarray(curve,dtype=float);scale=np.asarray(scale,dtype=float);p=curve/scale
    n,d=p.shape;half=d//2;ids=np.arange(anchors)*n//anchors
    tangent=np.roll(p,-1,0)-np.roll(p,1,0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-30)
    t=tangent[ids];rng=np.random.default_rng(seed);directions=[];names=[]
    def add(v,name,block=None):
        if block is None:v-=np.sum(v*t,1,keepdims=True)*t
        else:
            b=t[:,block];v[:,block]-=np.sum(v[:,block]*b,1,keepdims=True)*b/np.maximum(np.sum(b*b,1,keepdims=True),1e-30)
        directions.append(v);names.append(name)
    for i in range(NORMAL_COUNT):add(rng.normal(size=(anchors,d)),f'random_full_{i:02d}')
    for name,block in [('hidden_only',slice(0,half)),('cell_only',slice(half,d))]:
        v=np.zeros((anchors,d));v[:,block]=rng.normal(size=(anchors,half));add(v,name,block)
    radial=p[ids]-p.mean(0);add(radial.copy(),'full_radial')
    cradial=radial.copy();cradial[:,:half]=0;add(cradial,'cell_radial',slice(half,d))
    directions.append(t.copy());names.append('tangent')
    directions=np.stack(directions);norms=np.linalg.norm(directions,axis=2)
    directions/=np.maximum(norms[:,:,None],1e-30)
    radius=flow.curve_geometry(p)['radius'];amplitude=.02*radius
    kicks=amplitude*directions[:,None]*np.array([1.,-1.])[None,:,None,None]*scale
    perturbed=curve[ids][None,None]+kicks
    starts=np.concatenate((curve,perturbed.reshape(-1,d)))
    return starts,dict(count=n,anchors=anchors,anchor_indices=ids.tolist(),names=names,
        full_radius=radius,amplitude=amplitude,direction_norms=norms.tolist(),
        scaled_directions=directions,normal_dot_tangent=np.sum(directions[:-1]*t[None],2),
        physical_hidden_bound_excess=np.maximum(np.abs(starts[n:,:half])-1,0).max(1))



def bound_perturbations(starts,metadata,scale,method):
    """Generalized old bounded-pulse rule; retain actual directions explicitly.

    shared: one smaller two-sided amplitude for the entire case.
    clip_equalize: clip h endpoints only, then shrink all42 signed kicks at each
    phase to that phase's minimum realized scaled norm. Never rotate/project the
    resulting kicks back off the tangent, which could violate the bounds again.
    """
    assert method in ('shared','clip_equalize')
    scale=np.asarray(scale);n=metadata['count'];ids=np.asarray(metadata['anchor_indices'])
    nd=len(metadata['names']);a=len(ids);dims=starts.shape[1];half=dims//2
    base=starts[:n][ids];delta=starts[n:].reshape(nd,2,a,dims)-base[None,None]
    if method=='shared':
        dh=np.abs(delta[:,:,:,:half]);margin=np.maximum(1-np.abs(base[:,:half]),0)
        limits=np.divide(margin[None,None],dh,out=np.full_like(dh,np.inf),where=dh>0)
        multiplier=float(min(1.,.9*np.min(limits)))
        actual_end=base[None,None]+delta*multiplier
        target=np.repeat(metadata['amplitude']*multiplier,a)
    else:
        endpoint=base[None,None]+delta
        endpoint[:,:,:,:half]=np.clip(endpoint[:,:,:,:half],-1,1)
        clipped=endpoint-base[None,None]
        assert np.array_equal(clipped[:,:,:,half:],delta[:,:,:,half:])
        norms=np.linalg.norm(clipped/scale,axis=-1);target=norms.min(axis=(0,1))
        factors=np.divide(target[None,None],norms,out=np.zeros_like(norms),where=norms>0)
        actual_end=base[None,None]+clipped*factors[:,:,:,None]
        multiplier=None
    actual=actual_end-base[None,None];scaled=actual/scale
    lengths=np.linalg.norm(scaled,axis=-1);unit=scaled/np.maximum(lengths[:,:,:,None],1e-30)
    tangent=np.asarray(metadata['scaled_directions'])[-1]
    component=np.sum(unit*tangent[None,None],axis=-1)
    normal_component=np.sqrt(np.maximum(1-component**2,0))
    intended=np.asarray(metadata['scaled_directions'])[:,None]*np.array([1.,-1.])[None,:,None,None]
    cosine=np.sum(unit*intended,axis=-1)
    guard=np.abs(component)<=.05;guard[-1]=normal_component[-1]<=.05
    nonzero=lengths>=max(metadata['full_radius'],1e-30)*1e-12
    exact_bound=bool((np.abs(actual_end[:,:,:,:half])<=1).all())
    block_ok=True
    for name,block in [('hidden_only',slice(half,dims)),('cell_only',slice(0,half)),('cell_radial',slice(0,half))]:
        block_ok&=bool(np.count_nonzero(actual[metadata['names'].index(name),:,:,block])==0)
    output=starts.copy();output[n:]=actual_end.reshape(-1,dims)
    new=dict(metadata,bounded_method=method,shared_amplitude_multiplier=multiplier,
        bounded_target_norms=target.tolist(),actual_geometry_guard_all=bool(guard.all()),
        actual_nonzero_all=bool(nonzero.all()),exact_hidden_bounds=exact_bound,block_purity_exact=block_ok,
        physical_construction_conclusive=bool(guard.all()and nonzero.all()and exact_bound and block_ok),
        actual_scaled_directions=unit,actual_scaled_norms=lengths,actual_intended_cosine=cosine,
        actual_tangent_fraction=component,actual_normal_fraction=normal_component,actual_direction_guard=guard,
        actual_physical_kicks=actual,physical_hidden_bound_excess=np.maximum(np.abs(actual_end[:,:,:,:half])-1,0).max(-1).reshape(-1))
    assert exact_bound and block_ok
    return output,new


def initially_observable(result):
    return bool(result['tangent_nontrivial_fraction']>=.8 and
        all(r['normal_informative_fraction']>=.8 for r in result['directions'][:-1]))


def finite_normal_metrics(first,last,metadata):
    """Distances to the same evolved baseline, never re-center each kick."""
    n=metadata['count'];ids=np.asarray(metadata['anchor_indices']);a=len(ids);names=metadata['names'];d=len(names)
    shape=(d,2,a);base0=first[:n];base1=last[:n];kick0=first[n:];kick1=last[n:]
    radius=max(flow.curve_geometry(base0)['radius'],1e-30)
    initial,phase0,closest0,_=flow.nearest_closed_segments(kick0,base0)
    final,phase1,closest1,_=flow.nearest_closed_segments(kick1,base1)
    b0=np.broadcast_to(base0[ids],(d,2,a,base0.shape[1])).reshape(kick0.shape)
    b1=np.broadcast_to(base1[ids],(d,2,a,base1.shape[1])).reshape(kick1.shape)
    total0=np.linalg.norm(kick0-b0,axis=1).reshape(shape);total1=np.linalg.norm(kick1-b1,axis=1).reshape(shape)
    initial=initial.reshape(shape);final=final.reshape(shape)
    normal_nontrivial=initial>=1e-4*radius;total_nontrivial=total0>=1e-4*radius
    leakage=(~total_nontrivial)&(total1>=1e-4*radius)
    ng=final/np.maximum(initial,1e-30);tg=total1/np.maximum(total0,1e-30)
    tangent_valid=total_nontrivial[-1];tangent_final_valid=total1[-1]>=1e-4*radius;tangent_gain=tg[-1]
    ratio=ng/np.maximum(tangent_gain.mean(0)[None,None],1e-30)
    rows=[]
    for i,name in enumerate(names):
        valid=normal_nontrivial[i]&tangent_valid.all(0)[None,:]&tangent_final_valid.all(0)[None,:]
        good=ng[i][valid];rel=ratio[i][valid]
        passes=bool(valid.mean()>=.8 and not leakage[i].any()and len(good)and np.median(good)<=.5 and np.median(rel)<=.5)
        rows.append(dict(name=name,normal_informative_fraction=float(normal_nontrivial[i].mean()),
            total_informative_fraction=float(total_nontrivial[i].mean()),factor_leakage_count=int(leakage[i].sum()),
            informative_pair_fraction=float(valid.mean()),normal_gain=describe(good)if len(good)else None,
            normal_over_tangent_gain=describe(rel)if len(rel)else None,total_gain=describe(tg[i]),passes=passes))
    return dict(directions=rows,normal_directions=20,all_normal_directions_pass=all(r['passes']for r in rows[:-1]),
        tangent_nontrivial_fraction=float(tangent_valid.mean()),final_tangent_nontrivial_fraction=float(tangent_final_valid.mean()),tangent_total_gain=describe(tangent_gain),
        leakage_count=int(leakage[:-1].sum())),dict(initial_normal_distance=initial,final_normal_distance=final,
        initial_total_distance=total0,final_total_distance=total1,normal_gain=ng,total_gain=tg,
        normal_over_tangent_gain=ratio,normal_nontrivial=normal_nontrivial,total_nontrivial=total_nontrivial,
        final_tangent_nontrivial=tangent_final_valid,
        factor_leakage=leakage,initial_arc=phase0.reshape(shape),final_arc=phase1.reshape(shape),
        initial_phase_component=np.linalg.norm(closest0-b0,axis=1).reshape(shape),
        final_phase_component=np.linalg.norm(closest1-b1,axis=1).reshape(shape))


@torch.no_grad()
def finite_probe(transition,curve,scale,output,offsets,horizon,anchors=ANCHORS,bounded=None):
    """Public pure adapter for independent synthetic two-section controls."""
    starts,metadata=make_perturbations(curve,scale,anchors)
    if bounded is not None:starts,metadata=bound_perturbations(starts,metadata,scale,bounded)
    states=trajectory(transition,starts,horizon+max(offsets))
    first,last=observation_arrays(states,output,offsets,horizon)
    result,arrays=finite_normal_metrics(first,last,metadata)
    result['geometry']={str(c):geometry_screen(first[:len(curve)],last[:len(curve)],c)for c in (False,True)}
    result['perturbation_design']={k:v for k,v in metadata.items()if not isinstance(v,np.ndarray)}
    result['hidden_bound_excess_count']=int((metadata['physical_hidden_bound_excess']>0).sum())
    result['physical_conclusive']=bool(metadata.get('physical_construction_conclusive',result['hidden_bound_excess_count']==0)and initially_observable(result))
    result['physical_status']=('pass'if result['all_normal_directions_pass']else'failed_recovery')if result['physical_conclusive']else'inconclusive'
    arrays.update(first=first,last=last,starts=starts,**{k:v for k,v in metadata.items()if isinstance(v,np.ndarray)})
    return result,arrays


def physical_phase_comparison(first,last,headings,physical_rotation,center):
    """Convert nearest polygon segment to its KNOWN physical angle label.

    Polygon arclength is not an angle; no subtraction of unlike phase units.
    Labels merely describe transport after the geometric test, never fit a ring.
    """
    a=first-first.mean(0)if center else first;b=last-last.mean(0)if center else last
    _,_,closest,segment=flow.nearest_closed_segments(b,a)
    edges=np.roll(a,-1,0)-a
    fraction=np.sum((closest-a[segment])*edges[segment],1)/np.maximum(np.sum(edges[segment]**2,1),1e-30)
    headings=np.asarray(headings);delta=np.angle(np.exp(1j*(np.roll(headings,-1)-headings)))
    projected=headings[segment]+fraction*delta[segment]
    neural=np.angle(np.exp(1j*(projected-headings)))
    physical=np.asarray(physical_rotation);residual=np.angle(np.exp(1j*(neural-physical)))
    def circular(x):
        return dict(mean_rad=float(np.angle(np.exp(1j*x).mean())),absolute_rad=describe(np.abs(x)),absolute_max_rad=float(np.max(np.abs(x))),
            concentration=float(abs(np.exp(1j*x).mean())))
    return dict(neural_physical_label_displacement=circular(neural),physical_body_rotation=circular(physical),
        neural_minus_body_phase_change=circular(residual),
        coordinate='Known physical-angle interpolation along the nearest full-dimensional curve segment; descriptive only.')


@torch.no_grad()
def elicit_branches(net,family,incoming):
    n=len(family['target_choice_heading']);cue=family['audit']['cue_index'];answer={}
    for branch in BRANCHES:
        x=family['ordinary_x'if branch.endswith('ordinary')else'native_x'].double()
        state=tuple(v.expand(n,-1).clone()for v in incoming.double().reshape(1,-1).chunk(2,1))
        reset=cue if cue is not None else len(x)  # Turn-only: reset before first common LEFT suffix input.
        for t,item in enumerate(x):
            if branch.startswith('pre_cue_reset')and t==reset:state=tuple(torch.zeros_like(v)for v in state)
            state=net.lstm(item,state)
        if branch.startswith('pre_cue_reset')and reset==len(x):state=tuple(torch.zeros_like(v)for v in state)
        answer[branch]=torch.cat(state,1).numpy().copy()
    assert np.array_equal(answer['pre_cue_reset_ordinary'],np.repeat(answer['pre_cue_reset_ordinary'][:1],n,axis=0))
    if family['audit']['grid_sham_exact']or cue is None:
        assert np.array_equal(answer['native'],answer['ordinary'])
        assert np.array_equal(answer['pre_cue_reset_native'],answer['pre_cue_reset_ordinary'])
    return answer


def state_scales(curve):
    half=curve.shape[1]//2
    return np.maximum(np.repeat(np.sqrt(np.mean(curve.reshape(-1,2,half)**2,axis=(0,2))),half),1e-8)


def all_specs(regime):
    rows=[]
    if regime=='left':
        for window in LEFT_WINDOWS:
            for output in ('full_state','all_hidden','all_actor'):
                offsets=(0,)if output=='full_state'else(0,1)
                evolution=window-max(offsets)
                rows.append((window,evolution,offsets,output))
    else:
        for evolution in CYCLE_EVOLUTION:
            for output in ('full_state','all_hidden','all_actor'):
                offsets=(0,)if output=='full_state'else(0,1)
                rows.append((evolution+max(offsets),evolution,offsets,output))
    return rows


@torch.no_grad()
def analyze_curve(net,curve,transition,regime,headings,physical_rotations,out):
    """All dense rows first; expensive finite probes only after necessary gates."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    specs=all_specs(regime)
    needed={t for _,h,offsets,_ in specs for t in (*offsets,*[h+o for o in offsets])}
    states=trajectory(transition,curve,max(v[0]for v in specs),needed)
    np.savez_compressed(out/'baseline_states.npz',**{f't{k}':v for k,v in states.items()})
    rows=[];probe_cache={};scales={'raw':np.ones(curve.shape[1]),'h_c_RMS':state_scales(curve)}
    for window,evolution,offsets,kind in specs:
        for metric,scale in scales.items():
            outfn=observer(net,kind)
            if kind=='full_state':
                output=lambda z,s=torch.as_tensor(scale):z/s
            else:output=outfn
            first,last=observation_arrays(states,output,offsets,evolution)
            for centered in ((False,)if kind=='full_state'else(False,True)):
                record=geometry_screen(first,last,centered)
                record.update(window_updates=window,evolution_updates=evolution,observation_offsets=list(offsets),
                    window_seconds=window/15.,evolution_seconds=evolution/15.,observable=kind,metric=metric,
                    regime=regime,normal_test_run=False,broad_candidate=False,physical_status='not_tested_geometry_failed',
                    phase_comparison=physical_phase_comparison(first,last,headings,physical_rotations[evolution],centered))
                if kind=='all_actor':
                    pr0=np.concatenate([net.pi(torch.from_numpy(states[t][:,:curve.shape[1]//2])).softmax(1).numpy()for t in offsets],1)
                    pr1=np.concatenate([net.pi(torch.from_numpy(states[evolution+t][:,:curve.shape[1]//2])).softmax(1).numpy()for t in offsets],1)
                    record['actor_probability_reference_RMS']=float(np.sqrt(np.mean((pr0-pr0.mean(0))**2)))
                    record['actor_probability_evolved_RMS']=float(np.sqrt(np.mean((pr1-pr1.mean(0))**2)))
                if record['necessary_geometry_pass']:
                    key=(metric,kind,evolution,offsets)
                    if key not in probe_cache:
                        probe,arrays=finite_probe(transition,curve,scale,output,offsets,evolution)
                        name=f'normal_{metric}_{kind}_{evolution}.npz';np.savez_compressed(out/name,**arrays)
                        followups=[];physical=probe
                        if probe['hidden_bound_excess_count']:
                            for method in ('shared','clip_equalize'):
                                bounded,ba=finite_probe(transition,curve,scale,output,offsets,evolution,bounded=method)
                                bname=f'normal_{metric}_{kind}_{evolution}_{method}.npz'
                                np.savez_compressed(out/bname,**ba)
                                followups.append(dict(method=method,result=bounded,arrays=bname));physical=bounded
                                if bounded['physical_conclusive']:break
                        probe_cache[key]=(probe,name,followups,physical)
                    probe,name,followups,physical=probe_cache[key]
                    record.update(normal_test_run=True,normal_result=probe,normal_arrays=name,
                        bounded_followups=followups,physical_status=physical['physical_status'],
                        broad_candidate=bool(physical['physical_conclusive']and physical['all_normal_directions_pass']))
                rows.append(record)
    # Paired metric conjunction is the actual nomination, not any one favorable metric.
    for row in rows:
        other=[r for r in rows if r['metric']!=row['metric']and all(r[k]==row[k]for k in ('observable','centered','evolution_updates'))]
        assert len(other)==1
        row['both_metrics_candidate']=bool(row['broad_candidate']and other[0]['broad_candidate'])
    atomic_json(out/'result.json',dict(complete=True,regime=regime,rows=rows,
        curve_sha256=hashlib.sha256(np.asarray(curve).tobytes()).hexdigest(),
        baseline_states_sha256=parent.sha(out/'baseline_states.npz'),
        finite_probe_artifacts={p.name:parent.sha(p)for p in out.glob('normal_*.npz')}))
    return rows,states


def fixed_parameter_digest(net):
    h=hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):
        if not k.startswith('lstm.'):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_path(condition,endpoint):
    if endpoint=='original':return ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
    assert endpoint in ('clean','noise')
    return ROOT/f'results/native_turn_distillation/{condition}/{endpoint}/final.pt'


def validate_protocol(design):
    assert design['phases']==128 and design['branches']==list(BRANCHES)and design['no_optimization']
    for key in ('source_sha256','asset_sha256'):
        for name,digest in design[key].items():assert parent.sha(ROOT/name)==digest,name


def freeze():
    assert not PROTOCOL.exists()
    old=json.loads(parent.PROTOCOL.read_text());assets=dict(old['asset_sha256']);clock_audit={}
    assets[str(parent.PROTOCOL.relative_to(ROOT))]=parent.sha(parent.PROTOCOL)
    for condition,b in old['banks'].items():
        assets[b['path']]=b['sha256']
        bank=torch.load(ROOT/b['path'],map_location='cpu',weights_only=True)
        remaining=[bank['config']['episode_len']-int(c['native_bookkeeping']['t'])for c in bank['contexts']]
        assert len(remaining)==16 and min(remaining)>=104
        clock_audit[condition]=dict(contexts=16,required_remaining_updates=104,remaining_updates_by_context=remaining,all_contexts_fit=True)
    sources=dict(old['source_sha256'])
    for name in ('scripts/native_turn_geometry.py','tests/test_native_turn_geometry.py',
        'tests/test_native_turn_geometry_controls.py','docs/research/native_turn_geometry_plan.md',
        'scripts/native_turn_original_memory.py','scripts/lstm_heading_curve_flow.py',
        'scripts/lstm_endogenous_curve_recovery.py','scripts/lstm_heading_memory_probe.py'):
        sources[name]=parent.sha(ROOT/name)
    design=dict(version='native_actual_turn_geometry_v1',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=sources,asset_sha256=assets,parent_protocol_sha256=parent.sha(parent.PROTOCOL),
        native_episode_clock_audit=clock_audit,phases=PHASES,contexts_per_population=16,prefix_actions=[0,1],kinds=list(parent.KINDS),branches=list(BRANCHES),
        endpoints={c:{e:str(model_path(c,e).relative_to(ROOT))for e in ('original','clean','noise')}for c in parent.CONDITIONS},
        future_training_protocol=str(TRAINING_PROTOCOL.relative_to(ROOT)),
        future_terminal_requirements='Complete summary with condition/arm,180updates,1036800frames,259200decisions, fixed teacher/head, final checkpoint hash/size, matching policy/config/training protocol and source hashes. No intermediate endpoint substitution.',
        preparation='One archived physical/input family per population/context/fixed prefix/mode; unchanged CNN and actor required for all endpoints.',
        left_windows=list(LEFT_WINDOWS),cycle_evolution=list(CYCLE_EVOLUTION),
        full_state_offsets=[0],hidden_actor_offsets=[0,1],
        left_timing='Full state evolves the stated window; hidden/actor evolves window minus one and observes offsets0,1. Latest input is the stated window.',
        cycle_timing='ReferenceB: two truthful setup updates after choice. ReferenceF: next actual FORWARD, three updates after choice. Two-section output latest=evolution+1.',
        directions='16 fixed full normals, hidden-only, cell-only, full radial, cell radial, tangent; both signs2percent radius at8 fixed phase indices.',
        normal_seed=NORMAL_SEED,normal_directions=20,normal_anchors=ANCHORS,
        bounded_followup='Every geometry-passing row with nominal h-bound violations: shared amplitude capped at90percent of available two-sided h margin; if inconclusive, clip h endpoints and shrink all signs/directions at each phase to minimum realized scaled norm. Record actual directions; normal/tangent contamination at most5percent. Zero/unobservable/incorrect-type bounded kicks are inconclusive, not failed dynamics.',
        staging='All dense curve rows retained; finite-normal assay only for necessary-geometry passing rows. Actual nomination is broad pass in both raw and h/c-RMS metrics.',
        old_thresholds_unchanged=True,physical_phase='Known angle interpolation after geometry only; neural minus actual physical rotation retained.',
        training_outcome_selection=False,no_optimization=True,no_cloud=True)
    validate_protocol(design);atomic_json(PROTOCOL,design);print(parent.sha(PROTOCOL))


@torch.no_grad()
def prepare_inputs(condition):
    design=json.loads(PROTOCOL.read_text());validate_protocol(design)
    old=json.loads(parent.PROTOCOL.read_text());record=old['banks'][condition]
    bank=torch.load(ROOT/record['path'],map_location='cpu',weights_only=True);cfg=endpoint_config(bank['config'])
    checkpoint=torch.load(model_path(condition,'original'),map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):net=build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(checkpoint['policy']);pop=build_population(cfg,'cpu')
    folder=BASE/'inputs'/condition;assert not folder.exists()or not any(folder.iterdir());folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();records=[]
    for context in bank['contexts']:
        for prefix in (0,1):
            for kind in parent.KINDS:
                family=prepare_family(net,pop,cfg,context,kind,prefix)
                name=f"context{context['context_id']:02d}_prefix{prefix}_{kind}.pt"
                atomic_torch(folder/name,family)
                records.append(dict(context=context['context_id'],prefix=prefix,kind=kind,path=name,sha256=parent.sha(folder/name),
                    input_native_sha256=parent.tensor_sha(family['native_x']),input_ordinary_sha256=parent.tensor_sha(family['ordinary_x']),
                    audit=family['audit']))
                print(json.dumps(dict(prepared=condition,family=name,seconds=time.monotonic()-start)),flush=True)
    assert len(records)==64
    validate_protocol(design)
    atomic_json(folder/'summary.json',dict(complete=True,condition=condition,protocol_sha256=parent.sha(PROTOCOL),
        checkpoint_sha256=parent.sha(model_path(condition,'original')),policy_hash=state_digest(net),
        fixed_parameter_hash=fixed_parameter_digest(net),config=cfg,records=records,seconds=time.monotonic()-start))


@torch.no_grad()
def run(condition,endpoint):
    design=json.loads(PROTOCOL.read_text());validate_protocol(design)
    prepared=BASE/'inputs'/condition;prep=json.loads((prepared/'summary.json').read_text())
    assert prep['complete']and prep['protocol_sha256']==parent.sha(PROTOCOL)
    old=json.loads(parent.PROTOCOL.read_text());bank=torch.load(ROOT/old['banks'][condition]['path'],map_location='cpu',weights_only=True)
    cfg=endpoint_config(bank['config']);path=model_path(condition,endpoint)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    training_provenance=None
    if endpoint=='original':assert checkpoint['frames']==1_000_000_000
    else:
        terminal_path=path.parent/'summary.json';terminal=json.loads(terminal_path.read_text())
        training_design=json.loads(TRAINING_PROTOCOL.read_text())
        assert terminal['complete']and terminal['condition']==condition and terminal['arm']==endpoint
        assert terminal['updates']==checkpoint['updates']==180 and terminal['extra_frames']==checkpoint['extra_frames']==1_036_800
        assert terminal['policy_decisions']==259_200 and terminal['fixed_tensors_unchanged']and terminal['teacher_unchanged']
        assert terminal['artifacts']['final.pt']==dict(sha256=parent.sha(path),bytes=path.stat().st_size)
        assert terminal['protocol_sha256']==checkpoint['protocol_sha256']==parent.sha(TRAINING_PROTOCOL)
        assert terminal['initial_policy_hash']==checkpoint['teacher_policy_hash']==prep['policy_hash']
        assert checkpoint['config']==cfg and checkpoint['arm']==endpoint and checkpoint['original_frames']==1_000_000_000
        for name,digest in training_design['source_sha256'].items():assert parent.sha(ROOT/name)==digest,name
        training_provenance=dict(terminal_summary_sha256=parent.sha(terminal_path),training_protocol_sha256=parent.sha(TRAINING_PROTOCOL),
            terminal_complete=True,teacher_policy_hash=terminal['initial_policy_hash'],final_policy_hash=terminal['final_policy_hash'],
            frozen_tensor_hash=terminal['frozen_tensor_hash'])
    with torch.random.fork_rng(devices=[]):net=build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(checkpoint['policy']);assert fixed_parameter_digest(net)==prep['fixed_parameter_hash']
    if training_provenance:assert state_digest(net)==training_provenance['final_policy_hash']
    census=torch.load(ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt',map_location='cpu',weights_only=True)
    incoming,prefix_audit=memory.replay_own_prefixes(net,bank,census,endpoint=='original');del census
    digest=state_digest(net);net.double();folder=BASE/condition/endpoint
    assert not folder.exists()or not any(folder.iterdir());folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'prefix_replay.json',prefix_audit);allrows=[];history_states={};began=time.monotonic()
    for record in prep['records']:
        fp=prepared/record['path'];assert parent.sha(fp)==record['sha256']
        family=torch.load(fp,map_location='cpu',weights_only=True)
        branches=elicit_branches(net,family,incoming[record['context']])
        left=transition_from_inputs(net,family['left_x'][None]);fb=transition_from_inputs(net,family['fb_x'])
        leftrot={r['updates']:np.asarray(r['physical_heading_change_rad'])for r in family['audit']['left_physical']}
        # Missing output evolution offsets are computed from the same exact scalar
        # motor recurrence, including the known brake after eight LEFT actions.
        for t in range(61):
            if t not in leftrot:
                angle=parent.native_motion(cfg,[0]*min(t,8),2)
                if t>8:angle+=parent.native_motion(cfg,[0]*(t-8),2)
                leftrot[t]=np.repeat(angle,PHASES)
        for branch,curve in branches.items():
            # Exact shams are copied by reference in the terminal record, while
            # every requested case remains present in the denominator.
            sham=None
            if branch=='ordinary'and(family['audit']['grid_sham_exact']or family['audit']['cue_index']is None):sham='native'
            if branch=='pre_cue_reset_ordinary'and(family['audit']['grid_sham_exact']or family['audit']['cue_index']is None):sham='pre_cue_reset_native'
            label=f"context{record['context']:02d}_prefix{record['prefix']}_{record['kind']}_{branch}"
            source_label=label.rsplit('_'+branch,1)[0]+'_'+sham if sham else None
            if sham:
                copied=[r for r in allrows if r['family']==source_label]
                assert copied
                allrows.extend([{**r,'family':label,'branch':branch,'exact_sham_of':source_label}for r in copied])
                history_states[(record['context'],record['kind'],record['prefix'],branch)]=history_states[(record['context'],record['kind'],record['prefix'],sham)]
                continue
            leftstates=trajectory(left,curve,8);choice=leftstates[8]
            setup=trajectory(fb,choice,3);sectionB=setup[2];sectionF=setup[3]
            regime_starts={'left':(curve,left,family['reference_heading'].numpy(),leftrot,0),
                'fb_B':(sectionB,fb,family['choice_heading'].numpy(),{t:np.zeros(PHASES)for t in range(62)},10),
                'fb_F':(sectionF,transition_from_inputs(net,family['fb_x'].flip(0)),family['choice_heading'].numpy(),{t:np.zeros(PHASES)for t in range(62)},11)}
            savedhist={}
            for regime,(reference,transition,headings,rotations,absolute)in regime_starts.items():
                rows,states=analyze_curve(net,reference,transition,regime,headings,rotations,folder/label/regime)
                savedhist[regime]=folder/label/regime/'baseline_states.npz'
                for row in rows:
                    row.update(condition=condition,endpoint=endpoint,context=record['context'],prefix=record['prefix'],kind=record['kind'],branch=branch,
                        family=label,exact_sham_of=None,reference_updates_after_cue=absolute,
                        latest_updates_after_cue=absolute+row['window_updates'])
                allrows.extend(rows)
            history_states[(record['context'],record['kind'],record['prefix'],branch)]=savedhist
            print(json.dumps(dict(condition=condition,endpoint=endpoint,family=label,rows=len(allrows),seconds=time.monotonic()-began)),flush=True)
    natural=[]
    for context in range(16):
        for kind in parent.KINDS:
            for branch in BRANCHES:
                a=history_states[(context,kind,0,branch)];b=history_states[(context,kind,1,branch)]
                for regime in a:
                    with np.load(a[regime])as az,np.load(b[regime])as bz:
                        for key in sorted(az.files,key=lambda q:int(q[1:])):
                            t=int(key[1:]);assert key in bz.files
                            for output in ('full_state','all_hidden','all_actor'):
                                obs=observer(net,output)
                                aa=obs(torch.from_numpy(az[key])).numpy();bb=obs(torch.from_numpy(bz[key])).numpy()
                                radius=.5*(np.sqrt(np.mean(np.sum((aa-aa.mean(0))**2,1)))+np.sqrt(np.mean(np.sum((bb-bb.mean(0))**2,1))))
                                natural.append(dict(context=context,kind=kind,branch=branch,regime=regime,updates=t,observable=output,
                                    same_heading_history_difference=describe(np.linalg.norm(aa-bb,axis=1)),
                                    difference_over_angular_family_radius=describe(np.linalg.norm(aa-bb,axis=1)/max(radius,1e-30))))
    validate_protocol(design)
    artifacts={str(p.relative_to(folder)):dict(sha256=parent.sha(p),bytes=p.stat().st_size)for p in folder.rglob('*')if p.is_file()}
    atomic_json(folder/'summary.json',dict(complete=True,condition=condition,endpoint=endpoint,checkpoint_sha256=parent.sha(path),
        policy_hash_before_float64=digest,training_provenance=training_provenance,protocol_sha256=parent.sha(PROTOCOL),input_summary_sha256=parent.sha(prepared/'summary.json'),
        contexts=16,physical_families=64,branches=4,rows=allrows,natural_history_comparison=natural,
        individual_broad_pass_rows=sum(r['broad_candidate']for r in allrows),both_metrics_candidate_rows=sum(r['both_metrics_candidate']for r in allrows),
        seconds=time.monotonic()-began,artifacts=artifacts))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--prepare',action='store_true')
    p.add_argument('--condition',choices=parent.CONDITIONS);p.add_argument('--endpoint',choices=('original','clean','noise'),default='original')
    a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if a.freeze:freeze()
        elif a.prepare:assert a.condition;prepare_inputs(a.condition)
        else:assert a.condition;run(a.condition,a.endpoint)
if __name__=='__main__':main()
