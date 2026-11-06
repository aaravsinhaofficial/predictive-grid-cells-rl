"""Observed native turn-trajectory impulses; no fitted circle, training, or cloud."""
from __future__ import annotations
import argparse,datetime as dt,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from native_movement_cycles import (TraceArena,ENV_KEYS,PHYSICAL_KEYS,current_inputs,
    intervention_step,sha,verify_step)
from code.common import build_policy,build_population
from rl_goal_memory import atomic_json,atomic_torch,state_digest
PROTOCOL=ROOT/'docs/research/native_movement_pulse_protocol.json'
MANIFEST=ROOT/'docs/research/native_movement_pulse_source_manifest.json'
MODES=('native','current_grid','current_candidate_output_rescue')
PULSES=('sham','turn_plus','turn_minus','drift_plus','drift_minus','normal_plus','normal_minus')
AMPLITUDE=.02;HORIZON=30;OFF=np.pi/8


def crossing(phase,states,level):
    """First upward passage, retaining temporal order through reversals."""
    indices=np.flatnonzero((phase[:-1]<level)&(phase[1:]>=level))
    if not len(indices):raise ValueError(f'missing first passage {level}')
    i=int(indices[0]);weight=float((level-phase[i])/(phase[i+1]-phase[i]))
    return states[i]*(1-weight)+states[i+1]*weight,dict(left=i,right=i+1,weight=weight,level=float(level))


def select_cases(data,census):
    selected=[];missing=[]
    for slot in range(32):
        segments=[r for r in census['segments']if r['slot']==slot and r['known_goal'] and r['repeated_same_sign_pairs']>0]
        chosen=None
        for segment in segments:
            cycles=[r for r in census['cycles']if r['slot']==slot and r['segment_start']==segment['start_step']]
            for first,second in zip(cycles[:-1],cycles[1:]):
                if first['sign']==second['sign'] and second['start_step']==first['end_step']+1:
                    chosen=(segment,first,second);break
            if chosen:break
        if chosen is None:missing.append(dict(slot=slot,reason='no known-goal return with repeated same-sign net winding'));continue
        segment,first,second=chosen;anchor=second['end_step']+1
        # Do not select a later, more convenient cycle if the mechanical first is terminal.
        if anchor>segment['end_step'] or anchor>=len(data['x']):
            missing.append(dict(slot=slot,reason='first repeated crossing reaches terminal boundary',segment=segment));continue
        start,end=segment['start_step'],segment['end_step'];sign=first['sign']
        delta=(data['post']['hd'][start:end+1,slot]-data['pre']['hd'][start:end+1,slot]+np.pi)%(2*np.pi)-np.pi
        phase=np.r_[0,np.cumsum(delta.numpy().astype(np.float64))]*sign
        states=np.concatenate((data['state_before'][start:end+1,slot].numpy(),data['state_after'][end:end+1,slot].numpy()))
        # First crossing's integer level is determined by its actual first-passage time.
        center=float(round(phase[first['end_step']-start+1]/(2*np.pi))*2*np.pi)
        if center<=0:raise AssertionError('invalid winding center')
        zm,bm=crossing(phase,states,center-OFF);zp,bp=crossing(phase,states,center+OFF)
        z0,b0=crossing(phase,states,center);z1,b1=crossing(phase,states,center+2*np.pi)
        observed=states[:anchor-start+1]
        scale=np.r_[np.full(256,max(float(np.sqrt(np.mean(observed[:,:256]**2))),1e-8)),
                    np.full(256,max(float(np.sqrt(np.mean(observed[:,256:]**2))),1e-8))]
        def unit(v):
            v=np.asarray(v,dtype=np.float64)/scale;rms=float(np.sqrt(np.mean(v*v)))
            if rms<1e-10:raise ValueError('degenerate observed direction')
            return v/rms
        try:turn=unit(zp-zm);drift=unit(z1-z0)
        except ValueError as error:
            missing.append(dict(slot=slot,reason=str(error),segment=segment,anchor=anchor));continue
        rng=np.random.default_rng(908201+data['seed']*1000+slot)
        normal=rng.normal(size=512)
        basis=[]
        for v in (turn,drift):
            q=v.copy()
            for b in basis:q-=np.dot(q,b)*b
            if np.linalg.norm(q)>1e-8:basis.append(q/np.linalg.norm(q))
        for b in basis:normal-=np.dot(normal,b)*b
        normal/=np.sqrt(np.mean(normal*normal))
        directions=np.stack((np.zeros(512),turn,-turn,drift,-drift,normal,-normal))
        pulses=torch.tensor(directions*scale[None,:]*AMPLITUDE,dtype=torch.float32)
        incoming=data['state_before'][anchor,slot]
        selected.append(dict(slot=slot,anchor=anchor,segment=segment,first_cycle=first,second_cycle=second,
            horizon=min(HORIZON,len(data['x'])-anchor),replay_horizon=min(HORIZON,end-anchor+1),
            scale=torch.tensor(scale,dtype=torch.float32),pulses=pulses,
            metadata=dict(first_passage_brackets=[bm,bp,b0,b1],center_level=center,
                anchor_phase=float(phase[anchor-start]),turn_drift_cosine=float(np.mean(turn*drift)),
                normal_basis_rank=len(basis),initial_scaled_rms=(pulses/torch.tensor(scale)).square().mean(1).sqrt().tolist(),
                h_abs_max=(incoming[None,:256]+pulses[:,:256]).abs().max(1).values.tolist(),
                h_bound_overshoot_count=((incoming[None,:256]+pulses[:,:256]).abs()>1).sum(1).tolist())))
    return selected,missing


def clone_env(data,anchor,groups):
    env=TraceArena(data['config'],32,groups,898201+1000*data['seed'],'cpu')
    for key in ENV_KEYS:
        value=data['pre'][key][anchor];setattr(env,key,value.repeat((groups,)+(1,)*(value.ndim-1)).clone())
    for key,value in data['world'].items():setattr(env,key,value.repeat((groups,)+(1,)*(value.ndim-1)).clone())
    env.gen.set_state(data['environment_rng'][anchor].clone())
    obs=dict(pos=env.pos.clone(),dir=env.mdir.clone(),vel=data['obs_vel'][anchor].repeat(groups,1))
    return env,obs


def chunk_step(net,xn,xo,state):
    result=[]
    for group in range(21):
        sl=slice(group*32,(group+1)*32)
        z=intervention_step(net,xn[sl],xo[sl],(state[0][sl],state[1][sl]),MODES[group//7])
        result.append(z)
    return tuple(torch.cat([z[k]for z in result])for k in range(2))


def target_rows(tensor,slot):return tensor.reshape(21,32,*tensor.shape[1:])[:,slot].clone()


@torch.inference_mode()
def verify_selected_native_fork(net,pop,data,case):
    """Finish exact native-only reconstruction before any counterfactual branch."""
    anchor=case['anchor'];env,obs=clone_env(data,anchor,1)
    state=tuple(data['state_before'][anchor].chunk(2,1))
    embedding=data['x'][anchor,:,:256].clone()
    goal=data['x'][anchor,:,768:1280].clone()
    pa=data['x'][anchor,:,1280:1286].argmax(1);pa[data['x'][anchor,:,1280:1286].sum(1)==0]=-1
    pr=data['x'][anchor,:,1286].clone()
    for age in range(case['horizon']):
        k=anchor+age;xn,_,_=current_inputs(net,pop,obs,embedding,goal,pa,pr)
        assert torch.equal(xn,data['x'][k]),('native-only x',k)
        state=net.lstm(xn,state);logits=net.pi(state[0])
        assert torch.equal(torch.cat(state,1),data['state_after'][k])and torch.equal(logits,data['logits'][k])
        action=(data['uniform'][k,:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        assert torch.equal(action,data['action'][k])
        env.trace_first=None;obs,reward,done,info=env.step(action)
        assert all(torch.equal(env.trace_first[key],data['post'][key][k])for key in PHYSICAL_KEYS)
        assert torch.equal(info['hit'],data['hit'][k])and torch.equal(done,data['done'][k])
        if k+1<len(data['x']):assert torch.equal(env.gen.get_state(),data['environment_rng'][k+1].clone())
        hit=info['hit']
        if bool(hit.any()):goal[hit]=pop.activity(info['hit_pos'][hit],info['hit_dir'][hit],info['hit_vel'][hit])
        pa=action;pr=reward.clamp(-data['config']['reward_clip'],data['config']['reward_clip'])
        if bool(done.any()):
            state=tuple(v*(~done)[:,None]for v in state);goal[done]=0;pa[done]=-1;pr[done]=0
        if age+1<case['horizon']:embedding=net.embed(obs['rgb'])
    return dict(slot=case['slot'],anchor=anchor,steps=case['horizon'],all32_native_input_state_actor_action_physics_rng_exact=True)


@torch.inference_mode()
def run_case(net,pop,data,case):
    anchor,slot=case['anchor'],case['slot'];groups=21
    env,obs=clone_env(data,anchor,groups)
    z=data['state_before'][anchor].repeat(groups,1)
    z.reshape(groups,32,512)[:,slot]+=case['pulses'].repeat(3,1)
    initial=target_rows(z,slot);state=(z[:,:256].clone(),z[:,256:].clone())
    embedding=data['x'][anchor,:,:256].repeat(groups,1)
    goal=data['x'][anchor,:,768:1280].repeat(groups,1)
    pa=data['x'][anchor,:,1280:1286].argmax(1);pa[data['x'][anchor,:,1280:1286].sum(1)==0]=-1;pa=pa.repeat(groups)
    pr=data['x'][anchor,:,1286].repeat(groups)
    closed={k:[]for k in ('state','xn','xo','logits','action','pos','hit','done','active','path_step')}
    active=torch.ones(groups,dtype=torch.bool);last=data['pre']['pos'][anchor,slot].repeat(groups,1)
    goal0=data['pre']['goal'][anchor,slot].clone();held=last.clone();checks={k:0. for k in ('input','state','logits','physics','environment_rng')}
    for age in range(case['horizon']):
        k=anchor+age;xs=[];os=[]
        for group in range(groups):
            sl=slice(group*32,(group+1)*32);ob={key:value[sl]for key,value in obs.items()if key!='rgb'}
            xn,xo,_=current_inputs(net,pop,ob,embedding[sl],goal[sl],pa[sl],pr[sl]);xs.append(xn);os.append(xo)
        xn,xo=torch.cat(xs),torch.cat(os)
        # Full original32 group is checked, including slots irrelevant to the selected pulse.
        checks['input']=max(checks['input'],float((xn[:32]-data['x'][k]).abs().max()))
        assert torch.equal(xn[:32],data['x'][k]),('native input fork mismatch',k,checks)
        actual=chunk_step(net,xn,xo,state)
        logits=torch.cat([net.pi(actual[0][g*32:(g+1)*32])for g in range(groups)])
        checks['state']=max(checks['state'],float((torch.cat(actual,1)[:32]-data['state_after'][k]).abs().max()))
        checks['logits']=max(checks['logits'],float((logits[:32]-data['logits'][k]).abs().max()))
        assert checks['state']==0 and checks['logits']==0
        uniform=data['uniform'][k].repeat(groups)
        action=torch.cat([(uniform[g*32:(g+1)*32,None]>logits[g*32:(g+1)*32].softmax(1).cumsum(1)).sum(1).clamp_max(5)for g in range(groups)])
        assert torch.equal(action[:32],data['action'][k])
        closed['active'].append(active.clone());closed['state'].append(target_rows(torch.cat(actual,1),slot))
        closed['xn'].append(target_rows(xn,slot));closed['xo'].append(target_rows(xo,slot));closed['logits'].append(target_rows(logits,slot));closed['action'].append(target_rows(action,slot))
        env.trace_first=None;obs,reward,done,info=env.step(action)
        for key in PHYSICAL_KEYS:
            error=float((env.trace_first[key][:32].double()-data['post'][key][k].double()).abs().max());checks['physics']=max(checks['physics'],error)
        assert checks['physics']==0 and torch.equal(info['hit'][:32],data['hit'][k])and torch.equal(done[:32],data['done'][k])
        if k+1<len(data['x']):assert torch.equal(env.gen.get_state(),data['environment_rng'][k+1].clone())
        pos=target_rows(env.trace_first['pos'],slot);hit=target_rows(info['hit'],slot);terminal=target_rows(done,slot)
        step=(pos-last).norm(dim=1)*active;held[active]=pos[active];last=pos
        closed['pos'].append(held.clone());closed['path_step'].append(step)
        closed['hit'].append(hit&active);closed['done'].append(terminal&active)
        active&=~(hit|terminal)
        for group in range(groups):
            sl=slice(group*32,(group+1)*32);mask=info['hit'][sl]
            if bool(mask.any()):
                updated=goal[sl].clone();updated[mask]=pop.activity(info['hit_pos'][sl][mask],info['hit_dir'][sl][mask],info['hit_vel'][sl][mask]);goal[sl]=updated
        state=actual;pa=action;pr=reward.clamp(-data['config']['reward_clip'],data['config']['reward_clip'])
        if bool(done.any()):
            mask=~done;state=tuple(v*mask[:,None]for v in state);goal[done]=0;pa[done]=-1;pr[done]=0
        if age+1<case['horizon']:embedding=torch.cat([net.embed(obs['rgb'][g*32:(g+1)*32])for g in range(groups)])
    closed={k:torch.stack(v)for k,v in closed.items()}
    # Replay is the literal recorded native experience, including action/reward inputs.
    rz=data['state_before'][anchor].repeat(groups,1);rz.reshape(groups,32,512)[:,slot]+=case['pulses'].repeat(3,1)
    rs=(rz[:,:256].clone(),rz[:,256:].clone());replay={'state':[],'logits':[]}
    for age in range(case['replay_horizon']):
        k=anchor+age;xn=data['x'][k].repeat(groups,1);xo=xn.clone();xo[:,256:768]=data['current_ordinary'][k].repeat(groups,1)
        rs=chunk_step(net,xn,xo,rs);logits=torch.cat([net.pi(rs[0][g*32:(g+1)*32])for g in range(groups)])
        assert torch.equal(torch.cat(rs,1)[:32],data['state_after'][k])
        replay['state'].append(target_rows(torch.cat(rs,1),slot));replay['logits'].append(target_rows(logits,slot))
    replay={k:torch.stack(v)for k,v in replay.items()}
    if data['condition']=='gc100':
        for stream in (closed,replay):
            for key,value in stream.items():assert torch.equal(value[:,:7],value[:,7:14])and torch.equal(value[:,:7],value[:,14:21]),('Grid sham',key)
    return dict(initial_state=initial,initial_position=data['pre']['pos'][anchor,slot],goal=goal0,closed=closed,replay=replay,
        native_fork_checks=checks,grid_sham_exact=data['condition']=='gc100')


def summarize_case(raw,case):
    rows=[];scale=case['scale'];initial=raw['initial_state'];r0=raw['initial_position'];goal=raw['goal']
    for mode_i,mode in enumerate(MODES):
        base=mode_i*7
        for pulse_i,pulse in enumerate(PULSES):
            branch=base+pulse_i
            hit=raw['closed']['hit'][:,branch];hitsteps=torch.nonzero(hit).flatten()
            row=dict(mode=mode,pulse=pulse,slot=case['slot'],anchor=case['anchor'],horizon=case['horizon'],
                first_hit=bool(hit.any()),first_hit_step=int(hitsteps[0])+1 if len(hitsteps)else None,
                fixed_horizon_goal_progress_m=float((r0-goal).norm()-(raw['closed']['pos'][-1,branch]-goal).norm()),
                path_m=float(raw['closed']['path_step'][:,branch].sum()),responses={})
            initial_delta=initial[branch]-initial[base]
            for stream in ('closed','replay'):
                response=[];values=raw[stream]
                for horizon in (1,5,15,30):
                    if horizon>len(values['state']):continue
                    j=horizon-1
                    available=stream=='replay' or bool(raw['closed']['active'][:horizon,branch].all()and raw['closed']['active'][:horizon,base].all())
                    delta=values['state'][j,branch]-values['state'][j,base]
                    actor=values['logits'][j,branch]-values['logits'][j,base];actor-=actor.mean()
                    metrics=dict(full_hc_raw_rms=float(delta.square().mean().sqrt()),full_hc_scaled_rms=float((delta/scale).square().mean().sqrt()),
                        h_rms=float(delta[:256].square().mean().sqrt()),c_rms=float(delta[256:].square().mean().sqrt()),centered_actor_rms=float(actor.square().mean().sqrt()))
                    norm=float((initial_delta/scale).square().mean().sqrt())
                    metrics['scaled_state_gain']=metrics['full_hc_scaled_rms']/norm if norm>0 else None
                    response.append(dict(horizon=horizon,paired_before_terminal=available,**metrics))
                row['responses'][stream]=response
            rows.append(row)
    return rows


@torch.inference_mode()
def run(condition,seed,out):
    start=time.monotonic();out=Path(out);assert not out.exists()or not any(out.iterdir())
    manifest=json.loads(MANIFEST.read_text())
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    folder=ROOT/f'results/native_movement_cycles/{condition}_s{seed}';summary=json.loads((folder/'summary.json').read_text())
    assert sha(folder/'native_trace.pt')==summary['trace_sha256']
    data=torch.load(folder/'native_trace.pt',map_location='cpu',weights_only=True);census=json.loads((folder/'census.json').read_text())
    cfg=data['config'];net=build_policy(cfg,'cpu').eval();ckpt=torch.load(ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(ckpt['policy']);net.requires_grad_(False);digest=state_digest(net);pop=build_population(cfg,'cpu')
    verification=verify_step(net,pop,cfg);cases,missing=select_cases(data,census)
    out.mkdir(parents=True,exist_ok=True)
    serialized=[{k:v for k,v in case.items()if k not in ('scale','pulses')}for case in cases]
    atomic_json(out/'selection.json',dict(episodes=32,selected=serialized,missing=missing,selection_before_any_pulse=True))
    native_prechecks=[verify_selected_native_fork(net,pop,data,case)for case in cases]
    atomic_json(out/'native_only_prechecks.json',dict(complete=True,before_any_counterfactual=True,cases=native_prechecks))
    rows=[];checks=[]
    for i,case in enumerate(cases):
        raw=run_case(net,pop,data,case);path=out/f"slot{case['slot']:02d}.pt";atomic_torch(path,dict(case=case,**raw))
        rows.extend(summarize_case(raw,case));checks.append(raw['native_fork_checks'])
        print(json.dumps(dict(condition=condition,seed=seed,case=i+1,total=len(cases),slot=case['slot'],native_fork_checks=raw['native_fork_checks'],seconds=time.monotonic()-start)),flush=True)
    assert state_digest(net)==digest
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    report=dict(complete=True,condition=condition,seed=seed,episodes=32,eligible=len(cases),missing=missing,rows=rows,
        fork_checks=checks,verification=verification,no_training=True,no_cloud=True,policy_unchanged=True,
        protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(MANIFEST),seconds=time.monotonic()-start,
        raw_assets={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in out.glob('slot*.pt')})
    atomic_json(out/'summary.json',report);print(json.dumps({k:v for k,v in report.items()if k not in ('rows','missing','raw_assets')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=('gc100','pgc33'));p.add_argument('--seed',type=int,choices=(1,2));p.add_argument('--out',type=Path);p.add_argument('--freeze',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    if a.freeze:
        assert not MANIFEST.exists();old=json.loads((ROOT/'docs/research/native_movement_cycles_source_manifest.json').read_text())
        names=set(old['sha256'])|{'scripts/native_movement_pulses.py','tests/test_native_movement_pulses.py',str(PROTOCOL.relative_to(ROOT))}
        for condition in ('gc100','pgc33'):
            for seed in (1,2):
                folder=f'results/native_movement_cycles/{condition}_s{seed}'
                names|={f'{folder}/summary.json',f'{folder}/census.json'}
        atomic_json(MANIFEST,dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sha256={n:sha(ROOT/n)for n in sorted(names)}));print(sha(MANIFEST));return
    assert a.condition and a.seed and a.out;run(a.condition,a.seed,a.out)
if __name__=='__main__':main()
