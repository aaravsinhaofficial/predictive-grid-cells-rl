"""Bounded native-policy movement census. No training or model selection."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from code.common import build_policy,build_population
from code.env import wrap_pi
from lstm_code_swap import MatchedArena,gates_and_state,state_from_gates,summarize_episodes
from rl_goal_memory import atomic_json,atomic_torch,state_digest

PROTOCOL=ROOT/'docs/research/native_movement_cycles_protocol.json'
ADDENDUM=ROOT/'docs/research/native_movement_cycles_views_addendum.json'
MANIFEST=ROOT/'docs/research/native_movement_cycles_source_manifest.json'
SOURCE_NAMES=('scripts/native_movement_cycles.py','tests/test_native_movement_cycles.py',
    'scripts/lstm_code_swap.py','scripts/rl_goal_memory.py','code/__init__.py','code/common.py',
    'code/codes.py','code/env.py','code/nets.py')
ENV_KEYS=('pos','goal','mdir','hd','vf','vs','vr','spawn','t','partial','ep_idx','draw_idx',
          'travelled','ep_return','n_goals','first_goal_t','path_eff_sum','path_eff_n')
PHYSICAL_KEYS=('pos','hd','mdir','vf','vs','vr','t','n_goals','travelled')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TraceArena(MatchedArena):
    """Observe exact physics before native teleport; never change its transition."""
    trace_first=None
    def _respawn(self,mask):
        if self.trace_first is None:
            self.trace_first={key:getattr(self,key).detach().cpu().clone()for key in PHYSICAL_KEYS}
        return super()._respawn(mask)


def current_inputs(net,pop,obs,embedding,goal,pa,pr):
    native=pop.activity(obs['pos'],obs['dir'],obs['vel'])
    ordinary=pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0.)
    assert torch.equal(native[:,~pop.is_pgc],ordinary[:,~pop.is_pgc])
    xn=net._concat(embedding,native,goal,pa,pr,None)
    xo=xn.clone();xo[:,256:768]=ordinary
    assert torch.equal(xn[:,:256],xo[:,:256]) and torch.equal(xn[:,768:],xo[:,768:])
    return xn,xo,native


def intervention_step(net,xn,xo,state,mode):
    assert mode in ('native','current_grid','current_candidate_output_rescue')
    if mode=='native' or torch.equal(xn,xo):return net.lstm(xn,state)
    if mode=='current_grid':return net.lstm(xo,state)
    ordinary,_=gates_and_state(net,xo,state);native,_=gates_and_state(net,xn,state)
    return state_from_gates((ordinary[0],ordinary[1],native[2],native[3]),state[1])


@torch.inference_mode()
def verify_step(net,pop,cfg):
    generator=torch.Generator().manual_seed(889001)
    n=8;rgb=torch.rand(n,3,84,84,generator=generator)
    angle=torch.rand(n,generator=generator)*2*np.pi
    direction=torch.stack((angle.cos(),angle.sin()),1)
    obs=dict(pos=torch.rand(n,2,generator=generator)*2-1,dir=direction,vel=.2*direction)
    goal=pop.activity(torch.rand(n,2,generator=generator)*2-1,direction)
    pa=torch.arange(n)%6;pa[0]=-1;pr=torch.rand(n,generator=generator)
    state=tuple(torch.randn(n,256,generator=generator)*.1 for _ in range(2))
    embedding=net.embed(rgb);xn,xo,code=current_inputs(net,pop,obs,embedding,goal,pa,pr)
    logits,value,expected=net.step(rgb,code,goal,pa,pr,state)
    actual=net.lstm(xn,state)
    error=max(float((a-b).abs().max())for a,b in zip(expected,actual))
    assert error==0 and torch.equal(logits,net.pi(actual[0])) and torch.equal(value,net.v(actual[0]).squeeze(-1))
    gates,reconstructed=gates_and_state(net,xn,state)
    gate_error=max(float((a-b).abs().max())for a,b in zip(expected,reconstructed));assert gate_error<2e-4
    grid=build_population(dict(cfg,cond='gc100',pgc_frac=0.),'cpu')
    assert torch.equal(xo[:,256:768],grid.activity(obs['pos'],obs['dir'],obs['vel']))
    assert int(pop.is_pgc.sum())==(169 if cfg['cond']=='pgc33'else 0)
    if cfg['cond']=='gc100':
        assert torch.equal(xn,xo)
        for mode in ('native','current_grid','current_candidate_output_rescue'):
            assert all(torch.equal(a,b)for a,b in zip(actual,intervention_step(net,xn,xo,state,mode)))
    return dict(original_forward_state_logits_value_max_error=error,manual_gate_max_error=gate_error,
        independent_grid_population_matches_substitution=True,predictive_slots=int(pop.is_pgc.sum()),
        goal_and_noncurrent_inputs_unchanged=True,grid_sham_exact=cfg['cond']=='gc100',
        synthetic_inputs_only=True)


def closure(values):
    values=np.asarray(values,dtype=np.float64)
    radius=float(np.sqrt(np.mean(np.sum((values-values.mean(0))**2,axis=1))))
    distance=float(np.linalg.norm(values[-1]-values[0]))
    return dict(endpoint_distance=distance,rms_radius=radius,closure_over_radius=distance/radius if radius>1e-12 else None)


def census(data):
    pre={k:v.numpy()for k,v in data['pre'].items()};post={k:v.numpy()for k,v in data['post'].items()}
    before=data['state_before'].numpy();after=data['state_after'].numpy()
    hit=data['hit'].numpy();done=data['done'].numpy();T,n=hit.shape
    W=data['actor_weight'].numpy();bias=data['actor_bias'].numpy()
    cycles=[];segments=[]
    for slot in range(n):
        start=0
        while start<T:
            ends=np.flatnonzero(hit[start:,slot]|done[start:,slot]);end=start+int(ends[0])if len(ends)else T-1
            known=bool(pre['n_goals'][start,slot]>0)
            delta=(post['hd'][start:end+1,slot]-pre['hd'][start:end+1,slot]+np.pi)%(2*np.pi)-np.pi
            unwrapped=np.cumsum(delta,dtype=np.float64)
            motion_delta=np.arctan2(post['mdir'][start:end+1,slot,1],post['mdir'][start:end+1,slot,0])-np.arctan2(pre['mdir'][start:end+1,slot,1],pre['mdir'][start:end+1,slot,0])
            motion_delta=(motion_delta+np.pi)%(2*np.pi)-np.pi
            path=np.linalg.norm(post['pos'][start:end+1,slot]-pre['pos'][start:end+1,slot],axis=1)
            levels={1:0,-1:0};previous={1:start,-1:start};events=[]
            for local,theta in enumerate(unwrapped):
                for sign in (1,-1):
                    if sign*theta < (levels[sign]+1)*2*np.pi:continue
                    levels[sign]+=1;finish=start+local;begin=previous[sign];previous[sign]=finish+1
                    states=np.concatenate((before[begin:finish+1,slot],after[finish:finish+1,slot]),axis=0)
                    scale=np.r_[np.full(256,max(np.sqrt(np.mean(states[:,:256]**2)),1e-8)),
                                np.full(256,max(np.sqrt(np.mean(states[:,256:]**2)),1e-8))]
                    actor=states[:,:256]@W.T+bias;actor-=actor.mean(1,keepdims=True)
                    goal=pre['goal'][start,slot];p0=pre['pos'][begin,slot];p1=post['pos'][finish,slot]
                    row=dict(slot=slot,segment_start=start,segment_end=end,known_goal=known,sign=sign,
                        start_step=begin,end_step=finish,steps=finish-begin+1,
                        physical_closure_m=float(np.linalg.norm(p1-p0)),
                        path_m=float(path[begin-start:finish-start+1].sum()),
                        goal_progress_m=float(np.linalg.norm(p0-goal)-np.linalg.norm(p1-goal)),
                        segment_eventual_hit=bool(hit[end,slot]),
                        full_hc_raw=closure(states),full_hc_block_rms=closure(states/scale),
                        hidden_h=closure(states[:,:256]),centered_actor_logits=closure(actor))
                    cycles.append(row);events.append(row)
            repeated=sum(events[i]['sign']==events[i-1]['sign'] and events[i]['start_step']==events[i-1]['end_step']+1 for i in range(1,len(events)))
            goal=pre['goal'][start,slot]
            segments.append(dict(slot=slot,start_step=start,end_step=end,steps=end-start+1,known_goal=known,
                hit=bool(hit[end,slot]),censored_by_episode_end=not bool(hit[end,slot]),
                body_clockwise_turns=levels[-1],body_counterclockwise_turns=levels[1],
                body_turns=sum(levels.values()),repeated_same_sign_pairs=int(repeated),
                body_net_radians=float(unwrapped[-1]),travel_net_radians=float(motion_delta.sum()),
                travel_max_abs_radians=float(np.abs(np.cumsum(motion_delta)).max()),
                native_moving_fraction=float(np.mean(path>data['config'].get('v_min',.05)*data['dt'])),
                path_m=float(path.sum()),goal_progress_m=float(np.linalg.norm(pre['pos'][start,slot]-goal)-np.linalg.norm(post['pos'][end,slot]-goal))))
            start=end+1
    returns=[r for r in segments if r['known_goal']]
    by_cycle={}
    for name,predicate in [('with_fullturn',lambda r:r['body_turns']>0),('without_fullturn',lambda r:r['body_turns']==0)]:
        rows=[r for r in returns if predicate(r)]
        by_cycle[name]=dict(segments=len(rows),hits=sum(r['hit']for r in rows),
            censored=sum(r['censored_by_episode_end']for r in rows),
            mean_steps=float(np.mean([r['steps']for r in rows]))if rows else None,
            mean_progress_m=float(np.mean([r['goal_progress_m']for r in rows]))if rows else None,
            mean_path_m=float(np.mean([r['path_m']for r in rows]))if rows else None)
    return dict(episodes=n,segments=segments,cycles=cycles,total_heading_turns=len(cycles),
        episodes_with_heading_turns=len({r['slot']for r in cycles}),
        return_segments=len(returns),return_segments_with_heading_turns=sum(r['body_turns']>0 for r in returns),
        return_segments_with_repeated_turns=sum(r['repeated_same_sign_pairs']>0 for r in returns),
        physical_usefulness_descriptors=by_cycle,
        interpretation='Heading windings and neural self-similarity are a discovery census, not ring or attraction evidence; cycle-presence behavioral comparisons are descriptive.')


@torch.inference_mode()
def collect(condition,seed,out):
    started=time.monotonic();out=Path(out);assert not out.exists()or not any(out.iterdir())
    source_manifest=json.loads(MANIFEST.read_text())
    for name,digest in source_manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    cfg_path=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
    cfg=json.loads(cfg_path.read_text());ckpt=torch.load(cfg_path.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert ckpt['frames']==1_000_000_000 and cfg['episode_len']==1350 and cfg['extra_input']=='none'
    assert cfg['code_noise']==0 and cfg['goal_code_shift']=='intact' and cfg['img_noise']==.02
    eval_seed=898201+1000*seed;torch.manual_seed(eval_seed)
    net=build_policy(cfg,'cpu').eval();net.load_state_dict(ckpt['policy'],strict=True);net.requires_grad_(False)
    digest=state_digest(net);pop=build_population(cfg,'cpu');verification=verify_step(net,pop,cfg)
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'verification.json',verification)
    atomic_json(out/'protocol.json',dict(protocol_sha256=sha(PROTOCOL),addendum_sha256=sha(ADDENDUM),
        source_manifest_sha256=sha(MANIFEST),condition=condition,seed=seed,eval_seed=eval_seed,
        torch_version=str(torch.__version__),numpy_version=np.__version__,device='cpu',threads=torch.get_num_threads(),
        checkpoint_sha256=sha(cfg_path.with_suffix('.pt')),config_sha256=sha(cfg_path),policy_hash=digest))
    env=TraceArena(cfg,32,1,eval_seed,'cpu');obs=env.reset_all();state=net.zero_state(32,'cpu')
    world=dict(land_pos=env.land_pos.clone(),stripe_phase=env.stripe_phase.clone())
    goal=torch.zeros(32,512);pa=torch.full((32,),-1,dtype=torch.long);pr=torch.zeros(32)
    generator=torch.Generator().manual_seed(eval_seed+1)
    keys=('x','current_ordinary','state_before','state_after','logits','values','action','uniform','hit','done','hit_pos','hit_dir','hit_vel','reward','wall_hit','obs_vel','environment_rng')
    arrays={key:[]for key in keys};pre={key:[]for key in ENV_KEYS};post={key:[]for key in PHYSICAL_KEYS};rows=[]
    for step in range(1350):
        for key in ENV_KEYS:pre[key].append(getattr(env,key).clone())
        arrays['environment_rng'].append(env.gen.get_state().clone());arrays['obs_vel'].append(obs['vel'].clone())
        embedding=net.embed(obs['rgb']);xn,xo,code=current_inputs(net,pop,obs,embedding,goal,pa,pr)
        incoming=torch.cat(state,1);actual=net.lstm(xn,state);logits=net.pi(actual[0]);value=net.v(actual[0]).squeeze(-1)
        if step%135==0:
            baseline=net.step(obs['rgb'],code,goal,pa,pr,state)
            assert torch.equal(logits,baseline[0])and torch.equal(value,baseline[1])and all(torch.equal(a,b)for a,b in zip(actual,baseline[2]))
        uniform=torch.rand(32,generator=generator);action=(uniform[:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        arrays['x'].append(xn.clone());arrays['current_ordinary'].append(xo[:,256:768].clone())
        arrays['state_before'].append(incoming);arrays['state_after'].append(torch.cat(actual,1))
        for key,val in [('logits',logits),('values',value),('action',action),('uniform',uniform)]:arrays[key].append(val.clone())
        env.trace_first=None;obs,reward,done,info=env.step(action)
        assert env.trace_first is not None and torch.equal(env.trace_first['pos'],info['hit_pos'])
        for key in PHYSICAL_KEYS:post[key].append(env.trace_first[key])
        for key,val in [('hit',info['hit']),('done',done),('hit_pos',info['hit_pos']),('hit_dir',info['hit_dir']),
                        ('hit_vel',info['hit_vel']),('reward',reward),('wall_hit',info['hit_wall'])]:arrays[key].append(val.clone())
        hit=info['hit']
        if bool(hit.any()):goal[hit]=pop.activity(info['hit_pos'][hit],info['hit_dir'][hit],info['hit_vel'][hit])
        state=actual;pa=action;pr=reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            assert step==1349 and bool(done.all())
            es={key:val.tolist()for key,val in info['ep_stats'].items()}
            rows=[dict(slot=slot,episode=0,**{k:es[k][i]for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')})for i,slot in enumerate(es['env'])]
            goal.zero_();state=tuple(z*0 for z in state);pa=torch.full_like(action,-1);pr.zero_()
        if(step+1)%150==0:print(json.dumps(dict(condition=condition,seed=seed,step=step+1,total=1350,seconds=time.monotonic()-started)),flush=True)
    assert len(rows)==32 and state_digest(net)==digest
    data={key:torch.stack(values)for key,values in arrays.items()}
    data.update(pre={k:torch.stack(v)for k,v in pre.items()},post={k:torch.stack(v)for k,v in post.items()},
        world=world,config=cfg,condition=condition,seed=seed,dt=env.dt,actor_weight=net.pi.weight.clone(),actor_bias=net.pi.bias.clone(),
        rng_after_collection=env.gen.get_state(),action_rng_after_collection=generator.get_state(),
        source_manifest_sha256=sha(MANIFEST),input_schema='x=[nativeCNNembedding256,current512,nativegoal512,previousaction6,previousreward1]',
        neural_timing='state_before is incoming h/c before current input; state_after is outgoing h/c before physical action/teleport',
        no_optimization=True)
    atomic_torch(out/'native_trace.pt',data)
    result=census(data);atomic_json(out/'census.json',result)
    summary=dict(complete=True,condition=condition,seed=seed,episode_summary=summarize_episodes(rows,1350),
        census={k:v for k,v in result.items()if k not in ('cycles','segments')},
        seconds=time.monotonic()-started,trace_sha256=sha(out/'native_trace.pt'),trace_bytes=(out/'native_trace.pt').stat().st_size,
        census_sha256=sha(out/'census.json'),verification=verification,policy_unchanged=True,no_optimization=True,no_cloud=True)
    for name,expected in source_manifest['sha256'].items():assert sha(ROOT/name)==expected,name
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--condition',choices=('gc100','pgc33'))
    parser.add_argument('--seed',type=int,choices=(1,2));parser.add_argument('--out',type=Path)
    parser.add_argument('--freeze-sources',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    if args.freeze_sources:
        assert not MANIFEST.exists();protocol=json.loads(PROTOCOL.read_text())
        names=set(SOURCE_NAMES)|{str(PROTOCOL.relative_to(ROOT)),str(ADDENDUM.relative_to(ROOT))}|set(protocol['input_assets'])
        for name,digest in protocol['input_assets'].items():assert sha(ROOT/name)==digest
        atomic_json(MANIFEST,dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sha256={n:sha(ROOT/n)for n in sorted(names)}))
        print(json.dumps(dict(manifest=str(MANIFEST),sha256=sha(MANIFEST))));return
    assert args.condition and args.seed and args.out
    collect(args.condition,args.seed,args.out)


if __name__=='__main__':main()
