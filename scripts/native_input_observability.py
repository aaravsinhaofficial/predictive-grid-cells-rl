"""Bounded six-action observation of archived original LSTM states.

Draft implementation: original-network queries require a separately sealed
protocol. Retention is never replaced by a query trajectory.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
import native_turn_geometry as old
from code.common import build_env

BASE=ROOT/'results/native_input_observability'
PLAN=ROOT/'docs/research/native_input_observability_minimal_plan.md'
PROTOCOL=ROOT/'docs/research/native_input_observability_protocol.json'
DRAFT=ROOT/'docs/research/native_input_observability_protocol_draft.json'
DENSE=ROOT/'results/native_turn_geometry_resolution'
WINDOWS={'left':(2,7,14,29,59),'fb_B':(2,14,30,60),'fb_F':(2,14,30,60)}
REFERENCE={'left':0,'fb_B':10,'fb_F':11}
HELDOUT=((0,2,1),(1,3,0),(2,4,3),(3,5,2),(4,0,5),(5,1,4))
SCREEN_CAP=1800;CONDITIONAL_CAP=3600
PROBABILITY_FLOOR=1e-4;ARITHMETIC_TOLERANCE=1e-12;RESPONSE_FLOOR=1e-8
SNAPSHOT_KEYS=('pos','hd','mdir','vf','vs','vr','goal','spawn','t','n_goals','first_goal_t',
    'ep_return','travelled','path_eff_sum','path_eff_n','partial','ep_idx','draw_idx')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def probability_rms(x):return float(np.sqrt(np.mean((np.asarray(x)-np.asarray(x).mean(0))**2)))
def wrap(x):return np.angle(np.exp(1j*x))


def query_bank(base):
    assert base.shape==(1287,)
    bank=base.repeat(6,1).clone();bank[:,1280:1286]=torch.eye(6,dtype=base.dtype);bank[:,-1]=0
    return bank


class ActionObserver:
    """Exact common-input LSTM response; no CNN and no retention updates."""
    def __init__(self,cell,actor_weight,actor_bias,bank):
        self.cell=cell;self.weight=actor_weight.double();self.bias=actor_bias.double();self.bank=bank.double()
        assert len(bank)==6 and self.weight.shape[0]==6
        self.drive=F.linear(self.bank,cell.weight_ih,cell.bias_ih)

    @torch.no_grad()
    def profiles(self,state):
        state=torch.as_tensor(state,dtype=torch.float64);size=self.cell.hidden_size
        h,c=state[:,:size],state[:,size:]
        gates=F.linear(h,self.cell.weight_hh,self.cell.bias_hh)[:,None]+self.drive[None]
        ii,ff,gg,oo=gates.chunk(4,2)
        cell=ff.sigmoid()*c[:,None]+ii.sigmoid()*gg.tanh();hidden=oo.sigmoid()*cell.tanh()
        logits=F.linear(hidden,self.weight,self.bias)
        probabilities=logits.softmax(-1).reshape(len(state),36)
        centered=(logits-logits.mean(-1,keepdim=True)).reshape(len(state),36)
        return probabilities,centered

    @torch.no_grad()
    def reference(self,state):
        state=torch.as_tensor(state,dtype=torch.float64);h,c=state.chunk(2,1);out=[]
        for x in self.bank:
            hh,_=self.cell(x.expand(len(state),-1),(h,c));out.append(F.linear(hh,self.weight,self.bias).softmax(1))
        return torch.stack(out,1).reshape(len(state),36)

    def precision(self,state):
        error=float((self.profiles(state)[0]-self.reference(state)).abs().max())
        assert error<=ARITHMETIC_TOLERANCE,('query arithmetic',error)
        return dict(max_absolute_probability_error=error,tolerance=ARITHMETIC_TOLERANCE,passes=True)

    @torch.no_grad()
    def heldout(self,state,sequences=HELDOUT):
        state=torch.as_tensor(state,dtype=torch.float64);n=len(state);k=len(sequences)
        z=state[:,None,:].expand(n,k,-1).reshape(n*k,-1).clone()
        for step in range(len(sequences[0])):
            actions=torch.tensor([s[step]for s in sequences]).repeat(n)
            z=old.advance(self.cell,self.drive[actions],z)
        q=F.linear(z[:,:self.cell.hidden_size],self.weight,self.bias).softmax(1)
        return q.reshape(n,k,6).numpy()


def geometry(first,last):
    first=np.asarray(first);last=np.asarray(last)
    result=old.geometry_screen(first,last,False)
    result['gates']['initial_probability_RMS_at_least_1e_minus4']=probability_rms(first)>=PROBABILITY_FLOOR
    result['gates']['evolved_probability_RMS_at_least_1e_minus4']=probability_rms(last)>=PROBABILITY_FLOOR
    result['necessary_geometry_pass']=all(result['gates'].values())
    result.update(initial_probability_RMS=probability_rms(first),evolved_probability_RMS=probability_rms(last),
        probability_units='Unscaled36-vector; Euclidean family radius/6 equals per-probability RMS.',
        probability_visibility_floor=PROBABILITY_FLOOR)
    return result


def classify(first,metadata):
    n=metadata['count'];ids=np.asarray(metadata['anchor_indices']);nd=len(metadata['names']);a=len(ids)
    baseline=first[:n];changed=first[n:];radius=old.flow.curve_geometry(baseline)['radius']
    normal,_,closest,segment=old.flow.nearest_closed_segments(changed,baseline)
    anchor=np.broadcast_to(baseline[ids],(nd,2,a,baseline.shape[1])).reshape(changed.shape)
    total=np.linalg.norm(changed-anchor,axis=1);floor=max(RESPONSE_FLOOR,1e-4*radius/6)
    classes=np.where(total/6<floor,0,np.where(normal/6<floor,1,2)).reshape(nd,2,a)
    edges=np.roll(baseline,-1,0)-baseline
    fraction=np.sum((closest-baseline[segment])*edges[segment],1)/np.maximum(np.sum(edges[segment]**2,1),1e-30)
    phase=(segment+fraction)%n;anchor_ids=np.broadcast_to(ids,(nd,2,a)).reshape(-1)
    phase=np.where(classes.reshape(-1)==0,anchor_ids,phase).reshape(nd,2,a)
    # Tangent is a named phase control, never one of the20 normal directions.
    classes[-1]=3
    return dict(classes=classes,phase_labels=phase,normal_probability_RMS=normal.reshape(nd,2,a)/6,
        total_probability_RMS=total.reshape(nd,2,a)/6,visibility_floor=floor,
        class_names=['invisible','phase_only','observable_normal','tangent_control'])


def heldout_closure(first,last,psi0,psi1,metadata):
    """The initial discovery label is fixed before either held-out comparison."""
    classified=classify(first,metadata);n=metadata['count'];labels=classified['phase_labels'].reshape(-1)
    types=classified['classes'].reshape(-1);required=types!=2
    lo=np.floor(labels).astype(int)%n;fraction=labels-np.floor(labels);hi=(lo+1)%n
    arrays={k:v for k,v in classified.items()if isinstance(v,np.ndarray)};checks={}
    # The discovery factor itself must preserve the initial equivalence under
    # retention; a constant held-out profile cannot excuse a wrong phase here.
    families=[('discovery_evolved',np.asarray(last)[:,None,:]),('initial',psi0),('evolved',psi1)]
    for name,profiles in families:
        profiles=np.asarray(profiles);baseline=profiles[:n];actual=profiles[n:]
        expected=(1-fraction[:,None,None])*baseline[lo]+fraction[:,None,None]*baseline[hi]
        error=np.sqrt(np.mean((actual-expected)**2,axis=2))
        angular=np.sqrt(np.mean((baseline-baseline.mean(0))**2,axis=(0,2)))
        tolerance=np.maximum(RESPONSE_FLOOR,.05*angular)
        edge=np.sqrt(np.mean((baseline[hi]-baseline[lo])**2,axis=2))
        interpolation=(edge<=np.maximum(RESPONSE_FLOOR,.10*angular)[None])|(fraction[:,None]==0)
        passed=(error<=tolerance[None])&interpolation
        arrays[name+'_heldout_error_RMS']=error;arrays[name+'_heldout_tolerance']=tolerance
        arrays[name+'_heldout_edge_RMS']=edge;arrays[name+'_interpolation_resolved']=interpolation
        arrays[name+'_heldout_pass']=passed;arrays[name+'_heldout_expected']=expected
        checks[name]=dict(required_signed_kicks=int(required.sum()),heldout_sequences=profiles.shape[1],
            failed_equivalence_cells=int(((error>tolerance[None])&required[:,None]).sum()),
            unresolved_interpolation_cells=int(((~interpolation)&required[:,None]).sum()),
            maximum_required_error_RMS=float(error[required].max())if required.any()else None,
            passes=bool(passed[required].all()))
    counts={name:int((classified['classes']==i).sum())for i,name in enumerate(classified['class_names'])}
    return dict(passes=all(c['passes']for c in checks.values()),checks=checks,class_counts=counts,
        visibility_floor=classified['visibility_floor'],same_initial_phase_label=True,
        interpretation='Phase-only/invisible/control equivalence; observable normals remain separately tested for recovery.'),arrays


def finite_probe(transition,curve,scale,observer,horizon,bounded=None):
    starts,metadata=old.make_perturbations(curve,scale,anchors=8)
    if bounded is not None:starts,metadata=old.bound_perturbations(starts,metadata,scale,bounded)
    states=old.trajectory(transition,starts,horizon,{0,horizon})
    first=observer.profiles(states[0])[0].numpy();last=observer.profiles(states[horizon])[0].numpy()
    result,arrays=old.finite_normal_metrics(first,last,metadata)
    result['perturbation_design']={k:v for k,v in metadata.items()if not isinstance(v,np.ndarray)}
    result['hidden_bound_excess_count']=int((metadata['physical_hidden_bound_excess']>0).sum())
    result['physical_construction_conclusive']=bool(metadata.get('physical_construction_conclusive',result['hidden_bound_excess_count']==0))
    result['initial_observation_conclusive']=old.initially_observable(result)
    result['physical_conclusive']=result['physical_construction_conclusive']and result['initial_observation_conclusive']
    result['physical_status']=('inconclusive_physical_construction'if not result['physical_construction_conclusive']else
        'inconclusive_transverse_observability'if not result['initial_observation_conclusive']else
        'pass'if result['all_normal_directions_pass']else'failed_observable_recovery')
    classes=classify(first,metadata);result['response_class_counts']={name:int((classes['classes']==i).sum())for i,name in enumerate(classes['class_names'])}
    result['response_categories_by_direction']=[dict(name=name,**{label:int((classes['classes'][j]==i).sum())
        for i,label in enumerate(classes['class_names'])})for j,name in enumerate(metadata['names'])]
    result['qualification_scope']='Conservative existing full-normal screen. Invisible or phase-only responses are not counted as recovered normals; insufficient transverse visibility is inconclusive, even with held-out closure.'
    arrays.update(first=first,last=last,starts=starts,final_states=states[horizon],**{k:v for k,v in metadata.items()if isinstance(v,np.ndarray)})
    return result,arrays,metadata


def retention_guard(actual,expected,observer):
    actual=np.asarray(actual);expected=np.asarray(expected);error=np.abs(actual-expected)
    allowed=1e-12+1e-12*np.abs(expected)
    assert actual.shape==expected.shape and bool((error<=allowed).all()),'retention state arithmetic'
    qerror=float((observer.profiles(actual)[0]-observer.profiles(expected)[0]).abs().max())
    assert qerror<=ARITHMETIC_TOLERANCE,('retention probability arithmetic',qerror)
    return dict(max_absolute_state_error=float(error.max()),max_error_over_coordinate_tolerance=float((error/allowed).max()),
        state_absolute_tolerance=1e-12,state_coordinate_relative_tolerance=1e-12,max_absolute_probability_error=qerror,passes=True)


def fast_env(cfg,context,n):
    env=build_env(dict(cfg,img_noise=0.),n,983201,'cpu',stagger=False)
    # Rendering is unconsumed: the recorded black CNN input is used throughout.
    # Physics/reward/reset code remains the original ArenaEnv.step.
    env.render=lambda:torch.zeros((n,3,1,1))
    env.reset_all();env.noise_v=env.noise_r=0.
    env.goal[:]=context['goal_center'];env.land_pos[:]=context['land_pos'];env.stripe_phase[:]=context['stripe_phase']
    for k in old.parent.BOOKKEEPING:getattr(env,k)[:]=context['native_bookkeeping'][k]
    return env


def snapshot(env):return {k:getattr(env,k).clone()for k in SNAPSHOT_KEYS}
def restore(env,state,repeat=1):
    for k,v in state.items():getattr(env,k)[:]=v.repeat_interleave(repeat,0)


def physical_references(cfg,context,family):
    n=len(family['reference_heading']);env=fast_env(cfg,context,n)
    env.pos[:]=family['physical']['pos'][-1];env.hd[:]=family['reference_heading'];env.mdir[:]=family['physical']['mdir'][-1]
    env.spawn[:]=family['physical']['pos'][0];env.t[:]=torch.tensor(family['audit']['reference_episode_t'])
    env.travelled[:]=context['native_bookkeeping']['travelled']+(family['physical']['pos'][1:]-family['physical']['pos'][:-1]).norm(dim=2).sum(0)
    old.parent.brake(env);records={'left':{0:snapshot(env)},'fb_B':{},'fb_F':{}}
    expected={r['updates']:r for r in family['audit']['left_physical']}
    choice=None
    for t in range(1,61):
        old.parent.clean_step(env,torch.zeros(n,dtype=torch.long))
        if t==8:old.parent.brake(env);choice=snapshot(env)
        if t in WINDOWS['left']:records['left'][t]=snapshot(env)
        if t in expected:
            assert torch.equal(env.hd,torch.tensor(expected[t]['heading']))
            assert torch.equal(env.t,torch.tensor(expected[t]['episode_t']))
            assert torch.equal(env.pos,context['endpoint'].expand_as(env.pos))
    assert torch.equal(choice['hd'],family['choice_heading']);restore(env,choice)
    k=1-math.exp(-env.dt/env.tau);speed=env.v_max*k/(2-k)
    env.vf.fill_(-speed);env.vs.zero_();env.vr.zero_()
    for t in range(1,65):
        old.parent.clean_step(env,torch.full((n,),2 if t%2 else 3,dtype=torch.long))
        for regime,origin in [('fb_B',2),('fb_F',3)]:
            if t-origin in (0,)+WINDOWS[regime]:records[regime][t-origin]=snapshot(env)
    assert all(set(rows)=={0,*WINDOWS[regime]}for regime,rows in records.items())
    return records


def physical_tapes(cfg,context,state,bank,sequences):
    n=len(state['hd']);k=len(sequences);env=fast_env(cfg,context,n*k);restore(env,state,k)
    issues=[];input_hashes=[]
    for t in range(len(sequences[0])):
        action=torch.tensor([s[t]for s in sequences]).repeat(n)
        _,reward,done,info=env.step(action)
        for reason,flag in [('goal_hit',info['hit']),('wall_hit',info['hit_wall']),('episode_end',done),('nonzero_reward',reward!=0)]:
            if flag.any():issues.append(dict(step=t+1,reason=reason,phase_sequence_indices=flag.reshape(n,k).nonzero().tolist()))
        x=bank[action].clone();x[:,-1]=reward
        common=torch.equal(x.reshape(n,k,-1),x.reshape(n,k,-1)[:1].expand(n,-1,-1))
        goal_exact=torch.equal(env.goal,state['goal'].repeat_interleave(k,0))
        if not common:issues.append(dict(step=t+1,reason='query_input_not_common'))
        if not goal_exact:issues.append(dict(step=t+1,reason='physical_goal_changed'))
        expected=bank[torch.tensor([s[t]for s in sequences])]
        if not torch.equal(x.reshape(n,k,-1)[0],expected):issues.append(dict(step=t+1,reason='actual_input_differs_from_fixed_query'))
        input_hashes.append(tensor_sha(x))
    return dict(valid=not issues,issues=issues,input_sha256=input_hashes,queries=[list(s)for s in sequences],
        phases=n,measurement_updates=len(sequences[0]),native_physics_unchanged=True,unused_rendering_bypassed=True),snapshot(env)


def physical_utility(cfg,context,terminal):
    """One actual next action after each query, measured before teleport."""
    n=len(terminal['hd']);env=fast_env(cfg,context,n*6);restore(env,terminal,6)
    before=env.pos.clone();goal=env.goal.clone();unit=goal-before;unit/=unit.norm(dim=1,keepdim=True)
    _,reward,done,info=env.step(torch.arange(6).repeat(n))
    if done.any()or info['hit_wall'].any():return dict(valid=False,reason='utility_episode_end_or_wall'),None
    progress=((info['hit_pos']-before)*unit).sum(1).reshape(n,6).numpy()
    return dict(valid=True,preteleport=True,goal_hits=int(info['hit'].sum())),progress


def utility_scores(probability,progress):
    p=np.asarray(probability).reshape(-1,6,6);g=np.asarray(progress).reshape(p.shape)
    expected=np.sum(p*g,2);averaged=np.sum(p.mean(0)[None]*g,2)
    return dict(expected_goalward_mm=float(1000*expected.mean()),phase_averaged_policy_mm=float(1000*averaged.mean()),
        phase_specific_advantage_mm=float(1000*(expected-averaged).mean()),
        antipodal_policy_TV=float((.5*np.abs(p[:len(p)//2]-p[len(p)//2:]).sum(2)).mean()),
        interpretation='One next forced-action expectation, averaged across six query contexts; descriptive, not state erasure.')


def settings():
    return dict(schema=1,study='native_input_observability',conditions=['gc100','pgc33'],endpoints=['original'],
        phases=128,contexts=16,prefixes=[0,1],kinds=list(old.parent.KINDS),branches=list(old.BRANCHES),
        named_families_per_population=256,primary_cases_per_population=3328,diagnostic_profiles_per_primary=2,
        numerical_families={'gc100':128,'pgc33':192},retention_updates={k:list(v)for k,v in WINDOWS.items()},
        reference_updates_after_cue=REFERENCE,measurement_updates=1,heldout_measurement_updates=3,
        discovery_actions=list(range(6)),heldout_sequences=[list(s)for s in HELDOUT],
        profile='Unscaled concatenation of six6-action probability distributions; no division by sqrt(6).',
        initial_and_evolved_probability_RMS_floor=PROBABILITY_FLOOR,probability_arithmetic_tolerance=ARITHMETIC_TOLERANCE,
        retention_state_absolute_tolerance=1e-12,retention_state_coordinate_relative_tolerance=1e-12,
        response_absolute_RMS_floor=RESPONSE_FLOOR,response_relative_radius_fraction=1e-4,
        heldout_RMS_tolerance_fraction=.05,heldout_edge_RMS_fraction=.10,
        heldout_label='Initial discovery label, same baseline continuation; no held-out refitting.',
        qualification_scope='Existing full-normal visibility/recovery gates; no new quotient nomination gate. Invisible/phase-only fibers retain category-specific inconclusive outcomes.',
        normal_seed=old.NORMAL_SEED,normal_directions=20,tangent_controls=1,anchors=8,signs=[1,-1],
        metrics=['raw','h_c_RMS'],amplitude_fraction=.02,bounded_policy='Unchanged shared then clip_equalize policy.',
        screen_cap_seconds=SCREEN_CAP,conditional_cap_seconds=CONDITIONAL_CAP,threads=1,
        retries=False,no_optimization=True,no_cloud=True,
        stop_rule='No discovery-bank expansion, new hidden-profile rescue or continuation training after absent primary phase sensitivity, failed held-out closure or incomplete capped cohort.',
        launch_condition='Both original1024 cohorts complete with zero candidates; any dense candidate instead receives causal validation.')


def make_draft():
    sources=['scripts/native_input_observability.py','tests/test_native_input_observability.py',
        'tests/test_native_input_observability_review.py',str(PLAN.relative_to(ROOT))]
    return dict(**settings(),status='draft_only_no_original_queries',source_sha256={p:sha(ROOT/p)for p in sources})


def freeze():
    assert not PROTOCOL.exists();draft=read(DRAFT);assert draft==make_draft()
    assets={str(old.PROTOCOL.relative_to(ROOT)):sha(old.PROTOCOL)};sources=dict(draft['source_sha256'])
    prior=read(old.PROTOCOL)
    for p,s in prior['source_sha256'].items():assert sha(ROOT/p)==s;sources[p]=s
    for p,s in prior['asset_sha256'].items():assert sha(ROOT/p)==s;assets[p]=s
    for condition in settings()['conditions']:
        for path in (DENSE/condition/'original'/'summary.json',DENSE/condition/'original'/'execution.json',
            old.BASE/condition/'original'/'summary.json',old.BASE/'inputs'/condition/'summary.json'):
            doc=read(path);assert doc['complete'];assets[str(path.relative_to(ROOT))]=sha(path)
            if path.parent.parent==DENSE/condition:
                if path.name=='summary.json':assert doc['individual_candidate_rows']==doc['both_metrics_candidate_rows']==0
                else:assert doc['exit_code']==0
    record=dict(**settings(),source_sha256=sources,asset_sha256=assets,frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    old.atomic_json(PROTOCOL,record);print(sha(PROTOCOL))


def validate_protocol(record):
    assert all(record[k]==v for k,v in settings().items())
    for field in ('source_sha256','asset_sha256'):
        for path,digest in record[field].items():assert sha(ROOT/path)==digest,path


def policy_parts(checkpoint):
    policy=checkpoint['policy'];h=policy['lstm.weight_hh'].shape[1];d=policy['lstm.weight_ih'].shape[1]
    with torch.random.fork_rng(devices=[]):cell=torch.nn.LSTMCell(d,h,dtype=torch.float64)
    cell.load_state_dict({k:policy['lstm.'+k].double()for k in cell.state_dict()});cell.eval().requires_grad_(False)
    return cell,policy['pi.weight'].double(),policy['pi.bias'].double()


def policy_hash(policy):
    digest=hashlib.sha256()
    for name,value in sorted(policy.items()):digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def family_inventory(source):
    families={}
    for r in source['rows']:
        if r['regime']!='left':continue
        item={k:r[k]for k in ('context','prefix','kind','branch','family','exact_sham_of')}
        if r['family']in families:assert families[r['family']]==item
        families[r['family']]=item
    assert len(families)==256
    expected={(c,p,k,b)for c in range(16)for p in (0,1)for k in old.parent.KINDS for b in old.BRANCHES}
    assert {(r['context'],r['prefix'],r['kind'],r['branch'])for r in families.values()}==expected
    return sorted(families.values(),key=lambda r:r['family'])


def physical_label(item):return f"context{item['context']:02d}_prefix{item['prefix']}_{item['kind']}"
def case_key(row):return row['family'],row['regime'],row['retention_updates']
def source_states(folder,source,label,regime):
    name=label+'/'+regime+'/baseline_states.npz';path=folder/name;record=source['artifacts'][name]
    assert sha(path)==record['sha256']and path.stat().st_size==record['bytes']
    data=dict(np.load(path,allow_pickle=False));assert all(v.shape==(128,512)and np.isfinite(v).all()for v in data.values())
    return data,record['sha256']


def prepare_physics(folder,condition,prep,bank,deadline):
    contexts={c['context_id']:c for c in bank['contexts']};records=[];out=folder/'physics';out.mkdir()
    for r in sorted(prep['records'],key=lambda r:(r['context'],r['prefix'],r['kind'])):
        deadline();path=old.BASE/'inputs'/condition/r['path'];assert sha(path)==r['sha256']
        family=load(path);context=contexts[r['context']];label=physical_label(r)
        bank_x=query_bank(family['left_x']);refs=physical_references(bank['config'],context,family)
        probes={};terminals={}
        for regime,states in refs.items():
            probes[regime]={};terminals[regime]={}
            for t,state in states.items():
                deadline();audit,terminal=physical_tapes(bank['config'],context,state,bank_x,tuple((a,)for a in range(6)))
                probes[regime][t]=audit;terminals[regime][t]=terminal
        artifact=out/(label+'.pt')
        old.atomic_torch(artifact,dict(references=refs,one_action_terminals=terminals,query_bank=bank_x,
            left_x=family['left_x'],fb_x=family['fb_x'],probe_checks=probes))
        record=dict(context=r['context'],prefix=r['prefix'],kind=r['kind'],label=label,
            source_input_sha256=r['sha256'],physical_artifact=artifact.name,sha256=sha(artifact),probe_checks=probes)
        old.atomic_json(out/(label+'.json'),record);records.append(record)
        old.atomic_json(folder/'progress.json',dict(stage='physical_eligibility',complete=False,prepared=len(records),expected=64))
        print(json.dumps(dict(condition=condition,prepared_physics=len(records),expected=64)),flush=True)
    assert len(records)==64;old.atomic_json(out/'summary.json',dict(complete=True,records=records,physics_before_neural_screening=True))
    return records


def screen(folder,condition,source,prep,bank,cell,weight,bias,physics,deadline):
    inventory=family_inventory(source);rows=[];canonical={};source_folder=old.BASE/condition/'original'
    physical_records={r['label']:r for r in physics};profile_count=0
    for item in inventory:
        if item['exact_sham_of']:continue
        deadline();label=item['family'];physical=physical_records[physical_label(item)]
        data=load(folder/'physics'/physical['physical_artifact']);assert sha(folder/'physics'/physical['physical_artifact'])==physical['sha256']
        observer=ActionObserver(cell,weight,bias,data['query_bank']);target=folder/'profiles'/label;target.mkdir(parents=True)
        for regime,windows in WINDOWS.items():
            deadline();states,state_sha=source_states(source_folder,source,label,regime);profiles={};precision=[]
            for t in (0,)+windows:
                if not data['probe_checks'][regime][t]['valid']:continue
                q,l=observer.profiles(states[f't{t}']);profiles[t]=(q.numpy(),l.numpy());profile_count+=1
                if t in (0,max(windows)):
                    precision.append(dict(retention_updates=t,**observer.precision(states[f't{t}'][np.arange(8)*16])))
            np.savez_compressed(target/(regime+'.npz'),**{f'{kind}{t}':v for t,pair in profiles.items()for kind,v in zip(('probability_','logit_'),pair)})
            reference=data['references'][regime][0];headings=reference['hd'].numpy()
            for horizon in windows:
                row=dict(condition=condition,endpoint='original',**item,regime=regime,retention_updates=horizon,
                    measurement_updates=1,reference_updates_after_cue=REFERENCE[regime],
                    latest_observation_updates_after_cue=REFERENCE[regime]+horizon+1,
                    retention_seconds=horizon/15,measurement_seconds=1/15,
                    source_state_sha256=state_sha,physical_valid=0 in profiles and horizon in profiles,
                    geometry_qualified=False,broad_candidate=False,precision_checks=precision)
                if row['physical_valid']:
                    first,last=profiles[0][0],profiles[horizon][0];g=geometry(first,last)
                    rotation=wrap(data['references'][regime][horizon]['hd'].numpy()-headings)
                    row.update(primary_geometry=g,geometry_qualified=g['necessary_geometry_pass'],
                        physical_phase=old.physical_phase_comparison(first,last,headings,rotation,False),
                        diagnostic_probability_centered=old.geometry_screen(first,last,True),
                        diagnostic_logit_centered=old.geometry_screen(profiles[0][1],profiles[horizon][1],True))
                else:row['unavailability']={str(t):data['probe_checks'][regime][t]for t in (0,horizon)if t not in profiles}
                rows.append(row);canonical[case_key(row)]=row
                old.atomic_json(target/f'{regime}_t{horizon}.json',row)
        old.atomic_json(target/'summary.json',dict(complete=True,rows=[r for r in rows if r['family']==label]))
        print(json.dumps(dict(condition=condition,screened_family=label,primary_rows=len(rows))),flush=True)
    for item in inventory:
        if not item['exact_sham_of']:continue
        for regime,windows in WINDOWS.items():
            for horizon in windows:
                original=canonical[(item['exact_sham_of'],regime,horizon)]
                rows.append({**original,**item})
    assert len(rows)==3328 and len({case_key(r)for r in rows})==3328
    result=dict(complete=True,condition=condition,rows=rows,named_families=256,
        numerical_families=sum(not r['exact_sham_of']for r in inventory),probability_profile_arrays=profile_count,
        geometry_qualified_rows=sum(r['geometry_qualified']for r in rows),
        numerical_qualified_cases=[list(case_key(r))for r in rows if r['geometry_qualified']and not r['exact_sham_of']],
        unavailable_rows=sum(not r['physical_valid']for r in rows),source_summary_sha256=sha(source_folder/'summary.json'))
    old.atomic_json(folder/'screen_summary.json',result);return result


def conditional(folder,condition,source,bank,cell,weight,bias,physics,screened,deadline):
    contexts={c['context_id']:c for c in bank['contexts']};physical_records={r['label']:r for r in physics}
    selected=[r for r in screened['rows']if r['geometry_qualified']and not r['exact_sham_of']]
    rows=[];physical_cache={};utility_cache={};current_label=None;data=None
    for row in selected:
        deadline();label=row['family'];physical_name=physical_label(row);regime=row['regime'];horizon=row['retention_updates']
        if physical_name!=current_label:
            record=physical_records[physical_name];data=load(folder/'physics'/record['physical_artifact']);current_label=physical_name
        context=contexts[row['context']];observer=ActionObserver(cell,weight,bias,data['query_bank'])
        states,state_sha=source_states(old.BASE/condition/'original',source,label,regime);curve=states['t0']
        inputs=data['left_x'][None]if regime=='left'else(data['fb_x']if regime=='fb_B'else data['fb_x'].flip(0))
        transition=old.transition_from_inputs(SimpleNamespace(lstm=cell),inputs)
        # The retained-state function is the original map, not any query map.
        replay=old.trajectory(transition,curve,horizon,{horizon})[horizon]
        replay_check=retention_guard(replay,states[f't{horizon}'],observer)
        physical_checks={};utilities={}
        for t in (0,horizon):
            cache=(physical_name,regime,t)
            if cache not in physical_cache:
                physical_cache[cache]=physical_tapes(bank['config'],context,data['references'][regime][t],data['query_bank'],HELDOUT)[0]
            physical_checks[str(t)]=physical_cache[cache]
            if cache not in utility_cache:utility_cache[cache]=physical_utility(bank['config'],context,data['one_action_terminals'][regime][t])
            audit,progress=utility_cache[cache]
            utilities[str(t)]=dict(physics=audit,scores=utility_scores(observer.profiles(states[f't{t}'])[0].numpy(),progress)if audit['valid']else None)
        out=folder/'conditional'/label/regime/f't{horizon}';out.mkdir(parents=True)
        result={k:row[k]for k in ('condition','family','context','prefix','kind','branch','regime','retention_updates')}
        result.update(exact_sham_of=None,source_state_sha256=state_sha,retention_replay_check=replay_check,
            heldout_physics=physical_checks,utility=utilities,metrics={},broad_candidate=False)
        heldout_valid=all(v['valid']for v in physical_checks.values())
        for metric in ('raw','h_c_RMS'):
            deadline();scale=np.ones(curve.shape[1])if metric=='raw'else old.state_scales(curve)
            probe,arrays,metadata=finite_probe(transition,curve,scale,observer,horizon)
            probe['archived_baseline_check']=retention_guard(arrays['final_states'][:len(curve)],states[f't{horizon}'],observer)
            name=metric+'_nominal.npz';np.savez_compressed(out/name,**arrays)
            chosen=probe;chosen_arrays=arrays;chosen_meta=metadata;followups=[]
            if probe['hidden_bound_excess_count']:
                for method in ('shared','clip_equalize'):
                    deadline();bounded,ba,bm=finite_probe(transition,curve,scale,observer,horizon,method)
                    bounded['archived_baseline_check']=retention_guard(ba['final_states'][:len(curve)],states[f't{horizon}'],observer)
                    bn=metric+'_'+method+'.npz';np.savez_compressed(out/bn,**ba)
                    followups.append(dict(method=method,result=bounded,arrays=bn));chosen,chosen_arrays,chosen_meta=bounded,ba,bm
                    if bounded['physical_conclusive']:break
            closure=None
            if heldout_valid:
                deadline();psi0=observer.heldout(chosen_arrays['starts']);psi1=observer.heldout(chosen_arrays['final_states'])
                closure,ca=heldout_closure(chosen_arrays['first'],chosen_arrays['last'],psi0,psi1,chosen_meta)
                cn=metric+'_heldout.npz';np.savez_compressed(out/cn,psi_initial=psi0,psi_evolved=psi1,**ca)
                closure['arrays']=cn
            passed=bool(chosen['physical_conclusive']and chosen['all_normal_directions_pass']and closure is not None and closure['passes'])
            result['metrics'][metric]=dict(nominal=probe,nominal_arrays=name,bounded_followups=followups,
                physical_status=chosen['physical_status'],heldout_closure=closure,passes=passed)
        result['broad_candidate']=all(v['passes']for v in result['metrics'].values())
        old.atomic_json(out/'summary.json',dict(complete=True,**result));rows.append(result)
        old.atomic_json(folder/'conditional_progress.json',dict(complete=False,completed_cases=len(rows),expected_cases=len(selected)))
        print(json.dumps(dict(condition=condition,conditional_completed=len(rows),expected=len(selected))),flush=True)
    lookup={(r['family'],r['regime'],r['retention_updates']):r for r in rows}
    for row in screened['rows']:
        if row['geometry_qualified']and row['exact_sham_of']:
            original=lookup[(row['exact_sham_of'],row['regime'],row['retention_updates'])]
            rows.append({**original,**{k:row[k]for k in ('family','context','prefix','kind','branch','exact_sham_of')}})
    assert len(rows)==screened['geometry_qualified_rows']
    result=dict(complete=True,condition=condition,rows=rows,numerical_cases=len(selected),named_cases=len(rows),
        candidates=sum(r['broad_candidate']for r in rows),all_required_cases_retained=True)
    old.atomic_json(folder/'conditional_summary.json',result);return result


def run(condition):
    protocol=read(PROTOCOL);validate_protocol(protocol);folder=BASE/condition
    assert not folder.exists()or not any(folder.iterdir()),'No selective retry.'
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();utc=dt.datetime.now(dt.timezone.utc).isoformat()
    stage='screen';stage_start=time.monotonic();screened=None;finished=None;error=None;code=1
    def deadline():
        cap=SCREEN_CAP if stage=='screen'else CONDITIONAL_CAP
        if time.monotonic()-stage_start>=cap:raise TimeoutError(f'{stage} fixed cap reached')
    def alarm(signum,frame):raise TimeoutError(f'{stage} fixed cap reached')
    prior=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,SCREEN_CAP)
    try:
        source_path=old.BASE/condition/'original'/'summary.json';source=read(source_path)
        prep_path=old.BASE/'inputs'/condition/'summary.json';prep=read(prep_path)
        assert source['complete']and prep['complete']and len(source['rows'])==33280
        assert source['input_summary_sha256']==sha(prep_path)and source['protocol_sha256']==sha(old.PROTOCOL)
        parent_protocol=read(old.parent.PROTOCOL);bank_record=parent_protocol['banks'][condition]
        assert sha(ROOT/bank_record['path'])==bank_record['sha256'];bank=load(ROOT/bank_record['path'])
        assert bank['config']==prep['config']and bank['config']['action_repeat']==4 and bank['config']['episode_len']==1350
        assert len(family_inventory(source))==256
        old.atomic_json(folder/'cohort.json',dict(complete=True,condition=condition,protocol_sha256=sha(PROTOCOL),
            source_summary_sha256=sha(source_path),input_summary_sha256=sha(prep_path),families=family_inventory(source),settings=settings(),
            expected_primary_keys=[[r['family'],regime,t]for r in family_inventory(source)for regime,w in WINDOWS.items()for t in w]))
        physics=prepare_physics(folder,condition,prep,bank,deadline)
        checkpoint_path=old.model_path(condition,'original');assert sha(checkpoint_path)==source['checkpoint_sha256']
        checkpoint=load(checkpoint_path);assert checkpoint['frames']==1_000_000_000
        digest=policy_hash(checkpoint['policy']);assert digest==source['policy_hash_before_float64']
        cell,weight,bias=policy_parts(checkpoint)
        screened=screen(folder,condition,source,prep,bank,cell,weight,bias,physics,deadline);screen_seconds=time.monotonic()-stage_start
        deadline();stage='conditional';stage_start=time.monotonic();signal.setitimer(signal.ITIMER_REAL,CONDITIONAL_CAP)
        finished=conditional(folder,condition,source,bank,cell,weight,bias,physics,screened,deadline);deadline()
        assert policy_hash(checkpoint['policy'])==digest;validate_protocol(protocol)
        artifacts={str(p.relative_to(folder)):dict(sha256=sha(p),bytes=p.stat().st_size)for p in folder.rglob('*')if p.is_file()}
        old.atomic_json(folder/'summary.json',dict(complete=True,condition=condition,endpoint='original',protocol_sha256=sha(PROTOCOL),
            source_summary_sha256=sha(source_path),checkpoint_sha256=sha(checkpoint_path),policy_hash=digest,
            named_families=256,primary_cases=3328,geometry_qualified_rows=screened['geometry_qualified_rows'],
            candidates=finished['candidates'],unavailable_rows=screened['unavailable_rows'],
            screen_seconds=screen_seconds,conditional_seconds=time.monotonic()-stage_start,
            screen_summary_sha256=sha(folder/'screen_summary.json'),conditional_summary_sha256=sha(folder/'conditional_summary.json'),
            artifacts=artifacts,no_optimization=True,no_retention_query_substitution=True,no_cloud=True))
        code=0
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        completed=[list(case_key(read(p)))for p in (folder/'profiles').glob('*/*_t*.json')]
        conditional_completed=[list(case_key(read(p)))for p in (folder/'conditional').glob('*/*/*/summary.json')]
        planned=read(folder/'cohort.json')['expected_primary_keys']if(folder/'cohort.json').exists()else[]
        families=read(folder/'cohort.json')['families']if(folder/'cohort.json').exists()else[]
        aliases={r['family']:r['exact_sham_of']for r in families}
        complete_set={tuple(k)for k in completed}
        completed_named=[k for k in planned if (aliases.get(k[0])or k[0],k[1],k[2])in complete_set]
        complete_named_set={tuple(k)for k in completed_named}
        expected_conditional=screened['numerical_qualified_cases']if screened else[]
        complete_conditional_set={tuple(k)for k in conditional_completed}
        old.atomic_json(folder/'incomplete.json',dict(complete=False,error=error,stage=stage,
            elapsed_stage_seconds=time.monotonic()-stage_start,screen_complete=screened is not None,
            conditional_complete=finished is not None,cohort_retained=True,no_retry=True,
            expected_primary_keys=planned,completed_numerical_primary_keys=completed,
            completed_named_primary_keys=completed_named,
            pending_primary_keys=[k for k in planned if tuple(k)not in complete_named_set],
            expected_numerical_conditional_keys=expected_conditional,
            pending_numerical_conditional_keys=[k for k in expected_conditional if tuple(k)not in complete_conditional_set],
            completed_numerical_conditional_keys=conditional_completed,
            prepared_physical_families=[p.stem for p in (folder/'physics').glob('context*.json')]))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
        old.atomic_json(folder/'execution.json',dict(complete=code==0,exit_code=code,error=error,started_utc=utc,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),seconds=time.monotonic()-started,
            screen_cap_seconds=SCREEN_CAP,conditional_cap_seconds=CONDITIONAL_CAP,protocol_sha256=sha(PROTOCOL),
            summary_sha256=sha(folder/'summary.json')if code==0 else None))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-draft',action='store_true');p.add_argument('--freeze',action='store_true')
    p.add_argument('--condition',choices=('gc100','pgc33'));a=p.parse_args();assert sum((a.write_draft,a.freeze,bool(a.condition)))==1
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if a.write_draft:
            assert not PROTOCOL.exists();old.atomic_json(DRAFT,make_draft());print(str(DRAFT))
        elif a.freeze:freeze()
        else:run(a.condition)


if __name__=='__main__':main()
