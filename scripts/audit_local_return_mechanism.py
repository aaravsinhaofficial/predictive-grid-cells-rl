"""Independent read-only reconciliation of the frozen local-return mechanism.

No task/environment rollouts or new perturbation probes. Saved causal states
are replayed through the exact float32 LSTM, and native FREE physics through
separate NumPy equations. Geometry gates are recalculated from saved summaries;
unsaved observable perturbation samples are explicitly outside this audit.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.env import _TARGETS
from code.common import build_population
import local_goal_return_training as task
import lstm_local_return_family as family_helper
import certify_local_goal_return as certificate
from audit_local_goal_return_baseline import same_tree,policy_hash
from package_local_goal_return_outputs import verify_final_audit

BASE=ROOT/'results/lstm_local_return_mechanism'
ROUTING=ROOT/'docs/research/local_goal_return_training_launch.json'
CONDITIONS=('gc100','pgc33');CUES=('native','ordinary','plus60','minus60','native_reset')
VARIANTS=('blind_native','blind_ordinary','blind_plus60','blind_minus60','blind_reset_after_glimpse')
TIMES=(0,1,5,15,30,45,60);METRICS=('raw','relative_rms')
TRAINING_HASH='0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
MECHANISM_HASH='012825b8278ab571a3a2aa30dee544985c537031d5d113269229ac9d8af4f72a'
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def close(actual,expected,path='',atol=3e-6,rtol=3e-6):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),path
        for k,v in expected.items():close(actual[k],v,path+'/'+str(k),atol,rtol)
    elif isinstance(expected,(list,tuple)):
        assert isinstance(actual,(list,tuple)) and len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):close(a,b,path+'/'+str(i),atol,rtol)
    elif isinstance(expected,(np.ndarray,torch.Tensor)):
        a=np.asarray(actual);b=np.asarray(expected);assert a.shape==b.shape,path
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol),path
    elif expected is None or isinstance(expected,(str,bool)):
        assert actual==expected,path
    elif isinstance(expected,(float,int,np.number)):
        assert math.isfinite(float(actual)) and math.isfinite(float(expected)),path
        assert math.isclose(float(actual),float(expected),abs_tol=atol,rel_tol=rtol),f'{path}: {actual} != {expected}'
    else:assert actual==expected,path


def describe(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))


def radius(x):
    x=np.asarray(x,dtype=float);return float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))))


def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


def fixed_edits(curve,scale,subset):
    """Independent reconstruction of the fixed donor/radial/isotropic edits."""
    curve=np.asarray(curve,dtype=float);scale=np.asarray(scale);subset=np.asarray(subset)
    x=curve/scale;r=radius(x);result={'none':curve[subset].copy()};invalid=[]
    names=('plus30','minus30','radial_plus','radial_minus','isotropic_plus','isotropic_minus')
    if r<1e-12:return result,dict(radius=r,kick_norm=.02*r,invalid=list(names))
    tangent=np.roll(x,-1,0)-np.roll(x,1,0);tn=np.linalg.norm(tangent,axis=1,keepdims=True)
    tangent=tangent/np.maximum(tn,1e-30)
    radial=x-x.mean(0);radial-=np.sum(radial*tangent,1)[:,None]*tangent
    rn=np.linalg.norm(radial,axis=1,keepdims=True);radial/=np.maximum(rn,1e-30)
    rng=np.random.default_rng(188201);isotropic=rng.standard_normal(x.shape)
    isotropic/=np.linalg.norm(isotropic,axis=1,keepdims=True)
    assert len(curve)%12==0
    result['plus30']=curve[(subset+len(curve)//12)%len(curve)]
    result['minus30']=curve[(subset-len(curve)//12)%len(curve)]
    for label,direction in (('radial',radial),('isotropic',isotropic)):
        if label=='radial' and (np.any(rn[subset]<1e-12) or np.any(tn[subset]<1e-12)):
            invalid.extend(('radial_plus','radial_minus'));continue
        for name,sign in (('plus',1),('minus',-1)):
            result[f'{label}_{name}']=(x[subset]+sign*.02*r*direction[subset])*scale
    return result,dict(radius=r,kick_norm=.02*r,invalid=invalid,
        maximum_radial_tangent_dot=float(np.abs(np.sum(radial[subset]*tangent[subset],1)).max()))


def nearest_semantic(points,curve,theta):
    """Direct segment residuals in blocks, independently of producer helpers."""
    points=np.asarray(points,dtype=float);curve=np.asarray(curve,dtype=float);theta=np.asarray(theta)
    best=np.full(len(points),np.inf);phase=np.zeros(len(points))
    edges=np.roll(curve,-1,0)-curve
    for start in range(0,len(curve),32):
        stop=min(start+32,len(curve));e=edges[start:stop]
        difference=points[:,None,:]-curve[None,start:stop,:]
        fraction=np.clip(np.einsum('pjd,jd->pj',difference,e)/np.maximum(np.sum(e*e,1)[None,:],1e-40),0,1)
        residual=difference-fraction[:,:,None]*e[None,:,:]
        d2=np.einsum('pjd,pjd->pj',residual,residual);local=d2.argmin(1);ids=np.arange(len(points))
        values=d2[ids,local];improve=values<best
        j=start+local;closest=curve[j]+fraction[ids,local,None]*edges[j]
        # The frozen semantic map separately reprojects the rounded closest
        # point, with a 1e-30 floor. Tiny edges therefore need this exact step.
        semantic_fraction=np.sum((closest-curve[j])*edges[j],1)/np.maximum(np.sum(edges[j]**2,1),1e-30)
        angle=wrap(theta[j]+semantic_fraction*wrap(theta[(j+1)%len(curve)]-theta[j]))
        best[improve]=values[improve];phase[improve]=angle[improve]
        # Its reported distance is also measured to the rounded closest point.
        if start==0:reported=np.empty(len(points))
        reported[improve]=np.linalg.norm(points[improve]-closest[improve],axis=1)
    return reported,phase


def phase_record(points,correct,own,theta,expected,scale):
    correct=np.asarray(correct);own=np.asarray(own);scale=np.asarray(scale)
    r=radius(correct/scale);own_radius=radius(own/scale)
    distance,phase=nearest_semantic(np.asarray(points)/scale,correct/scale,theta)
    own_distance,own_phase=nearest_semantic(np.asarray(points)/scale,own/scale,theta)
    errors=wrap(phase-expected);ordered=np.sort(np.mod(phase,2*np.pi))
    coverage=1-np.diff(np.r_[ordered,ordered[0]+2*np.pi]).max()/(2*np.pi)
    return dict(phase_estimate_valid=r>=1e-12,
        phase_mae_deg=float(np.degrees(np.abs(errors)).mean()) if r>=1e-12 else None,
        signed_phase_bias_deg=float(np.degrees(np.angle(np.exp(1j*errors).mean()))) if r>=1e-12 else None,
        phase_error_concentration=float(abs(np.exp(1j*errors).mean())) if r>=1e-12 else None,
        phase_coverage=float(coverage) if r>=1e-12 else None,
        off_correct_curve_over_radius=describe(distance/r) if r>=1e-12 else None,
        own_branch_phase_mae_deg=float(np.degrees(np.abs(wrap(own_phase-expected))).mean()) if own_radius>=1e-12 else None,
        own_reference_radius=own_radius,own_reference_collapsed=own_radius<1e-12,
        off_own_curve=describe(own_distance),off_own_curve_over_radius=describe(own_distance/own_radius) if own_radius>=1e-12 else None)


def audit_navigation(rows,report,expected_ids):
    for cue,variant in zip(CUES,VARIANTS):
        selected=[r for r in rows if r['variant']==variant];saved=report[cue]
        assert len(selected)==len(expected_ids) and [r['heading_index'] for r in selected]==list(expected_ids)
        assert saved['rows']==selected
        expected=dict(trials=len(selected),successes=sum(r['hit'] for r in selected),
            success_fraction=sum(r['hit'] for r in selected)/len(selected),
            mean_goal_progress_m=sum(r['progress_m'] for r in selected)/len(selected),
            mean_first_action_goalward_progress_m=sum(r['first_action_goalward_progress_m'] for r in selected)/len(selected),
            mean_path_length_m=sum(r['free_path_m'] for r in selected)/len(selected),rows=selected)
        close(saved,expected,'navigation/'+cue,atol=1e-12,rtol=1e-12)


def audit_contrasts(results,reported):
    assert reported.keys()==results.keys()
    for construction,edits in results.items():
        expected_names={k for k in ('plus30','minus30') if k in edits}
        assert set(reported[construction])==expected_names
        for edit in expected_names:
            baseline=edits['none'];changed=edits[edit];record=reported[construction][edit]
            for metric in METRICS:
                a=baseline['phase']['60']['native'][metric]['phase_mae_deg']
                b=baseline['phase']['60']['ordinary'][metric]['phase_mae_deg']
                c=changed['phase']['60']['native'][metric]['phase_mae_deg']
                d=changed['phase']['60']['ordinary'][metric]['phase_mae_deg']
                direct=None if any(v is None for v in (a,b,c,d)) else d-c
                interaction=None if direct is None else (d-b)-(c-a)
                close(record['direct_native_vs_ordinary_phase_error_reduction_deg'][metric],direct)
                close(record['late_phase_error_reduction_deg'][metric],interaction)
            for key in ('success_fraction','mean_goal_progress_m'):
                a=baseline['navigation']['native'][key];b=baseline['navigation']['ordinary'][key]
                c=changed['navigation']['native'][key];d=changed['navigation']['ordinary'][key]
                close(record['direct_native_vs_ordinary_navigation_gain'][key],c-d)
                close(record['incremental_native_cue_navigation_rescue'][key],(c-a)-(d-b))


def clocks(plan):
    n=len(plan['heading']);assert bool((plan['delay']==60).all())
    a=15+int(plan['duration_a'][0]);probe=a+int(plan['duration_b'][0]);cue=probe+4;free=cue+60;end=free+30
    for key in ('duration_a','duration_b','turn_a','turn_b'):
        assert bool((plan[key]==plan[key][0]).all())
    phase=torch.full((160,n),6,dtype=torch.long);actions=torch.zeros((160,n),dtype=torch.long)
    for label,start,stop in ((0,0,15),(1,15,a),(2,a,probe),(3,probe,cue),(4,cue,free),(5,free,end)):
        phase[start:stop]=label
    actions[15:a]=plan['turn_a'];actions[a:probe]=plan['turn_b'];actions[probe:cue]=2
    return phase,actions,cue,free,end


def audit_trajectory(plan,cfg,raw,rows):
    n=len(plan['heading']);m=5*n;phase,forced,cue,free,end=clocks(plan)
    assert raw['phase'].shape==(160,m) and torch.equal(raw['phase'],phase.repeat(1,5))
    active=raw['active'].bool();mask=raw['policy_mask'].bool();terminal=raw['terminal'].bool()
    assert torch.equal(mask,active&(raw['phase']==5))
    assert torch.equal(raw['reward'],raw['raw_reward'].clamp(-cfg['reward_clip'],cfg['reward_clip']))
    assert bool((raw['raw_reward'][~mask]==0).all()) and bool((raw['pr']==0).all())
    assert bool(((raw['raw_reward']==0)|(raw['raw_reward']==10)).all())
    assert bool((raw['action'][active&~mask]==forced.repeat(1,5)[active&~mask]).all())
    assert not bool(raw['reset_before'].any())
    expected_reset=torch.zeros_like(raw['reset_after']);expected_reset[cue,4*n:]=True
    assert torch.equal(raw['reset_after'],expected_reset)
    assert bool((raw['pa'][0]==-1).all()) and bool((raw['pa'][~active]==-1).all())
    for t in range(1,160):assert torch.equal(raw['pa'][t][active[t]],raw['action'][t-1][active[t]])
    assert set(raw['first_free'])==set(range(m)) and len(rows)==m
    maximum_path_error=maximum_progress_error=0.
    for j,row in enumerate(rows):
        assert (row['variant'],row['heading_index'],row['context'])==(VARIANTS[j//n],int(plan['heading_index'][j%n]),int(plan['context'][j%n]))
        assert not row.get('invalid',False) and int(terminal[:,j].sum())==1
        last=int(terminal[:,j].nonzero()[0]);hit=bool((raw['raw_reward'][:,j]>0).any())
        assert torch.equal(active[:,j],torch.arange(160)<=last)
        assert last>=free and (last<end if hit else last==end-1)
        assert int((raw['raw_reward'][:,j]>0).sum())==int(hit)
        assert row['hit']==hit and row['hit_step']==(last-free+1 if hit else -1)
        assert row['free_steps']==last-free+1==int(mask[:,j].sum())
        first=raw['first_free'][j]
        for field,key in (('entropy','first_free_entropy'),('probabilities','first_free_probabilities'),
            ('logits','first_free_logits'),('value','first_free_value')):
            close(row[key],first[field],atol=1e-12,rtol=1e-12)
        assert torch.equal(first['position'],plan['endpoint'][j%n])
        assert torch.equal(first['input'][768:1280],raw['goal'][j])
        assert torch.equal(first['input'][1280:1286],torch.tensor([1.,0.,0.,0.,0.,0.])) and float(first['input'][-1])==0
        for key in ('position','heading','motors'):assert torch.equal(first[key],raw['first_free'][j%n][key])
        points=torch.cat((first['position'][None],raw['position'][free:last+1,j]),0)
        distance=(points-plan['goal'][j%n]).norm(dim=1);path=(points[1:]-points[:-1]).norm(dim=1).sum()
        maximum_path_error=max(maximum_path_error,abs(float(path)-row['free_path_m']))
        maximum_progress_error=max(maximum_progress_error,abs(float(distance[0]-distance[-1])-row['progress_m']))
        close(row['free_initial_distance_m'],float(distance[0]),atol=2e-6)
        close(row['first_action_goalward_progress_m'],float(distance[0]-distance[1]),atol=2e-6)
        assert bool((distance[:-1]>=cfg['goal_radius']-1e-5).all())
        assert (float(distance[-1])<cfg['goal_radius']+1e-5) if hit else float(distance[-1])>=cfg['goal_radius']-1e-5
        assert row['action_counts']==torch.bincount(raw['action'][free:last+1,j],minlength=6).tolist()
        assert torch.equal(raw['position'][last:,j],raw['position'][last:last+1,j].expand_as(raw['position'][last:,j]))
    assert maximum_path_error<3e-6 and maximum_progress_error<3e-6
    pos=np.stack([raw['first_free'][i]['position'].numpy() for i in range(m)]).astype(float)
    hd=np.array([float(raw['first_free'][i]['heading']) for i in range(m)])
    motors=np.stack([raw['first_free'][i]['motors'].numpy() for i in range(m)]).astype(float)
    tape=plan['motor'].repeat(1,1,5).numpy().astype(float);targets=np.asarray(_TARGETS)*[cfg['v_max'],cfg['v_max'],cfg['rot_max']]
    delta_t=cfg['action_repeat']/60;k=1-math.exp(-delta_t/.25);error=0.
    for t in range(free,end):
        action=raw['action'][t].numpy();use=mask[t].numpy()
        next_motors=motors+k*(targets[action]+tape[t].T*[.03,.03,.15]-motors)
        angle=wrap(hd+next_motors[:,2]*delta_t)
        movement=np.column_stack((next_motors[:,0]*np.cos(angle)-next_motors[:,1]*np.sin(angle),
            next_motors[:,0]*np.sin(angle)+next_motors[:,1]*np.cos(angle)))*delta_t
        attempted=pos+movement;new=np.clip(attempted,-cfg['arena_half']+.08,cfg['arena_half']-.08)
        walls=np.any(attempted!=new,1);next_motors[walls,:2]*=.2
        pos[use]=new[use];hd[use]=angle[use];motors[use]=next_motors[use]
        error=max(error,float(np.max(np.abs(pos-raw['position'][t].numpy()))))
    assert error<2e-5,('Native FREE physics mismatch',error)
    return dict(trials=m,first_hit_masks_rewards_and_rows_reconciled=True,
        native_free_physics_max_abs_error_m=error,path_error_m=maximum_path_error,progress_error_m=maximum_progress_error)


def cell_from_policy(policy,dtype=torch.float32):
    weights={k.removeprefix('lstm.'):v.to(dtype) for k,v in policy.items() if k.startswith('lstm.')}
    with torch.random.fork_rng(devices=[]):
        cell=torch.nn.LSTMCell(weights['weight_ih'].shape[1],weights['weight_hh'].shape[1],dtype=dtype)
    cell.load_state_dict(weights);return cell.eval().requires_grad_(False)


def advance(cell,z,x):return torch.cat(cell(x.expand(len(z),-1) if len(x)==1 else x,tuple(z.chunk(2,1))),1)


def actor(policy,z):
    h=z[:,:policy['pi.weight'].shape[1]]
    return torch.nn.functional.linear(h,policy['pi.weight'].to(z),policy['pi.bias'].to(z))


@torch.no_grad()
def audit_replay(policy,raw,plan,pre,cues,replacement,pgc,condition,controller='policy'):
    n=len(pre);m=5*n;nh=pre.shape[1]//2;cell=cell_from_policy(policy)
    expected=pre.repeat(5,1).float();difference=raw['precue_replay_difference']
    assert difference.shape==expected.shape and float(difference.abs().max())<3e-4
    actual=expected+difference
    for group in range(5):assert torch.equal(actual[:n],actual[group*n:(group+1)*n])
    intended=torch.zeros_like(expected) if replacement is None else replacement.repeat(5,1).float()-expected
    assert torch.equal(raw['intended_precue_edit'],intended),'Incorrect donor/perturbation state'
    expected_inputs=torch.cat([cues[c.replace('_reset','')] for c in CUES])
    x=raw['cue_inputs'];assert x.shape==(m,1287)
    assert float((x-expected_inputs).abs().max())<3e-5
    unchanged=torch.ones(1287,dtype=torch.bool);unchanged[256:768]=~pgc
    for group in range(1,5):assert torch.equal(x[:n,unchanged],x[group*n:(group+1)*n,unchanged])
    assert torch.equal(x[:n],x[4*n:])
    state=actual if replacement is None else replacement.repeat(5,1).float()
    state=advance(cell,state,x);state[4*n:]=0.
    dark=torch.stack([raw['first_free'][i]['input'] for i in range(m)])
    assert torch.equal(dark,dark[:1].expand_as(dark)),'FREE input is not common'
    errors={}
    for t in range(61):
        if t:
            state=advance(cell,state,dark)
        if t in TIMES:
            saved=raw['snapshots'][t];delta=(saved-state).abs()
            assert saved.shape==state.shape and torch.isfinite(saved).all()
            assert bool((delta<=3e-4+2e-6*saved.abs()).all()),f'State replay mismatch at{t}'
            errors[str(t)]=float(delta.max())
    saved=torch.stack([raw['first_free'][i]['state'] for i in range(m)])
    assert torch.equal(saved,raw['snapshots'][60]),'First FREE input was skipped/double processed'
    logits=actor(policy,saved);prob=logits.softmax(1)
    values=torch.nn.functional.linear(saved[:,:nh],policy['v.weight'],policy['v.bias']).flatten()
    _,_,_,free,_=clocks(plan)
    for i in range(m):
        row=raw['first_free'][i]
        close(row['logits'],logits[i],atol=2e-7,rtol=1e-6)
        close(row['value'],float(values[i]),atol=2e-7,rtol=1e-6)
        if controller=='policy':close(row['probabilities'],prob[i],atol=2e-7,rtol=1e-6)
        elif controller=='uniform':assert torch.equal(row['probabilities'],torch.full((6,),1/6))
        else:assert float(row['probabilities'].sum())==1 and int((row['probabilities']==1).sum())==1
        p=row['probabilities'];close(row['entropy'],float(-(p*p.clamp_min(1e-30).log()).sum()),atol=2e-6)
        a=int((plan['action'][free,i%n]>p.cumsum(0)).sum().clamp_max(5))
        assert int(raw['action'][free,i])==a,'First action is inconsistent with saved actor/tape'
    if condition=='gc100':
        for group in range(1,4):
            assert torch.equal(x[:n],x[group*n:(group+1)*n])
            for t in TIMES:assert torch.equal(raw['snapshots'][t][:n],raw['snapshots'][t][group*n:(group+1)*n])
            for key in ('action','position','raw_reward','terminal','active','policy_mask'):
                assert torch.equal(raw[key][:,:n],raw[key][:,group*n:(group+1)*n]),key
    return dict(state_replay_max_abs_error=errors,first_FREE_actor_reconstructed=True,
        PGC_only_input_difference=True,Grid_cue_shams=condition=='gc100',
        scope='Replay saved pre-cue through state60; reconstruct first FREE actor and sampling. Later FREE neural states were not saved; their physics and row arithmetic are independently checked.')


def audit_primary_gates(branch):
    expected={}
    for metric in METRICS:
        q=branch['invariance']['30'][metric];g=q['geometry'];r=q['horizons']['30'];p=r['phase_transport']
        expected[metric]=dict(radius_retains5percent=branch['trajectory']['30'][metric]['radius_retention']>=.05,
            largest_edge_below10percent_radius=g['max_edge_over_radius']<=.1,
            nonlocal_separation_above5percent_radius=g['nonlocal_distance_over_radius']['p10']>=.05,
            two_second_fixed_curve_residual_p90_below5percent_radius=r['residual_to_fixed_curve_over_radius']['p90']<=.05,
            phase_coverage80percent=p['circular_coverage']>=.8,phase_map_degree_one=abs(p['winding_degree']-1)<.1,
            phase_order_preserved=p['backward_edge_fraction']<=.05,radial_gain_below_half=r['radial_normal_gain']['median']<=.5,
            radial_over_tangent_below_half=r['radial_over_tangent_gain']['median']<=.5)
    assert branch['candidate_gates']==expected
    candidate=all(all(row.values()) for row in expected.values())
    assert branch['passes_both_metric_candidate_screen']==candidate
    return candidate


def audit_observable_gates(row,dimensions):
    assert row['full_state_dimensions']==512 and row['observable_dimensions']==dimensions
    assert tuple(row['observation_offsets'])==(0,5,10,15) and row['evolution_updates']==30
    geometry=row['observable_geometry'];r=geometry['radius'];expected={}
    assert set(row['candidate_gates'])=={'centered','noncentered'}
    for label,inv in row['invariance'].items():
        p=inv['phase_transport'];recorded=row['candidate_gates'][label]
        flags=('observable_radial_kick_nontrivial_in80percent','observable_tangent_kick_nontrivial_in80percent')
        assert all(type(recorded[k]) is bool for k in flags)
        expected[label]=dict(observable_radius_RMS_at_least1e_minus4=r/math.sqrt(dimensions)>=1e-4,
            max_edge_below10percent_radius=geometry['max_edge_over_radius']<=.1,
            nonlocal_separation_above5percent=geometry['nonlocal_distance_over_radius']['p10']>=.05,
            evolved_radius_retains5percent=row['evolved_observable_geometry']['radius']/max(r,1e-30)>=.05,
            p90_invariance_residual_below5percent=inv['residual_to_reference_curve_over_radius']['p90']<=.05,
            phase_coverage80percent=p['circular_coverage']>=.8,phase_degree_one=abs(p['winding_degree']-1)<.1,
            phase_order_preserved=p['backward_edge_fraction']<=.05,
            radial_gain_at_most_half=row['radial_observable_normal_gain']['median']<=.5,
            radial_over_tangent_at_most_half=row['radial_over_tangent_observable_gain']['median']<=.5,
            **{k:recorded[k] for k in flags})
        assert recorded==expected[label]
    assert row['centered_conditional_factor_candidate']==all(expected['centered'].values())
    assert row['noncentered_observable_candidate']==all(expected['noncentered'].values())
    return dict(independently_recalculated_thresholds_per_centering=10,provenance_only_flags=list(flags),
        centered=row['centered_conditional_factor_candidate'],noncentered=row['noncentered_observable_candidate'])


def assert_reset_and_physics(sham,raw,n,free):
    assert torch.equal(sham['cue_inputs'],raw['cue_inputs'])
    for key in ('position','action'):assert torch.equal(sham[key][:free],raw[key][:free]),key
    for t in TIMES:assert torch.equal(sham['snapshots'][t][-n:],raw['snapshots'][t][-n:]),'Post-cue reset did not erase edit'
    for key in ('position','action','raw_reward','terminal','active'):
        assert torch.equal(sham[key][:,-n:],raw[key][:,-n:]),'Reset behavior differs across edits'


def assert_edit_definition(saved,expected):
    close(saved['radius'],expected['radius'],atol=1e-12,rtol=1e-12)
    close(saved['kick_norm'],expected['kick_norm'],atol=1e-12,rtol=1e-12)
    assert set(saved['invalid_edits'])==set(expected['invalid'])
    if 'maximum_radial_tangent_dot' in expected:close(saved['maximum_radial_tangent_dot'],expected['maximum_radial_tangent_dot'],atol=1e-12)


def require_clean_context(folder):
    failed=list(Path(folder).glob('*failed.json'))
    assert not failed,f'Unresolved failure evidence: {failed}'


@torch.no_grad()
def audit_causal(folder,record,arrays,family,plan,cfg,policy,track):
    subset=np.arange(0,768,4);theta=arrays['theta'];nh=cfg['nh'];scales=dict(raw=np.ones(512),relative_rms=arrays['scale'])
    assert record['phases']==192 and record['clean_native_float32'] is True
    assert record['frozen_task_sha256']==sha(task.__file__)
    assert set(record['results'])==set(METRICS) and set(record['physical_controls'])=={'uniform','planner'}
    manifests={Path(item['path']).name:item for item in record['raw_files']};assert len(manifests)==len(record['raw_files'])
    for name,item in manifests.items():
        path=folder/'causal_arrays'/name;track(path)
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    assert set(manifests)=={p.name for p in (folder/'causal_arrays').glob('*.pt')}
    source=build_population(cfg,'cpu');pgc=source.is_pgc.cpu().bool()
    refs={cue:{t:arrays[cue+'_states'][TIMES.index(t)] for t in TIMES} for cue in CUES[:-1]}
    cell=cell_from_policy(policy,torch.float64);z=torch.zeros((768,512),dtype=torch.float64);refs['native_reset']={}
    for t in range(61):
        if t:z=advance(cell,z,torch.tensor(arrays['ordinary_input'],dtype=torch.float64))
        if t in TIMES:refs['native_reset'][t]=z.numpy().copy()
    used=set();reports={};reference_chunks={};certificate_first=[];certificate_goal=[]
    for construction,units in scales.items():
        edits,definition=fixed_edits(arrays['pre_cue_state'],units,subset)
        assert set(record['results'][construction])==set(edits)
        reports[construction]={}
        for edit,values in edits.items():
            summary=record['results'][construction][edit];assert_edit_definition(summary['edit_definition'],definition)
            rows=[];collected={cue:{t:[] for t in TIMES} for cue in CUES};checks=[]
            for begin in range(0,192,32):
                name=f'{construction}__{edit}__chunk{begin:03d}.pt';assert name in manifests;used.add(name)
                saved=torch.load(folder/'causal_arrays'/name,map_location='cpu',weights_only=True)
                selected=subset[begin:begin+32];small=family_helper.slice_plan(plan,selected)
                assert same_tree(saved['plan'],small) and saved['edit']==edit and saved['construction']==construction
                raw=saved['raw'];part=saved['rows'];assert summary['collection_checks'][begin//32]==saved['checks']
                check=audit_trajectory(small,cfg,raw,part)
                check.update(audit_replay(policy,raw,small,family['pre_cue_state'][selected],
                    {c:family['cue_inputs'][c][selected] for c in CUES[:-1]},
                    None if edit=='none' else torch.tensor(values[begin:begin+32],dtype=torch.float32),pgc,cfg['cond']))
                if construction=='raw' and edit=='none':
                    reference_chunks[begin]=raw
                    certificate_first.extend(raw['first_free'][128+i] for i in range(32));certificate_goal.append(raw['goal'][128:])
                else:assert_reset_and_physics(reference_chunks[begin],raw,32,clocks(small)[3])
                checks.append(check);rows.extend(part)
                for group,cue in enumerate(CUES):
                    for t in TIMES:collected[cue][t].append(raw['snapshots'][t][group*32:(group+1)*32].numpy())
            audit_navigation(rows,summary['navigation'],subset)
            assert set(summary['phase'])==set(map(str,TIMES))
            for t in TIMES:
                assert set(summary['phase'][str(t)])==set(CUES)
                for cue in CUES:
                    states=np.concatenate(collected[cue][t]);saved=summary['phase'][str(t)][cue]
                    for metric,scale in scales.items():
                        value=phase_record(states,refs['native'][t],refs[cue][t],theta,theta[subset],scale)
                        close(saved[metric],value,f'{construction}/{edit}/{cue}/{t}/{metric}',atol=2e-5,rtol=2e-5)
                    p=actor(policy,torch.tensor(states,dtype=torch.float32)).softmax(1).numpy()
                    p0=actor(policy,torch.tensor(refs[cue][t][subset],dtype=torch.float32)).softmax(1).numpy()
                    close(saved['actor_total_variation_from_own_unperturbed'],describe(.5*np.abs(p-p0).sum(1)))
            reports[construction][edit]=dict(chunks=checks,trials_per_cue=192,phase_and_navigation_reconciled=True)
    for controller,summary in record['physical_controls'].items():
        rows=[]
        for begin in range(0,192,32):
            name=f'control__{controller}__chunk{begin:03d}.pt';assert name in manifests;used.add(name)
            saved=torch.load(folder/'causal_arrays'/name,map_location='cpu',weights_only=True)
            selected=subset[begin:begin+32];small=family_helper.slice_plan(plan,selected);assert same_tree(small,saved['plan'])
            assert summary['checks'][begin//32]==saved['checks']
            audit_trajectory(small,cfg,saved['raw'],saved['rows'])
            audit_replay(policy,saved['raw'],small,family['pre_cue_state'][selected],
                {c:family['cue_inputs'][c][selected] for c in CUES[:-1]},None,pgc,cfg['cond'],controller)
            rows.extend(saved['rows'])
        audit_navigation(rows,summary['navigation'],subset)
    assert used==set(manifests);audit_contrasts(record['results'],record['rescue_contrasts'])
    prepared={new:torch.stack([r[old] for r in certificate_first]).numpy() for new,old in (
        ('first_free_input','input'),('erased_state','state'),('terminal_pos','position'),('terminal_hd','heading'),('terminal_motors','motors'))}
    free=clocks(plan)[3]
    prepared.update(goal_snapshot=torch.cat(certificate_goal).numpy(),motor=plan['motor'][free:free+30,:,subset].numpy(),
        uniform=plan['action'][free:free+30,subset].numpy())
    certified=certificate.certify(prepared,cfg,plan['goal'][0].numpy())
    saved=record['controlled_family_certificate'];close({k:saved[k] for k in certified},certified)
    assert saved['free_start_update']==free and saved['delay_override']==60 and saved['source_sha256']==sha(certificate.__file__)
    return dict(raw_files=len(used),all_chunks_reconciled=True,constructions=reports,
        controlled_family_certificate=certified,contrast_arithmetic_reconciled=True,
        postcue_reset_independent_of_all_precue_edits=True)


@torch.no_grad()
def audit_context(folder,checkpoint,condition,context,protocol_hash,track):
    require_clean_context(folder)
    for name in ('input_plan.pt','collection_attempt.json','family.pt','states.npz','primary.json','observable.json','causal.json'):track(folder/name)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=data['config'];policy=data['policy'];digest=sha(checkpoint)
    plan=torch.load(folder/'input_plan.pt',map_location='cpu',weights_only=True)
    expected,metadata=family_helper.dense_plan(cfg,context);assert same_tree(plan,expected)
    attempt=read(folder/'collection_attempt.json');assert attempt==dict(checkpoint_sha256=digest,protocol_sha256=protocol_hash,
        input_plan_sha256=sha(folder/'input_plan.pt'),plan_metadata=metadata)
    family=torch.load(folder/'family.pt',map_location='cpu',weights_only=True)
    assert family['condition']==condition and family['checkpoint_sha256']==digest and family['plan_metadata']==metadata
    with np.load(folder/'states.npz') as archive:arrays={k:archive[k] for k in archive.files}
    assert str(arrays['checkpoint_sha256'])==digest and np.array_equal(arrays['times'],TIMES)
    assert arrays['theta'].shape==(768,) and arrays['scale'].shape==(512,)
    assert np.array_equal(arrays['theta'],family['theta'].double().numpy())
    assert np.array_equal(arrays['pre_cue_state'],family['pre_cue_state'].double().numpy())
    assert np.array_equal(arrays['ordinary_input'],family['dark_input'].double().reshape(1,-1).numpy())
    assert np.array_equal(arrays['endpoint'],plan['endpoint'][0].numpy()) and np.array_equal(arrays['goal_center'],plan['goal'][0].numpy())
    native=family['starts']['native'].double();scale=np.repeat([float(native[:,:256].square().mean().sqrt()),
        float(native[:,256:].square().mean().sqrt())],256).clip(1e-8)
    assert np.array_equal(arrays['scale'],scale)
    primary=read(folder/'primary.json');observable=read(folder/'observable.json');causal=read(folder/'causal.json')
    for record in (primary,observable,causal):
        assert record['checkpoint_sha256']==digest and record['protocol_sha256']==protocol_hash
        assert record['states_sha256']==sha(folder/'states.npz')
    for record in (primary,causal):assert record['family_sha256']==sha(folder/'family.pt')
    assert primary['condition']==condition and primary['context']==context
    assert primary['collection_audit']==family['audit'] and primary['plan_metadata']==metadata
    for name,d in family['audit']['sources'].items():assert sha(ROOT/name)==d
    assert family['audit']['policy_sha256']==policy_hash(policy)
    assert family['audit']['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    assert family['audit']['common_updates_after_postcue_start']==60
    assert family['audit']['first_free_state_already_processed_first_free_input'] is True
    assert family['audit']['state_noise_sigma']==0
    assert torch.equal(family['phase_indices'],plan['heading_index'])
    velocity=family['nativephysics']['probe_velocity']
    close(family['theta'],torch.atan2(velocity[:,1],velocity[:,0]),atol=1e-7,rtol=0)
    assert set(primary['branches'])==set(CUES[:-1])==set(observable['branches'])
    cell=cell_from_policy(policy,torch.float64);double_replay={};primary_candidates={};observable_candidates={}
    for cue in CUES[:-1]:
        states=arrays[cue+'_states'];assert states.shape==(7,768,512)
        assert np.array_equal(states[0],family['starts'][cue].double().numpy())
        assert np.array_equal(arrays[cue+'_glimpse_inputs'],family['cue_inputs'][cue].double().numpy())
        z=torch.tensor(states[0],dtype=torch.float64);error=0.
        for t in range(1,61):
            z=advance(cell,z,torch.tensor(arrays['ordinary_input'],dtype=torch.float64))
            if t in TIMES:
                delta=float(np.max(np.abs(z.numpy()-states[TIMES.index(t)])));error=max(error,delta)
                assert delta<=2e-9,'Saved double reference trajectory changed'
        double_replay[cue]=error
        native_free=family['first_free_states'][cue].double().numpy();delta=np.abs(native_free-states[-1]);updates=family['audit']['first_free_clock']+1
        assert updates<=100 and np.abs(native_free[:,:256]).max()<=1+1e-6 and np.abs(native_free[:,256:]).max()<=updates+1e-4
        precision=dict(pointwise_atol=3e-4,pointwise_rtol=2e-6,pointwise_pass=bool(np.all(delta<=3e-4+2e-6*np.abs(native_free))),
            absolute_only_3e_minus4_pass=bool(delta.max()<3e-4),max_abs_error=float(delta.max()),clean_native_updates=updates,
            c_amplitude_bound_from_zero=float(updates),blocks={})
        for label,part in (('hidden',slice(0,256)),('cell',slice(256,512))):
            precision['blocks'][label]=dict(max_abs_error=float(delta[:,part].max()),mean_abs_error=float(delta[:,part].mean()),
                native_max_abs_activity=float(np.abs(native_free[:,part]).max()))
        close(primary['branches'][cue]['native_float32_first_free_replay'],precision,atol=1e-12,rtol=1e-12)
        assert precision['pointwise_pass'],'Numerical mismatch remains unresolved'
        primary_candidates[cue]=audit_primary_gates(primary['branches'][cue])
        observable_candidates[cue]={}
        assert set(observable['branches'][cue])=={'all_hidden','learned_action_scores'}
        for name,dimensions in (('all_hidden',1024),('learned_action_scores',24)):
            rows=observable['branches'][cue][name];assert set(rows)=={'raw_full_state_kicks','h_c_RMS_full_state_kicks'}
            observable_candidates[cue][name]={metric:audit_observable_gates(row,dimensions) for metric,row in rows.items()}
    if condition=='gc100':
        for cue in CUES[1:-1]:assert np.array_equal(arrays['native_states'],arrays[cue+'_states'])
    causal_check=audit_causal(folder,causal,arrays,family,plan,cfg,policy,track)
    assert sha(checkpoint)==digest
    return dict(checkpoint_sha256=digest,primary_sha256=sha(folder/'primary.json'),observable_sha256=sha(folder/'observable.json'),
        causal_sha256=sha(folder/'causal.json'),numerically_valid=True,double_reference_replay_max_abs_error=double_replay,
        primary_candidates=primary_candidates,observable_candidates=observable_candidates,causal=causal_check)


def gate_from_saved(report):
    row=report['checks']['stage4_raw_evaluation'];variants=row['variants']
    native=variants['blind_native'];erased=variants['blind_reset_after_glimpse']
    assert native['trials']==erased['trials']==256
    numeric=row['all_physical_checks'] and native['successes']>=167 and native['successes']-erased['successes']>=39
    expected=dict(near_baseline_pass=False,hard_numeric_pass=numeric,
        hard_certified_memory_pass=numeric and row['certificate']['all_pairs_certified'])
    assert row['gates']==expected
    return expected


def run(base=BASE,routing=ROUTING):
    base=Path(base);inputs={};models={};failures=[];pending=[]
    def track(path):
        path=Path(path);assert path.is_file(),f'Missing input: {path}'
        key=str(path.relative_to(ROOT));value=sha(path)
        assert key not in inputs or inputs[key]==value,f'Artifact changed during audit: {path}'
        inputs[key]=value;return path
    protocol_path=track(base/'protocol.json');protocol_hash=sha(protocol_path);assert protocol_hash==MECHANISM_HASH
    protocol=read(protocol_path)
    assert protocol['training_protocol_sha256']==TRAINING_HASH
    for group in ('source_sha256','design_sha256'):
        for name,digest in protocol[group].items():assert sha(track(ROOT/name))==digest,name
    assert protocol['behavior_protocol_sha256']==sha(track(ROOT/'results/local_goal_return_design/behavior_protocol.json'))
    route=read(track(routing));assert route['protocol_sha256']==TRAINING_HASH
    workers=route['workers'];assert len(workers)==2 and {r['condition'] for r in workers}==set(CONDITIONS)
    checks={};checkpoints={}
    for row in workers:
        condition=row['condition'];relative=Path(row['local_artifact_root']);child=Path(row['artifact_subdir'])
        assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[:2]==('results','ring_cloud_artifacts')
        assert not child.is_absolute() and '..' not in child.parts
        folder=ROOT/relative/child;audit_path=ROOT/f'results/local_goal_return_design/cloud_audit/{condition}.json'
        if not audit_path.exists():pending.append(str(audit_path));continue
        report=read(track(audit_path))
        if not report.get('complete'):pending.append(f'{condition}: fixed final audit pending');continue
        verify_final_audit(ROOT,folder,condition,report)
        for name,item in report['artifacts'].items():track(folder/name)
        checkpoints[condition]=track(folder/'final.pt');checks[condition]=gate_from_saved(report)
    result=dict(passed=True,complete=False,status='pending',neural_artifacts_complete=False,
        protocol_sha256=protocol_hash,auditor_sha256=sha(__file__),audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        input_sha256=inputs,models=models,failures=failures,pending=pending,behavior_gate_by_condition=checks,
        scope='No environment/policy rollout or new recovery probe. Saved causal h/c replay through state60 and firstFREE actor; all FREE paths/masks/actions/rewards reconciled independently. Gate arithmetic is recomputed from archived summaries, not perturbation responses. Two observable nontrivial-80% flags are provenance-checked only because individual samples/fractions were not saved. A positive candidate needs nativefloat32 replication and saved/reconciled perturbation samples before interpretation.')
    if pending:return result
    permitted=any(r['hard_certified_memory_pass'] for r in checks.values());result['behavior_gate_passed']=permitted
    saved_gate=read(track(base/'behavior_gate.json'))
    assert saved_gate['gate']['passed']==permitted and saved_gate['gate']['by_condition']==checks
    for c in CONDITIONS:assert saved_gate['checkpoints'][c]['sha256']==sha(checkpoints[c])
    if not permitted:
        result.update(complete=True,status='not_run_behavior_gate');return result
    for condition in CONDITIONS:
        models[condition]={}
        for context in (0,1):
            folder=base/condition/f'context{context}'
            try:models[condition][str(context)]=audit_context(folder,checkpoints[condition],condition,context,protocol_hash,track)
            except (AssertionError,KeyError,ValueError,FileNotFoundError,RuntimeError) as error:
                evidence=[str(p.relative_to(ROOT)) for p in folder.glob('*failed.json')]
                for name in evidence:track(ROOT/name)
                item=dict(condition=condition,context=context,error_type=type(error).__name__,error=str(error),failure_evidence=evidence)
                failures.append(item);models[condition][str(context)]=dict(complete=False,**item)
    # Detect concurrent source/artifact mutation after the final expensive read.
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,f'Input changed: {name}'
    complete=not failures;result.update(passed=complete,complete=complete,neural_artifacts_complete=complete,
        status='complete' if complete else 'incomplete_or_failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--routing',type=Path,default=ROUTING);p.add_argument('--out',type=Path,default=BASE/'independent_artifact_audit.json')
    args=p.parse_args();torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):report=run(args.base,args.routing)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','complete','status','neural_artifacts_complete')}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
