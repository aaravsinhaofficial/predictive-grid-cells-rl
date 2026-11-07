"""Saved-data-only report and arithmetic audit of the sealed turn-geometry assay.

No policy/environment is instantiated; no recurrent update or new outcome runs.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import native_turn_geometry as study
from rl_goal_memory import atomic_json

KEYS=('context','prefix','kind','branch','regime','window_updates','evolution_updates','observable','metric','centered')
METADATA=('condition','endpoint','context','prefix','kind','branch','family','exact_sham_of','reference_updates_after_cue','latest_updates_after_cue')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual,expected,path='value'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(path,set(actual)^set(expected))
        for k,v in expected.items():compare(actual[k],v,f'{path}.{k}')
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,b)in enumerate(zip(actual,expected)):compare(a,b,f'{path}[{i}]')
    elif isinstance(expected,(float,np.floating)):
        assert np.isclose(actual,expected,rtol=2e-8,atol=2e-10,equal_nan=False),(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)


def radius(x):return float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))))


def nearest(points,curve):
    """Independent direct polygon projection in every supplied dimension."""
    origin=curve.mean(0);p=points-origin;s=curve-origin;e=np.roll(s,-1,axis=0)-s
    e2=np.sum(e*e,axis=1)
    fraction=np.clip((p@e.T-np.sum(s*e,axis=1))/np.maximum(e2,1e-40),0,1)
    p2=np.sum(p*p,axis=1)[:,None];s2=np.sum(s*s,axis=1)[None]
    distance2=np.maximum(p2+s2-2*p@s.T-2*fraction*(p@e.T-np.sum(s*e,axis=1))+fraction*fraction*e2,0)
    index=distance2.argmin(1);f=fraction[np.arange(len(p)),index]
    closest=curve[index]+f[:,None]*e[index]
    length=np.sqrt(e2);arc=np.r_[0.,np.cumsum(length)]
    phase=np.mod((arc[index]+f*length[index])/max(arc[-1],1e-30),1)
    return np.linalg.norm(points-closest,axis=1),phase


def independent_gates(first,last,center):
    r=radius(first);safe=max(r,1e-30);r1=radius(last)
    length=np.linalg.norm(np.roll(first,-1,axis=0)-first,axis=1);arc=np.r_[0.,np.cumsum(length)];total=arc[-1]
    # Exact all-pairs Euclidean separation; it is geometry, not a PCA score.
    distance2=np.maximum(np.sum(first*first,1)[:,None]+np.sum(first*first,1)[None]-2*first@first.T,0)
    gap=np.abs(arc[:-1,None]-arc[None,:-1]);gap=np.minimum(gap,total-gap)
    distance2[gap<.1*max(total,1e-30)]=np.inf
    nonlocal_distance=np.sqrt(distance2.min(1));nonlocal_distance[~np.isfinite(nonlocal_distance)]=0
    a=first-first.mean(0)if center else first;b=last-last.mean(0)if center else last
    residual,phase=nearest(b,a);ordered=np.sort(phase)
    coverage=1-np.diff(np.r_[ordered,ordered[0]+1]).max()
    delta=((np.roll(phase,-1)-phase)*2*np.pi+np.pi)%(2*np.pi)-np.pi
    winding=delta.sum()/(2*np.pi);backward=(delta/(2*np.pi)<-1e-5).mean()
    return {k:bool(v)for k,v in dict(radius_RMS_at_least_1e_minus4=r/np.sqrt(first.shape[1])>=1e-4,
        max_edge_below_10percent_radius=length.max()/safe<=.1,
        nonlocal_separation_above_5percent=np.quantile(nonlocal_distance/safe,.1)>=.05,
        evolved_radius_retains_5percent=r1/safe>=.05,
        p90_invariance_residual_below_5percent=np.quantile(residual,.9)/safe<=.05,
        phase_coverage_80percent=coverage>=.8,phase_degree_one=abs(winding-1)<.1,
        phase_order_preserved=backward<=.05).items()}


def outputs(states,weights,row):
    offsets=row['observation_offsets'];evolution=row['evolution_updates'];kind=row['observable']
    reference=states['t0'];half=reference.shape[1]//2
    scale=np.ones(reference.shape[1])if row['metric']=='raw'else np.maximum(np.repeat(np.sqrt(np.mean(reference.reshape(-1,2,half)**2,axis=(0,2))),half),1e-8)
    def obs(z):
        if kind=='full_state':return z/scale
        if kind=='all_hidden':return z[:,:half]
        # Static fixed actor calculation from saved states. Never evaluates LSTM.
        logits=torch.nn.functional.linear(torch.from_numpy(z[:,:half]),weights['pi.weight'].double(),weights['pi.bias'].double())
        return (logits-logits.mean(1,keepdim=True)).numpy()
    return np.concatenate([obs(states[f't{t}'])for t in offsets],1),np.concatenate([obs(states[f't{evolution+t}'])for t in offsets],1)


def audit_normal(folder,probe,array_name,metric='raw'):
    """Recompute every finite gain from its archived complete observations."""
    metadata=probe['perturbation_design'];n=metadata['count'];a=metadata['anchors'];nd=len(metadata['names'])
    with np.load(folder/array_name,allow_pickle=False)as z:
        first,last=z['first'],z['last'];ids=np.asarray(metadata['anchor_indices']);shape=(nd,2,a)
        r=max(radius(first[:n]),1e-30)
        d0,_=nearest(first[n:],first[:n]);d1,_=nearest(last[n:],last[:n]);d0=d0.reshape(shape);d1=d1.reshape(shape)
        b0=np.broadcast_to(first[:n][ids],(nd,2,a,first.shape[1])).reshape(first[n:].shape)
        b1=np.broadcast_to(last[:n][ids],(nd,2,a,last.shape[1])).reshape(last[n:].shape)
        total0=np.linalg.norm(first[n:]-b0,axis=1).reshape(shape);total1=np.linalg.norm(last[n:]-b1,axis=1).reshape(shape)
        meaningful=d0>=1e-4*r;visible=total0>=1e-4*r;final_tangent=total1[-1]>=1e-4*r
        leakage=(~visible)&(total1>=1e-4*r)
        ng=d1/np.maximum(d0,1e-30);tg=total1/np.maximum(total0,1e-30)
        ratio=ng/np.maximum(tg[-1].mean(0)[None,None],1e-30)
        for name,value in [('initial_normal_distance',d0),('final_normal_distance',d1),('initial_total_distance',total0),('final_total_distance',total1),
            ('normal_gain',ng),('total_gain',tg),('normal_over_tangent_gain',ratio),('factor_leakage',leakage)]:
            np.testing.assert_allclose(z[name],value,rtol=2e-8,atol=2e-10,err_msg=f'{array_name}:{name}')
        passes=[]
        for j,row in enumerate(probe['directions']):
            valid=meaningful[j]&visible[-1].all(0)[None]&final_tangent.all(0)[None]
            passed=bool(valid.mean()>=.8 and not leakage[j].any()and valid.any()and np.median(ng[j][valid])<=.5 and np.median(ratio[j][valid])<=.5)
            assert row['passes']==passed,(array_name,row['name'])
            passes.append(passed)
        assert probe['all_normal_directions_pass']==all(passes[:-1])
        visible_initial=bool(visible[-1].mean()>=.8 and all(v.mean()>=.8 for v in meaningful[:-1]))
        starts=z['starts'];half=starts.shape[1]//2
        bound_count=int((np.maximum(np.abs(starts[n:,:half])-1,0).max(1)>0).sum())
        assert bound_count==probe['hidden_bound_excess_count']
        if 'bounded_method'in metadata:
            actual=z['actual_physical_kicks'];base=starts[:n][ids]
            np.testing.assert_array_equal(starts[n:].reshape(nd,2,a,-1)-base[None,None],actual)
            assert metric in ('raw','h_c_RMS')
            scale=np.ones(starts.shape[1])if metric=='raw'else np.maximum(np.repeat(
                np.sqrt(np.mean(starts[:n].reshape(n,2,half)**2,axis=(0,2))),half),1e-8)
            scaled_curve=starts[:n]/scale
            tangent=np.roll(scaled_curve,-1,0)-np.roll(scaled_curve,1,0)
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-30)
            tangent=tangent[ids]
            intended=z['scaled_directions']
            np.testing.assert_allclose(intended[-1],tangent,rtol=2e-12,atol=2e-12)
            compare(metadata['full_radius'],radius(scaled_curve),'perturbation radius')
            compare(metadata['amplitude'],.02*radius(scaled_curve),'perturbation amplitude')
            scaled=actual/scale;lengths=np.linalg.norm(scaled,axis=-1)
            unit=scaled/np.maximum(lengths[:,:,:,None],1e-30)
            component=np.sum(unit*tangent[None,None],axis=-1)
            normal_component=np.sqrt(np.maximum(1-component**2,0))
            signed=intended[:,None]*np.array([1.,-1.])[None,:,None,None]
            cosine=np.sum(unit*signed,axis=-1)
            for name,value in [('actual_scaled_norms',lengths),('actual_scaled_directions',unit),
                ('actual_intended_cosine',cosine),('actual_tangent_fraction',component),('actual_normal_fraction',normal_component)]:
                # sqrt(1-dot**2) magnifies last-bit tangent normalization near
                # zero; squared normal fractions preserve the original precision.
                observed,expected=(z[name]**2,value**2)if name=='actual_normal_fraction'else(z[name],value)
                np.testing.assert_allclose(observed,expected,rtol=2e-8,atol=2e-10,err_msg=f'{array_name}:{name}')
            # Reconstruct the bounded construction from the original intended
            # kicks, rather than accepting a stored target norm or multiplier.
            nominal_end=base[None,None]+metadata['amplitude']*signed*scale
            delta=nominal_end-base[None,None]
            if metadata['bounded_method']=='shared':
                dh=np.abs(delta[:,:,:,:half]);margin=np.maximum(1-np.abs(base[:,:half]),0)
                limit=np.divide(margin[None,None],dh,out=np.full_like(dh,np.inf),where=dh>0)
                multiplier=float(min(1.,.9*limit.min()))
                compare(metadata['shared_amplitude_multiplier'],multiplier,'shared amplitude multiplier')
                expected_end=base[None,None]+delta*multiplier
                targets=np.repeat(metadata['amplitude']*multiplier,a)
            else:
                assert metadata['bounded_method']=='clip_equalize'and metadata['shared_amplitude_multiplier']is None
                clipped=base[None,None]+delta
                clipped[:,:,:,:half]=np.clip(clipped[:,:,:,:half],-1,1)
                clipped-=base[None,None]
                original_lengths=np.linalg.norm(clipped/scale,axis=-1)
                targets=original_lengths.min(axis=(0,1))
                factors=np.divide(targets[None,None],original_lengths,out=np.zeros_like(original_lengths),where=original_lengths>0)
                expected_end=base[None,None]+clipped*factors[:,:,:,None]
            np.testing.assert_allclose(starts[n:].reshape(nd,2,a,-1),expected_end,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(metadata['bounded_target_norms'],targets,rtol=2e-10,atol=2e-12)
            np.testing.assert_allclose(lengths,np.broadcast_to(targets,lengths.shape),rtol=2e-8,atol=2e-10)
            guard=np.abs(component)<=.05;guard[-1]=normal_component[-1]<=.05
            assert metadata['actual_geometry_guard_all']==bool(guard.all())
            np.testing.assert_array_equal(z['actual_direction_guard'],guard)
            nonzero=lengths>=max(metadata['full_radius'],1e-30)*1e-12
            block_pure=all(np.count_nonzero(actual[metadata['names'].index(name),:,:,block])==0 for name,block in
                [('hidden_only',slice(half,2*half)),('cell_only',slice(0,half)),('cell_radial',slice(0,half))])
            assert metadata['block_purity_exact']==block_pure
            assert metadata['actual_nonzero_all']==bool(nonzero.all())and metadata['exact_hidden_bounds']==(bound_count==0)
            construct=bool(guard.all()and nonzero.all()and bound_count==0 and block_pure)
            assert metadata['physical_construction_conclusive']==construct
        else:construct=bound_count==0
        conclusive=bool(construct and visible_initial)
        assert probe['physical_conclusive']==conclusive
        status=('pass'if all(passes[:-1])else'failed_recovery')if conclusive else'inconclusive'
        assert probe['physical_status']==status
    return dict(array=array_name,sha256=sha(folder/array_name),directions=nd-1,physical_status=status)



def audit_natural_history(summary,folder,weights):
    records=summary['natural_history_comparison']
    keys=('context','kind','branch','regime','updates','observable')
    lookup={tuple(r[k]for k in keys):r for r in records};assert len(lookup)==len(records)
    route={}
    for row in summary['rows']:
        route[(row['context'],row['kind'],row['branch'],row['regime'],row['prefix'])]=row['exact_sham_of']or row['family']
    count=0
    for context in range(16):
        for kind in study.parent.KINDS:
            for branch in study.BRANCHES:
                for regime in ('left','fb_B','fb_F'):
                    paths=[folder/route[(context,kind,branch,regime,prefix)]/regime/'baseline_states.npz'for prefix in (0,1)]
                    with np.load(paths[0])as a,np.load(paths[1])as b:
                        assert set(a.files)==set(b.files)
                        for stamp in a.files:
                            t=int(stamp[1:]);za,zb=a[stamp],b[stamp];half=za.shape[1]//2
                            for output in ('full_state','all_hidden','all_actor'):
                                def observe(z):
                                    if output=='full_state':return z
                                    if output=='all_hidden':return z[:,:half]
                                    logits=torch.nn.functional.linear(torch.from_numpy(z[:,:half]),weights['pi.weight'].double(),weights['pi.bias'].double())
                                    return(logits-logits.mean(1,keepdim=True)).numpy()
                                aa,bb=observe(za),observe(zb);distance=np.linalg.norm(aa-bb,axis=1)
                                denominator=max(.5*(radius(aa)+radius(bb)),1e-30)
                                row=lookup[(context,kind,branch,regime,t,output)]
                                for name,values in [('same_heading_history_difference',distance),('difference_over_angular_family_radius',distance/denominator)]:
                                    expected=dict(mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
                                    compare(row[name],expected,'natural history')
                                count+=1
    assert count==len(records)
    return count



def validate_inputs(condition):
    folder=study.BASE/'inputs'/condition;path=folder/'summary.json';record=json.loads(path.read_text())
    assert record['complete']and record['condition']==condition and record['protocol_sha256']==sha(study.PROTOCOL)
    assert len(record['records'])==64
    assert {(r['context'],r['prefix'],r['kind'])for r in record['records']}=={(c,p,k)for c in range(16)for p in (0,1)for k in study.parent.KINDS}
    population=study.build_population(record['config'],'cpu')  # Analytic input map only; no policy/environment.
    outside=torch.ones(1287,dtype=torch.bool);outside[256+torch.as_tensor(population.pgc_idx,dtype=torch.long)]=False
    checked=[]
    for row in record['records']:
        p=folder/row['path'];assert sha(p)==row['sha256']
        data=torch.load(p,map_location='cpu',weights_only=True);audit=data['audit'];x=data['native_x'];ordinary=data['ordinary_x']
        assert x.shape==ordinary.shape and x.shape[1:]==(128,1287)
        assert study.parent.tensor_sha(x)==row['input_native_sha256']and study.parent.tensor_sha(ordinary)==row['input_ordinary_sha256']
        assert torch.equal(x[:,:,outside],ordinary[:,:,outside])
        cue=audit['cue_index']
        if cue is None or not len(population.pgc_idx):assert torch.equal(x,ordinary)
        if cue is not None:
            assert cue==len(x)-1 and torch.equal(x[:cue],ordinary[:cue])
            assert torch.equal(ordinary[cue],ordinary[cue,:1].expand_as(ordinary[cue]))
        assert torch.equal(x[:,:,768:1280],x[0,:1,768:1280][None].expand(len(x),128,-1))
        turn=[i for i,name in enumerate(audit['labels'])if name.startswith('turn_')]
        assert len(turn)==14 and torch.equal(x[turn,:,1280:1286].argmax(-1),torch.full((14,128),row['prefix']))
        assert data['fb_x'][:,1280:1286].argmax(1).tolist()==[2,3]
        assert int(data['left_x'][1280:1286].argmax())==0
        for vector,name in ((data['left_x'],'left'),(data['fb_x'][0],'forward'),(data['fb_x'][1],'backward')):
            assert study.parent.tensor_sha(vector)==audit[name+'_input_sha256']
            assert torch.equal(vector[768:1280],x[0,0,768:1280])
        gap=study.wrap_pi(torch.roll(data['reference_heading'],-1)-data['reference_heading'])
        assert float((gap-2*np.pi/128).abs().max())<2e-5
        assert audit['choice_heading_error_max_rad']<2e-5 and audit['endpoint_correction_max_m']<1e-5
        assert max(v['position_error_m']for v in audit['fb_physical']if v['action']==3)<1e-6
        assert max(v['heading_error_rad']for v in audit['fb_physical'])<1e-6
        assert record['config']['episode_len']-min(audit['reference_episode_t'])>=72
        checked.append(dict(context=row['context'],prefix=row['prefix'],kind=row['kind'],sha256=row['sha256']))
    return dict(complete=True,condition=condition,summary_sha256=sha(path),physical_families=64,phases_per_family=128,
        full_input_scope='Native/ordinary equality outside predictive slots, coherent fixed-prefix actions, common goal snapshot and truthful F/B labels.',records=checked)


def validate_endpoint(condition,endpoint):
    folder=study.BASE/condition/endpoint;path=folder/'summary.json';summary=json.loads(path.read_text())
    design=json.loads(study.PROTOCOL.read_text());study.validate_protocol(design)
    assert summary['complete']and summary['condition']==condition and summary['endpoint']==endpoint
    assert summary['protocol_sha256']==sha(study.PROTOCOL)
    assert summary['input_summary_sha256']==sha(study.BASE/'inputs'/condition/'summary.json')
    prefix=json.loads((folder/'prefix_replay.json').read_text())
    assert prefix['complete']and not prefix['cached_original_state_used_as_input']and prefix['original_batch_shape']==32
    assert prefix['require_original_cache']==(endpoint=='original')and len(prefix['contexts'])==16
    if endpoint=='original':assert all(r['original_cache_max_error']==0 for r in prefix['contexts'])
    assert summary['checkpoint_sha256']==sha(study.model_path(condition,endpoint))
    assert summary['contexts']==16 and summary['physical_families']==64 and summary['branches']==4
    rows=summary['rows'];assert len(rows)==33_280
    keys=[tuple(r[k]for k in KEYS)for r in rows];assert len(set(keys))==len(rows)
    expected=set()
    for context in range(16):
        for prefix in (0,1):
            for kind in study.parent.KINDS:
                for branch in study.BRANCHES:
                    for regime in ('left','fb_B','fb_F'):
                        for window,evolution,offsets,output in study.all_specs(regime):
                            for metric in ('raw','h_c_RMS'):
                                for centered in ((False,)if output=='full_state'else(False,True)):
                                    expected.add((context,prefix,kind,branch,regime,window,evolution,output,metric,centered))
    assert set(keys)==expected
    for name,record in summary['artifacts'].items():
        p=folder/name;assert p.stat().st_size==record['bytes']and sha(p)==record['sha256'],p
    lookup={tuple(r[k]for k in KEYS):r for r in rows}
    weights=torch.load(study.model_path(condition,endpoint),map_location='cpu',weights_only=True)['policy']
    raw_cache={};state_cache={};normal_cache={};numeric_rows=0
    for row in rows:
        family=row['exact_sham_of']or row['family'];sub=folder/family/row['regime'];key=str(sub)
        if key not in raw_cache:
            raw=json.loads((sub/'result.json').read_text());assert raw['complete']
            raw_cache[key]={tuple(r[k]for k in ('window_updates','evolution_updates','observable','metric','centered')):r for r in raw['rows']}
        raw=raw_cache[key][tuple(row[k]for k in ('window_updates','evolution_updates','observable','metric','centered'))]
        compare({k:v for k,v in row.items()if k not in METADATA},raw)
        # Exact shams have already been verified against the source record; do
        # not recompute identical high-dimensional arithmetic twice.
        if not row['exact_sham_of']:
            if key not in state_cache:state_cache={key:dict(np.load(sub/'baseline_states.npz',allow_pickle=False))}
            first,last=outputs(state_cache[key],weights,row)
            compare(row['gates'],independent_gates(first,last,row['centered']),'geometry gates')
            assert row['necessary_geometry_pass']==all(row['gates'].values())
            assert row['normal_test_run']==row['necessary_geometry_pass']
            numeric_rows+=1
            if row['normal_test_run']:
                candidates=[(row['normal_result'],row['normal_arrays'])]+[(r['result'],r['arrays'])for r in row['bounded_followups']]
                for probe,name in candidates:
                    cache=(key,name)
                    if cache not in normal_cache:normal_cache[cache]=audit_normal(sub,probe,name,row['metric'])
                physical=candidates[-1][0]
                assert row['physical_status']==physical['physical_status']
                assert row['broad_candidate']==bool(physical['physical_conclusive']and physical['all_normal_directions_pass'])
        assert row['latest_updates_after_cue']==row['reference_updates_after_cue']+row['window_updates']
        assert row['window_updates']==row['evolution_updates']+max(row['observation_offsets'])
        opposite={**row,'metric':'h_c_RMS'if row['metric']=='raw'else'raw'}
        other=lookup[tuple(opposite[k]for k in KEYS)]
        assert row['both_metrics_candidate']==bool(row['broad_candidate']and other['broad_candidate'])
    natural_count=audit_natural_history(summary,folder,weights)
    assert summary['individual_broad_pass_rows']==sum(r['broad_candidate']for r in rows)
    assert summary['both_metrics_candidate_rows']==sum(r['both_metrics_candidate']for r in rows)
    return summary,dict(condition=condition,endpoint=endpoint,complete=True,summary_sha256=sha(path),
        rows=len(rows),unique_keys=len(set(keys)),independently_recomputed_geometry_rows=numeric_rows,
        independently_recomputed_natural_history_rows=natural_count,
        exact_sham_rows=len(rows)-numeric_rows,artifact_hashes_checked=len(summary['artifacts']),
        independently_recomputed_normal_files=len(normal_cache),normal_files=list(normal_cache.values()))


def primary(row):
    return row['window_updates']==(3 if row['regime']=='left'or row['observable']!='full_state'else 2)


def summarize(records):
    grouped=[]
    for summary in records:
        rows=summary['rows']
        for regime in ('left','fb_B','fb_F'):
            for output,center in (('full_state',False),('all_hidden',True),('all_actor',True)):
                selected=[r for r in rows if r['metric']=='raw'and r['regime']==regime and r['observable']==output and r['centered']==center and primary(r)]
                assert len(selected)==256
                grouped.append(dict(condition=summary['condition'],endpoint=summary['endpoint'],regime=regime,observable=output,centered=center,
                    cases=len(selected),physical_candidates=sum(r['both_metrics_candidate']for r in selected),
                    necessary_geometry_pass_raw=sum(r['necessary_geometry_pass']for r in selected),
                    physical_inconclusive_raw=sum(r['physical_status']=='inconclusive'for r in selected),
                    normal_tests_raw=sum(r['normal_test_run']for r in selected),
                    median_fixed_curve_departure=float(np.median([r['residual_to_reference_curve_over_radius']['p90']for r in selected])),
                    max_actor_probability_RMS=max((r.get('actor_probability_reference_RMS',0.)for r in selected),default=0.)))
    return grouped


def render(records,groups,out):
    models=[(r['condition'],r['endpoint'])for r in records]
    labels=[f"{'Grid'if c=='gc100'else'PGC + grid'} · {e}"for c,e in models]
    fig,axes=plt.subplots(1,3,figsize=(10.5,max(3.2,.55*len(models)+1.6)),sharey=True)
    for ax,regime,title in zip(axes,('left','fb_B','fb_F'),('Turning','Forward/back · B','Forward/back · F')):
        values=np.zeros((len(models),3),dtype=int)
        for i,(condition,endpoint)in enumerate(models):
            for j,obs in enumerate(('full_state','all_hidden','all_actor')):
                row=next(r for r in groups if (r['condition'],r['endpoint'],r['regime'],r['observable'])==(condition,endpoint,regime,obs))
                values[i,j]=row['physical_candidates']
                ax.text(j,i,f"{values[i,j]}/256",ha='center',va='center',fontsize=10)
        ax.imshow(values,cmap='Blues',vmin=0,vmax=256,aspect='auto')
        ax.set_xticks(range(3),('Full state','Hidden output','Actor logits'),rotation=20,ha='right')
        ax.set_yticks(range(len(models)),labels);ax.set_title(title,fontsize=12)
        ax.tick_params(length=0);ax.spines[['top','bottom','left','right']].set_visible(False)
    fig.suptitle('Patterns meeting both geometry and physical recovery criteria',fontsize=14)
    fig.text(.5,.01,'Short task window · every context and cue/reset condition · 256 screening cases per cell',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'primary_candidates.png',dpi=180);fig.savefig(out/'primary_candidates.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--endpoints',nargs='+',choices=('original','clean','noise'),default=['original'])
    args=p.parse_args();assert args.endpoints in (['original'],['original','clean','noise'])
    out=study.BASE/('report_original'if args.endpoints==['original']else'report_all');assert not out.exists()or not any(out.iterdir())
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    results=[];audits=[];input_audits=[]
    with threadpool_limits(limits=2):
        for condition in study.parent.CONDITIONS:input_audits.append(validate_inputs(condition))
        for endpoint in args.endpoints:
            for condition in study.parent.CONDITIONS:
                result,audit=validate_endpoint(condition,endpoint);results.append(result);audits.append(audit)
                print(json.dumps(dict(audited=condition,endpoint=endpoint,rows=audit['rows'])),flush=True)
    grouped=summarize(results);render(results,grouped,out)
    atomic_json(out/'summary.json',dict(complete=True,scope=args.endpoints,models=len(results),raw_rows=sum(len(r['rows'])for r in results),
        source_sha256=sha(__file__),geometry_protocol_sha256=sha(study.PROTOCOL),primary=grouped,
        all_window_candidate_rows=[dict(condition=r['condition'],endpoint=r['endpoint'],individual=r['individual_broad_pass_rows'],both_metrics=r['both_metrics_candidate_rows'])for r in results]))
    atomic_json(out/'validation.json',dict(complete=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        mode='Saved arrays, static actor matrix and independent polygon/gain arithmetic only. No LSTM or environment execution.',
        source_sha256=sha(__file__),inputs=input_audits,models=audits,summary_sha256=sha(out/'summary.json')))
    lines=['# Actual turn-history geometry','',
        'We gave the trained networks a full circle of physically consistent headings, using the same turn sequence at every angle. Then every angle received the same future input.',
        '', 'A candidate had to keep a coherent angle-dependent pattern and pull many different disturbances back toward it. We tested all hidden/cell states, all hidden outputs and all actor logits, with both raw and h/c-normalized perturbations.', '',
        '![Short-window candidates](primary_candidates.png)','',
        '| Network | Turning: state / hidden / actor | F/B B-section | F/B F-section |',
        '| --- | --- | --- | --- |']
    for r in results:
        cells=[]
        for regime in ('left','fb_B','fb_F'):
            rr=[next(q for q in grouped if q['condition']==r['condition']and q['endpoint']==r['endpoint']and q['regime']==regime and q['observable']==o)for o in ('full_state','all_hidden','all_actor')]
            cells.append(' / '.join(str(q['physical_candidates'])for q in rr)+' of 256')
        lines.append(f"| {'Grid'if r['condition']=='gc100'else'PGC + grid'} · {r['endpoint']} | {' | '.join(cells)} |")
    lines.extend(['',
        'The256 cases per entry cover16 contexts, two continuous turn histories, two cue modes and four cue/reset conditions. They are screening cases, not256 independent trained networks. Every result, including all longer horizons and both centered/noncentered output tests, remains in the model summaries.', '',
        'Turning uses three updates from the cue boundary. The full-state forward/back test evolves two updates; its two-section output test observes through the third. The forward/back reference begins after the eight-update choice boundary and a truthful setup cycle; B and F are analyzed separately.', '',
        'A forward/back candidate is a memory pattern under a periodic physical drive. Its measured neural phase drift must be compared with the unchanged body heading before calling it stationary heading memory. Turning phase transport is compared with the actual body rotation. A geometric candidate alone does not quantify navigation benefit.', '',
        'Nominal kicks outside the hidden-state bounds remain recorded. Physical qualification uses a smaller shared amplitude or the clipped-and-size-matched follow-up. Zero, invisible or badly reoriented physical kicks are marked inconclusive, rather than failed recovery.', '',
        '[Numerical summary](summary.json) · [Saved-data validation](validation.json)'])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'artifacts.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in out.iterdir()if p.is_file()})
if __name__=='__main__':main()
