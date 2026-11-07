"""Read-only arithmetic/provenance audit of the separately sealed resolution study.

Only saved tensors, static actor multiplication and numerical geometry are used.
No policy, environment, LSTM transition or new outcome is instantiated/executed.
"""
from __future__ import annotations
import argparse
import collections
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
import native_turn_geometry_resolution as design
import report_native_turn_geometry as base
from rl_goal_memory import atomic_json

KEYS=design.KEYS
SPACING='max_edge_below_10percent_radius'
STATUSES=('sampling_unresolved','geometry_failed','physical_test_inconclusive','recovery_failed','screening_candidate')
LABELS=('Sampling unresolved','Other geometry failed','Physical kick inconclusive','Recovery failed','Screening candidate')
COLORS=('#bdbdbd','#cf9c77','#d7c668','#ca6e72','#4b9f84')
sha=base.sha
compare=base.compare


def read(path):return json.loads(Path(path).read_text())
def key(row):return tuple(row[k]for k in KEYS)
def physical_key(row):return row['context'],row['prefix'],row['kind']
def subkey(row):return tuple(row[k]for k in ('regime',)+design.ROWKEY)
def describe(x):
    x=np.asarray(x)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))


def selection(rows):
    """Independently reconstruct selection, including companions and exclusions."""
    lookup={key(r):r for r in rows};assert len(lookup)==len(rows)
    direct={k for k,r in lookup.items()if {n for n,v in r['gates'].items()if not v}=={SPACING}}
    selected=set(direct)
    for k in direct:
        partner=dict(lookup[k]);partner['metric']='h_c_RMS'if partner['metric']=='raw'else'raw'
        assert key(partner)in lookup;selected.add(key(partner))
    chosen=[r for r in rows if key(r)in selected];numeric=[r for r in chosen if not r['exact_sham_of']]
    excluded=collections.Counter(tuple(sorted(n for n,v in r['gates'].items()if not v))for r in rows if key(r)not in selected)
    cohort=dict(source_rows=len(rows),spacing_only_rows=len(direct),mandatory_metric_control_rows=len(selected-direct),
        selected_rows=len(chosen),numerical_rows=len(numeric),exact_sham_rows=len(chosen)-len(numeric),
        excluded_rows=len(rows)-len(chosen),physical_families=len({physical_key(r)for r in numeric}),
        branch_families=len({r['family']for r in numeric}),regime_families=len({(r['family'],r['regime'])for r in numeric}),
        source_keys=[list(key(r))for r in chosen],spacing_only_keys=[list(k)for k in sorted(direct)],
        exclusion_failure_sets=[dict(gates=list(k),count=v)for k,v in sorted(excluded.items())])
    return chosen,cohort


def state_guard(a,b):
    """Coordinate and angular-pattern tolerances, independently per h/c block."""
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and a.ndim==2 and a.shape[1]%2==0
    blocks={};half=a.shape[1]//2
    for name,sl in [('hidden',slice(0,half)),('cell',slice(half,None))]:
        x,y=a[:,sl],b[:,sl];error=x-y;limits=1e-8+1e-5*np.max(np.abs(y),0)
        yc=y-y.mean(0);shape=x-x.mean(0)-yc;angular=np.sqrt(np.mean(yc**2));sr=np.sqrt(np.mean(shape**2))
        sm=np.max(np.sqrt(np.mean(shape**2,1)));limit=1e-8+.001*angular;mean=x.mean(0)-y.mean(0)
        blocks[name]=dict(max_absolute_error=float(np.abs(error).max()),total_error_RMS=float(np.sqrt(np.mean(error**2))),
            max_error_over_coordinate_tolerance=float(np.max(np.abs(error)/limits)),coordinatewise_pass=bool(np.all(np.abs(error)<=limits)),
            angular_RMS=float(angular),centered_difference_RMS=float(sr),centered_difference_max_row_RMS=float(sm),
            common_mean_shift_RMS=float(np.sqrt(np.mean(mean**2))),common_mean_shift_max_absolute=float(np.abs(mean).max()),
            allowed_centered_difference_max_row_RMS=float(limit),centered_max_row_RMS_over_tolerance=float(sm/limit),
            centered_RMS_over_angular_RMS_with_floor=float(sr/max(angular,1e-8)),angular_shape_pass=bool(sm<=limit))
    return dict(passes=all(v['coordinatewise_pass']and v['angular_shape_pass']for v in blocks.values()),
        coordinate_absolute_tolerance=1e-8,coordinate_relative_tolerance=1e-5,angular_shape_fraction=.001,
        angular_shape_absolute_floor=1e-8,blocks=blocks)


def input_guard(dense,coarse):
    assert dense['audit']['phases']==1024 and coarse['audit']['phases']==128
    comparisons={}
    def add(name,a,b,tolerance=0.):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        error=float(np.abs(a-b).max())if a.size else 0.
        comparisons[name]=dict(max_absolute_error=error,allowed_error=tolerance,passes=error<=tolerance)
    for field in ('native_x','ordinary_x'):
        a=dense[field][:,::8].numpy();b=coarse[field].numpy()
        add(field+'_CNN',a[:,:,:256],b[:,:,:256],1e-5);add(field+'_nonvisual_exact',a[:,:,256:],b[:,:,256:])
    for field in ('target_choice_heading','choice_heading','reference_heading'):add(field,dense[field][::8],coarse[field])
    for field in ('left_x','fb_x'):add(field,dense[field],coarse[field])
    for field in coarse.get('physical',{}):add('physical_'+field,dense['physical'][field][:,::8],coarse['physical'][field])
    assert dense['audit']['labels']==coarse['audit']['labels']
    return dict(complete=True,passes=all(v['passes']for v in comparisons.values()),comparisons=comparisons,
        method='Actual dense-history samples at shared physical headings; no input or state interpolation.')


def polygon_projection(points,curve):
    """Direct nearest-segment calculation, chunked to bound temporary memory."""
    origin=curve.mean(0);c=curve-origin;e=np.roll(c,-1,0)-c;e2=np.sum(e*e,1)
    distance=[];indices=[];fractions=[]
    for start in range(0,len(points),128):
        p=points[start:start+128]-origin;dot=p@e.T-np.sum(c*e,1)
        fraction=np.clip(dot/np.maximum(e2,1e-40),0,1)
        d2=np.maximum(np.sum(p*p,1)[:,None]+np.sum(c*c,1)-2*p@c.T-2*fraction*dot+fraction**2*e2,0)
        index=np.argmin(d2,1);frac=fraction[np.arange(len(p)),index]
        closest=curve[index]+frac[:,None]*e[index]
        distance.extend(np.linalg.norm(points[start:start+128]-closest,axis=1));indices.extend(index);fractions.extend(frac)
    return np.asarray(distance),np.asarray(indices),np.asarray(fractions)


def polygon_convergence(a,b,c,d,center=False):
    radius=max(base.radius(c),1e-30);records={}
    for name,coarse,fine in [('initial',a,c),('evolved',b,d)]:
        if center:coarse=coarse-coarse.mean(0);fine=fine-fine.mean(0)
        distance=polygon_projection(fine,coarse)[0]/radius
        records[name]=dict(max_distance_over_initial_radius=float(distance.max()),p90_distance_over_initial_radius=float(np.quantile(distance,.9)))
    return dict(tolerance=.005,passes=all(r['max_distance_over_initial_radius']<=.005 for r in records.values()),comparisons=records,
        method='Distance of genuinely simulated 1024 states to 512-segment polygons; centered only for the declared centered observable.')


def scales(curve,metric):
    if metric=='raw':return np.ones(curve.shape[1])
    assert metric=='h_c_RMS';half=curve.shape[1]//2
    return np.repeat(np.maximum(np.sqrt(np.mean(curve.reshape(-1,2,half)**2,axis=(0,2))),1e-8),half)


def tangent_check(reference,metric):
    ts=[]
    for stride in (2,1):
        p=reference[::stride];p=p/scales(p,metric)
        t=np.roll(p,-1,0)-np.roll(p,1,0);t=t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-30)
        ts.append(t[np.arange(8)*len(p)//8])
    cos=np.sum(ts[0]*ts[1],1)
    return dict(cosines=cos.tolist(),minimum_cosine=float(cos.min()),mean_cosine=float(cos.mean()),
        interpretation='Descriptive agreement of physical perturbation tangent directions at the same eight angles.')


def nominal_design(curve,metric):
    """Rebuild fixed seed, full-circle anchors and all block-pure directions."""
    n,d=curve.shape;half=d//2;ids=np.arange(8)*n//8;scale=scales(curve,metric);p=curve/scale
    t=np.roll(p,-1,0)-np.roll(p,1,0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-30);t=t[ids]
    rng=np.random.default_rng(984701);vectors=[];names=[]
    def add(v,name,block=None):
        if block is None:v-=np.sum(v*t,1,keepdims=True)*t
        else:
            tb=t[:,block];v[:,block]-=np.sum(v[:,block]*tb,1,keepdims=True)*tb/np.maximum(np.sum(tb**2,1,keepdims=True),1e-30)
        vectors.append(v);names.append(name)
    for i in range(16):add(rng.normal(size=(8,d)),f'random_full_{i:02d}')
    for name,block in [('hidden_only',slice(0,half)),('cell_only',slice(half,d))]:
        v=np.zeros((8,d));v[:,block]=rng.normal(size=(8,half));add(v,name,block)
    radial=p[ids]-p.mean(0);add(radial.copy(),'full_radial')
    radial[:,:half]=0;add(radial,'cell_radial',slice(half,d))
    vectors.append(t.copy());names.append('tangent');vectors=np.stack(vectors);norms=np.linalg.norm(vectors,axis=-1)
    vectors/=np.maximum(norms[:,:,None],1e-30);r=base.radius(p);amp=.02*r
    starts=np.concatenate((curve,(curve[ids][None,None]+amp*vectors[:,None]*np.array([1.,-1.])[None,:,None,None]*scale).reshape(-1,d)))
    return starts,dict(count=n,anchors=8,anchor_indices=ids.tolist(),names=names,full_radius=r,amplitude=amp,
        direction_norms=norms.tolist(),scaled_directions=vectors,normal_dot_tangent=np.sum(vectors[:-1]*t[None],2))


def audit_probe(folder,probe,name,curve,metric,first,last,weights,row):
    """Bind the existing independent gain audit to actual prescribed kicks."""
    expected,metadata=nominal_design(curve,metric);stored=probe['perturbation_design']
    for field,value in metadata.items():
        if not isinstance(value,np.ndarray):compare(stored[field],value,field)
    with np.load(folder/name,allow_pickle=False)as data:
        for field in ('scaled_directions','normal_dot_tangent'):
            np.testing.assert_allclose(data[field],metadata[field],rtol=2e-12,atol=2e-12)
        n=len(curve);np.testing.assert_array_equal(data['starts'][:n],curve)
        if 'bounded_method'not in stored:np.testing.assert_array_equal(data['starts'],expected)
        np.testing.assert_allclose(data['first'][:n],first,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(data['last'][:n],last,rtol=2e-12,atol=2e-12)
        zero=observe(data['starts'],weights,row['observable'],scales(curve,metric))
        np.testing.assert_allclose(data['first'][:,:zero.shape[1]],zero,rtol=2e-12,atol=2e-12)
    return base.audit_normal(folder,probe,name,metric)


def observe(z,weights,kind,scale):
    half=z.shape[1]//2
    if kind=='full_state':return z/scale
    if kind=='all_hidden':return z[:,:half]
    assert kind=='all_actor'
    logits=torch.nn.functional.linear(torch.from_numpy(z[:,:half]),weights['pi.weight'].double(),weights['pi.bias'].double())
    return(logits-logits.mean(1,keepdim=True)).numpy()


def observations(states,weights,row,density):
    stride=1024//density;scale=scales(states['t0'][::stride],row['metric'])
    offsets=(0,)if row['observable']=='full_state'else(0,1);e=row['evolution_updates']
    def at(t):return observe(states[f't{t}'][::stride],weights,row['observable'],scale)
    return np.concatenate([at(t)for t in offsets],1),np.concatenate([at(e+t)for t in offsets],1)


def phase_audit(first,last,headings,rotation,center):
    a=first-first.mean(0)if center else first;b=last-last.mean(0)if center else last
    _,segment,fraction=polygon_projection(b,a)
    wrap=lambda x:np.angle(np.exp(1j*x))
    phase=wrap(headings[segment]+fraction*wrap(np.roll(headings,-1)-headings)[segment]-headings)
    def circular(x):return dict(mean_rad=float(np.angle(np.exp(1j*x).mean())),absolute_rad=describe(abs(x)),
        absolute_max_rad=float(abs(x).max()),concentration=float(abs(np.exp(1j*x).mean())))
    return dict(neural_physical_label_displacement=circular(phase),physical_body_rotation=circular(rotation),
        neural_minus_body_phase_change=circular(wrap(phase-rotation)),
        coordinate='Known physical-angle interpolation along the nearest full-dimensional curve segment; descriptive only.')


def physical_rotation(family,cfg,row):
    if row['regime']!='left':return np.zeros(1024)
    t=row['evolution_updates'];records={r['updates']:r['physical_heading_change_rad']for r in family['audit']['left_physical']}
    if t in records:return np.asarray(records[t])
    step=cfg['action_repeat']/60;k=1-np.exp(-step/.25)
    def turn(updates):
        speed=0.;angle=0.
        for _ in range(updates):speed+=(cfg['rot_max']-speed)*k;angle+=speed*step
        return angle
    return np.full(1024,turn(min(t,8))+turn(max(t-8,0)))


def audit_alias(row,canonical):
    assert row['regime']==canonical['regime']
    exclude=set(KEYS)|{'source_key','source_family','source_exact_sham_of','both_metrics_candidate'}
    compare({k:v for k,v in row.items()if k not in exclude},{k:v for k,v in canonical.items()if k not in exclude},'exact alias')


def pair_status(raw,rms):
    pair=(raw,rms)
    if any(not r['convergence']['passes']or any(not r['densities'][n]['gates'][SPACING]for n in ('512','1024'))for r in pair):return 'sampling_unresolved'
    if any(not r['resolved_geometry']for r in pair):return 'geometry_failed'
    if any(r['physical_status']=='inconclusive'for r in pair):return 'physical_test_inconclusive'
    if any(not r['broad_candidate']for r in pair):return 'recovery_failed'
    return 'screening_candidate'


def digest_weights(weights,fixed=False):
    h=hashlib.sha256()
    for name,tensor in sorted(weights.items()):
        if fixed and name.startswith('lstm.'):continue
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_hashes(folder,records):
    for name,r in records.items():
        path=folder/name;assert path.stat().st_size==r['bytes']and sha(path)==r['sha256'],path
    return len(records)


def protocol_audit():
    protocol=read(design.PROTOCOL);original=read(design.old.PROTOCOL);training=read(design.old.TRAINING_PROTOCOL)
    assert protocol['original_protocol_sha256']==sha(design.old.PROTOCOL)
    assert protocol['training_protocol_sha256']==sha(design.old.TRAINING_PROTOCOL)
    count=0
    for doc in (protocol,original,training):
        for field in ('source_sha256','asset_sha256'):
            for name,value in doc.get(field,{}).items():assert sha(ROOT/name)==value,name;count+=1
    assert protocol['densities']==[256,512,1024]and protocol['no_optimization']and not protocol['retries']
    return protocol,dict(protocol_sha256=sha(design.PROTOCOL),parent_protocol_sha256=sha(design.old.PROTOCOL),source_asset_hashes=count)


def validate_input(record,condition,prep,cache):
    path=ROOT/record['path'];identity=(condition,record['context'],record['prefix'],record['kind'])
    if identity in cache:
        compare(record,cache[identity]['record']);return cache[identity]['geometry']
    assert record['complete']and record['condition']==condition and record['protocol_sha256']==sha(design.PROTOCOL)
    assert sha(path)==record['dense_sha256'];compare(read(path.with_suffix('.json')),record)
    old=next(r for r in prep['records']if(r['context'],r['prefix'],r['kind'])==identity[1:])
    oldpath=design.old.BASE/'inputs'/condition/old['path'];assert sha(oldpath)==record['old_input_sha256']==old['sha256']
    assert record['fixed_parameter_hash']==prep['fixed_parameter_hash']
    dense=torch.load(path,map_location='cpu',weights_only=True);coarse=torch.load(oldpath,map_location='cpu',weights_only=True)
    audit=input_guard(dense,coarse);compare(record['shared_input_audit'],audit);assert audit['passes']
    for field,label in [('native_x','native_input_sha256'),('ordinary_x','ordinary_input_sha256')]:
        assert design.old.parent.tensor_sha(dense[field])==record[label]
    # Compare every heading and all future input columns, never a fitted angle.
    for field in ('target_choice_heading','choice_heading','reference_heading'):
        x=dense[field].numpy();delta=np.angle(np.exp(1j*(np.roll(x,-1)-x)))
        assert np.max(abs(delta-2*np.pi/1024))<2e-5
    assert dense['left_x'].shape==(1287,)and dense['fb_x'].shape==(2,1287)
    assert dense['native_x'].shape==dense['ordinary_x'].shape and dense['native_x'].shape[1:]==(1024,1287)
    geometry={k:dense[k]for k in ('reference_heading','choice_heading','audit')}
    cache[identity]=dict(record=record,geometry=geometry,audit=audit)
    return geometry


def validate_endpoint(condition,endpoint,protocol,input_cache):
    folder=design.BASE/condition/endpoint;summary=read(folder/'summary.json');execution=read(folder/'execution.json')
    assert summary['complete']and execution['complete']and execution['exit_code']==0 and execution['error']is None
    assert summary['condition']==condition and summary['endpoint']==endpoint
    assert summary['protocol_sha256']==execution['protocol_sha256']==sha(design.PROTOCOL)
    assert execution['summary_sha256']==sha(folder/'summary.json')and execution['cap_seconds']==3600
    sourcefolder=design.old.BASE/condition/endpoint;sourcepath=sourcefolder/'summary.json';source=read(sourcepath)
    assert source['complete']and source['condition']==condition and source['endpoint']==endpoint and len(source['rows'])==33280
    assert source['protocol_sha256']==sha(design.old.PROTOCOL)and summary['source_summary_sha256']==sha(sourcepath)
    selected,cohort=selection(source['rows']);compare(summary['cohort'],cohort)
    if endpoint=='original':
        compare(protocol['original_cohorts'][condition]['cohort'],cohort)
        assert protocol['original_cohorts'][condition]['summary_sha256']==sha(sourcepath)
    eligibility=read(folder/'eligibility.json')
    assert eligibility['complete']and eligibility['condition']==condition and eligibility['endpoint']==endpoint
    assert eligibility['source_summary_sha256']==sha(sourcepath)and eligibility['protocol_sha256']==sha(design.PROTOCOL)
    compare(eligibility['cohort'],cohort)
    compare(read(folder/'prefix_replay.json'),read(sourcefolder/'prefix_replay.json'),'identical full-prefix replay')
    hashes=verify_hashes(folder,summary['artifacts']);source_hashes=verify_hashes(sourcefolder,source['artifacts'])
    prep_path=design.old.BASE/'inputs'/condition/'summary.json';prep=read(prep_path)
    assert sha(prep_path)==source['input_summary_sha256']and prep['complete']and prep['condition']==condition
    assert prep['protocol_sha256']==sha(design.old.PROTOCOL)
    checkpoint_path=design.old.model_path(condition,endpoint);checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert sha(checkpoint_path)==summary['checkpoint_sha256']==source['checkpoint_sha256']
    weights=checkpoint['policy'];digest=digest_weights(weights)
    assert digest==summary['policy_hash_before_float64']==source['policy_hash_before_float64']
    assert digest_weights(weights,True)==prep['fixed_parameter_hash'];compare(summary['training_provenance'],source['training_provenance'])
    if endpoint=='original':assert checkpoint['frames']==1_000_000_000
    else:
        terminal_path=checkpoint_path.parent/'summary.json';terminal=read(terminal_path)
        assert terminal['complete']and terminal['condition']==condition and terminal['arm']==endpoint
        assert terminal['updates']==checkpoint['updates']==180 and terminal['extra_frames']==checkpoint['extra_frames']==1036800
        assert terminal['policy_decisions']==259200 and terminal['fixed_tensors_unchanged']and terminal['teacher_unchanged']
        assert terminal['final_policy_hash']==digest and terminal['initial_policy_hash']==checkpoint['teacher_policy_hash']==prep['policy_hash']
        assert terminal['protocol_sha256']==checkpoint['protocol_sha256']==sha(design.old.TRAINING_PROTOCOL)
        assert terminal['artifacts']['final.pt']==dict(sha256=sha(checkpoint_path),bytes=checkpoint_path.stat().st_size)
        assert summary['training_provenance']['terminal_summary_sha256']==sha(terminal_path)
        assert checkpoint['config']==prep['config']and checkpoint['arm']==endpoint and checkpoint['original_frames']==1_000_000_000
    numeric=[r for r in selected if not r['exact_sham_of']]
    expected_inputs={physical_key(r)for r in numeric};records={physical_key(r):r for r in summary['prepared_inputs']}
    assert set(records)==expected_inputs and len(records)==len(summary['prepared_inputs'])
    inputs={k:validate_input(r,condition,prep,input_cache)for k,r in records.items()}
    rows=summary['rows'];lookup={key(r):r for r in rows};source_lookup={key(r):r for r in selected}
    assert len(rows)==len(lookup)==cohort['selected_rows']and set(lookup)==set(source_lookup)
    groups=collections.defaultdict(list)
    for row in rows:
        src=source_lookup[key(row)];assert row['source_key']==list(key(row))
        assert row['source_family']==src['family']and row['source_exact_sham_of']==src['exact_sham_of']
        compare(row['source_gates'],src['gates'])
        if not row['source_exact_sham_of']:groups[(row['source_family'],row['regime'])].append(row)
    normal_cache={};checked_rows=0;shared_timestamps=0;phase_checks=0
    for (family,regime),group in groups.items():
        sub=folder/family/regime;raw=read(sub/'result.json');assert raw['complete']
        rawlookup={subkey(r):r for r in raw['rows']};assert len(rawlookup)==len(raw['rows'])==len(group)
        states=dict(np.load(sub/'baseline_states_1024.npz',allow_pickle=False))
        coarse=dict(np.load(sourcefolder/family/regime/'baseline_states.npz',allow_pickle=False))
        assert all(z.shape==(1024,512)and np.isfinite(z).all()for z in states.values())
        shared={stamp[1:]:state_guard(z[::8],coarse[stamp])for stamp,z in states.items()}
        compare(raw['shared_128_state_agreement'],shared);assert all(v['passes']for v in shared.values());shared_timestamps+=len(shared)
        geometry_cache={}
        for row in group:
            compare({k:v for k,v in row.items()if k!='both_metrics_candidate'},rawlookup[subkey(row)])
            compare(row['shared_128_state_agreement'],shared);compare(row['tangent_density_agreement'],tangent_check(states['t0'],row['metric']))
            input_family=inputs[physical_key(row)];headings=input_family['reference_heading'if regime=='left'else'choice_heading'].numpy()
            rotation=physical_rotation(input_family,prep['config'],row);obs={}
            for density in (256,512,1024):
                cachekey=(row['observable'],row['metric']if row['observable']=='full_state'else None,row['centered'],row['evolution_updates'],density)
                if cachekey not in geometry_cache:
                    first,last=observations(states,weights,row,density);gates=base.independent_gates(first,last,row['centered'])
                    a=first-first.mean(0)if row['centered']else first;b=last-last.mean(0)if row['centered']else last
                    residual=polygon_projection(b,a)[0]/max(base.radius(first),1e-30)
                    phase=phase_audit(first,last,headings[::1024//density],rotation[::1024//density],row['centered'])
                    geometry_cache[cachekey]=(first,last,gates,describe(residual),phase)
                first,last,gates,residual,phase=geometry_cache[cachekey];stored=row['densities'][str(density)];obs[density]=(first,last)
                compare(stored['gates'],gates);assert stored['necessary_geometry_pass']==all(gates.values())and stored['density']==density
                compare(stored['geometry']['radius'],base.radius(first));compare(stored['evolved_geometry']['radius'],base.radius(last))
                compare(stored['radius_retention'],base.radius(last)/max(base.radius(first),1e-30))
                compare(stored['residual_to_reference_curve_over_radius'],residual)
                compare(stored['phase_comparison'],phase);phase_checks+=1
            convergence=polygon_convergence(*obs[512],*obs[1024],row['centered']);compare(row['convergence'],convergence)
            qualifies=bool(convergence['passes']and all(row['densities'][str(n)]['necessary_geometry_pass']for n in (512,1024)))
            assert row['resolved_geometry']==row['normal_test_run']==qualifies
            if qualifies:
                assert set(row['recovery'])=={'512','1024'}
                physicals=[]
                for density in (512,1024):
                    rec=row['recovery'][str(density)];candidates=[(rec['nominal'],rec['nominal_arrays'])]+[(v['result'],v['arrays'])for v in rec['bounded_followups']]
                    for probe,name in candidates:
                        cachekey=(str(sub),name)
                        if cachekey not in normal_cache:normal_cache[cachekey]=audit_probe(sub,probe,name,states['t0'][::1024//density],row['metric'],*obs[density],weights,row)
                    follow=rec['bounded_followups']
                    if rec['nominal']['hidden_bound_excess_count']:
                        assert len(follow)in (1,2)and follow[0]['method']=='shared'
                        assert(len(follow)==1)==follow[0]['result']['physical_conclusive']
                        if len(follow)==2:assert follow[1]['method']=='clip_equalize'
                    else:assert not follow
                    physical=candidates[-1][0];physicals.append(physical)
                    assert rec['physical_status']==physical['physical_status']and rec['physical_conclusive']==physical['physical_conclusive']
                    assert rec['passes']==bool(physical['physical_conclusive']and physical['all_normal_directions_pass'])
                status='inconclusive'if any(not p['physical_conclusive']for p in physicals)else('pass'if all(p['all_normal_directions_pass']for p in physicals)else'failed_recovery')
                assert row['physical_status']==status and row['broad_candidate']==(status=='pass')
            else:assert 'recovery'not in row and not row['broad_candidate']and row['physical_status']=='not_tested_geometry_or_resolution_failed'
            checked_rows+=1
        print(json.dumps(dict(audited_family=family,regime=regime,condition=condition,endpoint=endpoint,rows=checked_rows)),flush=True)
    canonical={(r['source_family'],subkey(r)):r for r in rows if not r['source_exact_sham_of']}
    for row in rows:
        if row['source_exact_sham_of']:audit_alias(row,canonical[(row['source_exact_sham_of'],subkey(row))])
        partner=dict(row);partner['metric']='raw'if row['metric']=='h_c_RMS'else'h_c_RMS'
        assert row['both_metrics_candidate']==bool(row['broad_candidate']and lookup[key(partner)]['broad_candidate'])
    assert summary['individual_candidate_rows']==sum(r['broad_candidate']for r in rows)
    assert summary['both_metrics_candidate_rows']==sum(r['both_metrics_candidate']for r in rows)
    return summary,dict(complete=True,condition=condition,endpoint=endpoint,summary_sha256=sha(folder/'summary.json'),
        execution_sha256=sha(folder/'execution.json'),source_summary_sha256=sha(sourcepath),cohort=cohort,
        independently_audited_numerical_rows=checked_rows,exact_alias_rows=len(rows)-checked_rows,
        shared_state_timestamps=shared_timestamps,phase_comparisons=phase_checks,normal_files=list(normal_cache.values()),
        artifact_hashes_checked=hashes,source_artifact_hashes_checked=source_hashes,prepared_input_families=len(inputs))


def summarize(result):
    rows=result['rows'];lookup={key(r):r for r in rows};pairs=[]
    for row in rows:
        if row['metric']!='raw':continue
        other=lookup[key(dict(row,metric='h_c_RMS'))]
        pairs.append({k:row[k]for k in KEYS if k!='metric'}|dict(status=pair_status(row,other),exact_alias=bool(row['source_exact_sham_of']),
            geometry_by_density={n:bool(row['densities'][n]['necessary_geometry_pass']and other['densities'][n]['necessary_geometry_pass'])for n in ('256','512','1024')},
            both_metrics_candidate=row['both_metrics_candidate']))
    counts={name:sum(r['status']==name for r in pairs)for name in STATUSES}
    assert len(pairs)*2==len(rows)and counts['screening_candidate']*2==result['both_metrics_candidate_rows']
    groups=[]
    for group,values in __import__('itertools').groupby(sorted(pairs,key=lambda r:(r['regime'],r['observable'],r['centered'],r['window_updates'])),
        key=lambda r:(r['regime'],r['observable'],r['centered'],r['window_updates'])):
        values=list(values);groups.append(dict(zip(('regime','observable','centered','window_updates'),group))|dict(pairs=len(values),statuses=dict(collections.Counter(v['status']for v in values))))
    directions=collections.defaultdict(lambda:dict(tested_rows=0,physically_conclusive_rows=0,failed_recovery_rows=0))
    for row in rows:
        if not row['normal_test_run']:continue
        for density,rec in row['recovery'].items():
            physical=rec['bounded_followups'][-1]['result']if rec['bounded_followups']else rec['nominal']
            for direction in physical['directions'][:-1]:
                group=(row['regime'],row['observable'],row['metric'],density,direction['name'],bool(row['source_exact_sham_of']))
                count=directions[group];count['tested_rows']+=1
                count['physically_conclusive_rows']+=int(physical['physical_conclusive'])
                count['failed_recovery_rows']+=int(physical['physical_conclusive']and not direction['passes'])
    direction_rows=[dict(zip(('regime','observable','metric','density','direction','exact_alias'),k))|v for k,v in sorted(directions.items())]
    return dict(condition=result['condition'],endpoint=result['endpoint'],source_rows=result['cohort']['source_rows'],selected_rows=len(rows),
        selected_metric_pairs=len(pairs),exact_alias_pairs=sum(r['exact_alias']for r in pairs),statuses=counts,
        necessary_geometry_by_density={n:sum(r['geometry_by_density'][n]for r in pairs)for n in ('256','512','1024')},
        groups=groups,pairs=pairs,physical_recovery_by_direction=direction_rows)


def render(models,out):
    labels=[f"{'Grid'if r['condition']=='gc100'else'PGC + grid'} · {r['endpoint']}"for r in models]
    fig,axes=plt.subplots(1,2,figsize=(11,max(3.5,.55*len(models)+1.6)),gridspec_kw={'width_ratios':[1.4,1]})
    left=np.zeros(len(models));y=np.arange(len(models))
    for name,label,color in zip(STATUSES,LABELS,COLORS):
        values=np.array([r['statuses'][name]for r in models]);axes[0].barh(y,values,left=left,color=color,label=label);left+=values
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Selected raw/RMS pairs');axes[0].set_title('What finer sampling resolved')
    for i,total in enumerate(left):axes[0].text(total+.5,i,str(int(total)),va='center',fontsize=9)
    for i,r in enumerate(models):
        axes[1].plot([256,512,1024],[r['necessary_geometry_by_density'][str(n)]for n in (256,512,1024)],'o-',label=labels[i])
    axes[1].set_xticks([256,512,1024]);axes[1].set_xlabel('Actually simulated headings');axes[1].set_ylabel('Pairs passing geometry');axes[1].set_title('Same criteria at finer resolution')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False,fontsize=8)
    axes[1].legend(frameon=False,fontsize=8);fig.tight_layout()
    fig.savefig(out/'resolution_results.png',dpi=180,bbox_inches='tight');fig.savefig(out/'resolution_results.pdf',bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--endpoints',nargs='+',choices=('original','clean','noise'),default=['original'])
    args=parser.parse_args();assert args.endpoints in (['original'],['original','clean','noise'])
    # Refuse incomplete cohorts before creating a report directory or doing heavy arithmetic.
    for endpoint in args.endpoints:
        for condition in ('gc100','pgc33'):
            folder=design.BASE/condition/endpoint
            assert read(folder/'summary.json')['complete']and read(folder/'execution.json')['exit_code']==0
    out=design.BASE/('report_original'if args.endpoints==['original']else'report_all')
    assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    results=[];audits=[];cache={};torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        protocol,provenance=protocol_audit()
        for endpoint in args.endpoints:
            for condition in ('gc100','pgc33'):
                result,audit=validate_endpoint(condition,endpoint,protocol,cache);results.append(result);audits.append(audit)
    models=[summarize(r)for r in results];render(models,out)
    atomic_json(out/'summary.json',dict(complete=True,endpoints=args.endpoints,source_sha256=sha(__file__),protocol_sha256=sha(design.PROTOCOL),
        source_rows=sum(r['source_rows']for r in models),selected_rows=sum(r['selected_rows']for r in models),
        selected_pairs=sum(r['selected_metric_pairs']for r in models),models=models))
    atomic_json(out/'validation.json',dict(complete=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=sha(__file__),
        audit_source_sha256={name:sha(ROOT/name)for name in ('scripts/report_native_turn_geometry_resolution.py',
            'scripts/report_native_turn_geometry.py','tests/test_report_native_turn_geometry_resolution.py',
            'tests/test_report_native_turn_geometry.py')},
        arithmetic_versions=dict(python=sys.version.split()[0],numpy=np.__version__,torch=str(torch.__version__)),
        mode='Saved arrays and static actor multiplication only; no recurrent or environment execution.',provenance=provenance,models=audits,
        input_families=[dict(identity=list(k),record=v['record'],independent_shared_input_audit=v['audit'])for k,v in cache.items()],
        summary_sha256=sha(out/'summary.json')))
    lines=['# Finer sampling of the same angular families','',
        'We repeated every spacing-only case with 1,024 real physical histories. The 256, 512 and 1,024 curves contain actual simulated states. No states were interpolated.',
        '', '![Resolution results](resolution_results.png)','',
        '| Network | Selected pairs | Sampling unresolved | Other geometry failed | Kick inconclusive | Recovery failed | Candidates |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in models:
        lines.append(f"| {'Grid'if r['condition']=='gc100'else'PGC + grid'} · {r['endpoint']} | {r['selected_metric_pairs']} | "+' | '.join(str(r['statuses'][n])for n in STATUSES)+' |')
    lines.extend(['',
        'A pair contains the same case under raw and h/c-normalized disturbances. Both must pass. Exact sham controls remain in the totals; these are screening cases from fixed models, not independent trained networks.', '',
        'Sampling is unresolved when spacing still fails or the 512/1,024 polygons do not converge. Recovery is tested only after both densities meet the original geometry criteria. Invisible, zero or badly reoriented bounded kicks are physically inconclusive.', '',
        'The audit checks all selected and excluded source rows, genuine shared inputs, separate hidden/cell state agreement, nearest-segment convergence, physical-angle transport, eight full-circle perturbation anchors, all 20 normal directions and both signs. It also reconstructs bounded kick geometry from the saved physical vectors.', '',
        'A positive is a screening candidate. Eight disturbance anchors can miss a vulnerable angle; confirmation requires denser phase/Jacobian checks and a behavioral cue-repair experiment. Forward/back results concern a periodically driven memory family. Measured phase drift remains necessary before interpreting it as stationary heading memory.', '',
        'The original 128-heading results remain unchanged. This follow-up covers only the prospectively selected spacing-only cases and their required metric controls.', '',
        '[All counts and cases](summary.json) · [Independent saved-array checks](validation.json)'])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'artifacts.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in out.iterdir()if p.is_file()})


if __name__=='__main__':main()
