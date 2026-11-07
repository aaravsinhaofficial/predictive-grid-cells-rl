"""Separately sealed, genuine-history resolution refinement of the turn assay.

No training or fitted angular projection. The frozen 128-phase study is imported,
never edited. Native outcomes require a reviewed, sealed resolution protocol.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import native_turn_geometry as old

BASE=ROOT/'results/native_turn_geometry_resolution'
DRAFT=ROOT/'docs/research/native_turn_geometry_resolution_protocol_draft.json'
PROTOCOL=ROOT/'docs/research/native_turn_geometry_resolution_protocol.json'
DENSITIES=(256,512,1024)
SPACING='max_edge_below_10percent_radius'
KEYS=('context','prefix','kind','branch','regime','window_updates','evolution_updates','observable','metric','centered')
ROWKEY=('window_updates','evolution_updates','observable','metric','centered')
CAP_SECONDS=3600
CONVERGENCE=.005
INPUT_TOLERANCE=1e-5
STATE_RELATIVE_TOLERANCE=1e-5
STATE_ABSOLUTE_TOLERANCE=1e-8
STATE_SHAPE_FRACTION=1e-3
STATE_SHAPE_FLOOR=1e-8


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(row):return tuple(row[k]for k in KEYS)
def subkey(row):return tuple(row[k]for k in ROWKEY)
def physical_key(row):return row['context'],row['prefix'],row['kind']
def peak_rss_bytes():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin'else 1024)


def attach_shams(records,selected):
    """Keep the two periodic map sections distinct when copying exact aliases."""
    result=list(records)
    canonical={(r['source_family'],r['regime'],subkey(r)):r for r in records}
    assert len(canonical)==len(records)
    for row in selected:
        if row['exact_sham_of']:
            r=canonical[(row['exact_sham_of'],row['regime'],subkey(row))]
            result.append({**r,**{k:row[k]for k in KEYS},'source_key':list(key(row)),
                'source_family':row['family'],'source_exact_sham_of':row['exact_sham_of']})
    return result


def eligibility(rows):
    """Every exact spacing-only failure, plus its mandatory other-metric control."""
    lookup={key(r):r for r in rows};assert len(lookup)==len(rows)
    direct={key(r)for r in rows if not r['gates'][SPACING]and all(v for k,v in r['gates'].items()if k!=SPACING)}
    selected=set(direct)
    for k in direct:
        r=lookup[k];other={**r,'metric':'h_c_RMS'if r['metric']=='raw'else'raw'}
        assert key(other)in lookup
        selected.add(key(other))
    result=[r for r in rows if key(r)in selected]
    numeric=[r for r in result if not r['exact_sham_of']]
    return result,dict(source_rows=len(rows),spacing_only_rows=len(direct),mandatory_metric_control_rows=len(selected-direct),
        selected_rows=len(result),numerical_rows=len(numeric),exact_sham_rows=len(result)-len(numeric),
        excluded_rows=len(rows)-len(result),physical_families=len({physical_key(r)for r in numeric}),
        branch_families=len({r['family']for r in numeric}),regime_families=len({(r['family'],r['regime'])for r in numeric}),
        source_keys=[list(key(r))for r in result],spacing_only_keys=[list(k)for k in sorted(direct)],
        exclusion_failure_sets=[dict(gates=list(k),count=v)for k,v in sorted(collections.Counter(
            tuple(sorted(k for k,v in r['gates'].items()if not v))for r in rows if key(r)not in selected).items())])


def nested_indices(count,density):
    assert count==1024 and density in(128,256,512,1024)
    return np.arange(density)*(count//density)


def array_agreement(a,b,absolute=0.,relative=0.):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    error=float(np.max(np.abs(a-b)))if a.size else 0.
    allowed=absolute+relative*max(1.,float(np.max(np.abs(b)))if b.size else 1.)
    return dict(max_absolute_error=error,allowed_error=allowed,passes=bool(error<=allowed))


def shared_input_audit(dense,coarse):
    ids=nested_indices(1024,128);out={}
    assert dense['audit']['phases']==1024 and coarse['audit']['phases']==128
    for name in ('native_x','ordinary_x'):
        a=dense[name][:,ids].numpy();b=coarse[name].numpy()
        out[name+'_CNN']=array_agreement(a[:,:,:256],b[:,:,:256],absolute=INPUT_TOLERANCE)
        out[name+'_nonvisual_exact']=array_agreement(a[:,:,256:],b[:,:,256:])
    for name in ('target_choice_heading','choice_heading','reference_heading'):
        out[name]=array_agreement(dense[name][ids].numpy(),coarse[name].numpy())
    for name in ('left_x','fb_x'):
        out[name]=array_agreement(dense[name].numpy(),coarse[name].numpy())
    for name in coarse.get('physical',{}):
        out['physical_'+name]=array_agreement(dense['physical'][name][:,ids].numpy(),coarse['physical'][name].numpy())
    assert dense['audit']['labels']==coarse['audit']['labels']
    return dict(complete=True,passes=all(v['passes']for v in out.values()),comparisons=out,
        method='Actual dense-history samples at shared physical headings; no input or state interpolation.')


def shared_state_agreement(dense,coarse):
    """A large cell baseline cannot license hidden or angular-pattern errors."""
    dense=np.asarray(dense);coarse=np.asarray(coarse)
    assert dense.shape==coarse.shape and dense.ndim==2 and dense.shape[1]%2==0
    half=dense.shape[1]//2;blocks={}
    for name,sl in(('hidden',slice(0,half)),('cell',slice(half,2*half))):
        a=dense[:,sl];b=coarse[:,sl];difference=a-b
        allowed=STATE_ABSOLUTE_TOLERANCE+STATE_RELATIVE_TOLERANCE*np.max(np.abs(b),axis=0)
        coordinate_pass=bool(np.all(np.abs(difference)<=allowed))
        centered_b=b-b.mean(0);shape_difference=(a-a.mean(0))-centered_b
        angular_rms=float(np.sqrt(np.mean(centered_b**2)))
        shape_rms=float(np.sqrt(np.mean(shape_difference**2)))
        shape_max=float(np.sqrt(np.mean(shape_difference**2,axis=1)).max())
        shape_allowed=STATE_SHAPE_FLOOR+STATE_SHAPE_FRACTION*angular_rms
        common_mean=a.mean(0)-b.mean(0)
        blocks[name]=dict(max_absolute_error=float(np.max(np.abs(difference))),
            total_error_RMS=float(np.sqrt(np.mean(difference**2))),
            max_error_over_coordinate_tolerance=float(np.max(np.abs(difference)/allowed)),
            coordinatewise_pass=coordinate_pass,angular_RMS=angular_rms,
            centered_difference_RMS=shape_rms,centered_difference_max_row_RMS=shape_max,
            common_mean_shift_RMS=float(np.sqrt(np.mean(common_mean**2))),common_mean_shift_max_absolute=float(np.abs(common_mean).max()),
            allowed_centered_difference_max_row_RMS=shape_allowed,
            centered_max_row_RMS_over_tolerance=shape_max/shape_allowed,
            centered_RMS_over_angular_RMS_with_floor=shape_rms/max(angular_rms,STATE_SHAPE_FLOOR),
            angular_shape_pass=bool(shape_max<=shape_allowed))
    return dict(passes=all(r['coordinatewise_pass']and r['angular_shape_pass']for r in blocks.values()),
        coordinate_absolute_tolerance=STATE_ABSOLUTE_TOLERANCE,coordinate_relative_tolerance=STATE_RELATIVE_TOLERANCE,
        angular_shape_fraction=STATE_SHAPE_FRACTION,angular_shape_absolute_floor=STATE_SHAPE_FLOOR,blocks=blocks)


def convergence(first512,last512,first1024,last1024,center=False):
    """Numerical polygon convergence, separate from the unchanged scientific gates."""
    radius=max(float(np.sqrt(np.mean(np.sum((first1024-first1024.mean(0))**2,axis=1)))),1e-30);records={}
    for label,coarse,fine in (('initial',first512,first1024),('evolved',last512,last1024)):
        if center:coarse=coarse-coarse.mean(0);fine=fine-fine.mean(0)
        distance,_,_,_=old.flow.nearest_closed_segments(fine,coarse)
        records[label]=dict(max_distance_over_initial_radius=float(distance.max()/radius),
            p90_distance_over_initial_radius=float(np.quantile(distance,.9)/radius))
    return dict(tolerance=CONVERGENCE,passes=all(v['max_distance_over_initial_radius']<=CONVERGENCE for v in records.values()),
        comparisons=records,method='Distance of genuinely simulated 1024 states to 512-segment polygons; centered only for the declared centered observable.')


def tangent_agreement(reference,metric):
    tangents=[]
    for density in(512,1024):
        p=reference[nested_indices(len(reference),density)]
        scale=np.ones(p.shape[1])if metric=='raw'else old.state_scales(p)
        p=p/scale;t=np.roll(p,-1,0)-np.roll(p,1,0)
        t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-30)
        tangents.append(t[np.arange(8)*density//8])
    cosine=np.sum(tangents[0]*tangents[1],axis=1)
    return dict(cosines=cosine.tolist(),minimum_cosine=float(cosine.min()),mean_cosine=float(cosine.mean()),
        interpretation='Descriptive agreement of physical perturbation tangent directions at the same eight angles.')


def density_geometry(net,states,row,headings,rotation):
    results={};observations={};kind=row['observable'];evolution=row['evolution_updates'];offsets=row['observation_offsets']
    for density in DENSITIES:
        ids=nested_indices(len(states[0]),density);subset={t:z[ids]for t,z in states.items()}
        scale=np.ones(states[0].shape[1])if row['metric']=='raw'else old.state_scales(subset[0])
        output=(lambda z,s=torch.as_tensor(scale):z/s)if kind=='full_state'else old.observer(net,kind)
        first,last=old.observation_arrays(subset,output,offsets,evolution)
        record=old.geometry_screen(first,last,row['centered'])
        record.update(density=density,phase_comparison=old.physical_phase_comparison(first,last,headings[ids],rotation[ids],row['centered']))
        results[str(density)]=record;observations[density]=(first,last)
    check=convergence(*observations[512],*observations[1024],row['centered'])
    qualified=bool(check['passes']and all(results[str(n)]['necessary_geometry_pass']for n in(512,1024)))
    return results,check,qualified


def test_recovery(transition,curve,scale,output,row,folder,prefix):
    probe,arrays=old.finite_probe(transition,curve,scale,output,row['observation_offsets'],row['evolution_updates'],anchors=8)
    name=f'{prefix}_nominal.npz';np.savez_compressed(folder/name,**arrays)
    followups=[];physical=probe
    if probe['hidden_bound_excess_count']:
        for method in ('shared','clip_equalize'):
            bounded,ba=old.finite_probe(transition,curve,scale,output,row['observation_offsets'],row['evolution_updates'],anchors=8,bounded=method)
            bname=f'{prefix}_{method}.npz';np.savez_compressed(folder/bname,**ba)
            followups.append(dict(method=method,result=bounded,arrays=bname));physical=bounded
            if bounded['physical_conclusive']:break
    return dict(nominal=probe,nominal_arrays=name,bounded_followups=followups,
        physical_status=physical['physical_status'],physical_conclusive=physical['physical_conclusive'],
        passes=bool(physical['physical_conclusive']and physical['all_normal_directions_pass']))


def load_source(condition,endpoint):
    path=old.BASE/condition/endpoint/'summary.json';data=json.loads(path.read_text())
    assert data['complete']and data['condition']==condition and data['endpoint']==endpoint
    assert data['protocol_sha256']==sha(old.PROTOCOL)and len(data['rows'])==33280
    assert data['checkpoint_sha256']==sha(old.model_path(condition,endpoint))
    assert len({key(r)for r in data['rows']})==33280
    return data,path


def make_draft():
    originals={}
    for condition in old.parent.CONDITIONS:
        data,path=load_source(condition,'original');_,cohort=eligibility(data['rows'])
        originals[condition]=dict(summary_sha256=sha(path),cohort=cohort)
    sources=[Path(__file__).relative_to(ROOT).as_posix(),'tests/test_native_turn_geometry_resolution.py',
        'tests/test_native_turn_geometry_resolution_controls.py',
        'docs/research/native_turn_geometry_resolution_proposal.md']
    return dict(schema=1,study='native_turn_geometry_resolution',no_optimization=True,
        endpoints=['original','clean','noise'],conditions=list(old.parent.CONDITIONS),densities=list(DENSITIES),
        eligibility='Exact spacing-only failure in either raw/RMS row; mandatory other-metric row retained. All rows, windows, contexts and branches considered. Exact shams remain in denominator.',
        source_sha256={p:sha(ROOT/p)for p in sources},original_protocol_sha256=sha(old.PROTOCOL),
        training_protocol_sha256=sha(old.TRAINING_PROTOCOL),original_cohorts=originals,
        dense_input_method='Unchanged prepare_family(phases=1024), true rendered physical histories. Subsample actual1024 states at nested256/512/1024 indices.',
        input_tolerance=INPUT_TOLERANCE,input_tolerance_scope='CNN embeddings only, because convolution batch sizes differ; all physical trajectories/headings and nonvisual codes/goal/actions/rewards are exact.',
        shared_state_tolerance=dict(coordinate_relative=STATE_RELATIVE_TOLERANCE,coordinate_absolute=STATE_ABSOLUTE_TOLERANCE,
            coordinate_scale='Maximum across phases for each coordinate separately.',
            angular_shape_fraction=STATE_SHAPE_FRACTION,angular_shape_absolute_floor=STATE_SHAPE_FLOOR,
            angular_error='Maximum across phases of centered row RMS; additive absolute floor.',hidden_and_cell_blocks_separate=True),
        shared_common_future_input_exact=True,own_complete_prefix_replay=True,
        convergence=dict(densities=[512,1024],max_polygon_distance_over_initial_radius=CONVERGENCE,
            check_both_initial_and_evolved=True,original_geometry_gates_required_at_both=True),
        normal_tests=dict(densities=[512,1024],anchors=8,anchor_policy='Same eight physical angles across full circle at each density.',
            normal_directions=20,signs=2,amplitude=.02,normal_seed=old.NORMAL_SEED,
            source='Unchanged finite_probe and bounded-follow-up policy from sealed128 study.',
            raw_and_RMS_both_required=True),
        cap_seconds=CAP_SECONDS,cap_scope='Each population/endpoint cohort, including missing input preparation and complete prefix replay.',
        estimated_preparation_peak_GiB_per_worker=3,peak_resident_memory_recorded=True,
        retries=False,timeout_result='Incomplete ledger retained; no terminal complete summary, no silent case/density deletion.',
        original_outcomes_preserved=True,source_methods_unchanged=True,cloud=False)


def validate_design(design):
    assert design['schema']==1 and design['no_optimization']and design['densities']==list(DENSITIES)
    assert design['cap_seconds']==CAP_SECONDS and not design['retries']and not design['cloud']
    assert design['original_protocol_sha256']==sha(old.PROTOCOL)
    assert design['training_protocol_sha256']==sha(old.TRAINING_PROTOCOL)
    for path,digest in design['source_sha256'].items():assert sha(ROOT/path)==digest,path
    old.validate_protocol(json.loads(old.PROTOCOL.read_text()))
    for condition,record in design['original_cohorts'].items():
        data,path=load_source(condition,'original');assert record['summary_sha256']==sha(path)
        assert eligibility(data['rows'])[1]==record['cohort']


def load_policy(condition,endpoint,source):
    prep_path=old.BASE/'inputs'/condition/'summary.json'
    assert sha(prep_path)==source['input_summary_sha256']
    prep=json.loads(prep_path.read_text())
    assert prep['complete']and prep['condition']==condition and prep['protocol_sha256']==sha(old.PROTOCOL)
    assert prep['checkpoint_sha256']==sha(old.model_path(condition,'original'))
    parent_design=json.loads(old.parent.PROTOCOL.read_text())
    bank=torch.load(ROOT/parent_design['banks'][condition]['path'],map_location='cpu',weights_only=True)
    cfg=old.endpoint_config(bank['config']);assert prep['config']==cfg
    path=old.model_path(condition,endpoint)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True);provenance=None
    if endpoint=='original':assert checkpoint['frames']==1_000_000_000
    else:
        terminal_path=path.parent/'summary.json';terminal=json.loads(terminal_path.read_text())
        training=json.loads(old.TRAINING_PROTOCOL.read_text())
        assert terminal['complete']and terminal['condition']==condition and terminal['arm']==endpoint
        assert terminal['updates']==checkpoint['updates']==180 and terminal['extra_frames']==checkpoint['extra_frames']==1_036_800
        assert terminal['policy_decisions']==259_200 and terminal['fixed_tensors_unchanged']and terminal['teacher_unchanged']
        assert terminal['artifacts']['final.pt']==dict(sha256=sha(path),bytes=path.stat().st_size)
        assert terminal['protocol_sha256']==checkpoint['protocol_sha256']==sha(old.TRAINING_PROTOCOL)
        assert terminal['initial_policy_hash']==checkpoint['teacher_policy_hash']==prep['policy_hash']
        assert checkpoint['config']==cfg and checkpoint['arm']==endpoint and checkpoint['original_frames']==1_000_000_000
        for name,digest in training['source_sha256'].items():assert sha(ROOT/name)==digest,name
        provenance=dict(terminal_summary_sha256=sha(terminal_path),training_protocol_sha256=sha(old.TRAINING_PROTOCOL),
            terminal_complete=True,teacher_policy_hash=terminal['initial_policy_hash'],final_policy_hash=terminal['final_policy_hash'],
            frozen_tensor_hash=terminal['frozen_tensor_hash'])
        assert provenance==source['training_provenance']
    with torch.random.fork_rng(devices=[]):net=old.build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(checkpoint['policy']);assert old.fixed_parameter_digest(net)==prep['fixed_parameter_hash']
    assert old.state_digest(net)==source['policy_hash_before_float64']
    if provenance:assert old.state_digest(net)==terminal['final_policy_hash']
    return net,cfg,bank,prep,provenance


def prepared_family(net,pop,cfg,context,prefix,kind,condition,prep):
    folder=BASE/'inputs'/condition;folder.mkdir(parents=True,exist_ok=True)
    name=f"context{context['context_id']:02d}_prefix{prefix}_{kind}"
    path=folder/f'{name}.pt';meta=folder/f'{name}.json'
    old_record=next(r for r in prep['records']if(r['context'],r['prefix'],r['kind'])==(context['context_id'],prefix,kind))
    old_path=old.BASE/'inputs'/condition/old_record['path'];assert sha(old_path)==old_record['sha256']
    if path.exists()or meta.exists():
        assert path.exists()and meta.exists(),'Incomplete input cache; no selective retry.'
        record=json.loads(meta.read_text())
        assert record['complete']and record['protocol_sha256']==sha(PROTOCOL)and record['dense_sha256']==sha(path)
        assert record['old_input_sha256']==old_record['sha256']and record['fixed_parameter_hash']==prep['fixed_parameter_hash']
        return record
    family=old.prepare_family(net,pop,cfg,context,kind,prefix,phases=1024)
    coarse=torch.load(old_path,map_location='cpu',weights_only=True)
    audit=shared_input_audit(family,coarse);assert audit['passes'],audit
    old.atomic_torch(path,family)
    record=dict(complete=True,condition=condition,context=context['context_id'],prefix=prefix,kind=kind,
        protocol_sha256=sha(PROTOCOL),dense_sha256=sha(path),path=str(path.relative_to(ROOT)),
        old_input_sha256=old_record['sha256'],fixed_parameter_hash=prep['fixed_parameter_hash'],shared_input_audit=audit,
        native_input_sha256=old.parent.tensor_sha(family['native_x']),ordinary_input_sha256=old.parent.tensor_sha(family['ordinary_x']))
    old.atomic_json(meta,record);return record


def run_regime(net,reference,transition,rows,headings,rotations,folder,old_folder,source,deadline):
    folder.mkdir(parents=True,exist_ok=False)
    needed={0}
    for r in rows:needed.update(r['observation_offsets']);needed.update(r['evolution_updates']+o for o in r['observation_offsets'])
    states=old.trajectory(transition,reference,max(needed),needed)
    np.savez_compressed(folder/'baseline_states_1024.npz',**{f't{t}':z for t,z in states.items()})
    relative=str((old_folder/'baseline_states.npz').relative_to(old.BASE/source['condition']/source['endpoint']))
    assert sha(old_folder/'baseline_states.npz')==source['artifacts'][relative]['sha256']
    with np.load(old_folder/'baseline_states.npz',allow_pickle=False)as coarse:
        shared={str(t):shared_state_agreement(z[::8],coarse[f't{t}'])for t,z in states.items()}
    assert all(a['passes']for a in shared.values()),shared
    records=[];probe_cache={};geometry_cache={}
    for index,row in enumerate(rows):
        deadline()
        geo_key=(row['observable'],row['centered'],row['evolution_updates'],tuple(row['observation_offsets']),row['metric']if row['observable']=='full_state'else None)
        if geo_key not in geometry_cache:
            geometry_cache[geo_key]=density_geometry(net,states,row,headings,rotations[row['evolution_updates']])
        densities,check,qualified=geometry_cache[geo_key]
        result={k:row[k]for k in KEYS}
        result.update(source_key=list(key(row)),source_family=row['family'],source_exact_sham_of=None,
            source_gates=row['gates'],densities=densities,convergence=check,resolved_geometry=qualified,
            normal_test_run=False,broad_candidate=False,physical_status='not_tested_geometry_or_resolution_failed',
            shared_128_state_agreement=shared,tangent_density_agreement=tangent_agreement(reference,row['metric']))
        if qualified:
            probes={}
            for density in(512,1024):
                deadline();ids=nested_indices(len(reference),density);curve=reference[ids]
                scale=np.ones(curve.shape[1])if row['metric']=='raw'else old.state_scales(curve)
                output=(lambda z,s=torch.as_tensor(scale):z/s)if row['observable']=='full_state'else old.observer(net,row['observable'])
                cache=(density,row['metric'],row['observable'],row['evolution_updates'],tuple(row['observation_offsets']))
                if cache not in probe_cache:
                    prefix=f"normal_{density}_{row['metric']}_{row['observable']}_{row['evolution_updates']}"
                    probe_cache[cache]=test_recovery(transition,curve,scale,output,row,folder,prefix)
                probes[str(density)]=probe_cache[cache]
            status='inconclusive'if any(not p['physical_conclusive']for p in probes.values())else('pass'if all(p['passes']for p in probes.values())else'failed_recovery')
            result.update(normal_test_run=True,recovery=probes,physical_status=status,broad_candidate=status=='pass')
        records.append(result)
    old.atomic_json(folder/'result.json',dict(complete=True,rows=records,shared_128_state_agreement=shared))
    return records


def run(condition,endpoint):
    design=json.loads(PROTOCOL.read_text());validate_design(design)
    source,source_path=load_source(condition,endpoint);selected,cohort=eligibility(source['rows'])
    if endpoint=='original':assert cohort==design['original_cohorts'][condition]['cohort']
    folder=BASE/condition/endpoint;assert not folder.exists()or not any(folder.iterdir()),'No selective retries.'
    folder.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();started=dt.datetime.now(dt.timezone.utc).isoformat();completed=[];exit_code=1;error=None
    def deadline():
        if time.monotonic()-began>=CAP_SECONDS:raise TimeoutError('Sealed cohort wall-time cap reached')
    def alarm_handler(signum,frame):raise TimeoutError('Sealed cohort wall-time cap reached')
    previous=signal.signal(signal.SIGALRM,alarm_handler);signal.setitimer(signal.ITIMER_REAL,CAP_SECONDS)
    old.atomic_json(folder/'eligibility.json',dict(complete=True,source_summary_sha256=sha(source_path),
        protocol_sha256=sha(PROTOCOL),condition=condition,endpoint=endpoint,cohort=cohort))
    try:
        net,cfg,bank,prep,provenance=load_policy(condition,endpoint,source)
        census=torch.load(ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt',map_location='cpu',weights_only=True)
        incoming,prefix_audit=old.memory.replay_own_prefixes(net,bank,census,endpoint=='original');del census
        old.atomic_json(folder/'prefix_replay.json',prefix_audit)
        numeric=[r for r in selected if not r['exact_sham_of']]
        family_ids=sorted({physical_key(r)for r in numeric});contexts={c['context_id']:c for c in bank['contexts']}
        pop=old.build_population(cfg,'cpu');prepared={}
        for context,prefix,kind in family_ids:
            deadline();prepared[(context,prefix,kind)]=prepared_family(net,pop,cfg,contexts[context],prefix,kind,condition,prep)
            old.atomic_json(folder/'progress.json',dict(complete=False,phase='preparing_inputs',prepared=len(prepared),required=len(family_ids),seconds=time.monotonic()-began))
            print(json.dumps(dict(prepared=condition,endpoint=endpoint,family=[context,prefix,kind],seconds=time.monotonic()-began,peak_rss_bytes=peak_rss_bytes())),flush=True)
        digest=old.state_digest(net);net.double();records=[]
        for family_id in family_ids:
            deadline();record=prepared[family_id];family=torch.load(ROOT/record['path'],map_location='cpu',weights_only=True)
            assert sha(ROOT/record['path'])==record['dense_sha256']
            branches=old.elicit_branches(net,family,incoming[family_id[0]])
            left=old.transition_from_inputs(net,family['left_x'][None]);fb=old.transition_from_inputs(net,family['fb_x'])
            leftrot={r['updates']:np.asarray(r['physical_heading_change_rad'])for r in family['audit']['left_physical']}
            for t in range(61):
                if t not in leftrot:
                    angle=old.parent.native_motion(cfg,[0]*min(t,8),2)
                    if t>8:angle+=old.parent.native_motion(cfg,[0]*(t-8),2)
                    leftrot[t]=np.repeat(angle,1024)
            family_rows=[r for r in numeric if physical_key(r)==family_id]
            for branch in old.BRANCHES:
                branch_rows=[r for r in family_rows if r['branch']==branch]
                if not branch_rows:continue
                curve=branches[branch];choice=old.trajectory(left,curve,8,{8})[8];setup=old.trajectory(fb,choice,3,{2,3})
                regimes={'left':(curve,left,family['reference_heading'].numpy(),leftrot),
                    'fb_B':(setup[2],fb,family['choice_heading'].numpy(),{t:np.zeros(1024)for t in range(62)}),
                    'fb_F':(setup[3],old.transition_from_inputs(net,family['fb_x'].flip(0)),family['choice_heading'].numpy(),{t:np.zeros(1024)for t in range(62)})}
                for regime,(reference,transition,headings,rotations)in regimes.items():
                    todo=[r for r in branch_rows if r['regime']==regime]
                    if not todo:continue
                    deadline();label=todo[0]['family'];old_folder=old.BASE/condition/endpoint/label/regime
                    produced=run_regime(net,reference,transition,todo,headings,rotations,folder/label/regime,old_folder,source,deadline)
                    records.extend(produced);completed.extend(list(key(r))for r in todo)
                    old.atomic_json(folder/'progress.json',dict(complete=False,phase='geometry',completed_numerical_rows=len(completed),required_numerical_rows=cohort['numerical_rows'],seconds=time.monotonic()-began))
                    print(json.dumps(dict(condition=condition,endpoint=endpoint,family=label,regime=regime,completed_rows=len(completed),seconds=time.monotonic()-began)),flush=True)
        records=attach_shams(records,selected)
        lookup={key(r):r for r in records};assert len(lookup)==len(records)==cohort['selected_rows']
        assert set(lookup)=={key(r)for r in selected}
        for r in records:
            partner={**r,'metric':'h_c_RMS'if r['metric']=='raw'else'raw'}
            r['both_metrics_candidate']=bool(r['broad_candidate']and lookup[key(partner)]['broad_candidate'])
        deadline();validate_design(design)
        artifacts={str(p.relative_to(folder)):dict(sha256=sha(p),bytes=p.stat().st_size)for p in folder.rglob('*')if p.is_file()}
        old.atomic_json(folder/'summary.json',dict(complete=True,condition=condition,endpoint=endpoint,
            protocol_sha256=sha(PROTOCOL),source_summary_sha256=sha(source_path),checkpoint_sha256=sha(old.model_path(condition,endpoint)),
            policy_hash_before_float64=digest,training_provenance=provenance,cohort=cohort,rows=records,
            prepared_inputs=list(prepared.values()),seconds=time.monotonic()-began,artifacts=artifacts,
            individual_candidate_rows=sum(r['broad_candidate']for r in records),both_metrics_candidate_rows=sum(r['both_metrics_candidate']for r in records)))
        exit_code=0
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        old.atomic_json(folder/'incomplete.json',dict(complete=False,error=error,completed_numerical_source_keys=completed,
            expected_cohort=cohort,seconds=time.monotonic()-began,cap_seconds=CAP_SECONDS,no_retry_performed=True))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        old.atomic_json(folder/'execution.json',dict(complete=exit_code==0,exit_code=exit_code,error=error,
            started_utc=started,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),seconds=time.monotonic()-began,
            cap_seconds=CAP_SECONDS,protocol_sha256=sha(PROTOCOL),peak_rss_bytes=peak_rss_bytes(),
            summary_sha256=sha(folder/'summary.json')if exit_code==0 else None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-draft',action='store_true');parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--condition',choices=old.parent.CONDITIONS)
    parser.add_argument('--endpoint',choices=('original','clean','noise'),default='original')
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.write_draft:
            assert not PROTOCOL.exists();old.atomic_json(DRAFT,make_draft());print(str(DRAFT))
        elif args.freeze:
            assert not PROTOCOL.exists();design=json.loads(DRAFT.read_text());validate_design(design)
            assert design==make_draft();old.atomic_json(PROTOCOL,design);print(json.dumps(dict(protocol=str(PROTOCOL),sha256=sha(PROTOCOL))))
        else:
            assert args.condition;run(args.condition,args.endpoint)


if __name__=='__main__':main()
