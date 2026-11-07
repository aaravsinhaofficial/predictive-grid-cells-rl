"""Saved-only observer audit/report. Never executes a policy or environment."""
from __future__ import annotations
import collections
import copy
import datetime as dt
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
import native_input_observability as design
import report_native_turn_geometry as base
import report_native_turn_geometry_resolution as fine

OUT=design.BASE/'report'
sha=design.sha;read=design.read;compare=base.compare
META=('context','prefix','kind','branch','family','exact_sham_of')
INITIAL='initial_probability_RMS_at_least_1e_minus4'
EVOLVED='evolved_probability_RMS_at_least_1e_minus4'
STATUSES=('screen_pending','physics_unavailable','initial_probability_below_floor','evolved_probability_below_floor',
    'geometry_failed','conditional_pending','physical_kicks_inconclusive','transverse_observation_inconclusive',
    'heldout_physics_unavailable','heldout_interpolation_unresolved','heldout_equivalence_failed',
    'observable_recovery_failed','observed_candidate')


def primary_key(row):return row['family'],row['regime'],row['retention_updates']
def description(x):
    x=np.asarray(x)
    return dict(n=len(x),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),
        minimum=float(x.min()),maximum=float(x.max()))if len(x)else dict(n=0)


def probability_audit(x):
    x=np.asarray(x);assert x.ndim==2 and x.shape[1]==36 and np.isfinite(x).all()
    assert x.min()>=0 and x.max()<=1
    np.testing.assert_allclose(x.reshape(-1,6,6).sum(-1),1,atol=1e-12,rtol=0)
    return float(np.sqrt(np.mean((x-x.mean(0))**2)))


def static_softmax_audit(probability,centered_logits):
    probability_audit(probability);logits=np.asarray(centered_logits).reshape(-1,6,6)
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(logits.mean(-1),0,rtol=0,atol=1e-12)
    exponential=np.exp(logits-logits.max(-1,keepdims=True))
    expected=(exponential/exponential.sum(-1,keepdims=True)).reshape(-1,36)
    np.testing.assert_allclose(probability,expected,rtol=0,atol=1e-12)


def independent_probability_gates(first,last):
    gates=base.independent_gates(first,last,False)
    gates[INITIAL]=probability_audit(first)>=1e-4;gates[EVOLVED]=probability_audit(last)>=1e-4
    return gates


def audit_alias(row,canonical):
    assert row['regime']==canonical['regime']and row['retention_updates']==canonical['retention_updates']
    compare({k:v for k,v in row.items()if k not in META},{k:v for k,v in canonical.items()if k not in META},'alias')


def selected_probe(metric):
    return metric['bounded_followups'][-1]['result']if metric['bounded_followups']else metric['nominal']


def status(row,conditional=None):
    if row is None:return 'screen_pending'
    if not row['physical_valid']:return 'physics_unavailable'
    gates=row['primary_geometry']['gates']
    if not gates[INITIAL]:return 'initial_probability_below_floor'
    if not gates[EVOLVED]:return 'evolved_probability_below_floor'
    if not all(gates.values()):return 'geometry_failed'
    if conditional is None:return 'conditional_pending'
    probes=[selected_probe(v)for v in conditional['metrics'].values()]
    if any(not p['physical_construction_conclusive']for p in probes):return 'physical_kicks_inconclusive'
    if any(not p['initial_observation_conclusive']for p in probes):return 'transverse_observation_inconclusive'
    if any(not a['valid']for a in conditional['heldout_physics'].values()):return 'heldout_physics_unavailable'
    closures=[v['heldout_closure']for v in conditional['metrics'].values()]
    if any(any(c['unresolved_interpolation_cells']for c in x['checks'].values())for x in closures):return 'heldout_interpolation_unresolved'
    if any(not x['passes']for x in closures):return 'heldout_equivalence_failed'
    if any(not p['all_normal_directions_pass']for p in probes):return 'observable_recovery_failed'
    assert conditional['broad_candidate'];return 'observed_candidate'


def classify(first,metadata):
    n=metadata['count'];ids=np.asarray(metadata['anchor_indices']);nd=len(metadata['names']);a=len(ids)
    baseline=first[:n];changed=first[n:]
    distance,segment,fraction=fine.polygon_projection(changed,baseline)
    anchor=np.broadcast_to(baseline[ids],(nd,2,a,baseline.shape[1])).reshape(changed.shape)
    total=np.linalg.norm(changed-anchor,axis=1);floor=max(1e-8,1e-4*base.radius(baseline)/6)
    types=np.where(total/6<floor,0,np.where(distance/6<floor,1,2)).reshape(nd,2,a)
    labels=(segment+fraction)%n;labels=np.where(types.reshape(-1)==0,np.broadcast_to(ids,(nd,2,a)).reshape(-1),labels)
    types[-1]=3
    return dict(classes=types,phase_labels=labels.reshape(nd,2,a),normal_probability_RMS=distance.reshape(nd,2,a)/6,
        total_probability_RMS=total.reshape(nd,2,a)/6,visibility_floor=floor,
        class_names=['invisible','phase_only','observable_normal','tangent_control'])


def closure_audit(first,last,psi0,psi1,metadata):
    classes=classify(first,metadata);n=metadata['count'];labels=classes['phase_labels'].reshape(-1)
    required=classes['classes'].reshape(-1)!=2;lo=np.floor(labels).astype(int)%n;f=labels-np.floor(labels);hi=(lo+1)%n
    arrays={k:v for k,v in classes.items()if isinstance(v,np.ndarray)};checks={}
    for name,profiles in [('discovery_evolved',last[:,None,:]),('initial',psi0),('evolved',psi1)]:
        baseline=profiles[:n];expected=(1-f[:,None,None])*baseline[lo]+f[:,None,None]*baseline[hi]
        error=np.sqrt(np.mean((profiles[n:]-expected)**2,axis=2))
        angular=np.sqrt(np.mean((baseline-baseline.mean(0))**2,axis=(0,2)));tolerance=np.maximum(1e-8,.05*angular)
        edge=np.sqrt(np.mean((baseline[hi]-baseline[lo])**2,axis=2))
        resolved=(edge<=np.maximum(1e-8,.10*angular)[None])|(f[:,None]==0)
        passed=(error<=tolerance[None])&resolved
        arrays.update({name+'_heldout_error_RMS':error,name+'_heldout_tolerance':tolerance,
            name+'_heldout_edge_RMS':edge,name+'_interpolation_resolved':resolved,
            name+'_heldout_pass':passed,name+'_heldout_expected':expected})
        checks[name]=dict(required_signed_kicks=int(required.sum()),heldout_sequences=profiles.shape[1],
            failed_equivalence_cells=int(((error>tolerance[None])&required[:,None]).sum()),
            unresolved_interpolation_cells=int(((~resolved)&required[:,None]).sum()),
            maximum_required_error_RMS=float(error[required].max())if required.any()else None,passes=bool(passed[required].all()))
    return dict(passes=all(v['passes']for v in checks.values()),checks=checks,
        class_counts={name:int((classes['classes']==i).sum())for i,name in enumerate(classes['class_names'])},
        visibility_floor=classes['visibility_floor'],same_initial_phase_label=True,
        interpretation='Phase-only/invisible/control equivalence; observable normals remain separately tested for recovery.'),arrays


def audit_probe(folder,probe,name,metric,first,last,source_curve,source_last):
    arrays=dict(np.load(folder/name,allow_pickle=False));n=len(first)
    compare(arrays['first'][:n].tolist(),first.tolist(),'initial query baseline')
    compare(arrays['last'][:n].tolist(),last.tolist(),'final query baseline')
    np.testing.assert_array_equal(arrays['starts'][:n],source_curve)
    assert np.all(np.abs(arrays['final_states'][:n]-source_last)<=1e-12+1e-12*np.abs(source_last))
    expected,meta=fine.nominal_design(source_curve,metric)
    np.testing.assert_array_equal(arrays['scaled_directions'],meta['scaled_directions'])
    assert probe['perturbation_design']['anchor_indices']==meta['anchor_indices']
    if 'bounded_method'not in probe['perturbation_design']:np.testing.assert_array_equal(arrays['starts'],expected)
    legacy=copy.deepcopy(probe)
    legacy['physical_status']=('pass'if probe['all_normal_directions_pass']else'failed_recovery')if probe['physical_conclusive']else'inconclusive'
    base.audit_normal(folder,legacy,name,metric)
    classes=classify(arrays['first'],probe['perturbation_design'])
    compare(probe['response_class_counts'],{label:int((classes['classes']==i).sum())for i,label in enumerate(classes['class_names'])})
    compare(probe['response_categories_by_direction'],[dict(name=label,**{k:int((classes['classes'][j]==i).sum())
        for i,k in enumerate(classes['class_names'])})for j,label in enumerate(meta['names'])])
    visible=bool(probe['tangent_nontrivial_fraction']>=.8 and all(r['normal_informative_fraction']>=.8 for r in probe['directions'][:-1]))
    construction=bool(probe['perturbation_design'].get('physical_construction_conclusive',probe['hidden_bound_excess_count']==0))
    assert probe['initial_observation_conclusive']==visible and probe['physical_construction_conclusive']==construction
    expected_status=('inconclusive_physical_construction'if not construction else'inconclusive_transverse_observability'if not visible
        else'pass'if probe['all_normal_directions_pass']else'failed_observable_recovery')
    assert probe['physical_status']==expected_status
    return arrays


def physical_tape_metadata(audit,bank,n,sequences=tuple((a,)for a in range(6))):
    assert audit['valid']==(not audit['issues'])and audit['phases']==n
    assert audit['queries']==[list(s)for s in sequences]
    assert audit['measurement_updates']==len(sequences[0])
    assert len(audit['input_sha256'])==audit['measurement_updates']
    if audit['valid']:
        for t,digest in enumerate(audit['input_sha256']):
            actions=torch.tensor([seq[t]for seq in audit['queries']]).repeat(n)
            assert design.tensor_sha(bank[actions])==digest


def physics_audit(folder,prep):
    records=[];cached={};expected={(r['context'],r['prefix'],r['kind']):r for r in prep['records']}
    for path in sorted((folder/'physics').glob('context*.json')):
        r=read(path);source=expected[(r['context'],r['prefix'],r['kind'])]
        assert r['source_input_sha256']==source['sha256']
        p=folder/'physics'/r['physical_artifact'];assert sha(p)==r['sha256'];data=design.load(p)
        assert design.tensor_sha(data['left_x'])==source['audit']['left_input_sha256']
        for j,key in enumerate(('forward_input_sha256','backward_input_sha256')):assert design.tensor_sha(data['fb_x'][j])==source['audit'][key]
        bank=data['query_bank'];expected_bank=data['left_x'].repeat(6,1);expected_bank[:,1280:1286]=torch.eye(6);expected_bank[:,-1]=0
        assert torch.equal(bank,expected_bank);valid=unavailable=0
        for regime,windows in design.WINDOWS.items():
            assert set(data['references'][regime])=={0,*windows}
            for t in (0,)+windows:
                check=data['probe_checks'][regime][t];compare(r['probe_checks'][regime][str(t)],check)
                physical_tape_metadata(check,bank,128)
                reference=data['references'][regime][t];terminal=data['one_action_terminals'][regime][t]
                expected_t=torch.tensor(source['audit']['reference_episode_t'])+design.REFERENCE[regime]+t
                assert torch.equal(reference['t'],expected_t)
                if check['valid']:
                    valid+=1;assert torch.equal(terminal['t'],(reference['t']+1).repeat_interleave(6))
                    assert torch.equal(terminal['goal'],reference['goal'].repeat_interleave(6,0))
                else:unavailable+=1
                assert all(torch.isfinite(v).all()for v in reference.values())
        cached[r['label']]=data
        records.append(dict(label=r['label'],sha256=r['sha256'],valid_anchor_times=valid,unavailable_anchor_times=unavailable))
    return records,cached


def audit_conditional(folder,row,first,last,source_curve,source_last,physical):
    for t,audit in row['heldout_physics'].items():physical_tape_metadata(audit,physical['query_bank'],128,design.HELDOUT)
    metric_counts={};passed=[]
    for metric,entry in row['metrics'].items():
        assert metric in ('raw','h_c_RMS')
        probe=entry['nominal'];a=audit_probe(folder,probe,entry['nominal_arrays'],metric,first,last,source_curve,source_last)
        for follow in entry['bounded_followups']:
            probe=follow['result'];a=audit_probe(folder,probe,follow['arrays'],metric,first,last,source_curve,source_last)
        closure=entry['heldout_closure']
        if closure is not None:
            z=dict(np.load(folder/closure['arrays'],allow_pickle=False))
            expected,arrays=closure_audit(a['first'],a['last'],z['psi_initial'],z['psi_evolved'],probe['perturbation_design'])
            compare({k:v for k,v in closure.items()if k!='arrays'},expected)
            for name,value in arrays.items():np.testing.assert_allclose(z[name],value,rtol=2e-8,atol=2e-10,err_msg=name)
        valid=all(v['valid']for v in row['heldout_physics'].values());assert (closure is not None)==valid
        ok=bool(probe['physical_conclusive']and probe['all_normal_directions_pass']and closure is not None and closure['passes'])
        assert entry['passes']==ok and entry['physical_status']==probe['physical_status'];passed.append(ok)
        metric_counts[metric]=dict(physical_status=probe['physical_status'],class_counts=probe['response_class_counts'],
            directions=probe['response_categories_by_direction'],normal_directions_pass=sum(d['passes']for d in probe['directions'][:-1]),
            heldout_closure_pass=None if closure is None else closure['passes'],passes=ok)
    assert len(passed)==2 and row['broad_candidate']==all(passed)
    return metric_counts


def gather_rows(folder,cohort):
    screen=read(folder/'screen_summary.json')if(folder/'screen_summary.json').exists()else None
    conditionals=read(folder/'conditional_summary.json')if(folder/'conditional_summary.json').exists()else None
    rows=screen['rows']if screen else[read(p)for p in(folder/'profiles').glob('*/*_t*.json')]
    conditional=conditionals['rows']if conditionals else[read(p)for p in(folder/'conditional').glob('*/*/*/summary.json')]
    result={primary_key(r):r for r in rows};other={primary_key(r):r for r in conditional}
    assert len(result)==len(rows)and len(other)==len(conditional)
    for item in cohort['families']:
        if item['exact_sham_of']:
            for regime,windows in design.WINDOWS.items():
                for t in windows:
                    canonical=(item['exact_sham_of'],regime,t);key=(item['family'],regime,t)
                    for target in (result,other):
                        if canonical in target:
                            if key in target:audit_alias(target[key],target[canonical])
                            else:target[key]={**target[canonical],**item}
    return result,other,screen,conditionals


def audit_condition(condition):
    folder=design.BASE/condition;execution=read(folder/'execution.json');cohort=read(folder/'cohort.json')
    assert execution['protocol_sha256']==cohort['protocol_sha256']==sha(design.PROTOCOL)
    terminal=read(folder/'summary.json')if(folder/'summary.json').exists()else None
    if terminal:
        assert terminal['complete']and execution['complete']and execution['exit_code']==0
        assert execution['summary_sha256']==sha(folder/'summary.json')
        for stem in ('screen','conditional'):
            assert read(folder/(stem+'_summary.json'))['complete']
            assert terminal[stem+'_summary_sha256']==sha(folder/(stem+'_summary.json'))
        for path,r in terminal['artifacts'].items():assert (folder/path).stat().st_size==r['bytes']and sha(folder/path)==r['sha256'],path
    else:assert not execution['complete']and(folder/'incomplete.json').exists()
    source_path=design.old.BASE/condition/'original'/'summary.json';prep_path=design.old.BASE/'inputs'/condition/'summary.json'
    assert sha(source_path)==cohort['source_summary_sha256']and sha(prep_path)==cohort['input_summary_sha256']
    source=read(source_path);prep=read(prep_path);compare(cohort['families'],design.family_inventory(source))
    expected={(r['family'],regime,t)for r in cohort['families']for regime,windows in design.WINDOWS.items()for t in windows}
    assert len(expected)==3328 and expected=={tuple(k)for k in cohort['expected_primary_keys']}
    physics_records,physics=physics_audit(folder,prep)
    raw,conditional,screen,full_conditional=gather_rows(folder,cohort)
    assert set(raw)<=expected and set(conditional)<=set(raw)
    if terminal:assert len(raw)==3328 and len(physics_records)==64
    numerical={};output=[];audited_profiles=0;normal_cases=0;profile_cache={};state_cache={}
    families={r['family']:r for r in cohort['families']}
    for key in sorted(expected):
        label,regime,t=key;item=families[label];row=raw.get(key);canonical=(item['exact_sham_of']or label,regime,t)
        out=dict(condition=condition,**item,regime=regime,retention_updates=t,
            latest_observation_updates_after_cue=design.REFERENCE[regime]+t+1,status=status(row,conditional.get(key)))
        if row is not None:
            physical=physics[design.physical_label(item)];valid=all(physical['probe_checks'][regime][j]['valid']for j in(0,t))
            assert valid==row['physical_valid'];out['physical_valid']=valid
            assert row['measurement_updates']==1 and row['reference_updates_after_cue']==design.REFERENCE[regime]
            assert row['latest_observation_updates_after_cue']==design.REFERENCE[regime]+t+1
            out['geometry_qualified']=row['geometry_qualified']
            if not valid:out['unavailability']=row['unavailability']
            elif canonical not in numerical:
                cache=(canonical[0],regime)
                if cache not in profile_cache:
                    profile_cache.clear();state_cache.clear()
                    profile_cache[cache]=dict(np.load(folder/'profiles'/canonical[0]/(regime+'.npz'),allow_pickle=False))
                    saved_times=[q for q in(0,)+design.WINDOWS[regime]if physical['probe_checks'][regime][q]['valid']]
                    assert set(profile_cache[cache])=={kind+str(q)for kind in('probability_','logit_')for q in saved_times}
                    for q in saved_times:static_softmax_audit(profile_cache[cache]['probability_'+str(q)],profile_cache[cache]['logit_'+str(q)])
                    state_cache[cache]=design.source_states(design.old.BASE/condition/'original',source,canonical[0],regime)
                saved=profile_cache[cache];states,state_sha=state_cache[cache];assert row['source_state_sha256']==state_sha
                first,last=saved['probability_0'],saved[f'probability_{t}'];g=design.geometry(first,last)
                compare(row['primary_geometry'],g);assert independent_probability_gates(first,last)==g['gates']
                assert row['geometry_qualified']==all(g['gates'].values())
                for ckey,aa,bb in [('diagnostic_probability_centered',first,last),('diagnostic_logit_centered',saved['logit_0'],saved[f'logit_{t}'])]:
                    compare(row[ckey],design.old.geometry_screen(aa,bb,True))
                    assert base.independent_gates(aa,bb,True)==row[ckey]['gates']
                heads=physical['references'][regime][0]['hd'].numpy();rotation=design.wrap(physical['references'][regime][t]['hd'].numpy()-heads)
                compare(row['physical_phase'],fine.phase_audit(first,last,heads,rotation,False))
                precision=row['precision_checks'];assert all(p['passes']and p['max_absolute_probability_error']<=1e-12 for p in precision)
                assert {p['retention_updates']for p in precision}=={q for q in(0,max(design.WINDOWS[regime]))if physical['probe_checks'][regime][q]['valid']}
                p=last.reshape(128,6,6);tv=float((.5*np.abs(p[:64]-p[64:]).sum(-1)).mean())
                value=dict(initial_probability_RMS=probability_audit(first),evolved_probability_RMS=probability_audit(last),
                    evolved_antipodal_TV=tv,failed_gates=[name for name,passed in g['gates'].items()if not passed],gates=g['gates'],
                    radius_retention=g['radius_retention'],departure_p90=g['residual_to_reference_curve_over_radius']['p90'],
                    diagnostic_probability_centered_pass=row['diagnostic_probability_centered']['necessary_geometry_pass'],
                    diagnostic_logit_centered_pass=row['diagnostic_logit_centered']['necessary_geometry_pass'],
                    diagnostic_probability_centered_gates=row['diagnostic_probability_centered']['gates'],
                    diagnostic_logit_centered_gates=row['diagnostic_logit_centered']['gates'],
                    physical_phase=row['physical_phase'])
                if canonical in conditional:
                    c=conditional[canonical];normal_cases+=1
                    value['conditional_metrics']=audit_conditional(folder/'conditional'/canonical[0]/regime/f't{t}',c,first,last,states['t0'],states[f't{t}'],physical)
                numerical[canonical]=value;audited_profiles+=1
            if valid:out.update(numerical[canonical])
        output.append(out)
    if screen:
        assert screen['geometry_qualified_rows']==sum(r.get('geometry_qualified',False)for r in output)
        assert screen['unavailable_rows']==sum(r.get('physical_valid')is False for r in output)
        assert set(map(tuple,screen['numerical_qualified_cases']))=={primary_key(r)for r in raw.values()if r['geometry_qualified']and not r['exact_sham_of']}
    if full_conditional:
        assert len(conditional)==full_conditional['named_cases']==sum(r.get('geometry_qualified',False)for r in output)
        assert full_conditional['candidates']==sum(r['status']=='observed_candidate'for r in output)
    if terminal:
        assert terminal['candidates']==sum(r['status']=='observed_candidate'for r in output)
        assert terminal['geometry_qualified_rows']==sum(r.get('geometry_qualified',False)for r in output)
    validation=dict(condition=condition,execution_complete=bool(terminal),expected_primary_rows=3328,completed_screen_rows=len(raw),
        numerical_geometry_rows_recomputed=audited_profiles,numerical_normal_cases_recomputed=normal_cases,
        named_conditional_rows=len(conditional),physical_families=len(physics_records),physics_records=physics_records,
        artifact_hashes_checked=len(terminal['artifacts'])if terminal else 0,execution_sha256=sha(folder/'execution.json'),
        summary_sha256=sha(folder/'summary.json')if terminal else None,
        incomplete_sha256=sha(folder/'incomplete.json')if not terminal else None)
    return output,validation


def totals(rows):
    completed=[r for r in rows if r['status']!='screen_pending'];valid=[r for r in rows if r.get('physical_valid')]
    return dict(expected=len(rows),screened=len(completed),physical_valid=len(valid),
        reference_visible=sum(r.get('gates',{}).get(INITIAL,False)for r in rows),
        evolved_visible=sum(r.get('gates',{}).get(EVOLVED,False)for r in rows),
        both_visible=sum(r.get('gates',{}).get(INITIAL,False)and r.get('gates',{}).get(EVOLVED,False)for r in rows),
        geometry_qualified=sum(r.get('geometry_qualified',False)for r in rows),
        conditional_completed=sum('conditional_metrics'in r for r in rows),candidates=sum(r['status']=='observed_candidate'for r in rows),
        diagnostic_probability_centered_pass=sum(r.get('diagnostic_probability_centered_pass',False)for r in rows),
        diagnostic_logit_centered_pass=sum(r.get('diagnostic_logit_centered_pass',False)for r in rows),
        statuses={s:sum(r['status']==s for r in rows)for s in STATUSES},
        overlapping_failed_gates=dict(collections.Counter(g for r in rows for g in r.get('failed_gates',[]))),
        initial_probability_RMS=description([r['initial_probability_RMS']for r in valid]),
        evolved_probability_RMS=description([r['evolved_probability_RMS']for r in valid]),
        evolved_antipodal_TV=description([r['evolved_antipodal_TV']for r in valid]))


def render(summary):
    rows=summary['rows'];colors={'left':'#5279b4','fb_B':'#d98947','fb_F':'#469680'}
    fig,axes=plt.subplots(1,2,figsize=(8,3.7),sharey=True)
    for ax,condition,title in zip(axes,('gc100','pgc33'),('Grid','PGC + grid')):
        for i,regime in enumerate(design.WINDOWS):
            selected=[r for r in rows if r['condition']==condition and r['regime']==regime and r.get('physical_valid')]
            x=np.arange(len(selected));jitter=((x*0.61803398875)%1-.5)*.26
            y=np.array([r['evolved_probability_RMS']for r in selected])*100
            ax.scatter(i+jitter,y,s=3,color=colors[regime],alpha=.12,rasterized=True)
            if len(y):ax.plot([i-.18,i+.18],[np.median(y)]*2,color='black',lw=2)
        ax.axhline(.01,color='#666',ls='--',lw=1);ax.set_xticks(range(3),['LEFT','F/B: B','F/B: F'])
        ax.set_title(title);ax.set_yscale('symlog',linthresh=1e-5);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Angular variation in action probability\n(RMS, percentage points)')
    fig.tight_layout();fig.savefig(OUT/'probability_sensitivity.png',dpi=180);fig.savefig(OUT/'probability_sensitivity.pdf');plt.close(fig)
    gate_names=[INITIAL,EVOLVED,'max_edge_below_10percent_radius','nonlocal_separation_above_5percent',
        'p90_invariance_residual_below_5percent','phase_coverage_80percent','phase_degree_one','phase_order_preserved']
    labels=['Initial\nvisible','Later\nvisible','Spacing','Separated','Near same\ncurve','Phase\ncoverage','One\nturn','Phase\norder']
    groups=[(c,r)for c in ('gc100','pgc33')for r in design.WINDOWS];matrix=[];denominators=[]
    for condition,regime in groups:
        valid=[r for r in rows if r['condition']==condition and r['regime']==regime and r.get('physical_valid')]
        denominators.append(len(valid));matrix.append([100*sum(r['gates'][g]for r in valid)/len(valid)if valid else np.nan for g in gate_names])
    fig,ax=plt.subplots(figsize=(9,3.6));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(6),[f"{'Grid'if c=='gc100'else'PGC + grid'} · {r.replace('_', ': ')}"for c,r in groups])
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.0f}%'if np.isfinite(value)else'—',ha='center',va='center',color='white'if value>60 else'#222',fontsize=9)
    ax.set_title('Each criterion, checked separately');fig.colorbar(im,ax=ax,pad=.02,label='% of available cases passing')
    fig.tight_layout();fig.savefig(OUT/'geometry_criteria.png',dpi=180);fig.savefig(OUT/'geometry_criteria.pdf');plt.close(fig)


def write_readme(summary):
    lines=['# What the six-action observer found','',
        'We took independent copies of each saved RNN state, supplied each of the six truthful actions, and read the next action probabilities. This tests whether angular information becomes visible under different inputs. The queries were never inserted into the memory-retention trajectory.','',
        '![Probability sensitivity](probability_sensitivity.png)','',
        'Dots show every available case across all fixed windows, including exact aliases. Black lines are medians. The dashed line is the prespecified 0.01-percentage-point visibility floor. This is a numerical screen, not a navigation-benefit threshold.','',
        '| Original model | Available | Visible at both ends | Geometry passed | Conditional tests completed | Candidates |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for condition,label in [('gc100','Grid'),('pgc33','PGC + grid')]:
        v=summary['populations'][condition]['named'];lines.append(f"| {label} | {v['physical_valid']}/{v['expected']} | {v['both_visible']} | {v['geometry_qualified']} | {v['conditional_completed']} | {v['candidates']} |")
    lines.extend(['','![Separate geometry criteria](geometry_criteria.png)','',
        'A failed geometry criterion means the measured pattern did not meet that particular ring screen. It does not mean the RNN carried no information. The full table retains overlapping failures, initial and later probability sensitivity, and the centered-probability/logit diagnostics.','',
        'Qualifying curves receive the fixed broad perturbations in both raw and h/c-RMS metrics. Observable-normal recovery, phase-only responses, invisible responses, and held-out equivalence are separate results. Insufficient transverse visibility is inconclusive; it is not failed recovery. Held-out queries always use the original discovery phase label.','',
        f"In this batch, **{sum(v['named']['conditional_completed']for v in summary['populations'].values())} named cases** reached completed conditional tests. If that count is zero, no perturbation or held-out query was executed; there are no measured recovery failures to report.",'',
        'There is one original seed per population. The 3,328 cases per population are repeated measurements of those two networks, not thousands of independent models. LEFT has 1,280 cases; each F/B section has 1,024. Exact aliases remain in these denominators; numerical-only totals are also saved. F/B is a periodic-input diagnostic, and phase drift is reported against zero body rotation.','',
        f"Execution complete for both cohorts: **{summary['all_cohorts_complete']}**. Pending or physically unavailable cases remain explicit and are never counted as scientific negatives.",'',
        'This report only reads saved arrays and checks their arithmetic and provenance. It creates no environment, policy rollout, recurrent transition, or additional query. Full records: [summary](summary.json), [validation](validation.json).'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        protocol=read(design.PROTOCOL);design.validate_protocol(protocol);rows=[];validations=[]
        for condition in ('gc100','pgc33'):
            r,v=audit_condition(condition);rows.extend(r);validations.append(v);print(json.dumps(v,default=str),flush=True)
        populations={c:dict(named=totals([r for r in rows if r['condition']==c]),
            numerical=totals([r for r in rows if r['condition']==c and not r['exact_sham_of']]))for c in ('gc100','pgc33')}
        windows=[dict(condition=c,regime=regime,retention_updates=t,**totals([r for r in rows if r['condition']==c and r['regime']==regime and r['retention_updates']==t]))
            for c in ('gc100','pgc33')for regime,ws in design.WINDOWS.items()for t in ws]
        summary=dict(complete=True,all_cohorts_complete=all(v['execution_complete']for v in validations),
            original_models=2,seeds_per_population=1,expected_primary_rows=6656,rows=rows,populations=populations,windows=windows,
            interpretation='Repeated controlled histories of one original seed pair; visible information, geometry and attraction are distinct.',protocol_sha256=sha(design.PROTOCOL))
        design.old.atomic_json(OUT/'summary.json',summary)
        closure=['scripts/report_native_input_observability.py','tests/test_report_native_input_observability.py',
            'scripts/report_native_turn_geometry.py','scripts/report_native_turn_geometry_resolution.py',
            'tests/test_report_native_turn_geometry.py','tests/test_report_native_turn_geometry_resolution.py']
        validation=dict(complete=True,cohorts=validations,protocol_sha256=sha(design.PROTOCOL),summary_sha256=sha(OUT/'summary.json'),
            arithmetic_source_sha256={p:sha(ROOT/p)for p in closure},created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            no_model_or_environment_execution=True,scope='Saved probability/geometry/phase/normal/closure arithmetic plus physical-input hashes and archived provenance; recorded numerical precision is checked, never rerun.')
        design.old.atomic_json(OUT/'validation.json',validation);render(summary);write_readme(summary)
        design.old.atomic_json(OUT/'artifacts.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in OUT.iterdir()if p.is_file()and p.name!='artifacts.json'})


if __name__=='__main__':main()
