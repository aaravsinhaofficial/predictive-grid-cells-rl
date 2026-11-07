"""Independent saved-array audit; no network, environment or recurrent calls."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_turn_counterbalanced_readout'
PROTOCOL=ROOT/'docs/research/native_turn_counterbalanced_readout_protocol.json'
CONDITIONS=('gc100','pgc33');ENDPOINTS=('original','clean','noise')
KINDS=('turn_only','probe_glimpse');BRANCHES=('native','ordinary','pre_cue_reset_native','pre_cue_reset_ordinary')
FIELDS=('progress_mm','average_policy_progress_mm','progress_above_average_policy_mm','best_common_action_progress_mm',
        'mean_antipodal_policy_tv','body_goal_alignment_rad','turn_weighted_body_goal_alignment_rad')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def arr(x):return x.numpy()if isinstance(x,torch.Tensor)else np.asarray(x)
def wrap(x):return(np.asarray(x)+np.pi)%(2*np.pi)-np.pi
def softmax(x):
    v=np.exp(x-np.max(x,axis=1,keepdims=True));return v/v.sum(1,keepdims=True)
def close(a,b,tol=5e-12):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all()
    error=float(np.max(np.abs(a-b),initial=0));assert error<=tol,(error,tol)
    return error


def recompute(q,progress,alignment):
    q=np.asarray(q,dtype=float);progress=np.asarray(progress,dtype=float);alignment=np.asarray(alignment,dtype=float)
    assert q.shape==progress.shape==alignment.shape==(128,6)
    close(q.sum(1),np.ones(128));assert (q>=0).all()
    own=np.sum(q*progress,1)*1000;average=progress@q.mean(0)*1000;tv=.5*np.abs(q[:64]-q[64:]).sum(1)
    fields=dict(progress_mm=float(own.mean()),average_policy_progress_mm=float(average.mean()),
        progress_above_average_policy_mm=float(np.mean(own-average)),best_common_action_progress_mm=float(progress.mean(0).max()*1000),
        mean_antipodal_policy_tv=float(tv.mean()),body_goal_alignment_rad=float(np.sum(q*alignment,1).mean()),
        turn_weighted_body_goal_alignment_rad=float(np.sum(q[:,:2]*alignment[:,:2],1).mean()))
    return fields,dict(case_progress_mm=own,case_average_policy_progress_mm=average,antipodal_policy_tv=tv)


def physical_expected(terminal,cfg):
    """Independent algebra of one native, noise-free update from zero motors."""
    p=arr(terminal['pos']).astype(float);goal=arr(terminal['goal']).astype(float);hd=arr(terminal['hd']).astype(float)
    assert p.shape==(128,2)and all(np.count_nonzero(arr(terminal[k]))==0 for k in ('vf','vs','vr'))
    dt=cfg['action_repeat']/60.;k=1-math.exp(-dt/.25)
    target=np.array([[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],dtype=float)
    vf=target[:,0]*cfg['v_max']*k;vs=target[:,1]*cfg['v_max']*k
    newhd=wrap(hd[:,None]+target[:,2]*cfg['rot_max']*k*dt)
    displacement=np.stack((vf*np.cos(newhd)-vs*np.sin(newhd),vf*np.sin(newhd)+vs*np.cos(newhd)),2)*dt
    endpoint=p[:,None]+displacement
    assert np.max(np.abs(endpoint))<cfg['arena_half']-.08 and np.all(np.linalg.norm(goal[:,None]-endpoint,axis=2)>cfg['goal_radius'])
    assert np.max(arr(terminal['t']))+1<cfg['episode_len']
    vector=goal-p;unit=vector/np.linalg.norm(vector,axis=1,keepdims=True)
    progress=np.sum((endpoint-p[:,None])*unit[:,None],2)
    initial=np.abs(wrap(np.arctan2(vector[:,1],vector[:,0])-hd))
    remaining=goal[:,None]-endpoint;alignment=initial[:,None]-np.abs(wrap(np.arctan2(remaining[:,:,1],remaining[:,:,0])-newhd))
    return dict(endpoint=endpoint,heading=newhd,progress_m=progress,body_goal_angle_error_reduction_rad=alignment)


def coverage(rows):
    expected={(c,e,i,p,k,b)for c in CONDITIONS for e in ENDPOINTS for i in range(16)for p in (0,1)for k in KINDS for b in BRANCHES}
    tags=[tuple(r[k]for k in ('condition','endpoint','context','prefix','kind','branch'))for r in rows]
    assert len(tags)==len(set(tags))==1536 and set(tags)==expected
    return dict(zip(tags,rows))


def rejected_equivalence_cohort(summary):
    """The sealed failure stays rejected; this audit is only descriptive."""
    expected={('gc100',e,11,1,'probe_glimpse',b)for e in ENDPOINTS for b in BRANCHES}
    assert summary['complete']is False and len(summary['failures'])==12
    found={tuple(r[k]for k in ('condition','endpoint','context','prefix','kind','branch'))for r in summary['failures']}
    assert found==expected and all('prefix embedding' in r['reason']for r in summary['failures'])
    return expected


def verified_old_assignment_indices(target,bearing,old_target,actual_labels,prefix):
    target=np.asarray(target);actual_labels=np.asarray(actual_labels)
    assert target.shape==(128,)and actual_labels.shape==(16,)
    assert np.max(np.abs(wrap(target-(bearing+np.arange(128)*2*np.pi/128))))<3e-5
    cases=np.flatnonzero(actual_labels==prefix);assert len(cases)==8
    delta=np.abs(wrap(target[:,None]-np.asarray(old_target)[cases][None]));indices=delta.argmin(0)
    assert delta[indices,np.arange(8)].max()<3e-5
    assert set(indices)==set(range(prefix*64,prefix*64+64,8)),(prefix,indices)
    return cases,indices


def audit(out):
    errors={};verified={}
    def check(path,expected=None):
        path=Path(path);key=str(path.relative_to(ROOT))
        if key not in verified:verified[key]=dict(sha256=sha(path),bytes=path.stat().st_size)
        if expected is not None:assert verified[key]==expected,key
        return path
    def record_error(name,a,b,tol=5e-12):errors[name]=max(errors.get(name,0.),close(a,b,tol))
    design=read(check(PROTOCOL));summary=read(check(BASE/'summary.json'))
    rejected=rejected_equivalence_cohort(summary);assert(BASE/'incomplete.json').exists()
    assert summary['protocol_sha256']==sha(PROTOCOL)and summary['producer_sha256']==sha(ROOT/'scripts/native_turn_counterbalanced_readout.py')
    for category in ('source_sha256','asset_sha256'):
        for path,digest in design[category].items():assert sha(check(ROOT/path))==digest,path
    for path,entry in summary['input_sha256'].items():check(ROOT/path,entry)
    artifacts=read(BASE/'artifacts.json')
    for name,entry in artifacts.items():check(BASE/name,entry)
    cohort=read(BASE/'cohort.json');assert cohort['protocol_sha256']==sha(PROTOCOL)
    coverage(cohort['families']);rows=coverage(summary['rows'])
    shared=summary['shared_angle_checks'];assert shared['records']==1536 and shared['sha256']==sha(BASE/shared['path'])
    checks=coverage(read(BASE/shared['path'])['records']);assert len(summary['aggregates'])==144
    old=read(ROOT/'docs/research/native_turn_teacher_preflight_protocol.json');cache={};discrepancies=[];physics_count=0
    for condition in CONDITIONS:
        bankpath=ROOT/old['banks'][condition]['path'];assert sha(bankpath)==old['banks'][condition]['sha256'];bank=load(bankpath);cfg=bank['config']
        actor=load(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt')['policy'];w=arr(actor['pi.weight']).astype(float);b=arr(actor['pi.bias']).astype(float)
        geometries={e:read(ROOT/f'results/native_turn_geometry/{condition}/{e}/summary.json')for e in ENDPOINTS}
        prep=read(ROOT/f'results/native_turn_geometry/inputs/{condition}/summary.json');input_records={r['path']:r for r in prep['records']}
        paired_folders={e:(ROOT/f'results/native_turn_original_memory/{condition}'if e=='original'else ROOT/f'results/native_turn_distillation/{condition}/{e}/final_paired')for e in ENDPOINTS}
        paired_summaries={e:read(p/'summary.json')for e,p in paired_folders.items()}
        for context in bank['contexts']:
            ci=context['context_id']
            for kind in KINDS:
                rawpath=ROOT/f'results/native_turn_teacher_preflight/{condition}/context{ci:02d}_{kind}.pt';raw=load(rawpath)
                ids=[raw['labels'].index(f'turn_{j}')for j in range(14)];pa=arr(raw['previous_actions'][ids]);assert np.array_equal(pa,np.repeat(pa[:1],14,axis=0))
                assert set(pa[0])=={0,1}and np.count_nonzero(pa[0]==0)==8
                for field in ('teacher_x','student_native_x','student_ordinary_x'):assert np.array_equal(arr(raw[field][ids,:,1280:1286]),np.eye(6,dtype=np.float32)[pa])
                paired={}
                for e,p in paired_folders.items():
                    name=f'context{ci:02d}_{kind}';check(p/(name+'.json'),paired_summaries[e]['artifacts'][name+'.json'])
                    assert read(p/(name+'.json'))['source_raw_sha256']==sha(rawpath)
                    paired[e]=load(check(p/(name+'.pt'),paired_summaries[e]['artifacts'][name+'.pt']))
                for prefix in (0,1):
                    name=f'context{ci:02d}_prefix{prefix}_{kind}';bankpath=BASE/f'physics/{condition}/{name}.pt';physical=load(bankpath);terminal=physical['terminal'];physics=physical['physics'];physics_count+=1
                    family_path=ROOT/f'results/native_turn_geometry/inputs/{condition}/{name}.pt';assert physical['input_sha256']==sha(family_path)
                    assert sha(family_path)==input_records[name+'.pt']['sha256']
                    family=load(family_path);j8=next(r for r in family['audit']['left_physical']if r['updates']==8)
                    assert np.array_equal(arr(terminal['pos']),arr(family['physical']['pos'][-1]))and np.array_equal(arr(terminal['mdir']),arr(family['physical']['mdir'][-1]))
                    assert np.array_equal(arr(terminal['hd']),arr(family['choice_heading']))and np.array_equal(arr(terminal['t']),j8['episode_t'])
                    assert np.array_equal(arr(terminal['goal']),np.repeat(arr(context['goal_center'])[None],128,axis=0))
                    analytic=physical_expected(terminal,cfg)
                    for key in ('endpoint','progress_m'):record_error('analytic_'+key,arr(physics[key]),analytic[key],2e-7)
                    record_error('analytic_heading',wrap(arr(physics['heading'])-analytic['heading']),np.zeros((128,6)),2e-6)
                    record_error('analytic_alignment',arr(physics['body_goal_angle_error_reduction_rad']),analytic['body_goal_angle_error_reduction_rad'],3e-6)
                    vector=arr(context['goal_center']).astype(float)-arr(context['endpoint']);bearing=np.arctan2(vector[1],vector[0])
                    cases,ix=verified_old_assignment_indices(arr(family['target_choice_heading']),bearing,arr(raw['target_heading']),pa[0],prefix)
                    for endpoint in ENDPOINTS:
                        for branch in BRANCHES:
                            tag=(condition,endpoint,ci,prefix,kind,branch);row=rows[tag];audit=checks[tag]
                            assert row['shared_angle_checks_passed']==(tag not in rejected)
                            assert bool(audit.get('error'))==(tag in rejected)
                            label=name+'_'+branch;source=row['exact_sham_of']or label
                            if row['exact_sham_of']:
                                expected='native'if branch=='ordinary'else'pre_cue_reset_native'
                                assert branch.endswith('ordinary')and(condition=='gc100'or kind=='turn_only')and source==name+'_'+expected
                            statepath=ROOT/f'results/native_turn_geometry/{condition}/{endpoint}/{source}/left/baseline_states.npz'
                            check(statepath,geometries[endpoint]['artifacts'][source+'/left/baseline_states.npz']);assert sha(statepath)==row['source_state_sha256']
                            original_rows=[r for r in geometries[endpoint]['rows']if r['family']==label and r['regime']=='left']
                            assert original_rows and {r['exact_sham_of']for r in original_rows}=={row['exact_sham_of']}
                            with np.load(statepath)as z:state=z['t8']
                            with np.load(BASE/f'{condition}/{endpoint}/{label}.npz')as z:output={k:z[k]for k in z.files}
                            assert np.array_equal(output['h'],state[:,:256])and np.array_equal(output['heading'],arr(terminal['hd']))
                            logits=state[:,:256]@w.T+b;q=softmax(logits)
                            record_error('actor_logits',output['logits'],logits);record_error('actor_probability',output['probabilities'],q)
                            assert np.array_equal(output['action_progress_m'],arr(physics['progress_m']))and np.array_equal(output['action_alignment_rad'],arr(physics['body_goal_angle_error_reduction_rad']))
                            values,arrays=recompute(q,output['action_progress_m'],output['action_alignment_rad'])
                            for key,value in values.items():record_error('aggregate_'+key,row[key],value)
                            for key,value in arrays.items():record_error('phase_'+key,output[key],value)
                            record_error('mean_action_distribution',row['average_action_distribution'],q.mean(0))
                            if branch=='pre_cue_reset_ordinary':assert np.array_equal(state,np.repeat(state[:1],128,0))and np.array_equal(q,np.repeat(q[:1],128,0))
                            cache[tag]=q.copy()
                            if row['exact_sham_of']:
                                assert np.array_equal(q,cache[condition,endpoint,ci,prefix,kind,expected])
                            if tag not in rejected:assert audit['cases']==cases.tolist()and audit['geometry_indices']==ix.tolist()
                            v=paired[endpoint]['variants'][branch];oldq=arr(v['first_choice_probabilities'])[cases].astype(float);oldq/=oldq.sum(1,keepdims=True)
                            tolerance=design['numerical_provenance_tolerances'];oldstate=arr(v['first_choice_state_after_input'])[cases]
                            herror=float(np.max(np.abs(state[ix,:256]-oldstate[:,:256])));cerror=float(np.max(np.abs(state[ix,256:]-oldstate[:,256:])))
                            oldlog=arr(v['first_choice_logits'])[cases].astype(float)
                            logerror=float(np.max(np.abs((logits[ix]-logits[ix].mean(1,keepdims=True))-(oldlog-oldlog.mean(1,keepdims=True)))))
                            perror=float(np.max(np.abs(q[ix]-arr(v['first_choice_probabilities'])[cases])))
                            downstream_checks=dict(hidden=bool(herror<=tolerance['hidden']),
                                cell=bool(np.all(np.abs(state[ix,256:]-oldstate[:,256:])<=tolerance['cell_atol']+tolerance['cell_rtol']*np.abs(oldstate[:,256:]))),
                                centered_logits=bool(logerror<=tolerance['centered_logits']),probability=bool(perror<=tolerance['probability']))
                            if tag not in rejected:assert all(downstream_checks.values())
                            oldphysics=arr(raw['first_update_physics']['progress_m'])[cases].astype(float);newphysics=arr(physics['progress_m'])[ix].astype(float)
                            difference=1000*((q[ix]*newphysics).sum(1)-(oldq*oldphysics).sum(1))
                            bound=1000*((np.abs(q[ix]-oldq)*np.abs(oldphysics)).sum(1)+(q[ix]*np.abs(newphysics-oldphysics)).sum(1))
                            if tag not in rejected:
                                record_error('shared_expected_progress_difference',audit['shared_expected_progress_difference_mm'],difference)
                                record_error('shared_expected_progress_bound',audit['shared_expected_progress_absolute_error_bound_mm'],bound)
                            assert np.all(np.abs(difference)<=bound+1e-12)
                            discrepancies.append(dict(condition=condition,endpoint=endpoint,context=ci,prefix=prefix,kind=kind,branch=branch,
                                old_input_equivalence_rejected=tag in rejected,downstream_checks=downstream_checks,
                                hidden_max_abs=herror,cell_max_abs=cerror,centered_logits_max_abs=logerror,probability_max_abs=perror,
                                max_abs_shared_score_error_mm=float(np.max(np.abs(difference))),max_component_bound_mm=float(bound.max()),mean_abs_shared_score_error_mm=float(np.mean(np.abs(difference)))))
    assert physics_count==128 and len(cache)==1536
    aggregate_tags=set()
    for row in summary['aggregates']:
        c,e,k,b,p=(row[f]for f in ('condition','endpoint','kind','branch','prefix'));tag=(c,e,k,b,p);assert tag not in aggregate_tags;aggregate_tags.add(tag)
        selected=[r for t,r in rows.items()if(t[0],t[1],t[4],t[5])==(c,e,k,b)and(p=='both'or t[3]==p)]
        assert len(selected)==(32 if p=='both'else 16)and row['cases']==128*len(selected)
        for field in FIELDS:record_error('pooled_'+field,row[field],np.mean([r[field]for r in selected]))
    downstream_failure_keys={key:[{k:r[k]for k in ('condition','endpoint','context','prefix','kind','branch')}for r in discrepancies if not r['downstream_checks'][key]]for key in ('hidden','cell','centered_logits','probability')}
    record=dict(arithmetic_audit_complete=True,scope='Standalone descriptive audit of the internally consistent full-circle cohort from a failed-equivalence run.',
        scientific_run_complete=False,old16_input_equivalence='rejected',failed_equivalence_rows=summary['failures'],
        downstream_cell_failure_rows=[{k:r[k]for k in ('condition','endpoint','context','prefix','kind','branch')}for r in discrepancies if not r['downstream_checks']['cell']],
        downstream_failure_counts={k:len(v)for k,v in downstream_failure_keys.items()},downstream_failure_keys=downstream_failure_keys,
        maximum_shared_discrepancies={key:max(r[key]for r in discrepancies)for key in ('hidden_max_abs','cell_max_abs','centered_logits_max_abs','probability_max_abs','max_abs_shared_score_error_mm','max_component_bound_mm')},
        source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),summary_sha256=sha(BASE/'summary.json'),
        scientific_artifacts_sha256=sha(BASE/'artifacts.json'),families=1536,phase_rows=196608,physics_banks=128,aggregates=144,
        exact_shams=True,reset_ordinary_all_phase_identity=True,maximum_absolute_errors=errors,shared_case_precision=discrepancies,
        no_recurrent_execution=True,no_network_instantiation=True,no_environment_execution=True,no_optimization=True,
        verified_artifacts=verified)
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'report/independent_validation.json');args=p.parse_args();torch.set_num_threads(1);audit(args.out)
