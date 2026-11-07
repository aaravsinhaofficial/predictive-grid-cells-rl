"""Prospective static actor readout of archived full-circle native-turn states.

No recurrent transition, CNN, optimization or policy rollout is executed here.
The only new environment computation is a once-per-physical-family six-action
noise-free counterfactual, after the same archived first-choice motor brake.
--run requires a separately sealed, reviewed protocol. Draft generation reads
no neural states or action scores.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
BASE=ROOT/'results/native_turn_counterbalanced_readout'
PROTOCOL=ROOT/'docs/research/native_turn_counterbalanced_readout_protocol.json'
DRAFT=ROOT/'docs/research/native_turn_counterbalanced_readout_protocol_draft.json'
GEOMETRY=ROOT/'results/native_turn_geometry'
CONDITIONS=('gc100','pgc33')
ENDPOINTS=('original','clean','noise')
KINDS=('turn_only','probe_glimpse')
BRANCHES=('native','ordinary','pre_cue_reset_native','pre_cue_reset_ordinary')
PHASES=128
# Numerical provenance thresholds only; no scientific acceptance threshold.
TOL=dict(angle_rad=3e-5,position_m=1e-6,direction=3e-5,
    embedding=1e-5,current_code=3e-5,hidden=1e-4,
    cell_atol=1e-4,cell_rtol=1e-5,centered_logits=1e-3,probability=1e-4,
    action_progress_m=1e-7,action_alignment_rad=5e-5,action_endpoint_m=1e-6)
SOURCES=('scripts/native_turn_counterbalanced_readout.py','tests/test_native_turn_counterbalanced_readout.py',
    'scripts/decompose_native_turn_prefix.py','scripts/decompose_native_turn_imitation.py',
    'docs/research/native_turn_counterbalanced_readout_plan.md')
OLD_PROTOCOLS=('docs/research/native_turn_geometry_protocol.json',
    'docs/research/native_turn_teacher_preflight_protocol.json',
    'docs/research/native_turn_original_memory_protocol.json',
    'docs/research/native_turn_distillation_protocol.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def array(x):return x.detach().cpu().numpy()if isinstance(x,torch.Tensor)else np.asarray(x)
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))
def write(path,record):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def settings():
    return dict(version='counterbalanced_saved_turn_readout_v1',conditions=list(CONDITIONS),endpoints=list(ENDPOINTS),
        contexts=list(range(16)),prefix_actions=[0,1],kinds=list(KINDS),branches=list(BRANCHES),phases=128,
        expected_physics_families=128,expected_neural_families=1536,expected_case_rows=196608,
        state_archive='left/baseline_states.npz:t8',state_semantics='State AFTER eighth common LEFT input; this input is the first-choice observation. Do not feed it again.',
        physics='Actual native one-update six-action outcomes from the recorded choice heading/endpoint/travel direction, zero vf/vs/vr after the declared choice brake, native episode clock; image and motor noise zero as in the controlled preflight.',
        geometry='Both fixed LEFT and fixed RIGHT prefixes independently cover the full 360-degree circle. Every context, mode, branch and endpoint is included regardless of prior geometry or behavioral results.',
        static_readout='Float64 retained original actor on archived float64 h. No LSTM/CNN/policy rollout. Assert actor weights exactly equal at all endpoints.',
        checks='Compare actual prefix labels and angle matches to prior 16-case histories. Validate shared inputs, h/c, actor probabilities/logits and six-action physics at every shared angle. Failures retained and whole run incomplete; no dropped cases or adaptive tolerance.',
        numerical_provenance_tolerances=TOL,
        statistics='Equal weight for all128 phases and both prefixes per context; also retain each prefix separately. Expected signed goalward displacement, actual six-action angle alignment, antipodal policy TV, and averaged-policy counterfactual progress. No p-values, success gate, ring criterion or claim that facing the goal is an optimal action.',
        no_recurrent_execution=True,no_optimization=True,no_cloud=True,
        scope='One original seed per population; all original/clean/noise endpoints. Exploratory analysis specified after the prefix-associated result; no new readout scores observed before source/protocol seal.')


def paired_folder(condition,endpoint):
    return ROOT/'results/native_turn_original_memory'/condition if endpoint=='original'else ROOT/'results/native_turn_distillation'/condition/endpoint/'final_paired'


def checkpoint_path(condition,endpoint):
    return ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'if endpoint=='original'else ROOT/f'results/native_turn_distillation/{condition}/{endpoint}/final.pt'


def freeze():
    """Root-owned sealing; hashes metadata/inputs, without applying the actor."""
    assert not PROTOCOL.exists()
    sources={p:sha(ROOT/p)for p in SOURCES};assets={}
    for p in OLD_PROTOCOLS:
        doc=read(ROOT/p);assets[p]=sha(ROOT/p)
        for name,digest in doc['source_sha256'].items():
            assert sha(ROOT/name)==digest,name
            if name in sources:assert sources[name]==digest
            sources[name]=digest
    for condition in CONDITIONS:
        paths=[GEOMETRY/'inputs'/condition/'summary.json']
        for endpoint in ENDPOINTS:
            paths += [GEOMETRY/condition/endpoint/'summary.json',paired_folder(condition,endpoint)/'summary.json',checkpoint_path(condition,endpoint)]
            if endpoint!='original':paths += [paired_folder(condition,endpoint).parent/'summary.json']
        for path in paths:
            if path.suffix=='.json':assert read(path)['complete'],path
            assets[str(path.relative_to(ROOT))]=sha(path)
    for path in ('results/native_turn_distillation/report/independent_validation.json',
                 'results/native_turn_imitation_prefix/summary.json','results/native_turn_imitation_prefix/independent_validation.json'):
        assert read(ROOT/path)['complete'];assets[path]=sha(ROOT/path)
    write(PROTOCOL,dict(**settings(),frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=sources,asset_sha256=assets))
    print(sha(PROTOCOL))


def compare(name,a,b,atol,rtol=0.):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),name
    err=np.abs(a-b);allowed=atol+rtol*np.abs(b)
    assert bool(np.all(err<=allowed)),(name,float(err.max()),float(allowed.max()))
    return dict(max_abs=float(err.max(initial=0)),rms=float(np.sqrt(np.mean(err**2))),atol=atol,rtol=rtol)


def shared_indices(target,pair_target,actual_prefix,prefix):
    """Match angles and recorded action labels, never infer labels from parity."""
    target,pair_target,actual_prefix=map(np.asarray,(target,pair_target,actual_prefix))
    assert target.shape==(128,)and pair_target.shape==(16,)and actual_prefix.shape==(16,)
    cases=np.flatnonzero(actual_prefix==prefix);assert len(cases)==8
    d=np.abs(wrap(target[:,None]-pair_target[cases][None]));indices=d.argmin(0)
    assert len(set(indices))==8 and np.max(d[indices,np.arange(8)])<TOL['angle_rad']
    return cases,indices


def terminal_from_family(family,context):
    a=family['audit'];n=len(family['choice_heading']);assert n==PHASES
    j8=[r for r in a['left_physical']if r['updates']==8];assert len(j8)==1
    j8=j8[0];assert j8['after_choice_brake']and j8['physical_position_error_m']==0.
    assert a['labels'][-1]==('probe_3'if a['kind']=='probe_glimpse'else'turn_13')
    assert a['reference_input_index']==len(family['native_x'])-1
    t=torch.tensor(j8['episode_t'],dtype=family['physical']['t'].dtype)
    assert torch.equal(t,family['physical']['t'][-1]+8)and bool((t<1350).all())
    pos=family['physical']['pos'][-1].clone();assert torch.equal(pos,context['endpoint'].expand_as(pos))
    hd=family['choice_heading'].clone();compare('choice heading',wrap(array(hd)-array(j8['heading'])),np.zeros(n),1e-7)
    compare('target heading',wrap(array(hd)-array(family['target_choice_heading'])),np.zeros(n),TOL['angle_rad'])
    return SimpleNamespace(n=n,pos=pos,hd=hd,mdir=family['physical']['mdir'][-1].clone(),
        vf=torch.zeros(n),vs=torch.zeros(n),vr=torch.zeros(n),goal=context['goal_center'].expand(n,-1).clone(),t=t)


def validate_family(family,context,prefix,kind):
    a=family['audit'];n=len(family['choice_heading'])
    assert a['phases']==n==128 and a['context']==context['context_id']and a['prefix_action']==prefix and a['kind']==kind
    labels=a['labels'];ids=[labels.index(f'turn_{i}')for i in range(14)]
    for field in ('native_x','ordinary_x'):
        x=family[field];assert x.shape==(len(labels),128,1287)
        target=torch.zeros(14,128,6,dtype=x.dtype);target[:,:,prefix]=1
        assert torch.equal(x[ids,:,1280:1286],target)
        assert torch.equal(x[:,:,768:1280],context['native_x'][None,None,768:1280].expand(len(labels),128,-1))
    left=family['left_x'];assert left.shape==(1287,)and left[1280]==1 and torch.count_nonzero(left[1281:1286])==0 and left[-1]==0
    bearing=float(torch.atan2(*(context['goal_center']-context['endpoint']).flip(0)))
    compare('uniform full circle',wrap(array(family['target_choice_heading'])-(bearing+np.arange(128)*2*np.pi/128)),np.zeros(128),TOL['angle_rad'])
    assert a['full_left_input_common']


def static_actor(state,weight,bias):
    state=np.asarray(state,dtype=np.float64);weight=np.asarray(weight,dtype=np.float64);bias=np.asarray(bias,dtype=np.float64)
    assert state.ndim==2 and state.shape[1]==512 and weight.shape==(6,256)and bias.shape==(6,)
    assert np.isfinite(state).all()and np.max(np.abs(state[:,:256]))<=1+1e-12
    logits=state[:,:256]@weight.T+bias
    exp=np.exp(logits-logits.max(1,keepdims=True));return logits,exp/exp.sum(1,keepdims=True)


def family_scores(q,physics):
    q=np.asarray(q,dtype=float);progress=array(physics['progress_m']).astype(float)
    alignment=array(physics['body_goal_angle_error_reduction_rad']).astype(float)
    assert q.shape==progress.shape==alignment.shape==(128,6)
    assert np.all(q>=0)and np.allclose(q.sum(1),1,atol=1e-14,rtol=0)
    expected=np.sum(q*progress,1);mean=q.mean(0);averaged=progress@mean
    tv=.5*np.abs(q[:64]-q[64:]).sum(1)
    return dict(progress_mm=1000*float(expected.mean()),average_policy_progress_mm=1000*float(averaged.mean()),
        progress_above_average_policy_mm=1000*float((expected-averaged).mean()),
        best_common_action_progress_mm=1000*float(progress.mean(0).max()),
        mean_antipodal_policy_tv=float(tv.mean()),body_goal_alignment_rad=float(np.sum(q*alignment,1).mean()),
        turn_weighted_body_goal_alignment_rad=float(np.sum(q[:,:2]*alignment[:,:2],1).mean()),
        average_action_distribution=mean.tolist(),case_progress_mm=(1000*expected).tolist(),
        case_average_policy_progress_mm=(1000*averaged).tolist(),antipodal_policy_tv=tv.tolist())


def aggregate_rows(rows):
    """Equal phase/context weights; prefix-specific and counterbalanced totals."""
    fields=('progress_mm','average_policy_progress_mm','progress_above_average_policy_mm',
        'best_common_action_progress_mm','mean_antipodal_policy_tv','body_goal_alignment_rad','turn_weighted_body_goal_alignment_rad')
    answer=[]
    for c in CONDITIONS:
        for e in ENDPOINTS:
            for k in KINDS:
                for b in BRANCHES:
                    for prefix in (0,1,'both'):
                        subset=[r for r in rows if(r['condition'],r['endpoint'],r['kind'],r['branch'])==(c,e,k,b)and(prefix=='both'or r['prefix']==prefix)]
                        expected={(i,p)for i in range(16)for p in (0,1)if prefix=='both'or p==prefix}
                        assert len(subset)==len(expected)and{(r['context'],r['prefix'])for r in subset}==expected
                        answer.append(dict(condition=c,endpoint=e,kind=k,branch=b,prefix=prefix,contexts=16,
                            phases_per_prefix=128,cases=len(subset)*128,all_provenance_checks_passed=all(r['shared_angle_checks_passed']for r in subset),
                            **{f:float(np.mean([r[f]for r in subset]))for f in fields}))
    return answer


def validate_shared(family,terminal,physics,state,logits,q,raw,paired,branch,prefix):
    from decompose_native_turn_prefix import prefix_labels
    labels,_=prefix_labels(raw)
    cases,idx=shared_indices(array(family['target_choice_heading']),array(raw['target_heading']),labels,prefix)
    assert raw['labels'][-8:]==[f'suffix_{j}'for j in range(8)]
    # The reference input ends at the cue/prefix; exactly eight further LEFT
    # inputs reach the archived first-choice state, with no extra observation.
    field='student_ordinary_x'if branch.endswith('ordinary')else'student_native_x'
    before=raw[field][:-8,cases];current=family['ordinary_x'if branch.endswith('ordinary')else'native_x'][:,idx]
    checks={}
    for name,a,b in [('prefix',array(current),array(before)),('choice',array(family['left_x'])[None].repeat(8,0),array(raw[field][-1,cases]))]:
        checks[name+'_embedding']=compare(name+' embedding',a[...,:256],b[...,:256],TOL['embedding'])
        checks[name+'_current_code']=compare(name+' current code',a[...,256:768],b[...,256:768],TOL['current_code'])
        assert np.array_equal(a[...,768:],b[...,768:]),name+' goal/PA/reward'
    for key in ('pos','mdir','hd','vf','vs','vr','goal','t'):
        a=array(getattr(terminal,key))[idx];b=array(raw['physical'][key][-1,cases])
        if key=='hd':checks[key]=compare(key,wrap(a-b),np.zeros_like(a),TOL['angle_rad'])
        elif key in ('t','vf','vs','vr','goal'):assert np.array_equal(a,b),key
        else:checks[key]=compare(key,a,b,TOL['position_m']if key=='pos'else TOL['direction'])
    v=paired['variants'][branch];old=array(v['first_choice_state_after_input'])[cases]
    checks['hidden']=compare('hidden',state[idx,:256],old[:,:256],TOL['hidden'])
    checks['cell']=compare('cell',state[idx,256:],old[:,256:],TOL['cell_atol'],TOL['cell_rtol'])
    oldlog=array(v['first_choice_logits'])[cases].astype(float)
    checks['centered_logits']=compare('centered logits',logits[idx]-logits[idx].mean(1,keepdims=True),oldlog-oldlog.mean(1,keepdims=True),TOL['centered_logits'])
    checks['probability']=compare('probability',q[idx],array(v['first_choice_probabilities'])[cases],TOL['probability'])
    for key,tol in [('progress_m',TOL['action_progress_m']),('body_goal_angle_error_reduction_rad',TOL['action_alignment_rad']),('endpoint',TOL['action_endpoint_m'])]:
        checks[key]=compare(key,array(physics[key])[idx],array(raw['first_update_physics'][key])[cases],tol)
    checks['action_heading']=compare('action heading',wrap(array(physics['heading'])[idx]-array(raw['first_update_physics']['heading'])[cases]),np.zeros((8,6)),TOL['angle_rad'])
    assert np.array_equal(array(paired['first_update_physics']['progress_m']),array(raw['first_update_physics']['progress_m']))
    oldq=array(v['first_choice_probabilities'])[cases].astype(float);oldq/=oldq.sum(1,keepdims=True)
    oldphysics=array(raw['first_update_physics']['progress_m'])[cases].astype(float);newphysics=array(physics['progress_m'])[idx].astype(float)
    error=1000*(np.sum(q[idx]*newphysics,1)-np.sum(oldq*oldphysics,1))
    bound=1000*(np.sum(np.abs(q[idx]-oldq)*np.abs(oldphysics),1)+np.sum(q[idx]*np.abs(newphysics-oldphysics),1))
    assert np.all(np.abs(error)<=bound+1e-12)
    return dict(cases=cases.tolist(),geometry_indices=idx.tolist(),checks=checks,
        shared_expected_progress_difference_mm=error.tolist(),shared_expected_progress_absolute_error_bound_mm=bound.tolist(),
        precision_scope='Float64 archived recurrence/static actor versus prior float32 recurrence/actor. Exact observed score differences and component bound retained; tolerances are provenance checks, not a minimum scientifically meaningful effect.')


def run():
    """Only invoked by the owner after prospective review/seal."""
    import native_turn_teacher_preflight as parent
    from decompose_native_turn_prefix import prefix_labels
    design=read(PROTOCOL);assert all(design[k]==v for k,v in settings().items())
    for field in ('source_sha256','asset_sha256'):
        for name,digest in design[field].items():assert sha(ROOT/name)==digest,name
    assert not BASE.exists()or not any(BASE.iterdir());BASE.mkdir(parents=True,exist_ok=True)
    write(BASE/'cohort.json',dict(protocol_sha256=sha(PROTOCOL),all_cases_retained=True,
        families=[dict(condition=c,endpoint=e,context=i,prefix=p,kind=k,branch=b,phases=128)
            for c in CONDITIONS for e in ENDPOINTS for i in range(16)for p in (0,1)for k in KINDS for b in BRANCHES]))
    manifest={};rows=[];checks=[];failures=[];physics_count=0;case_count=0
    def register(path,expected=None):
        entry=dict(sha256=sha(path),bytes=path.stat().st_size)
        if expected is not None:assert entry==expected,path
        manifest[str(path.relative_to(ROOT))]=entry;return path
    def bound_read(path):return read(register(path))
    old=bound_read(parent.PROTOCOL);validation=bound_read(ROOT/'results/native_turn_distillation/report/independent_validation.json')
    assert validation['complete']
    for condition in CONDITIONS:
        bankref=old['banks'][condition];bankpath=register(ROOT/bankref['path']);assert sha(bankpath)==bankref['sha256'];bank=load(bankpath)
        prep_path=GEOMETRY/'inputs'/condition/'summary.json';prep=bound_read(prep_path)
        assert prep['complete']and prep['condition']==condition and prep['protocol_sha256']==sha(ROOT/OLD_PROTOCOLS[0])
        records={(r['context'],r['prefix'],r['kind']):r for r in prep['records']}
        assert set(records)=={(i,p,k)for i in range(16)for p in (0,1)for k in KINDS}
        originalcp=load(register(checkpoint_path(condition,'original')));assert originalcp['frames']==1_000_000_000
        assert prep['checkpoint_sha256']==sha(checkpoint_path(condition,'original'))
        original=originalcp['policy'];w=array(original['pi.weight']);b=array(original['pi.bias'])
        endpoints={}
        for endpoint in ENDPOINTS:
            folder=GEOMETRY/condition/endpoint;summary=bound_read(folder/'summary.json');cp=register(checkpoint_path(condition,endpoint));checkpoint=load(cp)
            assert summary['complete']and summary['condition']==condition and summary['endpoint']==endpoint
            assert summary['checkpoint_sha256']==sha(cp)and summary['input_summary_sha256']==sha(prep_path)and summary['protocol_sha256']==sha(ROOT/OLD_PROTOCOLS[0])
            assert torch.equal(checkpoint['policy']['pi.weight'],original['pi.weight'])and torch.equal(checkpoint['policy']['pi.bias'],original['pi.bias'])
            pairedpath=paired_folder(condition,endpoint);pairedsummary=bound_read(pairedpath/'summary.json')
            assert pairedsummary['complete']and sha(pairedpath/'summary.json')==validation['models'][condition][endpoint]['paired']['summary_sha256']
            if endpoint!='original':
                terminal=bound_read(pairedpath.parent/'summary.json')
                assert terminal['complete']and terminal['updates']==checkpoint['updates']==180
                assert terminal['artifacts']['final.pt']==dict(sha256=sha(cp),bytes=cp.stat().st_size)
                register(pairedpath/'summary.json',terminal['artifacts']['final_paired/summary.json'])
                assert sha(pairedpath.parent/'summary.json')==validation['endpoint_summary_sha256'][condition+'/'+endpoint]
                assert summary['training_provenance']['terminal_summary_sha256']==sha(pairedpath.parent/'summary.json')
            register(folder/'prefix_replay.json',summary['artifacts']['prefix_replay.json'])
            prefix=read(folder/'prefix_replay.json');assert prefix['complete']and not prefix['cached_original_state_used_as_input']and len(prefix['contexts'])==16
            endpoints[endpoint]=(folder,summary,pairedpath,pairedsummary)
        assert [c['context_id']for c in bank['contexts']]==list(range(16))
        for context in bank['contexts']:
            ci=context['context_id']
            common_left=None
            for kind in KINDS:
                name=f'context{ci:02d}_{kind}';rawpath=ROOT/'results/native_turn_teacher_preflight'/condition/(name+'.pt')
                raw=load(register(rawpath));prefix_labels(raw)
                saved={}
                for endpoint,(folder,summary,pairedpath,pairedsummary)in endpoints.items():
                    register(pairedpath/(name+'.json'),pairedsummary['artifacts'][name+'.json'])
                    assert read(pairedpath/(name+'.json'))['source_raw_sha256']==sha(rawpath)
                    saved[endpoint]=load(register(pairedpath/(name+'.pt'),pairedsummary['artifacts'][name+'.pt']))
                for prefix in (0,1):
                    record=records[ci,prefix,kind];fp=register(prep_path.parent/record['path']);assert sha(fp)==record['sha256'];family=load(fp)
                    validate_family(family,context,prefix,kind)
                    if common_left is None:common_left=family['left_x'].clone()
                    else:assert torch.equal(common_left,family['left_x']),'Different full first-choice input across prefix/mode'
                    label=f'context{ci:02d}_prefix{prefix}_{kind}';terminal=terminal_from_family(family,context)
                    write(BASE/'progress.json',dict(condition=condition,context=ci,kind=kind,prefix=prefix,completed_neural_families=len(rows),phase='physical_counterfactual'))
                    physics=parent.action_progress(bank['config'],context,terminal);physics_count+=1
                    physicsdir=BASE/'physics'/condition;physicsdir.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(physics=physics,terminal=vars(terminal),input_sha256=sha(fp)),physicsdir/(label+'.pt'))
                    for endpoint,(folder,summary,pairedpath,pairedsummary)in endpoints.items():
                        for branch in BRANCHES:
                            requested=label+'_'+branch;matches=[r for r in summary['rows']if r['family']==requested and r['regime']=='left']
                            assert matches and {r['branch']for r in matches}=={branch}
                            aliases={r['exact_sham_of']for r in matches};assert len(aliases)==1
                            alias=next(iter(aliases));source=alias or requested
                            if alias:
                                expected_sham='native'if branch=='ordinary'else'pre_cue_reset_native'
                                assert branch.endswith('ordinary')and(condition=='gc100'or kind=='turn_only')and alias==label+'_'+expected_sham
                            rel=source+'/left/baseline_states.npz';path=register(folder/rel,summary['artifacts'][rel])
                            with np.load(path)as z:state=z['t8'].copy()
                            assert state.shape==(128,512)
                            logits,q=static_actor(state,w,b)
                            if branch=='pre_cue_reset_ordinary':
                                assert np.array_equal(state,np.repeat(state[:1],128,axis=0))
                                assert np.array_equal(q,np.repeat(q[:1],128,axis=0))
                            try:
                                audit=validate_shared(family,terminal,physics,state,logits,q,raw,saved[endpoint],branch,prefix)
                            except AssertionError as exc:
                                failures.append(dict(condition=condition,endpoint=endpoint,context=ci,kind=kind,prefix=prefix,branch=branch,reason=str(exc)))
                                audit=dict(complete=False,error=str(exc))
                            out=BASE/condition/endpoint;out.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(out/(requested+'.npz'),h=state[:,:256],logits=logits,probabilities=q,
                                heading=array(terminal.hd),action_progress_m=array(physics['progress_m']),
                                action_alignment_rad=array(physics['body_goal_angle_error_reduction_rad']))
                            row=dict(condition=condition,endpoint=endpoint,context=ci,kind=kind,prefix=prefix,branch=branch,
                                exact_sham_of=alias,source_state_sha256=sha(path),shared_angle_checks_passed=not audit.get('error'),**family_scores(q,physics))
                            rows.append(row);checks.append(dict(condition=condition,endpoint=endpoint,context=ci,kind=kind,prefix=prefix,branch=branch,**audit));case_count+=128
    assert len(rows)==1536 and physics_count==128 and case_count==196608
    write(BASE/'summary.json',dict(complete=not failures,protocol_sha256=sha(PROTOCOL),producer_sha256=sha(__file__),
        rows=rows,aggregates=aggregate_rows(rows),shared_angle_checks=checks,failures=failures,expected_case_rows=case_count,physics_families=physics_count,
        input_sha256=manifest,no_recurrent_execution=True,no_optimization=True,no_cloud=True))
    fields=['condition','endpoint','context','kind','prefix','branch','shared_angle_checks_passed','progress_mm',
        'average_policy_progress_mm','progress_above_average_policy_mm','best_common_action_progress_mm',
        'mean_antipodal_policy_tv','body_goal_alignment_rad','turn_weighted_body_goal_alignment_rad']
    with (BASE/'all_families.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k]for k in fields}for r in rows)
    write(BASE/'artifacts.json',{str(p.relative_to(BASE)):dict(sha256=sha(p),bytes=p.stat().st_size)for p in BASE.rglob('*')if p.is_file()})
    if failures:raise RuntimeError(f'{len(failures)} provenance comparisons failed; all results retained, no completed scientific claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('--write-draft',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    assert sum((a.write_draft,a.freeze,a.run))==1
    torch.set_num_threads(1)
    if a.write_draft:write(DRAFT,dict(**settings(),status='draft_implementation_only_not_authorized_for_scores',source_sha256={s:sha(ROOT/s)for s in SOURCES}))
    elif a.freeze:freeze()
    else:
        try:run()
        except Exception as exc:
            if BASE.exists():write(BASE/'incomplete.json',dict(complete=False,exception_type=type(exc).__name__,reason=str(exc),no_retry=True))
            raise


if __name__=='__main__':main()
