"""Exploratory saved-probability decomposition; no policy/environment execution."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_turn_imitation_decomposition'
CONDITIONS=('gc100','pgc33')
ENDPOINTS=('original','clean','noise')
KINDS=('turn_only','probe_glimpse')
SCOPES=('probe_glimpse','turn_only','all_modes')
VARIANTS=('native','ordinary','cue_plus60','cue_minus60','pre_cue_reset_native',
          'pre_cue_reset_ordinary','choice_reset_native','choice_reset_ordinary')
FIELDS=('observed_kl','teacher_floor','mean_policy_mismatch','history_gain',
        'mean_policy_counterfactual_kl','conditional_case_mismatch')
TRAINING_SHA='ac37ba336f79299ed285c4bb0ef126296e5f8d46aee795e431d58544aab06c42'
MEMORY_SHA='52e6b8c145a87dedf4889ebe864fc7376c8447e4e68ac4c4774234545b93a8fd'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def tensor_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def probabilities(x):
    x=np.asarray(x,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==6 and np.isfinite(x).all()and (x>=0).all()
    assert np.all(x.sum(1)>0)
    return x/x.sum(1,keepdims=True)


def kl(p,q):
    """Exact zero-p convention; positive student probabilities, no clamping."""
    p=np.asarray(p,dtype=float);q=np.broadcast_to(np.asarray(q,dtype=float),p.shape)
    assert np.isfinite(p).all()and np.isfinite(q).all()and (p>=0).all()and (q>=0).all()
    active=p>0;assert np.all(q[active]>0),'Infinite KL is not silently regularized.'
    terms=np.zeros_like(p);terms[active]=p[active]*(np.log(p[active])-np.log(q[active]))
    return terms.sum(-1)


def decomposition(p,q):
    p=probabilities(p);q=probabilities(q);assert p.shape==q.shape and (q>0).all()
    pbar=p.mean(0);qbar=q.mean(0)
    floor=float(kl(p,pbar).mean());mismatch=float(kl(pbar[None],qbar)[0])
    observed=float(kl(p,q).mean());counterfactual=float(kl(p,qbar).mean())
    gain=counterfactual-observed
    # Independent chain-rule reconstruction using case posteriors given action.
    active=pbar>0;pp=p[:,active]/(len(p)*pbar[active]);qp=q[:,active]/(len(q)*qbar[active])
    posterior=float((pbar[active]*kl(pp.T,qp.T)).sum())
    assert abs(counterfactual-floor-mismatch)<5e-12
    assert abs(observed-(floor+mismatch-gain))<5e-12
    assert abs(observed-mismatch-posterior)<5e-12 and abs(floor-gain-posterior)<5e-12
    assert gain<=floor+5e-12 and observed>=mismatch-5e-12 and posterior>=-5e-12
    identical=bool(np.array_equal(q,np.repeat(q[:1],len(q),axis=0)))
    if identical:assert abs(gain)<5e-12
    return dict(cases=len(p),observed_kl=observed,teacher_floor=floor,mean_policy_mismatch=mismatch,
        history_gain=gain,mean_policy_counterfactual_kl=counterfactual,conditional_case_mismatch=posterior,
        identity_error=observed-(floor+mismatch-gain),chain_rule_error=observed-mismatch-posterior,
        student_identical_within_group=identical,teacher_mean=pbar.tolist(),student_mean=qbar.tolist(),
        minimum_student_probability=float(q.min()))


def group_cases(inputs,p,q,refs):
    assert len(inputs)==len(p)==len(q)==len(refs)
    groups={}
    for i,x in enumerate(inputs):groups.setdefault(tensor_sha(x),[]).append(i)
    records=[]
    for identity,indices in sorted(groups.items()):
        assert all(np.array_equal(inputs[indices[0]],inputs[i])for i in indices)
        r=decomposition(p[indices],q[indices]);r.update(input_sha256=identity,cases_ref=[refs[i]for i in indices])
        records.append(r)
    total=len(inputs)
    mean={field:sum(r[field]*r['cases']for r in records)/total for field in FIELDS}
    assert abs(mean['observed_kl']-mean['teacher_floor']-mean['mean_policy_mismatch']+mean['history_gain'])<5e-12
    return records,dict(cases=total,groups=len(records),**mean,
        max_group_identity_error=max(abs(r['identity_error'])for r in records),
        max_group_chain_rule_error=max(abs(r['chain_rule_error'])for r in records))


def improvement(before,after):
    assert before['cases']==after['cases']and before['groups']==after['groups']
    assert abs(before['teacher_floor']-after['teacher_floor'])<5e-12
    total=before['observed_kl']-after['observed_kl']
    average=before['mean_policy_mismatch']-after['mean_policy_mismatch']
    history=after['history_gain']-before['history_gain']
    assert abs(total-average-history)<5e-12
    return dict(kl_improvement=total,mean_policy_improvement=average,history_gain_change=history,
        reconstruction_error=total-average-history,
        mean_policy_fraction=average/total if abs(total)>1e-10 else None,
        history_fraction=history/total if abs(total)>1e-10 else None)


def teacher_groups(groups):
    """Exclude endpoint-dependent student fields from the shared-target check."""
    return {name:{**value,'records':[{k:v for k,v in r.items()if k!='variants'}for r in value['records']]}
            for name,value in groups.items()}


def load_and_analyze():
    manifest={}
    def register(path,expected=None):
        path=Path(path);value=dict(sha256=sha(path),bytes=path.stat().st_size)
        if expected is not None:assert value==expected,path
        key=str(path.relative_to(ROOT))
        if key in manifest:assert manifest[key]==value
        manifest[key]=value;return path
    def read(path):return json.loads(register(path).read_text())
    for file,expected in [('docs/research/native_turn_distillation_protocol.json',TRAINING_SHA),
                          ('docs/research/native_turn_original_memory_protocol.json',MEMORY_SHA)]:
        path=ROOT/file;assert sha(path)==expected
        doc=read(path)
        for name,digest in doc['source_sha256'].items():assert sha(register(ROOT/name))==digest
    validation=read(ROOT/'results/native_turn_distillation/report/independent_validation.json')
    report_path=ROOT/'results/native_turn_distillation/report/summary.json'
    assert validation['complete']and validation['report_summary_sha256']==sha(register(report_path))
    canonical={};aggregates=[];groups=[];zero_controls=0
    for condition in CONDITIONS:
        original_groups=None
        for endpoint in ENDPOINTS:
            folder=ROOT/'results/native_turn_original_memory'/condition if endpoint=='original'else ROOT/'results/native_turn_distillation'/condition/endpoint/'final_paired'
            summary=read(folder/'summary.json');assert summary['complete']and summary['condition']==condition
            assert validation['models'][condition][endpoint]['paired']['summary_sha256']==sha(folder/'summary.json')
            assert len(summary['rows'])==4096 and len(summary['pairs'])==2048
            if endpoint=='original':
                assert summary['protocol_sha256']==MEMORY_SHA;original_groups=teacher_groups(summary['exact_input_groups'])
            else:
                terminal=read(folder.parent/'summary.json');execution=read(ROOT/f'results/native_turn_distillation/{condition}_{endpoint}_execution.json')
                assert terminal['complete']and terminal['updates']==180 and execution['complete']and execution['exit_code']==0
                assert terminal['protocol_sha256']==TRAINING_SHA and summary['policy_hash']==terminal['final_policy_hash']
                register(folder/'summary.json',terminal['artifacts']['final_paired/summary.json'])
                assert validation['endpoint_summary_sha256'][f'{condition}/{endpoint}']==sha(folder.parent/'summary.json')
                assert validation['execution_sha256'][f'{condition}/{endpoint}']==sha(ROOT/f'results/native_turn_distillation/{condition}_{endpoint}_execution.json')
                assert teacher_groups(summary['exact_input_groups'])==original_groups
            for name,entry in summary['artifacts'].items():register(folder/name,entry)
            lookup={(r['context'],r['kind'],r['variant'],r['case']):r for r in summary['rows']}
            assert len(lookup)==4096
            data=[]
            for context in range(16):
                for kind in KINDS:
                    raw=torch.load(folder/f'context{context:02d}_{kind}.pt',map_location='cpu',weights_only=True)
                    p=raw['teacher_first_choice_probabilities'].numpy();saved=raw['variants'];assert set(saved)==set(VARIANTS)
                    x=saved['native']['first_choice_input'].numpy()
                    assert x.shape==(16,1287)and p.shape==(16,6)
                    key=(condition,context,kind)
                    if endpoint=='original':canonical[key]=(x.copy(),p.copy())
                    else:assert np.array_equal(x,canonical[key][0])and np.array_equal(p,canonical[key][1])
                    for variant in VARIANTS:
                        v=saved[variant];q=v['first_choice_probabilities'].numpy()
                        assert np.array_equal(v['first_choice_input'].numpy(),x)
                        values=kl(probabilities(p),probabilities(q))
                        for case in range(16):
                            assert abs(values[case]-lookup[context,kind,variant,case]['teacher_forward_kl_nats'])<1e-10
                            data.append(dict(context=context,kind=kind,case=case,variant=variant,x=x[case],p=p[case],q=q[case]))
            for scope in SCOPES:
                for variant in VARIANTS:
                    selected=[r for r in data if r['variant']==variant and(scope=='all_modes'or r['kind']==scope)]
                    refs=[{k:r[k]for k in ('context','kind','case')}for r in selected]
                    rr,mean=group_cases(np.stack([r['x']for r in selected]),np.stack([r['p']for r in selected]),np.stack([r['q']for r in selected]),refs)
                    archived=summary['exact_input_groups'][scope]
                    assert mean['cases']==archived['cases']and mean['groups']==archived['groups']
                    stored={r['input_sha256']:r for r in archived['records']}
                    assert set(stored)=={r['input_sha256']for r in rr}
                    for r in rr:
                        s=stored[r['input_sha256']]
                        assert r['cases']==s['count']and r['cases_ref']==[{k:z[k]for k in ('context','kind','case')}for z in s['cases']]
                    assert abs(mean['teacher_floor']-archived['case_weighted_minimum_forward_kl_nats'])<1e-12
                    if variant.startswith('choice_reset'):
                        assert all(r['student_identical_within_group']and abs(r['history_gain'])<5e-12 for r in rr)
                        zero_controls+=len(rr)
                    tags=dict(condition=condition,endpoint=endpoint,scope=scope,variant=variant)
                    aggregates.append(dict(**tags,**mean));groups.extend(dict(**tags,**r)for r in rr)
    lookup={(r['condition'],r['endpoint'],r['scope'],r['variant']):r for r in aggregates}
    changes=[]
    for condition in CONDITIONS:
        for endpoint in ('clean','noise'):
            for scope in SCOPES:
                for variant in VARIANTS:
                    before=lookup[condition,'original',scope,variant];after=lookup[condition,endpoint,scope,variant]
                    changes.append(dict(condition=condition,endpoint=endpoint,scope=scope,variant=variant,**improvement(before,after)))
    # Complete cue shams remain exact under this nonlinear probability averaging.
    for endpoint in ENDPOINTS:
        for scope in SCOPES:
            for a,b in [('native','ordinary'),('native','cue_plus60'),('native','cue_minus60'),('pre_cue_reset_native','pre_cue_reset_ordinary'),('choice_reset_native','choice_reset_ordinary')]:
                assert all(lookup['gc100',endpoint,scope,a][k]==lookup['gc100',endpoint,scope,b][k]for k in FIELDS)
        for condition in CONDITIONS:
            assert all(lookup[condition,endpoint,'turn_only','native'][k]==lookup[condition,endpoint,'turn_only',v][k]for v in ('ordinary','cue_plus60','cue_minus60')for k in FIELDS)
    assert len(aggregates)==144 and len(groups)==2304 and len(changes)==96 and zero_controls==576
    return dict(complete=True,exploratory_descriptive_reanalysis=True,no_new_rollout=True,no_optimization=True,
        analysis_started_after_four_arm_results=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        producer_sha256=sha(__file__),test_sha256=sha(ROOT/'tests/test_decompose_native_turn_imitation.py'),
        scope='All6 fixed endpoints, all8 paired variants, both modes plus their all-mode union; equal case weights within exact byte-identical current-input groups.',
        probability_convention='Float64 row normalization; exact zero-teacher convention; strictly positive student probabilities required, no smoothing or clamping.',
        interpretation='qbar is an averaged-policy counterfactual, not an executed state erasure, a memoryless neural mechanism, or a generalization measure. Signed history gain is descriptive rather than causal.',
        unique_saved_case_rows=24576,group_rows=len(groups),choice_reset_zero_gain_groups=zero_controls,
        aggregates=aggregates,groups=groups,improvements=changes,input_sha256=manifest)


def write_csv(path,rows,fields):
    with path.open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def render(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),sharey=True)
    for ax,condition,title in zip(axes,CONDITIONS,('Grid','PGC + grid')):
        rows=[next(r for r in result['improvements']if(r['condition'],r['endpoint'],r['scope'],r['variant'])==(condition,e,'probe_glimpse','native'))for e in ('clean','noise')]
        x=np.arange(2);average=np.array([r['mean_policy_improvement']for r in rows]);history=np.array([r['history_gain_change']for r in rows])
        ax.bar(x-.17,average,.32,color='#617a8a',label='Better average action policy')
        ax.bar(x+.17,history,.32,color='#299c8d',label='Change in history gain')
        ax.scatter(x,[r['kl_improvement']for r in rows],marker='D',s=27,color='black',label='Total improvement',zorder=3)
        ax.set_title(title);ax.set_xticks(x,['Clean','State noise']);ax.axhline(0,color='#555',lw=.8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Reduction in teacher KL (nats)');axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Why imitation improved: same first-choice inputs')
    fig.tight_layout();fig.savefig(out/'imitation_decomposition.png',dpi=180);fig.savefig(out/'imitation_decomposition.pdf');plt.close(fig)


def main():
    assert not BASE.exists()or not any(BASE.iterdir()),'Preserve earlier analysis output.'
    result=load_and_analyze();BASE.mkdir(parents=True,exist_ok=True)
    render(result,BASE)
    (BASE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    tags=['condition','endpoint','scope','variant']
    write_csv(BASE/'all_conditions.csv',result['aggregates'],tags+['cases','groups']+list(FIELDS))
    write_csv(BASE/'all_context_groups.csv',result['groups'],tags+['input_sha256','cases']+list(FIELDS)+['identity_error','chain_rule_error','student_identical_within_group'])
    write_csv(BASE/'all_improvements.csv',result['improvements'],tags+['kl_improvement','mean_policy_improvement','history_gain_change','mean_policy_fraction','history_fraction'])
    lines=['# Why imitation improved','',
        'We separated a better average action policy from using history to vary actions among cases with exactly the same current input. This exploratory analysis uses saved probabilities only.','',
        '![Imitation decomposition](imitation_decomposition.png)','',
        '| Model | Continuation | Total KL improvement | Better average policy | Change in history gain | Average-policy share |',
        '|---|---|---:|---:|---:|---:|']
    for r in result['improvements']:
        if r['scope']=='probe_glimpse'and r['variant']=='native':
            lines.append(f"| {'Grid'if r['condition']=='gc100'else'PGC + grid'} | {r['endpoint']} | {r['kl_improvement']:.6f} | {r['mean_policy_improvement']:.6f} | {r['history_gain_change']:.6f} | {100*r['mean_policy_fraction']:.2f}% |")
    lines+=['','All values are in nats. The two contributions add exactly to total improvement; negative history gain is retained.','',
        'Within each identical-input group, we average the teacher and student action probabilities. The **average-policy mismatch** asks whether those two averages agree. **History gain** asks whether the actual history-dependent student choices match the teacher better than using the student’s own average for every case.','',
        '**Observed KL = teacher memoryless floor + average-policy mismatch − history gain.** The identity also holds after weighting groups by their number of cases. Choice-state reset gives zero history gain in all576 checked groups.','',
        'A better average policy can substantially reduce imitation error without choosing the right action for each hidden heading. The averaged policy here is a mathematical comparison, not an executed state-erasure experiment or a measure of generalization. History gain is an association with the teacher’s choices, not a causal memory test.','',
        'Primary display: native probe cue, all16 contexts. All6 endpoints, all8 variants, turn-only/probe/combined groups and exact Grid shams are retained below. These are fixed models from one original seed pair, not independent network replicates.','',
        '[All conditions](all_conditions.csv) · [Every context/input group](all_context_groups.csv) · [All original-to-final changes](all_improvements.csv) · [Exact arithmetic and source hashes](summary.json)']
    (BASE/'README.md').write_text('\n'.join(lines)+'\n')
    artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in BASE.iterdir()if p.is_file()}
    (BASE/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(json.dumps(dict(complete=True,output=str(BASE),groups=result['group_rows'],zero_gain_controls=result['choice_reset_zero_gain_groups'])))


if __name__=='__main__':main()
