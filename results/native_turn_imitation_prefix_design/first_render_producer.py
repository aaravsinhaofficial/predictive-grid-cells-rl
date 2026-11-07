"""Exploratory prefix-conditioned averaging of saved action distributions only."""
from pathlib import Path
import datetime as dt
import json
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import decompose_native_turn_imitation as main

OUT=ROOT/'results/native_turn_imitation_prefix'
POLICIES=('full','prefix_average','group_average')
KL_FIELDS=('full_kl','prefix_average_kl','group_average_kl','total_history_gain','prefix_associated_gain','within_prefix_gain')
PROGRESS_FIELDS=('full_progress_mm','prefix_average_progress_mm','group_average_progress_mm',
                'total_progress_advantage_mm','prefix_associated_progress_mm','within_prefix_progress_mm')


def prefix_labels(raw):
    """Read actual PA tape, then independently check every recorded input."""
    ids=[i for i,label in enumerate(raw['labels'])if label.startswith('turn_')]
    assert [raw['labels'][i]for i in ids]==[f'turn_{i}'for i in range(14)]
    pa=raw['previous_actions'][ids].numpy();assert pa.shape==(14,16)
    assert np.array_equal(pa,np.repeat(pa[:1],14,axis=0))and set(pa[0])=={0,1}
    labels=pa[0].astype(int).copy();assert np.count_nonzero(labels==0)==np.count_nonzero(labels==1)==8
    expected=np.eye(6,dtype=np.float32)[pa]
    for field in ('teacher_x','student_native_x','student_ordinary_x'):
        actual=raw[field][ids,:,1280:1286].numpy()
        assert np.array_equal(actual,expected),(field,'recorded PA disagrees with actual action tape')
    return labels,dict(turn_input_indices=ids,turn_labels=[raw['labels'][i]for i in ids],
        prefix_action_by_case=labels.tolist(),prefix_actions={'0':'LEFT','1':'RIGHT'},
        source='All14 actual previous_actions entries, verified against teacher/native-student/ordinary-student input one-hots; not inferred from case parity.')


def prefix_decomposition(p,q,labels,physical):
    p=main.probabilities(p);q=main.probabilities(q);labels=np.asarray(labels)
    physical=np.asarray(physical,dtype=float)
    assert len(labels)==len(p)and p.shape==q.shape==physical.shape and np.isfinite(physical).all()
    assert set(labels)=={0,1}and (q>0).all()
    grouped=np.repeat(q.mean(0,keepdims=True),len(q),axis=0);prefix=np.empty_like(q)
    for label in (0,1):prefix[labels==label]=q[labels==label].mean(0)
    choices={'full':q,'prefix_average':prefix,'group_average':grouped}
    kl={name:float(main.kl(p,x).mean())for name,x in choices.items()}
    progress={name:float(np.mean(np.sum(x*physical,axis=1)))*1000 for name,x in choices.items()}
    result={name+'_kl':value for name,value in kl.items()}
    result.update({name+'_progress_mm':value for name,value in progress.items()})
    result.update(total_history_gain=kl['group_average']-kl['full'],
        prefix_associated_gain=kl['group_average']-kl['prefix_average'],within_prefix_gain=kl['prefix_average']-kl['full'],
        total_progress_advantage_mm=progress['full']-progress['group_average'],
        prefix_associated_progress_mm=progress['prefix_average']-progress['group_average'],
        within_prefix_progress_mm=progress['full']-progress['prefix_average'])
    assert abs(result['total_history_gain']-result['prefix_associated_gain']-result['within_prefix_gain'])<5e-12
    assert abs(result['total_progress_advantage_mm']-result['prefix_associated_progress_mm']-result['within_prefix_progress_mm'])<5e-12
    return dict(**result,cases=len(p),prefix_counts={str(label):int(np.count_nonzero(labels==label))for label in (0,1)},
        group_average_distribution=q.mean(0).tolist(),prefix_average_distributions={str(k):q[labels==k].mean(0).tolist()for k in (0,1)},
        case_metrics={name:dict(teacher_kl=main.kl(p,x).tolist(),signed_progress_mm=(1000*np.sum(x*physical,axis=1)).tolist())for name,x in choices.items()})


def analyze():
    manifest={}
    def register(path,expected=None):
        path=Path(path);value=dict(sha256=main.sha(path),bytes=path.stat().st_size)
        if expected is not None:assert value==expected,path
        manifest[str(path.relative_to(ROOT))]=value;return path
    def read(path,expected=None):return json.loads(register(path,expected).read_text())
    prior=read(main.BASE/'summary.json');audit=read(main.BASE/'independent_validation.json')
    assert prior['complete']and audit['complete']
    assert audit['summary_sha256']==main.sha(main.BASE/'summary.json')and audit['producer_sha256']==prior['producer_sha256']
    # The audited main output is fixed and remains unmodified by this supplement.
    assert prior['producer_sha256']==main.sha(register(ROOT/'scripts/decompose_native_turn_imitation.py'))
    original_artifacts=read(main.BASE/'artifacts.json')
    for name,entry in original_artifacts.items():register(main.BASE/name,entry)
    main_groups={(r['condition'],r['endpoint'],r['scope'],r['variant'],r['input_sha256']):r for r in prior['groups']}
    records=[];aggregates=[];histories={};canonical={}
    for condition in main.CONDITIONS:
        for endpoint in main.ENDPOINTS:
            folder=ROOT/'results/native_turn_original_memory'/condition if endpoint=='original'else ROOT/'results/native_turn_distillation'/condition/endpoint/'final_paired'
            summary=read(folder/'summary.json',prior['input_sha256'][str((folder/'summary.json').relative_to(ROOT))])
            row_lookup={(r['context'],r['kind'],r['variant'],r['case']):r for r in summary['rows']}
            data=[]
            for context in range(16):
                for kind in main.KINDS:
                    name=f'context{context:02d}_{kind}';path=folder/(name+'.pt');info=folder/(name+'.json')
                    register(path,prior['input_sha256'][str(path.relative_to(ROOT))]);meta=read(info,prior['input_sha256'][str(info.relative_to(ROOT))])
                    source=ROOT/'results/native_turn_teacher_preflight'/condition/(name+'.pt')
                    register(source);assert main.sha(source)==meta['source_raw_sha256']
                    identity=(condition,context,kind)
                    if identity not in histories:
                        raw=torch.load(source,map_location='cpu',weights_only=True)
                        labels,check=prefix_labels(raw)
                        histories[identity]=dict(condition=condition,context=context,kind=kind,path=str(source.relative_to(ROOT)),sha256=main.sha(source),**check)
                        canonical[identity]=dict(labels=labels,teacher=raw['teacher_first_choice_probabilities'].numpy(),
                            inputs=raw['student_native_x'][-1].numpy(),physical=raw['first_update_physics']['progress_m'].numpy())
                    ref=canonical[identity];saved=torch.load(path,map_location='cpu',weights_only=True)
                    assert np.array_equal(saved['teacher_first_choice_probabilities'].numpy(),ref['teacher'])
                    assert np.array_equal(saved['first_update_physics']['progress_m'].numpy(),ref['physical'])
                    for variant in main.VARIANTS:
                        value=saved['variants'][variant];x=value['first_choice_input'].numpy();q=value['first_choice_probabilities'].numpy()
                        assert np.array_equal(x,ref['inputs'])
                        p=main.probabilities(ref['teacher']);qn=main.probabilities(q)
                        kl=main.kl(p,qn);progress=np.sum(qn*ref['physical'],axis=1)
                        for case in range(16):
                            row=row_lookup[context,kind,variant,case]
                            assert abs(kl[case]-row['teacher_forward_kl_nats'])<1e-10
                            assert abs(progress[case]-row['expected_progress_m'])<1e-12
                            data.append(dict(context=context,kind=kind,case=case,variant=variant,
                                x=x[case],p=p[case],q=q[case],prefix=ref['labels'][case],physical=ref['physical'][case]))
            for scope in main.SCOPES:
                for variant in main.VARIANTS:
                    selected=[r for r in data if r['variant']==variant and(scope=='all_modes'or r['kind']==scope)]
                    by_input={}
                    for r in selected:by_input.setdefault(main.tensor_sha(r['x']),[]).append(r)
                    groups=[]
                    for identity,rows in sorted(by_input.items()):
                        assert all(np.array_equal(rows[0]['x'],r['x'])for r in rows)
                        value=prefix_decomposition(np.stack([r['p']for r in rows]),np.stack([r['q']for r in rows]),
                            np.array([r['prefix']for r in rows]),np.stack([r['physical']for r in rows]))
                        old=main_groups[condition,endpoint,scope,variant,identity]
                        assert value['cases']==old['cases']
                        assert abs(value['full_kl']-old['observed_kl'])<1e-12
                        assert abs(value['group_average_kl']-old['mean_policy_counterfactual_kl'])<1e-12
                        assert abs(value['total_history_gain']-old['history_gain'])<1e-12
                        tags=dict(condition=condition,endpoint=endpoint,scope=scope,variant=variant)
                        value.update(**tags,input_sha256=identity,cases_ref=[{k:r[k]for k in ('context','kind','case','prefix')}for r in rows])
                        if variant.startswith('choice_reset'):
                            for field in ('total_history_gain','prefix_associated_gain','within_prefix_gain','total_progress_advantage_mm','prefix_associated_progress_mm','within_prefix_progress_mm'):assert abs(value[field])<5e-12
                        records.append(value);groups.append(value)
                    total=sum(r['cases']for r in groups)
                    aggregate=dict(**tags,cases=total,groups=len(groups),**{field:sum(r[field]*r['cases']for r in groups)/total for field in KL_FIELDS+PROGRESS_FIELDS})
                    assert abs(aggregate['total_history_gain']-aggregate['prefix_associated_gain']-aggregate['within_prefix_gain'])<5e-12
                    assert abs(aggregate['total_progress_advantage_mm']-aggregate['prefix_associated_progress_mm']-aggregate['within_prefix_progress_mm'])<5e-12
                    aggregates.append(aggregate)
    assert len(records)==2304 and len(aggregates)==144 and len(histories)==64
    lookup={(r['condition'],r['endpoint'],r['scope'],r['variant']):r for r in aggregates}
    for endpoint in main.ENDPOINTS:
        for scope in main.SCOPES:
            for a,b in [('native','ordinary'),('native','cue_plus60'),('native','cue_minus60'),('pre_cue_reset_native','pre_cue_reset_ordinary'),('choice_reset_native','choice_reset_ordinary')]:
                assert all(lookup['gc100',endpoint,scope,a][k]==lookup['gc100',endpoint,scope,b][k]for k in KL_FIELDS+PROGRESS_FIELDS)
    return dict(complete=True,exploratory_descriptive_reanalysis=True,no_new_rollout=True,no_optimization=True,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),producer_sha256=main.sha(__file__),
        test_sha256=main.sha(ROOT/'tests/test_decompose_native_turn_prefix.py'),main_summary_sha256=main.sha(main.BASE/'summary.json'),
        scope='All6 fixed endpoints/all8 variants, probe/turn-only/combined exact-input groups; prefixes independently read from the actual archived14-step previous-action histories.',
        interpretation='Prefix and heading half-circle are correlated by construction. Prefix-conditioned averaging is a mathematical policy comparison, not an intervention or evidence of causal reliance on prefix, nor proof of continuous angular memory.',
        histories=list(histories.values()),aggregates=aggregates,groups=records,input_sha256=manifest)


def render(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    lookup={(r['condition'],r['endpoint'],r['scope'],r['variant']):r for r in result['aggregates']}
    for ax,c,title in zip(axes,main.CONDITIONS,('Grid','PGC + grid')):
        x=np.arange(3);rows=[lookup[c,e,'probe_glimpse','native']for e in main.ENDPOINTS]
        for j,(field,label,color)in enumerate([('full_progress_mm','Actual policy','#465d73'),('prefix_average_progress_mm','Average within LEFT/RIGHT prefix','#e0a54a'),('group_average_progress_mm','One average for all headings','#b7b7b7')]):
            ax.bar(x+(j-1)*.24,[r[field]for r in rows],.23,label=label,color=color)
        ax.set_xticks(x,['Original','Clean','State noise']);ax.set_title(title);ax.axhline(0,color='#666',lw=.8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Expected first-step goalward progress (mm)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=1,fontsize=8)
    fig.suptitle('How much survives averaging within each turn prefix?')
    fig.tight_layout(rect=(0,.18,1,1));fig.savefig(OUT/'prefix_comparison.png',dpi=180);fig.savefig(OUT/'prefix_comparison.pdf');plt.close(fig)


def main_cli():
    assert not OUT.exists()or not any(OUT.iterdir())
    result=analyze();OUT.mkdir(parents=True,exist_ok=True);render(result)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['condition','endpoint','scope','variant','cases','groups']+list(KL_FIELDS+PROGRESS_FIELDS)
    main.write_csv(OUT/'all_conditions.csv',result['aggregates'],fields)
    main.write_csv(OUT/'all_input_groups.csv',result['groups'],fields+['input_sha256'])
    primary=[r for r in result['aggregates']if r['scope']=='probe_glimpse'and r['variant']=='native']
    lines=['# Separating the two turn prefixes','',
        'The preparation used LEFT for one half of the heading circle and RIGHT for the other. We therefore compared each saved policy with its average within the actual turn prefix, and with one average across all headings. All prefix labels were read from the recorded action histories and checked against the input one-hots.','',
        '![Prefix comparison](prefix_comparison.png)','',
        '| Model | Endpoint | Actual progress (mm) | Within-prefix average (mm) | Whole-group average (mm) | Actual minus prefix average (mm) |',
        '|---|---|---:|---:|---:|---:|']
    for r in primary:lines.append(f"| {'Grid'if r['condition']=='gc100'else'PGC + grid'} | {r['endpoint']} | {r['full_progress_mm']:.6f} | {r['prefix_average_progress_mm']:.6f} | {r['group_average_progress_mm']:.6f} | {r['within_prefix_progress_mm']:.6f} |")
    lines+=['','The plot is deliberately zoomed to small effects; the fully informed teachers achieved about 1 mm per first update. All three columns are evaluated against the same saved six-action physical outcomes.','',
        '| Model | Endpoint | Total history gain (nats) | Prefix-associated part | Within-prefix part |',
        '|---|---|---:|---:|---:|']
    for r in primary:lines.append(f"| {'Grid'if r['condition']=='gc100'else'PGC + grid'} | {r['endpoint']} | {r['total_history_gain']:.6f} | {r['prefix_associated_gain']:.6f} | {r['within_prefix_gain']:.6f} |")
    lines+=['','The two signed parts add exactly to total history gain. The same decomposition holds for progress relative to the whole-group average. All endpoints, interventions, contexts and modes remain in the appendices.','',
        'Scope: one original seed per population, 16 contexts, both modes and all eight variants; original, clean and noisy endpoints are all retained.','',
        '**This is an association check.** Prefix is correlated with heading and other preceding observations. Averaging within a prefix does not causally erase only angular memory, and a surviving within-prefix difference is not by itself evidence of continuous angular memory or a ring. A counterbalanced task would be needed to separate reliance on these correlated features.','',
        '[All conditions](all_conditions.csv) · [Every input group](all_input_groups.csv) · [Actual action-history labels and source hashes](summary.json) · [Main decomposition](../native_turn_imitation_decomposition/README.md)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    (OUT/'artifacts.json').write_text(json.dumps({p.name:dict(sha256=main.sha(p),bytes=p.stat().st_size)for p in OUT.iterdir()if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(complete=True,groups=len(result['groups']),histories=len(result['histories']),output=str(OUT))))


if __name__=='__main__':main_cli()
