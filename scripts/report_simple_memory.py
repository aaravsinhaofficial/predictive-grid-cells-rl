"""Small, seed-level report of the frozen16-model memory study.

No policy execution or new analysis choices. Producer summaries are reduced to
one value per trained model; context/tape/episode counts remain explicit.
Missing and invalid artifacts never become zeros or scientific failures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PROTOCOL=ROOT/'docs/research/simple_memory_analysis_protocol.json'
SEEDS=(1,2,3,4)
ARMS=(('gc100',0),('pgc33',0),('gc100',60),('pgc33',60))
COHORT=tuple((c,g,s) for c,g in ARMS for s in SEEDS)
STAGES=('evaluation','activity','dynamics','interventions')
DEFAULTS=dict(evaluation='memory_eval_gap60.json',activity='activity_analysis/summary.json',
              dynamics='dynamics_analysis/summary.json',interventions='intervention_analysis/summary.json')
TARGETS=('body_heading','last_travel_direction','egocentric_goal_bearing')
BINS=('known_visible','blind_age_1_15','blind_age_16_30','blind_age_31_45','blind_age_46_60')
COLORS={'gc100':'#3379B5','pgc33':'#D87532'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(value,*keys):
    for key in keys:
        if not isinstance(value,dict) or key not in value: return None
        value=value[key]
    return value


def number(value):
    if value is None: return None
    if isinstance(value,(bool,np.bool_)): return float(value)
    if isinstance(value,(int,float,np.number)) and np.isfinite(value): return float(value)
    return None


def mean(values):
    values=[number(v) for v in values]
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def key(condition,gap,seed): return f'{condition}_gap{gap}_s{seed}'


def paired_estimate(effects):
    """Four planned seed slots; exact signs use available PAIRS only."""
    ordered=[number(effects.get(seed)) for seed in SEEDS]
    valid=np.asarray([v for v in ordered if v is not None],float)
    n=len(valid)
    result=dict(planned_seeds=list(SEEDS),seed_effects=ordered,n_pairs=n,
        missing_seeds=[s for s,v in zip(SEEDS,ordered) if v is None],mean=None,
        descriptive_ci95=None,exact_signflip_p=None,minimum_two_sided_exact_p=None,
        complete_four_pairs=n==4,independent_unit='trained seed; no episode/context pseudoreplication')
    if not n: return result
    center=float(valid.mean());result['mean']=center
    if n>=2:
        sem=float(stats.sem(valid))
        width=float(stats.t.ppf(.975,n-1)*sem)
        result['descriptive_ci95']=[center-width,center+width]
    signs=np.asarray(list(itertools.product((-1,1),repeat=n)),float)
    result['exact_signflip_p']=float(np.mean(np.abs((signs*valid).mean(1))>=abs(center)-1e-12))
    result['minimum_two_sided_exact_p']=min(1.,2/(2**n))
    return result


def factorial_estimates(models,metric):
    lookup={(r['condition'],r['training_task_max_gap'],r['training_seed']):r['metrics'].get(metric)
            for r in models}
    definitions={
        'pgc_minus_grid_blackout_training':[(1,'pgc33',60),(-1,'gc100',60)],
        'blackout_minus_native_grid':[(1,'gc100',60),(-1,'gc100',0)],
        'blackout_minus_native_pgc':[(1,'pgc33',60),(-1,'pgc33',0)],
        'population_by_training_interaction':[(1,'pgc33',60),(-1,'gc100',60),(-1,'pgc33',0),(1,'gc100',0)]}
    results={}
    for label,terms in definitions.items():
        effects={}
        for seed in SEEDS:
            values=[number(lookup.get((c,g,seed))) for _,c,g in terms]
            effects[seed]=None if any(v is None for v in values) else sum(t[0]*v for t,v in zip(terms,values))
        results[label]=paired_estimate(effects)
    return results


def ring_status(context,regime='constant'):
    if not context: return 'missing'
    if not context.get('qualified'): return 'unqualified'
    row=get(context,'regimes',regime)
    if row is None: return 'missing'
    if row.get('conditional_ring_candidate'): return 'passed'
    failed=[name for gates in row.get('gates',{}).values() for name,ok in gates.items() if not ok]
    physical={'radial_starts_within_hidden_bounds','radial_directions_nondegenerate'}
    if failed and any(name not in physical for name in failed): return 'failed'
    # Geometry-only success with failed numerical/physical confirmation is not
    # evidence against a ring, and an invalid radial construction cannot pass.
    return 'unqualified'


def context_card(dynamics,intervention,index):
    d=dynamics or {};i=intervention or {}
    same=bool(d and i and d.get('episode_id')==i.get('episode_id') and d.get('global_step')==i.get('global_step'))
    technical=bool(i.get('qualified') and get(i,'family','qualified')
                   and get(i,'behavior','validation_qualified'))
    geometry=bool(same and ring_status(d)=='passed'
                  and get(d,'regimes','constant','tested_normal_recovery_confirmed'))
    phase_reference=get(i,'cue_repair','rows','15','correct','none','correct_reference')
    reference_ok=bool(phase_reference and phase_reference.get('defined')
        and number(phase_reference.get('radius')) is not None
        and phase_reference['radius']/np.sqrt(512)>=1e-4
        and number(phase_reference.get('coverage')) is not None and phase_reference['coverage']>=.8)
    phase_qualified=technical and geometry and reference_ok
    card=dict(index=index,episode_id=d.get('episode_id',i.get('episode_id')),
        global_step=d.get('global_step',i.get('global_step')),
        dynamics_present=bool(d),intervention_present=bool(i),context_identity_match=same,
        ring={r:ring_status(d,r) for r in ('constant','forward_backward')},
        tested_recovery_confirmed={r:bool(get(d,'regimes',r,'tested_normal_recovery_confirmed')) for r in ('constant','forward_backward')},
        passive_occupancy_qualified=bool(get(d,'native_occupancy','qualified')),
        passive_broad_support=bool(get(d,'native_occupancy','broad_native_support')),
        technically_valid_native_behavior=technical,qualified_ring_and_recovery=geometry,
        phase_reference_qualified=reference_ok,phase_repair_qualified=phase_qualified,
        paired_geometry_behavior_qualified=phase_qualified,phase_descriptive={},phase_raw_descriptive={},behavior={},
        native_behavior_scope='Technically validated native anchors; does not require a ring or broad passive occupancy.')
    for alt in ('ordinary','wrong'):
        alternatives=('ordinary',) if alt=='ordinary' else ('wrong_plus','wrong_minus')
        raw_effects={}
        for edit in ('none','plus30','minus30'):
            values=[number(get(i,'cue_repair','rows','15',cue,edit,'raw_correct_reference','phase_mae_deg'))
                    for cue in (*alternatives,'correct')]
            raw_effects[edit]=None if any(v is None for v in values) else float(np.mean(values[:-1])-values[-1])
        raw_donor=None if any(raw_effects[e] is None for e in ('plus30','minus30')) else (raw_effects['plus30']+raw_effects['minus30'])/2
        for kind in ('donor_mean','donor_minus_no_edit'):
            suffix=f'{alt}_{kind}'
            card['phase_descriptive'][suffix]=number(get(i,'cue_repair','contrasts','15',alt+'_minus_correct',kind))
            card['phase_raw_descriptive'][suffix]=raw_donor if kind=='donor_mean' else (
                None if raw_donor is None or raw_effects['none'] is None else raw_donor-raw_effects['none'])
            card['behavior'][suffix]=number(get(i,'behavior','contrasts','correct_minus_'+alt,kind))
    records=get(i,'behavior','records') or []
    eligible=[r for r in records if r.get('qualified')]
    card['behavior_exposure']=dict(tapes=get(i,'behavior','tapes'),horizon=get(i,'behavior','horizon'),
        qualified_branches=len(eligible),unqualified_branches=len(records)-len(eligible),
        decision_range=[min(r['decisions'] for r in eligible),max(r['decisions'] for r in eligible)] if eligible else None,
        mean_blind_steps=mean([r.get('blind_steps') for r in eligible]),
        first_hit_censored_branches=sum(bool(r.get('first_hit_censored')) for r in eligible))
    return card


def extract_model(condition,gap,seed,stages,availability=None):
    """One compact card per model, never one observation per episode/phase."""
    metrics,curves,readouts,pca_counts={}, {}, {}, {}
    evaluation=stages.get('evaluation')
    if evaluation:
        variants=evaluation.get('variants',{})
        for name,row in variants.items(): metrics['score_'+name]=number(row.get('mean_score'))
        for which in ('onset','each'):
            for name in ('blind_reset_cost','seen_reset_cost','additional_reset_cost_in_blackout'):
                metrics[f'{which}_{name}']=number(get(evaluation,'reset_contrasts',which,name))
        for setting in ('blind','seen'):
            metrics['current_prediction_benefit_'+setting]=number(get(evaluation,'current_predictive_input_benefit',setting))
    activity=stages.get('activity')
    if activity:
        for subset in ('all','known_visible','known_blind'):
            value=get(activity,'covariance','h',subset)
            if not value: continue
            prefix='pca_'+subset+'_'
            counts=value.get('counts',{});per_episode=counts.get('episode_counts',[])
            pca_counts[subset]=dict(rows=counts.get('rows'),episodes=counts.get('episodes'),
                episode_count_range=[min(per_episode),max(per_episode)] if per_episode else None,
                weighting='Covariance/SS use observed row counts; full episode counts remain in producer summary.')
            metrics[prefix+'between_fraction']=number(value.get('between_fraction'))
            for kind in ('total','within'):
                for item in ('top10_share','participation_ratio'):
                    metrics[prefix+kind+'_'+item]=number(get(value,'spectra',kind,item))
                if subset in ('all','known_blind'):
                    curves[prefix+kind]=get(value,'spectra',kind,'cumulative')
            metrics[prefix+'episode_actor_fraction']=number(get(value,'actor_alignment','between_fraction'))
        for feature in ('h','hc'):
            readouts[feature]={}
            for target in TARGETS:
                readouts[feature][target]={}
                for label in BINS:
                    value=get(activity,'angular_readouts',feature,'targets',target,label)
                    if not value: continue
                    actual=value['actual'];null=value['null_replicates']
                    row=dict(mae=number(get(actual,'ridge','circular_mae_deg','episode_equal')),
                        constant_mae=number(get(actual,'constant','circular_mae_deg','episode_equal')),
                        r2=number(get(actual,'unit_angle_comparison','episode_equal','r2_vs_constant')),
                        sincos_r2=number(get(actual,'sincos_regression_comparison','episode_equal','r2_vs_constant')),
                        rows=actual.get('rows'),episodes=actual.get('episodes'),paired_rows=actual.get('paired_angle_rows'),
                        retained_fraction=actual.get('paired_retained_fraction'),
                        null_mae=[number(get(r,'ridge','circular_mae_deg','episode_equal')) for r in null])
                    readouts[feature][target][label]=row
                    metrics[f'readout_{feature}_{target}_{label}_r2']=row['r2']
    dynamics=stages.get('dynamics') or {}
    interventions=stages.get('interventions') or {}
    ds={r['index']:r for r in dynamics.get('contexts',[])}
    ins={r['index']:r for r in interventions.get('contexts',[])}
    if len(ds)!=len(dynamics.get('contexts',[])) or len(ins)!=len(interventions.get('contexts',[])):
        raise ValueError('Duplicate context index')
    if any(idx not in range(4) for idx in (*ds,*ins)): raise ValueError('Unexpected selected context index')
    cards=[context_card(ds.get(idx),ins.get(idx),idx) for idx in sorted(set(ds)|set(ins))]
    contexts_known=stages.get('dynamics') is not None or stages.get('interventions') is not None
    count=dict(requested_contexts=4,selected_contexts=len(cards) if contexts_known else None,
        not_selected=4-len(cards) if contexts_known else None,
        technically_valid_native_behavior=sum(r['technically_valid_native_behavior'] for r in cards),
        paired_geometry_behavior=sum(r['paired_geometry_behavior_qualified'] for r in cards))
    for regime in ('constant','forward_backward'):
        counts={label:sum(r['ring'][regime]==label for r in cards) for label in ('passed','failed','unqualified','missing')}
        counts.update(selected=len(cards) if contexts_known else None,
            not_selected=4-len(cards) if contexts_known else None,requested=4,
            tested_recovery_confirmed=sum(r['tested_recovery_confirmed'][regime] for r in cards))
        count[regime]=counts
        # These are observed context counts, not zero-filled missing model data.
        metrics[regime+'_ring_count']=float(counts['passed']) if stages.get('dynamics') is not None and ds else None
        metrics[regime+'_confirmed_recovery_count']=float(counts['tested_recovery_confirmed']) if stages.get('dynamics') is not None and ds else None
    for alt in ('ordinary','wrong'):
        for kind in ('donor_mean','donor_minus_no_edit'):
            suffix=f'{alt}_{kind}'
            metrics['phase_repair_'+suffix]=mean([r['phase_descriptive'][suffix] for r in cards if r['phase_repair_qualified']])
            metrics['phase_raw_sensitivity_'+suffix]=mean([r['phase_raw_descriptive'][suffix] for r in cards if r['phase_repair_qualified']])
            metrics['native_behavior_'+suffix]=mean([r['behavior'][suffix] for r in cards if r['technically_valid_native_behavior']])
            metrics['paired_geometry_behavior_'+suffix]=mean([r['behavior'][suffix] for r in cards if r['paired_geometry_behavior_qualified']])
    spectra={}
    for t in ('0','30'):
        for metric in ('raw','relative_rms'):
            rows=[get(r,'regimes','constant','local_spectra',t,metric) for r in ds.values() if r.get('qualified')]
            rows=[r for r in rows if r]
            prefix=f'local_t{t}_{metric}_'
            count['local_spectra_t'+t]=len(rows)
            for item in ('spectral_radius','largest_singular_value','top_eigenvector_tangent_overlap','top_singular_vector_tangent_overlap'):
                metrics[prefix+item]=mean([r.get(item) for r in rows])
            # Full secondary spectra remain in producer archives. The compact
            # report retains the primary30-update raw curves and all scalar summaries.
            if t=='30' and metric=='raw':
                spectra[prefix+'singular_values']=np.mean([r['singular_values'] for r in rows],axis=0).tolist() if rows else None
                spectra[prefix+'eigenvalue_modulus']=np.mean([r['ordered_eigenvalue_modulus'] for r in rows],axis=0).tolist() if rows else None
    return dict(model=key(condition,gap,seed),condition=condition,training_task_max_gap=gap,training_seed=seed,
        availability=availability or {s:'available' if stages.get(s) is not None else 'missing' for s in STAGES},
        metrics=metrics,curves=curves,pca_counts=pca_counts,readouts=readouts,
        context_counts=count,contexts=cards,local_spectra=spectra)


def aggregate(inputs):
    """Input items: identity plus optional producer dictionaries; exactly16 slots."""
    table={}
    for item in inputs:
        identity=(item['condition'],item['task_max_gap'],item['seed'])
        if identity not in COHORT or identity in table: raise ValueError('Duplicate or unexpected model identity')
        table[identity]=item
    models=[]
    for c,g,s in COHORT:
        item=table.get((c,g,s),{})
        models.append(extract_model(c,g,s,item.get('stages',{}),item.get('availability')))
    all_metrics=sorted({name for m in models for name in m['metrics']})
    contrasts={name:factorial_estimates(models,name) for name in all_metrics}
    available={stage:sum(m['availability'][stage]=='available' for m in models) for stage in STAGES}
    return dict(schema=1,status='complete' if all(v==16 for v in available.values()) else 'partial',
        planned_models=16,stage_available_models=available,models=models,paired_seed_effects=contrasts,
        interpretation='One trained seed is the independent unit. Four complete pairs have minimum two-sided exact p0.125. Intervals are descriptive; missing data are never zeros. A no-ring result and an unqualified test are different outcomes.')


class ArtifactError(ValueError): pass


def require(condition,message):
    if not condition: raise ArtifactError(message)


def artifact_check(path,digest):
    require(Path(path).is_file(),f'Missing artifact: {path}')
    require(sha(path)==digest,f'Artifact hash mismatch: {path}')


def load_pairs(paths,protocol_path=PROTOCOL,*,verify=None):
    """Eight prescribed pair packages, each authenticated by the operational runner."""
    from scripts import run_simple_memory_analysis as runner
    verify=runner.verify_pipeline if verify is None else verify
    protocol=json.loads(Path(protocol_path).read_text());pd=sha(protocol_path)
    require(protocol.get('status')=='frozen' and protocol.get('source_sha256'),'Analysis is not frozen')
    for name,digest in protocol['source_sha256'].items():artifact_check(ROOT/name,digest)
    output,provenance,plot_sources,seen=[],[],{},set()
    stage_map=dict(evaluation='behavior',activity='activity',dynamics='dynamics',interventions='interventions')
    for path in map(Path,paths):
        match=re.fullmatch(r'ring-20261105-simplemem-analysis-(native|blackout)-s([1-4])',path.name)
        require(match is not None,f'Unexpected analysis pair directory: {path}')
        gap,seed=(0 if match[1]=='native' else 60),int(match[2])
        require((gap,seed) not in seen,'Duplicate analysis pair');seen.add((gap,seed))
        if not path.is_dir():
            availability=dict.fromkeys(STAGES,'missing');error='Pair has not been collected'
            for condition in ('gc100','pgc33'):
                output.append(dict(condition=condition,task_max_gap=gap,seed=seed,stages={},availability=availability.copy()))
            provenance.append(dict(pair=str(path),status='missing',reason=error));continue
        try:
            verified=verify(path)
            require(verified.get('complete') is True,'Pipeline verifier did not confirm completion')
            pipeline=json.loads((path/'pipeline.json').read_text())
            identities={(r['endpoint']['condition'],r['endpoint']['training_task_max_gap'],r['endpoint']['training_seed'])
                        for r in pipeline['runs']}
            require(identities=={('gc100',gap,seed),('pgc33',gap,seed)} and len(pipeline['runs'])==2,
                    'Pair identity does not match the prescribed population/seed/training task')
            for record in pipeline['runs']:
                endpoint=record['endpoint'];condition=endpoint['condition'];run=runner.local_path(path,record['run'])
                stages={}
                for stage,producer in stage_map.items():
                    summary=runner.local_path(run,record['stages'][producer]['validation']['summary_path'])
                    stages[stage]=json.loads(summary.read_text())
                    plot_sources[(key(condition,gap,seed),stage)]=summary.parent
                output.append(dict(condition=condition,task_max_gap=gap,seed=seed,stages=stages,
                                   availability=dict.fromkeys(STAGES,'available')))
            provenance.append(dict(pair=str(path),status='verified',validation=verified))
        except (OSError,ValueError,KeyError,AssertionError) as exc:
            # A damaged pair never becomes two failed scientific models.
            output=[row for row in output if (row['task_max_gap'],row['seed'])!=(gap,seed)]
            for condition in ('gc100','pgc33'):
                output.append(dict(condition=condition,task_max_gap=gap,seed=seed,stages={},availability=dict.fromkeys(STAGES,'invalid')))
            provenance.append(dict(pair=str(path),status='invalid',reason=str(exc)))
    return output,dict(analysis_protocol_sha256=pd,pair_packages=provenance,
        pipeline_verifier_sha256=sha(ROOT/'scripts/run_simple_memory_analysis.py')),plot_sources,protocol


def group_models(summary,condition,gap):
    return [m for m in summary['models'] if m['condition']==condition and m['training_task_max_gap']==gap]


def plotted_mean(vectors):
    vectors=[v for v in vectors if v is not None]
    return np.mean(np.asarray(vectors,float),axis=0) if vectors else None


def paired_points(ax,summary,metric,other=None):
    displayed=False
    for j,(condition,gap) in enumerate(ARMS):
        group=group_models(summary,condition,gap)
        for k,row in enumerate(group):
            value=number(row['metrics'].get(metric))
            if value is not None:
                ax.scatter(j+(k-1.5)*.035-(.07 if other else 0),value,s=24,color=COLORS[condition]);displayed=True
            if other:
                value=number(row['metrics'].get(other))
                if value is not None:
                    ax.scatter(j+(k-1.5)*.035+.07,value,s=27,color=COLORS[condition],marker='x');displayed=True
        values=[m['metrics'].get(metric) for m in group]
        center=mean(values)
        if center is not None:ax.plot([j-.19,j+.19],[center,center],color=COLORS[condition],lw=2)
    for gap,start in ((0,0),(60,2)):
        left,right=group_models(summary,'gc100',gap),group_models(summary,'pgc33',gap)
        for a,b in zip(left,right):
            x,y=number(a['metrics'].get(metric)),number(b['metrics'].get(metric))
            if x is not None and y is not None:ax.plot([start,start+1],[x,y],color='.8',lw=.7,zorder=0)
    ax.axhline(0,color='.8',lw=.7,zorder=0)
    ax.set(xticks=range(4),xticklabels=['Grid\nnative','PGC\nnative','Grid\nblackout','PGC\nblackout'])
    if not displayed:ax.text(.5,.55,'Not available',transform=ax.transAxes,ha='center',color='.5')
    if other:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([],[],color='.3',marker='o',ls='',label='vs ordinary'),
                           Line2D([],[],color='.3',marker='x',ls='',label='vs wrong')],frameon=False,fontsize=7)


def save_figure(fig,folder,name):
    fig.savefig(folder/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_figures(summary,inputs,plot_sources,out,*,context_figures=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(out)/'figures';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for ax,metric,title,ylabel in zip(axes,
        ['score_blind','onset_additional_reset_cost_in_blackout','current_prediction_benefit_blind'],
        ['Navigation','Memory erasure','Current prediction'],
        ['Episode reward','Extra reset cost in blackout','Reward lost after code replacement']):
        paired_points(ax,summary,metric);ax.set(title=title,ylabel=ylabel)
    save_figure(fig,folder,'behavior')

    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for kind,ax in zip(('total','within'),axes[:2]):
        displayed=False
        for condition,gap in ARMS:
            curve=plotted_mean([m['curves'].get('pca_all_'+kind) for m in group_models(summary,condition,gap)])
            if curve is not None:
                displayed=True
                ax.plot(np.arange(1,len(curve)+1),100*curve,color=COLORS[condition],ls='-' if gap==60 else '--',
                        label=('Grid' if condition=='gc100' else 'PGC')+(' / blackout' if gap else ' / native'))
        ax.set(xlabel='Principal components',ylabel='Variance (%)',xlim=(1,100),ylim=(0,100),
               title='All activity' if kind=='total' else 'Episode means removed')
        if not displayed:ax.text(.5,.5,'Not available',transform=ax.transAxes,ha='center',color='.5')
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(frameon=False,fontsize=6)
    paired_points(axes[2],summary,'pca_all_between_fraction')
    axes[2].set(title='Episode-average shifts',ylabel='Fraction of total variance')
    save_figure(fig,folder,'activity')

    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for row,gap in enumerate((0,60)):
        for col,target in enumerate(TARGETS):
            ax=axes[row,col]
            displayed=False
            for condition in ('gc100','pgc33'):
                group=group_models(summary,condition,gap)
                for field,ls in (('mae','-'),('constant_mae',':')):
                    values=[mean([get(m,'readouts','h',target,label,field) for m in group]) for label in BINS]
                    displayed|=any(v is not None for v in values)
                    ax.plot(range(5),[np.nan if v is None else v for v in values],ls=ls,marker='o' if field=='mae' else None,
                        markersize=3,color=COLORS[condition],label=('Grid' if condition=='gc100' else 'PGC')+(' constant' if field=='constant_mae' else ''))
                null_range=[]
                for label in BINS:
                    nulls=[get(m,'readouts','h',target,label,'null_mae') for m in group]
                    nulls=[n for n in nulls if n]
                    replicate_means=[mean([n[r] for n in nulls]) for r in range(20)] if nulls else []
                    valid=[v for v in replicate_means if v is not None]
                    null_range.append((min(valid),max(valid)) if valid else (np.nan,np.nan))
                ax.fill_between(range(5),np.asarray(null_range)[:,0],np.asarray(null_range)[:,1],color=COLORS[condition],alpha=.09)
            ax.set(xticks=range(5),xticklabels=['Visible','1–15','16–30','31–45','46–60'],ylim=(0,180),
                xlabel='Blind age (updates)',title=target.replace('_',' '))
            if col==0:ax.set_ylabel(('Native training' if gap==0 else 'Blackout training')+'\nAngle error (degrees)')
            if not displayed:ax.text(.5,.5,'Not available',transform=ax.transAxes,ha='center',color='.5')
    axes[0,0].legend(frameon=False,fontsize=6)
    save_figure(fig,folder,'angular_information')

    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    paired_points(axes[0,0],summary,'constant_ring_count')
    paired_points(axes[0,1],summary,'constant_confirmed_recovery_count')
    axes[0,0].set(title='Qualified conditional families',ylabel='Contexts per model',ylim=(-.15,4.3),yticks=range(5))
    axes[0,1].set(title='Confirmed tested-normal recovery',ylabel='Contexts per model',ylim=(-.15,4.3),yticks=range(5))
    paired_points(axes[1,0],summary,'phase_repair_ordinary_donor_mean','phase_repair_wrong_donor_mean')
    paired_points(axes[1,1],summary,'paired_geometry_behavior_ordinary_donor_mean','paired_geometry_behavior_wrong_donor_mean')
    axes[1,0].set(title='Correct cue: state repair',ylabel='Angle-error reduction (degrees)')
    axes[1,1].set(title='Same contexts: navigation',ylabel='Reward improvement')
    save_figure(fig,folder,'mechanism')

    # Exact local eigenvalues are display-only points, never independent samples.
    fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
    raw_inputs={(r['condition'],r['task_max_gap'],r['seed']):r for r in inputs}
    for col,gap in enumerate((0,60)):
        displayed_eigen,displayed_singular=False,False
        for condition in ('gc100','pgc33'):
            values=[]
            for m in group_models(summary,condition,gap):
                stage=get(raw_inputs.get((condition,gap,m['training_seed']),{}),'stages','dynamics')
                directory=plot_sources.get((m['model'],'dynamics'))
                if not stage or directory is None:continue
                for context in stage.get('contexts',[]):
                    if not context.get('qualified'):continue
                    with np.load(directory/context['arrays']['path'],allow_pickle=False) as arrays:
                        name='constant_spectra_t30_eigenvalues'
                        if name in arrays: values.append(arrays[name])
            if values:
                displayed_eigen=True
                values=np.concatenate(values)
                axes[0,col].scatter(values.real,values.imag,s=2,alpha=.2,color=COLORS[condition],rasterized=True)
            singular=plotted_mean([m['local_spectra'].get('local_t30_raw_singular_values') for m in group_models(summary,condition,gap)])
            if singular is not None:
                displayed_singular=True
                axes[1,col].plot(np.arange(1,len(singular)+1),singular,color=COLORS[condition],label='Grid' if condition=='gc100' else 'PGC')
        angle=np.linspace(0,2*np.pi,256)
        axes[0,col].plot(np.cos(angle),np.sin(angle),color='.7',ls='--',lw=.7)
        axes[0,col].set(aspect='equal',xlabel='Real eigenvalue part',ylabel='Imaginary part',title='Native training' if not gap else 'Blackout training')
        axes[1,col].axhline(1,color='.7',ls='--',lw=.7)
        axes[1,col].set(xlabel='Singular-value rank',ylabel='One-step amplification')
        if not displayed_eigen:axes[0,col].text(.5,.5,'Not available',transform=axes[0,col].transAxes,ha='center',color='.5')
        if not displayed_singular:axes[1,col].text(.5,.5,'Not available',transform=axes[1,col].transAxes,ha='center',color='.5')
    if axes[1,0].get_legend_handles_labels()[0]:axes[1,0].legend(frameon=False)
    save_figure(fig,folder,'local_spectra')
    gallery=[]
    if context_figures:
        gallery_dir=folder/'contexts';gallery_dir.mkdir(exist_ok=True)
        for item in inputs:
            model=key(item['condition'],item['task_max_gap'],item['seed'])
            stage=item.get('stages',{}).get('dynamics');directory=plot_sources.get((model,'dynamics'))
            if not stage or directory is None:continue
            for context in stage['contexts']:
                if not context.get('qualified'):
                    gallery.append(dict(model=model,context=context['index'],status='unqualified',path=None));continue
                with np.load(directory/context['arrays']['path'],allow_pickle=False) as arrays:
                    z0=arrays['constant_z_0'];origin=z0.mean(0)
                    _,_,v=np.linalg.svd(z0-origin,full_matrices=False);basis=v[:2].T
                    for regime in ('constant','forward_backward'):
                        result=context['regimes'][regime]
                        fig,axs=plt.subplots(1,4,figsize=(12,3),layout='constrained')
                        for t,color in ((0,'.65'),(30,'#3379B5'),(60,'#D87532')):
                            curve=(arrays[f'{regime}_z_{t}']-origin)@basis
                            axs[0].plot(np.r_[curve[:,0],curve[0,0]],np.r_[curve[:,1],curve[0,1]],color=color,label=f'{t/15:g}s')
                        axs[0].set(xlabel='PC1 (display only)',ylabel='PC2',aspect='equal');axs[0].legend(frameon=False,fontsize=7)
                        for metric,ls in (('raw','-'),('relative_rms','--')):
                            ts=sorted(map(int,result['trajectory']))
                            label='raw' if metric=='raw' else 'h/c scaled'
                            axs[1].plot(np.asarray(ts)/15,[result['trajectory'][str(t)][metric]['radius_retention'] for t in ts],ls=ls,label=label)
                            inv=result['invariance'][metric];steps=list(map(int,inv['horizons']))
                            seconds=np.asarray(steps)*(1 if regime=='constant' else 2)/15
                            axs[2].plot(seconds,[inv['horizons'][str(t)]['residual_to_fixed_curve_over_radius']['p90'] for t in steps],ls=ls,label=metric)
                            axs[3].plot(seconds,[inv['horizons'][str(t)]['phase_transport']['circular_coverage'] for t in steps],ls=ls,label=metric)
                        axs[1].set(xlabel='Seconds',ylabel='Family radius retained');axs[1].legend(frameon=False,fontsize=7)
                        axs[2].axhline(.05,color='.7',ls=':',lw=.7);axs[2].set(xlabel='Seconds after reference',ylabel='Distance / reference radius')
                        axs[3].axhline(.8,color='.7',ls=':',lw=.7);axs[3].set(xlabel='Seconds after reference',ylabel='Phase coverage',ylim=(0,1.02))
                        status=ring_status(context,regime);fig.suptitle(f'{model} / context {context["index"]} / {regime}: {status}',fontsize=9)
                        name=f'{model}_context{context["index"]}_{regime}'
                        save_figure(fig,gallery_dir,name)
                        gallery.append(dict(model=model,context=context['index'],regime=regime,status=status,path=f'figures/contexts/{name}.png'))
    return gallery


def fmt(value,digits=2): return 'unavailable' if value is None else f'{value:+.{digits}f}'


def readme(summary,gallery):
    available=summary['stage_available_models']
    lines=['# Native navigation with brief sensory blackouts','',
        f"Results available: {available['evaluation']}/16 navigation evaluations; {available['activity']}/16 activity analyses; "
        f"{available['dynamics']}/16 dynamics analyses; {available['interventions']}/16 intervention analyses.",'',
        'Each dot is one trained seed. We compare Grid and PGC + grid models trained with normal vision or brief sensory blackouts. Each complete model was trained for 1 billion environment frames.','',
        '![Navigation and memory tests](figures/behavior.png)','',
        'Left: native navigation reward during the blackout test. Middle: the extra cost of erasing memory at blackout onset, after subtracting the cost of the same scheduled erasure with vision. Right: reward lost when current predictive entries are replaced with ordinary grid code. The saved goal message stays unchanged.','']
    primary=get(summary,'paired_seed_effects','score_blind','pgc_minus_grid_blackout_training')
    if primary and primary['n_pairs']:
        ci=primary['descriptive_ci95']
        interval='unavailable' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'
        lines += [f"PGC minus Grid reward in blackout-trained models: **{fmt(primary['mean'])}**, "
                  f"with descriptive 95% interval {interval} ({primary['n_pairs']}/4 paired seeds).",'']
    lines += ['![Activity variance](figures/activity.png)','',
        'Between-episode variance measures how far episode activity centers shift; a higher fraction suggests stronger context-dependent activity. Subtracting each episode’s mean reveals variation within episodes. The JSON also reports these activity measurements using only known-goal blackout periods.','',
        '![Held-out angular information](figures/angular_information.png)','',
        'Solid curves show angular error on episodes excluded from fitting. Dotted curves use one constant angle learned from the training episodes. Shaded ranges show the 20 episode-rotated-label null fits. Improvement over the constant baseline measures additional angular information in activity.','',
        '| Blackout-trained model | Heading R² | Travel direction R² | Goal bearing R² |',
        '| --- | ---: | ---: | ---: |']
    for condition in ('gc100','pgc33'):
        group=group_models(summary,condition,60)
        values=[mean([get(m,'readouts','h',target,'blind_age_46_60','r2') for m in group]) for target in TARGETS]
        lines.append('| '+('Grid' if condition=='gc100' else 'PGC + grid')+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', 'R² here is improvement over the fixed constant-angle predictor at blind ages 46–60. Negative values mean the constant did better. Missing age bins remain unavailable; all bins and both feature sets are retained in the JSON.','',
        '![Conditional dynamics and cue repair](figures/mechanism.png)','',
        'Top: how many selected contexts pass the full-state conditional-ring and recovery checks. Unqualified and unselected contexts identify where this evidence remains unavailable. Bottom: how much a correct one-update cue repairs ±30° state substitutions compared with ordinary or wrong cues, and how navigation changes in those same qualified contexts. The JSON also retains behavioral effects at every technically valid native anchor.','',
        '![Local recurrent spectra](figures/local_spectra.png)','',
        'Local derivatives measure changes of small state errors with the input fixed, evaluated after 30 blind updates. Eigenvalues describe local recurrent modes; singular values show possible one-step amplification. Recovery and behavior tests link these measurements to memory use.','',
        '| Model | Selected | Ring pass / fail / unqualified | Confirmed recovery | Valid behavior |',
        '| --- | ---: | --- | ---: | ---: |']
    for model in summary['models']:
        counts=model['context_counts'];ring=counts['constant']
        selected=counts['selected_contexts']
        if selected is None: values=['unavailable']*4
        else:values=[str(selected),f"{ring['passed']} / {ring['failed']} / {ring['unqualified']} (missing {ring['missing']})",
                     str(ring['tested_recovery_confirmed']),str(counts['technically_valid_native_behavior'])]
        lines.append('| '+model['model']+' | '+' | '.join(values)+' |')
    lines += ['', 'Four paired training seeds give limited statistical resolution: the smallest two-sided exact sign-flip p-value is 0.125. Intervals are descriptive. The trained seeds provide the independent comparisons; episodes, contexts, phases and random future tapes are measurements within each seed.','',
        'Together these measurements distinguish conditional structure, cue repair and navigation benefit in the tested contexts. Broad passive native occupancy measures support across additional naturally visited states.','',
        '[Numerical summary](summary.json) · [Report provenance and rules](report_protocol.json)','']
    if gallery:
        lines += ['## Every selected context','',
            'The same PCA projection is fixed from the initial full-state family and used only for display. The gallery includes every selected context, including failed and unqualified checks.','']
        for item in gallery:
            label=f"{item['model']} / context {item['context']} / {item.get('regime','not run')}: {item['status']}"
            lines.append(f"- [{label}]({item['path']})" if item['path'] else '- '+label)
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--root',type=Path,help='Parent directory of the eight prescribed collected pair packages')
    source.add_argument('--pairs',type=Path,nargs='+',help='Explicit prescribed pair directories; each must pass verify_pipeline')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--no-context-figures',action='store_true',help='Numerical report remains complete; omit optional display gallery')
    args=parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Report output must be empty')
    if not 1<=args.threads<=8:raise ValueError('Use1..8 CPU threads')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        paths=args.pairs or [args.root/f'ring-20261105-simplemem-analysis-{task}-s{seed}'
                            for task in ('native','blackout') for seed in SEEDS]
        inputs,provenance,plot_sources,protocol=load_pairs(paths,args.protocol)
        summary=aggregate(inputs)
        args.out.mkdir(parents=True,exist_ok=True)
        report_protocol=dict(schema=1,role='Implementation of aggregation already prespecified before model outcomes',
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),analysis_protocol_sha256=sha(args.protocol),
            frozen_aggregation=protocol['aggregation_and_figures'],
            source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),
                'tests/test_report_simple_memory.py':sha(ROOT/'tests/test_report_simple_memory.py'),
                'scripts/run_simple_memory_analysis.py':sha(ROOT/'scripts/run_simple_memory_analysis.py')},
            independent_unit='Training seed; four planned paired seeds',
            phase_scope='Primary constant conditional_ring_candidate AND tested_normal_recovery_confirmed AND technically valid native behavior; paired behavior uses those same contexts.',
            phase_reference_eligibility=dict(update=15,
                reference='Unedited correct-pulse full-state reference: cue_repair.rows[15].correct.none.correct_reference',
                require_defined_and_finite=True,full_state_dimensions=512,
                radius_over_sqrt_dimensions_min=1e-4,phase_coverage_min=.8,
                origin='Pre-data reuse of the frozen geometry resolution and phase-coverage cutoffs for interpreting phase on a noncollapsed reference.',
                scientific_protocol_changed=False),
            broad_behavior_scope='All technically valid native anchors, regardless of ring or passive occupancy.',
            missing='Missing/invalid producer outputs are not scientific failures and never imputed as zero.',
            input_provenance=provenance,context_gallery_requested=not args.no_context_figures)
        (args.out/'report_protocol.json').write_text(json.dumps(report_protocol,indent=2,allow_nan=False)+'\n')
        gallery=make_figures(summary,inputs,plot_sources,args.out,context_figures=not args.no_context_figures)
        summary.update(provenance=provenance,report_protocol_sha256=sha(args.out/'report_protocol.json'),
                       report_source_sha256=sha(__file__),context_gallery=gallery)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        (args.out/'README.md').write_text(readme(summary,gallery))
    print(json.dumps(dict(status=summary['status'],available=summary['stage_available_models'],out=str(args.out))))


if __name__=='__main__':main()
