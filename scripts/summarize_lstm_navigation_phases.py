"""Summarize phase localization; every inferential replicate is a training seed."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_navigation_phases import describe_trace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_navigation_phases_1B'

def estimate(x):
    x=np.asarray(x,float)
    return dict(values=x.tolist(),n=len(x),mean=float(x.mean()),sem=float(stats.sem(x)))

def paired(a,b):
    d=np.asarray(b)-np.asarray(a);n=len(d);se=stats.sem(d)
    flips=np.array([np.mean(d*s) for s in itertools.product((-1,1),repeat=n)])
    ci=stats.t.interval(.95,n-1,loc=d.mean(),scale=se) if se else [d.mean(),d.mean()]
    return dict(difference=float(d.mean()),differences=d.tolist(),ci95=list(map(float,ci)),
        paired_t_p=float(stats.ttest_1samp(d,0).pvalue) if se else (1. if d.mean()==0 else 0.),
        exact_signflip_p=float(np.mean(np.abs(flips)>=abs(d.mean())-1e-12)))

def holm(family):
    for raw,out in [('paired_t_p','holm_t_p'),('exact_signflip_p','holm_exact_p')]:
        running=0.
        for i,key in enumerate(sorted(family,key=lambda k:family[k][raw])):
            running=max(running,min(1.,(len(family)-i)*family[key][raw]));family[key][out]=running

def matched_trips(a,b):
    aa={(r['episode'],r['slot'],r['return_ordinal']):r for r in a if r['success']}
    bb={(r['episode'],r['slot'],r['return_ordinal']):r for r in b if r['success']}
    keys=sorted(aa.keys()&bb.keys());assert keys
    max_distance=max(abs(aa[k]['start_distance']-bb[k]['start_distance']) for k in keys)
    assert max_distance<1e-5,max_distance
    fields=['seconds','before_near_seconds','after_first_near_seconds','excess_path','near_exit_count']
    return dict(n=len(keys),maximum_spawn_distance_difference=max_distance,
                metrics={f:dict(a=float(np.mean([aa[k][f] for k in keys])),b=float(np.mean([bb[k][f] for k in keys])),
                                difference=float(np.mean([bb[k][f]-aa[k][f] for k in keys]))) for f in fields})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=list(range(1,9)))
    ap.add_argument('--refresh-from-trace',action='store_true');args=ap.parse_args()
    runs={}
    for s in args.seeds:
        for c in ['gc100','pgc33']:
            p=OUT/f'{c}_s{s}.json';r=json.loads(p.read_text())
            if args.refresh_from_trace:
                z=np.load(OUT/f'{c}_s{s}_trace.npz');r['phase_analysis']=describe_trace(z['trace'],1350,r['n_envs'],list(z['variants']))
                p.write_text(json.dumps(r,indent=2)+'\n')
            assert r['baseline_records_match_exactly'];runs[c,s]=r
    conditions=[('gc100','intact'),('pgc33','intact'),('pgc33','goal_grid'),('pgc33','current_grid'),('pgc33','both_grid')]
    estimates={}
    for c,v in conditions:
        rows=[runs[c,s] for s in args.seeds]
        e={}
        for m in ['mean_score','mean_first_goal_step','frac_found_goal','mean_path_eff','post_discovery_goal_rate']:
            e[m]=estimate([r['variants'][v][m] for r in rows])
        e['first_discovery_score']=estimate([10*r['variants'][v]['frac_found_goal'] for r in rows])
        e['return_score']=estimate([r['variants'][v]['mean_score']-10*r['variants'][v]['frac_found_goal'] for r in rows])
        e['trip_summary']={k:estimate([r['phase_analysis'][v]['trip_summary'][k] for r in rows])
                           for k in rows[0]['phase_analysis'][v]['trip_summary']}
        e['first_return_20s']={k:estimate([r['phase_analysis'][v]['first_return_20s'][k] for r in rows])
                              for k in rows[0]['phase_analysis'][v]['first_return_20s']}
        e['phases']={phase:{k:estimate([r['phase_analysis'][v]['phases'][phase][k] for r in rows])
                    for k in rows[0]['phase_analysis'][v]['phases'][phase]}
                    for phase in rows[0]['phase_analysis'][v]['phases']}
        estimates[f'{c}/{v}']=e
    comparisons={}
    for name,ca,va,cb,vb in [('native_pgc_minus_grid','gc100','intact','pgc33','intact'),
                            ('goal_deshift_minus_intact','pgc33','intact','pgc33','goal_grid')]:
        sa=estimates[f'{ca}/{va}'];sb=estimates[f'{cb}/{vb}']
        family={m:paired(sa[m]['values'],sb[m]['values']) for m in
                ['mean_score','mean_first_goal_step','frac_found_goal','mean_path_eff','post_discovery_goal_rate']}
        for m in ['mean_complete_before_near_seconds','mean_complete_after_first_near_seconds',
                  'mean_complete_excess_path_m','fraction_near_attempts_with_exit']:
            family[m]=paired(sa['trip_summary'][m]['values'],sb['trip_summary'][m]['values'])
        for phase in ['near_under_0.6m','far_over_0.9m','first_second_after_respawn']:
            m='mean_distance_progress_m_per_second'
            family[phase+'_progress']=paired(sa['phases'][phase][m]['values'],sb['phases'][phase][m]['values'])
        holm(family)
        matches=[matched_trips(runs[ca,s]['phase_analysis'][va]['trips'],runs[cb,s]['phase_analysis'][vb]['trips']) for s in args.seeds]
        matched={m:paired([r['metrics'][m]['a'] for r in matches],[r['metrics'][m]['b'] for r in matches]) for m in matches[0]['metrics']}
        holm(matched)
        comparisons[name]=dict(exploratory_family=family,matched_complete_trips=matched,matched_counts=[r['n'] for r in matches],
            matched_max_spawn_distance_error=max(r['maximum_spawn_distance_difference'] for r in matches))
    first_family={}
    for s in args.seeds:
        pa=runs['pgc33',s]['phase_analysis']['intact']['first_return_20s_rows']
        pb=runs['pgc33',s]['phase_analysis']['goal_grid']['first_return_20s_rows']
        assert [(r['episode'],r['slot'],r['first_goal_step'],r['start_distance']) for r in pa]==[(r['episode'],r['slot'],r['first_goal_step'],r['start_distance']) for r in pb]
        for rr in pa+pb:
            assert abs(rr['capped_return_seconds']-rr['before_near_seconds']-rr['after_first_near_seconds'])<1e-10
    for m in ['capped_return_seconds','before_near_seconds','after_first_near_seconds','success_within_20s']:
        first_family[m]=paired(estimates['pgc33/intact']['first_return_20s'][m]['values'],
                               estimates['pgc33/goal_grid']['first_return_20s'][m]['values'])
    holm(first_family)
    comparisons['first_return_goal_deshift_minus_intact']=dict(exploratory_family=first_family,
        eligible_counts=[len(runs['pgc33',s]['phase_analysis']['intact']['first_return_20s_rows']) for s in args.seeds],
        identical_eligibility_first_goal_time_and_start_distance=True,
        scope='First return only; first discovery before decision1050 leaves300 decisions. Endpoint=min(actual return time,20seconds). All eligible episodes retained, including failed returns. Time components sum exactly to capped duration. Protocol addition made after first two descriptive phase seeds, before inspecting first-return endpoints.')
    result=dict(seeds=args.seeds,estimates=estimates,comparisons=comparisons,
        baseline_records_match_exactly=True,statistics='Paired training-seed contrasts; 12 exploratory comparisons per contrast family. Holm t and exact sign-flip separately. Five matched-complete-trip contrasts form separate exploratory families. Conditional successful-trip quantities are not unconditional policy effects.',
        limitations=['Native models visit different states; phase-conditioned changes need not identify the causal mediator.',
                     'Matched trip comparisons require both controllers to finish that ordinal return; censoring may select easier trips.',
                     'Near threshold .6m chosen before inspecting traces, twice actual goal radius .3m. All raw step exposures and failed/censored return attempts retained.'])
    strategies=dict(action_names=['left','right','forward','backward','strafe_left','strafe_right'],
        models=[dict(condition=c,seed=s,**runs[c,s]['phase_analysis']['intact']['action_strategy']) for c in ['gc100','pgc33'] for s in args.seeds],
        scope='Exploratory characterization of sampled native-policy actions after first discovery. Preferred turn and translation chosen from observed frequencies; action-index conventions follow code/env.py. Native-controller hybrids can change preferred movement axis or turn direction.')
    result['action_strategy']=strategies
    (OUT/'action_strategy.json').write_text(json.dumps(strategies,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol['completed_seeds']=args.seeds
    protocol['first_return_20s']='Exploratory addition after first two phase seeds; eligibility first_goal_step<1050, exact300-decision followup, capped20s return time; before/afterfirstnear components; four-comparison Holm family.'
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    figure(result);report(result)
    for key,e in estimates.items():
        print(key, 'score',round(e['mean_score']['mean'],2),'firstgoal',round(e['mean_first_goal_step']['mean'],1),
              'before/after near',*[round(e['trip_summary'][m]['mean'],3) for m in ['mean_complete_before_near_seconds','mean_complete_after_first_near_seconds']],
              'near exits',round(e['trip_summary']['fraction_near_attempts_with_exit']['mean'],4))

def figure(r):
    e=r['estimates'];names=['gc100/intact','pgc33/intact','pgc33/goal_grid'];labels=['Grid','PGC','PGC: ordinary goal']
    colors=['#3379B5','#D87532','#9BA2A8']
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for ax,m,title in zip(axes,['mean_complete_before_near_seconds','mean_complete_after_first_near_seconds','fraction_near_attempts_with_exit'],
                          ['Getting near the goal','Finishing the approach','Leaving the near-goal area']):
        mult=100 if 'fraction' in m else 1
        vals=[e[n]['trip_summary'][m]['mean']*mult for n in names]
        errors=[e[n]['trip_summary'][m]['sem']*mult for n in names]
        ax.bar(range(3),vals,color=colors,yerr=errors,capsize=3,width=.6)
        ax.set(xticks=range(3),xticklabels=['Grid','PGC','PGC\nordinary goal'],title=title,ylabel='Attempts (%)' if mult==100 else 'Seconds per completed return')
        ax.spines[['top','right']].set_visible(False)
        for i,(v,err) in enumerate(zip(vals,errors)):ax.text(i,v+err+.03*max(vals),f'{v:.2f}',ha='center',fontsize=10)
        ax.set_ylim(0,max(np.array(vals)+errors)*1.22)
    fig.supxlabel('Near = within 0.6 m; goal radius = 0.3 m. Mean ± SEM across training seeds.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'navigation_phases.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    names=['pgc33/intact','pgc33/goal_grid'];xx=np.arange(2)
    before=np.array([e[n]['first_return_20s']['before_near_seconds']['mean'] for n in names])
    after=np.array([e[n]['first_return_20s']['after_first_near_seconds']['mean'] for n in names])
    axes[0].bar(xx,before,color='#8B939B',width=.55,label='Before first near entry')
    axes[0].bar(xx,after,bottom=before,color='#D87532',width=.55,label='After first near entry')
    for i,n in enumerate(names):
        z=e[n]['first_return_20s']['capped_return_seconds'];axes[0].errorbar(i,z['mean'],yerr=z['sem'],color='#253442',capsize=3)
        axes[0].text(i,z['mean']+z['sem']+.15,f"{z['mean']:.2f}s",ha='center')
    axes[0].set(xticks=xx,xticklabels=['Predictive goal','Ordinary goal'],ylabel='First-return time, capped at 20 s',ylim=(0,max(before+after)*1.3),title='Same start; keep failed returns')
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    fam=r['comparisons']['first_return_goal_deshift_minus_intact']['exploratory_family']
    for i,(m,label) in enumerate([('capped_return_seconds','Total time'),('before_near_seconds','Before near'),('after_first_near_seconds','After near')]):
        z=fam[m];v=np.array(z['differences']);axes[1].scatter(v,np.full(len(v),i)+np.linspace(-.09,.09,len(v)),s=15,color='#D87532',alpha=.45)
        axes[1].errorbar(z['difference'],i,xerr=[[z['difference']-z['ci95'][0]],[z['ci95'][1]-z['difference']]],fmt='o',color='#D87532',capsize=3)
    axes[1].axvline(0,color='#8B939B',lw=1)
    axes[1].set(yticks=range(3),yticklabels=['Total time','Before near','After near'],ylim=(2.5,-.5),xlabel='Ordinary − predictive goal (seconds)',title='Paired seed differences')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Identical first discovery and spawn · eligibility before intervention · mean ± SEM / paired 95% intervals',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'first_return_20s.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def report(r):
    e=r['estimates'];g=e['gc100/intact'];p=e['pgc33/intact'];d=e['pgc33/goal_grid']
    lines=['# Where the navigation advantage appears','',
           '**The PGC advantage appears after goal discovery: better return navigation, despite no earlier discovery on this evaluation set.**','',
           '![Return navigation phases](navigation_phases.png)','',
           '| Measure | Grid | PGC | PGC with ordinary goal |','|---|---:|---:|---:|']
    for label,m in [('Score','mean_score'),('First goal step (failure = 1350)','mean_first_goal_step'),('Goal discovery fraction','frac_found_goal'),('Return path efficiency','mean_path_eff'),('Goals / 1000 post-discovery steps','post_discovery_goal_rate')]:
        lines.append('| '+label+' | '+' | '.join(f"{x[m]['mean']:.3f}" for x in [g,p,d])+' |')
    lines+=['',f"Score separates exactly into first-discovery reward and later return rewards. PGC minus grid is **{p['first_discovery_score']['mean']-g['first_discovery_score']['mean']:+.3f} points** from first discovery and **{p['return_score']['mean']-g['return_score']['mean']:+.3f} points** from subsequent returns. This accounting is descriptive, not a mediation percentage.",'',
        '## What the phases measure','',
        'Getting near ends at the first entry within 0.6 m of the true goal center. Finishing the approach includes all subsequent time until goal pickup, including excursions back outside that area. The actual pickup radius is 0.3 m. An attempt that starts inside 0.6 m has zero getting-near time. Durations in the figure condition on completed returns; the exit fraction includes all attempts that reached the near area, including censored attempts.','',
        'The original input-swap intervention changes only goal encoding. Its first-discovery behavior is identical to intact. Native grid versus PGC comparisons also involve different learned policies and visited states, so their phase differences alone do not establish a causal mechanism.','',
        'Matched successful-return comparisons pair the same episode, environment slot, and return number. Their spawn distances match exactly (within numerical tolerance), controlling initial trip difficulty. Requiring both trips to finish can still select easier cases. Full step exposures and censored returns are retained.','',
        '| Matched completed trips | Total time change | Before near | After near |',
        '|---|---:|---:|---:|']
    for contrast,label in [('native_pgc_minus_grid','PGC − grid'),('goal_deshift_minus_intact','Ordinary goal − intact PGC')]:
        z=r['comparisons'][contrast]['matched_complete_trips']
        lines.append('| '+label+' | '+' | '.join(f"{z[m]['difference']:+.3f} s" for m in ['seconds','before_near_seconds','after_first_near_seconds'])+' |')
    lines+=['','These differences concern matched trips that **both** controllers completed. They do not give a percentage of the score advantage explained. Full paired intervals and five-comparison corrections are in the numerical summary.','',
        '## First return: a comparison that retains failures','',
        '![First return with predictive versus ordinary goal code](first_return_20s.png)','',
        'For goal-only substitution, the first goal discovery, first return spawn, and recurrent state before the changed goal input are identical. We select episodes whose first discovery leaves at least 20 seconds, then measure first-return time capped at 20 seconds. Failed returns remain in this endpoint. The episode-specific goal snapshot does not refresh until that first return succeeds.','',
        '| First-return measure | PGC intact | Ordinary goal | Change | 95% paired interval | Holm t p | Holm exact p |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,label in [('capped_return_seconds','Time capped at20s'),('before_near_seconds','Before first near entry'),('after_first_near_seconds','After first near entry'),('success_within_20s','Success within20s')]:
        z=r['comparisons']['first_return_goal_deshift_minus_intact']['exploratory_family'][m]
        lines.append(f"| {label} | {p['first_return_20s'][m]['mean']:.3f} | {d['first_return_20s'][m]['mean']:.3f} | {z['difference']:+.3f} | [{z['ci95'][0]:+.3f}, {z['ci95'][1]:+.3f}] | {z['holm_t_p']:.4g} | {z['holm_exact_p']:.4g} |")
    lines+=['','The mean capped first-return time increases by 0.33 seconds after removing predictive goal input. That comparison remains uncertain after correction for the four tested endpoints, and the phase components are also uncertain. It agrees in direction with the longer evaluation but does not independently establish concentration of the effect near the goal.','',
        'The before/after components sum exactly to capped time. The first-return analysis was added after the first two descriptive phase seeds, before inspecting these endpoints; it is exploratory. Its four comparisons form a separate Holm family. The native grid-versus-PGC first-discovery histories differ, so this matched causal comparison applies to the goal-only intervention within PGC-trained models.','',
        '## A mostly turn-or-move policy','',
        'Across the full post-discovery traces, **15 of 16 native models use one turn direction and one translation action for at least 99.86% of sampled actions**. Translation can mean forward, backward, or sideways; the preferred axis differs across models. PGC seed5 is a genuine three-action exception: left turn28.0%, strafe left58.8%, backward13.2%. Thus a lower forward-action probability alone would not imply worse navigation.','',
        'Only seed pairs1 and5 share their preferred turn and movement axes, and PGC seed5 additionally uses backward movement. Switching between grid and PGC controllers can therefore incur a change-of-strategy cost. These action-frequency descriptions were added after examining the completed behavior traces; they are exploratory. [Full action counts](action_strategy.json).','',
        '## Validation and reproduction','',
        f"Every per-episode record reproduces the original code-swap evaluation exactly, for every instrumented variant. {len(r['seeds'])} training seeds are the statistical replicates. All comparisons here are exploratory; the numerical summary retains paired intervals and both Holm-corrected t and exact sign-flip tests.",'',
        '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_navigation_phases.py --device mps','python scripts/summarize_lstm_navigation_phases.py --refresh-from-trace','```','',
        'The rollout also saves local native-policy snapshots before the current LSTM input, for separate matched-state action analysis. No model is retrained. Raw NumPy arrays and checkpoint weights are local artifacts. [Numerical results](summary.json) · [Protocol](protocol.json)']
    text='\n'.join(lines)+'\n'
    radius_section=OUT/'radius_sensitivity.md'
    if radius_section.exists():
        text=text.replace('## A mostly turn-or-move policy',radius_section.read_text()+'\n## A mostly turn-or-move policy',1)
    for old,new in [('at20s','at 20 s'),('within20s','within 20 s'),('seed5','seed 5'),('turn28.0%','turn 28.0%'),
                    ('left58.8%','left 58.8%'),('backward13.2%','backward 13.2%'),('pairs1 and5','pairs 1 and 5')]:
        text=text.replace(old,new)
    (OUT/'README.md').write_text(text)

if __name__=='__main__':main()
