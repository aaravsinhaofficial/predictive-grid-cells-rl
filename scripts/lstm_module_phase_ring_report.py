"""Simple complete-cohort display for the frozen module-phase experiment."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_module_phase_ring'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    all_rows=[]
    for model in summary['results']:
        for family in model['families']:
            for anchor,record in family['anchors'].items():
                all_rows.append({'condition':model['condition'],'seed':model['seed'],
                    'module':family['module'],'axis':family['axis'],'regime':family['regime'],
                    'anchor':int(anchor),'record':record,
                    'hidden_retention':family['expression'][anchor]['hidden_radius']/max(family['expression']['0']['hidden_radius'],1e-30)})
    table=[]
    for condition in ('gc100','pgc33'):
        for regime in ('actual_goal_held','all_goal_withheld'):
            for anchor in (75,450,1500):
                rows=[r for r in all_rows if (r['condition'],r['regime'],r['anchor'])==(condition,regime,anchor)]
                measured=[r['record']['metrics']['relative_rms'] for r in rows]
                keys=measured[0]['gates']
                table.append({'condition':condition,'regime':regime,'anchor_seconds':anchor/15,
                    'cases':len(rows),'candidates':sum(r['record']['candidate'] for r in rows),
                    'full_radial_probes':sum(r['record']['radial_recovery_tested'] for r in rows),
                    'relative_state_radius_retention_median':float(np.median([r['radius_retention'] for r in measured])),
                    'raw_state_radius_retention_median':float(np.median([r['record']['metrics']['raw']['radius_retention'] for r in rows])),
                    'hidden_radius_retention_median':float(np.median([r['hidden_retention'] for r in rows])),
                    'fixed_curve_departure_p90_median':float(np.median([r['fixed_curve_departure_over_radius']['p90'] for r in measured])),
                    'phase_coverage_median':float(np.median([r['phase_transport']['circular_coverage'] for r in measured])),
                    'relative_metric_gate_pass_counts':{key:sum(r['gates'][key] for r in measured) for key in keys}})
    (OUT/'aggregate.json').write_text(json.dumps(table,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11,3.25),layout='constrained')
    colors={'gc100':'#2563a6','pgc33':'#e16b35'}
    names={'gc100':'Grid','pgc33':'PGC + grid'}
    for ax,regime,title in zip(axes[:2],('actual_goal_held','all_goal_withheld'),('Common recorded goal cue','Goal cue removed')):
        for condition in ('gc100','pgc33'):
            values=np.array([[r['record']['metrics']['relative_rms']['radius_retention'] for r in all_rows
                if (r['condition'],r['regime'],r['anchor'])==(condition,regime,anchor)] for anchor in (75,450,1500)])
            x=np.array([5,30,100])
            ax.plot(x,100*np.median(values,axis=1),marker='o',color=colors[condition],label=names[condition])
            ax.fill_between(x,100*values.min(1),100*values.max(1),alpha=.12,color=colors[condition])
        ax.set(title=title,xlabel='Time after input becomes common (s)',ylabel='State separation remaining (%)',xscale='log',xticks=[5,30,100])
        ax.set_xticklabels(['5','30','100']);ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False,fontsize=9)
    for i,condition in enumerate(('gc100','pgc33')):
        values=np.array([r['record']['metrics']['relative_rms']['fixed_curve_departure_over_radius']['p90'] for r in all_rows if r['condition']==condition and r['anchor']==450])
        jitter=np.linspace(-.12,.12,len(values))
        axes[2].scatter(i+jitter,np.maximum(values,1e-8),color=colors[condition],s=17,alpha=.6)
    axes[2].axhline(.05,color='black',ls='--',lw=1)
    axes[2].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],ylabel='5 s departure / current curve radius',
                title='Does the state family stay in place?',yscale='log')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'module_phase_screen.png',dpi=200)
    fig.savefig(OUT/'module_phase_screen.pdf')
    print(json.dumps({'candidates':summary['candidate_count'],'all_cases':len(all_rows),'aggregates':table}),flush=True)


if __name__=='__main__':main()
