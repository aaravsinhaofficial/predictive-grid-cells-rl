"""Complete-cohort summary of the two-phase tangent/transverse experiment."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_module_phase_torus'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['cases']==768 and len(summary['results'])==4
    rows=[r for model in summary['results'] for r in model['records']]
    table=[]
    for condition in ('gc100','pgc33'):
        for observable in ('full_state','all_hidden_future','all_actor_scores_future'):
            values=[r for r in rows if r['condition']==condition and r['observable']==observable and r['horizon_updates']==30]
            normal_worst=[]
            for row in values:
                gains=[d['normal_gain_median_informative'] for d in row['recovery']['per_direction'] if d['normal_gain_median_informative'] is not None]
                normal_worst.append(max(gains) if gains else np.nan)
            table.append({'condition':condition,'observable':observable,'cases':len(values),
                'two_phase_gain_median':float(np.median([r['recovery']['minimum_phase_gain']['median'] for r in values])),
                'pooled_normal_gain_median':float(np.median([r['recovery']['normal_gain']['median'] for r in values])),
                'worst_normal_direction_median':float(np.nanmedian(normal_worst)),
                'rank2_fraction_median':float(np.median([r['recovery']['phase_rank2_fraction'] for r in values])),
                'all_normal_direction_passes':sum(r['recovery']['gates']['all_normal_directions_qualify_and_recover'] for r in values),
                'centered_geometry_passes':sum(all(r['geometry']['centering']['centered']['gates'].values()) for r in values),
                'centered_nominations':sum(r['candidates']['centered'] for r in values),
                'noncentered_nominations':sum(r['candidates']['noncentered'] for r in values)})
    cross=[]
    for row in rows:
        if row['observable']=='full_state' and row['horizon_updates']==30:
            for axis,value in enumerate(row['old_slice_radial_second_phase_fraction']):
                cross.append({k:row[k] for k in ('condition','seed','module','regime','anchor_updates','metric')}|
                             {'axis':axis,'fraction_median':value['median'],'fraction_p90':value['p90']})
    model_counts=[{'condition':m['condition'],'seed':m['seed'],'cases':len(m['records']),
                  'centered':sum(r['candidates']['centered'] for r in m['records']),
                  'noncentered':sum(r['candidates']['noncentered'] for r in m['records']),
                  'elicitation_FD_max_relative_error':max(r['relative_error'] for r in m['elicitation_FD_checks'])}
                 for m in summary['results']]
    (OUT/'aggregate.json').write_text(json.dumps({'table':table,'old_slice_second_phase':cross,'model_counts':model_counts},indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.6),layout='constrained')
    for i,condition in enumerate(('gc100','pgc33')):
        values=np.array([r['fraction_median'] for r in cross if r['condition']==condition])
        axes[0].scatter(i+np.linspace(-.16,.16,len(values)),100*values,s=15,alpha=.5,
                        color='#2563a6' if condition=='gc100' else '#e16b35')
    axes[0].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],ylim=(-2,102),
                ylabel='Old radial kick along the second phase (%)',title='One phase can hide the other')
    names={'full_state':'Full state','all_hidden_future':'Hidden outputs','all_actor_scores_future':'Action scores'}
    for i,observable in enumerate(names):
        for j,condition in enumerate(('gc100','pgc33')):
            row=next(r for r in table if (r['condition'],r['observable'])==(condition,observable))
            x=i*3+j
            axes[1].scatter(x,100*row['two_phase_gain_median'],marker='o',s=52,color='#2563a6' if j==0 else '#e16b35')
            axes[1].scatter(x,100*row['worst_normal_direction_median'],marker='x',s=55,color='#2563a6' if j==0 else '#e16b35')
    axes[1].axhline(100,color='.6',lw=.8)
    axes[1].axhline(50,color='black',ls='--',lw=1)
    axes[1].set(xticks=[.5,3.5,6.5],xticklabels=list(names.values()),ylabel='Displacement remaining after 2 s (%)',
                title='Phase retention and error recovery',ylim=(0,230))
    axes[1].scatter([],[],marker='o',color='black',label='Weaker phase direction')
    axes[1].scatter([],[],marker='x',color='black',label='Most persistent sampled error')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'module_phase_torus_screen.png',dpi=200)
    fig.savefig(OUT/'module_phase_torus_screen.pdf')
    print(json.dumps({'model_counts':model_counts,'table':table}),flush=True)


if __name__=='__main__':main()
