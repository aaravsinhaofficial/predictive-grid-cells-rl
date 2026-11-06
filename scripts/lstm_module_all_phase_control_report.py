"""Summarize the local known-eight-phase control without claiming global topology."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_module_all_phase_control'


def median(values):
    values=[v for v in values if v is not None]
    return float(np.median(values)) if values else None


def qualified(row,direction):
    c=row['comparison'];d=c['per_direction'][direction]
    return (c['observable_radius_RMS']>=1e-4 and d['eight_phase_initial_visible_fraction']>=.8
            and d['coefficient_locality_pass_fraction']>=.8 and d['eight_phase_normal_gain_median'] is not None
            and d['initially_silent_later_total_visible_fraction']==0)


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['cases']==768 and len(summary['results'])==4
    rows=[r for m in summary['results'] for r in m['records']]
    table=[];directions=[]
    for condition in ('gc100','pgc33'):
        for observable in ('full_state','all_hidden_future','all_actor_scores_future'):
            selected=[r for r in rows if r['condition']==condition and r['observable']==observable and r['horizon_updates']==30]
            values=[r['comparison']['per_direction'][6] for r in selected]
            accepted=[r['comparison']['per_direction'][6] for r in selected if qualified(r,6)]
            table.append({'condition':condition,'observable':observable,'cases':len(selected),
                'cell_centroid_gain_two_phase_all_median':median([d['two_phase_normal_gain_median'] for d in values]),
                'cell_centroid_gain_eight_phase_all_median':median([d['eight_phase_normal_gain_median'] for d in values]),
                'additional_phase_explained_fraction_all_median':median([d['additional_phase_explained_fraction_median'] for d in values]),
                'qualified_cases':len(accepted),
                'qualified_eight_phase_gain_median':median([d['eight_phase_normal_gain_median'] for d in accepted]),
                'qualified_explained_fraction_median':median([d['additional_phase_explained_fraction_median'] for d in accepted]),
                'median_locality_pass_fraction':median([d['coefficient_locality_pass_fraction'] for d in values]),
                'absolute_observable_amplitude_passes':sum(r['comparison']['observable_radius_RMS']>=1e-4 for r in selected)})
            if observable=='full_state':
                for k,label in [(4,'Activity-only push (h)'),(6,'Stored-pattern strength (c)'),(5,'Random memory push (c)')]:
                    accepted=[r['comparison']['per_direction'][k] for r in selected if qualified(r,k)]
                    directions.append({'condition':condition,'direction':k,'label':label,'qualified_cases':len(accepted),
                        'normal_gain_eight_phase_median':median([d['eight_phase_normal_gain_median'] for d in accepted])})
    verification={'maximum_elicitation_FD_relative_error':max(max(m['elicitation_FD_relative_errors']) for m in summary['results']),
        'maximum_parent_state_difference':max(r['state_max_abs_error'] for m in summary['results'] for r in m['parent_replay_agreement']),
        'maximum_parent_selected_phase_JVP_difference':max(r['selected_phase_JVP_max_abs_error'] for m in summary['results'] for r in m['parent_replay_agreement'])}
    (OUT/'aggregate.json').write_text(json.dumps({'table':table,'full_state_direction_comparison':directions,'verification':verification},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    for j,condition in enumerate(('gc100','pgc33')):
        values=[r for r in directions if r['condition']==condition]
        x=np.arange(3)+(j-.5)*.32
        ax.bar(x,[100*r['normal_gain_eight_phase_median'] for r in values],width=.3,
               color='#2563a6' if j==0 else '#e16b35',label='Grid' if j==0 else 'PGC + grid')
    ax.axhline(100,color='.5',lw=.8);ax.axhline(50,color='black',ls='--',lw=1)
    ax.set(xticks=range(3),xticklabels=['Change in activity\n(h)','Change in memory strength\n(c)','Random memory error\n(c)'],
           ylabel='Local error remaining after 2 s (%)',title='Activity settles faster than memory errors',ylim=(0,225))
    ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'all_phase_memory_control.png',dpi=200);fig.savefig(OUT/'all_phase_memory_control.pdf')
    print(json.dumps({'table':table,'directions':directions,'verification':verification}),flush=True)


if __name__=='__main__':main()
