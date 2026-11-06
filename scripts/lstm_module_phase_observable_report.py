"""Show every geometry-qualified module-phase output probe, including failures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_module_phase_observable'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    rows=[r for m in summary['results'] for r in m['rows']]
    promoted=[r for r in rows if r['full_recovery_tested']]
    positives=[r for r in rows if r['centered_candidate'] or r['noncentered_candidate']]
    counts=[{'condition':m['condition'],'seed':m['seed'],'screens':len(m['rows']),
             'promoted':sum(r['full_recovery_tested'] for r in m['rows']),
             'centered':sum(r['centered_candidate'] for r in m['rows']),
             'noncentered':sum(r['noncentered_candidate'] for r in m['rows'])} for m in summary['results']]
    concise=[]
    for row in positives:
        r={k:row[k] for k in ('condition','seed','module','axis','regime','anchor_updates','metric','observable','centered_candidate','noncentered_candidate')}
        probe=row['full_probe']
        r.update(radial_gain=probe['radial_observable_normal_gain']['median'],
                 tangent_gain=probe['tangent_observable_total_gain']['median'],
                 radial_over_tangent=probe['radial_over_tangent_observable_gain']['median'],
                 actor_probability_family_RMS=probe['actor']['reference']['family_probability_RMS'])
        concise.append(r)
    (OUT/'aggregate.json').write_text(json.dumps({'counts':counts,'candidates':concise},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained')
    groups=[('gc100','all_hidden'),('pgc33','all_hidden'),('gc100','all_centered_actor_scores'),('pgc33','all_centered_actor_scores')]
    for i,(condition,observable) in enumerate(groups):
        subset=[r for r in promoted if (r['condition'],r['observable'])==(condition,observable)]
        values=np.array([r['full_probe']['radial_over_tangent_observable_gain']['median'] for r in subset])
        if len(values):
            jitter=np.linspace(-.16,.16,len(values))
            ax.scatter(i+jitter,values,s=22,alpha=.65,color='#2563a6' if condition=='gc100' else '#e16b35')
        else:ax.text(i,.9,'No qualifying\nshape',ha='center',fontsize=8)
    ax.axhline(.5,color='black',ls='--',lw=1)
    ax.axhline(1,color='.6',lw=.8)
    ax.set(xticks=range(4),xticklabels=['Grid\nhidden activity','PGC + grid\nhidden activity','Grid\naction scores','PGC + grid\naction scores'],
           ylabel='Off-curve / along-curve retention after 2 s',title='Does the network correct off-curve errors faster?',yscale='log')
    ax.set_yticks([.1,.2,.5,1.]);ax.set_yticklabels(['0.1','0.2','0.5','1.0'])
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'module_phase_observable_recovery.png',dpi=200)
    fig.savefig(OUT/'module_phase_observable_recovery.pdf')
    print(json.dumps({'counts':counts,'candidate_rows':len(concise)}),flush=True)


if __name__=='__main__':main()
