"""Compact descriptive display of every fixed40M observable branch."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_reorientation_observable'


def main():
    data=json.loads((OUT/'summary.json').read_text());assert data['all32_checks_complete']
    groups=[('gc100',0),('gc100',1),('pgc33',0),('pgc33',1)]
    cues=[('native','Native','#20252a'),('ordinary','Ordinary','#357eb0'),
          ('plus60','+60°','#d78430'),('minus60','−60°','#8c5ca3')]
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.5),layout='constrained')
    for index,(cue,label,color) in enumerate(cues):
        values=[next(r for r in data['rows'] if (r['condition'],r['context'])==group and r['cue']==cue and r['metric']=='raw_full_state_kicks') for group in groups]
        x=np.arange(4)+(index-1.5)*.11
        axes[0].scatter(x,[r['radial_over_tangent'] for r in values],label=label,color=color,s=32)
        axes[1].scatter(x,[r['centered_departure_p90'] for r in values],label=label,color=color,s=32)
    axes[0].axhline(.5,color='.55',ls='--',lw=1)
    axes[1].axhline(.05,color='.55',ls='--',lw=1)
    axes[0].set(title='Radial vs angular persistence',ylabel='Radial / angular change',ylim=(0,.85))
    axes[1].set(title='The activity family changes shape',ylabel='Departure from starting curve (radii)',ylim=(0,.85))
    axes[0].legend(frameon=False,ncol=2,fontsize=8,loc='upper right')
    for ax in axes:
        ax.set_xticks(range(4),['Grid\nContext 1','Grid\nContext 2','PGC + grid\nContext 1','PGC + grid\nContext 2'],fontsize=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13)
    fig.supxlabel('All four cues · raw full-state kicks · all 256 future hidden outputs',fontsize=9)
    fig.savefig(OUT/'observable_recovery.png',dpi=180);plt.close(fig)
    lines=['# Future activity after the task-native direction cue','',
        '**Some radial disturbances shrink faster than angular disturbances. The activity families change shape and no longer map consistently onto their earlier curve. None passes the unchanged conditional-ring criteria.**','',
        '![Radial recovery and preservation of the activity family](observable_recovery.png)','',
        'We observed every hidden unit at four future times, then repeated that measurement two seconds later. This tests for a stable ring in all hidden units’ future activity, allowing a common shift of the activity pattern. A circular latent variable expressed through a changing activity pattern remains compatible with these observations. No decoder plane or neuron subset was selected.','',
        'The four branches receive a native predictive cue, ordinary spatial cue, or predictive cue rotated by ±60° exactly once. Their subsequent inputs are identical. The analysis starts one second after that cue and finishes at the first free-action input, before the actor moves.','',
        '**Left:** values below 0.5 meet the selective radial-recovery part of the test. PGC values are similar with the native cue removed or rotated. **Right:** every branch changes substantially relative to its starting activity curve, even after subtracting common background drift; all exceed the 0.05-radius invariance threshold. Mapping later activity onto the earlier curve also fails the phase-order criterion. This measures their changing correspondence; direction remains readable. Both criteria must pass alongside the other frozen guards.','',
        '| Model / context | Cue | Radial/angular | Curve departure |',
        '|---|---|---:|---:|']
    for condition,context in groups:
        for cue,label,_ in cues:
            r=next(r for r in data['rows'] if (r['condition'],r['context'],r['cue'],r['metric'])==(condition,context,cue,'raw_full_state_kicks'))
            lines.append(f"| {'Grid' if condition=='gc100' else 'PGC + grid'} / {context+1} | {label} | {r['radial_over_tangent']:.3f} | {r['centered_departure_p90']:.3f} |")
    lines+=['',
        'The second Grid context also falls below the minimum observable-activity amplitude. Its small radial/angular ratio therefore accompanies a weak, changing trace. Exact Grid cue shams are retained in the figure/table; their repeated points are not independent observations.','',
        f"Across both contexts, four cue branches and both full-state perturbation metrics, **{data['centered_candidates']}/{data['branch_metric_checks']} centered and {data['noncentered_candidates']}/{data['branch_metric_checks']} noncentered checks pass**. These are nested checks of one original seed pair, not 32 trained agents. All hidden/cell RMS-scaled results and every gate are retained in [the numerical summary](summary.json). Saved15→60 trajectories replayed exactly before the probes. Actor probability differences are reported separately; this geometry analysis does not simulate free navigation.",'',
        '[Frozen protocol](protocol.json) · [Independent review and known-system controls](review.json) · [Interpretation audit](interpretation_review.json) · [Primary and causal results](../lstm_reorientation_mechanism/final) · [Adapter](../../scripts/lstm_endogenous_reorientation_observable.py)','',
        'Reproduce each fixed population with `OPENBLAS_NUM_THREADS=2 python scripts/lstm_endogenous_reorientation_observable.py --condition gc100` (then `pgc33 --require-complete`). Rebuild this figure with `python scripts/plot_reorientation_observable.py`. The validated `observable_probe` and its gates were reused unchanged.','']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
