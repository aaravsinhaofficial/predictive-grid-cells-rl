"""Minimal plots and complete pilot summary of the frozen cue-pulse assay."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);args=ap.parse_args()
    records=[json.loads(p.read_text()) for p in sorted(args.folder.rglob('result.json'))]
    if not records:raise ValueError('No results')
    pgc=[r for r in records if r['audit']['condition']=='pgc33']
    models={}
    for r in pgc:models.setdefault(r['audit']['checkpoint_sha256'],[]).append(r)
    times=np.array([0,1,5,15,30,45]);seconds=times/15
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,metric,title in zip(axes,('raw','relative_rms'),('Raw state distances','Hidden and cell blocks scaled')):
        for name,color,label in [('ordinary','#444444','Ordinary current'),('correct','#D55E00','One correct PGC pulse'),('wrong','#7B61A8','One rotated PGC pulse')]:
            curves=[]
            for contexts in models.values():
                context_curves=[]
                for record in contexts:
                    rows=[record['contrasts'][str(t)][metric] for t in times]
                    if name=='ordinary':values=[r['ordinary_phase_error_deg'] for r in rows]
                    elif name=='correct':values=[r['single_correct_phase_error_deg'] for r in rows]
                    else:values=[(r['single_wrong_plus_phase_error_deg']+r['single_wrong_minus_phase_error_deg'])/2 for r in rows]
                    context_curves.append(values)
                values=np.mean(context_curves,axis=0)
                curves.append(values);ax.plot(seconds,values,color=color,alpha=.25,lw=1)
            ax.plot(seconds,np.mean(curves,axis=0),color=color,lw=2,label=label,marker='o',ms=3)
        ax.set(title=title,xlabel='Seconds after state substitution',ylim=(0,45),xlim=(-.05,3.05))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Closest-curve direction error (degrees)')
    axes[1].legend(frameon=False,fontsize=9,loc='lower right')
    fig.tight_layout();fig.savefig(args.folder/'cue_pulse.png',dpi=180);fig.savefig(args.folder/'cue_pulse.pdf');plt.close(fig)
    compact=[]
    for r in records:
        compact.append(dict(audit=r['audit'],immediate=r['contrasts']['1'],three_seconds_later=r['contrasts']['45']))
    (args.folder/'summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    max_late=max(abs(r['contrasts']['45'][m]['single_correct_improvement_deg']) for r in pgc for m in ('raw','relative_rms'))
    max_actor=max(r['trajectory'][str(t)][b]['edits'][e]['paired_actor_total_variation']['mean']
                  for r in records for t in times for b in ('ordinary','correct_pulse') for e in ('plus30','minus30'))
    text=("# A brief predictive cue after a direction-memory disturbance\n\n"
          "![Cue-pulse response](cue_pulse.png)\n\n"
          "Each thin line is a trained PGC model, averaging its contexts when there are two; thick lines average models. Both signs of the 30° state substitution and the ±60° wrong-cue rotations are retained. The two panels show different state-distance metrics. Grid-model cue branches are an exact identity control. Sampled angles and contexts are not independent trained-model repetitions.\n\n"
          "We first let the original renderer, spatial maps and LSTM process actual forward travel from different directions to the same location. After an explicit stop, the network receives ordinary current grid code and black RGB. At 1 second we replace its full recurrent state with the naturally elicited state for a direction 30° away. A predictive cue is restored for exactly one update (1/15 second), or rotated deliberately; then ordinary code resumes. The saved goal and all other input slots stay unchanged.\n\n"
          "The angle is measured by the nearest point on the full 512-coordinate reference curve, not by a PCA projection. Correct-pulse trajectories supply the reference for both correct and wrong cues. All results also include off-curve distance, reference geometry, phase coverage, matched-branch errors and original actor probabilities. A curve angle is only interpretable together with those checks.\n\n")
    text+=(f"Across {len(models)} PGC models and {len(pgc)} contexts, the largest absolute change in the 30° discrepancy three seconds after a correct pulse is **{max_late:.3f}°**, considering both distance metrics. ")
    if max_late<1:
        text+=("The substituted direction largely remains. Any larger immediate shift in the block-scaled metric should be read alongside the off-curve distances: a brief input response can move the state toward the reference without durably repairing its memory. ")
    text+=(f"The largest mean action-probability change caused by the phase substitution across the ordinary/correct-pulse branches is {max_actor:.6f}.\n\n"
           "The separate radial-recovery and observable-history assays characterize the geometry of the trace. Persistent cue influence alone could also occur in a non-attracting memory system.\n\n")
    text+=("[Protocol](../protocol.json) · [All pilot contrasts](summary.json). Individual `result.json` files retain every branch, both distance metrics and input/checkpoint hashes. `pulse_states.npz` retains full trajectories locally. Reproduce with `scripts/lstm_travel_cue_pulse.py` and regenerate this figure with `scripts/plot_lstm_travel_cue_pulse.py`.\n")
    (args.folder/'README.md').write_text(text)


if __name__=='__main__':main()
