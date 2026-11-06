"""Summarize every fixed action-input case without selecting candidates."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lstm_previous_action_sweep import ACTIONS, MODELS, OUT, protocol
from lstm_endogenous_curve_recovery import sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    design=json.loads((OUT/'protocol.json').read_text())
    assert design==protocol()
    rows=[]
    for condition,seed in MODELS:
        for action in ACTIONS:
            path=OUT/f'{condition}_s{seed}'/f'{action}.json'
            result=json.loads(path.read_text())
            assert result['source_sha256']==sha(ROOT/'scripts/lstm_previous_action_sweep.py')
            for metric in ('raw','relative_rms'):
                full=result['full_state'];q=full['invariance']['30'][metric]['horizons']['30']
                obs=result['observable'][metric]
                rows.append(dict(condition=condition,seed=seed,action=action,metric=metric,
                    full_state_candidate=full['passes_both_metric_candidate_screen'],
                    full_state_metric_candidate=all(full['candidate_gates'][metric].values()),
                    radial_over_tangent=q['radial_over_tangent_gain']['median'],
                    curve_departure_p90=q['residual_to_fixed_curve_over_radius']['p90'],
                    centered_observable_candidate=obs['centered_conditional_factor_candidate'],
                    noncentered_observable_candidate=obs['noncentered_observable_candidate'],
                    observable_radial_over_tangent=obs['radial_over_tangent_observable_gain']['median'],
                    observable_failed_centered_gates=[k for k,v in obs['candidate_gates']['centered'].items() if not v],
                    left_replay_error=result['left_identity_replay_max_error'],source_sha256=sha(path)))
    assert len(rows)==48
    summary=dict(protocol_sha256=sha(OUT/'protocol.json'),model_action_cases=24,
        both_metric_full_state_candidates=sum(r['full_state_candidate'] for r in rows if r['metric']=='raw'),
        centered_observable_candidates=sum(r['centered_observable_candidate'] for r in rows),
        noncentered_observable_candidates=sum(r['noncentered_observable_candidate'] for r in rows),
        max_left_replay_error=max(r['left_replay_error'] for r in rows if r['action']=='left'),rows=rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True,layout='constrained')
    for ax,metric,title in zip(axes,('raw','relative_rms'),('Original activity units','Hidden / cell scales equalized')):
        for condition,seed in MODELS:
            y=[next(r['radial_over_tangent'] for r in rows if
                (r['condition'],r['seed'],r['action'],r['metric'])==(condition,seed,a,metric)) for a in ACTIONS]
            ax.plot(range(6),y,color='#2878a4' if condition=='gc100' else '#d27927',
                ls='-' if seed==1 else '--',marker='o' if seed==1 else 's',ms=4,
                label=f'{"Grid" if condition=="gc100" else "PGC + grid"}, seed {seed}')
        ax.axhline(.5,color='.5',ls=':',lw=1)
        ax.set(title=title,xticks=range(6),xticklabels=['Left','Right','Forward','Back','Side L','Side R'],
               xlabel='Previous-action input',ylim=(0,max(1.25,max(r['radial_over_tangent'] for r in rows)*1.1)))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Radial / angular disturbance gain')
    axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    fig.suptitle('Changing the action input leaves radial errors persistent')
    fig.supxlabel('Same starting states and all other inputs · dotted line: selective-recovery threshold',fontsize=9)
    fig.savefig(OUT/'previous_action_recovery.png',dpi=180);plt.close(fig)
    lo=min(r['radial_over_tangent'] for r in rows if r['metric']=='raw')
    hi=max(r['radial_over_tangent'] for r in rows if r['metric']=='raw')
    text=f'''# Does the previous-action input switch on a ring?

**All 24 model/action cases fail the unchanged full-state ring criteria.** The future-hidden-activity check also finds no candidate in all 48 action/metric comparisons. Changing the previous-action input alone does not reveal a ring in these original pilot models.

![Recovery across all six action inputs](previous_action_recovery.png)

Each line is one original trained model. A ratio near one means radial errors survive about as strongly as changes around the direction curve. The raw-unit ratios range from **{lo:.3f} to {hi:.3f}**. The dotted threshold is one part of the frozen candidate test; closed geometry, preservation of direction order, coverage and invariance must also pass.

## Why this test?

A [2026 fly-navigation study](https://www.nature.com/articles/s41586-026-11144-9) describes a goal circuit whose persistent activity depends on behavioral state. Our earlier action-context test changed the approach movement but still released the network under a turn input. This new experiment tests the previous-action input directly.

We reused each original pilot model’s full 512-coordinate state family after the same physical approach. We then changed only the six previous-action entries to left, right, forward, backward, left strafe or right strafe. The starting states, ordinary current code, saved goal, black visual input and reward were identical across these six branches. Both populations and seeds 1–2 were included, with no new outcome-based selection.

This is a **clamped-input experiment on the RNN**. In the translational conditions, location is held fixed while the action message says “move.” These branches measure input-dependent neural dynamics; they are not simulated navigation trajectories.

The LEFT branches reproduce every archived trajectory exactly: maximum error **{summary['max_left_replay_error']:.1g}**. All criteria and the future-activity observation map are reused unchanged. Some future-activity comparisons suppress radial disturbances more strongly, while failing curve preservation and direction ordering. Every gate and case is retained in the [complete summary](summary.json).

[Frozen protocol](protocol.json) · [Independent review](review.json) · [Source](../../scripts/lstm_previous_action_sweep.py)
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()
