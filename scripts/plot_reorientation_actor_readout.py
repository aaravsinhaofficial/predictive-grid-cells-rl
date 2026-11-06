"""Simple display of all frozen40M actor readout outcomes; no new analysis."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_reorientation_actor_readout'


def compact(r):
    runner=next(q for q in r['pairwise_margins'] if q['other_action']==r['runner_up_mean_logit_action'])
    return dict(population=r['condition'],context=r['context']+1,cue=r['cue'],seconds=r['time_seconds'],
        hidden_variation_in_actor_sensitive_directions_percent=100*r['actor_rowspace_hidden_variance_fraction'],
        centered_logit_variation_rms=r['centered_logit_family_rms'],
        mean_preferred_action=r['preferred_mean_logit_action'],preferred_action_margin=r['preferred_mean_logit_margin'],
        directional_margin_rms=runner['directional_margin_rms'],
        translation_probability_percent=100*r['probabilities']['total_translation_mean'],
        probability_variation_rms=r['probabilities']['family_rms'],
        fraction_angles_changing_preferred_action=r['probabilities']['fraction_phases_changing_mean_preferred_action'],
        directional_first_harmonic_fraction=r['direction_first_harmonic']['fraction_centered_logit_variance'])


def main():
    data=json.loads((OUT/'summary.json').read_text());assert data['complete'] and len(data['rows'])==64
    rows=[compact(r) for r in data['rows']]
    with (OUT/'all_conditions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[('gc100',1),('gc100',2),('pgc33',1),('pgc33',2)]
    cues=['native','ordinary','plus60','minus60']
    fig,axes=plt.subplots(1,3,figsize=(10.3,3.9),layout='constrained')
    for x,group in enumerate(groups):
        color='#357eb0' if group[0]=='gc100' else '#d66b2c'
        for j,cue in enumerate(cues):
            r=next(q for q in rows if (q['population'],q['context'])==group and q['seconds']==4 and q['cue']==cue)
            xpos=x+(j-1.5)*.10
            axes[0].scatter(xpos,r['centered_logit_variation_rms'],color=color,s=31)
            axes[1].scatter(xpos,r['preferred_action_margin'],color=color,marker='s',s=31)
            axes[1].scatter(xpos,r['directional_margin_rms'],edgecolors=color,facecolors='none',marker='o',s=31)
            axes[2].scatter(xpos,r['probability_variation_rms'],color=color,s=31)
    axes[0].set(title='Signal reaches action scores',ylabel='Direction-dependent score variation\n(RMS vector norm, logit units)',yscale='log')
    axes[1].set(title='The default action dominates',ylabel='Logit margin',ylim=(-.4,15.5))
    axes[1].scatter([],[],color='.35',marker='s',label='Mean advantage')
    axes[1].scatter([],[],edgecolors='.35',facecolors='none',marker='o',label='Directional variation')
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    axes[2].set(title='Choices barely vary',ylabel='Action-probability variation (RMS vector norm)',yscale='log')
    for ax in axes:
        ax.set_xticks(range(4),['Grid\nContext 1','Grid\nContext 2','PGC + grid\nContext 1','PGC + grid\nContext 2'],fontsize=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.supxlabel('At 4 s · every cue branch shown · all four times retained in the table',fontsize=9)
    fig.savefig(OUT/'actor_bottleneck.png',dpi=180);fig.savefig(OUT/'actor_bottleneck.pdf');plt.close(fig)
    simple,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    for row in data['rows']:
        if row['condition']=='pgc33' and row['time_updates']==60:
            assert row['preferred_mean_logit_action']=='turn_right'
            assert row['runner_up_mean_logit_action'] in ('forward','backward','strafe_left','strafe_right')
    features=[('preferred_action_margin',-.20,'#d66b2c','Default turn preference'),
              ('directional_margin_rms',.20,'#718290','Direction-dependent change')]
    for index,context in enumerate((1,2)):
        native_row=next(r for r in rows if r['population']=='pgc33' and
                        r['context']==context and r['cue']=='native' and r['seconds']==4)
        for key,offset,color,label in features:
            value=native_row[key]
            ax.bar(index+offset,value,width=.32,color=color,alpha=.8,
                   label=label if index==0 else None)
            for j,cue in enumerate(cues):
                row=next(r for r in rows if r['population']=='pgc33' and
                         r['context']==context and r['cue']==cue and r['seconds']==4)
                ax.scatter(index+offset+(j-1.5)*.055,row[key],s=18,color='.2',zorder=3)
            ax.text(index+offset,value+.35,f'{value:.2f}',ha='center',fontsize=10)
    ax.set(title='The turning preference overwhelms the direction signal',
           xticks=(0,1),xticklabels=('Context 1','Context 2'),
           ylabel='Action-score difference',ylim=(0,17.4))
    ax.legend(frameon=False,ncol=2,loc='upper center',fontsize=9)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    simple.supxlabel('PGC + grid · fixed 40M model · dots show every cue condition',fontsize=9)
    simple.savefig(OUT/'pgc_action_bottleneck_simple.png',dpi=180);plt.close(simple)
    native=[r for r in rows if r['cue']=='native' and r['seconds']==4]
    text=['# Why the direction trace barely changes actions','',
        '**The PGC direction signal reaches the action scores. A much larger average preference for turning right overwhelms it.** In these contexts, the main bottleneck is the nearly deterministic action readout.','',
        '![The turning preference overwhelms direction](pgc_action_bottleneck_simple.png)','',
        'The tall bars show the average right-turn score advantage over the best movement action. The short bars show how much direction changes that same score difference (RMS across angles). Bars use the native cue; dots retain every cue condition. [Detailed figure including Grid](actor_bottleneck.png).','',
        'The final actor is a simple weighted sum of the 256 hidden activities, followed by softmax, which converts six scores into action probabilities. We passed the saved activity through those exact learned weights. We changed no states, weights, inputs or actions and ran no new navigation. This targeted follow-up was frozen after observing the earlier turning behavior and before calculating these values.','',
        'Here, the average is across controlled directions within one context. It is not the episode average used in the earlier PCA analysis.','',
        '**1. The signal reaches the actor.** We measured how much direction-dependent hidden variation lies in directions that the action weights can read. In the PGC contexts at four seconds, this is 5.68% and 2.50%. The five-dimensional actor-sensitive space occupies 1.95% of the 256-dimensional hidden space; that geometric reference is not a significance test. In these contexts, most hidden variation lies outside that small space. The PGC signal nevertheless produces score variation of 1.369 and 0.294 logit units. It has not simply disappeared into an actor-null direction.','',
        '**2. A strong default wins.** The average right-turn score exceeds the best translation score by 12.55 and 13.79 logit units. Direction changes the relevant pairwise margins by only 1.825 and 0.184 units RMS. No tested angle changes the highest-scoring action. The large baseline comes mainly from the average hidden activity read through the actor weights, rather than the actor bias parameter alone.','',
        '**3. The probabilities barely move.** The native PGC cue yields total translation probabilities of about 0.00283% and 0.000117% at four seconds. Softmax is operating in a nearly saturated regime: meaningful score variation produces very small probability changes. This explains why retaining direction information—and even shifting its neural representation—scarcely changes the observed movements. Grid also develops a strong default in these contexts, but it prefers strafing right and its late directional score variation is much smaller.','',
        '| Model / context | Hidden variation in actor-sensitive directions | Score variation | Default-action margin | Translation probability |',
        '|---|---:|---:|---:|---:|']
    for r in native:
        text.append(f"| {'Grid' if r['population']=='gc100' else 'PGC + grid'} / {r['context']} | {r['hidden_variation_in_actor_sensitive_directions_percent']:.2f}% | {r['centered_logit_variation_rms']:.6g} | {r['preferred_action_margin']:.2f} | {r['translation_probability_percent']:.6g}% |")
    text+=['',
        'The figures show all four cue variants at the final four-second point. The table above uses the native cue. Score variation is the RMS length of the six-score change vector across angles: square each centered score change, sum across actions, average across angles, then take the square root. Probability variation uses the same vector convention. The simple bar figure instead compares the average and directional RMS of the same pairwise score difference. The [complete 64-row table](all_conditions.csv) retains both models, both contexts, all four cues, and 0/1/2/4 seconds; the [full numerical record](summary.json) also retains singular values, every pairwise margin, exact probabilities and the separate linearized softmax calculation. Across every row, the top-scoring action remains unchanged across angles. Grid cue identities are exact.','',
        'We subtract only a score shared by all six actions, which cannot affect softmax. The linear reconstruction and that invariance are checked exactly to numerical tolerance. A fixed cosine/sine projection describes angular structure in the scores; it is not a trained decoder or evidence of behavioral generalization. Actor-sensitive hidden variance and score variance are reported separately from probability variation.','',
        'These results identify a concrete readout bottleneck in the fixed 40M pilot. They support testing whether exploration training can make existing directional information useful to the actor. The already frozen entropy continuation addresses that question separately; this analysis neither changes its protocol nor predicts its result. The native contexts differ across models, and the repeated angles are not independent trained agents.','',
        '[Frozen follow-up protocol](protocol.json) · [Review and five synthetic tests](review.json) · [Independent numerical audit](independent_review.json) · [Read-only numerical source](../../scripts/lstm_reorientation_actor_readout.py) · [Earlier state and navigation interventions](../lstm_reorientation_mechanism/final/README.md)','',
        'Reproduce with `OPENBLAS_NUM_THREADS=2 python scripts/lstm_reorientation_actor_readout.py`; rebuild this display with `python scripts/plot_reorientation_actor_readout.py`.']
    assert all(r['fraction_angles_changing_preferred_action']==0 for r in rows),'Rewrite qualitative claim if any phase changes preferred action'
    (OUT/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
