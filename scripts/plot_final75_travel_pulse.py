"""Show both fixed75M training arms separately; contexts remain nested."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_heading_memory/travel_cue_pulse/final75'


def main():
    protocol=json.loads((OUT/'cohort_protocol.json').read_text())
    cases=[]
    for case in protocol['cases']:
        result=json.loads((ROOT/case['output']/'result.json').read_text())
        cases.append(dict(case=case,result=result))
    assert len(cases)==8
    times=[0,1,5,15,30,45];seconds=np.asarray(times)/15
    styles=[('ordinary_phase_error_deg','#555555','Ordinary input'),
            ('single_correct_phase_error_deg','#cf6f2e','One correct pulse'),
            ('wrong','#785ea6','One rotated pulse')]
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True,layout='constrained')
    rows=[]
    for row,arm in enumerate(('v2','v3')):
        records=[c['result'] for c in cases if c['case']['arm']==arm and c['case']['condition']=='pgc33']
        for col,metric in enumerate(('raw','relative_rms')):
            ax=axes[row,col]
            for key,color,label in styles:
                context_curves=[]
                for result in records:
                    values=[]
                    for t in times:
                        contrast=result['contrasts'][str(t)][metric]
                        values.append((contrast['single_wrong_plus_phase_error_deg']+
                                       contrast['single_wrong_minus_phase_error_deg'])/2 if key=='wrong' else contrast[key])
                    context_curves.append(values)
                    ax.plot(seconds,values,color=color,lw=1,alpha=.25)
                ax.plot(seconds,np.mean(context_curves,axis=0),color=color,lw=2,label=label)
            ax.set_ylim(0,45);ax.set_yticks([0,15,30,45]);ax.set_xticks([0,1,2,3])
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
            if row==0:ax.set_title('Raw state distances' if col==0 else 'Hidden/cell blocks scaled')
            if col==0:ax.set_ylabel(('Noiseless training' if arm=='v2' else 'Noise training')+'\nDirection error (°)')
            if row==1:ax.set_xlabel('Seconds after memory disturbance')
        rows.append(dict(arm=arm,contexts=2,late_correct_improvement_deg={
            m:[r['contrasts']['45'][m]['single_correct_improvement_deg'] for r in records]
            for m in ('raw','relative_rms')}))
    axes[0,1].legend(frameon=False,fontsize=8,loc='lower right')
    fig.suptitle('Does one predictive pulse repair a 30° memory error?')
    fig.savefig(OUT/'cue_pulse_by_training_arm.png',dpi=180);plt.close(fig)
    max_late=max(abs(x) for r in rows for values in r['late_correct_improvement_deg'].values() for x in values)
    summary=dict(arms=rows,maximum_absolute_late_effect_deg=max_late,
        grid_branches_exact_identities=all(c['result']['audit']['grid_identity'] for c in cases if c['case']['condition']=='gc100'),
        inference='Both arms start from the same original seed1 PGC model and same25M parent. Two nested contexts each. No independent-seed significance test; all8prespecified cases retained.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# A brief cue after longer training

**The substituted direction largely remains in both training arms.** Three seconds after the correct pulse, its largest effect on the 30° discrepancy is {max_late:.3f}°, across both PGC models, all four contexts and both distance metrics.

![Cue repair in both training arms](cue_pulse_by_training_arm.png)

We displaced the complete LSTM state to the state elicited by movement 30° away, then supplied one correct predictive pulse, an ordinary-grid pulse, or a deliberately rotated predictive pulse. The cue lasts one update. The saved goal and every other input stay fixed. The plot averages the two signs of the state displacement and, for the wrong cue, both rotation signs.

Rows keep noiseless and noise-trained models separate. Each thick line averages two contexts of one model; thin lines show those contexts. Both arms descend from the same original seed-1 checkpoint. Grid cue branches are exact identity controls. All eight prespecified model/context cases ran successfully.

The angle uses the nearest point on the full512-state curve. Every result also retains distance off that curve, phase coverage, matched-branch state errors and actor effects. The companion geometry/recovery assays determine whether the family behaves as an attracting ring; the short pulse response is a separate measurement.

These late pulses refer to the stored actual travel direction after movement has stopped. The new blind-reorientation task separately tests a pulse at the actual movement endpoint, before the remembered direction is used for a blind turn.

[Frozen cohort](cohort_protocol.json) · [All numerical summaries](summary.json) · [Original pulse method](../protocol.json). Individual result files retain every condition, both metrics and exact checkpoint/input hashes. Recreate this figure with `python scripts/plot_final75_travel_pulse.py`.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps(summary))


if __name__=='__main__':main()
