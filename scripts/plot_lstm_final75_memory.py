"""Report all fixed75M primary memory contexts; no selection or new gates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results/lstm_final75_memory';rows=[];heading=[]
    groups=[('v2','gc100'),('v2','pgc33'),('v3','gc100'),('v3','pgc33')]
    for arm,condition in groups:
        for context in range(2):
            root=folder/arm/condition
            record=json.loads((root/'travel'/f'{condition}_s1'/f'context{context}.json').read_text())
            o=record['ordinary_release'];row=dict(arm=arm,condition=condition,context=context,
                candidate=o['passes_both_metric_candidate_screen'],checkpoint_sha256=record['checkpoint_sha256'],
                refitted_phase_mae_deg=o['trajectory']['60']['time_refit_information_only']['heldout_mae_deg'],
                fixed_phase_mae_deg=o['trajectory']['60']['fixed_travel_phase']['heldout_mae_deg'],
                actor_probability_family_rms=o['trajectory']['60']['actor_probability_family_rms'])
            for metric in ('raw','relative_rms'):
                q=o['invariance']['30'][metric]['horizons']['30']
                row[metric]=dict(radial_gain=q['radial_normal_gain']['median'],
                    radial_over_tangent=q['radial_over_tangent_gain']['median'],
                    drift_radii=q['residual_to_fixed_curve_over_radius']['p90'])
            rows.append(row)
            h=json.loads((root/'heading'/f'context{context}.json').read_text())
            heading.append(dict(arm=arm,condition=condition,context=context,
                candidates={key:value['passes_both_metric_candidate_screen'] for key,value in h['branches'].items()}))
    summary=dict(travel_contexts=rows,heading_contexts=heading,
        trained_seed_scope='Same original seed1 pair, independently continued with/without state noise; contexts are nested repeated measurements, not independent trained seeds',
        travel_candidates=sum(r['candidate'] for r in rows),heading_candidates=sum(any(h['candidates'].values()) for h in heading))
    (folder/'primary_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2));plt.rcParams.update({'font.size':10})
    for j,(arm,condition) in enumerate(groups):
        group=[r for r in rows if (r['arm'],r['condition'])==(arm,condition)]
        color='#347bb5' if condition=='gc100' else '#d86b2a'
        values=[[r['refitted_phase_mae_deg'] for r in group],[r['raw']['radial_over_tangent'] for r in group],[r['raw']['drift_radii'] for r in group]]
        for ax,v in zip(axes,values):
            ax.plot([j,j],[min(v),max(v)],color=color,alpha=.5,lw=2)
            ax.scatter([j-.035,j+.035],v,color=color,s=38,zorder=3,marker='o' if arm=='v2' else 's')
    labels=['Grid\nNo noise','PGC + grid\nNo noise','Grid\nNoise','PGC + grid\nNoise']
    for ax in axes:
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_xlim(-.5,3.5);ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0].set(title='Direction remains readable',ylabel='Refitted direction error at 4 s (degrees)')
    axes[1].set(title='Radial errors persist',ylabel='Radial / tangent gain (2–4 s)');axes[1].axhline(.5,color='.5',ls='--',lw=1);axes[1].axhline(1,color='.75',lw=1)
    axes[2].set(title='The full activity curve drifts',ylabel='Distance from 2 s curve at 4 s (radii)');axes[2].axhline(.05,color='.5',ls='--',lw=1)
    fig.supxlabel('Fixed final checkpoints · two contexts per model · one original training-seed pair',fontsize=10)
    fig.tight_layout();fig.savefig(folder/'primary_memory.png',dpi=180);fig.savefig(folder/'primary_memory.pdf');plt.close(fig)
    text=['# Fixed 75M continuations: primary memory results','',
        '**None of the eight heading contexts or eight travel contexts passes the unchanged full-state ring criteria.** All fixed final models are retained: the seed-1 Grid/PGC pair continued with or without activity noise. No intermediate checkpoint was selected.','',
        '![All fixed final travel contexts](primary_memory.png)','',
        'Direction can still be recovered with an analysis readout fitted at each time. Radial disturbances persist about as strongly as angular differences, while the full-state family moves away from its earlier curve. The added training, including the chosen activity-noise condition, preserves a changing direction trace in these tests.','',
        '| Training | Population | Context | Refitted error at 4 s | Radial/tangent gain | Curve drift (radii) |','|---|---|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {'Noiseless' if r['arm']=='v2' else 'Activity noise'} | {'Grid' if r['condition']=='gc100' else 'PGC + grid'} | {r['context']} | {r['refitted_phase_mae_deg']:.2f}° | {r['raw']['radial_over_tangent']:.3f} | {r['raw']['drift_radii']:.2f} |")
    text.extend(['','The figure/table use raw full-state distances. The [complete numeric summary](primary_summary.json) retains the separate hidden/cell RMS-scaled results and both heading-input conditions. Lines show the two contexts of one trained model; they are not confidence intervals. The refitted readout measures accessible information and is not a ring criterion.','',
        'Heading tests retain native-current and ordinary-current branches. Travel tests remove only the current predictive component after a real forward approach; saved-goal values stay native. Both use fresh histories from the final weights under their trained blackout schedule, and primary recovery ends within four seconds. Independent future-hidden-activity, cue-pulse and finite-time spectral analyses are reported separately.','',
        'The separate [single-cue repair test](../lstm_heading_memory/travel_cue_pulse/final75/README.md) also retains all eight travel contexts. Across the four PGC contexts and both distance metrics, a correct predictive pulse changes the late phase error by at most 0.14°; all Grid sham branches are exact identities. The [simple pulse figure](../lstm_heading_memory/travel_cue_pulse/final75/cue_pulse_by_training_arm.png) distinguishes the immediate response from the much smaller persistent change. This measures acute cue repair in the same frozen network, separately from autonomous radial recovery.','',
        'The independent [future-hidden-activity analysis](../lstm_endogenous_ring/observable_history/final75/summary.json) finds zero candidates across all 48 branch/metric checks, including the version that subtracts shared background drift. The [full-state finite-time spectra](../lstm_final75_finite_spectra/summary.json) show multiple slow cell-state directions and strong temporary amplification outside the angular direction. Together these favor a persistent, changing memory trace over selective restoration toward a circular state. These checks reuse the same four trained models; they are not independent seed replications.','',
        'The noiseless worker adds a verified runtime metadata field to checkpoint provenance. The [separate metadata adapter](runtime_adapter_protocol.json) checks its protocol hash/backend/condition before calling the original validator; the frozen numerical wrapper and gates are unchanged. Initial metadata-only attempts ended before context collection. Noise-arm checkpoints use the original wrapper directly.','',
        '[Frozen primary protocol](protocol.json) · [Original numerical wrapper](../../scripts/lstm_final75_memory.py) · [Runtime-only adapter](../../scripts/lstm_final75_memory_runtime_adapter.py)','',
        'These are one original training-seed pair and two nested contexts per continuation. The new blind-reorientation training task and its prospective assay remain a separate experiment.'])
    assert summary['travel_candidates']==summary['heading_candidates']==0,'Rewrite descriptive result if any candidate passes'
    (folder/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
