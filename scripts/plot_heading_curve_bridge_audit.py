"""Descriptive full-cohort geometry report; no new confirmatory tests."""
import json
from pathlib import Path

import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path('results/ring_heading/geometry_audit')
    data = json.loads((root / 'curve_bridge_sensitivity_all_available.json').read_text())
    conditions = ['grid', 'predictive', 'grid_heading']
    names = ['Grid', 'PGC + grid', 'Explicit heading']
    colors = ['#337CB2', '#D87532', '#7A9E7E']
    rows = [data['model_summaries'][c]['models'] for c in conditions]
    assert all(len(r)==16 for r in rows)
    metrics = ['maximum_adjacent_jump_over_radius_across_contexts', 'minimum_slow_alignment_across_contexts']
    fig, axes = plt.subplots(1,3,figsize=(11.3,3.8),layout='constrained')
    for ax,key,title,ylabel in zip(axes[:2],metrics,
            ['Sharp hidden-state transitions', 'Weakest local alignment'],
            ['Largest adjacent gap / state radius', 'Slow mode–curve alignment']):
        for i,(rr,color) in enumerate(zip(rows,colors)):
            values=np.array([r[key] for r in rr]); xx=i+np.linspace(-.18,.18,16)
            ax.scatter(xx,values,color=color,s=24,alpha=.75)
            ax.plot([i-.25,i+.25],[np.median(values)]*2,color='black',lw=2)
        ax.set(title=title,ylabel=ylabel,xticks=range(3),xticklabels=names,ylim=(0,None))
        ax.tick_params(axis='x',labelsize=8)
    axes[1].set_ylim(0,1.05)
    ax=axes[2]
    for i,(rr,color) in enumerate(zip(rows,colors)):
        original=np.array([r['normal_10pct_distance_over_current_radius_step100_all'] for r in rr])*100
        restricted=np.array([r['normal_10pct_distance_over_current_radius_step100_without_long_interiors'] for r in rr])*100
        for a,b in zip(original,restricted):ax.plot([i-.15,i+.15],[a,b],color=color,alpha=.25,lw=.8)
        ax.scatter(np.full(16,i-.15),original,color=color,s=17,marker='o',label='Original' if i==0 else None)
        ax.scatter(np.full(16,i+.15),restricted,color=color,s=20,marker='x',label='Omit long bridges' if i==0 else None)
    ax.set(title='Recovery survives bridge removal',ylabel='Distance / current radius after 5 s (%)',
           xticks=range(3),xticklabels=names,ylim=(0,None))
    ax.tick_params(axis='x',labelsize=8);ax.legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('16 training seeds per condition · Descriptive audit · Worst of 3 positions for geometry; mean of 3 for recovery',fontsize=8.5)
    for ext in ('png','pdf','svg'):fig.savefig(root/f'geometry_bridge_audit.{ext}',dpi=200)
    plt.close(fig)
    lines=['# Full-state geometry: useful local memory, with sharp exceptions','',
        'The learned networks preserve direction, but some have very sharp changes in other hidden features. A circular decoded output alone does not describe that full geometry.','',
        '![Geometry audit](geometry_bridge_audit.png)','',
        'Each dot is one training seed. The first two panels use the worst of three tested positions after four initialization cues and two seconds with all inputs zero. Horizontal lines mark medians. The last panel begins with disturbances of 10% of state radius and measures recovery after five more seconds; it averages the three positions.','',
        '| Descriptive measure | Grid | PGC + grid | Explicit heading |','|---|---:|---:|---:|']
    numbers=[]
    for rr in rows:
        numbers.append({'gap_median':np.median([r[metrics[0]] for r in rr]),
            'gap_over1':sum(r[metrics[0]]>1 for r in rr),
            'alignment_median':np.median([r[metrics[1]] for r in rr]),
            'alignment_below_half':sum(r[metrics[1]]<.5 for r in rr),
            'residual_original':np.median([r['normal_10pct_distance_over_current_radius_step100_all'] for r in rr])*100,
            'residual_restricted':np.median([r['normal_10pct_distance_over_current_radius_step100_without_long_interiors'] for r in rr])*100})
    for key,label,fmt in [('gap_median','Median largest adjacent gap / radius','.3f'),
            ('gap_over1','Seeds with an adjacent gap larger than radius','d'),
            ('alignment_median','Median of each seed’s weakest slow-mode alignment','.3f'),
            ('alignment_below_half','Seeds with any slow-mode alignment below 0.5','d'),
            ('residual_original','Median 5 s residual distance / current radius, original (%)','.3f'),
            ('residual_restricted','Median 5 s residual, long bridge interiors omitted (%)','.3f')]:
        lines.append('| '+label+' | '+' | '.join(format(n[key],fmt) for n in numbers)+' |')
    lines.extend(['','The gap and alignment thresholds are descriptive summaries, not validated ring classifiers or exclusion criteria. All 16 seeds remain in the functional analyses.','',
        '**What the recovery check changes.** A sparse curve joins neighboring measured states with straight lines. A long line can cross hidden states the network never occupies. We repeat the distance measurement after omitting interiors of segments longer than five times the curve’s median segment, while retaining every measured state. The residual distance increases, but recovery remains strong. This sensitivity check does not establish that every omitted segment is physically invalid.','',
        '**What this suggests.** PGC models usually have smoother conditional direction-memory geometry and fewer severe alignment exceptions than Grid models in this sampled assay. The learned direction memory is still not a globally uniform smooth ring in every network. Local correction and the causal benefits of the direction input remain meaningful without that stronger claim.','',
        'A dense follow-up of seed 2001 resolves extremely narrow continuous transitions, largely in the readout-null dimensions. This is a separate exploratory investigation, not a new cohort selection rule. The 2 s initial curves are not necessarily the asymptotic attractors. Later cue-free evolution can sharpen additional transitions.','',
        'Methods: 128 initial directions, 3 positions, 16 seeds in each of 3 conditions; underlying state maps and weights unchanged. No new hypothesis test is introduced here. [Raw audit](curve_bridge_sensitivity_all_available.json) · [Audit code](../../../scripts/audit_heading_curve_bridges.py) · [Figure code](../../../scripts/plot_heading_curve_bridge_audit.py).'])
    (root/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
