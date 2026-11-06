"""Compact descriptive figure for every fixed75M travel context."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lstm_final75_finite_spectra'


def main():
    spectra=json.loads((OUT/'summary.json').read_text())
    observable=json.loads((ROOT/'results/lstm_endogenous_ring/observable_history/final75/summary.json').read_text())
    models=[('v2','gc100'),('v2','pgc33'),('v3','gc100'),('v3','pgc33')]
    labels=['Grid\nNo noise','PGC + grid\nNo noise','Grid\nNoise','PGC + grid\nNoise']
    colors=['#2878a4','#d27927','#2878a4','#d27927']
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    table=[];slow_modes=[];cell_only=[]
    for x,(arm,condition) in enumerate(models):
        s=sorted([r for r in spectra['rows'] if r['arm']==arm and r['condition']==condition and r['metric']=='raw_full_state'],key=lambda r:r['context'])
        o=sorted([r for r in observable['rows'] if r['arm']==arm and r['condition']==condition and r['family']=='travel' and r['metric']=='raw_full_state_kicks'],key=lambda r:r['context'])
        assert len(s)==len(o)==2
        for j,(sr,ob) in enumerate(zip(s,o)):
            xpos=x+(-.06 if j==0 else .06)
            axes[0].scatter(xpos,sr['slow_abs_eigenvalue_ge_099_count']['median'],color=colors[x],s=38,zorder=3)
            axes[1].scatter(xpos,ob['radial_over_tangent_median'],color=colors[x],s=38,zorder=3)
            for offset,key,color,marker in [(-.24,'tangent_total_gain','#48535c','o'),
                    (0,'radial_normal_gain_median','#2878a4','s'),(.24,'maximum_transverse_gain','#a4526b','^')]:
                value=sr[key]['median'] if isinstance(sr[key],dict) else sr[key]
                axes[2].scatter(x+offset+(-.025 if j==0 else .025),value,color=color,marker=marker,s=28,zorder=3)
            a=np.load(OUT/arm/condition/f'context{sr["context"]}.npz')
            slow=np.abs(a['raw_full_state__eigenvalues'])>=.99
            silent=a['raw_full_state__eigenvector_hidden_fraction']==0
            slow_modes.append(int(slow.sum()));cell_only.append(int((slow&silent).sum()))
        table.append((labels[x].replace('\n',' / '),
            '/'.join(f'{r["slow_abs_eigenvalue_ge_099_count"]["median"]:g}' for r in s),
            '/'.join(f'{r["radial_over_tangent_median"]:.2f}' for r in o),
            '/'.join(f'{r["maximum_transverse_gain"]["median"]:.1f}' for r in s)))
    axes[0].set(title='Many slow modes',ylabel='Slow-mode count (median across phases)',ylim=(0,18),yticks=[0,4,8,12,16])
    axes[1].axhline(1,color='#8a8a8a',lw=.8)
    axes[1].axhline(.5,color='#8a8a8a',lw=.8,ls='--')
    axes[1].set(title='Radial changes persist',ylabel='Radial / angular change in activity',ylim=(.4,1.2),yticks=[.5,.75,1.])
    for color,marker,label in [('#48535c','o','Angular'),('#2878a4','s','Radial'),('#a4526b','^','Worst normal')]:
        axes[2].scatter([],[],color=color,marker=marker,s=25,label=label)
    axes[2].axhline(1,color='#8a8a8a',lw=.8)
    axes[2].set(title='Full-state response',ylabel='Gain over 2 seconds',yscale='log',ylim=(.65,35),yticks=[1,5,10,25])
    axes[2].set_yticklabels(['1','5','10','25'])
    axes[2].legend(frameon=False,fontsize=8,loc='upper left')
    for ax in axes:
        ax.set_xticks(range(4),labels,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.13)
    fig.supxlabel('Two contexts per fixed model · one training seed per population · raw h/c units',fontsize=9)
    fig.savefig(OUT/'memory_spectra.png',dpi=180);plt.close(fig)
    silent_fraction=sum(cell_only)/sum(slow_modes)
    lines=['# Memory traces and spectra after 75M additional frames','',
        '**Direction-related activity persists, while radial changes recover about as slowly as angular changes. The full state contains several slow cell-memory modes.**','',
        '![Observable recovery and full-state spectra](memory_spectra.png)','',
        'Each dot summarizes one fixed travel context; spectral counts and gains are medians across 16 phases. Both original seed-1 agents received 75.072M additional frames; the noise arms added recurrent noise during the final 50.048M. These are descriptive comparisons of four models, not estimates across independent training seeds.','',
        '**Left:** at each sampled phase, the 2-second state-update product has 6–16 eigenvalues with magnitude at least 0.99. '
        f'Across the sampled phases, {100*silent_fraction:.1f}% of slow eigenvectors have exactly zero hidden-state component in float64; they are locally actor-silent cell-state modes. '
        'Slow memory therefore occupies multiple state directions.','',
        '**Middle:** radial and angular kicks were measured through all 256 hidden outputs over the frozen observation window. '
        'A ratio near 1 means radial changes persist about as much as angular changes. Every travel-context median lies above the prespecified 0.5 selective-recovery threshold (dashed line). '
        'Across heading/travel branches and both coordinate metrics, all 48 observable-factor screens retain their negative result. The activity families also deform and lose phase order.','',
        '**Right:** the linearized ordinary-input map from 2s to 4s retains angular and radial disturbances near unit size; its worst transverse direction grows 8.3–25.0× (context medians). '
        '**This maximum includes spectator cell-state dimensions. It is a full-state sensitivity, not a measure of behavioral instability.** '
        'The observable recovery test in the middle addresses activity that can influence the actor.','',
        '| Model | Median slow-mode counts | Observable radial/angular | Median worst normal gain |',
        '|---|---:|---:|---:|']
    lines += [f'| {a} | {b} | {c} | {d} |' for a,b,c,d in table]
    lines += ['',
        'Each table entry lists context 0 / context 1. Spectral medians use the 16 phases within each context: a value such as 10.5 is the median of integer mode counts, not a fractional mode.','',
        'The spectra use all 512 state coordinates at 16 uniformly preselected phases in every one of the eight travel contexts. '
        'They describe a fixed-input finite-time propagator, not a Floquet or period-return map. '
        'Raw and h/c-RMS coordinates, all eigenvalues/singular values, eigenvector alignment, the top 16 full eigenvectors and products are retained. No new ring gate or favorable-context selection was added.','',
        'Numerical replay agrees with the saved trajectories to 5.69×10⁻¹⁴ and with the native LSTMCell exactly. '
        'The angular finite-difference propagation error has context medians below 0.019, with an outlying maximum 0.575 in block-scaled coordinates; all phases remain included. '
        'High tangent alignment is also a smooth-flow consistency check.','',
        '[Spectral summary](summary.json) · [Observable summary](../lstm_endogenous_ring/observable_history/final75/summary.json) · '
        '[Primary tests](../lstm_final75_memory/README.md) · [Frozen spectral protocol](protocol.json) · '
        '[Numerical validation](numerical_validation.json) · [Artifact verification](artifact_verification.json)','',
        'Reproduce the figure with `python scripts/plot_lstm_final75_secondary.py`. '
        'The frozen calculations are in `scripts/lstm_final75_finite_spectra.py` and `scripts/lstm_endogenous_continued_observable_v2.py`; '
        'their exact source and checkpoint hashes accompany the results.','']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
