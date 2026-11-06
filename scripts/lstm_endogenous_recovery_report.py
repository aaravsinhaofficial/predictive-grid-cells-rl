"""Aggregate every original-checkpoint corrected-recovery pilot context."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_endogenous_curve_recovery import sha,save


def main():
    root=ROOT/'results/lstm_endogenous_ring'
    primary=json.loads((root/'corrected_recovery/protocol.json').read_text())
    for source,digest in primary['sources'].items():assert sha(ROOT/source)==digest,source
    secondary=json.loads((root/'metric_audit/protocol.json').read_text())
    assert secondary['source_sha256']==sha(ROOT/'scripts/lstm_endogenous_metric_audit.py')
    models=[];all_spectra=[];maxfd=0.;maxreconstruct=0.
    for condition in ('gc100','pgc33'):
        for seed in (1,2):
            points=[];checks=[]
            for context in range(4):
                p=root/f'corrected_recovery/{condition}_s{seed}/context{context}.json'
                q=root/f'metric_audit/{condition}_s{seed}/context{context}.json'
                a,b=json.loads(p.read_text()),json.loads(q.read_text())
                checkpoint=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt'
                assert a['checkpoint_sha256']==b['checkpoint_sha256']==sha(checkpoint)
                assert a['source_sha256']==primary['sources']['scripts/lstm_endogenous_curve_recovery.py']
                assert b['source_sha256']==secondary['source_sha256']
                assert (a['condition'],a['seed'],a['context'])==(condition,seed,context)
                maxreconstruct=max(maxreconstruct,a['native_step_error'])
                maxfd=max(maxfd,*[v['finite_difference_relative_error']for ss in a['local_recovery'].values()for v in ss.values()])
                points.append(a);checks.append(b);all_spectra.extend(b['local_spectra'])
            times=list(checks[0]['longer_recovery'])
            gain={}
            for metric in ('block_RMS','raw'):
                gain[metric]={key:[float(np.mean([b['longer_recovery'][t][metric]['tangent_norm_gain']['median'] if key=='tangent' else b['longer_recovery'][t][metric]['normal_direction_sets'][metric]['transverse_gain']['median']for b in checks]))for t in times]for key in ('tangent','normal')}
            specs=[s for b in checks for s in b['local_spectra']]
            models.append(dict(condition=condition,seed=seed,contexts=4,
                candidate_contexts=sum(a['passes_candidate_screen']for a in points),
                times_updates=list(map(int,times)),gains=gain,
                slow_eigenvalue_count_median=float(np.median([s['eigenvalues_modulus_at_least_099']for s in specs])),
                closest_one_tangent_alignment_median=float(np.median([s['nearest_one_tangent_alignment']for s in specs])),
                radius5s_block_mean=float(np.mean([a['trajectory']['75']['radius_retention']for a in points])),
                radius5s_raw_mean=float(np.mean([b['longer_recovery']['0']['raw']['family_radius_from_original']for b in checks])),
                normal_old_1s_scaled=float(np.mean([a['local_recovery']['75']['goal_withheld']['horizons']['15']['normal_total_gain_old_metric']['median']for a in points])),
                normal_corrected_1s_scaled=float(np.mean([a['local_recovery']['75']['goal_withheld']['horizons']['15']['normal_transverse_gain_corrected']['median']for a in points]))))
    result=dict(completed_contexts=16,models=models,all_hashes_verified=True,
        max_lstm_step_reconstruction_error=maxreconstruct,max_finite_difference_relative_error=maxfd,
        candidate_contexts=sum(m['candidate_contexts']for m in models),
        statistics='Two trained seed pairs. All contexts retained; descriptive model averages only. No population p-values.',
        distinction='The local phase coordinate under an advected common-input family is constant by construction; phase gain1 there is not independent evidence of memory. Physical tangent size, family radius, longer normal recovery and spectra are reported.')
    save(root/'summary.json',result)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    colors={'gc100':'#367fb5','pgc33':'#db7933'}
    names={'gc100':'Grid','pgc33':'PGC + grid'}
    for condition in ('gc100','pgc33'):
        group=[m for m in models if m['condition']==condition]
        times=np.array(group[0]['times_updates'])[1:]/15
        for key,style in [('tangent','-'),('normal','--')]:
            vals=np.array([m['gains']['raw'][key][1:]for m in group])
            axes[0].plot(times,vals.mean(0),style,color=colors[condition],lw=2,label=f'{names[condition]}: {key}')
        for j,m in enumerate(group):
            offset=(j-.5)*.1
            axes[1].plot(np.array([0,1])+offset,[m['normal_old_1s_scaled'],m['normal_corrected_1s_scaled']],'-o',color=colors[condition],alpha=.7)
            axes[2].scatter((0 if condition=='gc100' else 1)+offset,m['slow_eigenvalue_count_median'],s=55,c=colors[condition])
    axes[0].axhline(1,color='.6',lw=1);axes[0].set(xscale='log',yscale='log',xlabel='Recovery time after 5 s settling (s)',ylabel='Perturbation size / initial, raw (h, c)',title='Early contraction gives way to growth')
    axes[0].set_yticks([.5,1,2,5],labels=['0.5','1','2','5'])
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(xticks=[0,1],xticklabels=['Distance to\noriginal trajectory','Distance across\nadvected curve'],ylabel='One-second gain, block-RMS metric',title='Removing phase shift changes little')
    axes[1].set_xlim(-.25,1.25);axes[1].axhline(1,color='.6',lw=1)
    axes[2].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],ylabel='Number of eigenvalues with |λ| ≥ 0.99',title='Many slow state directions')
    axes[2].set_xlim(-.3,1.3);axes[2].set_ylim(bottom=0)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for suffix in ('png','pdf','svg'):fig.savefig(root/f'corrected_goal_recovery.{suffix}',dpi=180)
    text='''# Corrected goal-curve recovery in the original LSTMs

We tested all four cached native contexts in each of the original Grid and PGC models of seeds1–2. The original weights, renderer-derived visual embedding, spatial maps, and actual histories were preserved.

The new check addresses a specific possible false negative. A disturbance away from a ring can settle onto a different phase. Measuring its distance to the original trajectory then confuses phase displacement with failure to return to the ring. We propagated the exact512-dimensional LSTM differential, separated displacement along the evolving goal-state curve from residual displacement across it, and checked the calculation against finite perturbations. A synthetic sheared ring verifies that this correction detects transverse recovery even when the old pointwise error grows.

![Corrected recovery, coordinate sensitivity and local spectra](corrected_goal_recovery.png)

**Outcome:** none of the16contexts meets the frozen candidate screen. After5seconds with one common physical input and the goal slots blank, removing induced phase displacement changes the normal-error result only slightly. The primary screen's mean context-median gain changes4.93→4.79 for Grid and5.26→5.23 for PGC in block-RMS coordinates. The frozen combination of family-size and recovery criteria is unmet.

Raw coordinates change the early finite-time magnitude. An all-context follow-up retained both coordinate systems and both raw-isotropic and block-isotropic disturbances. With raw-isotropic disturbances, one-second transverse gains average0.50Grid/0.54PGC, then rise to1.28/1.08 at5seconds and5.40/5.28 at100seconds. Tangent sizes over those same horizons are0.94/0.97,0.80/0.87,and0.63/0.36. Thus the short raw-coordinate contraction does not continue as a protected one-dimensional goal-memory ring in this assay.

The instantaneous full-state spectra contain many slow directions: median24Grid/20.5PGC eigenvalues have modulus≥0.99 across the prespecified contexts/phases. The eigenvector closest to eigenvalue1 is poorly aligned with the goal tangent in many cases. These spectra belong to slow trajectory states, not optimized fixed points; finite-time growth is not equated with an unstable eigenvalue. Some slow c-state directions may be invisible to behavior, so this full-state result does not determine whether a task-relevant angular factor exists inside a larger dynamical system.

**Method details:** each valid goal-code family first experiences15actual recorded physical updates, with only goal slots varying. We then repeat the endpoint physical input while withholding goal input; this is an explicit dynamical intervention, not a closed-loop navigation trajectory. The primary probes begin after0,1,5,and10seconds of common-input settling. All96sampled goal phases, all16contexts, and all matched native/ordinary/withheld restoration conditions are retained. The main candidate screen requires≥20%initial family radius after5seconds, then tangent gain≥0.5, transverse gain<0.5,and their ratio>2over the next1second. These are screening criteria, not a definition that every ring must meet them. Slow rings can contain stable fixed points and saddles rather than a continuum of exact fixed points.

The block-RMS metric divides each h/c block by its initial family-wide RMS; raw coordinates are reported separately. A phase coordinate that remains1under propagation with the reference family is tautological; the report instead keeps physical tangent size, family radius and actual state motion visible. No circular readout or imposed ring enters the dynamics.

All source/checkpoint hashes pass, all finite-amplitude checks pass, and three focused mathematical tests pass. The follow-up did not change the frozen first-screen source or its thresholds. [Numerical summary](summary.json) · [Frozen primary protocol](corrected_recovery/protocol.json) · [Descriptive metric audit](metric_audit/protocol.json). The subsequent [fixed-point and saturation audit](fixed_points_and_saturation.md) checks whether full-state drift could hide a useful stable factor.

Reproduce:

```bash
python scripts/lstm_endogenous_curve_recovery.py
python scripts/lstm_endogenous_metric_audit.py
python scripts/lstm_endogenous_recovery_report.py
python -m pytest -q tests/test_lstm_endogenous_curve_recovery.py
```
'''
    (root/'README.md').write_text(text)
    print(json.dumps({k:v for k,v in result.items()if k!='models'},indent=2))


if __name__=='__main__':main()
