"""Report every saved original observable-history control; no selected cases."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_endogenous_curve_recovery import sha,save


def main():
    root=ROOT/'results/lstm_endogenous_ring/observable_history'
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['source_sha256']==sha(ROOT/'scripts/lstm_endogenous_observable_flow.py')
    rows=[]
    for p in sorted((root/'original').glob('*/*.json')):
        d=json.loads(p.read_text());assert d['source_sha256']==protocol['source_sha256']
        for name,branch in d['branches'].items():
            for metric,r in branch.items():
                center=r['invariance']['centered'];plain=r['invariance']['noncentered']
                rows.append(dict(model=p.parent.name,input=p.stem,branch=name,metric=metric,
                    centered_candidate=r['centered_conditional_factor_candidate'],
                    radial_gain=r['radial_observable_normal_gain']['median'],tangent_gain=r['tangent_observable_total_gain']['median'],
                    radial_over_tangent=r['radial_over_tangent_observable_gain']['median'],
                    centered_invariance_p90=center['residual_to_reference_curve_over_radius']['p90'],
                    noncentered_invariance_p90=plain['residual_to_reference_curve_over_radius']['p90'],
                    observable_radius_retention=center['evolved_radius_retention'],phase_coverage=center['phase_transport']['circular_coverage'],
                    failed_gates=[k for k,v in r['candidate_gates']['centered'].items()if not v],actor=r['actor']))
    assert len(rows)==12
    save(root/'original_summary.json',dict(rows=rows,source_and_protocol_verified=True,candidates=sum(r['centered_candidate']for r in rows)))
    labels=sorted(set((r['model'],r['input'])for r in rows));fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for x,label in enumerate(labels):
        color='#367fb5' if label[0].startswith('gc') else '#db7933'
        for metric,marker,offset in [('raw_full_state_kicks','o',-.1),('h_c_RMS_full_state_kicks','s',.1)]:
            r=next(r for r in rows if (r['model'],r['input'])==label and r['metric']==metric)
            for ax,key in zip(axes,['radial_over_tangent','centered_invariance_p90']):ax.scatter(x+offset,r[key],color=color,marker=marker,s=45)
    ticklabels=[('Grid' if model.startswith('gc') else 'PGC')+' '+model.rsplit('_',1)[1]+'\n'+inp for model,inp in labels]
    axes[0].set(ylabel='Radial gain / phase gain',title='Some directions recover faster than phase',yscale='log')
    axes[0].axhline(.5,color='.5',lw=1,ls='--');axes[0].axhline(1,color='.8',lw=1)
    axes[1].set(ylabel='Distance / reference-curve radius',title='The activity curve still changes after centering')
    axes[1].axhline(.05,color='.5',lw=1,ls='--');axes[1].set_ylim(bottom=0)
    for ax in axes:ax.set_xticks(range(len(labels)),ticklabels,fontsize=8);ax.spines[['top','right']].set_visible(False)
    axes[0].scatter([],[],color='.4',marker='o',label='Raw-state kicks');axes[0].scatter([],[],color='.4',marker='s',label='h/c-scaled kicks')
    axes[0].legend(frameon=False,fontsize=8)
    for ext in ('png','pdf','svg'):fig.savefig(root/f'original_observable_control.{ext}',dpi=180)
    (root/'README.md').write_text('''# Does an observable ring survive after common drift is removed?

We made the test sensitive to an angular subcomputation inside a larger drifting LSTM. For every state, the measurement includes **all 256 hidden outputs now and at three later times**. This fixed 1,024-dimensional observable history exposes internal memory when it affects future activity. It uses no fitted decoder or chosen two-dimensional plane.

We compared the activity curves both before and after subtracting their common phase-family mean. The mean subtraction removes shared background motion; differences among phases remain. The same actual full-state radial and tangent kicks were used in both comparisons, with a guard against perturbations that are invisible in the measured activity. Everything ends within four seconds: a one-second reference, two-second evolution, and one final second of observation.

![Observable-history sensitivity](original_observable_control.png)

**All six original native/released cases fail the frozen conditional-ring-factor screen in both perturbation constructions.** There is nevertheless a specific dynamical difference: PGC seed1 has substantially faster observable radial recovery than phase recovery, with a raw-state ratio of 0.154. That ratio remains 0.120 after replacing current predictive inputs with ordinary inputs. The corresponding Grid ratios are 1.140 and 0.689; PGC seed2 is 0.549 under native input. These are two seed pairs, so the individual outcomes remain visible rather than being presented as a population significance claim.

The PGC seed1 activity curve also changes by about 0.95 of its reference radius after common drift is removed, and loses phase coverage/order. Its faster radial recovery therefore suggests anisotropic short-term dynamics. It does not supply a stable angular curve, and the ordinary-input comparison does not attribute the recovery to an active predictive cue. Some observable curves also have large sampled edges; those resolution failures are retained explicitly.

We also applied the identical frozen control to **last-travel-direction families**, which vary the direction actually represented by PGC input. Only their ordinary-current released states were used, so every phase received the same input. All four original models failed the conditional-factor screen. Their raw radial/phase ratios were Grid 0.922/0.541 and PGC 0.331/0.616; after mean removal, curve-change errors were 1.766/0.416 and 0.663/0.987 radii, respectively. This preserves the PGC seed1 recovery finding while keeping it distinct from stable ring geometry. [All travel-direction results](travel_original/summary.json).

The code passes three controls: a known ring combined with shared observable drift passes the centered test; a shrinking plane sampled around a circle fails; and an output-silent circle fails the nontrivial-perturbation guard. The primary full-state tests are unchanged. This is an explicitly secondary, conditional ring-factor analysis.

[Frozen protocol](protocol.json) · [All original outcomes](original_summary.json). A serialization-only fix was needed before any result could be written; the earlier source and protocol are retained beside the current protocol, and the numerical calculation and thresholds were unchanged.

The same frozen implementation accepts the final continued models and their native/ordinary saved curves:

```bash
python scripts/lstm_endogenous_observable_flow.py --originals
python scripts/lstm_endogenous_observable_report.py
python scripts/lstm_endogenous_observable_flow.py --checkpoint FINAL.pt --archive context0_states.npz --label continued/MODEL
python -m pytest -q tests/test_lstm_endogenous_observable_flow.py
```
''')
    continued=[]
    for family,paths in [('heading',sorted((root/'continued').glob('*/context*_states.json'))),('travel',sorted((root/'travel_continued').glob('*/*_s*/context*.json')))]:
        for p in paths:
            d=json.loads(p.read_text())
            assert (d.get('observable_source_sha256') or d['source_sha256'])==protocol['source_sha256']
            assert sha(d['checkpoint'])==d['checkpoint_sha256']
            for name,branch in d['branches'].items():
                for metric,r in branch.items():
                    continued.append(dict(family=family,condition=d['condition'],context=p.stem.split('_')[0],input=name,metric=metric,
                        candidate=r['centered_conditional_factor_candidate'],radial_over_tangent=r['radial_over_tangent_observable_gain']['median'],
                        radial_gain=r['radial_observable_normal_gain']['median'],tangent_gain=r['tangent_observable_total_gain']['median'],
                        centered_invariance_p90=r['invariance']['centered']['residual_to_reference_curve_over_radius']['p90'],
                        failed_gates=[k for k,v in r['candidate_gates']['centered'].items()if not v]))
    if continued:
        save(root/'continued_summary.json',dict(rows=continued,metric_results=len(continued),candidates=sum(r['candidate']for r in continued),
            interpretation='Fixed final25.024M pair, both fresh contexts and all matched input branches. Same frozen observable-history calculation; no model selection.'))
        text='\n## Fixed 25.024M continued pair\n\n'
        text+=f'All {len(continued)} metric-specific tests are retained; {sum(r["candidate"]for r in continued)} pass the conditional-factor screen. These are the same fixed final Grid/PGC models used by the primary analysis, with fresh native contexts and no checkpoint selection.\n\n'
        text+='| Family | Model | Context | Current input | Radial / phase gain | Centered curve-change error |\n|---|---|---|---|---:|---:|\n'
        for r in continued:
            if r['metric']!='raw_full_state_kicks':continue
            name='Grid' if r['condition']=='gc100' else 'PGC + grid'
            text+=f'| {r["family"]} | {name} | {r["context"]} | {r["input"]} | {r["radial_over_tangent"]:.3f} | {r["centered_invariance_p90"]:.3f} |\n'
        text+='\nThe table uses raw-state kicks; both constructions and every failed gate are in the [continued summary](continued_summary.json). Curve-change error is measured in reference-curve radii. All observations finish within four seconds.\n'
        with (root/'README.md').open('a')as handle:handle.write(text)


if __name__=='__main__':main()
