"""Keep all held-out seed pairs and contexts in the frozen observable audit."""
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
    root=ROOT/'results/lstm_endogenous_ring/observable_history/travel_heldout_original'
    summary=json.loads((root/'summary.json').read_text());rows=summary['rows']
    assert summary['cases']==24 and len(rows)==48
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['observable_source_sha256']==sha(ROOT/'scripts/lstm_endogenous_observable_flow.py')
    for p in root.glob('*_s*/context*.json'):
        d=json.loads(p.read_text());assert d['observable_source_sha256']==protocol['observable_source_sha256']
        assert d['source_sha256']==protocol['source_sha256'];assert sha(d['checkpoint'])==d['checkpoint_sha256']
    pairs=[]
    for metric in ('raw_full_state_kicks','h_c_RMS_full_state_kicks'):
        for seed in range(3,9):
            pair=dict(seed=seed,metric=metric)
            for condition in ('gc100','pgc33'):
                group=[r for r in rows if r['model']==f'{condition}_s{seed}' and r['metric']==metric]
                assert len(group)==2
                pair[condition]=dict(mean_context_radial_over_tangent=float(np.mean([r['radial_over_tangent']for r in group])),
                    context_radial_over_tangent=[r['radial_over_tangent']for r in group],
                    mean_context_centered_invariance_p90=float(np.mean([r['centered_invariance_p90']for r in group])),
                    candidate_contexts=sum(r['candidate']for r in group))
            pair['pgc_minus_grid']=pair['pgc33']['mean_context_radial_over_tangent']-pair['gc100']['mean_context_radial_over_tangent'];pairs.append(pair)
    result=dict(seed_pairs=pairs,contexts=24,metric_results=48,candidate_metric_results=sum(r['candidate']for r in rows),
        all_hashes_verified=True,statistics='Descriptive held-out seed variation. Each seed statistic averages two preselected native contexts. No context or checkpoint selection; no population p-values.')
    save(root/'seed_summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(7.8,3.8),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['raw_full_state_kicks','h_c_RMS_full_state_kicks'],['Raw-state kicks','h/c-scaled kicks']):
        for pair in [p for p in pairs if p['metric']==metric]:
            values=[pair[c]['mean_context_radial_over_tangent']for c in ('gc100','pgc33')]
            ax.plot([0,1],values,'-',color='.65',alpha=.7,lw=1)
            ax.scatter([0,1],values,c=['#367fb5','#db7933'],s=35,zorder=3)
        ax.axhline(.5,color='.5',ls='--',lw=1);ax.set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],title=title,
            ylabel='Mean radial / phase gain',yscale='log',xlim=(-.2,1.2),ylim=(.25,2.5));ax.spines[['top','right']].set_visible(False)
        ax.set_yticks([.25,.5,1,2],labels=['0.25','0.5','1','2']);ax.minorticks_off()
    fig.suptitle('Six held-out seed pairs; two contexts per model',fontsize=12)
    for ext in ('png','pdf','svg'):fig.savefig(root/f'heldout_seed_variation.{ext}',dpi=180)
    counts={metric:sum(p['pgc_minus_grid']<0 for p in pairs if p['metric']==metric)for metric in ('raw_full_state_kicks','h_c_RMS_full_state_kicks')}
    text='# Held-out observable travel-direction audit\n\n'
    text+='The calculation and criteria were frozen on seeds 1–2, then applied unchanged to seeds 3–8. Every one of the 24 native travel contexts is retained. Only the common ordinary-current release is used, so persistent phase-specific predictive inputs cannot create an apparent autonomous ring.\n\n'
    text+=f'**{result["candidate_metric_results"]} of 48 metric-specific results pass the conditional-factor screen.** The PGC radial/phase ratio is lower in {counts["raw_full_state_kicks"]}/6 seed pairs for raw-state kicks and {counts["h_c_RMS_full_state_kicks"]}/6 for h/c-scaled kicks. Each seed value is the mean of its two context medians; these comparisons describe held-out variation rather than a population significance claim.\n\n'
    text+='![Held-out paired variation](heldout_seed_variation.png)\n\n| Seed | Grid ratio | PGC ratio | PGC minus Grid |\n|---|---:|---:|---:|\n'
    for pair in [p for p in pairs if p['metric']=='raw_full_state_kicks']:
        text+=f'| {pair["seed"]} | {pair["gc100"]["mean_context_radial_over_tangent"]:.3f} | {pair["pgc33"]["mean_context_radial_over_tangent"]:.3f} | {pair["pgc_minus_grid"]:+.3f} |\n'
    text+='\nLower radial/phase gain means stronger recovery of radial disturbances relative to preservation of phase. It remains a partial property: stable geometry, phase coverage and useful output must also be present. The complete failed-gate lists and both metrics are retained in the [all-context summary](summary.json); [seed summary](seed_summary.json); [frozen adapter protocol](protocol.json).\n'
    (root/'README.md').write_text(text)
    print(json.dumps(dict(candidate_metric_results=result['candidate_metric_results'],pgc_lower_seed_pairs=counts,pairs=pairs),indent=2))


if __name__=='__main__':main()
