"""Summarize original and fixed25M every-update h/c reset controls."""
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
    out=ROOT/'results/lstm_endogenous_ring/every_blind_update_reset';rows=[];artifacts=[]
    for condition,tag in [('gc100','grid'),('pgc33','pgc')]:
        worker=ROOT/f'results/ring_cloud_artifacts/ring-20261105-vision-long-{tag}-01'
        v1=ROOT/f'results/ring_cloud_artifacts/ring-20261105-vision-{tag}-01'
        for cohort,folder,reference_file in [('original','every_blind_reset_original','baseline.json'),('continued25M','every_blind_reset_v1final','final_evaluation.json')]:
            path=worker/folder/'result.json';d=json.loads(path.read_text());reference=json.loads((v1/reference_file).read_text())
            assert d['source_sha256']==sha(ROOT/'scripts/lstm_endogenous_every_blind_reset.py')
            checkpoint=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt' if cohort=='original' else v1/'final.pt'
            assert d['checkpoint_sha256']==sha(checkpoint)
            assert all(d['variants'][v]['rows']==reference['variants'][v]['rows']for v in ('blind','seen','blind_current_grid','seen_current_grid'))
            assert d['all_nonreset_rows_identical']
            artifacts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),checkpoint_sha256=sha(checkpoint)))
            for visibility in ('blind','seen'):
                usual=d['variants'][visibility]['mean_score'];onset=reference['variants'][visibility+'_reset']['mean_score'];every=d['variants'][visibility+'_reset']['mean_score']
                rows.append(dict(condition=condition,cohort=cohort,visibility=visibility,usual_score=usual,onset_reset_score=onset,every_update_reset_score=every,
                    onset_cost=usual-onset,every_update_cost=usual-every,resets=d['state_reset_counts'][visibility+'_reset']))
    result=dict(rows=rows,artifacts=artifacts,all_four_nonreset_row_identity_checks_pass=True,
        episodes_per_model_and_branch=64,training_seed_pairs=1,
        intervention='Reset h and c before every scheduled blind update. In the seen control, retain RGB while applying the same reset scheduling rule.',
        preserved='Previous action/reward inputs, current spatial code, saved external goal, physical state and its history, trained weights, rewards and first-discovery trajectory.',
        interpretation='Tests dependence on h/c carryover under this intervention. Available inputs and physical state still carry history. Seen effects can include a reset-induced state-distribution shift; no unique stored-information claim. Descriptive one-seed-pair pilot.')
    save(out/'summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.5),layout='constrained',sharey=True)
    colors={'gc100':'#367fb5','pgc33':'#db7933'};names={'gc100':'Grid','pgc33':'PGC + grid'}
    for ax,visibility,title in zip(axes,['blind','seen'],['Blackout gains survive h/c reset','Visible behavior depends on h/c carryover']):
        for condition in ('gc100','pgc33'):
            group=[next(r for r in rows if r['condition']==condition and r['cohort']==cohort and r['visibility']==visibility)for cohort in ('original','continued25M')]
            ax.plot([0,1],[r['usual_score']for r in group],'-o',color=colors[condition],label=names[condition]+' usual')
            ax.plot([0,1],[r['every_update_reset_score']for r in group],'--s',color=colors[condition],label=names[condition]+' scheduled h/c resets')
        ax.set(xticks=[0,1],xticklabels=['Original','After 25M'],title=title,ylabel='Mean episode score',ylim=(0,95),xlim=(-.15,1.15));ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.supxlabel('Reset before every scheduled blackout update; the visible control keeps its images.',fontsize=8)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'every_blind_update_reset.{ext}',dpi=180)
    text='''# Does blackout behavior require recurrent carryover on every step?

We reset both LSTM variables, `h` and `c`, **before every scheduled blind update**. The earlier control reset them only once, when the blackout began. The ordinary evaluation function was unchanged; a small clock adapter returned the next scheduled-blind mask instead of only the onset signal. Its tests verify exact clock timing, identical clock RNG/state, and no extra resets before goal discovery.

![Every-update recurrent-state reset](every_blind_update_reset.png)

**Most of the learned blackout improvement survives removing h/c carryover on every blind update.** The continued Grid model scores 43.59 normally and 44.22 with this intervention; continued PGC scores 47.81 and 45.78. Each result uses 64 episodes from the same fixed seed-1 models. The unchanged branches reproduce every corresponding episode row in the parent CUDA evaluation exactly.

Previous action/reward, current spatial codes, the external saved-goal code and the environment's physical state remain available. They still carry information about history. The result suggests the blackout improvement is largely supported by computation from those available inputs and externally carried state, rather than requiring maintained h/c across blind updates.

Applying the same reset schedule while keeping images visible has much larger effects. This shows dependence on recurrent state under the intervention; it also changes the state distribution seen by the network, so it does not uniquely identify stored information. The original and continued outcomes, including onset-only controls, are retained below.

| Model | Vision | Usual h/c | Reset at onset | Reset every scheduled update |
|---|---|---:|---:|---:|
'''
    for r in rows:
        model=('Grid' if r['condition']=='gc100'else 'PGC + grid')+(' original' if r['cohort']=='original'else ' +25M')
        text+=f'| {model} | {r["visibility"]} | {r["usual_score"]:.2f} | {r["onset_reset_score"]:.2f} | {r["every_update_reset_score"]:.2f} |\n'
    text+='\nThe every-update branches contain about 32,000–40,000 resets each. These are descriptive results from one trained seed pair, with no population-level significance claim. [Summary and artifact identities](summary.json) · [Frozen intervention protocol](protocol.json). The discarded local MPS attempt completed no model outcome; authoritative results are the same-CUDA cloud diagnostics.\n'
    (out/'README.md').write_text(text)
    print(json.dumps(dict(all_checks_pass=True,rows=rows),indent=2))


if __name__=='__main__':main()
