"""Exploratory subtraction of the predictive episode-specific hidden offset.

This complements restoration tests. It subtracts a past-calibrated predictive
contribution, not the entire episode mean. The six underlying policies retain
their original predictive current and goal encoding.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_lstm_episode_offset_rescue import estimate, paired, contrasts, keyed, aggregate_calibration

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'results/lstm_episode_offset_rescue_1B'
OUT = MAIN / 'ablation'
SEEDS = tuple(range(1,9))
VARIANTS = ('intact','remove_episode_mean','remove_wrong_episode_mean',
            'remove_between_episode_mean','remove_episode_mean_feedback','remove_wrong_episode_mean_feedback')
LABELS = dict(intact='Intact',
    remove_episode_mean='Subtract correct offset',
    remove_wrong_episode_mean='Subtract wrong offset',
    remove_between_episode_mean='Subtract in episode directions',
    remove_episode_mean_feedback='Correct subtraction + feedback',
    remove_wrong_episode_mean_feedback='Wrong subtraction + feedback')
CONTRASTS = (
    ('correct_minus_intact','intact','remove_episode_mean','Correct subtraction − intact'),
    ('correct_minus_wrong','remove_wrong_episode_mean','remove_episode_mean','Correct subtraction − wrong subtraction'),
    ('between_minus_intact','intact','remove_between_episode_mean','PCA subtraction − intact'),
    ('feedback_minus_intact','intact','remove_episode_mean_feedback','Correct feedback subtraction − intact'),
    ('feedback_minus_wrong','remove_wrong_episode_mean_feedback','remove_episode_mean_feedback','Correct feedback − wrong feedback'),
)


def validate(runs,main_dir):
    first_checks=prefix_checks=prior_checks=main_checks=0
    eligible_counts={}
    for seed,run in runs.items():
        assert (run['seed'],run['condition'],run['checkpoint_frames'])==(seed,'pgc33',1_000_000_000)
        assert (run['episodes'],run['n_envs'],run['eval_seed'])==(128,64,737373)
        assert set(run['variants'])==set(VARIANTS)
        assert run['basis']['rank']==10
        checks=run['checks']
        assert checks['prefix_exact'] and checks['first_goal_records_identical'] and checks['intact_baseline_match']
        assert checks['full_rescue_logit_max_error']<2e-4
        baseline=keyed(run['variants']['intact']['rows'])
        assert len(baseline)==128
        eligible={key for key,row in baseline.items() if row['eligible']}
        assert len(eligible)>1
        assert all(bool(row['eligible']) == (0 <= row['first_goal_t'] < 450) for row in baseline.values())
        eligible_counts[str(seed)]=len(eligible)
        for variant in VARIANTS:
            summary=run['variants'][variant];rows=keyed(summary['rows'])
            assert rows.keys()==baseline.keys()
            assert {key for key,row in rows.items() if row['eligible']}==eligible
            assert summary['episodes']==128 and summary['eligible_episodes']==len(eligible)
            assert np.isclose(summary['mean_score'],np.mean([r['ret'] for r in rows.values()]),atol=1e-9)
            assert np.isclose(summary['eligible_mean_score'],np.mean([rows[key]['ret'] for key in eligible]),atol=1e-9)
            assert np.isclose(summary['post_handoff_score'],np.mean([rows[key]['post_score'] for key in eligible]),atol=1e-9)
            for key,row in rows.items():
                assert np.isclose(row['post_score']+row['prefix_score'],row['ret'],atol=1e-8)
                assert row['first_goal_t']==baseline[key]['first_goal_t']
                assert row['prefix_score']==baseline[key]['prefix_score']
                if variant!='intact':first_checks+=1;prefix_checks+=1
                if key not in eligible:
                    assert row==baseline[key],(seed,variant,key,'ineligible changed')
            diag=run['patch_diagnostics'][variant]
            assert diag['samples']>0
            assert diag['patch_reference']=='mixed_hidden'
            for field in ('mean_patch_l2','maximum_absolute_actor_hidden'):
                assert np.isfinite(diag[field]) and diag[field]>=0
            for field in ('units_outside_unit_range_fraction','samples_outside_unit_range_fraction'):
                assert 0<=diag[field]<=1
        batches=run['calibration']['batches']
        assert len(batches)==2 and sum(b['eligible_count'] for b in batches)==len(eligible)
        for batch in batches:
            assert batch['eligible_count']>1 and batch['total_slots']==64
            assert batch['wrong_derangement_available'] and batch['wrong_norm_max_error']<2e-4
            future=batch['future_validation']
            assert (future['future_start'],future['future_stop_exclusive'],future['future_steps'])==(600,1350,750)
            assert future['eligible_count']==batch['eligible_count']
            for space in ('centered_hidden','centered_actor_logits'):
                record=future[space]
                assert 0<=record['nonzero_episode_pairs']<=batch['eligible_count']
                assert all(record[key]>=0 for key in ('squared_error','target_energy','prediction_energy'))
                for field in ('cosine','mean_episode_cosine'):
                    assert record[field] is None or -1-1e-8 <= record[field] <= 1+1e-8
                if record['target_energy']>1e-12:
                    assert np.isclose(record['prediction_r2'],1-record['squared_error']/record['target_energy'],atol=1e-8)
        old_path=ROOT/f'results/lstm_code_swap_1B/pgc33_s{seed}.json'
        if old_path.exists():
            old=json.loads(old_path.read_text())
            assert old['checkpoint_sha256']==run['checkpoint_sha256']
            original=keyed(old['variants']['intact']['rows']);assert original.keys()==baseline.keys()
            for key,row in original.items():
                assert all(row[field]==baseline[key][field] for field in ('ret','n_goals','first_goal_t','path_eff','path_eff_n'))
                prior_checks+=1
        rescue_path=main_dir/f'pgc33_s{seed}.json'
        if rescue_path.exists():
            rescue=json.loads(rescue_path.read_text())
            assert rescue['checkpoint_sha256']==run['checkpoint_sha256']
            assert keyed(rescue['variants']['intact']['rows'])==baseline
            main_checks+=len(baseline)
    return dict(first_goal_episode_checks=first_checks,prefix_score_episode_checks=prefix_checks,
        first_goal_and_prefix_invariance=True,ineligible_records_identical=True,
        prior_intact_episode_checks=prior_checks,prior_intact_records_match_exactly=prior_checks==1024,
        main_rescue_intact_episode_checks=main_checks,main_rescue_intact_records_match_exactly=main_checks==1024,
        eligible_episodes_by_seed=eligible_counts,
        max_full_rescue_logit_error=max(r['checks']['full_rescue_logit_max_error'] for r in runs.values()))


def aggregate(runs,protocol,main_dir):
    assert (protocol['calibration_start'],protocol['calibration_stop_exclusive'],protocol['handoff_step'])==(450,600,600)
    assert {(row['reference'],row['intervention']) for row in protocol['exploratory_ablation_contrasts']}=={(a,b) for _,a,b,_ in CONTRASTS}
    validation=validate(runs,main_dir)
    estimates={metric:{variant:estimate([runs[s]['variants'][variant][metric] for s in SEEDS]) for variant in VARIANTS}
               for metric in ('post_handoff_score','mean_score','eligible_mean_score')}
    comparisons=contrasts(estimates['post_handoff_score'],CONTRASTS)
    full_score={name:dict(paired(estimates['mean_score'][a]['values'],estimates['mean_score'][b]['values']),label=label)
                for name,a,b,label in CONTRASTS}
    fields=('mean_patch_l2','units_outside_unit_range_fraction','samples_outside_unit_range_fraction','maximum_absolute_actor_hidden')
    diagnostics={variant:dict(estimates={field:estimate([runs[s]['patch_diagnostics'][variant][field] for s in SEEDS]) for field in fields},
                    by_seed={str(s):runs[s]['patch_diagnostics'][variant] for s in SEEDS}) for variant in VARIANTS}
    return dict(protocol=protocol,validation=validation,estimates=estimates,exploratory_post_handoff=comparisons,
        descriptive_full_score=full_score,patch_diagnostics=diagnostics,calibration=aggregate_calibration(runs),
        future_offset_validation=aggregate_future_validation(runs),
        checkpoint_sha256={str(s):runs[s]['checkpoint_sha256'] for s in SEEDS},
        statistics='Exploratory supplement added after observing the first rescue seed, before full rescue results. The five post-handoff score contrasts form one Holm family, corrected separately for paired t and exhaustive two-sided sign-flip tests. Eight training-seed means are the replicates. CIs are marginal paired 95% t intervals; figures use SEM. Full scores and patch diagnostics are descriptive.',
        intervention='Subtract only the frozen centered predictive offset delta_e minus grand_delta, estimated as mixed-goal minus ordinary-goal hidden activity during the past common prefix. Underlying current and goal inputs remain predictive. This is not subtraction of the whole episode mean.',
        limitations=[
            'The first four arms change only actor input; feedback arms additionally modify mixed-shadow h for the next recurrent step, while c and the grid shadow remain unchanged.',
            'Correct and wrong offsets have matched norms. Harm from both without a clear difference does not establish episode-specific information use.',
            'The fixed offset is an estimate from a 150-step prefix and may fail to represent later changing goal and history effects.',
            'Subtracting the predictive contribution can move hidden activity outside its natural [-1,1] range; observed violations are reported without clipping.',
            'This identifies effects within PGC-trained models, not the cause of their training advantage over separate grid-trained models.',
            'No future hidden states or rewards are used to estimate the offset. Other contemporary batch episodes enter the diagnostic shared/wrong controls.',
        ])


def aggregate_future_validation(runs):
    """Average batch-level diagnostics; never pool energies to redefine R²."""
    fields=('future_mean_energy_fraction','future_episode_mean_energy_fraction',
            'future_dynamic_mean_squared_l2','future_constant_mean_squared_l2',
            'future_grand_delta_l2','past_grand_delta_l2')
    spaces=('centered_hidden','centered_actor_logits')
    comparisons=('cosine','mean_episode_cosine','prediction_r2','squared_error',
                 'target_energy','prediction_energy','nonzero_episode_pairs')
    def weighted(values,weights):
        good=np.asarray([value is not None and np.isfinite(value) for value in values])
        return float(np.average(np.asarray(values,dtype=float)[good],weights=weights[good])) if good.any() else None
    def nullable_estimate(values):
        finite=np.asarray([value for value in values if value is not None and np.isfinite(value)],float)
        return dict(values=values,n_valid_seeds=len(finite),mean=float(finite.mean()) if len(finite) else None,
                    sem=float(finite.std(ddof=1)/np.sqrt(len(finite))) if len(finite)>1 else None)
    by_seed={}
    for seed,run in runs.items():
        batches=[batch['future_validation'] for batch in run['calibration']['batches']]
        weights=np.asarray([batch['eligible_count'] for batch in batches],float)
        record={field:weighted([batch[field] for batch in batches],weights) for field in fields}
        record['eligible_counts']=weights.astype(int).tolist()
        record['batches']=batches
        for space in spaces:
            record[space]={field:weighted([batch[space][field] for batch in batches],weights) for field in comparisons}
            record[space]['valid_batches']={field:sum(batch[space][field] is not None for batch in batches) for field in comparisons}
        by_seed[str(seed)]=record
    estimates={field:nullable_estimate([by_seed[str(seed)][field] for seed in SEEDS]) for field in fields}
    for space in spaces:
        estimates[space]={field:nullable_estimate([by_seed[str(seed)][space][field] for seed in SEEDS]) for field in comparisons}
    return dict(by_seed=by_seed,estimates=estimates,
        scope='Post-episode intact-branch diagnostic only; these future values never enter the policy, offset estimates, or eligibility. Within each seed, batch metrics are averaged using eligible counts; then seeds receive equal weight. R² is the mean of batch-level 1-SSE/target-energy values, not pooled R². Past and future episode offsets are separately centered over the same eligible episodes. No fitting or rescaling. Undefined zero-energy metrics are omitted from their average, with valid counts retained. All quantities are descriptive.')


def figure(result,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
    scores=result['estimates']['post_handoff_score']
    upper=max(r['mean']+r['sem'] for r in scores.values())*1.18
    for i,name in enumerate(VARIANTS):
        row=scores[name];color='#D87532' if 'wrong' not in name else '#9DA5AC'
        axes[0].barh(i,row['mean'],height=.5,color=color)
        axes[0].errorbar(row['mean'],i,xerr=row['sem'],fmt='none',color='#253442',capsize=3,lw=1)
        axes[0].text(row['mean']+row['sem']+upper*.015,i,f'{row["mean"]:.1f}',va='center')
    axes[0].axvline(scores['intact']['mean'],color='#D87532',ls=':',lw=1)
    axes[0].set(yticks=range(len(VARIANTS)),yticklabels=[LABELS[v] for v in VARIANTS],
                ylim=(len(VARIANTS)-.5,-.5),xlim=(0,upper),xlabel='Final 50-second score (mean ± SEM)',
                title='Subtracting a predictive episode offset')
    axes[0].spines[['left','bottom']].set_visible(False);axes[0].tick_params(axis='y',length=0)
    for i,(name,_,_,label) in enumerate(CONTRASTS):
        row=result['exploratory_post_handoff'][name];lo,hi=row['ci95'];mean=row['difference']
        axes[1].scatter(row['differences'],i+np.linspace(-.1,.1,8),s=18,color='#D87532',alpha=.35)
        axes[1].errorbar(mean,i,xerr=np.asarray([[mean-lo],[hi-mean]]),fmt='o',color='#D87532',capsize=3,lw=1.5)
    axes[1].axvline(0,color='#6F7882',lw=1)
    axes[1].set(yticks=range(len(CONTRASTS)),yticklabels=[c[3] for c in CONTRASTS],ylim=(len(CONTRASTS)-.5,-.5),
                xlabel='Score difference (95% paired interval)',title='Does the correct subtraction hurt more?')
    fig.supxlabel('Eight frozen PGC-model seeds. Eligible episodes only. Current and goal encoding remain predictive. All comparisons exploratory.',fontsize=10)
    fig.savefig(out/'episode_offset_ablation.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'episode_offset_ablation.pdf',bbox_inches='tight')
    plt.close(fig)


def report(result,out):
    comparisons=result['exploratory_post_handoff']
    def negative_supported(name):
        row=comparisons[name]
        return row['difference']<0 and row['paired_t_p_holm']<.05 and row['exact_signflip_p_holm']<.05
    if negative_supported('correct_minus_intact') and negative_supported('correct_minus_wrong'):
        headline='**Subtracting the correct predictive episode offset hurts navigation, and hurts more than subtracting a norm-matched wrong offset.** Both comparisons pass the two corrected tests. This supports a useful episode-specific contribution under this intervention.'
    elif negative_supported('correct_minus_intact'):
        headline='**Subtracting the correct predictive episode offset hurts navigation, but its disadvantage against the matched wrong-offset control is not established by both corrected tests.** Generic perturbation effects remain an alternative.'
    else:
        headline='**In the actor-only test, subtracting the correct offset was not reliably more harmful than subtracting a same-sized wrong offset.** This does not demonstrate an episode-specific contribution to navigation; uncertainty is not proof that all episode-related activity is irrelevant.'
    scores=result['estimates']['post_handoff_score']
    feedback_summary=(f'Actor-only scores are **{scores["remove_episode_mean"]["mean"]:.2f}** after correct subtraction and '
        f'**{scores["remove_wrong_episode_mean"]["mean"]:.2f}** after wrong subtraction, versus **{scores["intact"]["mean"]:.2f}** intact. '
        f'Feeding correct subtraction back through the RNN lowers score to **{scores["remove_episode_mean_feedback"]["mean"]:.2f}**; '
        f'wrong-offset feedback scores **{scores["remove_wrong_episode_mean_feedback"]["mean"]:.2f}**. '
        '**Correct-offset feedback hurts relative to intact**, including after both statistical corrections, but its extra harm compared with the wrong-offset control remains uncertain.')
    table=['| Exploratory comparison | Score difference | 95% interval | Holm t p | Holm exact p |','|---|---:|---:|---:|---:|']
    for row in comparisons.values():
        table.append(f'| {row["label"]} | {row["difference"]:+.2f} | [{row["ci95"][0]:+.2f}, {row["ci95"][1]:+.2f}] | {row["paired_t_p_holm"]:.4g} | {row["exact_signflip_p_holm"]:.4g} |')
    diagnostics=['| Intervention | Mean patch L2 | Hidden units outside [−1,1] | Samples with any outside unit | Largest observed abs(h) |',
                 '|---|---:|---:|---:|---:|']
    for name in VARIANTS:
        d=result['patch_diagnostics'][name]
        e=d['estimates']
        maximum=max(v['maximum_absolute_actor_hidden'] for v in d['by_seed'].values())
        diagnostics.append(f'| {LABELS[name]} | {e["mean_patch_l2"]["mean"]:.3f} | {100*e["units_outside_unit_range_fraction"]["mean"]:.3f}% | {100*e["samples_outside_unit_range_fraction"]["mean"]:.2f}% | {maximum:.3f} |')
    future=result['future_offset_validation']['estimates']
    def fmt(value):return f'{value:.3f}' if value is not None else 'undefined'
    future_table=['| Past offset predicting future offset | Cosine similarity | Prediction R² |','|---|---:|---:|']
    for space,label in [('centered_hidden','Centered hidden activity'),('centered_actor_logits','Centered action logits')]:
        future_table.append(f'| {label} | {fmt(future[space]["cosine"]["mean"])} | {fmt(future[space]["prediction_r2"]["mean"])} |')
    text=f'''# Subtracting the predictive episode offset

{headline}

{feedback_summary}

![Post-handoff ablation scores and paired differences](episode_offset_ablation.png)

The plotted outcome is score during the final **50 seconds**, among episodes eligible before the intervention. It is not the earlier 90-second full-episode score. This is an exploratory necessity-oriented complement to the [rescue experiment](../README.md), added after inspecting the first rescue seed and before full rescue results.

## What was subtracted

All branches follow the same intact policy for the first 600 steps. For episodes already knowing the goal at step 450, mixed-goal and ordinary-goal shadow states experience the same prefix. Their average hidden difference over steps 450–599 gives a predictive contribution for that episode.

We subtract its **episode-specific deviation from the batch's shared mean**. We do not subtract the whole episode-average activity pattern. The patch is frozen before any post-handoff activity or reward is observed.

- **Correct offset:** subtract that episode's centered predictive contribution from the original mixed hidden state at the actor readout.
- **Wrong offset:** subtract another eligible episode's centered contribution, rescaled to exactly the same norm.
- **PCA offset:** subtract the correct contribution only along the ten episode-mean directions learned from an independent earlier replay.
- **Feedback:** additionally pass the subtracted mixed h into the next recurrent step. Cell state c and the ordinary-goal shadow are untouched.

Current and saved goal inputs remain predictive. The non-feedback branches change only the actor's input. The feedback branches also alter the evolving recurrence and must be interpreted separately.

## Exploratory comparisons

{chr(10).join(table)}

The five comparisons form one Holm family, corrected separately for paired t and exhaustive sign-flip tests. Eight training seeds are the statistical replicates. Negative correct-minus-wrong differences mean the correct subtraction is more harmful. If both offsets hurt similarly, that does not demonstrate episode-specific information use.

## Patch size and hidden-state range

{chr(10).join(diagnostics)}

Fractions and patch norms are means of eight seed means; the final column is the largest value across seeds. Natural LSTM hidden outputs lie in [−1,1]. Patches are not clipped. Exceeding that range, especially with recurrent feedback, is an intervention side effect and limits mechanistic interpretation. Raw per-seed diagnostics are retained in [summary.json](summary.json).

This tests an estimated predictive contribution in frozen PGC-trained networks. It does not establish that every episode offset is goal memory, that all episode information is necessary, or why PGC training outperformed grid-only training. A finite prefix estimate can also miss changing signals and later context interactions.

## Did the prefix estimate match the later offset?

{chr(10).join(future_table)}

After the episode ended, we compared the frozen prefix estimate with the intact branch's actual mean mixed-minus-grid difference during steps 600–1349. Past and future offsets were separately centered across the same eligible episodes. This diagnostic never influenced actions, offsets, or eligibility.

R² is `1 − squared error / future offset energy`, with no fitting or rescaling; it can be negative. Each seed averages its two batch metrics using eligible counts, then the eight seeds receive equal weight. These are **averaged batch R² values, not pooled R²**. Poor agreement, especially in action-logit space, limits what failed constant-offset rescues or ablations can establish. Full energy fractions and raw batch diagnostics are retained in the summary.

The future mean within each episode accounts for **{100*future['future_mean_energy_fraction']['mean']:.4f}%** of the **hidden mixed-minus-grid signal energy** (`future_mean_energy_fraction = {future['future_mean_energy_fraction']['mean']:.10f}`). After removing the shared mean across episodes, the episode-specific means account for **{100*future['future_episode_mean_energy_fraction']['mean']:.4f}%** (`future_episode_mean_energy_fraction = {future['future_episode_mean_energy_fraction']['mean']:.10f}`). The denominator is the mean squared size of the changing hidden-state difference over steps 600–1349 along the intact trajectory. These are **not percentages of total PCA variance or navigation performance**; they are descriptive averages of batch energy fractions.

## Checks and reproduction

Eligibility, first-goal times, and prefix scores match exactly across variants. Ineligible episodes are unchanged. Intact episode records are compared with both the original code-swap evaluation and the main rescue run. Wrong controls require a derangement and matched norms. Calibration uses only the past prefix.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_episode_offset_rescue.py --device mps --episodes 128 --n-envs 64 \\
  --variants intact remove_episode_mean remove_wrong_episode_mean remove_between_episode_mean \\
  remove_episode_mean_feedback remove_wrong_episode_mean_feedback \\
  --out results/lstm_episode_offset_rescue_1B/ablation
python scripts/summarize_lstm_episode_offset_ablation.py
```

The first command runs the six ablation conditions for all eight seeds using the original local checkpoints; the second summarizes completed runs. Existing completed seed files are reused. The matching main rescue files are also required to check the intact baseline. [Protocol](protocol.json) · [Numerical results](summary.json) · [Main rescue test](../README.md)
'''
    (out/'README.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--rescue-dir',type=Path,default=MAIN)
    args=parser.parse_args()
    paths={s:args.out/f'pgc33_s{s}.json' for s in SEEDS}
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing:parser.error('All eight ablation files are required: '+', '.join(missing))
    missing_rescue=[str(args.rescue_dir/f'pgc33_s{s}.json') for s in SEEDS if not (args.rescue_dir/f'pgc33_s{s}.json').exists()]
    if missing_rescue:parser.error('Matching main rescue records are required: '+', '.join(missing_rescue))
    runs={s:json.loads(path.read_text()) for s,path in paths.items()}
    result=aggregate(runs,json.loads((args.out/'protocol.json').read_text()),args.rescue_dir)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    figure(result,args.out);report(result,args.out)
    for name,row in result['estimates']['post_handoff_score'].items():print(name,f'{row["mean"]:.4f} ± {row["sem"]:.4f}')
    for name,row in result['exploratory_post_handoff'].items():
        print(name,f'delta={row["difference"]:+.4f}, CI={row["ci95"]}, Holm t={row["paired_t_p_holm"]:.6g}, Holm exact={row["exact_signflip_p_holm"]:.6g}')
    print('Validation',json.dumps(result['validation']))


if __name__=='__main__':
    main()
