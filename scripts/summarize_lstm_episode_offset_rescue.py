"""Summarize frozen-PGC episode-offset rescues after a common prefix.

The primary outcome is score during the final 50 seconds, among episodes
eligible at calibration. Fixed-offset specificity tests are separate from
exploratory dynamic and recurrent-feedback interventions.
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_episode_offset_rescue_1B'
SEEDS = tuple(range(1, 9))
VARIANTS = ('intact', 'goal_grid', 'global_mean', 'episode_mean',
            'wrong_episode_mean', 'between_episode_mean', 'between_dynamic',
            'random_dynamic', 'episode_mean_feedback', 'wrong_mean_feedback')
LABELS = dict(intact='Intact', goal_grid='Ordinary goal code',
    global_mean='Shared offset', episode_mean='Correct episode offset',
    wrong_episode_mean='Wrong episode offset',
    between_episode_mean='Mean: episode directions',
    between_dynamic='Changing: episode directions',
    random_dynamic='Changing: random directions',
    episode_mean_feedback='Correct offset + feedback',
    wrong_mean_feedback='Wrong offset + feedback')
# All differences are intervention minus reference.
PRIMARY = (
    ('episode_minus_global', 'global_mean', 'episode_mean', 'Correct episode − shared mean'),
    ('episode_minus_wrong', 'wrong_episode_mean', 'episode_mean', 'Correct episode − wrong episode'),
    ('between_mean_minus_global', 'global_mean', 'between_episode_mean', 'PCA episode mean − shared mean'),
)
EXPLORATORY = (
    ('goal_grid_minus_intact', 'intact', 'goal_grid', 'Goal substitution − intact'),
    ('between_dynamic_minus_goal_grid', 'goal_grid', 'between_dynamic', 'Dynamic PCA rescue − substitution'),
    ('between_dynamic_minus_random', 'random_dynamic', 'between_dynamic', 'Dynamic PCA rescue − random control'),
    ('feedback_minus_goal_grid', 'goal_grid', 'episode_mean_feedback', 'Mean feedback − substitution'),
    ('feedback_minus_wrong', 'wrong_mean_feedback', 'episode_mean_feedback', 'Correct feedback − wrong feedback'),
)


def estimate(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (8,) and np.isfinite(values).all()
    return dict(values=values.tolist(), n=8, mean=float(values.mean()), sem=float(stats.sem(values)))


def paired(reference, intervention):
    a, b = np.asarray(reference, float), np.asarray(intervention, float)
    difference = b - a
    se = float(stats.sem(difference))
    ci = stats.t.interval(.95, len(a)-1, loc=difference.mean(), scale=se) if se else (difference.mean(), difference.mean())
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(a))))
    return dict(reference_values=a.tolist(), intervention_values=b.tolist(),
        differences=difference.tolist(), difference=float(difference.mean()),
        ci95=list(map(float, ci)), paired_t_p=float(stats.ttest_rel(b, a).pvalue) if se else (1. if difference.mean()==0 else 0.),
        exact_signflip_p=float(np.mean(np.abs((signs*difference).mean(1)) >= abs(difference.mean())-1e-12)),
        positive_seeds=int((difference>0).sum()), negative_seeds=int((difference<0).sum()))


def contrasts(estimates, specifications):
    result = {}
    for name, reference, intervention, label in specifications:
        result[name] = paired(estimates[reference]['values'], estimates[intervention]['values'])
        result[name].update(reference=reference, intervention=intervention, label=label)
    for field in ('paired_t_p', 'exact_signflip_p'):
        running = 0.
        for i, name in enumerate(sorted(result, key=lambda name: result[name][field])):
            running = max(running, min(1., (len(result)-i)*result[name][field]))
            result[name][field+'_holm'] = running
    return result


def keyed(rows):
    result = {(r['slot'], r['episode']): r for r in rows}
    assert len(result) == len(rows), 'Duplicate episode keys'
    return result


def validate(runs):
    first_checks, prefix_checks, prior_checks = 0, 0, 0
    prefix_means = {}
    for seed, run in runs.items():
        assert (run['seed'], run['condition'], run['checkpoint_frames']) == (seed, 'pgc33', 1_000_000_000)
        assert (run['episodes'], run['n_envs'], run['eval_seed']) == (128, 64, 737373)
        assert set(run['variants']) == set(VARIANTS)
        assert run['checkpoint_sha256']
        assert run['basis']['rank'] == 10
        checks = run['checks']
        assert checks['prefix_exact'] and checks['first_goal_records_identical'] and checks['intact_baseline_match']
        assert checks['full_rescue_logit_max_error'] < 2e-4
        baseline = keyed(run['variants']['intact']['rows'])
        eligible = {key for key, row in baseline.items() if row['eligible']}
        assert len(baseline) == 128 and len(eligible) > 1
        prefix_means[str(seed)] = {}
        for variant in VARIANTS:
            summary = run['variants'][variant]
            rows = keyed(summary['rows'])
            assert rows.keys() == baseline.keys()
            assert {key for key, row in rows.items() if row['eligible']} == eligible
            assert summary['episodes'] == 128
            assert summary['eligible_episodes'] == len(eligible)
            assert np.isclose(summary['mean_score'], np.mean([r['ret'] for r in rows.values()]), atol=1e-9)
            assert np.isclose(summary['eligible_mean_score'], np.mean([rows[key]['ret'] for key in eligible]), atol=1e-9)
            assert np.isclose(summary['post_handoff_score'], np.mean([rows[key]['post_score'] for key in eligible]), atol=1e-9)
            for key, row in rows.items():
                assert row['eligible'] == (0 <= row['first_goal_t'] < 450)
                assert np.isclose(row['post_score'] + row['prefix_score'], row['ret'], atol=1e-8)
                assert row['first_goal_t'] == baseline[key]['first_goal_t'], (seed, variant, key, 'first goal')
                assert row['prefix_score'] == baseline[key]['prefix_score'], (seed, variant, key, 'prefix score')
                if not row['eligible']:
                    assert row == baseline[key], (seed, variant, key, 'ineligible episode')
                if variant != 'intact':
                    first_checks += 1
                    prefix_checks += 1
            prefix_means[str(seed)][variant] = dict(
                all_episode_mean=float(np.mean([r['prefix_score'] for r in rows.values()])),
                eligible_mean=float(np.mean([rows[key]['prefix_score'] for key in eligible])))
            assert np.isclose(summary['mean_prefix_score'], prefix_means[str(seed)][variant]['all_episode_mean'])
            assert np.isclose(summary['eligible_mean_prefix_score'], prefix_means[str(seed)][variant]['eligible_mean'])
        batches = run['calibration']['batches']
        assert len(batches) == 2
        assert {batch['episode'] for batch in batches} == {0, 1}
        assert sum(batch['eligible_count'] for batch in batches) == len(eligible)
        for batch in batches:
            assert (batch['calibration_start'], batch['calibration_stop_exclusive']) == (450, 600)
            assert batch['eligible_count'] == sum(row['eligible'] for row in baseline.values() if row['episode'] == batch['episode'])
            assert batch['eligible_count'] > 1 and batch['total_slots'] == 64
            assert batch['wrong_derangement_available']
            assert batch['wrong_zero_norm_count'] == 0
            assert batch['wrong_norm_max_error'] < 2e-4
        assert run['patch_diagnostics']['random_control']['zero_norm_failures'] == 0
        assert run['patch_diagnostics']['random_control']['norm_match_max_absolute_error'] < 2e-4
        old_path = ROOT / f'results/lstm_code_swap_1B/pgc33_s{seed}.json'
        if old_path.exists():
            old = json.loads(old_path.read_text())
            assert old['checkpoint_sha256'] == run['checkpoint_sha256']
            original = keyed(old['variants']['intact']['rows'])
            assert original.keys() == baseline.keys()
            for key, row in original.items():
                assert all(row[field] == baseline[key][field] for field in ('ret','n_goals','first_goal_t','path_eff','path_eff_n'))
                prior_checks += 1
    return dict(first_goal_episode_checks=first_checks, prefix_score_episode_checks=prefix_checks,
                first_goal_and_prefix_invariance=True, prior_intact_episode_checks=prior_checks,
                prior_intact_records_match_exactly=prior_checks == 1024,
                max_full_rescue_logit_error=max(r['checks']['full_rescue_logit_max_error'] for r in runs.values()),
                eligible_episodes_by_seed={str(s):runs[s]['variants']['intact']['eligible_episodes'] for s in SEEDS},
                prefix_means=prefix_means)


def aggregate_calibration(runs):
    fields = ('mean_delta_l2', 'grand_delta_l2', 'mean_episode_delta_l2', 'between_projection_energy_fraction',
              'mean_delta_energy_fraction', 'episode_delta_energy_fraction',
              'grand_centered_logit_energy', 'mean_episode_centered_logit_energy')
    by_seed = {}
    for seed, run in runs.items():
        batches = run['calibration']['batches']
        weights = np.asarray([batch['eligible_count'] for batch in batches], float)
        record = {}
        for field in fields:
            record[field] = float(np.average([batch[field] for batch in batches], weights=weights))
        for encoding in ('mixed','grid'):
            record[encoding] = {field:float(np.average([batch[encoding][field] for batch in batches], weights=weights))
                               for field in ('total_variance','between_episode_variance','between_episode_fraction')}
        by_seed[str(seed)] = record
    aggregates = {field:estimate([by_seed[str(seed)][field] for seed in SEEDS]) for field in fields}
    for encoding in ('mixed','grid'):
        aggregates[encoding] = {field:estimate([by_seed[str(seed)][encoding][field] for seed in SEEDS])
                                for field in ('total_variance','between_episode_variance','between_episode_fraction')}
    return dict(by_seed=by_seed, estimates=aggregates,
        scope='Descriptive eligible-count-weighted averages of two calibration batches per training seed. Batch covariance fractions are averaged, not recomputed from pooled samples. These are mixed-versus-grid goal inputs within the same PGC-trained network, not comparisons of separately trained populations.')


def aggregate(runs, protocol):
    assert (protocol['calibration_start'], protocol['calibration_stop_exclusive'], protocol['handoff_step']) == (450, 600, 600)
    assert {(r['reference'], r['intervention']) for r in protocol['primary_contrasts']} == {(a, b) for _, a, b, _ in PRIMARY}
    validation = validate(runs)
    estimates = {metric:{variant:estimate([runs[seed]['variants'][variant][metric] for seed in SEEDS]) for variant in VARIANTS}
                 for metric in ('post_handoff_score','mean_score','eligible_mean_score')}
    primary = contrasts(estimates['post_handoff_score'], PRIMARY)
    exploratory = contrasts(estimates['post_handoff_score'], EXPLORATORY)
    full_score_descriptive = {}
    for name, reference, intervention, label in PRIMARY + EXPLORATORY:
        full_score_descriptive[name] = paired(estimates['mean_score'][reference]['values'],estimates['mean_score'][intervention]['values'])
        full_score_descriptive[name].update(reference=reference,intervention=intervention,label=label,
                                         scope='Descriptive secondary endpoint; no additional significance claims.')
    return dict(protocol=protocol,validation=validation,estimates=estimates,
        primary_post_handoff=primary,exploratory_post_handoff=exploratory,
        descriptive_full_score=full_score_descriptive,calibration=aggregate_calibration(runs),
        patch_diagnostics={variant:{field:estimate([runs[seed]['patch_diagnostics'][variant][field] for seed in SEEDS])
                           for field in ('mean_patch_l2','units_outside_unit_range_fraction','samples_outside_unit_range_fraction','maximum_absolute_actor_hidden')}
                           for variant in VARIANTS},
        random_control_diagnostics={str(seed):runs[seed]['patch_diagnostics']['random_control'] for seed in SEEDS},
        checkpoint_sha256={str(seed):runs[seed]['checkpoint_sha256'] for seed in SEEDS},
        statistics='Eight training-seed means are the replicates. Primary endpoint: final 750-step (50-second) score among episodes eligible at calibration. Three primary paired contrasts and five exploratory contrasts form separate Holm families, each corrected separately for paired t and exact two-sided sign-flip tests. CIs are marginal 95% paired t intervals; plot errors are SEM. Full-episode score and calibration measurements are descriptive.',
        limitations=[
            'Only frozen PGC-trained networks are intervened on; this does not establish why their training outperformed grid-only training.',
            'Offsets are estimated from the same episode prefix, frozen before the handoff, and use no later observations or rewards.',
            'An episode-specific difference may reflect goal-by-context/history interactions; it is not necessarily a pure goal coordinate.',
            'The fixed-offset primary tests modify the actor input only; they do not replace recurrent h/c or isolate all future recurrent effects.',
            'Dynamic subspace rescue changes a time-varying signal and cannot by itself prove that constant episode-average shifts are causal.',
            'The feedback arms also alter the next recurrent h input while leaving c unchanged; they are distinct exploratory interventions.',
            'An effective rescue would demonstrate sufficiency under the tested intervention, not exclusive necessity or the full source of the PGC training advantage.',
            'A failed rescue can reflect an inaccurate or nontransferable constant-offset approximation and does not prove that all contextual activity is irrelevant.',
        ])


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def figures(result,out):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.5,3.1),layout='constrained')
    for i,(name,_,_,label) in enumerate(PRIMARY):
        row=result['primary_post_handoff'][name]
        color='#D87532'
        ax.scatter(row['differences'],i+np.linspace(-.12,.12,8),s=22,color=color,alpha=.4)
        lo,hi=row['ci95'];mean=row['difference']
        ax.errorbar(mean,i,xerr=np.asarray([[mean-lo],[hi-mean]]),fmt='o',color=color,capsize=4,ms=7,lw=2)
    ax.axvline(0,color='#6F7882',lw=1)
    ax.set(yticks=range(len(PRIMARY)),yticklabels=[row[3] for row in PRIMARY],
           ylim=(len(PRIMARY)-.5,-.5),xlabel='Post-handoff score difference (95% paired interval)',
           title='Does the correct episode offset help?')
    fig.supxlabel('Dots: eight training seeds. Positive values favor the episode-specific rescue.',fontsize=10)
    save(fig,out,'episode_offset_primary')
    names=[('intact','goal_grid','global_mean','episode_mean','wrong_episode_mean','between_episode_mean'),
           ('goal_grid','between_dynamic','random_dynamic','episode_mean_feedback','wrong_mean_feedback')]
    titles=['Fixed mean offsets: actor readout only','Changing signals and recurrent feedback']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    scores=result['estimates']['post_handoff_score']
    upper=max(r['mean']+r['sem'] for r in scores.values())*1.16
    for ax,variants,title in zip(axes,names,titles):
        for i,variant in enumerate(variants):
            row=scores[variant];color='#D87532' if variant not in ('goal_grid','wrong_episode_mean','wrong_mean_feedback','random_dynamic') else '#9DA5AC'
            ax.barh(i,row['mean'],height=.5,color=color)
            ax.errorbar(row['mean'],i,xerr=row['sem'],fmt='none',color='#253442',capsize=3,lw=1)
            ax.text(row['mean']+row['sem']+upper*.015,i,f'{row["mean"]:.1f}',va='center',fontsize=10)
        ax.axvline(scores['intact']['mean'],color='#D87532',lw=1,ls=':')
        ax.set(yticks=range(len(variants)),yticklabels=[LABELS[v] for v in variants],
               ylim=(len(variants)-.5,-.5),xlim=(0,upper),xlabel='Final 50-second score (mean ± SEM)',title=title)
        ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.supxlabel('Frozen PGC-trained models. Eight seeds; only episodes eligible before handoff. Dotted line: intact.',fontsize=10)
    save(fig,out,'episode_offset_conditions')


def report(result,out):
    estimates=result['estimates']['post_handoff_score']
    def score(name):return estimates[name]['mean']
    def table(family):
        lines=['| Comparison | Score difference | 95% interval | Holm t p | Holm exact p |',
               '|---|---:|---:|---:|---:|']
        for row in family.values():
            lines.append(f'| {row["label"]} | {row["difference"]:+.2f} | [{row["ci95"][0]:+.2f}, {row["ci95"][1]:+.2f}] | {row["paired_t_p_holm"]:.4g} | {row["exact_signflip_p_holm"]:.4g} |')
        return '\n'.join(lines)
    supported=[row['label'] for row in result['primary_post_handoff'].values()
               if row['difference']>0 and row['paired_t_p_holm']<.05 and row['exact_signflip_p_holm']<.05]
    conclusion=('The fixed-offset comparisons supported by both corrected tests are: **'+ '; '.join(supported)+'**.' if supported
                else '**Adding back the fixed episode offset did not demonstrate a navigation benefit.** The correct offset was not reliably better than a shared offset or a same-sized wrong-episode offset.')
    supplement=''
    supplement_path=out/'ablation/summary.json'
    if supplement_path.exists():
        subtraction=json.loads(supplement_path.read_text())
        contrast=subtraction['exploratory_post_handoff']['correct_minus_wrong']
        future=subtraction['future_offset_validation']['estimates']
        supplement=(f'The [subtraction test](ablation/README.md) also found an uncertain correct-versus-wrong effect: '
            f'{contrast["difference"]:+.2f} points, 95% interval [{contrast["ci95"][0]:+.2f}, {contrast["ci95"][1]:+.2f}]. '
            f'Its [past-to-future check](ablation/README.md#did-the-prefix-estimate-match-the-later-offset) found that the frozen prefix offset predicted the later mean poorly: '
            f'R² = {future["centered_hidden"]["prediction_r2"]["mean"]:.3f} in hidden activity and '
            f'{future["centered_actor_logits"]["prediction_r2"]["mean"]:.3f} in action logits. '
            'That limits what the failed fixed-offset rescue can establish: a short past average is an imperfect substitute for the changing predictive signal.\n')
    text=f'''# Do episode-specific activity offsets help navigation?

{conclusion}

![Paired fixed-offset rescue comparisons](episode_offset_primary.png)

The primary outcome is **score during the final 50 seconds**, among episodes eligible before the intervention. These numbers should not be compared directly with the earlier 90-second scores near 104. The intact condition scores **{score('intact'):.2f}** here, and ordinary goal-code substitution scores **{score('goal_grid'):.2f}**.

Restoring the correct fixed offset scores **{score('episode_mean'):.2f}**, and restoring a same-sized wrong offset scores **{score('wrong_episode_mean'):.2f}**. Neither recovers the lost score in its mean outcome; their difference remains uncertain.

## The simple test

Every branch first follows the same intact policy for 600 steps. During steps 450–599, after the goal is already known, two shadow RNN states receive the same experience: one gets the predictive goal code and the other gets ordinary goal code. Their average hidden-state difference estimates an offset for that episode. The estimate is frozen before any later action or reward is observed.

After step 600, the main fixed-offset interventions feed the actor ordinary-goal hidden activity plus:

- **Shared mean:** the average offset across eligible episodes.
- **Correct episode mean:** that episode's own offset.
- **Wrong episode mean:** another eligible episode's deviation from the shared mean, rescaled to match the correct deviation's size.
- **PCA episode mean:** the shared mean plus the correct episode deviation projected onto ten episode-mean directions learned from an independent earlier replay.

Live population inputs stay predictive. These fixed-offset arms change only the actor input; their recurrent h/c states continue under ordinary goal input. The correct-versus-shared and correct-versus-wrong controls distinguish episode-specific content from a generic offset or perturbation size.

## Primary comparisons

{table(result['primary_post_handoff'])}

Eight paired training seeds are the replicates. The three primary comparisons form one Holm correction family. Positive differences favor the episode-specific rescue. Individual dots show seed differences; intervals are marginal 95% paired t intervals.

## Dynamic and recurrent controls

![All post-handoff score conditions](episode_offset_conditions.png)

The **dynamic PCA rescue** adds the current mixed-versus-grid hidden difference only along the independently learned episode-mean subspace. Its control uses a rank-matched random subspace outside that span, with patch norm matched at each step. These patches vary with time: success would identify useful information in the subspace, not prove that constant episode offsets are sufficient.

The **feedback** arms additionally feed the corrected h back into the next ordinary-goal recurrent update; c is unchanged. They test a different mechanism from actor-only restoration.

{table(result['exploratory_post_handoff'])}

These five comparisons form a separate exploratory Holm family. All ten variants and full-episode scores remain in [summary.json](summary.json).

Restoring the changing signal gives a small mean improvement, but its advantage over the same-sized random control remains uncertain. Feeding fixed offsets back into recurrence scores **{score('episode_mean_feedback'):.2f}** with the correct offset and **{score('wrong_mean_feedback'):.2f}** with the wrong one. Both means are below the ordinary-goal baseline; the correct-versus-wrong comparison does not establish an episode-specific effect.

## Interpretation limits and checks

The test isolates information produced by changing the goal encoding within the same frozen PGC model. It does not equate every episode offset with a pure goal coordinate: goal effects can interact with visual context and history. It also does not establish why PGC training outperformed grid-only training.

During the matched calibration prefix, episode averages account for **{100*result['calibration']['estimates']['mixed']['between_episode_fraction']['mean']:.2f}%** of hidden variance with predictive goal input and **{100*result['calibration']['estimates']['grid']['between_episode_fraction']['mean']:.2f}%** with ordinary goal input. Both retain large episode-associated variation. These are descriptive batch averages from the **same PGC-trained RNN with unchanged predictive live input**; they are not the earlier 30.2% versus 21.0% comparison of separately trained populations on a different replay.

A successful constant-offset rescue would show sufficiency under this intervention. A failed rescue would not rule out changing goal signals, recurrent interactions, or context features inadequately estimated by the short prefix. Dynamic rescue and feedback must be interpreted separately from the fixed-offset primary comparisons.

{supplement}

All branches have identical first-goal and prefix-score records. Intact records reproduce the earlier evaluation exactly. Wrong-offset controls require a genuine derangement within each eligible batch and matched norms. Calibration and independent-basis provenance are retained in the per-seed files; all offsets use only prefix information.

Activation patches are not clipped. The summary retains how often patched h leaves its natural [−1, +1] range, the maximum magnitude, and random-control rescaling. Feedback effects therefore include a deliberately altered recurrent state and should not be interpreted as ordinary operation of an unmodified LSTM.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_episode_offset_rescue.py --device mps --episodes 128 --n-envs 64
python scripts/summarize_lstm_episode_offset_rescue.py
```

The first command runs all eight seeds using the original local checkpoints; the second summarizes completed runs. Existing completed seed files are reused. See [protocol.json](protocol.json), [full statistics](summary.json), the [subtraction and future-offset checks](ablation/README.md), and the [eigenvector analysis](../lstm_eigenspectra_1B/README.md).
'''
    (out/'README.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    paths={seed:args.out/f'pgc33_s{seed}.json' for seed in SEEDS}
    missing=[str(path) for path in paths.values() if not path.exists()]
    if missing:parser.error('All eight model files are required: '+', '.join(missing))
    runs={seed:json.loads(path.read_text()) for seed,path in paths.items()}
    protocol=json.loads((args.out/'protocol.json').read_text())
    result=aggregate(runs,protocol)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    figures(result,args.out)
    report(result,args.out)
    for name,row in result['estimates']['post_handoff_score'].items():print(name,f'{row["mean"]:.4f} ± {row["sem"]:.4f}')
    for family in ('primary_post_handoff','exploratory_post_handoff'):
        print(family)
        for name,row in result[family].items():
            print(name,f'delta={row["difference"]:+.4f}, CI={row["ci95"]}, Holm t={row["paired_t_p_holm"]:.6g}, Holm exact={row["exact_signflip_p_holm"]:.6g}')
    print('Validation',json.dumps(result['validation']))


if __name__=='__main__':
    main()
