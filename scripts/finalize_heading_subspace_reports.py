"""Readable complete-cohort reports without modifying frozen evaluations."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/ring_heading'
SEEDS = list(range(2001, 2017))


def mean(rows, key): return float(np.mean([r[key] for r in rows]))


def render_figure(rows, variants, labels, output, footer):
    """Reporting-only labels distinguish decoded coverage from full-state topology."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette = {'native':'#D87532', 'ordinary':'#8E98A3', 'remove_direction':'#8C64B4',
               'restore_direction':'#317BB5', 'random_restore':'#D1D5DA',
               'mean_only_restore':'#7FAD91', 'mean_preserving_removal':'#8C64B4'}
    colors = [palette[v] for v in variants]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
    for ax, prefix, scale, title, ylabel in zip(axes, ['heading_', 'navigation_', 'coverage_'], [1, 100, 100],
            ['Heading at new places', 'Navigation at new places', 'Decoded heading coverage'],
            ['Error (degrees)', 'Reaching the goal (%)', 'Heading circle covered (%)']):
        def value(row, variant):
            if variant == 'random_restore':
                return np.mean([value(row, f'sham{j}_restore') for j in range(3)])
            return row[f'anchor_{variant}_circle_coverage'] if prefix == 'coverage_' else row[prefix + variant]
        data = [np.array([value(r, v) for r in rows]) * scale for v in variants]
        ax.bar(range(len(variants)), [v.mean() for v in data], color=colors, width=.72, capsize=3,
               yerr=[stats.sem(v) * stats.t.ppf(.975, len(v)-1) for v in data])
        for i, values in enumerate(data):
            ax.scatter(i + np.linspace(-.22, .22, len(values)), values, s=10, color='#44505C', alpha=.5, zorder=3)
        upper = max(95., max(float(v.max()) for v in data)+5,
                    max(float(v.mean()+stats.sem(v)*stats.t.ppf(.975,len(v)-1)) for v in data)+5) if prefix=='heading_' else 110.
        ax.set(title=title, ylabel=ylabel, xticks=range(len(variants)), xticklabels=labels, ylim=(0,upper))
        ax.tick_params(axis='x', labelsize=8); ax.spines[['top','right']].set_visible(False)
    fig.supxlabel(footer, fontsize=9)
    for extension in ('png','pdf','svg'): fig.savefig(output.with_suffix('.'+extension), dpi=200)
    plt.close(fig)


def tests_table(tests, scale=1., unit='°'):
    lines = ['| Paired comparison | Difference [95% interval] | Exact Holm p | Seeds with positive difference |',
             '|---|---:|---:|---:|']
    labels = {'native_benefit': 'Native benefit over ordinary', 'removal_cost': 'Cost of removing the plane',
              'restoration_benefit': 'Full-plane restoration benefit over ordinary',
              'specificity_over_random': 'Full-plane restoration benefit over random',
              'mean_only_benefit': 'Mean-only benefit over ordinary',
              'full_plane_over_mean_only': 'Full-plane benefit over mean-only',
              'mean_preserving_removal_cost': 'Cost of removing variation while retaining the mean'}
    for key, test in tests.items():
        lo, hi = np.array(test['ci95']) * scale
        lines.append(f"| {labels[key]} | {test['mean_difference']*scale:.2f}{unit} [{lo:.2f}, {hi:.2f}] | {test['holm_exact_p']:.6g} | {test['positive_seeds']}/16 |")
    return '\n'.join(lines)


def main():
    subdir, tondir = BASE / 'subspace_confirmation', BASE / 'tonic_direction_confirmation'
    sub = json.loads((subdir / 'summary.json').read_text())
    ton = json.loads((tondir / 'summary.json').read_text())
    for data in (sub, ton):
        assert data['complete'] is True and data['completed_seeds'] == SEEDS
        assert [r['seed'] for r in data['rows']] == SEEDS
    validation = {'all_16_seeds_in_both': True, 'shared_heading_max_error': 0., 'shared_navigation_success_max_error': 0.}
    for a, b in zip(sub['rows'], ton['rows']):
        for variant in ('native', 'ordinary', 'restore_direction'):
            validation['shared_heading_max_error'] = max(validation['shared_heading_max_error'], abs(a['heading_' + variant] - b['heading_' + variant]))
            validation['shared_navigation_success_max_error'] = max(validation['shared_navigation_success_max_error'], abs(a['navigation_' + variant] - b['navigation_' + variant]))
    assert validation['shared_heading_max_error'] < 1e-5
    assert validation['shared_navigation_success_max_error'] == 0.
    (tondir / 'cross_study_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    rows = sub['rows']
    table = ['| Drive entering the same frozen RNN | Heading error at new places | Navigation success | Static decoded heading coverage |',
             '|---|---:|---:|---:|']
    variants = [('native', 'Native'), ('ordinary', 'Ordinary'), ('remove_direction', 'Remove direction plane'),
                ('restore_direction', 'Restore direction plane')]
    for v, label in variants:
        table.append(f"| {label} | {mean(rows,'heading_'+v):.1f}° | {100*mean(rows,'navigation_'+v):.1f}% | {100*mean(rows,'anchor_'+v+'_circle_coverage'):.1f}% |")
    sham_heading = np.mean([[r[f'heading_sham{j}_restore'] for j in range(3)] for r in rows])
    sham_nav = np.mean([[r[f'navigation_sham{j}_restore'] for j in range(3)] for r in rows])
    sham_cover = np.mean([[r[f'anchor_sham{j}_restore_circle_coverage'] for j in range(3)] for r in rows])
    table.append(f'| Three random-plane restorations, averaged within model | {sham_heading:.1f}° | {100*sham_nav:.1f}% | {100*sham_cover:.1f}% |')
    text = f'''# The learned direction component: all 16 models

**Complete cohort: every planned predictive-model seed, 2001–2016, is included.** The pilot is excluded. Each dot below is one trained model, not one simulated episode.

![Direction-plane causal confirmation](direction_plane_confirmation.png)

{chr(10).join(table)}

We found two directions in each frozen encoder's drive associated with cosine and sine of measured travel direction. We fitted them using separate calibration positions, then removed or restored only the observed predictive-versus-ordinary difference inside that fixed plane. All recurrent weights stayed unchanged. The experiment supplied the paired native and counterfactual ordinary inputs; no true-heading label entered the intervention.

The plane contains {100*mean(rows,'projected_energy_fraction'):.1f}% of the predictive-versus-ordinary drive difference's energy at held-out positions, averaged over models. Its held-out direction-readout error is {mean(rows,'direction_readout_error_deg'):.1f}°. These are measurements of a useful rank-2 component, not a claim that the entire input difference is intrinsically two-dimensional. The three random-plane controls match preactivation-drive norm per input; their values are averaged within each trained model before comparisons.

Forced heading uses 256 matched trajectories per model, after four shared native initialization steps and excluding the first eight steps. Navigation uses 128 matched episodes per model, with each branch controlling its own physical path. Both functional endpoints above use a 10 m translation. Static decoded-heading coverage uses five fixed in-region contexts at 5 s after substitution of neighboring states initialized at headings ±10° apart; it accompanies phase/full-state perturbation measurements so collapse cannot masquerade as useful correction. Angular readout coverage is not a test of smooth full-state ring geometry.

The seed is the inferential unit. Positive differences below favor useful native/direction-plane input. Each endpoint has its own four-comparison Holm family, as fixed in the [protocol](PROTOCOL.md). Confidence intervals are nominal paired-t intervals; p values are exact two-sided paired sign-flip tests with Holm correction.

Heading comparisons (degrees):

{tests_table(sub['heading_tests'])}

Navigation comparisons (percentage points):

{tests_table(sub['navigation_tests'],100.,' pp')}

The [separate fixed-mean diagnostic](../tonic_direction_confirmation/README.md) tests whether the effect can be reproduced by a global constant offset. It leaves these tests and data unchanged. Whether the full recurrent state forms a smooth ring is a separate geometry question: coarse conditional curves can hide sharp changes in hidden dimensions even when decoded direction is smooth. The present result concerns useful direction information and input-supported correction; it does not require an unqualified global-ring interpretation.

Validation includes checkpoint-bound calibration artifacts, unchanged checkpoint hashes, native/ordinary full-state and independent navigation identities, deletion/residual equivalence, per-input sham norms, and zero intervention during blackouts. A [documented numerical-validation amendment](NUMERICAL_VALIDATION_AMENDMENT.md) handles float32 roundoff in seed 2012's unused residual-only alias; no scientific computation or contrast changes. Exact per-model source snapshots, [paired seed values](paired_seeds.csv), [full summary](summary.json), and the [source manifest](frozen_manifest.json) are archived here. This cohort contains the newly trained heading models, not the original navigation LSTMs or published internal-PGC path integrators.
'''
    (subdir / 'README.md').write_text(text)
    rows = ton['rows']
    table = ['| Drive entering the same frozen RNN | Heading error at new places | Navigation success | Static decoded heading coverage |',
             '|---|---:|---:|---:|']
    for v, label in [('native','Native'), ('ordinary','Ordinary'), ('restore_direction','Restore full direction plane'),
                     ('mean_only_restore','Restore only the fixed mean'), ('mean_preserving_removal','Keep the mean; remove variation')]:
        table.append(f"| {label} | {mean(rows,'heading_'+v):.1f}° | {100*mean(rows,'navigation_'+v):.1f}% | {100*mean(rows,'anchor_'+v+'_circle_coverage'):.1f}% |")
    energies = np.array([r['mean_projected_energy_fraction'] for r in rows]) * 100
    text = f'''# Fixed mean versus changing direction signal: all 16 models

**Complete additional diagnostic: all 16 planned predictive models are included.** This analysis is separate from the frozen original direction-plane comparison.

![Mean versus varying component](tonic_direction_confirmation.png)

{chr(10).join(table)}

We split each model's fitted direction-plane drive difference into its fixed independent-calibration mean and its variation around that mean. We tested restoring the mean alone and removing the variation while retaining the mean. The only online gate on the fixed mean is the observed spatial-availability flag, so it disappears during blackouts. No true direction or position is supplied to the intervention.

The fixed mean accounts for {energies.mean():.4f}% of projected-drive energy on average across models (range {energies.min():.4f}–{energies.max():.4f}%). This control addresses a simple global input offset. It does not remove every position-dependent average or match all nonlinear effects of fluctuations in drive magnitude. Variation within the fitted plane can also depend on position and other observed sensors.

The settings, episodes, calibration data, and projectors are identical to the original subspace assay. Shared native, ordinary, and full-plane restoration heading metrics agree to maximum error {validation['shared_heading_max_error']:.2g} degrees, and navigation success agrees exactly. All new blackout effects are zero. Frozen original source files and test families remain unchanged. Seed 2012 uses the same [validation-only roundoff amendment](../subspace_confirmation/NUMERICAL_VALIDATION_AMENDMENT.md) as the original subspace assay. [Cross-study validation](cross_study_validation.json).

Three additional exploratory contrasts were fixed for each endpoint in the [diagnostic protocol](PROTOCOL.md). Positive differences indicate benefit for the named source of input or cost of its removal. Intervals are nominal paired-t intervals; exact paired sign-flip p values are Holm-corrected separately within each three-test family.

Heading comparisons (degrees):

{tests_table(ton['heading_exploratory_tests'])}

Navigation comparisons (percentage points):

{tests_table(ton['navigation_exploratory_tests'],100.,' pp')}

The diagnostic was specified after the pilot and after six confirmation models had their original subspace analyses computed. It is a recorded mechanistic follow-up, not an externally preregistered primary study. [Paired seed values](paired_seeds.csv) · [Full summary](summary.json) · [Source manifest](source_manifest.json).
'''
    (tondir / 'README.md').write_text(text)
    render_figure(sub['rows'], ['native','ordinary','remove_direction','restore_direction','random_restore'],
        ['Native','Ordinary','Remove\ndirection','Restore\ndirection','Random\nrestore'],
        subdir/'direction_plane_confirmation', '16 frozen trained models · Dots: training seeds · 95% confidence intervals')
    render_figure(ton['rows'], ['native','ordinary','restore_direction','mean_only_restore','mean_preserving_removal'],
        ['Native','Ordinary','Full plane','Mean only','Remove\nvariation'],
        tondir/'tonic_direction_confirmation', '16 frozen trained models · Additional mean-versus-varying-drive diagnostic · 95% intervals')
    for folder in (subdir,tondir):
        manifest={'reporting_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'summary_sha256':hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
                  'data_and_tests_unchanged':True,'third_panel':'decoded heading coverage, not full-state ring topology'}
        (folder/'reporting_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'completed_models':16,'subspace_report':str(subdir/'README.md'),
                      'tonic_report':str(tondir/'README.md'),'cross_validation':validation},indent=2))


if __name__ == '__main__': main()
