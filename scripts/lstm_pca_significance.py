"""Paired significance checks for the seven PCA metrics shown to the user.

Eight training-seed pairs are the units; activity samples are not replicates.
Both test families are reported without selecting whichever gives significance.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_variable_1B/pca_spectrum'
METRICS = [('pc3', 'Variance in first 3 PCs', 'percentage points', 100),
           ('pc10', 'Variance in first 10 PCs', 'percentage points', 100),
           ('p50', 'PCs for 50% variance', 'PCs', 1),
           ('p80', 'PCs for 80% variance', 'PCs', 1),
           ('p90', 'PCs for 90% variance', 'PCs', 1),
           ('p95', 'PCs for 95% variance', 'PCs', 1),
           ('p99', 'PCs for 99% variance', 'PCs', 1)]


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    corrected = np.empty_like(p)
    corrected[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected


def main():
    source = OUT / 'summary.json'
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = json.loads(source.read_text())
    by = {(r['condition'], r['seed']): r for r in summary['models'] if r['protocol'] == 'variable'}
    signs = np.array(list(itertools.product((-1, 1), repeat=8)))
    rows = []
    for key, label, units, scale in METRICS:
        group = 'cumulative_variance' if key.startswith('pc') else 'dimensions'
        a, b = [np.array([by[c, s][group][key] for s in range(1, 9)]) * scale
                for c in ('gc100', 'pgc33')]
        difference = b - a
        ttest = stats.ttest_rel(b, a, alternative='two-sided')
        ci = ttest.confidence_interval(.95)
        null = (signs * difference).mean(1)
        observed = difference.mean()
        exact = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
        # Independent library implementation of paired label exchanges.
        reference = stats.permutation_test((b, a), lambda x, y: np.mean(x-y),
            permutation_type='samples', vectorized=False, n_resamples=np.inf,
            alternative='two-sided').pvalue
        assert np.isclose(exact, reference, rtol=0, atol=1e-12)
        rows.append(dict(metric=key, label=label, units=units, n_pairs=8,
            grid=a.tolist(), pgc=b.tolist(), differences=difference.tolist(),
            mean_difference=float(observed), sem=float(stats.sem(difference)),
            unadjusted_t_ci95=[float(ci.low), float(ci.high)],
            t_statistic=float(ttest.statistic), t_df=float(ttest.df),
            paired_t_p=float(ttest.pvalue), exact_signflip_p=exact))
    for field in ('paired_t_p', 'exact_signflip_p'):
        for row, adjusted in zip(rows, holm([r[field] for r in rows])):
            row[field + '_holm7'] = float(adjusted)
    result = dict(protocol=dict(
        family=[r['metric'] for r in rows], family_size=7, alpha=.05,
        family_scope='The seven rows explicitly selected by the user; exploratory post-hoc family, not a preregistered endpoint. Other analyses in this project are not covered by this correction.',
        independent_units='Eight matched training seeds, not time samples; one shared diagnostic replay.',
        direction='PGC minus grid; every test two-sided.',
        parametric='Paired t test, df=7; small-sample validity relies on independent, approximately normal seed-wise differences.',
        exact='Mean-difference paired sign-flip test enumerating all 2^8=256 assignments; assumes paired exchangeability/sign symmetry under the null. No Monte Carlo pseudocount.',
        multiplicity='Holm across the same seven metrics, separately for each test method; controls within-family error even when metrics are dependent. Both methods reported.',
        intervals='95% paired-t confidence intervals are marginal, not simultaneous and not multiplicity adjusted.',
        exact_resolution='Smallest two-sided exact p is 2/256=.0078125. For this seven-test Holm family the smallest possible adjusted p is .0546875, just above .05.',
        interpretation='Inference concerns PCA variance concentration under this replay, not gridness, topology, stability, or causal performance benefit.'),
        source=str(source.relative_to(ROOT)), source_sha256=fingerprint, results=rows)
    (OUT / 'significance.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Significance of the seven reported PCA differences', '',
        'Eight matched training seeds are the independent units. All tests are two-sided and the seven selected rows form one exploratory family. Differences are PGC minus grid. Holm correction is applied separately to both test methods; neither method is selected according to its result.', '',
        '| Metric | Mean difference | Marginal 95% t CI | Paired t p | Holm t p | Exact sign-flip p | Holm exact p |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lo, hi = r['unadjusted_t_ci95']
        lines.append(f'| {r["label"]} | {r["mean_difference"]:+.3f} {r["units"]} | [{lo:.3f}, {hi:.3f}] | '
            f'{r["paired_t_p"]:.6g} | {r["paired_t_p_holm7"]:.6g} | {r["exact_signflip_p"]:.6g} | {r["exact_signflip_p_holm7"]:.6g} |')
    lines += ['',
        'The paired-t analysis rejects equality for the first five rows after Holm correction. The exact sign-flip analysis rejects none after correction; its four smallest adjusted p-values are 0.0546875. This reflects the coarse resolution of only eight pairs: the smallest attainable two-sided exact p is 2/256, which becomes 7 × 2/256 = 0.0546875 at the first Holm step. This is not evidence that the effects are zero.', '',
        'Paired t tests rely on independent, approximately normal seed differences for small-sample calibration. Exact sign-flip tests instead require within-pair exchangeability/sign symmetry under the null; they are not assumption-free. The confidence intervals above are unadjusted and should not be read as simultaneous intervals for all seven quantities.', '',
        'These tests were requested after the spectra were inspected. They are exploratory, conditional on one shared replay, and do not establish toroidal structure, grid-cell tuning, dynamical stability, or why rewards differ. The correction does not cover every exploratory analysis performed elsewhere in the project.', '',
        'Methods: [SciPy paired t test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [SciPy paired permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [Holm adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).', '',
        'Reproduce: `python scripts/lstm_pca_significance.py`', '']
    (OUT / 'significance.md').write_text('\n'.join(lines))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fingerprint
    for r in rows:
        print(f'{r["metric"]}: delta={r["mean_difference"]:+.4f}, CI={r["unadjusted_t_ci95"]}, '
              f't p={r["paired_t_p"]:.8g}, Holm t={r["paired_t_p_holm7"]:.8g}; '
              f'exact={r["exact_signflip_p"]:.8g}, Holm exact={r["exact_signflip_p_holm7"]:.8g}')


if __name__ == '__main__':
    main()
