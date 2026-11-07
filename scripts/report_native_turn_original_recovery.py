"""Saved-summary supplement for every original-resolution finite-recovery case.

No policy, recurrent, environment, or physical simulation imports/execution.
The unchanged resolution reporter has already rebuilt the underlying arrays.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_geometry_resolution'
OUT = BASE / 'original_recovery_supplement'
KEYS = ('context', 'prefix', 'kind', 'branch', 'regime', 'window_updates',
        'evolution_updates', 'observable', 'centered')
NAMES = ['hidden_only', 'cell_only'] + [f'random_full_{i:02d}' for i in range(16)] + [
    'full_radial', 'cell_radial', 'tangent']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def extract(source, report, validation):
    """Retain all tested pairs, metrics, densities, directions and visibility."""
    pairs = [p for p in report['pairs'] if p['status'] == 'recovery_failed']
    wanted = {tuple(p[k] for k in KEYS) for p in pairs}
    selected = [r for r in source['rows'] if r['normal_test_run']]
    assert {tuple(r[k] for k in KEYS) for r in selected} == wanted
    assert len(selected) == 2 * len(wanted)
    assert validation['complete'] and source['complete']
    rows, probes = [], []
    for r in selected:
        assert not r['source_exact_sham_of'] and r['resolved_geometry']
        assert r['observable'] == 'all_actor' and r['centered']
        assert r['window_updates'] == 3 and r['evolution_updates'] == 2
        assert set(r['recovery']) == {'512', '1024'}
        for density, rec in r['recovery'].items():
            follow = rec['bounded_followups'][-1] if rec['bounded_followups'] else None
            q = follow['result'] if follow else rec['nominal']
            ds = {d['name']: d for d in q['directions']}
            assert set(ds) == set(NAMES) and len(q['directions']) == 21
            assert q['physical_conclusive'] and q['hidden_bound_excess_count'] == 0
            assert q['physical_status'] == 'failed_recovery'
            assert q['tangent_nontrivial_fraction'] == q['final_tangent_nontrivial_fraction'] == 1
            assert q['leakage_count'] == 0
            identity = {k: r[k] for k in KEYS}
            identity.update(metric=r['metric'], density=int(density),
                            method=follow['method'] if follow else 'nominal')
            probes.append(dict(**identity, physical_conclusive=q['physical_conclusive'],
                normal_directions=q['normal_directions'],
                initial_tangent_visibility=q['tangent_nontrivial_fraction'],
                final_tangent_visibility=q['final_tangent_nontrivial_fraction'],
                normal_directions_passed=[d['name'] for d in q['directions'][:-1] if d['passes']],
                convergence=r['convergence'],
                common_mean_shift_over_radius=r['densities'][density]['common_mean_shift_over_radius'],
                invariance_p90=r['densities'][density]['residual_to_reference_curve_over_radius']['p90']))
            for name in NAMES:
                d = ds[name]
                # Tangent is a separately labeled total-displacement reference.
                stats = q['tangent_total_gain'] if name == 'tangent' else d['normal_gain']
                assert stats and all(math.isfinite(v) for v in stats.values())
                rows.append(dict(**identity, direction=name,
                    gain_type='total_tangent' if name == 'tangent' else 'off_curve_normal',
                    gain_median=stats['median'], gain_p10=stats['p10'], gain_p90=stats['p90'],
                    normal_over_tangent_median=None if name == 'tangent' else d['normal_over_tangent_gain']['median'],
                    normal_informative_fraction=d['normal_informative_fraction'],
                    informative_pair_fraction=d['informative_pair_fraction'],
                    leakage_count=d['factor_leakage_count'],
                    direction_passes=None if name == 'tangent' else d['passes']))
    assert len(rows) == len(selected) * 2 * 21
    return pairs, probes, rows


def range_values(rows, metric, names, column='gain_median', density=1024):
    vals = [r[column] for r in rows if r['metric'] == metric and r['density'] == density and r['direction'] in names]
    return [min(vals), max(vals)]


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(17, 9), sharex=True)
    cases = [(0, 'fb_B'), (0, 'fb_F'), (1, 'fb_B'), (1, 'fb_F')]
    colors = ['#1b6ca8', '#c45a24', '#3e8c68', '#8a5da4']
    markers = ['o', 's', '^', 'D']
    x = np.arange(len(NAMES))
    for ax, metric, title in zip(axes, ['raw', 'h_c_RMS'],
            ['Raw h/c kick metric · bounded by clipping and equalizing',
             'h/c RMS kick metric · bounded by a shared shrink']):
        values = []
        for case, color, marker in zip(cases, colors, markers):
            subset = {r['direction']: r for r in rows if r['metric'] == metric and r['density'] == 1024 and (r['prefix'], r['regime']) == case}
            y = [subset[name]['gain_median'] for name in NAMES]
            values.append(y)
            label = f"{'LEFT' if case[0] == 0 else 'RIGHT'} prefix · {case[1][-1]} section"
            ax.scatter(x + (cases.index(case) - 1.5) * .12, y, c=color, marker=marker,
                       s=33, zorder=3, label=label, edgecolors='white', linewidths=.35)
        values = np.asarray(values)
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() < 1.8
        ax.vlines(x, values.min(0), values.max(0), color='#777777', linewidth=3, alpha=.45, zorder=1)
        ax.axhline(.5, color='#555555', linestyle='--', lw=1, label='Normal-gain criterion ≤ 0.5')
        ax.axhline(1, color='#aaaaaa', linestyle=':', lw=1)
        ax.axvspan(19.5, 20.5, color='#eeeeee', zorder=0)
        ax.set_ylim(-.04, 1.8); ax.set_xlim(-.65, 20.65)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_ylabel('Remaining effect on\naction-output pattern (gain)')
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    labels = ['h only', 'c only'] + [f'R{i:02d}' for i in range(16)] + ['Full\nradial', 'Cell\nradial', 'Tangent*']
    axes[-1].set_xticks(x, labels, fontsize=9)
    axes[-1].set_xlabel('All 20 normal directions retained; R00–R15 are the fixed random full-state directions')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.42), ncol=5, frameon=False, fontsize=10)
    fig.suptitle('Action-output effects of hidden-only kicks shrink; cell-sensitive errors persist', y=.985, fontsize=16)
    fig.text(.5, .925, 'One original PGC network · one context (8) · native probe after pre-cue reset · two prefixes × two periodic sections', ha='center', fontsize=11)
    fig.text(.08, .035, 'Dots: medians over 8 anchors × 2 signs; grey spans: range of the four case medians (not confidence intervals).\n'
        'Normal gain: remaining off-curve actor-score error after 2 updates. *Tangent: total actor-output displacement gain, a distinct reference.\n'
        'Three-update observation window; centered curve shape; 1,024 actual headings. Raw/RMS labels describe physical kick construction.', fontsize=10)
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.81, hspace=.28)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'recovery_gains.{ext}', dpi=170)
    plt.close(fig)


def main():
    rp = BASE / 'report_original'
    summary, validation = read(rp / 'summary.json'), read(rp / 'validation.json')
    assert summary['complete'] and validation['complete']
    assert validation['summary_sha256'] == sha(rp / 'summary.json')
    model = next(m for m in summary['models'] if m['condition'] == 'pgc33')
    audit = next(m for m in validation['models'] if m['condition'] == 'pgc33')
    path = BASE / 'pgc33/original/summary.json'
    assert sha(path) == audit['summary_sha256']
    pairs, probes, rows = extract(read(path), model, audit)
    assert len(pairs) == 4 and {p['context'] for p in pairs} == {8}
    assert {p['branch'] for p in pairs} == {'pre_cue_reset_native'}
    assert {p['kind'] for p in pairs} == {'probe_glimpse'}
    assert {(p['prefix'], p['regime']) for p in pairs} == {(p, r) for p in (0, 1) for r in ('fb_B', 'fb_F')}
    OUT.mkdir(exist_ok=False)
    with (OUT / 'all_directions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ranges = {m: {name: range_values(rows, m, names) for name, names in {
        'hidden_only': ['hidden_only'], 'cell_only': ['cell_only'],
        'random_full': [f'random_full_{i:02d}' for i in range(16)],
        'full_radial': ['full_radial'], 'cell_radial': ['cell_radial'], 'tangent_total': ['tangent']}.items()}
        for m in ('raw', 'h_c_RMS')}
    changes = []
    for r in rows:
        if r['density'] != 1024: continue
        other = next(v for v in rows if v['density'] == 512 and all(v[k] == r[k] for k in (*KEYS, 'metric', 'direction')))
        changes.append(abs(r['gain_median'] - other['gain_median']))
    record = dict(complete=True, descriptive_only=True, no_new_model_or_environment_execution=True,
        source_sha256=sha(__file__), protocol_sha256=summary['protocol_sha256'],
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (path, rp/'summary.json', rp/'validation.json')},
        selected_pairs=pairs, probe_records=probes, all_direction_rows=len(rows),
        median_gain_ranges_1024=ranges, max_512_1024_change_of_direction_median=max(changes),
        original_candidate_count=0, classification_unchanged=True,
        unit='One original PGC model and one context, two prefixes and two periodic sections; not four independent replicates.',
        observable='Off-curve distance in two-observation centered actor-score outputs after physical h/c kicks. Tangent reference uses total output displacement, not off-curve distance.')
    (OUT/'summary.json').write_text(json.dumps(record, indent=2)+'\n')
    figure(rows)
    table = '\n'.join('| '+m+' | '+' | '.join(f'{v[0]:.3f}–{v[1]:.3f}' for v in ranges[m].values())+' |' for m in ranges)
    text = f'''# Cell-sensitive output errors survive the resolved PGC disturbance tests

The only four original-model comparisons that reached finite recovery all belong to **one PGC context (8), after a pre-cue reset and a native probe**: LEFT/RIGHT preparation prefixes × backward/forward sections of the same periodic input. They are not four independent contexts or networks. No intact-history or Grid superiority claim follows.

![All retained disturbance directions](recovery_gains.png)

After two recurrent updates (0.133 s), the action-output effect of hidden-only kicks shrinks by about 92–93%. Cell-only effects instead grow, while radial effects persist. Tangent output differences retain about 97% of their size. These are actor-score off-curve gains after physical state kicks, **not hidden/cell state-norm recovery**. The actor observation uses two consecutive score vectors, spanning a three-update (0.2 s) window.

| Physical kick metric | Hidden-only | Cell-only | 16 random normals | Full radial | Cell radial | Tangent total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

Numbers are ranges of direction medians across the four related cases at 1,024 headings. In the raw metric, all 16 random normals and hidden-only pass; cell-only, full-radial and cell-radial fail. In the RMS metric, only hidden-only passes. Every normal remains observable, both initial and final tangent references remain visible, no factor leakage occurs, and the final physical kicks satisfy bounds. Thus these are conclusive failures of the fixed broad-recovery criterion, not missing/zero disturbances. The criterion requires every normal's median gain and gain relative to tangent to be at most 0.5.

The same classifications hold at 512 headings; the largest change in a direction's median gain from 512 to 1,024 is {max(changes):.6f}. The full CSV retains all {len(rows)} direction rows (four cases × two metrics × two densities × 21 directions), including successful directions and tangent controls.

The geometry gate concerns the angular shape after removing its common mean shift; it does not establish a stationary absolute actor-output family. These tests use periodically alternating native forward/backward action inputs. No endogenous ring is nominated. Grid's 162 metric pairs remain sampling-unresolved; PGC has 102 unresolved pairs plus these four recovery failures. Exact aliases remain in those main-report denominators, while these four comparisons are not declared exact aliases.

[All directions and both densities](all_directions.csv) · [Exact keys, visibility, convergence, mean drift and provenance](summary.json) · [Unchanged main report](../report_original/README.md)
'''
    (OUT/'README.md').write_text(text)
    (OUT/'artifacts.json').write_text(json.dumps({p.name: dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)),pairs=len(pairs),direction_rows=len(rows),ranges=ranges,max_density_difference=max(changes))))


if __name__ == '__main__':
    main()
