"""Presentation-only companion for the saved original recovery table."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_geometry_resolution'
OUT = BASE / 'original_recovery_supplement'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protected = [OUT / p for p in ('summary.json', 'all_directions.csv', 'recovery_gains.png', 'recovery_gains.pdf')]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    saved = json.loads((OUT / 'artifacts.json').read_text())
    for p in protected:
        assert saved[p.name]['sha256'] == sha(p)
    rows = list(csv.DictReader((OUT / 'all_directions.csv').open()))
    assert len(rows) == 336
    selected = [r for r in rows if r['density'] == '1024']
    assert len(selected) == 168
    others = [f'random_full_{i:02d}' for i in range(16)] + ['full_radial', 'cell_radial']
    groups = [['hidden_only'], ['cell_only'], others, ['tangent']]
    cases = [('0', 'fb_B'), ('0', 'fb_F'), ('1', 'fb_B'), ('1', 'fb_F')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    plotted = []
    for ax, metric, title in zip(axes, ('raw', 'h_c_RMS'), ('Raw kick metric', 'h/c RMS kick metric')):
        subset = {(r['prefix'], r['regime'], r['direction']): r for r in selected if r['metric'] == metric}
        assert len(subset) == 84
        for group_index, names in enumerate(groups):
            for d, name in enumerate(names):
                dx = 0 if len(names) == 1 else -.27 + .54 * d / (len(names)-1)
                for j, case in enumerate(cases):
                    r = subset[(*case, name)]
                    value = float(r['gain_median'])
                    assert 0 <= value < 1.65
                    plotted.append(dict(metric=metric, prefix=case[0], section=case[1], direction=name, median=value))
                    ax.scatter(group_index + dx + (j-1.5)*(.05 if len(names)==1 else .009), value,
                               s=21, marker=('o', 's', '^', 'D')[j], linewidths=0,
                               color='#bf692a' if name == 'tangent' else '#237b9b', alpha=.8, zorder=3)
        ax.axhline(.5, color='#777777', linestyle='--', linewidth=1)
        ax.axhline(1, color='#bbbbbb', linestyle=':', linewidth=1)
        ax.set_ylim(-.035, 1.65); ax.set_xlim(-.5, 3.5)
        ax.set_xticks(range(4), ['Hidden-only', 'Cell-only', 'Other\nnormals', 'Tangent'], fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.12)
    axes[0].set_ylabel('Remaining effect on action scores', fontsize=12)
    fig.suptitle('Cell-sensitive action-score errors persist', fontsize=17, y=.96)
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.82, wspace=.12)
    for ext in ('png', 'pdf'):
        target = OUT / f'recovery_compact.{ext}'
        assert not target.exists()
        fig.savefig(target, dpi=100)
    plt.close(fig)
    assert len(plotted) == 168
    archive = BASE / 'original_recovery_presentation'
    archive.mkdir(exist_ok=False)
    original = OUT / 'README.md'
    archived = archive / 'initial_README.md'
    shutil.copyfile(original, archived)
    old = original.read_text()
    needle = '![All retained disturbance directions](recovery_gains.png)'
    assert old.count(needle) == 1
    replacement = '''![Compact recovery comparison](recovery_compact.png)

All plotted dots are case-direction medians, not independent replicates or fitted estimates. “Other normals” retains all 16 fixed random directions plus full-radial and cell-radial kicks. Blue points show normal off-curve action-score gain; orange tangent points show total action-score displacement gain. The dashed line is the normal-gain criterion (0.5); the dotted line is unchanged size (1). The four marker shapes denote the two preparation prefixes × two periodic sections in the same context.

[Full direction-by-direction figure](recovery_gains.png) · [Compact PDF](recovery_compact.pdf)'''
    original.write_text(old.replace(needle, replacement))
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    record = dict(complete=True, presentation_only=True, source_sha256=sha(__file__),
        protected_artifacts_unchanged=before, plotted_case_direction_medians=len(plotted),
        no_new_scores_or_model_execution=True,
        archived_original_readme=dict(path=str(archived.relative_to(ROOT)), sha256=sha(archived)),
        revised_readme_sha256=sha(original),
        outputs={p.name: sha(p) for p in (OUT/'recovery_compact.png', OUT/'recovery_compact.pdf')})
    (OUT / 'compact_presentation.json').write_text(json.dumps(record, indent=2)+'\n')
    (OUT/'artifacts.json').write_text(json.dumps({p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifacts.json'},indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
