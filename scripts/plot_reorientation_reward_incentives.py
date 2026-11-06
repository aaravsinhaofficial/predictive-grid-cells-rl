"""Simple display of the completed task's time and reward allocation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rl_blind_reorientation_review/reward_allocation'


def main():
    path = OUT / 'summary.json'
    data = json.loads(path.read_text())
    rows = [next(r for r in data['evaluations'] if
        (r['condition'], r['stage'], r['variant']) == (condition, 4, 'blind_native'))
        for condition in ('gc100', 'pgc33')]
    assert all(r['episodes'] == 64 and r['learning_frames'] == 40_058_880 for r in rows)
    phases = [('waiting_visible', 'Vision available', '#cbd2d8'),
              ('forced', 'Forced blind actions', '#8ca5b8'),
              ('free', 'Free blind movement', '#df8e42')]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), layout='constrained', sharey=True)
    for ax, key, scale, title in zip(axes,
            ('aggregate_update_percent', 'goal_allocation_fraction'), (1, 100),
            ('Time spent', 'Goals earned')):
        for y, row in enumerate(rows):
            left = 0.
            for phase, label, color in phases:
                value = row[key][phase] * scale
                ax.barh(y, value, left=left, height=.48, color=color,
                    label=label if y == 0 and ax is axes[0] else None)
                if value >= 8:
                    ax.text(left + value / 2, y, f'{value:.1f}%', ha='center', va='center', fontsize=8)
                left += value
            assert abs(left - 100) < 1e-8
        ax.set(title=title, xlim=(0, 100), xticks=[0, 50, 100], xlabel='Share (%)')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_yticks([0, 1], ['Grid', 'PGC + grid'])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Most rewards arrive with vision', fontsize=13)
    fig.savefig(OUT / 'reward_incentives.png', dpi=180)
    plt.close(fig)
    record = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source=str(path.relative_to(ROOT)), selection='Both fixed final40M models, blind_native; all64episodes each',
        rows=[{k: r[k] for k in ('condition', 'episodes', 'aggregate_update_percent',
            'goal_allocation_fraction', 'totals')} for r in rows])
    (OUT / 'reward_incentives.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
