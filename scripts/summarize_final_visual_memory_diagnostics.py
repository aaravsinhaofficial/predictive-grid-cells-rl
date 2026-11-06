"""Summarize every fixed final checkpoint with exact reset/noise identity checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'results/ring_cloud_artifacts'
OUTPUT = ROOT / 'results/ring_visual_memory'
MODELS = [
    ('noiseless_gc100', 'v2_gc100', 'vision-long-grid', 'Grid\nstandard training'),
    ('noiseless_pgc33', 'v2_pgc33', 'vision-long-pgc', 'PGC + grid\nstandard training'),
    ('noise_gc100', 'v3_gc100', 'vision-noise-grid', 'Grid\nnoise training'),
    ('noise_pgc33', 'v3_pgc33', 'vision-noise-pgc', 'PGC + grid\nnoise training'),
]


def read(path):
    return json.loads(path.read_text())


def main():
    noise_folder = ARTIFACTS / 'ring-20261105-vision-noisy-eval-01'
    reset_folder = ARTIFACTS / 'ring-20261105-vision-reset-eval-01'
    assert read(noise_folder / 'summary.json')['completed']
    assert read(reset_folder / 'summary.json')['completed']
    rows = {}
    for name, reset_name, worker, label in MODELS:
        trained = ARTIFACTS / f'ring-20261105-{worker}-01'
        clean = read(trained / 'final_evaluation.json')
        reset = read(reset_folder / f'{reset_name}.json')
        noise = read(noise_folder / f'{name}.json')
        assert all(reset['exact_nonreset_row_identity'].values())
        assert reset['checkpoint_sha256'] == noise['checkpoint_sha256'] == read(trained / 'summary.json')['final_sha256']
        for variant in ('blind', 'seen', 'blind_current_grid', 'seen_current_grid'):
            assert reset['variants'][variant]['rows'] == clean['variants'][variant]['rows']
        rows[name] = dict(label=label, checkpoint_sha256=noise['checkpoint_sha256'],
            clean_scores={k: v['mean_score'] for k, v in clean['variants'].items()},
            every_step_reset_scores={k: v['mean_score'] for k, v in reset['variants'].items()},
            noisy_scores={k: v['mean_score'] for k, v in noise['variants'].items()},
            every_step_reset_contrasts=reset['descriptive_contrasts'],
            raw_noise_sha256=noise['noise_draw_audit']['raw_draw_sha256'])
    assert len({r['raw_noise_sha256'] for r in rows.values()}) == 1
    (OUTPUT / 'final_diagnostics_summary.json').write_text(json.dumps(dict(
        all_four_fixed_endpoints=True, exact_nonreset_rows=True, exact_common_raw_noise=True,
        training_frames_beyond_original=75_072_000, episodes_per_model_variant=64,
        trained_seed_pairs_per_arm=1, models=rows), indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    positions = np.arange(len(rows)); width = .34
    for ax, variant, title in zip(axes, ('blind', 'seen'), ('During blackouts', 'With vision')):
        kept = [r['clean_scores'][variant] for r in rows.values()]
        cleared = [r['every_step_reset_scores'][variant + '_reset'] for r in rows.values()]
        ax.bar(positions - width/2, kept, width, label='Memory kept', color='#357bb8')
        ax.bar(positions + width/2, cleared, width, label='Memory cleared', color='#a8adb2')
        ax.set_title(title); ax.set_xticks(positions, [r['label'] for r in rows.values()], fontsize=9)
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(90, max(kept+cleared)*1.12))
    axes[0].set_ylabel('Navigation score'); axes[1].legend(frameon=False, loc='upper right')
    fig.text(.5, .015, 'Hidden and cell state cleared at every scheduled blackout step; vision controls use the same schedule.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(OUTPUT / 'final_memory_reset.png', dpi=180); plt.close(fig)
    print(json.dumps({k: {'blind': r['clean_scores']['blind'], 'every_reset': r['every_step_reset_scores']['blind_reset']} for k, r in rows.items()}))


if __name__ == '__main__':
    main()
