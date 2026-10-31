"""Small figures for the episode-offset tests; read saved results, fit nothing."""
import numpy as np
from plot_lstm_mechanism_summary import (
    GRID, PGC, GRAY, INK, read, save, sem, horizontal_scores,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def offset_scores():
    rescue = read('lstm_episode_offset_rescue_1B')['estimates']['post_handoff_score']
    ablation = read('lstm_episode_offset_rescue_1B/ablation')['estimates']['post_handoff_score']
    assert rescue['intact']['values'] == ablation['intact']['values']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), layout='constrained')
    horizontal_scores(axes[0], [
        ('Original', rescue['intact'], PGC),
        ('Ordinary goal', rescue['goal_grid'], '#E8AD82'),
        ('+ correct offset', rescue['episode_mean'], PGC),
        ('+ wrong offset', rescue['wrong_episode_mean'], GRAY),
    ], PGC, xmax=101)
    horizontal_scores(axes[1], [
        ('Original', ablation['intact'], PGC),
        ('− correct offset', ablation['remove_episode_mean'], PGC),
        ('− wrong offset', ablation['remove_wrong_episode_mean'], GRAY),
    ], PGC, xmax=101)
    for ax in axes:
        ax.set(xlabel='Final 50-second score', xticks=[0, 40, 80], xlim=(0, 101))
    axes[0].set_title('Add the fixed offset back')
    axes[1].set_title('Subtract it from the original')
    fig.supxlabel('Actor input only · eligible episodes · 8 PGC-model seeds · mean ± SEM',
                  fontsize=9, color='#687A86')
    save(fig, '05_fixed_offset_test')


def spectra_summary():
    rows = read('lstm_eigenspectra_1B')['models']
    old = read('lstm_variance_sources_1B')['models']
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.9), layout='constrained')
    for c, color, shift in [('gc100', GRID, -.17), ('pgc33', PGC, .17)]:
        values = [
            [[r['metrics'][field] for r in rows if r['condition'] == c]
             for field in ('total_participation_ratio', 'within_participation_ratio')],
            [[100 * r['between_episode_share'] for r in old if r['condition'] == c],
             [100 * r['metrics']['episode_actor_variance_share'] for r in rows
              if r['condition'] == c]],
        ]
        for ax, groups in zip(axes, values):
            for j, group in enumerate(groups):
                assert len(group) == 8
                mean, error = np.mean(group), sem(group)
                x = j + shift
                ax.bar(x, mean, width=.30, color=color, zorder=3)
                ax.errorbar(x, mean, yerr=error, color=INK, capsize=3, lw=1, zorder=4)
                ax.text(x, mean + error + .8, f'{mean:.1f}', ha='center', fontsize=10)
    axes[0].set(title='How spread out is the activity?', ylabel='Effective directions (participation ratio)',
                xticks=[0, 1], xticklabels=['All activity', 'Episode means removed'], ylim=(0, 44))
    axes[1].set(title='How much comes from episode means?', ylabel='Share of variance (%)',
                xticks=[0, 1], xticklabels=['Hidden activity', 'Action logits'], ylim=(0, 44))
    for ax in axes:
        ax.yaxis.grid(True, color='#E8EBED', lw=.65)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.legend(handles=[Patch(color=GRID, label='Grid'), Patch(color=PGC, label='PGC + grid')],
               loc='outside upper center', ncol=2, frameon=False, fontsize=10)
    fig.supxlabel('Same physical replay · 8 paired training seeds · mean ± SEM',
                  fontsize=9, color='#687A86')
    save(fig, '06_spectra_summary')


if __name__ == '__main__':
    offset_scores()
    spectra_summary()
