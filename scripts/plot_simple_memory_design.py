"""Illustrate the observation rule; this figure contains no experimental results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'results/ring_simple_memory'

def main():
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.2), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1]})
    blue, gray = '#2473ad', '#dadde2'
    for ax, name in zip(axes, ['Normal training', 'Memory training']):
        ax.set_xlim(0, 9); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.set_title(name, loc='left', fontsize=12, pad=5)
    axes[0].broken_barh([(0, 9)], (.12, .62), facecolors=blue)
    axes[0].text(4.5, .43, 'Image + current grid code available', ha='center', va='center', color='white')
    axes[1].broken_barh([(0, 1), (4, 1), (7, 1)], (.12, .62), facecolors=blue)
    axes[1].broken_barh([(1, 3), (5, 2), (8, 1)], (.12, .62), facecolors=gray)
    axes[1].text(2.5, .43, 'Both inputs zero', ha='center', va='center', fontsize=10)
    axes[1].text(6, .43, 'Zero', ha='center', va='center', fontsize=10)
    axes[1].set_xticks(range(10)); axes[1].set_xlabel('Seconds after finding the goal')
    fig.text(.5, .015, 'Same navigation task. Same LSTM. Example schedule after the training ramp.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'task.png', dpi=180)
    fig.savefig(OUT/'task.svg')
    plt.close(fig)

if __name__ == '__main__':
    main()
