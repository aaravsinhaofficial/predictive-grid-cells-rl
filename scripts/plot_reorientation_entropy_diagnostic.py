"""Display every completed fixed-checkpoint action diagnostic without selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rl_blind_reorientation_review/entropy_diagnostic'
ORDER=['incoming','after_warmup','stage1','stage2','stage3','stage4']
LABELS=['Original','Warmup','Stage 1','Stage 2','Stage 3','Stage 4']


def main():
    data=json.loads((OUT/'summary.json').read_text());rows=data['checkpoints']
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.1))
    for condition,color,offset,label in [('gc100','#347bb5',-.07,'Grid'),('pgc33','#d86b2a',.07,'PGC + grid')]:
        rr=[r for r in rows if r['condition']==condition and r['phases']['free_blind']['updates']]
        x=[ORDER.index(r['endpoint'])+offset for r in rr]
        axes[0].scatter(x,[100*r['phases']['free_blind']['translation_probability'] for r in rr],color=color,s=42,label=label)
        axes[1].scatter(x,[r['phases']['free_blind']['entropy_nats']['mean'] for r in rr],color=color,s=42,label=label)
    for ax in axes:
        ax.set_xticks(range(6),LABELS,fontsize=8);ax.set_xlim(-.5,5.5)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0].set(title='Chance of moving',ylabel='Translation probability during FREE (%)',ylim=(0,100))
    axes[1].set(title='Variety of choices',ylabel='Mean policy entropy during FREE (nats)',ylim=(0,np.log(6)*1.05))
    axes[1].axhline(np.log(6),color='.6',ls='--',lw=1)
    axes[0].legend(frameon=False,fontsize=9)
    fig.supxlabel('One 160-update rollout from each saved state · private CPU streams · no learning',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'free_action_choices.png',dpi=180);fig.savefig(OUT/'free_action_choices.pdf');plt.close(fig)
    text=['# Does the actor explore movement during the free window?','',
        '**Read-only diagnostic, frozen before its rollouts.** Every available prescribed original, post-warmup and stage checkpoint is retained. Each contributes one 160-update rollout from its exact saved non-RNG runtime. No optimizer runs and all checkpoint/weight hashes remain unchanged.','',
        '![All fixed checkpoint action diagnostics](free_action_choices.png)','',
        '| Model | Checkpoint | FREE steps / exposed slots | FREE entropy (nats) | Translation probability | Actual counts: L, R, F, B, SL, SR |',
        '|---|---|---:|---:|---:|---|']
    for r in rows:
        f=r['phases']['free_blind'];n=f['updates']
        entropy='—' if not n else f"{f['entropy_nats']['mean']:.4f}"
        prob='—' if not n else f"{100*f['translation_probability']:.3f}%"
        text.append(f"| {'Grid' if r['condition']=='gc100' else 'PGC + grid'} | {r['endpoint']} | {n} / {r['free_slots']} | {entropy} | {prob} | {', '.join(map(str,f['actual_action_counts']))} |")
    text+=['','Translation means any forward, backward or strafe action; it does not mean goalward movement. L/R are left/right turns. Each rollout contains 32 environment slots. The original incoming states provide only the exposure actually encountered; no extra rollouts are added to obtain a desired number of FREE samples. Within each model, incoming and post-warmup weights are identical; their saved histories differ.','',
        'Entropy is measured separately for each decision, then averaged. A uniform choice among six actions has entropy 1.792 nats (dashed line); values near zero mean the actor is almost certain of one action at each state. Probabilities are measured before action sampling, alongside the actual sampled counts. A broad repertoire across different states can coexist with low entropy at individual states.','',
        '**The usual training-log entropy is a different quantity:**','',
        '| Model | Log interval | Logged batches | Masked entropy / all updates | Entropy / policy-controlled updates |',
        '|---|---|---:|---:|---:|']
    for condition,item in data['training_logs'].items():
        for stage,r in item['groups'].items():
            text.append(f"| {'Grid' if condition=='gc100' else 'PGC + grid'} | {stage} ({r['first_update']}–{r['last_update']}) | {r['logged_rollouts']} | {r['masked_per_all_step_entropy_nats']['mean']:.4f} | {r['policy_controlled_entropy_nats_mean']:.4f} |")
    text+=['','The log masks forced actions and divides by all 160 updates. Its policy-controlled average includes waiting, visible and FREE decisions. It cannot identify a FREE-specific turning bias by itself. All captured complete log rows remain in [the JSON](summary.json).','',
        'The original entropy coefficient is 0.00008. The [primary-literature and loss-convention audit](../../../docs/research/reorientation_exploration_audit.md) explains the A3C/A2C precedent of 0.01 and why a numerical coefficient comparison is not an exploration guarantee. This diagnostic changes neither coefficient nor active training.','',
        'CUDA RNG blobs are preserved in provenance and each rollout archive. Future draws use the declared private CPU seeds because CUDA generator state cannot be loaded into a CPU generator. This is a fresh short trajectory from a saved state, not exact replay of the cloud worker’s future. State noise and native physical randomness remain enabled. Sampled steps/checkpoints are nested observations of one original seed pair.','',
        '[Frozen protocol](protocol.json) · [All numbers and source hashes](summary.json) · [Diagnostic source](../../../scripts/lstm_reorientation_entropy_diagnostic.py)','',
        'Reproduce or append newly available fixed stages: `OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_diagnostic.py`, then `OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/plot_reorientation_entropy_diagnostic.py`. Existing results are reused only when checkpoint and frozen diagnostic hashes match.']
    (OUT/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
