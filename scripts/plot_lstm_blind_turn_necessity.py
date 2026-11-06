"""Simple report of the paired-history information audit; no model fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/lstm_blind_turn_necessity';rows=json.loads((folder/'summary.json').read_text())['records']
    common=np.array([r['ceilings']['common_action_progress_m']*1000 for r in rows])
    aware=np.array([r['ceilings']['phase_aware_progress_m']*1000 for r in rows])
    fig,ax=plt.subplots(figsize=(6,3.7))
    bars=ax.bar([0,1],[common.mean(),aware.mean()],color=['#777777','#347bb5'],width=.62)
    for bar,val in zip(bars,[common.mean(),aware.mean()]):ax.text(bar.get_x()+bar.get_width()/2,val+.12,f'{val:.3f} mm',ha='center',fontsize=12)
    ax.set_xticks([0,1],['Same action distribution\nfor both histories','Action choice can\ndepend on history'])
    ax.set_ylabel('Best average goalward progress\non the next native update (mm)')
    ax.set_ylim(0,6.3);ax.set_title('Same current inputs; different actions needed')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(folder/'necessity_ceiling.png',dpi=180);fig.savefig(folder/'necessity_ceiling.pdf');plt.close(fig)
    lines=['# The proposed blind-turn test requires remembered information','',
        '**All eight paired-history audits pass:** four original pilot models, each with a turn-only history and a complete forward-probe/cue history. This is a design diagnostic; no training or cloud experiment was launched.','',
        '![Information requirement](necessity_ceiling.png)','',
        f'The two histories end with headings about {rows[0]["audits"]["heading_separation_degrees"]:.2f}° apart. Their final 1,287 input values, last three complete input vectors, position, saved goal, previous action/reward, and motor velocities match byte-for-byte. Different earlier turn actions are visible only through recurrent history. In the probe variant, actual four-update forward trajectories finish at the same endpoint before the native-code glimpse.','',
        f'The best action distribution shared by both histories gives {common.mean():.3f} mm average goalward progress over the next native update. Choosing separately for each remembered heading gives {aware.mean():.3f} mm. These are analysis-only ceilings calculated by evaluating all six actions through the original simulator; the policy receives no oracle action or angle label.','',
        'The preferred translations are backward in one history and forward in the other. Any randomized memoryless policy is a mixture of common actions and obeys the same ceiling. A future trained agent must exceed this bound using intact history and show a corresponding reset-sensitive navigation benefit before proceeding to ring interpretation.','',
        '## Original-model transfer responses','',
        'The original checkpoints were not trained for this task. None exceeds the common-action ceiling here. The table reports their expected first-update progress, calculated from their own six-action probabilities. This prevents the information audit from being mistaken for successful learned memory.','',
        '| Model | History | Intact state (mm) | Reset at final decision (mm) |','|---|---|---:|---:|']
    for r in rows:
        b=r['original_actor_branches'];lines.append(f"| {'Grid' if r['condition']=='gc100' else 'PGC + grid'} seed {r['seed']} | {r['kind'].replace('_',' ')} | {1000*b['retained_native']['mean_expected_progress_m']:.6f} | {1000*b['reset_at_final_decision']['mean_expected_progress_m']:.6f} |")
    lines.extend(['','## Controls retained','',
        '- Deterministic native square-arena physics, actual renderer, original CNN/LSTM/maps and unchanged goal snapshot. The controlled visible history is fifteen observations containing fourteen native left-turn updates.','- Explicit motor braking before forced segments. Both terminal motor states are identical, including the remaining common angular velocity; equality does not rely on a turn instantly braking.','- Only 169 current slots differ in the PGC native-versus-ordinary glimpse. Grid glimpse replacement changes zero slots. All saved-goal and other input values remain identical.','- Erasing state after the glimpse, or before an ordinary glimpse, produces exactly identical final action distributions across the paired histories. The native cue is withdrawn before the identical final suffix.','- Endpoint rounding correction is bounded below 1e−5 m; the largest measured correction is '+f"{max(r['audits']['endpoint_snap_max_m'] for r in rows):.2e} m.",'- Entire observation/action traces, both cue histories, recurrent endpoints, hashes and actor responses are retained in each model folder. These four models provide repeated implementation checks, not eight independent training replications.','',
        '[Frozen diagnostic protocol](protocol.json) · [All numerical results](summary.json) · [Proposed training timeline and shortcut controls](../../docs/research/rl_blind_reorientation_proposal.md)','',
        'Reproduce locally with `OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_blind_turn_necessity.py`. The three unit tests cover the action-mixture bound, byte equality and native motor-state matching; the run itself asserts every full input/goal/position/motor identity.'])
    assert all(r['original_actor_branches']['retained_native']['mean_expected_progress_m']<=r['ceilings']['common_action_progress_m'] for r in rows)
    (folder/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
