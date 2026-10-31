"""Summarize the phase-specific input and intact-policy hybrid experiments."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_lstm_code_swap import estimate,paired,holm
from summarize_lstm_goal_geometry import holm_signflips

ROOT=Path(__file__).resolve().parents[1]
LABELS=dict(intact='Original PGC',goal_grid='Ordinary goal everywhere',goal_near='Ordinary goal near',
    goal_far='Ordinary goal far',goal_early='Ordinary goal just after pickup',
    current_grid_known='Ordinary current everywhere',current_near='Ordinary current near',
    current_far='Ordinary current far',grid='Grid throughout',pgc_after_discovery='PGC for all returns',
    pgc_near='PGC near; grid far',pgc_far='PGC far; grid near')
METRICS=('mean_score','mean_path_eff','post_discovery_goal_rate','mean_first_goal_step','frac_found_goal')


def summarize(kind):
    name='lstm_phase_intervention_1B' if kind=='phase' else 'lstm_policy_hybrid_1B'
    out=ROOT/'results'/name
    protocol=json.loads((out/'protocol.json').read_text())
    runs=[json.loads((out/(f'pgc33_s{s}.json' if kind=='phase' else f's{s}.json')).read_text()) for s in range(1,9)]
    variants=protocol['variants']
    primary=protocol['primary_score_comparisons']
    contrasts=[(r['reference'],r['intervention']) for r in primary] if kind=='phase' else primary
    base='intact' if kind=='phase' else 'grid'
    primary_set={tuple(c) for c in contrasts}
    allcomparisons=contrasts+[(base,v) for v in variants if v!=base and (base,v) not in primary_set]
    if kind=='hybrid':allcomparisons += [('pgc_after_discovery',v) for v in ('pgc_near','pgc_far')]
    for s,run in enumerate(runs,1):
        assert run['seed']==s and (run['episodes'],run['n_envs'],run['eval_seed'])==(128,64,737373)
        assert run['near_radius']==.6
        assert all(run['checks'][k] for k in run['checks'])
        oldpath=ROOT/f'results/lstm_code_swap_1B/{"pgc33" if kind=="phase" else "gc100"}_s{s}.json'
        old=json.loads(oldpath.read_text())
        assert run['variants'][base]['rows']==old['variants']['intact']['rows']
        for v in variants:
            rows=run['variants'][v]['rows'];br=run['variants'][base]['rows']
            assert len(rows)==128
            assert [r['first_goal_t'] for r in rows]==[r['first_goal_t'] for r in br]
            assert np.isclose(np.mean([r['ret'] for r in rows]),run['variants'][v]['mean_score'])
            assert all(r['ret']==10*r['n_goals'] for r in rows)
    estimates={m:{v:estimate([r['variants'][v][m] for r in runs]) for v in variants} for m in METRICS}
    tests={}
    for m in METRICS:
        families={'primary' if m=='mean_score' else 'secondary':{},'exploratory':{}}
        for a,b in allcomparisons:
            family=('primary' if m=='mean_score' else 'secondary') if (a,b) in primary_set else 'exploratory'
            r=paired(estimates[m][a]['values'],estimates[m][b]['values'])
            r.update(reference=a,intervention=b)
            families[family][f'{b}_minus_{a}']=r
        for family in families.values():holm(family);holm_signflips(family)
        tests[m]=families
    exposure={v:{k:estimate([r['exposure'][v][k] for r in runs]) for k in runs[0]['exposure'][v]} for v in variants}
    summary=dict(protocol=protocol,estimates=estimates,comparisons=tests,exposure=exposure,
        validation=dict(complete_seed_pairs=8,baseline_episode_records_exact=1024,
                        first_discovery_unchanged=True,episode_score_math_exact=True),
        checkpoints={str(r['seed']):r['checkpoint_sha256'] for r in runs},
        statistics='Eight training seeds are replicates. Marginal95%paired t intervals; Holm t and exactsignflip within listed primary and exploratory families separately. Other endpoints secondary. Research followup on existing evaluation layouts.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8.1,4.8 if kind=='phase' else 3.2),layout='constrained')
    scores=estimates['mean_score']
    for j,v in enumerate(variants):
        d=scores[v];color='#3379B5' if v=='grid' else '#D87532' if v in ('intact','pgc_after_discovery','pgc_near','pgc_far') else '#929DA6'
        ax.barh(j,d['mean'],height=.56,color=color)
        ax.errorbar(d['mean'],j,xerr=d['sem'],fmt='none',ecolor='#253442',capsize=3)
        ax.text(d['mean']+d['sem']+1.5,j,f'{d["mean"]:.1f}',va='center')
    ax.set(yticks=range(len(variants)),yticklabels=[LABELS[v] for v in variants],
           ylim=(len(variants)-.4,-.6),xlim=(0,max(d['mean']+d['sem'] for d in scores.values())+13),
           xlabel='90-second score (mean ± SEM across 8 seeds)',
           title='When is predictive input useful?' if kind=='phase' else 'Which controller helps on the return trip?')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel('Near: within 60 cm of the true goal center. All interventions begin after discovery.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'{kind}_scores.{ext}',dpi=180)
    plt.close(fig)
    lines=[f'# {"When predictive input helps" if kind=="phase" else "Switching trained controllers during the return trip"}',
           '',f'![Navigation scores]({kind}_scores.png)','']
    if kind=='phase':
        lines += ['Only the predictive shift is removed; no cells are silenced and no weights or hidden states are patched. Every branch follows the original PGC policy until first discovery. The goal and live current code are tested separately.',
                  '', '“Near” is within 0.6m of the true goal center; “far” is outside. “Just after pickup” means the first15 decisions (one second) after each goal hit and respawn. The distance-based intervention uses privileged goal coordinates.']
    else:
        lines += ['Every branch uses the grid policy until the first goal is found. Then we choose the grid or PGC policy for all returns, only near the goal, or only far from it. Both frozen RNNs keep their own states and receive their native codes plus the branch’s actual images, actions, and rewards. No activity vector is transferred between models.',
                  '', 'This compares return controllers after identical discovery experience. The PGC-after-discovery condition is not the original PGC policy during exploration. Selecting a controller by true goal distance is an oracle diagnostic.',
                  '', f'Using PGC for the whole return raises mean score from **{scores["grid"]["mean"]:.2f} to {scores["pgc_after_discovery"]["mean"]:.2f}**. The paired-t and exact tests disagree after correction, so the interval and both tests are retained below. Switching between controllers within each return lowers performance relative to keeping the PGC controller throughout.',
                  '', 'The [action-strategy analysis](../lstm_navigation_phases_1B/action_strategy.json) shows that most paired networks prefer different movement axes or turn directions. Switching may therefore introduce coordination costs. That possibility was recognized after the experiment was specified; it is not itself a tested cause of the losses. These mixed-controller results do not cleanly allocate the native performance advantage to near or far navigation.']
    lines += ['', '## Scores', '', '| Condition | Mean score | SEM |','|---|---:|---:|']
    lines += [f'| {LABELS[v]} | {scores[v]["mean"]:.2f} | {scores[v]["sem"]:.2f} |' for v in variants]
    lines += ['', '## Paired comparisons', '', 'The score counts every goal in the90-second episode. First-discovery records match exactly across branches, so these score differences arise after discovery. Seeds are the statistical replicates. Intervals are marginal95%intervals, not multiplicity-adjusted intervals.', '']
    for family,records in tests['mean_score'].items():
        lines += [f'### {family.capitalize()}', '', '| Difference | Points | 95% interval | Holm t p | Holm exact p |','|---|---:|---:|---:|---:|']
        for r in records.values():
            lo,hi=r['ci95'];lines.append(f'| {LABELS[r["intervention"]]} − {LABELS[r["reference"]]} | {r["difference"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r["holm_t_p"]:.4g} | {r["holm_signflip_p"]:.4g} |')
        lines += ['']
    lines += ['## Scope and checks', '', 'The intervention changes later trajectories. Near/far exposure is therefore descriptive, not a matched amount of altered input, and neural history crosses phase boundaries. A phase effect does not imply that only that phase matters or equal a percentage of the population advantage. Switching inputs or controllers can itself change behavior.', '',
              'All1,024 baseline episode records reproduce the earlier evaluation exactly. Source checkpoints are unchanged. Full per-episode records, secondary return-efficiency endpoints, exposure counts, and source hashes are retained in [summary.json](summary.json) and the eight seed files.', '', '```sh',
              f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/{"lstm_phase_intervention" if kind=="phase" else "lstm_policy_hybrid"}.py',
              f'python scripts/summarize_lstm_navigation_interventions.py --kind {kind}', '```', '',
              '[Protocol](protocol.json) · [Combined explanation](../../mechanisms/navigation-explanation.md)']
    text='\n'.join(lines)+'\n'
    for old,new in [('first15','first 15'),('the90','the 90'),('marginal95%intervals','marginal 95% intervals'),('All1,024','All 1,024')]:text=text.replace(old,new)
    (out/'README.md').write_text(text)
    print(kind,{v:round(d['mean'],3) for v,d in scores.items()})
    for key,r in tests['mean_score']['primary'].items():print(key,r['difference'],r['ci95'],r['holm_t_p'],r['holm_signflip_p'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['phase','hybrid','both'],default='both')
    a=p.parse_args()
    for kind in (['phase','hybrid'] if a.kind=='both' else [a.kind]):summarize(kind)
