"""Report the two fixed original-policy qualification evaluations; no rollout."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/goal_cue_memory/qualification'
OUT = BASE / 'report'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_counts(grid, pgc):
    assert len(grid) == len(pgc) == 256
    keyed = [{(r['context'], r['angle_index']): r for r in rows} for rows in (grid, pgc)]
    assert set(keyed[0]) == set(keyed[1]) == {(c, a) for c in range(16) for a in range(16)}
    result = []
    for context in range(16):
        row = dict(context=context, trials=16, grid=0, pgc=0, both=0, neither=0, grid_only=0, pgc_only=0)
        for angle in range(16):
            g, p = (k[(context, angle)] for k in keyed)
            assert not g.get('invalid', False) and not p.get('invalid', False)
            g, p = g['hit'], p['hit']
            assert isinstance(g, bool) and isinstance(p, bool)
            row['grid'] += g; row['pgc'] += p
            row['both' if g and p else 'grid_only' if g else 'pgc_only' if p else 'neither'] += 1
        row['difference'] = row['pgc'] - row['grid']
        result.append(row)
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data, evidence = {}, {}
    for condition in ('gc100', 'pgc33'):
        folder = BASE / condition; ap = BASE / 'audits' / f'{condition}.json'
        audit = json.loads(ap.read_text())
        assert audit['passed'] and audit['complete'] and not audit['failures'] and not audit['pending']
        for rel, record in audit['artifacts'].items():
            path = folder / rel
            assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
        data[condition] = json.loads((folder/'qualification_evaluation/summary.json').read_text())
        assert data[condition]['variant_order'] == ['continuous']
        evidence[condition] = dict(audit_path=str(ap.relative_to(ROOT)), audit_sha256=sha(ap),
            evaluation_summary_sha256=sha(folder/'qualification_evaluation/summary.json'),
            checkpoint_policy_hash=data[condition]['checkpoint_policy_hash'])
    rows = paired_counts(*(data[c]['variants']['continuous']['rows'] for c in ('gc100','pgc33')))
    totals = {key: sum(r[key] for r in rows) for key in rows[0] if key != 'context'}
    assert totals['grid'] == 181 and totals['pgc'] == 236
    execution_path = ROOT/'results/goal_cue_memory_design/qualification_execution.json'
    execution = json.loads(execution_path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'paired_contexts.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.4), gridspec_kw={'width_ratios':[1, 1.7]})
    colors = ['#2878b5', '#e87532']; labels = ['Grid', 'PGC + grid']
    axes[0].bar(labels, [181/256*100, 236/256*100], color=colors, width=.6)
    for x, n in enumerate((181,236)):
        axes[0].text(x, n/256*100-9, f'{n}/256', ha='center', fontsize=10, color='white')
    axes[0].axhline(75, color='0.4', ls='--', lw=1)
    axes[0].set(ylabel='Goals reached (%)', ylim=(0,105), title='Goal code stays available')
    for r in rows:
        offset=(r['context']-7.5)*.012
        axes[1].plot([offset,1+offset], [r['grid'],r['pgc']], color='0.75', lw=1, zorder=1)
    for x,key in enumerate(('grid','pgc')):
        axes[1].scatter([x+(r['context']-7.5)*.012 for r in rows], [r[key] for r in rows], color=colors[x], s=28, zorder=2)
    axes[1].set(xticks=[0,1], xticklabels=labels, ylabel='Goals reached / 16',
                ylim=(0,17), xlim=(-.3,1.3), title='Each line is one matched context')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/'qualification.png', dpi=180)
    fig.savefig(OUT/'qualification.pdf')
    plt.close(fig)
    summary = dict(version='fixed_original_qualification_report_v1', complete=True,
        producer_sha256=sha(__file__), protocol_sha256=execution['protocol_sha256'],
        execution_path=str(execution_path.relative_to(ROOT)), execution_sha256=sha(execution_path),
        evidence=evidence, paired_contexts=rows, paired_totals=totals,
        contexts_pgc_higher=sum(r['difference']>0 for r in rows),
        contexts_equal=sum(r['difference']==0 for r in rows),
        contexts_grid_higher=sum(r['difference']<0 for r in rows),
        difference_percentage_points=(236-181)/256*100,
        both_originals_qualified=False, training_performed=False, memory_bank_evaluated=False,
        cloud_cost_usd=0, scope='One original training seed pair; all 16 matched contexts and all 256 prescribed trial pairs. Descriptive counts only; no seed-population inference.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text = f'''# Original networks: goal code remains available

![Original qualification](qualification.png)

**PGC+grid reached 236/256 goals (92.2%); Grid reached 181/256 (70.7%).** The difference is 55 goals, or 21.5 percentage points, in this one original training-seed pair. No new training was done.

We first obtained each goal code by an actual native goal pickup. Then we cleared setup history, put both agents at the same central position, and gave them a stationary wait followed by six seconds to return to the unmarked goal. **The goal code stayed on throughout this qualification.** The native six actions and movement noise were retained; image noise was set to zero.

Across the 256 matched trial pairs, both succeeded on {totals['both']}; only PGC+grid succeeded on {totals['pgc_only']}; only Grid succeeded on {totals['grid_only']}; neither succeeded on {totals['neither']}. PGC+grid scored higher in {summary['contexts_pgc_higher']}/16 contexts, tied in {summary['contexts_equal']}, and scored lower in {summary['contexts_grid_higher']}. [All context counts](paired_contexts.csv) are retained. These contexts are not independent training seeds.

This suggests that the original PGC+grid controller uses the supplied goal information more successfully in this controlled return task. Memory after removing that information is a separate question.

**The planned paired training prerequisite failed:** both original models had to reach at least 192/256 goals (75%, dashed line). Grid reached 181. This proposal therefore stopped before its memory-bank baseline or training. No cloud worker was created or reserved; cloud cost was **$0**.

Both independent audits passed all 512 trials. They checked native movement and saved inputs, verified unchanged original actor/CNN/LSTM weights and the shared fresh critic, and reproduced the complete stationary neural sequence and first movement update exactly. [Paired audit](../audits/pair_review.json) · [Exact protocol](../../../../docs/research/goal_cue_memory_training_protocol.json) · [Execution and storage record](../../../goal_cue_memory_design/qualification_execution.json).

The two local CPU evaluations took {execution['runs'][0]['evaluation_seconds']:.1f} and {execution['runs'][1]['evaluation_seconds']:.1f} seconds inside the evaluators. Their complete outputs occupy {sum(r['all_output_bytes'] for r in execution['runs'])/1e6:.1f} MB, including lossless raw tensors. [Machine-readable report and source hashes](summary.json).

The [proposed brief-goal task drawing](../../../goal_cue_memory_design/task_design.png) illustrates the memory proposal; this qualification instead kept the goal code available.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps(dict(paired_totals=totals, contexts_pgc_higher=summary['contexts_pgc_higher'],out=str(OUT))))


if __name__ == '__main__':
    main()
