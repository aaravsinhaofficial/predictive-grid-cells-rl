"""Report both audited fixed endpoints; never render partial training outcomes.

Only saved results are read. No neural execution, training, checkpoint selection,
or trial-level significance test is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/research/local_goal_return_training_protocol.json'
PROTOCOL_HASH = '0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
MANIFEST = ROOT/'docs/research/local_goal_return_training_launch.json'
AUDITS = ROOT/'results/local_goal_return_design/cloud_audit'
OUT = ROOT/'results/ring_local_goal_return/report'
CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}
COLORS = {'gc100': '#357eb0', 'pgc33': '#d97536'}
BLIND = ('blind_native', 'blind_ordinary', 'blind_plus60', 'blind_minus60',
         'blind_reset_after_glimpse', 'blind_reset_every_blind')
VARIANTS = BLIND + tuple(v.replace('blind_', 'seen_', 1) for v in BLIND)
VARIANT_LABELS = dict(zip(VARIANTS, (
    'Blind: native cue', 'Blind: ordinary cue', 'Blind: cue +60°', 'Blind: cue −60°',
    'Blind: erase after cue', 'Blind: erase every blind input',
    'Visible: native cue', 'Visible: ordinary cue', 'Visible: cue +60°', 'Visible: cue −60°',
    'Visible: erase after cue', 'Visible: erase on matched blind clock')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def summarize_rows(rows):
    """Keep every prescribed heading/context, including failed invalid trials."""
    assert len(rows) == 256
    assert sorted((r['context'], r['heading_index']) for r in rows) == [
        (c, h) for c in range(16) for h in range(16)], 'Duplicate or missing physical trial'
    assert all(r['hit'] in (False, True, 0, 1) for r in rows)
    assert all(not r['hit'] for r in rows if r.get('invalid', False))
    successes = [sum(bool(r['hit']) for r in rows if r['context'] == c) for c in range(16)]
    valid_contexts = sum(not any(r.get('invalid', False) for r in rows if r['context'] == c)
                         for c in range(16))
    return dict(trials=256, contexts=16, valid_contexts=valid_contexts,
        successes=sum(successes), success_fraction=sum(successes)/256,
        context_successes=successes, context_success_fraction=[n/16 for n in successes],
        invalid_trials=sum(bool(r.get('invalid', False)) for r in rows))


def aggregate_evaluation(report):
    assert report['complete'] and report['bank'] == 'final_hard'
    assert set(report['variants']) == set(VARIANTS) and len(report['rows']) == 256*12
    assert {r['variant'] for r in report['rows']} == set(VARIANTS)
    values = {}
    for variant in VARIANTS:
        stats = summarize_rows([r for r in report['rows'] if r['variant'] == variant])
        for key in ('trials', 'successes', 'success_fraction', 'invalid_trials'):
            assert stats[key] == report['variants'][variant][key], (variant, key)
        values[variant] = stats
    native = values['blind_native']['successes']
    erased = values['blind_reset_after_glimpse']['successes']
    certificate = report['certificate']
    assert certificate['equally_weighted_pairs'] == 128 and certificate['heading_trials'] == 256
    certified = certificate['certified_pairs']; uncertified = certificate['uncertified_pairs']
    assert certified + uncertified == 128 and min(certified, uncertified) >= 0
    bound = (.5*certified + uncertified)/128
    assert certificate['common_information_success_upper_bound'] == bound
    assert certificate['all_pairs_certified'] == (certified == 128)
    numeric = report['all_physical_checks'] and native >= 167 and native-erased >= 39
    expected = dict(near_baseline_pass=False, hard_numeric_pass=numeric,
                    hard_certified_memory_pass=numeric and certified == 128)
    assert report['gates'] == expected
    return dict(variants=values, gates=expected, all_physical_checks=report['all_physical_checks'],
        intact_successes=native, postcue_erased_successes=erased,
        postcue_erasure_cost_successes=native-erased,
        postcue_erasure_cost_percentage_points=100*(native-erased)/256,
        certified_pairs=certified, uncertified_pairs=uncertified,
        common_information_success_upper_bound=bound)


def safe_relative(value):
    p = Path(value)
    assert not p.is_absolute() and '..' not in p.parts
    return p


def verify_audit(folder, condition, audit_path):
    audit = read(audit_path)
    assert audit['passed'] and audit['complete'] and audit['status'] == 'complete', \
        'Both final independent audits must be complete before reporting'
    assert not audit['failures'] and not audit['pending']
    assert audit['condition'] == condition and audit['protocol_sha256'] == PROTOCOL_HASH
    # The recorded absolute root is historical provenance; relocation preserves suffix and bytes.
    relative = folder.relative_to(ROOT)
    assert Path(audit['folder']).parts[-len(relative.parts):] == relative.parts
    assert audit['auditor_sha256'] == sha(ROOT/'scripts/audit_local_goal_return_training.py')
    for path, digest in audit['audit_dependency_sha256'].items():
        assert sha(ROOT/safe_relative(path)) == digest
    assert 'final.pt' in audit['artifacts'] and 'summary.json' in audit['artifacts']
    for path, item in audit['artifacts'].items():
        actual = folder/safe_relative(path)
        assert actual.stat().st_size == item['bytes'] and sha(actual) == item['sha256'], path
    return audit


def training_windows(rows, width=32):
    """Fixed moving means, restarted at stage boundaries; entropy weights decisions."""
    assert width == 32 and [r['update'] for r in rows] == list(range(1, 4097))
    result = []
    for stage in range(1, 5):
        local = rows[(stage-1)*1024:stage*1024]
        assert all(r['stage'] == stage and r['completed_trials'] == 32 for r in local)
        hits = np.array([r['free_hits'] for r in local], dtype=float)
        decisions = np.array([r['free_updates'] for r in local], dtype=float)
        entropy = np.array([r['entropy_per_free_decision'] for r in local], dtype=float)
        assert np.all((hits >= 0) & (hits <= 32)) and np.all(decisions > 0)
        assert np.isfinite(entropy).all() and np.all((entropy >= 0) & (entropy <= math.log(6)+2e-6))
        kernel = np.ones(width)
        successes = np.convolve(hits, kernel, mode='valid')/(32*width)
        means = np.convolve(entropy*decisions, kernel, mode='valid')/np.convolve(decisions, kernel, mode='valid')
        result.append(dict(stage=stage, updates=[r['update'] for r in local[width-1:]],
                           success_fraction=successes.tolist(), entropy_per_decision=means.tolist()))
    return result


def load_complete():
    assert sha(PROTOCOL) == PROTOCOL_HASH
    protocol = read(PROTOCOL); manifest = read(MANIFEST)
    assert manifest['protocol_sha256'] == PROTOCOL_HASH
    routes = manifest['workers']
    assert len(routes) == 2 and {r['condition'] for r in routes} == set(CONDITIONS)
    sources = {str(PROTOCOL.relative_to(ROOT)): PROTOCOL_HASH, str(MANIFEST.relative_to(ROOT)): sha(MANIFEST)}
    package_hash = hashlib.sha256(json.dumps(manifest['package_manifest'],sort_keys=True).encode()).hexdigest()
    assert all(r['package_manifest_sha256'] == package_hash for r in routes)
    for entry in manifest['package_manifest']:
        path = safe_relative(entry['path'])
        assert sha(ROOT/path) == entry['sha256'], str(path)
        sources[str(path)] = entry['sha256']
    for path, digest in {**protocol['source_sha256'], **protocol['original_assets'],
                         **protocol['passed_baseline_assets']}.items():
        assert sha(ROOT/safe_relative(path)) == digest, path
        sources[path] = digest
    # Verify both complete audits before interpreting either model's outcomes.
    checked = {}
    for condition in CONDITIONS:
        route = next(r for r in routes if r['condition'] == condition)
        assert route['artifact_subdir'] == 'experiment'
        folder = ROOT/safe_relative(route['local_artifact_root'])/'experiment'
        audit_path = AUDITS/f'{condition}.json'
        audit = verify_audit(folder, condition, audit_path)
        checked[condition] = (folder, audit)
        sources[str(audit_path.relative_to(ROOT))] = sha(audit_path)
        for name, item in audit['artifacts'].items():
            sources[str((folder/name).relative_to(ROOT))] = item['sha256']
        sources.update(audit['audit_dependency_sha256'])
        sources['scripts/audit_local_goal_return_training.py'] = audit['auditor_sha256']
    from audit_local_goal_return_training import accepted_metrics
    models = {}
    bank = protocol['evaluation']['sealed_banks']['final_hard']
    for condition in CONDITIONS:
        folder, audit = checked[condition]
        summary = read(folder/'summary.json')
        assert summary['completed'] and summary['counters']['updates'] == 4096
        assert summary['counters']['trials'] == 131072 and summary['counters']['completed_stages'] == 4
        baseline_path = ROOT/f'results/local_goal_return_baseline/{condition}/final_hard/summary.json'
        baseline = read(baseline_path)
        assert baseline['bank_sha256'] == bank['sha256']
        stages = [dict(stage=0, training_updates=0, training_trials=0,
                       reset_head_baseline=True, **aggregate_evaluation(baseline))]
        for stage in range(1, 5):
            path = folder/f'stage{stage}_evaluation/summary.json'; report = read(path)
            assert report['condition'] == condition and report['model_seed'] == 1
            assert report['bank_sha256'] == bank['sha256'] and report['clean_state_noise_sigma'] == 0
            assert report['behavior_protocol_sha256'] == protocol['evaluation']['behavior_protocol_sha256']
            assert report['checkpoint_policy_hash'] == audit['checks'][f'stage{stage}_before_evaluation.pt']['policy_sha256']
            stages.append(dict(stage=stage, training_updates=stage*1024, training_trials=stage*32768,
                               reset_head_baseline=False, **aggregate_evaluation(report)))
        metrics, history = accepted_metrics(folder)
        assert len(metrics) == 4096 and not history['partial_last_row']
        assert history == {k:audit['checks']['all_update_metrics_and_checkpoint_counters'][k] for k in history}
        models[condition] = dict(route=next(r for r in routes if r['condition']==condition),
            stages=stages, final=stages[-1], training_counters=summary['counters'],
            active_physics_frames=summary['active_physics_frames'],
            metric_history=history, training_trace=training_windows(metrics))
    return models, sources


def plot(models, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios':[1, 1.35]}, sharey=True)
    fig.subplots_adjust(left=.065,right=.99,bottom=.24,top=.79,wspace=.17)
    styles = {'blind_native':('-', 'o'), 'blind_ordinary':('--', 's'),
              'blind_reset_after_glimpse':(':', '^')}
    for condition in CONDITIONS:
        for variant, (line, marker) in styles.items():
            values=[100*s['variants'][variant]['success_fraction'] for s in models[condition]['stages']]
            axes[0].plot(range(5),values,color=COLORS[condition],ls=line,marker=marker,lw=1.8,ms=4)
    axes[0].set(title='Same hard trials after every stage',xlabel='Completed training stage',
                xticks=range(5),xticklabels=['Reset\nheads','1','2','3','4'],ylabel='Successful trials (%)')
    style_legend=[Line2D([0],[0],color='#47555e',ls=style[0],marker=style[1],ms=4,label=label)
                  for style,label in zip(styles.values(),('Native cue','Ordinary cue','Erase after cue'))]
    axes[0].legend(handles=style_legend,frameon=False,loc='upper left',fontsize=9)
    x=np.arange(6)
    for condition,offset in zip(CONDITIONS,(-.19,.19)):
        values=[100*models[condition]['final']['variants'][v]['success_fraction'] for v in BLIND]
        axes[1].bar(x+offset,values,width=.35,color=COLORS[condition],alpha=.88,label=LABELS[condition])
    axes[1].set(title='Final checkpoint: cue and memory controls',xticks=x,
        xticklabels=['Native','Ordinary','+60°','−60°','Erase\nafter cue','Erase\nevery step'])
    for ax in axes:
        ax.set_ylim(0,104);ax.set_yticks([0,25,50,75,100]);ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.14);ax.tick_params(axis='x',length=0,pad=7)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[c],lw=4,label=LABELS[c]) for c in CONDITIONS],
               loc='upper right',bbox_to_anchor=(.98,.99),frameon=False,ncol=2)
    fig.text(.065,.94,'Learning a local return to the goal',fontsize=17,color='#25353e')
    fig.text(.065,.055,'One original seed pair · 16 matched contexts · 256 trials per condition · all four fixed stages',
             fontsize=10,color='#65727b')
    for suffix in ('png','pdf'):fig.savefig(out/f'local_return.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),sharex=True)
    fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.20,wspace=.26)
    for condition in CONDITIONS:
        for window in models[condition]['training_trace']:
            axes[0].plot(window['updates'],100*np.array(window['success_fraction']),color=COLORS[condition],lw=1.5)
            axes[1].plot(window['updates'],window['entropy_per_decision'],color=COLORS[condition],lw=1.5)
    for ax in axes:
        for boundary in (1024,2048,3072):ax.axvline(boundary,color='#69757e',ls=':',lw=.8,alpha=.55)
        ax.set_xlim(0,4096);ax.set_xticks([0,1024,2048,3072,4096]);ax.set_xlabel('Learning updates')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
    axes[0].set(ylabel='Training trials with a goal hit (%)',ylim=(0,100))
    axes[1].set(ylabel='Entropy per FREE action (nats)',ylim=(0,math.log(6)*1.05))
    fig.legend(handles=[Line2D([0],[0],color=COLORS[c],lw=3,label=LABELS[c]) for c in CONDITIONS],
               loc='upper right',bbox_to_anchor=(.99,1.),frameon=False,ncol=2)
    fig.text(.075,.94,'Reward discovery during training',fontsize=16,color='#25353e')
    fig.text(.075,.035,'Fixed 32-update moving means; windows restart at each distance/delay stage.',fontsize=10,color='#65727b')
    for suffix in ('png','pdf'):fig.savefig(out/f'training.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)


def write_readme(models, out):
    final={c:models[c]['final'] for c in CONDITIONS}
    g,p=(final[c] for c in CONDITIONS)
    lead=(f"At the fixed endpoint, Grid returned successfully in **{g['intact_successes']}/256** trials and "
          f"PGC + grid in **{p['intact_successes']}/256**. Erasing recurrent state after the cue changed "
          f"those counts to **{g['postcue_erased_successes']}** and **{p['postcue_erased_successes']}**.")
    lines=['# Local goal-return training','',lead,'','![All fixed stages and final controls](local_return.png)','',
        'Both models retained their original 1B spatial/visual/recurrent features initially, received identical fresh action/value heads, then trained all weights using goal-pickup reward. Each completed four fixed stages, 4,096 updates and 131,072 trials. Evaluation always used the same hard trial bank.','',
        '| Final control | Grid successes | PGC + grid successes | Valid contexts: Grid / PGC |',
        '|---|---:|---:|---:|']
    for variant in VARIANTS:
        a,b=(final[c]['variants'][variant] for c in CONDITIONS)
        lines.append(f"| {VARIANT_LABELS[variant]} | {a['successes']}/256 | {b['successes']}/256 | {a['valid_contexts']}/16 · {b['valid_contexts']}/16 |")
    lines += ['', 'Every condition includes all 256 trials in 16 matched physical contexts; invalid trials count as failures. The visible controls use the same scheduled cue changes and state erasures.','',
        '| Frozen final criterion | Grid | PGC + grid |','|---|---:|---:|']
    for label,values in (
        ('Native successes ≥167/256',[str(final[c]['intact_successes']) for c in CONDITIONS]),
        ('Post-cue erasure loss ≥39 successes',[str(final[c]['postcue_erasure_cost_successes']) for c in CONDITIONS]),
        ('Certified opposite-heading pairs',[f"{final[c]['certified_pairs']}/128" for c in CONDITIONS]),
        ('Numeric behavior gate',['Pass' if final[c]['gates']['hard_numeric_pass'] else 'Fail' for c in CONDITIONS]),
        ('Certified memory gate',['Pass' if final[c]['gates']['hard_certified_memory_pass'] else 'Fail' for c in CONDITIONS])):
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    bounds=[100*final[c]['common_information_success_upper_bound'] for c in CONDITIONS]
    lines += ['',f'The saved physical tapes give common-information success bounds of {bounds[0]:g}% (Grid) and {bounds[1]:g}% (PGC + grid). These bounds apply to the specified opposite-heading trials after heading information is erased.','',
        '![Reward discovery and action entropy](training.png)','',
        'Training curves average fixed 32-update windows within each stage. Action entropy is weighted by the number of FREE decisions. Setup, forced actions and padding contribute no actor reward.','',
        'This is one original training-seed pair. The counts describe its matched physical trials; the report does not treat 256 headings as 256 independently trained networks. All four stage results, context counts, input hashes and accepted training history are in [summary.json](summary.json).','',
        'Reproduce after both final audits pass: `python scripts/plot_local_goal_return_training.py`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();models,sources=load_complete()
    args.out.mkdir(parents=True,exist_ok=True)
    summary=dict(complete=True,original_model_seed=1,models=models,input_sha256=sources,
        reporting_source_sha256=sha(__file__),protocol_sha256=PROTOCOL_HASH,
        matched_physical_contexts=16,headings_per_context=16,trial_level_significance_tests=False,
        fixed_training_updates_per_model=4096,fixed_training_trials_per_model=131072,
        gate_any_population_pass=any(models[c]['final']['gates']['hard_certified_memory_pass'] for c in CONDITIONS))
    plot(models,args.out);write_readme(models,args.out)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(complete=True,out=str(args.out),gate_any_population_pass=summary['gate_any_population_pass'])))


if __name__=='__main__':main()
