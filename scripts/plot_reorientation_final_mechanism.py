"""Describe every fixed40M mechanism context without changing analyses or gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/lstm_reorientation_mechanism/final'
CONDITIONS = ('gc100', 'pgc33')
CUES = ('native', 'ordinary', 'plus60', 'minus60')


def load(path):
    return json.loads(path.read_text())


def main():
    observable = load(ROOT/'results/lstm_reorientation_observable/summary.json')
    independent = load(OUT/'independent_artifact_audit.json')
    assert independent['passed'] and independent['complete']
    assert observable['all32_checks_complete']
    rows = []
    sources = {}
    for condition in CONDITIONS:
        for context in range(2):
            folder = OUT/condition
            primary_path = folder/f'context{context}.json'
            causal_path = folder/'causal'/f'context{context}'/'result.json'
            paths_path = causal_path.with_name('navigation_paths.npz')
            primary, causal = load(primary_path), load(causal_path)
            paths = np.load(paths_path)
            for p in (primary_path, causal_path, paths_path):
                sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
            branches = {}
            for cue in CUES:
                b = primary['branches'][cue]
                q = b['invariance']['30']['raw']['horizons']['30']
                branches[cue] = dict(candidate=b['passes_both_metric_candidate_screen'],
                    hidden_radius_retention=b['trajectory']['60']['hidden_radius_retention'],
                    cell_radius_retention=b['trajectory']['60']['cell_radius_retention'],
                    refitted_direction_mae_deg=b['trajectory']['60']['time_refit_information_only']['heldout_mae_deg'],
                    actor_probability_family_rms=b['trajectory']['60']['actor_probability_family_rms'],
                    fullstate_radial_over_tangent=q['radial_over_tangent_gain']['median'],
                    fullstate_curve_departure_p90=q['residual_to_fixed_curve_over_radius']['p90'],
                    observable=[r for r in observable['rows'] if (r['condition'],r['context'],r['cue'])==(condition,context,cue)])
            action_effects = {}
            for construction in ('raw', 'relative_rms'):
                native = paths[f'{construction}__native__none__actions']
                effects = {}
                for key in paths.files:
                    if key.startswith(construction+'__') and key.endswith('__actions'):
                        value = paths[key]; assert value.shape == native.shape == (30,192)
                        # A first-hit stop would leave -1 entries; retain the exact tape comparison.
                        effects[key] = dict(changed_actions=int(np.count_nonzero(value != native)),
                            sampled_action_entries=int(native.size))
                action_effects[construction] = effects
            native_nav = causal['branches']['raw__native']['navigation']['none']
            cue_identity = all(np.array_equal(paths[f'{m}__{cue}__{edit}__actions'],
                paths[f'{m}__native__{edit}__actions']) for m in ('raw','relative_rms')
                for cue in CUES for edit in causal['branches'][m+'__native']['navigation'])
            rows.append(dict(condition=condition,context=context,
                checkpoint_sha256=primary['checkpoint_sha256'],branches=branches,
                phase_repair_contrasts=causal['phase_repair_contrasts'],
                native_navigation=native_nav,action_effects=action_effects,
                cue_actions_exact_for_each_edit=cue_identity,
                goal_center_distance_m=float(np.linalg.norm(np.array(primary['physics']['endpoint'])-
                    np.load(folder/f'context{context}_states.npz')['goal_center'])),
                audit=causal['audit']))
            rows[-1]['opportunity'] = next(q['opportunity'] for q in independent['models'][condition]['causal'] if q['context']==context)
    summary=dict(scope='One original seed1 pair, fixed40,058,880-frame endpoints; two physically selected contexts each; phases and interventions are nested measurements',
        rows=rows,source_sha256=sources,fullstate_branch_candidates=sum(b['candidate'] for r in rows for b in r['branches'].values()),
        observable_centered_candidates=observable['centered_candidates'],
        observable_branch_metric_checks=observable['branch_metric_checks'])
    (OUT/'mechanism_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    fig, axes = plt.subplots(1,3,figsize=(10.2,3.7),layout='constrained')
    for x,r in enumerate(rows):
        color = '#357eb0' if r['condition']=='gc100' else '#d66b2c'
        for cue,offset,marker in [('native',-.07,'o'),('ordinary',.07,'s')]:
            b = r['branches'][cue]
            axes[0].scatter(x+offset,b['hidden_radius_retention'],color=color,marker=marker,s=42)
            obs = next(q for q in b['observable'] if q['metric']=='raw_full_state_kicks')
            axes[1].scatter(x+offset,obs['radial_over_tangent'],color=color,marker=marker,s=42)
            error = r['phase_repair_contrasts']['raw']['60']['raw'][cue+'_error_deg']
            axes[2].scatter(x+offset,error,color=color,marker=marker,s=42)
    axes[0].set(title='Direction traces survive',ylabel='Hidden activity radius at 4 s / start',ylim=(0,.85))
    axes[1].set(title='Some errors fade faster',ylabel='Radial / angular persistence\nin future hidden activity',ylim=(0,.85))
    axes[1].axhline(.5,color='.6',ls='--',lw=1)
    axes[1].annotate('Signal nearly gone',xy=(1,.111),xytext=(.1,.23),fontsize=8,
        color='.35',arrowprops=dict(arrowstyle='-',color='.6',lw=.7))
    axes[2].set(title='Little lasting cue repair',ylabel='Error after a ±30° state shift (degrees)',ylim=(29.75,30.25))
    axes[2].axhline(30,color='.6',ls='--',lw=1)
    for ax in axes:
        ax.set_xticks(range(4),['Grid\nContext 1','Grid\nContext 2','PGC + grid\nContext 1','PGC + grid\nContext 2'],fontsize=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13)
    axes[0].scatter([],[],color='.3',marker='o',label='Native cue')
    axes[0].scatter([],[],color='.3',marker='s',label='Ordinary cue')
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    fig.supxlabel('Fixed final models · two contexts each · one original training-seed pair',fontsize=9)
    fig.savefig(OUT/'mechanism_summary.png',dpi=180);fig.savefig(OUT/'mechanism_summary.pdf');plt.close(fig)

    lines=['# What changed inside the new blind-reorientation agents?','',
        '**The PGC model retains stronger direction-related hidden activity in these contexts. Some disturbances fade selectively, while the predictive pulse produces almost no lasting repair of the full-state phase estimate and scarcely changes the next movements.**','',
        '![Direction traces, selective recovery and cue repair](mechanism_summary.png)','',
        'We used the actual square arena, maps, goal code and trained LSTM. For each final model, we took the first two physically eligible native task histories and replayed their visible/turn/forward-probe sequence across 768 directions, ending at the same location. We then supplied one correct, ordinary, or ±60° wrong current cue and held subsequent inputs identical within that model. The saved goal code stayed intact. The primary comparison is the complete hidden + cell state; a second analysis observes every hidden unit over the following second.','',
        '**The positive difference:** after four seconds the PGC hidden-state family retains about 51–75% of its starting radius, versus about 0.013–5.4% for Grid in these four contexts. In the future-hidden-activity analysis, prescribed radial disturbances in PGC persist about 38–46% as strongly as angular disturbances for raw state kicks. Scaling hidden and cell coordinates gives ratios of 0.44 and 0.61. Similar values remain with the predictive pulse removed or rotated, suggesting a property of the trained agent and its preceding history, rather than an acute stabilization effect of that pulse. The second Grid context also has a low radial ratio, but its observable activity is below the frozen amplitude guard.','',
        '**What that activity is doing:** direction remains readable from both populations with a new analysis readout fitted at each time. The activity families drift and deform, so their later patterns no longer map consistently onto the earlier curve. No branch passes the unchanged full-state criteria; none of the 32 future-hidden-activity branch/metric checks passes either. The measured organization is a changing directional trace with unequal sensitivity to disturbances. The action tests below show how little this trace controls the subsequent blind behavior.','',
        '**Causal test:** just before the cue, we substituted a state belonging to a direction 30° away, or applied a small radial/random disturbance. We compared a correct predictive cue with ordinary and wrong-direction cues, withdrew the cue, and ran the original actor for two seconds with matched action and motor-noise draws. A separate reset removes all hidden and cell state immediately after the cue. All 192 directions, failures and zero-hit windows are retained.','',
        '| Population / context | Late error: ordinary → native | Mean path | Mean goalward progress | +30° edit: changed actions | Reset: changed actions |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        q=r['phase_repair_contrasts']['raw']['60']['raw'];n=r['native_navigation'];a=r['action_effects']['raw']
        lines.append(f"| {'Grid' if r['condition']=='gc100' else 'PGC + grid'} / {r['context']+1} | {q['ordinary_error_deg']:.3f}° → {q['native_error_deg']:.3f}° | {n['path_length_m']['mean']:.3f} m | {n['total_goalward_progress_m']['mean']:+.4f} m | {a['raw__native__plus30__actions']['changed_actions']} / 5,760 | {a['raw__native_reset__none__actions']['changed_actions']} / 5,760 |")
    lines+=['',
        'The table uses raw full-state geometry and the unperturbed native-cue movement baseline. The underlying per-context `causal/context*/result.json` files retain both perturbation metrics, both ±30° edits, ±2% radial/random edits at both prescribed times, every cue, both reference conventions and actor-probability changes. Action entries are paired samples from one policy, not independent experimental replicates.','',
        'All native, ordinary and wrong-cue variants produce identical sampled actions for each matched edit in these contexts. PGC movement is mostly rotation with little translation. Its remaining direction information has very weak influence on the free actor here. Grid cue shams are exact, and a post-cue reset exactly erases every prior state-edit difference.','',
        '**Goal opportunity:** the two Grid contexts and the first PGC context start too far from the goal disk for any 30-step action sequence to hit it under the saved motor-noise tapes. Their zero-hit counts are therefore opportunity-limited. Phase repair, action changes and continuous progress remain measurable. The second PGC context is not ruled out by that speed bound: its goal-disk gap is 0.491 m and the maximum path bound is 0.702 m. It still moves only about 0.026 m and has zero hits. Every context is retained. The controlled forced delay is noiseless to match the neural flow; free movement uses native motor noise. Native noisy whole-episode evaluations are a separate result.','',
        'The native physical contexts and saved goal codes differ between the two models. Thus the population comparison describes these trained agents in their own task histories; the cue-removal and perturbation comparisons within each frozen agent isolate the acute effect. These are fixed final checkpoints of one original training-seed pair, not a population-level significance test.','',
        '[Complete numerical summary](mechanism_summary.json) · [Independent artifact and goal-opportunity audit](independent_artifact_audit.json) · [All future-hidden-activity results](../../lstm_reorientation_observable/README.md) · [Interpretation audit](../../lstm_reorientation_observable/interpretation_review.json) · [Frozen protocol](../../../docs/research/reorientation_mechanism_protocol.json) · [Exact reproduction commands and timing](../../../docs/research/reorientation_mechanism_reproduction.md)','',
        'Rebuild this display with `python scripts/plot_reorientation_final_mechanism.py`. Every underlying numerical analysis and threshold is unchanged.']
    assert summary['fullstate_branch_candidates']==summary['observable_centered_candidates']==0
    assert all(r['cue_actions_exact_for_each_edit'] for r in rows),'Rewrite cue-action statement if a nonidentity is observed'
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
