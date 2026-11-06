"""Render the complete frozen normal-audit cohort without significance tests."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    data=json.loads((HERE/'summary.json').read_text())
    rows=data['results']
    # Every case is Grid seed 2; retain the population's blue in both metrics.
    colors={'raw_full_state_kicks':'#356FA3','h_c_RMS_full_state_kicks':'#356FA3'}
    metrics=list(colors)
    fig,axes=plt.subplots(1,2,figsize=(9,4.3),sharey=True)
    compact=[]
    for ax,metric in zip(axes,metrics):
        selected=[r for r in rows if r['original_case']['metric']==metric]
        for row in selected:
            directions=row['horizons']['30']['directions']
            values={r['name']:r['normal_gain']['median'] for r in directions}
            y=[values['hidden_only'],float(np.median([values[f'random_full_{k:02d}']for k in range(16)])),values['cell_only']]
            ax.plot(range(3),y,'o-',color=colors[metric],alpha=.55,linewidth=1.1,markersize=4)
            compact.append(dict(case=row['case'],hidden_only=y[0],mixed_direction_median=y[1],cell_only=y[2],
                radial=values['existing_radial'],passes_by_horizon={h:row['horizons'][h]['normal_directions_passing_old_median_gains'] for h in ('15','30','75')}))
        ax.axhline(.5,color='#999999',linestyle='--',linewidth=1)
        ax.set_xticks(range(3),['Hidden\nonly','Mixed\nhidden + cell','Cell\nonly'])
        title='Equal raw distance' if metric==metrics[0] else 'Scale-adjusted distance'
        ax.set_title(f'{title}\n{len(selected)} candidate cases',fontsize=12)
        ax.set_xlim(-.2,2.2);ax.set_ylim(-.04,1.2)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel('Error remaining after 2 seconds',fontsize=12)
    fig.suptitle('Hidden-state errors fade. Cell-state errors persist.',fontsize=15,y=.98)
    fig.text(.5,.015,'Each line is one earlier candidate. All cases are from Grid seed 2.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.065,1,.9])
    for ext in ('png','svg','pdf'):fig.savefig(HERE/f'normal_recovery_by_state_component.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    (HERE/'figure_values.json').write_text(json.dumps(compact,indent=2)+'\n')
    protocol=json.loads((HERE/'protocol.json').read_text())
    checks={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in protocol['source_sha256'].items()}
    assert all(checks.values()),checks
    assert hashlib.sha256((ROOT/'results/lstm_module_phase_observable/summary.json').read_bytes()).hexdigest()==protocol['parent_sha256']
    (HERE/'source_integrity_check.json').write_text(json.dumps(dict(all_match=True,checks=checks,parent_matches=True),indent=2)+'\n')
    lines=['# Broader perturbations of the 14 activity-curve candidates','',
        '**The earlier radial recovery is real, but recovery depends strongly on which state component is perturbed. Hidden-state errors fade; cell-state errors persist.**','',
        '![Recovery by state component](normal_recovery_by_state_component.png)','',
        'Each line is one earlier candidate, all from the Grid model trained with seed 2. The plot shows the median fraction of distance to the evolving activity curve remaining after two seconds. “Mixed” is the median across 16 fixed random directions. The dashed line is the previous half-error criterion. These repeated probes are not independent training seeds; no significance test is applied.','',
        '## What we tested','',
        'All 14 positive cases from the completed four-model observable screen were included. At eight fixed points on each curve, we changed the full recurrent state in both directions along 16 random directions, one hidden-only direction, one cell-only direction, and the previous radial and tangent controls. Every normal was projected off the observed curve tangent. Perturbations were 2% of the original curve radius using the same raw or scale-adjusted distance as each discovery case.','',
        'We then replayed identical future inputs and measured the unchanged fixed observables: every hidden output, or every trained actor score, at four future offsets. Distance was measured to the corresponding evolving unperturbed curve. Perturbed families were never centered separately. All finite kicks, observed coordinates, distances, observed arc displacements, and normal/tangential components are retained.','',
        '## Results','',
        '| Recovery interval | Cases passing all 19 tested normal-direction criteria |','|---|---:|',
        *[f"| {h//15} second{'s' if h!=15 else ''} | {data['broad_pass_counts'][str(h)]}/14 |" for h in (15,30,75)],'',
        'At two seconds, every candidate failed the cell-only probe: its median error retention was 0.72–1.06. With raw distances, most random mixed directions contracted quickly because their visible effect was dominated by hidden-state changes. After adjusting hidden and cell distances by their original scales, mixed errors generally persisted too. All initial observable-normal displacement guards were satisfied; the failure was not caused by invisible or numerically zero perturbations.','',
        'One case passed every tested normal-direction median criterion at five seconds: module 2, axis 0, goal withheld, the 30-second anchor, actor-score observable, raw-distance kicks. Its scale-adjusted counterpart passed 11/19 directions, so the broader recovery does not hold across both kick metrics. The original 14 discovery labels remain unchanged.','',
        '## Meaning','',
        'The network quickly corrects some departures from the activity curve while retaining other cell-state disturbances that affect future activity. This is more specific than a general claim that all disturbances return to a one-dimensional ring. A second persistent phase of a torus could account for some of those directions; the separate two-phase experiment tests that possibility. Native, physically consistent access to the candidate states and behavioral use also remain separate follow-ups.','',
        'The audit covers a fixed finite set of directions and eight locations per curve. Its “all directions” count refers only to these 19 tested normal directions, using the existing median gain and normal-to-tangent criteria. It is not an exhaustive stability proof. Some full-state kicks exceed the natural hidden-output bounds; every overshoot count is recorded. Cell-only kicks leave hidden coordinates unchanged, so their persistence is not explained by that issue.','',
        '## Validation and reproduction','',
        'An analytic attracting ring with an unobserved drifting clock passes the new assay; a contracting plane fails the preferential-recovery comparison. Two tests pass, checking perturbation norms, signs, block specificity, tangent projection, and those positive/negative controls. Every frozen source hash and the completed parent-summary hash match.','',
        'Run `/opt/anaconda3/bin/python scripts/lstm_module_phase_normal_audit.py` from the repository root. Results include `protocol.json`, `controls.json`, `summary.json`, and one `result.json` plus `all_perturbations.npz` per candidate. `figure_values.json` contains every plotted value.','']
    (HERE/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cases=len(rows),broad_pass_counts=data['broad_pass_counts'],source_integrity=True)))


if __name__=='__main__':main()
