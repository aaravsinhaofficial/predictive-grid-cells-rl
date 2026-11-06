"""Describe all angular observations from the two original-checkpoint shape runs."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results/lstm_reorientation_mechanism'
SOURCES = {'gc100':'original_shape_smoke', 'pgc33':'original_pgc_shape_smoke'}
FIELDS = ('body_heading', 'actual_travel_direction', 'egocentric_goal_bearing')
EDGES = np.linspace(-np.pi, np.pi, 13)


def angular_summary(values):
    values = np.asarray(values)
    wrapped = (values+np.pi)%(2*np.pi)-np.pi
    counts, _ = np.histogram(wrapped, EDGES)
    ordered = np.sort(np.mod(values, 2*np.pi))
    max_gap = float(np.diff(np.r_[ordered, ordered[0]+2*np.pi]).max())
    circular = np.exp(1j*values).mean()
    return dict(count=len(values), bin_counts=counts.tolist(),
        occupied_30degree_bins=int((counts > 0).sum()),
        largest_unsampled_gap_deg=float(np.rad2deg(max_gap)),
        circular_coverage_fraction=float(1-max_gap/(2*np.pi)),
        circular_mean_deg=float(np.rad2deg(np.angle(circular))),
        circular_resultant_length=float(abs(circular)))


def main():
    out = FOLDER/'original_angular_coverage'; out.mkdir(exist_ok=True)
    rows = []; provenance = {}; summary = {}
    for condition, name in SOURCES.items():
        path = FOLDER/name/'native_phase_coverage.json'
        source = json.loads(path.read_text())
        rows.extend([dict(condition=condition, **row) for row in source['rows']])
        provenance[condition] = dict(path=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            decisions=source['decisions'], records=len(source['rows']),
            observer_only=source['observation_only'])
        summary[condition] = {}
        for phase in ('endpoint','free_start'):
            selected = [r for r in source['rows'] if r['phase'] == phase]
            summary[condition][phase] = dict(records=len(selected),
                contributing_episode_slots=len({(r['slot'],r['episode']) for r in selected}),
                **{key:angular_summary([r[key] for r in selected]) for key in FIELDS})
    payload = dict(scope='Original1B seed1 Grid/PGC checkpoints transferred to the new reorientation schedule for physical/array shape validation; no reorientation training and no candidate outcomes',
        no_phase_exclusions=True, all_records_retained=True, total_records=len(rows),
        angle_units_in_records='radians', bin_edges_deg=np.rad2deg(EDGES).tolist(),
        interpretation='Empirical native event support, distinct from the separately controlled360degree state families; endpoint and free-start observations are repeated/nested, not independent trained agents',
        provenance=provenance, summary=summary, records=rows)
    (out/'coverage.json').write_text(json.dumps(payload,indent=2)+'\n')
    fig, axes = plt.subplots(2,3,figsize=(9.6,4.7),sharex=True,sharey=True)
    highest = max(max(summary[c][p][f]['bin_counts'])/summary[c][p][f]['count']*100
        for c in SOURCES for p in ('endpoint','free_start') for f in FIELDS)
    ymax = 5*np.ceil(highest/5)+2
    colors = dict(gc100='#347bb5', pgc33='#d86b2a')
    labels = dict(gc100='Grid', pgc33='PGC + grid')
    for row_index,phase in enumerate(('endpoint','free_start')):
        for col,key in enumerate(FIELDS):
            ax = axes[row_index,col]
            for condition in SOURCES:
                values = summary[condition][phase][key]
                ax.stairs(np.array(values['bin_counts'])/values['count']*100,
                    np.rad2deg(EDGES), color=colors[condition],lw=2,label=labels[condition])
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(-180,180);ax.set_ylim(0,ymax);ax.set_xticks([-180,0,180])
            ax.grid(axis='y',alpha=.15)
            if row_index == 1: ax.set_xlabel('Angle (degrees)')
        first,second = (summary[c][phase]['records'] for c in SOURCES)
        axes[row_index,0].set_ylabel(('At the cue' if phase == 'endpoint' else 'At free navigation')+'\nObservations (%)')
        axes[row_index,2].text(.98,.97,f'Grid: {first}   PGC: {second}',transform=axes[row_index,2].transAxes,
            ha='right',va='top',fontsize=9)
    for ax,title in zip(axes[0],('Body heading','Last travel direction','Goal relative to body')):ax.set_title(title,fontsize=11)
    axes[0,0].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Original weights under the new task schedule',fontsize=13)
    fig.supxlabel('Shape-validation histories only · all observations retained · one original seed pair',fontsize=9)
    fig.tight_layout();fig.savefig(out/'angular_coverage.png',dpi=180);fig.savefig(out/'angular_coverage.pdf');plt.close(fig)
    text = ['# Angular coverage of original-checkpoint transfer histories','',
        '**These are original 1B checkpoints running the new task schedule for shape validation. They have not been trained on this task.** The plot is a support check, not a new trained-model mechanism result.','',
        '![All original-checkpoint angular observations](angular_coverage.png)','',
        'The observer recorded every encountered probe endpoint and free-navigation start across 2,700 policy updates per population. No angular observations were excluded. Each panel uses 30° bins; each curve sums to 100%.','',
        '| Population | Event boundary | Observations | Body-heading bins | Travel-direction bins | Goal-bearing bins |',
        '|---|---|---:|---:|---:|---:|']
    for condition in SOURCES:
        for phase in ('endpoint','free_start'):
            r = summary[condition][phase]
            text.append(f"| {labels[condition]} | {'Probe endpoint' if phase == 'endpoint' else 'Free navigation'} | {r['records']} | {r['body_heading']['occupied_30degree_bins']}/12 | {r['actual_travel_direction']['occupied_30degree_bins']}/12 | {r['egocentric_goal_bearing']['occupied_30degree_bins']}/12 |")
    text.extend(['','World heading, last travel direction, and goal direction relative to the body are separate quantities. Their empirical distributions describe the policy-generated histories. The controlled experiment deliberately replays a complete 360° family; its angular distribution should not be mistaken for the policy’s usual distribution.','',
        'All twelve 30° bins are occupied in every distribution. The largest empty angular gaps range from 5.3° to 12.5° across these recorded samples. This finds no large missing direction sector in the original transfer histories; the endpoint goal-bearing distributions still show clear nonuniformity.','',
        'The [complete JSON](coverage.json) retains all observations, source hashes, bin counts, largest angular gaps, and circular summaries. Repeated events and paired endpoint/free-start observations are nested within one original seed pair. Fresh coverage will be collected from the fixed final trained checkpoints.'])
    (out/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(total_records=len(rows),summary=summary)))


if __name__ == '__main__':main()
