"""Report every frozen action-context sensitivity, without action selection."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_heading_memory_probe import save


def main():
    folder=ROOT/'results/lstm_travel_memory/action_context_original'
    actions=['forward','backward','strafe_left','strafe_right'];rows=[]
    for action in actions:
        for condition in ('gc100','pgc33'):
            for seed in (1,2):
                r=json.loads((folder/action/f'{condition}_s{seed}/context0.json').read_text())
                o=r['ordinary_release'];q=o['trajectory']['60'];g=o['invariance']['30']
                row=dict(action=action,condition=condition,seed=seed,candidate=o['passes_both_metric_candidate_screen'],
                    phase_mae_deg=q['time_refit_information_only']['heldout_mae_deg'],fixed_phase_mae_deg=q['fixed_travel_phase']['heldout_mae_deg'],
                    actor_probability_rms=q['actor_probability_family_rms'],endpoint_error_m=r['physics']['endpoint_error_before_snap_m'])
                for metric in ('raw','relative_rms'):
                    h=g[metric]['horizons']['30']
                    row[metric]=dict(radial_over_tangent=h['radial_over_tangent_gain']['median'],
                        radial_gain=h['radial_normal_gain']['median'],departure_p90_radii=h['residual_to_fixed_curve_over_radius']['p90'])
                rows.append(row)
    save(folder/'action_summary.json',dict(unit='One context per independently trained model; four actions are repeated measurements, not independent seeds',rows=rows))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.9))
    settings=[('phase_mae_deg','Travel direction remains readable','Refitted error at 4 s (degrees)'),('radial_over_tangent','Radial and angular changes persist similarly','Radial / tangent gain (2 s→4 s)'),('departure_p90_radii','The full activity curve drifts','Distance from 2 s curve at 4 s (radii)')]
    colors={'gc100':'#3477b4','pgc33':'#e46b31'}
    for ax,(key,title,ylabel) in zip(axes,settings):
        for condition in ('gc100','pgc33'):
            for seed in (1,2):
                group=[next(r for r in rows if r['action']==a and r['condition']==condition and r['seed']==seed) for a in actions]
                y=[r[key] if key=='phase_mae_deg' else r['raw'][key] for r in group]
                offset=(-.08 if condition=='gc100' else .08)+(-.025 if seed==1 else .025)
                ax.plot(np.arange(4)+offset,y,'o-',color=colors[condition],alpha=.8,lw=1,ms=5,
                    label=('Grid' if condition=='gc100' else 'PGC + grid') if seed==1 else None)
        if key=='radial_over_tangent':ax.axhline(.5,color='.55',ls='--',lw=1);ax.axhline(1,color='.8',lw=1)
        if key=='departure_p90_radii':ax.axhline(.05,color='.55',ls='--',lw=1)
        ax.set_xticks(range(4),['Forward','Back','Left','Right']);ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(folder/'action_context.png',dpi=180);fig.savefig(folder/'action_context.pdf');plt.close(fig)
    lines=['# The result survives changes in movement action','',
        'We repeated the same travel-memory test using forward, backward, leftward and rightward movement. Each family followed the same world-space paths and reached the same endpoint. Body orientation and the previous-action input matched the action. Every family began from the same native recurrent context for that model.','',
        '![All four movement actions](action_context.png)','',
        'All 16 model/action combinations retain readable travel-direction information. None meets the unchanged full-state attracting-ring criteria. Radial disturbances persist about as strongly as changes along the activity curve, and the whole curve moves away from its earlier location. This pattern suggests a changing memory trace across movement contexts.','',
        '| Action | Model | Refitted travel error at 4 s | Radial/tangent gain, 2→4 s | Drift from 2 s curve (radii) |','|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['action'].replace('_',' ')} | {'Grid' if r['condition']=='gc100' else 'PGC + grid'} seed{r['seed']} | {r['phase_mae_deg']:.2f}° | {r['raw']['radial_over_tangent']:.3f} | {r['raw']['departure_p90_radii']:.2f} |")
    lines.extend(['','The table and figure use raw full-state distances; [the numeric summary](action_summary.json) also reports the unchanged h/c RMS-scaled analysis. The refitted decoder measures available information; it is fitted separately at each time and does not define a ring. Lines connect repeated action measurements of the same trained model, not independent replications.','',
        'The original forward condition was regenerated and matched every published state/input exactly. Actual endpoint errors were below 1e−5 m before the documented numerical snap. The explicit stop and common blind-turn input were identical across actions. Native predictive current inputs were replaced by ordinary current codes during the autonomous test; the saved goal code remained intact.','',
        'This is a separately frozen coverage check using original seeds 1–2 and context 0. The actor did not choose the scripted approach sequence; its preceding recurrent context came from a real policy history. See the [frozen protocol](protocol.json), [all outcomes](summary.json), and per-action folders.'])
    assert not any(r['candidate'] for r in rows),'Rewrite result text if any candidate passed'
    (folder/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
