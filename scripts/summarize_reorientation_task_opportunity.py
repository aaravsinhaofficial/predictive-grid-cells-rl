"""Read-only fixed40M opportunity audit; distance budgets are nominal, not oracles."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rl_blind_reorientation_review/task_opportunity'
WORKERS={'gc100':'ring-20261105-reorient-grid-01','pgc33':'ring-20261105-reorient-pgc-02'}
STAGES=((0,'baseline.json'),(1,'stage1_evaluation.json'),(2,'stage2_evaluation.json'),
        (3,'stage3_evaluation.json'),(4,'stage4_evaluation.json'))


def summarize(events,threshold=None):
    chosen=[r for r in events if threshold is None or r['initial_goal_distance_m']<=threshold]
    return dict(windows=len(chosen),hits=sum(r['goal_hit'] for r in chosen),
        mean_path_m=float(np.mean([r['path_m'] for r in chosen])) if chosen else None,
        mean_progress_m=float(np.mean([r['progress_m'] for r in chosen])) if chosen else None,
        total_updates=sum(r['free_updates'] for r in chosen))


def main():
    rows=[];hashes={}
    for condition,worker in WORKERS.items():
        folder=ROOT/'results/ring_cloud_artifacts'/worker
        cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
        dt=cfg['action_repeat']/60;duration=30*dt
        nominal_fullspeed=cfg['v_max']*duration
        alpha=math.exp(-dt/.25)
        nominal_fromrest=cfg['v_max']*dt*(30-alpha*(1-alpha**30)/(1-alpha))
        for stage,filename in STAGES:
            path=folder/filename
            assert path.exists(),path
            data=json.loads(path.read_text());hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for variant in ('blind_native','seen_native'):
                events=data['free_window'][variant]['rows'];episodes=data['variants'][variant]['rows']
                assert len(episodes)==64
                assert sum(r['goal_hit'] for r in events)==data['free_window'][variant]['free_hits']
                distances=[r['initial_goal_distance_m'] for r in events]
                rows.append(dict(condition=condition,stage=stage,variant=variant,original_episodes=64,
                    all_windows=summarize(events),distance_quantiles_m=np.quantile(distances,[0,.1,.5,.9,1]).tolist() if distances else None,
                    distance_strata={str(t):summarize(events,t) for t in (.5,.7,1.)},
                    windows_beyond_nominal_fullspeed_budget=sum(d>cfg['goal_radius']+nominal_fullspeed for d in distances),
                    windows_beyond_nominal_acceleration_budget=sum(d>cfg['goal_radius']+nominal_fromrest for d in distances),
                    nominal_fullspeed_path_budget_m=nominal_fullspeed,nominal_noiseless_fromrest_path_budget_m=nominal_fromrest,
                    goal_radius_m=cfg['goal_radius'],free_duration_s=duration,
                    free_window_rows=events,original_episode_rows=episodes))
    result=dict(version='fixed40M_task_opportunity_readonly_v1',all20endpoints_retained=len(rows)==20,
        qualifications='Distance strata are descriptive, post-outcome diagnostics. The0.7mfullspeed andfromrest budgets ignore native Gaussian motor noise, actual motor state and turning geometry; they are not rigorous unreachable/achievable labels for native evaluation. Passing a distance cutoff does not prove a successful action sequence. Exact controlled192phase tape-specific bounds are in the separate mechanism audit. Seen/blind later histories can diverge.',
        credit=dict(max_event_updates=129,bptt_updates=160,gamma=.99,discount30=.99**30,
            discount60=.99**60,discount89=.99**89,discount128=.99**128,
            interpretation='The whole event is inside oneBPTTsegment; reward is attenuated but not omitted. Long waiting/visible periods and sparse successfulfreehits change incentives and learning signal.'),
        rows=rows,source_sha256=hashes)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Physical opportunity in the fixed40M task','',
        'All original episodes and every free window are retained. Near/far strata are descriptive checks; the bounds below do not incorporate the actual native motor-noise tape or turning trajectory.','',
        '| Model / stage / vision | Free windows | Start ≤0.5m: hits/windows | Start ≤0.7m: hits/windows | Start >1.0m | Mean free path |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['distance_strata']['0.5'],r['distance_strata']['0.7']
        lines.append(f"| {r['condition']} / {r['stage']} / {r['variant']} | {r['all_windows']['windows']} | {a['hits']}/{a['windows']} | {b['hits']}/{b['windows']} | {r['windows_beyond_nominal_fullspeed_budget']} | {r['all_windows']['mean_path_m']:.3f}m |")
    lines+=['',
        'The goal radius is 0.3 m. A 0.5 m starting distance leaves 0.2 m to the goal disk; 0.7 m leaves 0.4 m. At 0.35 m/s, a 2 s free window has a generous 0.7 m noiseless path budget before acceleration and turning costs. Native Gaussian motor noise prevents using this nominal budget as a strict impossibility proof.','',
        'At the fixed final PGC endpoint, 125 blind windows begin within 0.7 m and 61 within 0.5 m, yet none hits the goal; the near-window mean path is about 0.027 m. Distance alone therefore does not explain the failed behavior. The seen PGC condition has 37 hits in 123 windows within 0.7 m, although later seen/blind paths and window starts can differ.','',
        'Conversely, 113 of 312 final PGC blind windows begin beyond 1 m, exceeding the generous noiseless 2 s budget. This reduces the fraction of trials that can teach successful local navigation. Three of the four separate controlled mechanism contexts are also rigorously opportunity-limited under their saved noise tapes; that exact bound applies to those controlled contexts only.','',
        'The maximum 129-step reorientation chain fits inside the 160-step BPTT segment. Discounting at 0.99 gives 0.4088 weight to a reward 89 updates after the cue and 0.2763 at 128 updates; credit is present but attenuated. These checks favor an incentive/exploration bottleneck over a claim that the LSTM architecture cannot hold useful direction.','',
        '[Every original episode/window and source hash](summary.json) · [Reward allocation and training-stage exposures](../reward_allocation/README.md) · [Exact controlled-context opportunity audit](../../lstm_reorientation_mechanism/final/independent_artifact_audit.json)','',
        'Reproduce with `python scripts/summarize_reorientation_task_opportunity.py`. No experiment or frozen training/analysis source is changed.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
