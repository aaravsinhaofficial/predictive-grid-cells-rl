"""Read-only reward/opportunity accounting for all saved reorientation endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rl_blind_reorientation_review/reward_allocation'
WORKERS={'gc100':'ring-20261105-reorient-grid-01','pgc33':'ring-20261105-reorient-pgc-02'}
PHASES=('waiting','visible','turn_a','turn_b','forward_probe','left_delay','free_blind')
VARIANTS=tuple(f'{vision}_{cue}' for vision in ('blind','seen') for cue in
    ('native','ordinary','plus60','minus60','reset_after_glimpse','reset_every_blind'))
FORCED=('turn_a','turn_b','forward_probe','left_delay')
DELAYS=(4,15,30,60)
STAGE_FRAMES=10_014_720


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_variant(data,variant,episode_len=1350):
    originals=data['variants'][variant]['rows']; n=len(originals)
    assert n==data['episodes_per_variant']
    episodes={(r['slot'],r['episode']):dict(slot=r['slot'],episode=r['episode'],
        total_goals=int(r['n_goals']),free_goals=0,forced_goals=0,visible_goals=0,
        free_updates=0,admitted_events=0) for r in originals}
    assert len(episodes)==n
    free=data['free_window'][variant]
    for row in free['rows']:
        episode=episodes[row['slot'],row['episode']]
        episode['free_goals']+=int(row['goal_hit']);episode['free_updates']+=int(row['free_updates'])
    windows=data['window_rows'][variant]
    ending={p:0 for p in PHASES}
    for row in windows:
        episode=episodes[row['slot'],row['episode']];episode['admitted_events']+=1
        if row['goal_hit']:
            ending[row['end_phase']]+=1
            if row['end_phase'] in FORCED:episode['forced_goals']+=1
            elif row['end_phase']=='visible':episode['visible_goals']+=1
            else:assert row['end_phase']=='free_blind'
    for episode in episodes.values():
        episode['waiting_goals']=episode['total_goals']-episode['free_goals']-episode['forced_goals']-episode['visible_goals']
        assert episode['waiting_goals']>=0
        episode['waiting_post_discovery_goals']=episode['waiting_goals']-int(episode['total_goals']>0)
        assert episode['waiting_post_discovery_goals']>=0
    totals={k:sum(r[k] for r in episodes.values()) for k in
        ('total_goals','free_goals','forced_goals','visible_goals','waiting_goals','waiting_post_discovery_goals','free_updates','admitted_events')}
    assert totals['free_goals']==free['free_hits']==ending['free_blind']
    assert totals['forced_goals']==free['forced_hits']
    assert totals['free_updates']==free['free_updates']
    index=VARIANTS.index(variant)
    schedule={k:v[index] for k,v in data['schedule'].items()}
    updates=dict(zip(PHASES,schedule.pop('phase_update_counts')))
    assert sum(updates.values())==n*episode_len
    assert updates['free_blind']==totals['free_updates']
    assert schedule['admitted']==len(windows)==totals['admitted_events']
    assert schedule['admitted']==sum(schedule[k] for k in ('completed','interrupted_reward','interrupted_done'))
    assert schedule['coverage_violations']==0
    assert schedule['interrupted_reward']==sum(r['goal_hit'] and not r['episode_end'] for r in windows)
    assert schedule['interrupted_done']==sum(r['episode_end'] for r in windows)
    assert schedule['completed']==sum(not r['goal_hit'] and not r['episode_end'] for r in windows)
    counts=dict(waiting_visible=updates['waiting']+updates['visible'],forced=sum(updates[p] for p in FORCED),free=updates['free_blind'])
    goals=dict(waiting_visible=totals['waiting_goals']+totals['visible_goals'],forced=totals['forced_goals'],free=totals['free_goals'])
    return dict(variant=variant,episodes=n,episode_updates=episode_len,episodes_without_event=sum(r['admitted_events']==0 for r in episodes.values()),
        totals=totals,goals_per_original_episode={k:v/n for k,v in goals.items()},total_goals_per_original_episode=totals['total_goals']/n,
        goal_allocation_fraction={k:v/totals['total_goals'] if totals['total_goals'] else None for k,v in goals.items()},
        updates_by_phase=updates,phase_update_percent={k:100*v/(n*episode_len) for k,v in updates.items()},
        aggregate_update_percent={k:100*v/(n*episode_len) for k,v in counts.items()},
        goals_per_100_corresponding_updates={k:100*goals[k]/counts[k] if counts[k] else None for k in counts},
        schedule=schedule,free_windows_reached=len(free['rows']),free_windows_with30updates=sum(r['free_updates']==30 for r in free['rows']),
        event_goal_hits_by_end_phase=ending,episode_rows=[episodes[k] for k in sorted(episodes)])


def training_counters(data):
    runtime=data['runtime_snapshot']
    return dict(phase_counts=runtime['clock']['phase_counts'].sum(0).cpu().numpy().astype(np.int64),
        free_hits=int(runtime['input_audit']['free_goal_hits'].sum()),
        forced_hits=int(runtime['input_audit']['forced_goal_hits'].sum()),
        schedule={k:int(v.sum()) for k,v in runtime['clock_stats'].items()})


def training_difference(before,after,expected_updates):
    phase=after['phase_counts']-before['phase_counts'];assert (phase>=0).all() and phase.sum()==expected_updates
    free=after['free_hits']-before['free_hits'];forced=after['forced_hits']-before['forced_hits']
    assert min(free,forced)>=0
    updates=dict(zip(PHASES,phase.tolist()));forced_updates=sum(updates[p] for p in FORCED)
    schedule={k:after['schedule'][k]-v for k,v in before['schedule'].items()}
    assert min(schedule.values())>=0
    return dict(decision_updates=expected_updates,free_goal_hits=free,forced_goal_hits=forced,
        free_goals_per_100_free_updates=100*free/updates['free_blind'] if updates['free_blind'] else None,
        forced_goals_per_100_forced_updates=100*forced/forced_updates if forced_updates else None,
        updates_by_phase=updates,phase_update_percent={k:100*v/expected_updates for k,v in updates.items()},schedule=schedule,
        total_training_goals=None,waiting_visible_training_goals=None,
        missing_quantities='Cumulative total/visible reward is not saved; recent score means are not used to reconstruct it')


def collect(require_complete=False,include_training=True):
    evaluations=[];training=[];available={}
    for condition,worker in WORKERS.items():
        folder=ROOT/'results/ring_cloud_artifacts'/worker
        found=[]
        for stage in range(5):
            path=folder/('baseline.json' if stage==0 else f'stage{stage}_evaluation.json')
            if not path.exists():continue
            data=json.loads(path.read_text());assert data['clean_no_noise'] and data['max_delay']==60
            assert tuple(data['variants'])==VARIANTS
            for variant in ('blind_native','seen_native'):
                evaluations.append(dict(condition=condition,stage=stage,learning_frames=stage*STAGE_FRAMES,
                    max_training_delay=None if stage==0 else DELAYS[stage-1],max_evaluation_delay=data['max_delay'],
                    source=str(path.relative_to(ROOT)),source_sha256=sha(path),**summarize_variant(data,variant)))
            found.append(stage)
        available[condition]=dict(saved_stages=found,missing_stages=sorted(set(range(5))-set(found)))
        if require_complete:assert found==list(range(5)),condition
        if include_training and (folder/'after_warmup.pt').exists():
            import torch
            previous_path=folder/'after_warmup.pt';data=torch.load(previous_path,map_location='cpu',weights_only=True)
            assert data['extra_frames']==0 and data['total_updates']==200
            previous=training_counters(data);del data
            for stage,delay in enumerate(DELAYS,1):
                path=folder/f'stage{stage}_delay{delay}.pt'
                if not path.exists():break
                data=torch.load(path,map_location='cpu',weights_only=True)
                assert data['extra_frames']==stage*STAGE_FRAMES and data['total_updates']==200+stage*489
                current=training_counters(data);del data
                training.append(dict(condition=condition,stage=stage,max_training_delay=delay,
                    from_checkpoint=str(previous_path.relative_to(ROOT)),from_sha256=sha(previous_path),
                    to_checkpoint=str(path.relative_to(ROOT)),to_sha256=sha(path),
                    warmup_excluded=True,**training_difference(previous,current,STAGE_FRAMES//4)))
                previous=current;previous_path=path
    return dict(scope='Read-only descriptive accounting; all saved fixed endpoints and all original evaluation episodes retained; active curriculum unchanged',
        analysis_source_sha256=sha(__file__),
        evaluation_scope='Clean recurrent states, native physical randomness, fixed final-task delay30–60 updates for baseline and every stage',
        training_scope='Exact differences between saved cumulative counters at consecutive fixed checkpoints; first interval starts after200 zero-LR warmup updates; ongoing noisy training at its own curriculum delay',
        available=available,evaluations=evaluations,training_intervals=training,
        inference='One original training-seed pair. Ratios are exposure-normalized descriptions, not independent-seed significance tests. Evaluation allocation is separate from training allocation.')


def render(data):
    label=lambda r:f"{'Grid' if r['condition']=='gc100' else 'PGC'} / {'base' if r['stage']==0 else 'S'+str(r['stage'])} / {'blind' if r['variant'].startswith('blind') else 'seen'}"
    fmt=lambda value:'—' if value is None else f'{value:.3f}'
    blind=[r for r in data['evaluations'] if r['variant']=='blind_native']
    reward_share=[100*r['goal_allocation_fraction']['waiting_visible'] for r in blind if r['totals']['total_goals']]
    time_share=[r['aggregate_update_percent']['waiting_visible'] for r in blind]
    allocation=(f"In the saved blind evaluations with at least one goal, waiting/visible navigation accounts for **{min(reward_share):.1f}–{max(reward_share):.1f}% of goals**, while occupying {min(time_share):.1f}–{max(time_share):.1f}% of updates. Whole-episode reward can therefore remain substantial while free blind navigation is weak."
        if reward_share else 'No blind-evaluation goals are recorded; goal-allocation fractions are undefined. All zero-goal endpoints remain in the tables.')
    lines=['# Where rewards and opportunities occur','',
        '**Read-only audit of every saved fixed evaluation endpoint.** All original episodes are retained, including those with no discovery or no admitted event. Clean evaluation always uses the final task’s 30–60-update turn delay; early-stage training has shorter delays. These are separate measurements.','',
        allocation,'',
        '| Model / endpoint / vision | Total goals/episode | Free | Forced | Waiting + visible | Free goals/100 free updates |',
        '|---|---:|---:|---:|---:|---:|']
    for r in data['evaluations']:
        g=r['goals_per_original_episode'];rate=r['goals_per_100_corresponding_updates']['free']
        lines.append(f"| {label(r)} | {r['total_goals_per_original_episode']:.3f} | {g['free']:.3f} | {g['forced']:.3f} | {g['waiting_visible']:.3f} | {fmt(rate)} |")
    lines+=['','Free means the agent-controlled 30-update test window. Forced includes both labelled turns, the forward probe, and the left-turn delay. Waiting + visible includes initial goal discovery, native visible navigation between events, and each event’s visible refresh. The JSON separates waiting and visible rewards.','',
        '| Model / endpoint / vision | Free updates % | Forced updates % | Waiting + visible % | Admitted | Completed without interruption | Reward-ended | Episode-ended |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['evaluations']:
        p=r['aggregate_update_percent'];s=r['schedule']
        lines.append(f"| {label(r)} | {p['free']:.1f} | {p['forced']:.1f} | {p['waiting_visible']:.1f} | {s['admitted']} | {s['completed']} | {s['interrupted_reward']} | {s['interrupted_done']} |")
    lines+=['','A reward ends its event, including a successful free-navigation hit. “Completed without interruption” means the full event ended without a goal hit; it is not a success count. Exact counts/shares for all seven phases, free-window completion, waiting reasons, and every episode are retained in [the JSON](summary.json).','',
        '**Separate training counter differences, with warmup excluded:**','',
        '| Model / stage | Training delay range | Free goals | Free updates | Free goals/100 free updates | Forced goals/100 forced updates |',
        '|---|---:|---:|---:|---:|---:|']
    for r in data['training_intervals']:
        delay=r['max_training_delay']
        lines.append(f"| {'Grid' if r['condition']=='gc100' else 'PGC'} / S{r['stage']} | {(delay+1)//2}–{delay} | {r['free_goal_hits']} | {r['updates_by_phase']['free_blind']} | {fmt(r['free_goals_per_100_free_updates'])} | {fmt(r['forced_goals_per_100_forced_updates'])} |")
    lines+=['','These training rates come from exact cumulative hit/phase-counter differences at fixed checkpoints, starting after warmup. They include training state noise and the stage’s own delay. Total training goal counts and their waiting/visible allocation are unavailable from these counters; recent score averages are not substituted. Training and clean evaluation also differ in policy evolution, state noise and environment samples, so their difference is not a controlled delay-only effect.','',
        'Reproduce with `OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/summarize_reorientation_reward_allocation.py`; add `--require-complete` after both four-stage runs finish. Only baseline and the four prescribed stage endpoints are considered; latest/intermediate checkpoints are ignored. No training settings or frozen mechanism sources are changed.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-complete',action='store_true');p.add_argument('--without-training',action='store_true');args=p.parse_args()
    data=collect(args.require_complete,not args.without_training);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(data,indent=2)+'\n');render(data)
    print(json.dumps(dict(available=data['available'],evaluation_rows=len(data['evaluations']),training_intervals=len(data['training_intervals']))))


if __name__=='__main__':main()
