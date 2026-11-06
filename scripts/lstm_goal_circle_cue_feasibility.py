"""First-pickup geometry sanity check for an original-task goal-memory ring.

Reads existing actual first-hit records. No simulation, training, decoding
oracle supplied to a policy, or checkpoint edits. Ground-truth pickup/arrival
stand in for PERFECT inversion of the analytic noiseless map snapshots.
"""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE=ROOT/'results/ring_cloud_artifacts/ring-20261105-rl-goal-memory-t4-01'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def phase_error(a,b):
    return np.abs(np.angle(np.exp(1j*(a-b))))


def metrics(goal,pickup,direction,radius=.65,shift=.20):
    ahead=pickup+shift*direction
    goal_phase=np.arctan2(goal[:,1],goal[:,0])
    phases={name:np.arctan2(pos[:,1],pos[:,0])for name,pos in [('pickup',pickup),('ahead',ahead)]}
    projection={name:radius*np.stack((np.cos(phase),np.sin(phase)),1)for name,phase in phases.items()}
    inward=goal-pickup
    arrival_cos=np.sum(inward*direction,axis=1)/np.linalg.norm(inward,axis=1)
    result=dict(goal=goal,pickup=pickup,arrival_direction=direction,ahead=ahead,arrival_cosine=arrival_cos,
        pickup_distance_m=np.linalg.norm(pickup-goal,axis=1),ahead_distance_m=np.linalg.norm(ahead-goal,axis=1))
    for name in phases:
        result[name+'_angle_error_deg']=np.rad2deg(phase_error(phases[name],goal_phase))
        result[name+'_projected_position']=projection[name]
        result[name+'_projected_distance_m']=np.linalg.norm(projection[name]-goal,axis=1)
    return result


def summarize(data,mask):
    if not mask.any():return dict(n=0)
    p=data['pickup_angle_error_deg'][mask];a=data['ahead_angle_error_deg'][mask]
    return dict(n=int(mask.sum()),pickup_angle_mae_deg=float(p.mean()),ahead_angle_mae_deg=float(a.mean()),
        paired_angle_improvement_deg=float((p-a).mean()),pickup_angle_median_deg=float(np.median(p)),
        ahead_angle_median_deg=float(np.median(a)),ahead_better_fraction=float(np.mean(a<p)),
        pickup_projected_distance_m=float(data['pickup_projected_distance_m'][mask].mean()),
        ahead_projected_distance_m=float(data['ahead_projected_distance_m'][mask].mean()),
        pickup_projection_inside_reward_fraction=float(np.mean(data['pickup_projected_distance_m'][mask]<.3)),
        ahead_projection_inside_reward_fraction=float(np.mean(data['ahead_projected_distance_m'][mask]<.3)),
        raw_pickup_distance_m=float(data['pickup_distance_m'][mask].mean()),
        raw_ahead_distance_m=float(data['ahead_distance_m'][mask].mean()),
        mean_arrival_cosine=float(data['arrival_cosine'][mask].mean()))


def analyze(path,condition,out,protocol):
    r=json.loads(path.read_text());hits=r['first_hit_geometry'];rows=r['variants']['flash']['rows']
    assert r['all_first_discoveries_identical'] is True
    assert len(rows)==r['episodes_per_variant']==64
    episode_len=1350
    identifiers=[(int(h['slot']),int(h['step'])//episode_len)for h in hits]
    expected=[(int(x['slot']),int(x['episode']))for x in rows if x['first_goal_step']>=0]
    assert len(set(identifiers))==len(identifiers)
    assert set(identifiers)==set(expected)
    goal=np.asarray([h['goal']for h in hits]);pickup=np.asarray([h['hit_pos']for h in hits]);direction=np.asarray([h['hit_dir']for h in hits])
    assert np.allclose(np.linalg.norm(goal,axis=1),.65,atol=1e-6)
    assert np.allclose(np.linalg.norm(direction,axis=1),1,atol=1e-6)
    assert np.max(np.linalg.norm(pickup-goal,axis=1))<.3
    assert np.min(np.linalg.norm(pickup,axis=1))>.34
    data=metrics(goal,pickup,direction)
    all_mask=np.ones(len(hits),dtype=bool)
    bins={name:summarize(data,(data['arrival_cosine']>=low)&(data['arrival_cosine']<high))for name,low,high in [('glancing_lt_0.5',-1.001,.5),('oblique_0.5_to_0.8',.5,.8),('more_direct_ge_0.8',.8,1.001)]}
    assert sum(v['n']for v in bins.values())==len(hits)
    result=dict(actor_condition=condition,original_training_seed=1,evaluation_seed=r['seed'],
        source=str(path),source_sha256=sha(path),stage='baseline_original_frozen_1B_actor',
        episodes=len(rows),episodes_with_first_pickup=len(hits),episodes_without_discovery=len(rows)-len(hits),
        all_discovered_first_pickups_included=True,no_outlier_exclusions=True,
        geometry=summarize(data,all_mask),arrival_bins=bins,
        ideal_grid_arrival_control=dict(angle_mae_deg=float(data['ahead_angle_error_deg'].mean()),
            exact_same_point_as_predictive=True,measured_history_reconstruction=False,
            explanation='If Grid has the same exact arrival direction, x+0.20d is algebraically identical. Previous positions were not saved, so this is an ideal information-matched identity, not an empirical decoder/history result.'),
        limitation='Ground-truth decoded-position-equivalent geometry only: map inversion has not been performed; true goals are used only to score estimates. One training seed, no inferential population-level statistics.')
    write_json(out/f'{condition}_s1.json',result)
    np.savez_compressed(out/f'{condition}_s1_pickups.npz',**data,episode=np.array([a[1]for a in identifiers]),slot=np.array([a[0]for a in identifiers]))
    return result


def report(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    for i,r in enumerate(results):
        g=r['geometry'];axes[0].bar(i-.18,g['pickup_angle_mae_deg'],.34,color='#4C83B5',label='Pickup'if i==0 else None)
        axes[0].bar(i+.18,g['ahead_angle_mae_deg'],.34,color='#D97736',label='20 cm ahead'if i==0 else None)
        bins=list(r['arrival_bins'].values());axes[1].plot(range(3),[b.get('paired_angle_improvement_deg',np.nan)for b in bins],marker='o',label=r['actor_condition'])
    labels=['Grid actor'if r['actor_condition']=='gc100'else'PGC actor'for r in results]
    axes[0].set(xticks=range(len(results)),xticklabels=labels,ylabel='Mean goal-angle error (degrees)',title='A better point to remember?');axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(3),xticklabels=['Glancing','Oblique','More direct'],ylabel='Pickup error − ahead error (degrees)',title='Does arrival geometry matter?');axes[1].axhline(0,color='.7',lw=.8);axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'goal_angle_cue.png',dpi=190);plt.close(fig)
    lines=['# Is there a useful goal-angle cue to store?','',
        '**This is a geometry feasibility check using actual first pickups from the original frozen RL agents in the same square arena, with goal centers restricted to a0.65m circle. No ring was built or trained.**','',
        '![First-pickup goal-angle estimates](goal_angle_cue.png)','',
        'For each actual pickup, we compare the angle of the ordinary pickup point with the angle of the point20cm along arrival—the point represented by predictive goal cells. Both estimates are projected onto the known goal circle. True goal coordinates are used only to measure error.','',
        '**Stage limitation:** pickup and arrival coordinates come from evaluation records. They represent the ideal result of perfectly inverting the noiseless spatial maps; map inversion has not yet been tested. These coordinates were never supplied to an actor by this analysis.','',
        '|Original frozen actor|Actual pickups / episodes|Pickup angle error|Ahead angle error|Ahead point better|','|---|---:|---:|---:|---:|']
    for r in results:
        g=r['geometry'];lines.append(f'|{r["actor_condition"]}, seed1|{r["episodes_with_first_pickup"]}/{r["episodes"]}|{g["pickup_angle_mae_deg"]:.2f}°|{g["ahead_angle_mae_deg"]:.2f}°|{100*g["ahead_better_fraction"]:.1f}%|')
    lines+=['','Every discovered first pickup is included. Episodes without discovery are reported in the denominator; they cannot supply a goal cue. The same pickup set is used for both estimates within each actor. First discoveries precede every goal-message intervention.','',
        '**Grid with arrival history:** if ordinary-grid history recovers the same arrival direction, it produces exactly the same ahead point. That equality is geometric, not a measured history-decoding result: these baseline records did not save the preceding positions. A later implementation must test the actual map/history reconstruction.','',
        '**What this can show:** whether a predictive snapshot contains a more accurate circular goal estimate worth storing. It does not show that a ring exists in the original LSTM, that the actor reads this angle, or that predictive input changes cue-free ring stability. One original training seed per population is a feasibility sample; pickups are not independent trained-network replicates, so no population-significance test is reported.','',
        'The arrival-angle bins are descriptive, fixed before these files were inspected: inward cosine below0.5,0.5–0.8, and at least0.8. Detailed paired errors, projected-point reward-disk containment, missing discoveries, and source hashes are in the JSON files.','',
        'Reproduce with `python scripts/lstm_goal_circle_cue_feasibility.py --source results/ring_cloud_artifacts/ring-20261105-rl-goal-memory-t4-01`. Original environment and evaluator sources remain unchanged.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE);ap.add_argument('--out',type=Path,default=ROOT/'results/original_goal_circle_cue_feasibility');ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(stage='Ground-truth decoded-position-equivalent geometry feasibility; not a map-decoder or ring test',
        source=str(args.source),actor_seeds=[1],conditions=['gc100','pgc33'],baseline_only=True,
        evaluation_episodes_per_actor=64,include='Every actual first pickup; failed discoveries retained in inclusion counts; no exclusions',
        primary_metric='Mean paired reduction in absolute circular goal-angle error, pickup versus20cm ahead, separately within each original actor',
        circle_radius_m=.65,pgc_shift_m=.20,arrival_cosine_bins=[-1.001,.5,.8,1.001],
        control='Ideal Grid+samearrival equality only; actual history unavailable in these existing records',
        statistics='No inferential significance tests: one preselected original training seed per population',source_sha256=sha(__file__))
    if not(args.out/'protocol.json').exists():write_json(args.out/'protocol.json',protocol)
    if args.freeze_only:return
    results=[]
    for condition in ['gc100','pgc33']:
        path=args.source/condition/'baseline.json'
        if path.exists():results.append(analyze(path,condition,args.out,protocol))
    if not results:raise FileNotFoundError('No baseline geometry files collected locally yet')
    write_json(args.out/'summary.json',dict(status='complete'if len(results)==2 else'partial_waiting_for_second_actor',models=results))
    report(results,args.out)
    for r in results:print(r['actor_condition'],r['geometry'])

if __name__=='__main__':main()
