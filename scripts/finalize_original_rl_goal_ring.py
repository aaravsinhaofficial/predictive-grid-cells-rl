"""Frozen-family statistics and simple figure for the original-actor ring study."""
import argparse
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy import stats
from original_rl_goal_ring import sha,save_json,VARIANTS


def comparison(differences):
    d=np.asarray(differences,dtype=float);n=len(d);mean=float(d.mean());se=float(stats.sem(d))
    interval=stats.t.interval(.95,n-1,loc=mean,scale=se)if se else(mean,mean)
    permutations=np.array([np.mean(d*np.array(sign))for sign in itertools.product((-1,1),repeat=n)])
    return dict(n_training_seeds=n,seed_differences_seconds=d.tolist(),mean_difference_seconds=mean,
        ci95_seconds=list(map(float,interval)),paired_t_p=float(stats.ttest_1samp(d,0).pvalue)if se else(1. if mean==0 else 0.),
        exact_two_sided_signflip_p=float(np.mean(np.abs(permutations)>=abs(mean)-1e-12)))


def holm(comparisons):
    for source,target in [('paired_t_p','holm_t_p'),('exact_two_sided_signflip_p','holm_exact_p')]:
        running=0.
        for i,k in enumerate(sorted(comparisons,key=lambda k:comparisons[k][source])):
            running=max(running,min(1.,(len(comparisons)-i)*comparisons[k][source]));comparisons[k][target]=running


def initial_commands(result):
    rows=result['first_postpickup_commands'];by={v:{(r['slot'],r['episode']):r for r in rows if r['variant']==v}for v in VARIANTS[1:]}
    eligible={(r['slot'],r['episode'])for r in result['variants']['ring_ahead']['rows']if 0<=r['first_goal_step']<1050}
    values={}
    for name,angle in [('ring_ahead_plus45',np.pi/4),('ring_ahead_minus45',-np.pi/4)]:
        tv=[]
        for key in sorted(eligible):
            a=by['ring_ahead'][key];b=by[name][key]
            assert np.allclose(a['position'],b['position'],atol=1e-6)and abs(a['heading']-b['heading'])<1e-6
            za,zb=np.array(a['ring_target']),np.array(b['ring_target'])
            delta=np.arctan2(np.cross(za,zb),np.dot(za,zb))
            assert abs(delta-angle)<2e-6
            tv.append(.5*np.abs(np.array(a['probabilities'])-b['probabilities']).sum())
        values[name]=dict(eligible_matched_commands=len(tv),mean_action_total_variation=float(np.mean(tv)),
            maximum_action_total_variation=float(np.max(tv)))
    return values


def cue_phase_accuracy(result):
    events=result['first_hit_geometry']
    goal={(r['slot'],r['episode']):np.asarray(r['goal'])for r in events}
    geometric={};actual={}
    for variant,name in [('ring_pickup','pickup'),('ring_ahead','ahead')]:
        errors=[]
        for event in events:
            point=np.asarray(event['pickup'])
            if name=='ahead':point=point+.2*np.asarray(event['arrival'])
            target=np.asarray(event['goal'])
            errors.append(np.rad2deg(abs(np.arctan2(np.cross(point,target),np.dot(point,target)))))
        geometric[name]=dict(n=len(errors),mean_absolute_error_deg=float(np.mean(errors)))
        observed=[]
        for row in result['first_postpickup_commands']:
            if row['variant']!=variant:continue
            point=np.asarray(row['ring_target']);target=goal[(row['slot'],row['episode'])]
            observed.append(np.rad2deg(abs(np.arctan2(np.cross(point,target),np.dot(point,target)))))
        actual[name]=dict(n=len(observed),mean_absolute_error_deg=float(np.mean(observed)))
    return dict(all_first_pickup_geometry=geometric,actual_rate_decoded_ring_targets=actual,
                note='All recorded first pickups enter geometry. Actual ring targets are independently read from the next actor input; any final-step pickup without a subsequent actor update is counted only in geometry. True goals score estimates and never enter inference.')


def load_and_validate(out):
    protocol=json.loads((out/'confirmation/protocol.json').read_text())
    for path,expected in protocol['source_sha256'].items():assert sha(ROOT/path)==expected,(path,'source changed')
    models=[]
    for seed in range(2,9):
        path=out/f'confirmation/seed{seed}/results.json';result=json.loads(path.read_text())
        assert result['original_actor']==f'gc100_lstm_s{seed}_1000M'
        assert result['checkpoint_sha256']==protocol['original_checkpoint_sha256'][str(seed)]
        assert result['episodes_per_variant']==64 and result['eval_seed']==63001
        assert result['validation']['all_first_discoveries_identical']
        assert result['original_weights_unchanged']
        reference=None
        for variant in VARIANTS:
            rows=result['variants'][variant]['rows'];assert len(rows)==64
            keys=[(r['slot'],r['episode'],r['first_goal_step'])for r in rows]
            if reference is None:reference=keys
            else:assert keys==reference
            eligible=[r for r in rows if 0<=r['first_goal_step']<1050]
            assert len(eligible)==result['variants'][variant]['eligible_first_returns']
            manual=np.mean([min(r['first_return_steps'],300)if r['first_return_steps']>=0 else 300 for r in eligible])
            assert abs(manual-result['variants'][variant]['eligible_first_return_capped20s_steps'])<1e-10
        assert result['validation']['decoder_live_max_pickup_error_m']<2e-5
        assert result['validation']['decoder_live_max_ahead_error_m']<2e-5
        models.append(dict(seed=seed,source_sha256=sha(path),result=result,initial_commands=initial_commands(result),cue_phase=cue_phase_accuracy(result)))
    return protocol,models


def plot(out,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    blue,orange,purple='#4C83B5','#DD7738','#896BA5'
    pilot=json.loads((out/'seed1/results.json').read_text());event=pilot['first_hit_geometry'][0]
    goal=np.array(event['goal']);pickup=np.array(event['pickup']);ahead=pickup+.2*np.array(event['arrival'])
    qx=.65*pickup/np.linalg.norm(pickup);qa=.65*ahead/np.linalg.norm(ahead)
    ax=axes[0];theta=np.linspace(0,2*np.pi,200)
    ax.plot(.65*np.cos(theta),.65*np.sin(theta),color='.75',lw=1)
    ax.add_patch(plt.Circle(goal,.3,color='#DFE9D6',alpha=.8))
    ax.scatter(*goal,facecolors='none',edgecolors='#5C9449',marker='*',s=130,linewidths=1.6,label='True goal',zorder=6)
    for q,color,label in [(qx,blue,'Pickup estimate'),(qa,orange,'Predictive estimate')]:
        ax.plot([0,q[0]],[0,q[1]],':',color=color,lw=1)
        ax.scatter(*q,s=45,color=color,label=label,zorder=4)
    ax.scatter(*pickup,color=blue,s=15);ax.scatter(*ahead,color=orange,s=15)
    ax.annotate('',ahead,pickup,arrowprops=dict(arrowstyle='->',color=orange,lw=1.5))
    ax.set_aspect('equal');ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),title='Actual goal geometry');ax.axis('off')
    ax.legend(fontsize=7,loc='lower right')
    times=np.array([[m['result']['variants'][v]['eligible_first_return_capped20s_steps']/15 for v in ('ring_pickup','ring_ahead')]for m in models])
    for row in times:axes[1].plot([0,1],row,color='.75',lw=1,zorder=1)
    axes[1].scatter(np.zeros(len(times)),times[:,0],color=blue,s=25,zorder=3)
    axes[1].scatter(np.ones(len(times)),times[:,1],color=orange,s=25,zorder=3)
    axes[1].plot([0,1],times.mean(0),color='k',lw=2,zorder=2)
    axes[1].set(xticks=[0,1],xticklabels=['Pickup cue','Predictive cue'],ylabel='First return time (s)',title='Same trained navigation policy',xlim=(-.25,1.25))
    diffs=np.array([[m['result']['variants'][v]['eligible_first_return_capped20s_steps']/15-m['result']['variants']['ring_ahead']['eligible_first_return_capped20s_steps']/15 for v in ('ring_ahead_minus45','ring_ahead_plus45')]for m in models])
    for i in range(2):
        axes[2].bar(i,diffs[:,i].mean(),color=purple,alpha=.5,width=.55)
        axes[2].scatter(np.full(len(diffs),i)+np.linspace(-.08,.08,len(diffs)),diffs[:,i],color=purple,s=20)
    axes[2].axhline(0,color='.65',lw=.8);axes[2].set(xticks=[0,1],xticklabels=['−45°','+45°'],ylabel='Extra first-return time (s)',title='Rotate the stored goal')
    for ax in axes[1:]:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'goal_memory_mechanism.png',dpi=200);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/original_rl_goal_ring');args=ap.parse_args()
    protocol,models=load_and_validate(args.out)
    means={v:{name:float(np.mean([m['result']['variants'][v][name]for m in models]))for name in ['mean_score','eligible_first_return_within20s_fraction','eligible_first_return_capped20s_steps']}for v in VARIANTS}
    for row in means.values():row['eligible_first_return_capped20s_seconds']=row['eligible_first_return_capped20s_steps']/15
    primary=comparison([(m['result']['variants']['ring_pickup']['eligible_first_return_capped20s_steps']-m['result']['variants']['ring_ahead']['eligible_first_return_capped20s_steps'])/15 for m in models])
    secondary={name:comparison([(m['result']['variants'][name]['eligible_first_return_capped20s_steps']-m['result']['variants']['ring_ahead']['eligible_first_return_capped20s_steps'])/15 for m in models])for name in ['ring_ahead_plus45','ring_ahead_minus45']};holm(secondary)
    pilot=json.loads((args.out/'seed1/results.json').read_text())
    payload=dict(stage='Complete seven-heldout-original-actor replication; seed1 descriptivepilot excluded fromtests',
        primary=primary,secondary_holm_family=secondary,mean_outcomes=means,protocol=protocol,
        model_rows=[dict(seed=m['seed'],checkpoint_sha256=m['result']['checkpoint_sha256'],results_sha256=m['source_sha256'],
            eligible_first_returns=m['result']['variants']['ring_ahead']['eligible_first_returns'],
            initial_commands=m['initial_commands'],cue_phase_accuracy=m['cue_phase'],mean_scores={v:m['result']['variants'][v]['mean_score']for v in VARIANTS},
            primary_return_seconds={v:m['result']['variants'][v]['eligible_first_return_capped20s_steps']/15 for v in VARIANTS},
            decoder_max_error_m=max(m['result']['validation']['decoder_live_max_pickup_error_m'],m['result']['validation']['decoder_live_max_ahead_error_m']))for m in models],
        descriptive_seed1={v:dict(mean_score=pilot['variants'][v]['mean_score'],all_discovered_firstreturn_seconds=pilot['variants'][v]['first_return_capped20s_steps']/15)for v in VARIANTS},
        claim='Predictive map cue initializes a more useful goal location in a designed ring feeding the same original trained RL actor. Intrinsic ring dynamics are identical; radial contraction is not functionally exercised by noiseless unit-radius navigation/readout.')
    payload['mean_cue_phase_mae_deg']={name:float(np.mean([m['cue_phase']['actual_rate_decoded_ring_targets'][name]['mean_absolute_error_deg']for m in models]))for name in ('pickup','ahead')}
    save_json(args.out/'confirmation/summary.json',payload);plot(args.out,models)
    p=primary
    lines=['# A goal-memory ring connected to the original navigation agent','',
        '**This model uses the original trained RL policies, original square arena, original hexagonal spatial maps, and original six actions. The added goal-memory ring is explicitly designed.**','',
        '![Goal geometry, return behavior, and memory-phase intervention](goal_memory_mechanism.png)','',
        'At a real reward pickup, ordinary map cells describe where the agent entered the reward disk. Predictive cells describe the point 20 cm farther along its arrival direction. We decode those observed rates, store the estimated goal angle in a 2D ring, and regenerate the original 512-cell goal-input code from that memory. The frozen original LSTM chooses every action. No hidden goal coordinate, true arrival angle, or external navigation controller enters the model.','',
        'We first checked original Grid actor seed 1, then froze the protocol and tested the remaining seven original Grid actors. Every cue condition uses the **same actor weights and current-code inputs**. The goal prior is a 0.65 m circle, so goal location is one circular variable.','',
        '|Goal input|First return time ↓|Episode reward ↑|','|---|---:|---:|']
    names={'native_saved_code':'Original saved goal code','ring_pickup':'Ring, ordinary pickup cue','ring_ahead':'Ring, predictive ahead cue','ring_ahead_plus45':'Predictive ring, phase +45°','ring_ahead_minus45':'Predictive ring, phase −45°'}
    for v in VARIANTS:lines.append(f'|{names[v]}|{means[v]["eligible_first_return_capped20s_seconds"]:.2f} s|{means[v]["mean_score"]:.2f}|')
    lines+=['',f'**Primary result:** predictive initialization gives a mean first-return improvement of **{p["mean_difference_seconds"]:.3f} s**. Evidence is borderline and depends on the statistical method: exact two-sided sign-flip p={p["exact_two_sided_signflip_p"]:.5f}, paired t p={p["paired_t_p"]:.5f}, and the 95% seed-paired interval [{p["ci95_seconds"][0]:.4f}, {p["ci95_seconds"][1]:.4f}] s crosses zero. The statistical unit is the trained policy: **seven previously untested seeds**, not episodes. Seed 1 remains a separate pilot.','',
        f'**Descriptive reward result:** the predictive cue increases episode reward by {means["ring_ahead"]["mean_score"]-means["ring_pickup"]["mean_score"]:.2f} points on average, with improvement in all seven heldout actors. Reward was not the new confirmatory primary endpoint; this consistent pattern is reported descriptively.','',
        f'**The stored phase is more accurate:** actual rate-decoded goal memories have mean angular error {payload["mean_cue_phase_mae_deg"]["pickup"]:.2f}° from ordinary pickup cues versus {payload["mean_cue_phase_mae_deg"]["ahead"]:.2f}° from predictive ahead cues. Each trained actor is weighted equally; per-seed errors and inclusion counts are in the summary.','',
        'The primary endpoint includes only episodes whose shared first discovery occurs by zero-based step 1049, giving 300 subsequent updates (20 s). Each failed eligible return contributes 20 s. This criterion is fixed by a pre-intervention event; every episode remains in the reward totals. A one-step timing correction and JSON serialization fix occurred before any heldout evaluation, with the original source/protocol archived.','',
        '**Causal phase test:** after the same first pickup, we rotate only the stored ring angle by ±45°. The actor, physical state, vision and current spatial code remain unchanged at that point. The ensuing policy and environment dynamics are real closed-loop RL behavior.']
    lines.append('')
    for name,comp in secondary.items():lines.append(f'- {names[name]} adds {comp["mean_difference_seconds"]:.2f} s to first-return time; Holm-corrected exact p = {comp["holm_exact_p"]:.5f}.')
    lines+=['','The two phase-pulse tests form a separate prespecified Holm family. The first actor distributions after pickup are compared at identical physical observations, before trajectories diverge. One-second paths are not used as an unbiased causal endpoint because early returns censor those records.','',
        '**What the ring does:** its stored phase persists without a cue; radial errors shrink by 0.9 per update. The phase multiplier is 1 and the radial multiplier is 0.9 for both cue sources. Those dynamics are designed, and the numerical Jacobian checks match the equations. Unit-radius writes and normalized readout mean radial contraction is not functionally exercised in the navigation comparison.','',
        '**What predictive input changes:** the remembered goal estimate, rather than intrinsic ring stability. This demonstrates a concrete route from actual predictive map activity → stored goal location → the original trained navigation policy. It does not discover a ring inside the original LSTM or show that a ring is uniquely necessary for memory; simple persistent phase storage would share the relevant cue-free behavior.','',
        'The exact known-map inverse starts from observed rates and an independent position dictionary, then refines the fit. Independent random-position tests and all live pickup reconstructions pass; true positions are diagnostic checks, never inverse initializers. Original checkpoint/config and frozen evaluator hashes are verified.','',
        '**Information-matched history control:** ordinary spatial-code history can reconstruct the same arrival cue. In 64 actual first pickups, a rate-only history decoder recovered an ahead point within 9.6e−8 m and regenerated the ring goal code within 1.2e−6 activity units of the PGC-derived version. Thus predictive cells package information from arrival history directly into the saved snapshot; they do not provide uniquely unavailable information. This does not claim that the original Grid LSTM learned the explicit decoder/history calculation. [Measured history control](history_control/README.md).','',
        '[Frozen confirmation protocol](confirmation/protocol.json) · [Complete results](confirmation/summary.json) · [Pre-outcome correction](confirmation/PREOUTCOME_AMENDMENT.md) · [Original frozen-agent cue geometry](../original_goal_circle_cue_feasibility/README.md) · [Original LSTM state-family test](../original_goal_bearing_manifold/pilot/README.md)','',
        'Reproduce: `python scripts/original_rl_goal_ring_cohort.py`, then `python scripts/finalize_original_rl_goal_ring.py`. Original checkpoints are required.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=primary,secondary=secondary,means=means),indent=2))

if __name__=='__main__':main()
