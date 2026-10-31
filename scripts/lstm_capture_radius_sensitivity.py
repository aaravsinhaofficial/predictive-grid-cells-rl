"""First-return capture-radius counterfactual on already-recorded PGC trajectories.

No model runs. Original discovery, goal snapshot, first return spawn and policy
are held fixed. Before the first crossing, a changed reward criterion cannot
change the observed behavior. All thresholds are >= the actual 0.30 m radius,
so their first crossings are observable before any original return teleport.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_navigation_phases_1B'
RADII = np.array([.30, .35, .40, .45, .60])
SEEDS = tuple(range(1,9))
EPISODE_LEN, N_ENVS, WINDOW = 1350, 64, 300
DT = 4 / 60
VARIANTS = ('intact', 'goal_grid')


def estimate(values):
    values=np.asarray(values,float)
    return dict(values=values.tolist(),mean=float(values.mean()),sem=float(stats.sem(values)))


def zero_test(values):
    d=np.asarray(values,float);se=stats.sem(d)
    ci=stats.t.interval(.95,len(d)-1,loc=d.mean(),scale=se) if se else [d.mean(),d.mean()]
    permutations=np.array([np.mean(d*s) for s in itertools.product((-1,1),repeat=len(d))])
    return dict(n=len(d),differences=d.tolist(),difference=float(d.mean()),ci95=list(map(float,ci)),
                paired_t_p=float(stats.ttest_1samp(d,0).pvalue) if se else (1. if not d.mean() else 0.),
                exact_signflip_p=float(np.mean(np.abs(permutations)>=abs(d.mean())-1e-12)))


def extract(trace, fields, variants):
    ix={str(f):i for i,f in enumerate(fields)}
    results={}
    event_checks=0
    for v in VARIANTS:
        j=list(variants).index(v);x=trace[:,j*N_ENVS:(j+1)*N_ENVS]
        rows=[]
        for episode in range(len(x)//EPISODE_LEN):
            for slot in range(N_ENVS):
                s=x[episode*EPISODE_LEN:(episode+1)*EPISODE_LEN,slot]
                hits=np.flatnonzero(s[:,ix['hit']]>0)
                if not len(hits) or hits[0]>=EPISODE_LEN-WINDOW:continue
                start=int(hits[0]+1)
                assert start+WINDOW<=EPISODE_LEN
                next_hits=hits[(hits>=start)&(hits<start+WINDOW)]
                observed_return=bool(len(next_hits))
                stop=int(next_hits[0]+1) if observed_return else start+WINDOW
                z=s[start:stop]
                distances=z[:,ix['distance_after']]
                # distance_after uses the old goal and pre-teleport hit_pos.
                np.testing.assert_array_equal(distances<.30,z[:,ix['hit']]>0)
                assert np.all(z[:,ix['known']]>0)
                assert np.all(z[:,ix['goal_count']]==1)
                event_checks+=len(z)
                times=[];captured=[]
                for radius in RADII:
                    crossings=np.flatnonzero(distances<radius)
                    capture=bool(len(crossings))
                    elapsed=(int(crossings[0])+1)*DT if capture else WINDOW*DT
                    times.append(elapsed);captured.append(capture)
                    if observed_return:assert capture
                assert times[0]==(stop-start)*DT
                assert captured[0]==observed_return
                assert all(a>=b-1e-12 for a,b in zip(times,times[1:]))
                rows.append(dict(episode=episode,slot=slot,first_goal_step=int(hits[0]),
                    first_spawn_distance=float(z[0,ix['distance_before']]),
                    capped_times_seconds=times,captured_within_20s=captured))
        results[v]=rows
    key=lambda r:(r['episode'],r['slot'],r['first_goal_step'],r['first_spawn_distance'])
    assert [key(r) for r in results['intact']]==[key(r) for r in results['goal_grid']]
    return results,event_checks


def main():
    protocol=dict(seeds=SEEDS,radii_m=RADII.tolist(),variants=VARIANTS,
        eligibility='First original goal discovery before decision1050, leaving exactly300 decisions (20 seconds) observable after its respawn; same837episodecohort as the previous first-return endpoint.',
        endpoint='First post-step pre-teleport distance strictly below radius, counted as (crossing_index+1)/15 seconds. If no crossing,20seconds. Scan ends at original first-return hit or20seconds. Initial spawn is held fixed, even if it lies inside an enlarged radius; reward still requires a post-step check.',
        primary='One exploratory paired-training-seed contrast: [ordinary-minus-predictive capped-time penalty at0.30m] minus [same penalty at0.45m]. Specified before inspecting this radius analysis; positive means enlargement reduces the predictive-goal advantage. Paired t interval/test and exhaustive sign-flip test. No multiplicity correction needed within this one-contrast family; the broader investigation is exploratory.',
        other_points='0.35,0.40,0.60m curve points and all per-radius means are descriptive.',
        causal_scope='Counterfactual first-return reward criterion conditional on original discovery, goal message and spawn. Before the first threshold crossing the policies receive the original inputs, zero return reward, and original history. This does not simulate a new arena with larger goals during exploration, changed spawn rejection, or subsequent repeated returns.',
        checks=['0.30m distances must match actual hit flags exactly before first-return teleport.',
                '0.30m times and episode eligibility must reproduce prior first-return summaries.',
                'Matched variants share original discovery times and first-spawn distances.',
                'Larger-radius capture times cannot increase within a recorded trajectory.'])
    # Persist chosen contrast and endpoint before loading any radius outcomes.
    (OUT/'radius_sensitivity_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    prior=json.loads((OUT/'summary.json').read_text())
    sources={};by_seed={};checks=0
    for seed in SEEDS:
        path=OUT/f'pgc33_s{seed}_trace.npz'
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            records,nchecks=extract(z['trace'],z['fields'],z['variants'])
        checks+=nchecks
        n=len(records['intact'])
        previous_count=prior['comparisons']['first_return_goal_deshift_minus_intact']['eligible_counts'][seed-1]
        assert n==previous_count
        row=dict(n_eligible=n,rows=records,means={},capture_fractions={},spawn_inside_counts=[])
        for v in VARIANTS:
            times=np.array([r['capped_times_seconds'] for r in records[v]])
            captures=np.array([r['captured_within_20s'] for r in records[v]])
            row['means'][v]=times.mean(0).tolist();row['capture_fractions'][v]=captures.mean(0).tolist()
            expected=prior['estimates'][f'pgc33/{v}']['first_return_20s']['capped_return_seconds']['values'][seed-1]
            assert abs(times[:,0].mean()-expected)<1e-12
            previous_rows=json.loads((OUT/f'pgc33_s{seed}.json').read_text())['phase_analysis'][v]['first_return_20s_rows']
            for new,old in zip(records[v],previous_rows):
                assert (new['episode'],new['slot'],new['first_goal_step'])==(old['episode'],old['slot'],old['first_goal_step'])
                assert new['capped_times_seconds'][0]==old['capped_return_seconds']
        spawns=np.array([r['first_spawn_distance'] for r in records['intact']])
        row['spawn_inside_counts']=[int((spawns<radius).sum()) for radius in RADII]
        row['penalty_seconds']=(np.array(row['means']['goal_grid'])-row['means']['intact']).tolist()
        by_seed[str(seed)]=row
    estimates={v:[estimate([by_seed[str(s)]['means'][v][i] for s in SEEDS]) for i in range(len(RADII))] for v in VARIANTS}
    penalties=[estimate([by_seed[str(s)]['penalty_seconds'][i] for s in SEEDS]) for i in range(len(RADII))]
    contrast=np.array(penalties[0]['values'])-np.array(penalties[3]['values'])
    primary=zero_test(contrast)
    primary['description']='Penalty at0.30m minus penalty at0.45m; positive means relaxing capture precision reduces the goal-deshift penalty.'
    for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    result=dict(protocol=protocol,estimates=estimates,penalties=penalties,primary_contrast=primary,by_seed=by_seed,
        source_sha256=sources,validation=dict(total_eligible_episodes=sum(r['n_eligible'] for r in by_seed.values()),
        post_step_event_checks=checks,actual_radius_events_match_exactly=True,original_first_return_times_match_exactly=True,
        eligibility_and_first_spawn_match_exactly=True,source_hashes_unchanged=True),
        spawn_inside_counts_total=np.sum([r['spawn_inside_counts'] for r in by_seed.values()],axis=0).tolist())
    (OUT/'radius_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    figure(result);report(result)
    print('Radiuscm / intact / ordinary / penalty')
    for i,radius in enumerate(RADII):print(round(radius*100),*[round(estimates[v][i]['mean'],5) for v in VARIANTS],round(penalties[i]['mean'],5))
    print('Primary contrast',primary)
    print('Validation',result['validation'],'spawninside',result['spawn_inside_counts_total'])


def figure(r):
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained')
    x=RADII*100
    for v,label,color in [('intact','Predictive goal','#D87532'),('goal_grid','Ordinary goal','#8B939B')]:
        vals=[e['mean'] for e in r['estimates'][v]];errors=[e['sem'] for e in r['estimates'][v]]
        axes[0].errorbar(x,vals,yerr=errors,marker='o',color=color,capsize=3,label=label)
    axes[0].set(xlabel='Capture radius (cm)',ylabel='First-return time, capped at 20 s',title='Relax the capture criterion')
    axes[0].legend(frameon=False,fontsize=9)
    vals=np.array([e['values'] for e in r['penalties']])
    for j in range(vals.shape[1]):axes[1].plot(x,vals[:,j],color='#B5B9BC',alpha=.5,lw=.7)
    axes[1].errorbar(x,[e['mean'] for e in r['penalties']],yerr=[e['sem'] for e in r['penalties']],color='#253442',marker='o',capsize=3)
    axes[1].axhline(0,color='#8B939B',lw=.8)
    axes[1].set(xlabel='Capture radius (cm)',ylabel='Ordinary − predictive goal (seconds)',title='Does the predictive advantage shrink?')
    for ax in axes:
        ax.axvline(45,color='#C3C7CB',ls=':',lw=1)
        ax.set_xticks(x);ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Same 837 eligible first returns; original discovery and spawn held fixed. Mean ± SEM across 8 seeds.',fontsize=8)
    for ext in ['png','pdf']:fig.savefig(OUT/f'radius_sensitivity.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def report(r):
    p=r['primary_contrast'];lo,hi=p['ci95']
    supported=p['paired_t_p']<.05 and p['exact_signflip_p']<.05
    lead=('The predictive-goal advantage becomes smaller when capture requires less precision.' if supported else
          'This test does not establish that relaxing capture precision reduces the predictive-goal advantage.')
    lines=['## Capture-radius sensitivity','',f'**{lead}**','',
        '![First-return capture time at different acceptance radii](radius_sensitivity.png)','',
        'We reused the same837 eligible first-return trajectories, keeping the original first discovery, goal message, starting position, and policy fixed. We asked when each trajectory would first satisfy a larger capture radius. Only the criterion for ending this first return changes. No new model rollouts were needed.','',
        '| Capture radius | Predictive goal | Ordinary goal | Ordinary − predictive | Original spawns inside enlarged disk |',
        '|---|---:|---:|---:|---:|']
    for i,radius in enumerate(RADII):
        lines.append(f"| {radius*100:.0f} cm | {r['estimates']['intact'][i]['mean']:.3f} s | {r['estimates']['goal_grid'][i]['mean']:.3f} s | {r['penalties'][i]['mean']:+.3f} s | {r['spawn_inside_counts_total'][i]} /837 |")
    lines+=['',f"The one preselected exploratory comparison is the penalty at30cm minus the penalty at45cm: **{p['difference']:+.3f}s**, 95% paired interval **[{lo:+.3f},{hi:+.3f}]**, paired-t p={p['paired_t_p']:.4g}, exact sign-flip p={p['exact_signflip_p']:.4g}. Positive means that widening the capture region reduces the disadvantage of ordinary goal input. Other curve points are descriptive.",'',
        'Capture uses the first **post-step** distance strictly below the radius, matching the environment reward check. A first-step capture takes1/15s, including when the fixed original spawn is already inside the enlarged disk. Distances were recorded before any pickup teleport. The30cm endpoint reproduces every original first-return time and hit event exactly. Larger-radius first crossings always occur before any original return teleport, so preceding behavior remains observable.','',
        'This is a first-return reward-criterion counterfactual conditional on the original discovery and spawn. It is **not** a new evaluation of larger goals throughout the task: discovery, respawn rejection, later rewards, and subsequent returns would also change in that experiment. It does not test a native grid-trained controller, whose discovery history differs.','',
        'The contrast and radii were selected before inspecting this analysis, after the earlier navigation results; this is an exploratory follow-up. Eight training seeds are the replicates, with equal weight. Failures remain capped at20seconds. [Numerical results](radius_sensitivity.json) · [Protocol](radius_sensitivity_protocol.json).','',
        '```sh','python scripts/lstm_capture_radius_sensitivity.py','```']
    text='\n'.join(lines)+'\n'
    for old,new in [('same837','same 837'),('at30cm','at 30 cm'),('at45cm','at 45 cm'),('takes1/15s','takes 1/15 s'),('The30cm','The 30 cm'),('at20seconds','at 20 seconds'),('/837','/ 837')]:text=text.replace(old,new)
    (OUT/'radius_sensitivity.md').write_text(text)
    readme=OUT/'README.md'
    original=readme.read_text()
    start=original.find('## Capture-radius sensitivity')
    if start>=0:
        stop=original.find('\n## ',start+3)
        original=original[:start]+(original[stop+1:] if stop>=0 else '')
    marker='## A mostly turn-or-move policy'
    assert marker in original
    readme.write_text(original.replace(marker,text+'\n'+marker,1))


if __name__=='__main__':main()
