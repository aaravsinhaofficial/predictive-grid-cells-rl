"""Two matched gate-recovery panels with paired training-seed bootstrap CIs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=int,default=50000)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_attractor/goal_gate_behavior')
    a=p.parse_args()
    behavior_path=a.out/'summary.json'
    neural_path=ROOT/'results/ring_attractor/gate_stability/summary.json'
    behavior=json.loads(behavior_path.read_text())['models'];neural=json.loads(neural_path.read_text())['models']
    by_behavior={r['seed']:r for r in behavior};by_neural={r['seed']:r for r in neural}
    seeds=sorted(by_behavior)
    assert seeds==list(range(1,9)) and all(s in by_neural for s in seeds)
    condition_names=['Ordinary goal','Forget','Candidate + output','Intact']
    bnames=['goal_ordinary','restore_forget','restore_candidate_output','intact']
    nnames=['plain','forget','candidate_output','native']
    raw={'memory':np.array([[by_neural[s]['measurements']['0.0001'][k]['c_error']['60'] for k in nnames] for s in seeds]),
         'navigation':np.array([[by_behavior[s]['variants'][k]['mean_score'] for k in bnames] for s in seeds])}
    # Identical resampled seed indices in both assays preserve their pairing.
    indices=np.random.default_rng(a.seed).integers(0,len(seeds),(a.bootstrap,len(seeds)))
    summary=dict(seeds=seeds,conditions=condition_names,bootstrap_resamples=a.bootstrap,bootstrap_seed=a.seed,
        method='Ratio of seed means; 95% percentile intervals from paired seed bootstrap. Same resampled seed indices for both assays; no bootstrap samples discarded. Ordinary/intact anchors are0/100bydefinition.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (behavior_path,neural_path)},assays={})
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.8),layout='constrained')
    colors=['#9299A1','#587DA5','#D87532','#6F573B']
    for ax,(assay,values),title in zip(axes,raw.items(),['Memory protection','Navigation']):
        mean=values.mean(0)
        # The same formula handles a lower-is-better error and higher-is-better reward.
        point=100*(mean-mean[0])/(mean[3]-mean[0])
        sampled=values[indices].mean(1);denominator=sampled[:,3]-sampled[:,0]
        assert not np.any(denominator==0)
        ratios=100*(sampled-sampled[:,:1])/denominator[:,None]
        intervals=np.quantile(ratios,[.025,.975],axis=0)
        summary['assays'][assay]=dict(raw_seed_values=values.tolist(),raw_seed_mean=mean.tolist(),
            mean_gap_recovered_percent=point.tolist(),ci95_percent=intervals.T.tolist(),
            bootstrap_denominator_opposite_sign_count=int(np.sum(np.sign(denominator)!=np.sign(mean[3]-mean[0]))))
        ax.bar(np.arange(4),point,color=colors,width=.65)
        ax.errorbar(np.arange(4),point,yerr=np.stack((point-intervals[0],intervals[1]-point)),fmt='none',color='black',capsize=4,lw=1.4)
        ax.scatter([0],[0],color=colors[0],s=20,zorder=3)
        ax.axhline(0,color='#AAAAAA',lw=.7);ax.axhline(100,color='#CCCCCC',ls='--',lw=.7)
        ax.set(title=title,ylabel='Intact–ordinary gap recovered (%)',xticks=np.arange(4),xticklabels=condition_names,
               ylim=(min(-20,float(intervals.min())-8),max(120,float(intervals.max())+8)))
        ax.tick_params(axis='x',labelrotation=30,labelsize=9);ax.spines[['top','right']].set_visible(False)
        for j in (1,2):ax.text(j,point[j]/2,f'{point[j]:.0f}%',ha='center',va='center',fontsize=11,color='white',weight='bold')
    # Identical scales make the two recovery profiles directly readable.
    bounds=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_ylim(bounds)
    fig.supxlabel('Same goal-input gate interventions; separate assays · 8 seeds · 95% paired bootstrap intervals',fontsize=8.5)
    for ext in ('png','pdf','svg'):fig.savefig(a.out/f'goal_gate_recovery.{ext}',dpi=190)
    plt.close(fig)
    (a.out/'recovery_intervals.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['assays'],indent=2))


if __name__=='__main__':main()
