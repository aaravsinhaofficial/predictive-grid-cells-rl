"""Summarize every fixed visual-memory curriculum stage without selecting models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
NAMES={'gc100':'ring-20261105-vision-grid-01','pgc33':'ring-20261105-vision-pgc-01'}
VARIANTS=('blind','blind_reset','seen','seen_reset','blind_current_grid','seen_current_grid')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(condition,name,require_complete):
    folder=ROOT/'results/ring_cloud_artifacts'/name
    stages=[]
    if (folder/'baseline.json').exists():
        stages.append(dict(stage=0,learning_frames=0,result=json.loads((folder/'baseline.json').read_text())))
    for k in range(1,6):
        p=folder/f'stage{k}_evaluation.json'
        if p.exists():stages.append(dict(stage=k,learning_frames=k*5_004_800,result=json.loads(p.read_text())))
    if require_complete and len(stages)!=6:raise ValueError(f'Incomplete fixed curriculum: {condition}')
    for item in stages:
        result=item['result']
        assert result['episodes_per_variant']==64 and result['all_first_discoveries_identical']
        assert result['max_blind_gap']==60
        if condition=='gc100':
            for a,b in [('blind','blind_current_grid'),('seen','seen_current_grid')]:
                assert result['variants'][a]['rows']==result['variants'][b]['rows']
        scores={v:result['variants'][v]['mean_score'] for v in VARIANTS}
        returns={v:result['variants'][v]['mean_returns_per_episode'] for v in VARIANTS}
        item.update(scores=scores,returns=returns,contrasts=result['descriptive_contrasts'],
                    blind_seen_return_ratio=returns['blind']/max(returns['seen'],1e-12),
                    blind_seen_score_ratio=scores['blind']/max(scores['seen'],1e-12),
                    blind_exposure_fraction=result['blind_fraction_of_post_discovery_updates'],
                    state_reset_counts=result['state_reset_counts'],
                    body_heading_vs_travel_direction_rms_deg=result['body_heading_vs_travel_direction_rms_deg'])
        del item['result']
    if stages:
        baseline = stages[0]
        assert baseline['stage'] == 0
        for item in stages:
            item['blind_original_seen_return_ratio'] = item['returns']['blind']/max(baseline['returns']['seen'],1e-12)
            item['blind_original_seen_score_ratio'] = item['scores']['blind']/max(baseline['scores']['seen'],1e-12)
            item['seen_original_seen_score_ratio'] = item['scores']['seen']/max(baseline['scores']['seen'],1e-12)
            item['practical_health_reference_met'] = item['blind_seen_return_ratio'] >= .70
            item['practical_memory_reference_met'] = item['contrasts']['additional_reset_cost_in_darkness'] >= 5
    if require_complete:
        report=json.loads((folder/'summary.json').read_text())
        assert report['completed'] and report['training']['completed']
        assert report['training']['extra_frames']==25_024_000
        assert report['final_checkpoint_sha256']==sha(folder/'final.pt')
        protocol=json.loads((folder/'protocol.json').read_text())
        for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
        for name,digest in protocol['original_asset_sha256'].items():assert sha(ROOT/name)==digest,name
    return dict(condition=condition,stages=stages,completed=len(stages)==6,
        inference='One original training seed; descriptive paired episodes, no population p-values.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    out=ROOT/'results/ring_visual_memory';out.mkdir(parents=True,exist_ok=True)
    data={condition:collect(condition,name,not args.partial) for condition,name in NAMES.items()}
    (out/'curriculum_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained',sharey=True)
    styles={'blind':('#d86b2a','-','Blackouts'), 'blind_reset':('#d86b2a','--','Blackouts + reset'),
            'seen':('#347bb5','-','Vision'), 'seen_reset':('#347bb5','--','Vision + reset')}
    for ax,(condition,item) in zip(axes,data.items()):
        stages=item['stages'];x=[r['learning_frames']/1e6 for r in stages]
        for variant,(color,line,label) in styles.items():
            ax.plot(x,[r['scores'][variant] for r in stages],marker='o',color=color,ls=line,label=label)
        if stages:
            ax.axhline(stages[0]['scores']['seen'], color='#777777', ls=':', lw=1.2,
                       label='Original vision score')
        ax.set(title='Grid' if condition=='gc100' else 'PGC + grid',xlabel='Additional RL frames (millions)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Navigation score');axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Visual-memory curriculum (in progress)' if args.partial else 'Learned navigation during visual blackouts')
    fig.supxlabel('One original seed pair · same final blackout test at every stage · all checkpoints retained',fontsize=9)
    fig.savefig(out/'visual_memory_learning.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for condition, color in [('gc100','#347bb5'),('pgc33','#d86b2a')]:
        stages=data[condition]['stages'];x=[r['learning_frames']/1e6 for r in stages]
        label='Grid' if condition=='gc100' else 'PGC + grid'
        axes[0].plot(x,[r['contrasts']['additional_reset_cost_in_darkness'] for r in stages],
                     marker='o',color=color,label=label)
        for variant,ls,branch in [('blind','-','Native'),('blind_reset','--','Reset'),
                                  ('blind_current_grid',':','Ordinary current code')]:
            axes[2].plot(x,[100*r['blind_exposure_fraction'][variant] for r in stages],
                         marker='o',color=color,ls=ls,label=f'{label}: {branch}')
    stages=data['pgc33']['stages'];x=[r['learning_frames']/1e6 for r in stages]
    for key,color,label in [('blind_current_prediction_benefit','#d86b2a','Blackouts'),
                             ('seen_current_prediction_benefit','#347bb5','Vision')]:
        axes[1].plot(x,[r['contrasts'][key] for r in stages],marker='o',color=color,label=label)
    axes[0].set(title='Additional memory benefit in darkness',ylabel='Score difference')
    axes[1].set(title='Benefit of current predictive input',ylabel='Score difference')
    axes[2].set(title='Actual blackout exposure',ylabel='Post-discovery updates (%)')
    for ax in axes:
        ax.set_xlabel('Additional RL frames (millions)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
        ax.legend(frameon=False,fontsize=8)
    for ax in axes[:2]:ax.axhline(0,color='#777777',lw=1)
    if args.partial:fig.suptitle('Visual-memory checks (in progress)')
    fig.supxlabel('One original seed pair · same final blackout test at every stage · descriptive episode comparisons',fontsize=9)
    fig.savefig(out/'visual_memory_checks.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({c:{'completed':d['completed'],'stages':len(d['stages'])} for c,d in data.items()}))


if __name__=='__main__':main()
