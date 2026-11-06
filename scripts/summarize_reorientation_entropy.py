"""Prespecified paired summaries of all four matched entropy continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from summarize_blind_reorientation import summarize, episode_values, METRICS

OUT=ROOT/'results/rl_blind_reorientation_entropy'
LAUNCH=ROOT/'docs/research/rl_blind_reorientation_entropy_launch.json'
TRAINING=ROOT/'docs/research/rl_blind_reorientation_entropy_protocol.json'
CONDITIONS=('gc100','pgc33');ARMS=('original','explore')
BOOTSTRAPS=10_000;SEED=198207;BLOCK_FRAMES=10_014_720;PARENT_FRAMES=40_058_880


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return dict(version='matched_reorientation_entropy_behavior_v1',
        training_protocol_sha256=sha(TRAINING),
        sources={name:sha(ROOT/name) for name in (
            'scripts/summarize_reorientation_entropy.py','scripts/summarize_blind_reorientation.py')},
        cohort='Both populations, original seed1, original versus explore entropy. Exact40M parent baseline and every five fixed extension endpoints. Final90,132,480total extra frames is primary; no stage selection.',
        primary='Free-navigation goal hits and progress per original episode, including no-event/no-discovery zeros. Original frozen episode_values and summarize reused unchanged.',
        contrasts='Within each population, explore minus original for blind-native outcomes; difference of their native-minus-memory-reset costs. Same fixed episode/slot keys; pairedslot resampling retains both episodes and all contrast components.',
        uncertainty=dict(bootstrap_resamples=BOOTSTRAPS,seed=SEED,clusters='32slots,2episodes each',
            interpretation='Descriptive95% evaluation uncertainty conditional on these two fixed policies. No trained-seed significance or multiple-comparison claim.'),
        caveats='Different policies can change event timing and free exposure. Per-original-episode outcomes remain primary; rates perfreeupdate are descriptive. Stages and windows are not trainedseed replications.',
        routing='Read workers[{condition,arm,name}] from explicit launch manifest. Preserve capacity failures separately; no ambiguous symlink routing.')


def paired_components(components,metric):
    """A signed contrast; each component is (coefficient,result,variant)."""
    keys=None;difference=None
    for coefficient,result,variant in components:
        current,values=episode_values(result,variant)
        if keys is None:
            keys=current;difference=np.zeros(len(keys))
        assert current==keys,'Unmatched episode/slot keys'
        difference+=coefficient*values[:,METRICS.index(metric)]
    assert keys
    slots=sorted({key[0] for key in keys})
    groups=[difference[[key[0]==slot for key in keys]] for slot in slots]
    assert len({len(g) for g in groups})==1
    means=np.array([g.mean() for g in groups]);rng=np.random.default_rng(SEED)
    estimates=means[rng.integers(len(slots),size=(BOOTSTRAPS,len(slots)))].mean(1)
    return dict(mean=float(difference.mean()),ci95=np.quantile(estimates,[.025,.975]).tolist(),
        episodes=len(keys),paired_slots=len(slots),metric=metric,units='per original episode')


def collect(require_complete=False):
    launch=json.loads(LAUNCH.read_text());routes={}
    for worker in launch['workers']:
        key=(worker['condition'],worker['arm']);assert key not in routes
        assert key[0] in CONDITIONS and key[1] in ARMS
        routes[key]=worker['name']
    assert len(routes)==4
    cases={};raw={}
    for condition in CONDITIONS:
        cases[condition]={}
        for arm in ARMS:
            worker=routes[condition,arm];folder=ROOT/'results/ring_cloud_artifacts'/worker
            endpoints=[]
            for block in range(6):
                path=folder/('baseline.json' if block==0 else f'block{block}_evaluation.json')
                if not path.exists():continue
                result=json.loads(path.read_text())
                assert result['episodes_per_variant']==64
                raw[condition,arm,block]=result
                endpoints.append(dict(block=block,extension_frames=block*BLOCK_FRAMES,
                    total_extra_frames=PARENT_FRAMES+block*BLOCK_FRAMES,
                    source=str(path.relative_to(ROOT)),source_sha256=sha(path),**summarize(result)))
            complete=False
            if (folder/'summary.json').exists():
                metadata=json.loads((folder/'summary.json').read_text())
                complete=bool(metadata['completed']) and len(endpoints)==6
                if complete:
                    assert metadata['condition']==condition and metadata['arm']==arm
                    assert metadata['training']['extra_frames']==90_132_480
                    assert metadata['training']['extension_frames']==50_073_600
            if require_complete:assert complete,(condition,arm)
            cases[condition][arm]=dict(worker=worker,completed=complete,endpoints=endpoints)
    comparisons=[]
    for condition in CONDITIONS:
        for block in range(6):
            if not all((condition,arm,block) in raw for arm in ARMS):continue
            original=raw[condition,'original',block];explore=raw[condition,'explore',block]
            effects={}
            for metric in ('free_hits','progress_m'):
                effects['explore_minus_original_'+metric]=paired_components(
                    [(1,explore,'blind_native'),(-1,original,'blind_native')],metric)
                for reset in ('reset_after_glimpse','reset_every_blind'):
                    effects['memory_cost_difference_'+reset+'_'+metric]=paired_components(
                        [(1,explore,'blind_native'),(-1,explore,'blind_'+reset),
                         (-1,original,'blind_native'),(1,original,'blind_'+reset)],metric)
            comparisons.append(dict(condition=condition,block=block,effects=effects))
    return dict(conditions=cases,comparisons=comparisons,launch_manifest_sha256=sha(LAUNCH),
                analysis_protocol_sha256=sha(OUT/'analysis_protocol.json'))


def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.6),sharey=True,layout='constrained')
    for ax,condition,title in zip(axes,CONDITIONS,('Grid','PGC + grid')):
        for arm,color,label in [('original','#2878a4','Original entropy weight'),
                                ('explore','#d27927','Higher entropy weight')]:
            rows=data['conditions'][condition][arm]['endpoints']
            ax.plot([r['extension_frames']/1e6 for r in rows],
                    [r['variants']['blind_native']['free_hits'] for r in rows],
                    marker='o',color=color,label=label)
        ax.set(title=title,xlabel='Further training frames (millions)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Blind-navigation goals per episode');axes[1].legend(frameon=False,fontsize=8)
    complete=all(v['completed'] for p in data['conditions'].values() for v in p.values())
    fig.suptitle('Does a higher entropy weight help blind navigation?'+('' if complete else ' — running'))
    fig.savefig(OUT/'exploration_learning.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'analysis_protocol.json';design=protocol()
    if path.exists():assert json.loads(path.read_text())==design
    else:
        assert args.freeze_only,'Freeze before reading new results';path.write_text(json.dumps(design,indent=2)+'\n')
    if args.freeze_only:return
    data=collect(args.require_complete);(OUT/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    figure(data)
    print(json.dumps({c:{a:len(v['endpoints']) for a,v in p.items()} for c,p in data['conditions'].items()}))


if __name__=='__main__':main()
