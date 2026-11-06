"""Seed-level descriptive summary of the frozen held-out travel-memory cohort."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lstm_heading_memory_probe import sha,save

OUT=ROOT/'results/lstm_travel_memory/heldout_original'


def values(record):
    branch=record['ordinary_release'];at4=branch['trajectory']['60'];primary=branch['invariance']['30']
    result=dict(fixed_readout_error_deg=at4['fixed_travel_phase']['heldout_mae_deg'],
                refitted_readout_error_deg=at4['time_refit_information_only']['heldout_mae_deg'])
    for metric in ('raw','relative_rms'):
        row=primary[metric]['horizons']['30']
        result[f'{metric}_radial_over_tangent']=row['radial_over_tangent_gain']['median']
        result[f'{metric}_fixed_curve_departure']=row['residual_to_fixed_curve_over_radius']['p90']
    return result


def main():
    files=sorted(OUT.glob('*_s*/context[01].json'));assert len(files)==24
    records=[json.loads(path.read_text()) for path in files]
    seeds=[]
    for condition in ('gc100','pgc33'):
        for seed in range(3,9):
            contexts=[r for r in records if r['condition']==condition and r['seed']==seed]
            assert len(contexts)==2
            vv=[values(r) for r in contexts]
            aggregate={key:dict(mean=float(np.mean([v[key] for v in vv])),
                                minimum=min(v[key] for v in vv),maximum=max(v[key] for v in vv)) for key in vv[0]}
            seeds.append(dict(condition=condition,seed=seed,metrics=aggregate,
                              candidate_contexts=sum(r['ordinary_release']['passes_both_metric_candidate_screen'] for r in contexts)))
    save(OUT/'seed_summary.json',dict(replication_unit='trained model seed; two nested native contexts',
        methods_fixed_on_seeds=[1,2],heldout_seeds=list(range(3,9)),models=12,contexts=24,
        source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),
        results=seeds,context_source_sha256={str(p.relative_to(OUT)):sha(p) for p in files}))
    colors={'gc100':'#2878b8','pgc33':'#e66b2f'}
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.1),layout='constrained')
    for j,condition in enumerate(('gc100','pgc33')):
        rows=[r for r in seeds if r['condition']==condition]
        for i,row in enumerate(rows):
            x=j+np.linspace(-.13,.13,6)[i];color=colors[condition]
            for key,face in [('fixed_readout_error_deg',color),('refitted_readout_error_deg','white')]:
                v=row['metrics'][key]
                axes[0].plot([x,x],[v['minimum'],v['maximum']],color=color,lw=.8,alpha=.5)
                axes[0].scatter(x,v['mean'],s=24,edgecolor=color,facecolor=face,zorder=3)
            for ax,key in [(axes[1],'raw_radial_over_tangent'),(axes[2],'raw_fixed_curve_departure')]:
                v=row['metrics'][key]
                ax.plot([x,x],[v['minimum'],v['maximum']],color=color,lw=.8,alpha=.5)
                ax.scatter(x,v['mean'],s=24,color=color,zorder=3)
    axes[0].set(ylabel='Angle error (degrees)',title='Direction information survives',ylim=(-3,100))
    axes[0].legend(handles=[Line2D([0],[0],marker='o',ls='',color='.3',label='Fixed readout'),
                            Line2D([0],[0],marker='o',ls='',markerfacecolor='white',color='.3',label='Refitted readout')],frameon=False,fontsize=8,loc='center')
    axes[1].set(ylabel='Radial / tangential gain',title='Radial disturbances persist',ylim=(0,1.3))
    axes[1].axhline(.5,color='.6',ls=':',lw=1);axes[1].axhline(1,color='.8',lw=.8)
    axes[2].set(ylabel='Departure / curve radius',title='The full-state curve moves',ylim=(0,9))
    axes[2].axhline(.05,color='.6',ls=':',lw=1)
    for ax in axes:
        ax.set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],xlim=(-.4,1.4))
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'heldout_travel_memory.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'heldout_travel_memory.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
