"""Simple flow figure and explicit post-screen common-background diagnostic."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_heading_curve_flow import nearest_closed_segments
from lstm_heading_memory_probe import save,sha
from lstm_endogenous_curve_recovery import describe

OUT=ROOT/'results/lstm_heading_memory/curve_flow'


def background_diagnostic(records):
    rows=[]
    for r in records:
        folder=OUT/f'{r["condition"]}_s{r["seed"]}'
        a=np.load(folder/'context0_states.npz');times=list(a['times']);pairs={}
        for before,after in ((75,450),(450,1500)):
            z0=a['states'][times.index(before)];z1=a['states'][times.index(after)]
            metrics={}
            for name,scale in [('raw',np.ones(512)),('relative_rms',a['relative_rms_scale'])]:
                x0=z0/scale;x1=z1/scale;mean0=x0.mean(0);mean1=x1.mean(0)
                radius=r['trajectory'][str(before)][name]['radius']
                common=mean1-mean0
                energy=float(np.mean(np.sum((x1-x0)**2,axis=1)))
                shifted=x1-common
                residual,_,_,_=nearest_closed_segments(shifted,x0)
                metrics[name]=dict(common_mean_displacement_over_radius=float(np.linalg.norm(common)/radius),
                    mean_shift_fraction_matched_displacement_energy=float(np.sum(common**2)/max(energy,1e-40)),
                    residual_after_mean_only_alignment_over_radius=describe(residual/radius))
            pairs[f'{before}_to_{after}']=metrics
        cell_share={}
        for t in (75,450,1500):
            z=a['states'][times.index(t)];centered=z-z.mean(0)
            cell_share[str(t)]=float(np.sum(centered[:,256:]**2)/np.sum(centered**2))
        rows.append(dict(condition=r['condition'],seed=r['seed'],pairs=pairs,cell_share_of_raw_variance=cell_share))
    save(OUT/'mean_shift_diagnostic.json',dict(label='Post-screen diagnostic: remove only the common state translation between observed time points. No phase fitting, state rotation or amplitude rescaling. This does not alter the frozen full-state invariance test.',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/plot_lstm_heading_curve_flow.py','scripts/lstm_heading_curve_flow.py']},results=rows))
    return rows


def main():
    records=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_s*/context0.json'))]
    background_diagnostic(records)
    colors={'gc100':'#2878b8','pgc33':'#e66b2f'}
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for r in records:
        color=colors[r['condition']];label=('Grid' if r['condition']=='gc100' else 'PGC + grid') if r['seed']==1 else None
        ts=np.array([0,15,75,150,450,1500]);seconds=ts/15
        for field,style in [('hidden_radius_retention','-'),('cell_radius_retention','--')]:
            values=[r['trajectory'][str(t)]['behavioral_expression'][field] for t in ts]
            axes[0].plot(seconds,values,style,color=color,alpha=.75,label=label if style=='-' else None)
        for metric,style in [('raw','-'),('relative_rms','--')]:
            ts2=np.array([75,450,1500]);ys=[r['invariance'][str(t)][metric]['horizons']['75']['residual_to_fixed_curve_over_radius']['p90'] for t in ts2]
            axes[1].plot(ts2/15,ys,style,color=color,alpha=.75)
            ts3=np.array([450,1500]);ys3=[r['invariance'][str(t)][metric]['horizons']['75']['radial_over_tangent_gain']['median'] for t in ts3]
            axes[2].plot(ts3/15,ys3,style+'o',color=color,alpha=.75,ms=3)
    axes[0].set(xlabel='Seconds in darkness',ylabel='Remaining variation',yscale='log',title='Heading trace remains in cell state')
    axes[0].legend(frameon=False,fontsize=8,title='Solid: hidden   Dashed: cell',title_fontsize=8)
    axes[1].set(xlabel='Time of reference curve (seconds)',ylabel='5-s departure / curve radius',yscale='log',title='The full-state curve keeps moving')
    axes[1].axhline(.05,color='.6',ls=':',lw=1)
    axes[2].set(xlabel='Time of reference curve (seconds)',ylabel='Radial / tangential gain',ylim=(0,1.2),title='Radial disturbances persist')
    axes[2].axhline(.5,color='.6',ls=':',lw=1);axes[2].axhline(1,color='.8',lw=.8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'heading_curve_flow.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'heading_curve_flow.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
