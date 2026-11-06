"""Simple, explicitly pilot-labelled views of learned memory and code rescue."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ring_attractor/pilot'
BLUE,ORANGE='#3479B6','#D87532'


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    # Fit a display PCA within the one model; all diagnostics use 64-D state.
    with np.load(OUT/'predictive_s1001_3000_dynamics.npz') as a:
        states=a['base'];times=a['times'];phase=a['initial_phase']
    state=states[np.flatnonzero(times==100)[0]]
    center=state.mean(0)
    _,s,v=np.linalg.svd(state-center,full_matrices=False)
    projection=(state-center)@v[:2].T
    ax=axs[0,0]
    ax.plot(*np.r_[projection,projection[:1]].T,color=ORANGE,lw=1.5)
    ax.scatter(*projection.T,c=phase,cmap='twilight',s=13,zorder=3)
    ax.set(title='Memory after cues disappear',xlabel='PC 1',ylabel='PC 2',aspect='equal')
    ax.set_xticks([]);ax.set_yticks([])
    ax.text(.02,.02,f'2 PCs: {100*(s[:2]**2).sum()/(s*s).sum():.0f}% of variance',
            transform=ax.transAxes,fontsize=9)
    ax=axs[0,1]
    for cond,color,label in [('grid',BLUE,'Grid'),('predictive',ORANGE,'PGC + grid')]:
        j=json.loads((OUT/f'{cond}_s1001_3000_dynamics.json').read_text())
        rows=j['memory']
        ax.plot([r['seconds'] for r in rows],[r['angular_mae_deg'] for r in rows],color=color,label=label)
    ax.set(title='Memory drift without cues',xlabel='Time without cues (s)',ylabel='Location error (degrees)')
    ax.legend(frameon=False)
    ax=axs[1,0]
    j=json.loads((OUT/'predictive_s1001_3000_dynamics.json').read_text())
    rows=j['perturbation']['0.1']
    t=[r['seconds'] for r in rows]
    ax.plot(t,[r['normal_distance_radius']/rows[0]['normal_distance_radius'] for r in rows],
            color=ORANGE,label='Distance away from ring')
    initial=rows[0]['tangent_phase_displacement']
    ax.plot(t,[r['tangent_phase_displacement']/initial for r in rows],color='#616161',label='Shift along ring')
    ax.set(title='Different responses to disturbance',xlabel='Time after disturbance (s)',ylabel='Fraction of initial change')
    ax.legend(frameon=False,fontsize=8)
    ax=axs[1,1]
    j=json.loads((OUT/'predictive_s1001_3000_motion.json').read_text())
    variants=['native','ordinary','reverse','restore_at_15']
    means=[np.mean([r['baseline_mae_deg'] for r in j['rows'] if r['variant']==v and r['step']==60]) for v in variants]
    ax.bar(range(4),means,color=[ORANGE,'#89949E','#B8BEC5',ORANGE],width=.65)
    ax.set_xticks(range(4),['Intact','Ordinary','Reversed','Restored'],rotation=15)
    ax.set(title='Predictive input aligns moving memory',ylabel='Location error (degrees)')
    fig.suptitle('A learned ring-like memory: first pilot',fontsize=16)
    fig.supxlabel('One paired training seed · Full-state diagnostics · Exploratory results',fontsize=9)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'ring_pilot.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
