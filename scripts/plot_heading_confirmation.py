"""Plot complete seed-level confirmation and full-state memory evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/ring_heading/figures"
BLUE,ORANGE,GREEN,GRAY="#2878B5","#E66A32","#338569","#858585"


def mean_ci(x):
    x=np.asarray(x,dtype=float)
    mean=x.mean(0)
    half=stats.t.ppf(.975,len(x)-1)*stats.sem(x,axis=0)
    return mean,half


def paired_points(ax,groups,labels,colors):
    groups=np.asarray(groups)
    for row in groups.T:
        ax.plot(range(len(groups)),row,color="#b9c0c5",alpha=.35,lw=.8,zorder=1)
    for i,(values,color) in enumerate(zip(groups,colors)):
        mean,half=mean_ci(values)
        ax.scatter(np.full(len(values),i),values,s=18,color=color,alpha=.65,zorder=2)
        ax.errorbar(i+.10,mean,yerr=half,fmt="_",color=color,capsize=4,ms=12,lw=2,zorder=3)
    ax.set_xticks(range(len(groups)),labels)


def main():
    summary=json.loads((ROOT/"results/ring_heading/confirmation_summary.json").read_text())
    cue=json.loads((ROOT/"results/ring_heading/cue_spectra_confirmation/summary.json").read_text())
    if summary["seeds"]!=list(range(2001,2017)) or cue["n"]!=16:
        raise ValueError("Complete locked cohort required")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "font.family":"DejaVu Sans","savefig.dpi":200})
    fig,axs=plt.subplots(2,2,figsize=(10,8),layout="constrained")
    rows=summary["per_seed"]
    ax=axs[0,0]
    paired_points(ax,[[r[c]["same_region_native_mae_deg"] for r in rows] for c in ("grid","predictive","grid_heading")],
                  ["Grid","PGC + grid","Explicit direction"],[BLUE,ORANGE,GREEN])
    ax.set(ylabel="Heading error (degrees)",ylim=(0,None),title="A  Direction estimates improve")
    ax=axs[0,1]
    memory_scalars=[]
    for condition,color in (("grid",BLUE),("predictive",ORANGE)):
        normal,tangent=[],[]
        for seed in summary["seeds"]:
            path=ROOT/f"results/ring_heading/confirmation_evaluation/memory_{condition}_s{seed}.json"
            report=json.loads(path.read_text())
            context_rows=[context["perturbations"]["0.1"] for context in report["models"]["trained"][:2]]
            time=np.array([r["seconds_after_settling"] for r in context_rows[0]])
            normal.append(np.mean([[r["normal_distance_radius"]/context[0]["normal_distance_radius"] for r in context] for context in context_rows],axis=0))
            tangent.append(np.mean([[r["decoded_tangent_retention"]["median"] for r in context] for context in context_rows],axis=0))
            one=next(i for i,t in enumerate(time) if t==1)
            memory_scalars.append({"seed":seed,"condition":condition,"normal_retained_1s":float(normal[-1][one]),
                                   "phase_retained_1s":float(tangent[-1][one])})
        keep=time<=5
        label="Grid" if condition=="grid" else "PGC + grid"
        for values,style,suffix in ((normal,"-","across"),(tangent,"--","around")):
            mean,half=mean_ci(values)
            ax.plot(time[keep],mean[keep],style,color=color,lw=2,label=label+": "+suffix)
            ax.fill_between(time[keep],(mean-half)[keep],(mean+half)[keep],color=color,alpha=.10,lw=0)
    ax.set(xlabel="Seconds after disturbance (input off)",ylabel="Disturbance remaining",ylim=(-.03,1.3),
           title="B  Fast local recovery, slower phase change")
    ax.set_yticks([0,.5,1],["0%","50%","100%"])
    ax.legend(frameon=False,fontsize=8)
    ax=axs[1,0]
    paired_points(ax,[[r["cue_free_"+v]["phase_gain"] for r in cue["per_seed"]] for v in ("blank","ordinary","native")],
                  ["Input off","Ordinary code","Predictive code"],[BLUE,GRAY,ORANGE])
    ax.axhline(1,color="#cccccc",lw=1,ls=":")
    ax.set(ylabel="One-step phase gain",title="C  Predictive input reduces phase gain")
    ax=axs[1,1]
    arrays=np.load(ROOT/"results/ring_heading/cue_spectra_confirmation/predictive_s2001.npz")
    circle=np.linspace(0,2*np.pi,300)
    ax.plot(np.cos(circle),np.sin(circle),color="#cccccc",lw=1)
    for variant,color,label in (("blank",BLUE,"Input off"),("ordinary",GRAY,"Ordinary"),("native",ORANGE,"Predictive")):
        eigensets=arrays[f"context4_cue_free_{variant}_eigenvalues"][::12]
        eigen=eigensets.ravel()
        ax.scatter(eigen.real,eigen.imag,s=9,alpha=.25,color=color,label=label,rasterized=True)
        slow=eigensets[np.arange(len(eigensets)),np.argmin(np.abs(eigensets-1),axis=1)]
        ax.scatter(slow.real,slow.imag,s=27,facecolors="none",edgecolors=color,lw=1.2)
    ax.set(xlabel="Eigenvalue: real part",ylabel="Imaginary part",title="D  Input changes the Jacobian spectrum",
           xlim=(.50,1.04),ylim=(-.20,.20))
    ax.set_aspect("equal",adjustable="box")
    ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax.text(.98,.03,"Example: seed 2001",transform=ax.transAxes,ha="right",fontsize=8,color="#666666")
    for ax in axs.flat:
        ax.grid(axis="y",color="#ededed",lw=.7);ax.set_axisbelow(True)
    fig.suptitle("Predictive input anchors learned direction memory",fontsize=15)
    fig.supxlabel("16 independent training seeds • Mean ± 95% interval across seeds • Fixed 6,000-update checkpoints",fontsize=9)
    OUT.mkdir(parents=True,exist_ok=True)
    for extension in ("png","pdf","svg"):
        fig.savefig(OUT/("heading_confirmation."+extension))
    (OUT/"heading_confirmation_data.json").write_text(json.dumps({"seeds":summary["seeds"],"memory_scalars":memory_scalars,
        "spectral_example_selection":"First confirmation seed2001, center position, every12th of72initialphases; fixed without selecting an effect.",
        "phase_gain_note":"Coordinate derivative, not eigenvalue. Full-state finite recovery and coverage are assessed separately."},indent=2)+"\n")


if __name__=="__main__":main()
