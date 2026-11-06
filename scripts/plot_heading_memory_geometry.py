"""Show prespecified example memory geometry without selecting a successful seed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/ring_heading/figures"


def main():
    plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans",
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200})
    fig,axs=plt.subplots(2,3,figsize=(9,5.8),layout="constrained")
    provenance=[]
    for row,condition in enumerate(("grid","predictive")):
        path=ROOT/f"results/ring_heading/confirmation_evaluation/memory_{condition}_s2001.npz"
        data=np.load(path)
        base=data["trained_context0_base"]
        reference=base[0]
        center=reference.mean(0)
        _,singular,vt=np.linalg.svd(reference-center,full_matrices=False)
        basis=vt[:2].T
        # One fixed basis and origin across every delay, fitted only at blank2s.
        projected=(base-center)@basis
        selected=[int(np.flatnonzero(data["sample_times"]==step)[0]) for step in (0,100,600)]
        limit=float(np.max(np.abs(projected[selected])))*1.12
        fraction=float(np.sum(singular[:2]**2)/np.sum(singular**2))
        for col,(index,seconds) in enumerate(zip(selected,(2,7,32))):
            ax=axs[row,col]
            ax.scatter(projected[0,:,0],projected[0,:,1],facecolors="none",edgecolors="#d4d4d4",s=12,lw=.6,zorder=1)
            points=ax.scatter(projected[index,:,0],projected[index,:,1],c=data["phase"],
                              cmap="twilight",vmin=0,vmax=2*np.pi,s=15,zorder=2)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xticks=[],yticks=[])
            ax.set_aspect("equal")
            if row==0:ax.set_title(f"{seconds} seconds without input",fontsize=11)
            if col==0:
                label="Grid" if condition=="grid" else "PGC + grid"
                ax.set_ylabel(label+"\nPC2",fontsize=11)
            if row==1:ax.set_xlabel("PC1")
        provenance.append({"condition":condition,"seed":2001,"context":0,
                           "variance_in_displayed_plane":fraction,"source":str(path.relative_to(ROOT)),
                           "basis":"SVD of centered trained state after40blank updates; held fixed across delays"})
    bar=fig.colorbar(points,ax=axs.ravel().tolist(),fraction=.022,pad=.03,ticks=[0,np.pi,2*np.pi])
    bar.ax.set_yticklabels(["0°","180°","360°"])
    bar.set_label("Initial direction")
    fig.suptitle("Direction-memory states drift over longer blank delays",fontsize=15)
    fig.supxlabel("Example: first confirmation seed (2001) • Same projection across delays • Full-state tests reported separately",fontsize=8.5)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/("memory_geometry."+ext))
    (OUT/"memory_geometry_data.json").write_text(json.dumps({"selection":"First planned seed, first context; no outcome-based selection", "provenance":provenance},indent=2)+"\n")


if __name__=="__main__":main()
