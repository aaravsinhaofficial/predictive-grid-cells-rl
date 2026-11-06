"""Compare the published learned ring to the unchanged decoder-free native assay."""
from pathlib import Path
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

import published_ssm_ring_benchmark as benchmark
import lstm_heading_curve_flow as previous

OUT = ROOT / "results/ring_nonlinear_design/published_benchmark_1000s"


def main():
    torch.set_num_threads(1)
    model = benchmark.Model()
    archive = np.load(OUT / "map_trajectories.npz", allow_pickle=False)
    curve = archive["loop"]
    # Runtime horizon override; the old source is neither edited nor copied.
    # Its general decoder-free geometric routines are reused verbatim.
    previous.FLOW = (1, 50, 250)
    w, bias = torch.tensor(model.recurrent), torch.tensor(model.bias)
    transition = lambda x: .8*x + .2*torch.nn.functional.softplus(x @ w.T + bias)
    protocol = {"old_source": "scripts/lstm_heading_curve_flow.py",
                "old_source_sha256": hashlib.sha256((ROOT/"scripts/lstm_heading_curve_flow.py").read_bytes()).hexdigest(),
                "new_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "horizons_seconds": [.02,1,5], "arc_samples":384,
                "norm": "all256 original coordinates", "radial_kick": ".02 loop RMS",
                "negative": "2D isotropic contraction under identical geometric assay"}
    (OUT/"old_assay_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    positive = previous.invariance_probe(None,None,curve,np.ones(256),384,transition)
    phase = np.arange(512)*2*np.pi/512
    circle = np.c_[np.cos(phase),np.sin(phase)]
    negative = previous.invariance_probe(None,None,circle,np.ones(2),384,
                                        lambda x: np.exp(-.02)*x)
    for result in (positive,negative):
        g, f = result["geometry"], result["horizons"]["250"]
        result["same_geometric_5s_criteria"] = {
            "no_large_closing_bridges":g["max_edge_over_radius"]<=.1,
            "nonlocal_segments_separated":g["nonlocal_distance_over_radius"]["p10"]>=.05,
            "fixed_curve_departure_p90_at_most_5percent":f["residual_to_fixed_curve_over_radius"]["p90"]<=.05,
            "phase_coverage_at_least80percent":f["phase_transport"]["circular_coverage"]>=.8,
            "phase_degree_one":abs(f["phase_transport"]["winding_degree"]-1)<.1,
            "phase_order_preserved":f["phase_transport"]["backward_edge_fraction"]<=.05,
            "radial_gain_at_most_half":f["radial_normal_gain"]["median"]<=.5,
            "radial_to_tangent_gain_at_most_half":f["radial_over_tangent_gain"]["median"]<=.5}
    result = {"protocol":protocol,"published_trained_ring":positive,"contracting_plane":negative}
    (OUT/"old_assay_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({name: {"checks":r["same_geometric_5s_criteria"],
                           "five_seconds":r["horizons"]["250"]}
                      for name,r in [("learned_ring",positive),("contracting_plane",negative)]}),flush=True)
    plot(curve,positive,negative)


def plot(curve,positive,negative):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((OUT/"summary.json").read_text())
    center = curve.mean(0)
    _,_,v = np.linalg.svd(curve-center,full_matrices=False)
    xy = (curve-center)@v[:2].T
    fp = np.load(benchmark.VENDOR/"Data/fpts.npy",allow_pickle=False)
    other = np.load(benchmark.VENDOR/"Data/allFpts.npy",allow_pickle=False)
    stable=(fp[1]-center)@v[:2].T
    saddle=(other[0]-center)@v[:2].T
    fig, axes = plt.subplots(1,3,figsize=(10.7,3.2),layout="constrained")
    axes[0].plot(*np.r_[xy,xy[:1]].T,color="#2563a6",lw=2.1)
    axes[0].scatter(*stable,color="#e16b35",s=38,zorder=4,label="Stable point")
    axes[0].scatter(*saddle,color="#222222",s=34,marker="s",zorder=4,label="Saddle")
    axes[0].set(aspect="equal",title="Published memory RNN",xlabel="PC 1 (display only)",ylabel="PC 2")
    axes[0].legend(frameon=False,fontsize=8,loc="upper right")
    values=summary["full_network"]["map"]["all_direction_transverse_variational_test"]
    tangent=np.array([r["tangent_gain"] for r in values])
    normal=np.array([r["worst_transverse_gain"] for r in values])
    axes[1].scatter(np.zeros(8),tangent,color="#e16b35",s=25,label="Along ring")
    axes[1].scatter(np.ones(8),normal,color="#2563a6",s=25,label="Off ring")
    axes[1].axhline(1,color=".6",lw=.8,ls="--")
    axes[1].set(xticks=[0,1],xticklabels=["Along ring","Off ring"],ylabel="Perturbation remaining after 1 s",ylim=(0,1.13),xlim=(-.5,1.5),title="Errors shrink; phase persists")
    vals=[positive["horizons"]["250"]["radial_over_tangent_gain"]["median"],
          negative["horizons"]["250"]["radial_over_tangent_gain"]["median"]]
    axes[2].bar([0,1],vals,color=["#2563a6","#999999"],width=.58)
    axes[2].text(0,.027,f"{vals[0]:.5f}",ha="center",fontsize=9)
    axes[2].set(xticks=[0,1],xticklabels=["Learned ring","Shrinking circle"],ylabel="Off-ring / along-ring retention",ylim=(0,1.13),title="Same test rejects a false ring")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"published_ring_calibration.png",dpi=200)
    fig.savefig(OUT/"published_ring_calibration.pdf")


if __name__=="__main__":
    main()
