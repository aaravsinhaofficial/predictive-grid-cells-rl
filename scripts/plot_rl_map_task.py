"""Show the actual square-arena geometry and one of its analytic hex maps."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch
from code.common import build_population


def main():
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / "models/pgc33/pgc33_lstm_s1_1000M.json").read_text())
    pop = build_population(cfg, "cpu")
    cell = int(torch.where(pop.is_pgc)[0][0])
    axis = np.linspace(-1.25, 1.25, 200, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    xy = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())))
    direction = torch.tensor([[1., 0.]]).expand(len(xy), -1)
    with torch.inference_mode():
        regular = pop.activity(xy, direction, direction*.25, shift_scale=0.)[:, cell]
        predictive = pop.activity(xy, direction, direction*.25)[:, cell]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8), layout="constrained")
    ax = axes[0]
    ax.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, fill=False, ec="#394752", lw=2))
    ax.add_patch(Circle((0, 0), .65, fill=False, ec="#aeb9bf", lw=1.5, ls="--"))
    goal = np.array([.65*np.cos(.65), .65*np.sin(.65)])
    pickup = goal-np.array([.29, 0.])
    ahead = pickup+np.array([.2, 0.])
    ax.add_patch(Circle(goal, .30, fc="#dbead2", ec="#6a8c55", alpha=.95))
    ax.scatter(*pickup, s=42, c="#3578b6", zorder=4)
    ax.scatter(*ahead, s=42, c="#e77832", zorder=4)
    ax.annotate("", xy=ahead, xytext=pickup,
                arrowprops={"arrowstyle": "->", "color": "#e77832", "lw": 2})
    ax.scatter(-.8, -.7, s=45, c="#26343e")
    ax.text(-.8, -.87, "Agent", ha="center", fontsize=11)
    ax.text(0, -.10, "Possible goals", ha="center", fontsize=10, color="#687780")
    ax.set_title("Same square arena")
    ax.set(xlim=(-1.31, 1.31), ylim=(-1.31, 1.31), aspect="equal")
    ax.axis("off")
    for ax, values, title in zip(axes[1:], [regular, predictive],
                                  ["Ordinary grid cell", "Same cell, 20 cm ahead"]):
        ax.imshow(values.numpy().reshape(xx.shape), origin="lower", extent=(-1.25,1.25,-1.25,1.25),
                  cmap="viridis", vmin=0, vmax=max(float(regular.max()), float(predictive.max())))
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.supxlabel("Goal-memory variant: goal locations lie on a circle; the agent still moves freely in 2D.",
                  fontsize=11, color="#43525d")
    out = ROOT / "mechanisms/figures"
    out.mkdir(exist_ok=True, parents=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"rl_goal_memory_task.{suffix}", dpi=180, bbox_inches="tight")
    (out / "rl_goal_memory_task.json").write_text(json.dumps({
        "source": "scripts/plot_rl_map_task.py", "population_seed": cfg["seed"],
        "cell_index": cell, "code_config": str(Path("models/pgc33/pgc33_lstm_s1_1000M.json")),
        "map_shift_m": .2, "shift_direction": [1,0],
        "geometry_is_illustration": True, "rate_maps_are_actual_analytic_population": True
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
