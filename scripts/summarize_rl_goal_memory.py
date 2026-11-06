"""Validate and summarize the single-pair original-arena RL memory pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("gc100", "pgc33")
VARIANTS = ("flash", "flash_reset", "continuous", "continuous_reset", "no_goal")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(source):
    master = read(source / "summary.json")
    if master["completed"] is not True or set(master["conditions"]) != set(CONDITIONS):
        raise ValueError("Both frozen pilot conditions must be complete")
    design = master["protocol"]
    frozen_path = ROOT / "docs/research/rl_goal_memory_pilot_protocol.json"
    frozen = read(frozen_path)
    if design["frozen_protocol_sha256"] != sha(frozen_path):
        raise ValueError("Frozen protocol hash mismatch")
    for key in ("conditions", "circle_radius_m", "flash_policy_updates", "extra_frames_per_condition",
                "warmup_updates", "training_seed", "evaluation_seed", "evaluation_episodes_per_variant",
                "evaluation_envs_per_variant", "variants", "source_sha256", "original_asset_sha256"):
        if design[key] != frozen[key]:
            raise ValueError(f"Protocol mismatch: {key}")
    for field in ("source_sha256", "original_asset_sha256"):
        for path, expected in design[field].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Original source/asset changed: {path}")
    result = {"scope": "Descriptive single preselected seed pair; episodes are not trained-network replicates",
              "source": str(source.relative_to(ROOT)), "frozen_protocol_sha256": sha(frozen_path),
              "evaluation_episodes_per_branch": design["evaluation_episodes_per_variant"],
              "conditions": {}, "hypothesis_tests": None}
    for condition in CONDITIONS:
        folder = source / condition
        baseline, training, final = (read(folder / name) for name in
                                     ("baseline.json", "training.json", "final_evaluation.json"))
        if baseline != master["conditions"][condition]["baseline"] or final != master["conditions"][condition]["final"]:
            raise ValueError("Per-condition evaluation differs from final combined artifact")
        if training != master["conditions"][condition]["training"]:
            raise ValueError("Training record differs from final combined artifact")
        checkpoint_path = folder / "final.pt"
        if sha(checkpoint_path) != master["conditions"][condition]["final_checkpoint_sha256"]:
            raise ValueError("Final checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]
        increment = cfg["n_envs"] * cfg["rollout"] * cfg["action_repeat"]
        expected_frames = math.ceil(design["extra_frames_per_condition"] / increment) * increment
        if not checkpoint["completed"] or checkpoint["extra_frames"] != expected_frames:
            raise ValueError("Training was partial or exceeded the fixed update budget")
        if training["extra_frames"] != expected_frames or training["warmup_updates"] != 200:
            raise ValueError("Incorrect learned-frame/warmup accounting")
        if training["warmup_exposure_frames"] != 200 * increment or training["warmup_policy_hash_unchanged"] is not True:
            raise ValueError("Warmup did not preserve the policy and exposure accounting")
        original = torch.load(ROOT / f"models/{condition}/{condition}_lstm_s1_1000M.pt",
                              map_location="cpu", weights_only=True)["policy"]
        warmup = torch.load(folder / "after_warmup.pt", map_location="cpu", weights_only=True)
        if warmup["extra_frames"] != 0 or not all(torch.equal(original[k], warmup["policy"][k]) for k in original):
            raise ValueError("Warmup weights differ from the immutable original checkpoint")
        if all(torch.equal(original[k], checkpoint["policy"][k]) for k in original):
            raise ValueError("Fine-tuning did not update any policy parameter")
        stages = {}
        for stage, evaluation in (("before", baseline), ("after", final)):
            if set(evaluation["variants"]) != set(VARIANTS) or not evaluation["all_first_discoveries_identical"]:
                raise ValueError("Evaluation branches are incomplete or not coupled through discovery")
            reference_discoveries = None
            branches = {}
            for variant in VARIANTS:
                values = evaluation["variants"][variant]
                rows = values["rows"]
                if len(rows) != design["evaluation_episodes_per_variant"] or values["episodes"] != len(rows):
                    raise ValueError("Missing evaluation episodes")
                if len({(r["slot"], r["episode"]) for r in rows}) != len(rows):
                    raise ValueError("Duplicate episode rows")
                discoveries = [r["first_goal_step"] for r in rows]
                if reference_discoveries is None:
                    reference_discoveries = discoveries
                elif discoveries != reference_discoveries:
                    raise ValueError("An intervention changed pre-intervention discovery")
                scores = np.array([r["score"] for r in rows])
                returns = np.array([max(0, r["n_goals"] - 1) for r in rows])
                if not np.allclose(scores, [10 * r["n_goals"] for r in rows], rtol=0, atol=0):
                    raise ValueError("Raw score no longer equals original +10 pickup rewards")
                if not np.isclose(scores.mean(), values["mean_score"]) or not np.isclose(returns.mean(), values["mean_returns_per_episode"]):
                    raise ValueError("Summary does not reproduce all episode rows")
                branches[variant] = {"mean_score": float(scores.mean()),
                                     "mean_returns_per_episode": float(returns.mean()),
                                     "discovered_episodes": sum(x >= 0 for x in discoveries),
                                     "at_least_two_goals_episodes": sum(r["n_goals"] >= 2 for r in rows),
                                     "post_discovery_returns_per_1000_steps": values["post_discovery_returns_per_1000_steps"]}
            differences = {}
            for metric in ("mean_score", "mean_returns_per_episode"):
                value = lambda name: branches[name][metric]
                flash_reset = value("flash") - value("flash_reset")
                continuous_reset = value("continuous") - value("continuous_reset")
                differences[metric] = {"flash_minus_no_goal": value("flash") - value("no_goal"),
                                       "flash_reset_cost": flash_reset,
                                       "continuous_reset_cost": continuous_reset,
                                       "additional_reset_cost_without_continuous_cue": flash_reset - continuous_reset}
            stages[stage] = {"branches": branches, "differences": differences,
                             "state_reset_counts": evaluation["state_reset_counts"],
                             "first_hit_geometry_records": len(evaluation["first_hit_geometry"])}
        result["conditions"][condition] = {"stages": stages,
                "learned_frames": training["extra_frames"], "warmup_exposure_frames": training["warmup_exposure_frames"],
                "warmup_updates": training["warmup_updates"], "learning_updates": training["learning_updates"],
                "training_seconds": training["seconds"], "final_checkpoint_sha256": sha(checkpoint_path),
                "warmup_exactly_preserved_original_weights": True}
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("Brief goal\ncue", "Brief cue\n+ reset", "Continuous\ngoal cue", "Continuous\n+ reset", "No goal\ncue")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = .34
    max_score = max(v["mean_score"] for c in result["conditions"].values()
                    for s in c["stages"].values() for v in s["branches"].values())
    for ax, stage, title in zip(axes, ("before", "after"), ("Before fine-tuning", "After 5M reward-only frames")):
        x = np.arange(len(VARIANTS))
        for j, (condition, color, name) in enumerate((("gc100", "#2878B5", "Grid"), ("pgc33", "#E66A32", "PGC + grid"))):
            values = [result["conditions"][condition]["stages"][stage]["branches"][v]["mean_score"] for v in VARIANTS]
            bars = ax.bar(x + (j - .5) * width, values, width, color=color, label=name)
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, max_score * 1.19)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#dddddd", alpha=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean navigation score", fontsize=11)
    axes[1].legend(frameon=False, loc="upper right", fontsize=10)
    fig.suptitle("Testing goal memory in the original square room", fontsize=16, y=1.01)
    fig.text(.5, -.015, "One original seed pair · 64 matched episodes per branch · +10 per pickup · Reset clears both LSTM states",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(out / "goal_memory_before_after.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def report(result, out):
    grid = result["conditions"]["gc100"]["stages"]
    pgc = result["conditions"]["pgc33"]["stages"]
    g_reset = grid["after"]["differences"]["mean_score"]["flash_reset_cost"]
    p_reset = pgc["after"]["differences"]["mean_score"]["flash_reset_cost"]
    p_before = pgc["before"]["differences"]["mean_score"]["flash_minus_no_goal"]
    p_after = pgc["after"]["differences"]["mean_score"]["flash_minus_no_goal"]
    lines = ["# Original-agent goal-memory pilot", "",
             "**Fine-tuning mainly improved searching.** Navigation improved substantially even when no saved goal message was supplied. After training, clearing both recurrent states as the brief cue disappeared cost only "
             f"**{g_reset:.2f} score points for Grid and {p_reset:.2f} for PGC + grid**. The results favor more effective search over newly learned, useful retention of the brief goal message.", "",
             f"The PGC flash advantage over no goal message was {p_before:.2f} points before training and {p_after:.2f} afterward. Most of that small advantage survives the state reset. A brief cue can change the first few actions and the resulting pose/momentum, which persist after the hidden-state reset; this is a plausible explanation for a small cue benefit without sustained recurrent goal storage.", "",
             "This is one original Grid/PGC seed pair, before and after reward-only fine-tuning. The two task changes are circular goal centers within the same square room and a four-update goal-code flash. Episodes are descriptive trials, not independent trained-network replicates.", "",
             "![Before and after fine-tuning](goal_memory_before_after.png)", "",
             "Each cell below is **mean raw score / mean subsequent returns per episode**. Failed discoveries and failed returns remain in these means. Score is +10 per pickup; subsequent returns exclude the first discovery.", "",
             "| Network | Stage | Brief cue | Brief cue + reset | Continuous cue | Continuous + reset | No goal cue |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for stage in ("before", "after"):
            branches = result["conditions"][condition]["stages"][stage]["branches"]
            cells = [f"{branches[v]['mean_score']:.2f} / {branches[v]['mean_returns_per_episode']:.2f}" for v in VARIANTS]
            lines.append("| " + " | ".join(["Grid" if condition == "gc100" else "PGC + grid", stage] + cells) + " |")
    lines += ["", "Reset occurs immediately before the first update without the brief cue. It clears h and c while preserving the environment, previous action, and previous reward. The continuous+reset branch measures the corresponding disruption when the goal message is still externally available.", "",
              "| Network | Stage | Brief cue − no goal | Brief-cue reset cost | Continuous-cue reset cost | Additional reset cost without continuous cue |",
              "|---|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for stage in ("before", "after"):
            d = result["conditions"][condition]["stages"][stage]["differences"]["mean_score"]
            vals = [d[k] for k in ("flash_minus_no_goal", "flash_reset_cost", "continuous_reset_cost", "additional_reset_cost_without_continuous_cue")]
            lines.append("| " + " | ".join(["Grid" if condition == "gc100" else "PGC + grid", stage] + [f"{x:+.2f}" for x in vals]) + " |")
    lines += ["", "Positive reset costs mean performance fell after erasing the recurrent state. A larger cost after the cue disappears is compatible with useful retained goal information, but the reset is not selective for a proven goal subspace. No ring architecture or auxiliary target was imposed, and successful goal memory alone would not demonstrate a neural ring attractor.", "",
              "Each model received **5,004,800 learned frames** (391 complete original A2C updates), plus **2,560,000 zero-learning-rate exposure frames** (200 warmup updates). Warmup weights were independently checked tensor by tensor against the immutable original checkpoint. All five branches reproduce identical first-discovery times within each evaluation.", "",
              "The source, original asset, protocol, and final checkpoint hashes were validated before producing this report. No p-values or population confidence intervals are computed for a single seed pair.", "",
              "[Numerical summary](pilot_summary.json) · [Frozen protocol](../../docs/research/rl_goal_memory_pilot_protocol.json) · [Methods and interpretation](../../docs/research/rl_goal_memory_pilot.md)."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/ring_cloud_artifacts/ring-20261105-rl-goal-memory-t4-01")
    parser.add_argument("--out", type=Path, default=ROOT / "results/ring_rl_goal_memory")
    args = parser.parse_args()
    result = analyze(args.source.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.out)
    report(result, args.out)
    print(json.dumps({"report": str(args.out / "README.md"), "conditions": list(result["conditions"])}))
