"""Evaluate a released, completed goal-flash RL checkpoint without retraining."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from code.common import build_policy
from scripts.rl_goal_memory import evaluate, atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, required=True,
                    help="Use the evaluation seed recorded in the pilot protocol for reproduction")
    args = ap.parse_args()
    torch.set_num_threads(2)
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not data.get("completed"):
        raise ValueError("A completed pilot checkpoint is required")
    net = build_policy(data["config"], args.device)
    net.load_state_dict(data["policy"])
    result = evaluate(net, data["config"], device=args.device, seed=args.seed,
                      episodes=args.episodes, n_envs=args.n_envs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print({name: item["mean_score"] for name, item in result["variants"].items()})


if __name__ == "__main__":
    main()
