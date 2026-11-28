"""Properties of the codes themselves, without any agent (a check that a score difference is not a trivial
side effect of how a code is built).

For every open-room code and seed, the population is rebuilt from the config.json of its training run, so the
cells are exactly the trained ones. 4,500 random places and directions of travel give:

  mean, sd        activity of a cell (every code should be equally active)
  norm            length of the population vector
  cos_10cm        similarity of the codes of two places 10 cm apart
  cos_far_p999    among pairs of places more than 40 cm apart: the similarity of the most alike 0.1 %
                  (how easily two distant places are mixed up)
  place_r2        how well a linear readout of the code predicts 81 Gaussian place fields (held-out places)
  xy_rmse_cm      error of a linear readout of x and y

  python -m scripts.codes.code_stats        -> results/code_variations/data/code_stats.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.common import build_population  # noqa: E402
from scripts.codes.matrix import OPEN_CORE, OPEN_SCREEN  # noqa: E402

torch.set_num_threads(2)
H = 1.25


def find(code, seed):
  for tag in ('os', 'oc'):
    p = os.path.join(ROOT, 'runs_codes', tag, f'{code}_open_ffb_s{seed}', 'config.json')
    if os.path.exists(p):
      return json.load(open(p))
  return None


def stats(code, seed, n=4500):
  cfg = find(code, seed)
  if cfg is None:
    return None
  pop = build_population(cfg, 'cpu')
  rng = np.random.default_rng(1000 + seed)
  pos = torch.tensor(rng.uniform(-H, H, (n, 2)), dtype=torch.float32)
  ang = rng.uniform(0, 2 * np.pi, n)
  d = torch.tensor(np.stack([np.cos(ang), np.sin(ang)], 1), dtype=torch.float32)
  with torch.no_grad():
    A = pop.activity(pos, dir=d).double().numpy()                     # [n, 512]
  nrm = np.linalg.norm(A, axis=1)
  out = dict(mean=A.mean(), sd=A.std(), norm=nrm.mean())
  P = pos.numpy()
  i, j = rng.integers(0, n, 60000), rng.integers(0, n, 60000)
  dist = np.linalg.norm(P[i] - P[j], axis=1)
  An = A - A.mean(0)
  An = An / np.linalg.norm(An, axis=1, keepdims=True)
  cs = (An[i] * An[j]).sum(1)
  out['cos_far_p999'] = np.quantile(cs[dist > 0.40], 0.999)
  out['cos_10cm'] = cs[(dist > 0.08) & (dist < 0.12)].mean()
  c = np.linspace(-H + 0.15, H - 0.15, 9)
  C = np.stack(np.meshgrid(c, c), -1).reshape(-1, 2)
  T = np.exp(-((P[:, None, :] - C[None]) ** 2).sum(-1) / (2 * 0.15 ** 2))
  X = np.concatenate([A, np.ones((n, 1))], 1)
  tr, te = slice(0, 3000), slice(3000, n)
  G = X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1])
  pred = X[te] @ np.linalg.solve(G, X[tr].T @ T[tr])
  out['place_r2'] = 1 - ((pred - T[te]) ** 2).sum() / ((T[te] - T[tr].mean(0)) ** 2).sum()
  out['xy_rmse_cm'] = 100 * np.sqrt(((X[te] @ np.linalg.solve(G, X[tr].T @ P[tr]) - P[te]) ** 2).sum(1).mean())
  return {k: float(v) for k, v in out.items()}


def main():
  res = {}
  for code in list(OPEN_CORE) + list(OPEN_SCREEN):
    rows = [r for r in (stats(code, s) for s in range(1, 9)) if r]
    if not rows:
      continue
    res[code] = dict(n=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
    m = res[code]
    print(f'{code:9s} n={m["n"]} mean {m["mean"]:.3f} sd {m["sd"]:.3f} norm {m["norm"]:.2f} cos10cm {m["cos_10cm"]:.3f} '
          f'cosFar {m["cos_far_p999"]:.3f} placeR2 {m["place_r2"]:.3f} xyRMSE {m["xy_rmse_cm"]:.1f} cm', flush=True)
  out = os.path.join(ROOT, 'results', 'code_variations', 'data', 'code_stats.json')
  json.dump(res, open(out, 'w'), indent=1)
  print('wrote', out)


if __name__ == '__main__':
  main()
