import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popcode.codes import GridPopulation  # noqa: E402


def test_grid_periodicity_and_range():
  pop = GridPopulation(n_cells=64, n_modules=4, seed=3, device='cpu')
  pos = torch.rand(50, 2) * 2 - 1
  r = pop.activity(pos)
  assert r.shape == (50, 64)
  assert r.min() >= 0 and r.max() <= 1
  # Shift every position by one lattice vector of its cell's module -> identical.
  lam = torch.tensor(pop.spacings[pop.module], dtype=torch.float32)
  th = torch.tensor(pop.orientation[pop.module], dtype=torch.float32)
  a1 = lam[:, None] * torch.stack([torch.cos(th + np.pi / 6), torch.sin(th + np.pi / 6)], 1)
  for c in (0, 17, 63):
    r2 = pop.activity(pos + a1[c][None])
    assert torch.allclose(r[:, c], r2[:, c], atol=1e-4), c
  # Peak at the phase itself.
  assert torch.allclose(pop.activity(pop.phi)[torch.arange(64), torch.arange(64)],
                        torch.ones(64), atol=1e-4)


def test_pgc_is_displaced_grid_cell():
  gc = GridPopulation(n_cells=128, pgc_frac=0.0, seed=5, device='cpu')
  mixed = GridPopulation(n_cells=128, pgc_frac=0.33, pgc_shift=0.2, seed=5, device='cpu')
  assert mixed.n_pgc == round(0.33 * 128) == 42
  assert torch.allclose(gc.phi, mixed.phi) and torch.allclose(gc.K, mixed.K)
  pos = torch.rand(40, 2) * 2 - 1
  d = torch.randn(40, 2)
  d = d / d.norm(dim=1, keepdim=True)
  r_gc = gc.activity(pos)
  r_mix = mixed.activity(pos, dir=d)
  # grid cells unchanged, PGCs equal the grid pattern 20 cm ahead
  assert torch.allclose(r_gc[:, mixed.gc_idx], r_mix[:, mixed.gc_idx], atol=1e-5)
  r_ahead = gc.activity(pos + 0.2 * d)
  assert torch.allclose(r_ahead[:, mixed.pgc_idx], r_mix[:, mixed.pgc_idx], atol=1e-5)
  assert not torch.allclose(r_gc[:, mixed.pgc_idx], r_mix[:, mixed.pgc_idx], atol=1e-2)
  # shift_scale=0 turns PGCs back into grid cells (pgc_as_gc probe)
  assert torch.allclose(r_gc, mixed.activity(pos, dir=d, shift_scale=0.0), atol=1e-5)
  # stratified across modules
  per = mixed.describe()['pgc_per_module']
  assert sum(per) == 42 and max(per) - min(per) <= 1


def test_time_mode_uses_velocity():
  gc = GridPopulation(n_cells=32, seed=1, device='cpu')
  tm = GridPopulation(n_cells=32, pgc_frac=1.0, pgc_shift_mode='time', pgc_tau=0.5,
                      seed=1, device='cpu')
  pos = torch.rand(10, 2)
  vel = torch.randn(10, 2) * 0.3
  assert torch.allclose(tm.activity(pos, vel=vel), gc.activity(pos + 0.5 * vel), atol=1e-5)


def test_ratemaps_shape():
  pop = GridPopulation(n_cells=16, pgc_frac=0.5, seed=0, device='cpu')
  m = pop.ratemaps(res=30, half=1.25)
  assert m.shape == (16, 30, 30) and np.isfinite(m).all()
