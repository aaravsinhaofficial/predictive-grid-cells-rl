"""Code variations of code/codes.py: per-cell look-ahead distributions, sideways
look-aheads, within-module spread, lattice deformation, field jitter and
learnable look-ahead vectors."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.codes import GridPopulation  # noqa: E402
from code.common import CodeSource, build_population  # noqa: E402

MEC = dict(pgc_frac=0.33, pgc_shift_dist='tnorm', pgc_shift_dist_args=[0.27, 0.11, 0.02, 0.5])


def states(B=64, seed=0):
  g = torch.Generator().manual_seed(seed)
  pos = (torch.rand(B, 2, generator=g) * 2 - 1) * 1.17
  ang = torch.rand(B, generator=g) * 2 * math.pi
  d = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
  return pos, d


def test_defaults_are_the_original_population():
  # Fingerprints recorded from the population before the variations existed
  # (activity compared bit for bit at the time).
  assert GridPopulation(seed=1, pgc_frac=0.33).lattice_hash() == 'f76ba20987b9c5f2'
  p = GridPopulation(seed=3, pgc_frac=0.33)
  assert not p.has_lat and not p.has_warp and not p.learned_shift
  assert np.allclose(p.deform, np.eye(2))


def test_shift_distributions_keep_the_cells_and_change_only_the_shifts():
  base = GridPopulation(seed=4, pgc_frac=0.33)
  for kw in (MEC,
             dict(pgc_frac=0.33, pgc_shift_dist='unif', pgc_shift_dist_args=[0.0, 0.4]),
             dict(pgc_frac=0.33, pgc_shift_dist='set', pgc_shift_dist_args=[0.1, 0.3]),
             dict(pgc_frac=0.33, pgc_shift_dist='emp', pgc_shift_dist_args=[0.05, 0.1, 0.2])):
    p = GridPopulation(seed=4, **kw)
    assert np.array_equal(p.pgc_idx, base.pgc_idx)
    assert torch.equal(p.K, base.K) and torch.equal(p.phi, base.phi)
    s = p.shift.numpy()
    assert np.all(s[p.gc_idx] == 0) and np.all(s[p.pgc_idx] > 0)
  lam = base.spacings[base.module]
  p = GridPopulation(seed=4, **MEC)
  f = p.shift.numpy()[p.pgc_idx] / lam[p.pgc_idx]
  assert f.min() >= 0.02 - 1e-6 and f.max() <= 0.5 + 1e-6
  assert abs(f.mean() - 0.27) < 0.03 and abs(f.std() - 0.11) < 0.03
  p = GridPopulation(seed=4, pgc_frac=0.33, pgc_shift_dist='unif', pgc_shift_dist_args=[0.0, 0.4])
  s = p.shift.numpy()[p.pgc_idx]
  assert s.max() <= 0.4 and abs(s.mean() - 0.2) < 0.03
  p = GridPopulation(seed=4, pgc_frac=0.33, pgc_shift_dist='set', pgc_shift_dist_args=[0.1, 0.3])
  for m in range(4):
    s = p.shift.numpy()[(p.module == m) & p._np['is_pgc']]
    assert abs(int(np.isclose(s, 0.1).sum()) - int(np.isclose(s, 0.3).sum())) <= 1
  p = GridPopulation(seed=4, pgc_frac=0.33, pgc_shift_dist='emp', pgc_shift_dist_args=[0.05, 0.1, 0.2])
  f = p.shift.numpy()[p.pgc_idx] / lam[p.pgc_idx]
  assert set(np.round(f, 4)) <= {0.05, 0.1, 0.2}


def test_ten_percent_rule_gives_the_four_distances():
  p = GridPopulation(seed=1, pgc_frac=0.33, pgc_shift_rule='spacing', pgc_shift_frac=0.10)
  d = p.describe()['pgc_shift_m_per_module']
  assert np.allclose(np.array(d) * 100, [4.0, 5.68, 8.07, 11.45], atol=0.01)


def test_sideways_and_backward_look_aheads():
  pos, d = states()
  grid = GridPopulation(seed=2, pgc_frac=0.0)
  left = torch.stack([-d[:, 1], d[:, 0]], 1)
  side = GridPopulation(seed=2, pgc_frac=0.33, pgc_shift_angle=90.0)
  a = side.activity(pos, dir=d)
  ref = grid.activity(pos + 0.20 * left)
  assert torch.allclose(a[:, side.pgc_idx], ref[:, side.pgc_idx], atol=1e-5)
  assert torch.allclose(a[:, side.gc_idx], grid.activity(pos)[:, side.gc_idx], atol=1e-6)
  back = GridPopulation(seed=2, pgc_frac=0.33, pgc_shift_angle=180.0)
  retro = GridPopulation(seed=2, pgc_frac=0.33, pgc_shift=-0.20)
  assert torch.allclose(back.activity(pos, dir=d), retro.activity(pos, dir=d), atol=1e-5)
  # de-shifting probe removes the sideways shift too
  assert torch.allclose(side.activity(pos, dir=d, shift_scale=0.0), grid.activity(pos), atol=1e-6)


def test_within_module_spread():
  base = GridPopulation(seed=5, **MEC)
  p = GridPopulation(seed=5, spacing_sd=0.05, orient_sd_deg=2.0, **MEC)
  assert torch.equal(p.shift, base.shift)                   # look-aheads are untouched
  lam_m = base.spacings[base.module]
  rel = p.cell_spacing / lam_m - 1.0
  assert abs(rel.std() - 0.05) < 0.006 and abs(rel.mean()) < 0.006
  dth = np.degrees(p.cell_orientation - base.orientation[base.module])
  assert abs(dth.std() - 2.0) < 0.25
  # every cell is still periodic over its own lattice
  pos, d = states(32)
  th, lam = p.cell_orientation, p.cell_spacing
  a1 = torch.tensor(lam[:, None] * np.stack([np.cos(th + np.pi / 6), np.sin(th + np.pi / 6)], -1),
                    dtype=torch.float32)
  r0 = p.activity(pos, dir=d, shift_scale=0.0)
  for n in (0, 77, 300, 511):
    r1 = p.activity(pos + a1[n][None], dir=d, shift_scale=0.0)
    assert torch.allclose(r0[:, n], r1[:, n], atol=2e-4)
  # the same spread is drawn whatever the PGC arrangement
  g = GridPopulation(seed=5, pgc_frac=0.0, spacing_sd=0.05, orient_sd_deg=2.0)
  assert torch.equal(g.K, p.K) and torch.equal(g.phi, p.phi)


def test_ellipticity_and_shear():
  base = GridPopulation(seed=6, pgc_frac=0.0)
  p = GridPopulation(seed=6, pgc_frac=0.0, lattice_ellip=1.3)
  for m in range(4):
    sv = np.linalg.svd(p.deform[m], compute_uv=False)
    assert abs(sv[0] / sv[1] - 1.3) < 1e-9 and abs(sv[0] * sv[1] - 1.0) < 1e-9
  # periodic over the deformed lattice vectors, not over the original ones
  pos, d = states(48)
  n = 9
  m = base.module[n]
  lam, th = base.spacings[m], base.orientation[m]
  a1 = lam * np.array([np.cos(th + np.pi / 6), np.sin(th + np.pi / 6)])
  a1d = torch.tensor(p.deform[m] @ a1, dtype=torch.float32)
  r0 = p.activity(pos)[:, n]
  assert torch.allclose(r0, p.activity(pos + a1d[None])[:, n], atol=2e-4)
  assert not torch.allclose(r0, p.activity(pos + torch.tensor(a1, dtype=torch.float32)[None])[:, n], atol=1e-2)
  # the deformed map is the original map read at D^-1 x
  Dinv = torch.tensor(np.linalg.inv(p.deform[m]), dtype=torch.float32)
  assert torch.allclose(r0, base.activity(pos @ Dinv.T)[:, n], atol=2e-4)
  s = GridPopulation(seed=6, pgc_frac=0.0, lattice_shear=0.1)
  for m in range(4):
    D = s.deform[m]
    assert abs(abs(D[0, 1]) + abs(D[1, 0]) - 0.1) < 1e-9 and np.allclose(np.diag(D), 1.0)


def test_field_jitter_moves_fields_by_the_stated_amount():
  base = GridPopulation(seed=7, pgc_frac=0.0)
  p = GridPopulation(seed=7, pgc_frac=0.0, field_jitter=0.06)
  g = torch.Generator().manual_seed(1)
  pos = (torch.rand(4000, 2, generator=g) * 2 - 1) * 1.25
  x = pos[:, None, :].expand(-1, p.n_cells, -1)
  ph = torch.einsum('bnd,nkd->bnk', x, p.warp_Q) + p.warp_psi[None]
  W = torch.einsum('bnk,nkd->bnd', torch.sin(ph), p.warp_A)
  rms = W.pow(2).sum(-1).mean(0).sqrt().numpy()             # per cell
  lam = base.spacings[base.module]
  # per cell the RMS inside the room scatters (a coarse cell's warp has about one
  # wavelength in the room); over cells it averages to the stated value
  assert np.all(np.abs(rms / lam - 0.06) < 0.035)
  assert abs(float(np.mean(rms / lam)) - 0.06) < 0.004
  fine = base.module == 0
  assert np.all(np.abs(rms[fine] / lam[fine] - 0.06) < 0.02)
  a, b = p.activity(pos[:200]), base.activity(pos[:200])
  assert a.min() >= 0 and a.max() <= 1 and not torch.allclose(a, b, atol=1e-2)
  # jitter composes with look-aheads: the warped map is read at the look-ahead point
  q = GridPopulation(seed=7, pgc_frac=0.33, field_jitter=0.06)
  pos2, d2 = states(40)
  assert torch.allclose(q.activity(pos2, dir=d2)[:, q.pgc_idx],
                        p.activity(pos2 + 0.2 * d2)[:, q.pgc_idx], atol=1e-5)


def test_variations_draw_from_independent_streams():
  a = GridPopulation(seed=8, **MEC)
  b = GridPopulation(seed=8, field_jitter=0.05, lattice_ellip=1.2, spacing_sd=0.03, orient_sd_deg=1.0, **MEC)
  assert torch.equal(a.shift, b.shift) and np.array_equal(a.pgc_idx, b.pgc_idx)
  c = GridPopulation(seed=8, pgc_frac=0.0, field_jitter=0.05)
  assert torch.equal(c.warp_A, b.warp_A) is False        # amplitude scales with the cell's own spacing
  d = GridPopulation(seed=8, pgc_frac=0.33, field_jitter=0.05)
  assert torch.equal(c.warp_A, d.warp_A) and torch.equal(c.warp_Q, d.warp_Q)
  assert a.lattice_hash() != b.lattice_hash() != c.lattice_hash()


def test_learnable_shift_matches_stored_shift_and_has_correct_gradient():
  pos, d = states(16)
  p = GridPopulation(seed=9, pgc_frac=0.33, pgc_shift_angle=30.0)
  sh = torch.stack([p.shift, p.shift_lat], 1).clone().double().requires_grad_(True)
  assert torch.allclose(p.activity(pos, dir=d), p.activity(pos, dir=d, shift=sh.float()), atol=1e-6)
  g = GridPopulation(seed=9, pgc_frac=0.0)
  s0 = torch.zeros(g.n_cells, 2, requires_grad=True)
  out = g.activity(pos, dir=d, shift=s0)
  assert torch.allclose(out, g.activity(pos), atol=1e-6)
  w = torch.randn(out.shape, generator=torch.Generator().manual_seed(0))
  (out * w).sum().backward()
  eps, n = 1e-3, 123
  for j in (0, 1):
    sp, sm = torch.zeros(g.n_cells, 2), torch.zeros(g.n_cells, 2)
    sp[n, j], sm[n, j] = eps, -eps
    fd = ((g.activity(pos, dir=d, shift=sp) * w).sum() - (g.activity(pos, dir=d, shift=sm) * w).sum()) / (2 * eps)
    assert abs(float(fd) - float(s0.grad[n, j])) < 2e-2 * max(1.0, abs(float(fd)))
  g.set_shift(torch.full((g.n_cells,), 0.1))
  assert torch.allclose(g.activity(pos, dir=d), GridPopulation(seed=9, pgc_frac=1.0, pgc_shift=0.1).activity(pos, dir=d),
                        atol=1e-5)


def test_code_source_passes_learnable_shift_and_cfg_builds_variants():
  cfg = dict(n_cells=512, n_modules=4, spacing_min=0.4, spacing_ratio=1.42, pgc_frac=0.33, pgc_shift=0.2,
             pgc_shift_mode='space', pgc_tau=0.5, pgc_shift_jitter=0.0, sharpness=2.0, code_seed=3,
             code_mode='full', arena_half=1.25, v_max=0.35, pgc_shift_dist='tnorm',
             pgc_shift_dist_args='0.27,0.11,0.02,0.5', spacing_sd=0.03, orient_sd_deg=1.0,
             lattice_ellip=1.15, field_jitter=0.04, pgc_shift_angle=0.0)
  pop = build_population(cfg, 'cpu')
  assert pop.describe()['pgc_shift_dist'] == 'tnorm' and pop.has_warp
  src = CodeSource(cfg, pop, 'cpu')
  pos, d = states(8)
  sh = torch.stack([pop.shift, pop.shift_lat], 1).clone().requires_grad_(True)
  a = src(pos, d, d * 0.3, shift=sh)
  assert torch.allclose(a, src(pos, d, d * 0.3), atol=1e-6)
  g = src.goal(pos, d, d * 0.3, shift=sh)
  (a.sum() + g.sum()).backward()
  assert sh.grad.abs().sum() > 0
  old = {k: v for k, v in cfg.items() if not k.startswith(('pgc_shift_dist', 'spacing_sd', 'orient', 'lattice', 'field'))}
  assert build_population(old, 'cpu').lattice_hash() == GridPopulation(seed=3, pgc_frac=0.33).lattice_hash()


def test_alternating_left_right_look_ahead_and_recorded_fractions():
  pos, d = states()
  p = GridPopulation(seed=2, pgc_frac=0.33, pgc_shift_angle=30.0, pgc_shift_angle_alt=True)
  lat = p.shift_lat.numpy()[p.pgc_idx]
  assert abs(int((lat > 0).sum()) - int((lat < 0).sum())) <= 4
  assert np.allclose(np.abs(lat), 0.2 * math.sin(math.radians(30)), atol=1e-6)
  assert np.allclose(p.shift.numpy()[p.pgc_idx], 0.2 * math.cos(math.radians(30)), atol=1e-6)
  from code.codes_data import GARDNER_SHIFT_FRAC
  assert len(GARDNER_SHIFT_FRAC) == 108 and abs(np.mean(GARDNER_SHIFT_FRAC) - 0.0975) < 1e-3
  cfg = dict(n_cells=512, n_modules=4, spacing_min=0.4, spacing_ratio=1.42, pgc_frac=0.33, pgc_shift=0.2,
             pgc_shift_mode='space', pgc_tau=0.5, pgc_shift_jitter=0.0, sharpness=2.0, code_seed=3,
             pgc_shift_dist='gardner_frac')
  g = build_population(cfg, 'cpu')
  f = g.shift.numpy()[g.pgc_idx] / g.spacings[g.module][g.pgc_idx]
  assert set(np.round(f, 4)) <= set(np.round(GARDNER_SHIFT_FRAC, 4)) and abs(f.mean() - 0.0975) < 0.02
  # 'gardner': the recorded shifts in centimetres as they are, the same in every module
  from code.codes_data import GARDNER_SHIFT_M
  g = build_population(dict(cfg, pgc_shift_dist='gardner'), 'cpu')
  d = g.shift.numpy()[g.pgc_idx]
  cm = lambda v: set(np.round(np.asarray(v, dtype=np.float64) * 100).astype(int))
  assert cm(d) <= cm(GARDNER_SHIFT_M) and abs(d.mean() - 0.083) < 0.015
  assert abs(np.mean(GARDNER_SHIFT_M) - 0.0833) < 1e-3 and np.median(GARDNER_SHIFT_M) == 0.06


def test_old_run_cannot_be_resumed_with_a_variation_switched_on(tmp_path):
  import json
  from code.train import check_resume_config
  cfg = dict(cond='pgc33', pgc_frac=0.33, code_mode='full', code_seed=1, n_cells=512, n_modules=4,
             spacing_min=0.4, spacing_ratio=1.42, pgc_shift=0.2, pgc_shift_mode='space', pgc_tau=0.5,
             pgc_shift_jitter=0.0, sharpness=2.0, seed=1, out=str(tmp_path))
  stored = {k: v for k, v in cfg.items() if k != 'out'}      # an old config: no variation keys
  json.dump(stored, open(tmp_path / 'config.json', 'w'))
  check_resume_config(dict(cfg), str(tmp_path / 'config.json'))            # unchanged: accepted
  for k, v in (('field_jitter', 0.06), ('spacing_sd', 0.06), ('lattice_shear', 0.13), ('learn_shift', 1),
               ('pgc_shift_angle', 90.0)):
    with pytest.raises(SystemExit):
      check_resume_config(dict(cfg, **{k: v}), str(tmp_path / 'config.json'))
