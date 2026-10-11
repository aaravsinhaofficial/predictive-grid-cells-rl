"""Idealised grid-cell and predictive-grid-cell (PGC) population codes.

Every cell is an analytic rate map, so the population's activity at any agent
state is a lookup -- nothing is learned (the "perfect cells" design agreed with
the PI on 2026-10-09: "we imagine we have a perfect grid cell or predictive grid
cell and we have its map with respect to space ... we don't even have to
simulate it").

Grid cell
    Classic three-cosine hexagonal pattern with module spacing lambda, module
    orientation theta and a per-cell phase phi:
        s(x) = sum_{i=0..2} cos(k_i . (x - phi)),
        k_i  = 4 pi / (sqrt(3) lambda) * (cos(theta + i pi/3), sin(theta + i pi/3)),
        r(x) = ((s(x) + 1.5) / 4.5) ** sharpness            in [0, 1].
    The maxima form a hexagonal lattice with nearest-neighbour spacing lambda
    (lattice basis a1 = lambda (cos(theta+30deg), sin(theta+30deg)),
    a2 = lambda (cos(theta+90deg), sin(theta+90deg)); phases are sampled
    uniformly over that unit cell).

Predictive grid cell
    The SAME hexagonal pattern evaluated at the position displaced ahead of the
    agent along its direction of travel:
        space mode:  r(x_t + d   * u_t)      d   = preferred shift (m)
        time  mode:  r(x_t + tau * v_t)      tau = look-ahead (s)
    with u_t the unit heading-of-motion vector and v_t the allocentric
    velocity. This is the operational definition used by the predictive-grid-
    cell analysis pipeline (Ouchi & Fujisawa 2024; `space_projection='heading'`
    and `shift_mode='time'` in predictive-grid-cell-analysis/code/scores.py):
    a PGC's gridness peaks when its activity is aligned with the position
    d cm (or tau s) AHEAD on the trajectory. The RNN cohort's predictive units
    shift by a median ~16-21 cm, so the default is d = 0.20 m.

A population is fully determined by (seed, n_cells, n_modules, spacings,
pgc_frac, pgc_shift, ...), so the SAME population (same phases/orientations)
can be instantiated for several conditions, differing only in which cells are
predictive -- that is what makes the gc-vs-pgc comparison within a seed paired.
"""
import math

import numpy as np
import torch


class GridPopulation:
  """Analytic grid / predictive-grid population code."""

  def __init__(self, n_cells=512, n_modules=4, spacing_min=0.40,
               spacing_ratio=1.42, pgc_frac=0.0, pgc_shift=0.20,
               pgc_shift_mode='space', pgc_tau=0.5, pgc_shift_jitter=0.0,
               sharpness=2.0, noise=0.0, seed=0, device='cpu'):
    if pgc_shift_mode not in ('space', 'time'):
      raise ValueError(f'pgc_shift_mode must be space|time, got {pgc_shift_mode!r}')
    self.n_cells = int(n_cells)
    self.n_modules = int(n_modules)
    self.pgc_frac = float(pgc_frac)
    self.pgc_shift = float(pgc_shift)
    self.pgc_shift_mode = pgc_shift_mode
    self.pgc_tau = float(pgc_tau)
    self.pgc_shift_jitter = float(pgc_shift_jitter)
    self.sharpness = float(sharpness)
    self.noise = float(noise)
    self.seed = int(seed)
    self.device = torch.device(device)

    rng = np.random.default_rng(self.seed)
    self.spacings = spacing_min * spacing_ratio ** np.arange(self.n_modules)  # [M]
    self.module = np.arange(self.n_cells) % self.n_modules                    # [N]
    self.orientation = rng.uniform(0.0, np.pi / 3.0, self.n_modules)          # [M]
    lam = self.spacings[self.module]                                          # [N]
    th = self.orientation[self.module]                                        # [N]

    # Wave vectors k_i (i = 0, 1, 2) at 60-degree increments.
    kmag = 4.0 * np.pi / (np.sqrt(3.0) * lam)
    ang = th[:, None] + np.arange(3)[None, :] * np.pi / 3.0                  # [N,3]
    K = kmag[:, None, None] * np.stack([np.cos(ang), np.sin(ang)], -1)       # [N,3,2]

    # Phases uniform over the unit cell of the lattice of maxima.
    a1 = lam[:, None] * np.stack([np.cos(th + np.pi / 6), np.sin(th + np.pi / 6)], -1)
    a2 = lam[:, None] * np.stack([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)], -1)
    uv = rng.uniform(0.0, 1.0, (self.n_cells, 2))
    phi = uv[:, :1] * a1 + uv[:, 1:] * a2                                     # [N,2]

    # PGC assignment, stratified by module so every module keeps the same
    # predictive share (the RNN cohort's PGCs span all modules).
    is_pgc = np.zeros(self.n_cells, dtype=bool)
    k_total = int(round(self.pgc_frac * self.n_cells))
    per = [k_total // self.n_modules + (1 if m < k_total % self.n_modules else 0)
           for m in range(self.n_modules)]
    for m in range(self.n_modules):
      idx = np.where(self.module == m)[0]
      if per[m] > len(idx):
        raise ValueError('pgc_frac too large for module size')
      chosen = rng.choice(idx, size=per[m], replace=False)
      is_pgc[chosen] = True
    base = self.pgc_shift if pgc_shift_mode == 'space' else self.pgc_tau
    if self.pgc_shift_jitter > 0:
      jit = np.exp(rng.normal(0.0, self.pgc_shift_jitter, self.n_cells))
    else:
      jit = np.ones(self.n_cells)
    shifts = np.where(is_pgc, base * jit, 0.0)

    f32 = dict(dtype=torch.float32, device=self.device)
    self.K = torch.as_tensor(K, **f32)                 # [N,3,2]
    self.phi = torch.as_tensor(phi, **f32)             # [N,2]
    self.shift = torch.as_tensor(shifts, **f32)        # [N] (m or s; 0 for grid cells)
    self.is_pgc = torch.as_tensor(is_pgc, device=self.device)
    self.pgc_idx = np.where(is_pgc)[0]
    self.gc_idx = np.where(~is_pgc)[0]
    self.n_pgc = int(is_pgc.sum())
    self.has_pgc = self.n_pgc > 0
    self._np = dict(K=K, phi=phi, shifts=shifts, is_pgc=is_pgc)

  # ------------------------------------------------------------------ core
  def activity(self, pos, dir=None, vel=None, shift_scale=1.0):
    """Population activity [B, N] in [0, 1].

    pos [B,2] allocentric position (m); dir [B,2] unit heading-of-motion
    (space mode); vel [B,2] allocentric velocity (m/s, time mode).
    shift_scale multiplies every PGC's shift (0 turns PGCs into plain grid
    cells -- the `pgc_as_gc` evaluation probe).
    """
    pos = pos.to(self.device, torch.float32)
    B = pos.shape[0]
    x = pos[:, None, :].expand(B, self.n_cells, 2)
    if self.has_pgc and shift_scale != 0.0:
      src = dir if self.pgc_shift_mode == 'space' else vel
      if src is None:
        raise ValueError('PGC population needs dir (space mode) or vel (time mode)')
      src = src.to(self.device, torch.float32)
      x = x + (shift_scale * self.shift)[None, :, None] * src[:, None, :]
    proj = torch.einsum('bnd,nid->bni', x - self.phi[None], self.K)        # [B,N,3]
    s = torch.cos(proj).sum(-1)                                            # [B,N] in [-1.5, 3]
    r = ((s + 1.5) / 4.5).clamp(0.0, 1.0)
    if self.sharpness != 1.0:
      r = r.pow(self.sharpness)
    if self.noise > 0:
      r = (r + self.noise * torch.randn_like(r)).clamp(0.0, 1.0)
    return r

  def ratemaps(self, res=40, half=1.25, dir=(1.0, 0.0), vel=None, shift_scale=1.0):
    """Analytic rate maps [N, res, res] (index [i, j] = x bin i, y bin j) for a
    fixed travel direction (PGC maps are the grid pattern displaced along
    `dir`)."""
    xs = torch.linspace(-half, half, res, device=self.device)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    pos = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    d = torch.tensor(dir, dtype=torch.float32, device=self.device).expand(pos.shape[0], 2)
    v = (torch.tensor(vel, dtype=torch.float32, device=self.device).expand(pos.shape[0], 2)
         if vel is not None else d * 0.35)
    with torch.no_grad():
      r = self.activity(pos, dir=d, vel=v, shift_scale=shift_scale)
    return r.T.reshape(self.n_cells, res, res).cpu().numpy()

  # ------------------------------------------------------------- metadata
  def describe(self):
    return dict(
        n_cells=self.n_cells, n_modules=self.n_modules,
        spacings_m=[float(s) for s in self.spacings],
        orientation_deg=[float(np.degrees(o)) for o in self.orientation],
        pgc_frac=self.pgc_frac, n_pgc=self.n_pgc, n_gc=self.n_cells - self.n_pgc,
        pgc_shift_mode=self.pgc_shift_mode,
        pgc_shift_m=self.pgc_shift if self.pgc_shift_mode == 'space' else None,
        pgc_tau_s=self.pgc_tau if self.pgc_shift_mode == 'time' else None,
        pgc_shift_jitter=self.pgc_shift_jitter, sharpness=self.sharpness,
        noise=self.noise, seed=self.seed,
        pgc_idx=[int(i) for i in self.pgc_idx],
        pgc_per_module=[int(((self.module == m) & self._np['is_pgc']).sum())
                        for m in range(self.n_modules)],
    )

  def to(self, device):
    self.device = torch.device(device)
    for k in ('K', 'phi', 'shift', 'is_pgc'):
      setattr(self, k, getattr(self, k).to(self.device))
    return self
