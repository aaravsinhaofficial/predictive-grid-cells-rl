"""Idealised grid-cell and predictive-grid-cell (PGC) population codes.

Every cell is an analytic rate map, so the population's activity at any agent
state is a lookup -- nothing is learned. The cells are "perfect": each has a
fixed map with respect to space (or, for a predictive cell, with respect to
the position ahead of the agent) and is read off that map every step rather
than simulated.

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

Code variations (all off by default; with the defaults the population is
bit-identical to the original one). Each variation draws from its own random
stream keyed by (seed, stream id), so switching one on never changes the
numbers another one draws, nor the original phases / orientations / PGC
indices:
  shift distributions   pgc_shift_dist: every PGC gets its own look-ahead
                        ('tnorm': a fraction of its module spacing from a
                        truncated normal, e.g. the recorded MEC distribution;
                        'emp': a fraction drawn from a list of recorded values;
                        'empm': metres drawn from a list of recorded shifts;
                        'unif': metres, uniform in a range; 'set': a few fixed
                        distances shared out evenly within each module)
  shift direction       pgc_shift_angle: the look-ahead points this many
                        degrees to the left of the direction of travel
                        (0 ahead, 90 sideways, 180 behind)
  within-module spread  spacing_sd / orient_sd_deg: every cell's spacing and
                        orientation are drawn from a Gaussian around its
                        module's values
  lattice deformation   lattice_ellip / lattice_shear: each module's lattice
                        is stretched along a random axis (ellipticity = long /
                        short axis of the six inner peaks) or sheared along a
                        wall axis
  field jitter          field_jitter: every cell's map is warped by a smooth
                        random displacement field (RMS = field_jitter x
                        spacing, wavelength 1.5-2.5 spacings), which moves
                        each firing field a little in its own direction
  learnable shifts      set_shift(): per-cell (forward, sideways) look-ahead
                        vectors supplied from outside (code/train.py
                        --learn_shift trains them with the policy)
"""
import hashlib
import math

import numpy as np
import torch


class GridPopulation:
  """Analytic grid / predictive-grid population code."""

  def __init__(self, n_cells=512, n_modules=4, spacing_min=0.40,
               spacing_ratio=1.42, pgc_frac=0.0, pgc_shift=0.20,
               pgc_shift_mode='space', pgc_tau=0.5, pgc_shift_jitter=0.0,
               sharpness=2.0, noise=0.0, seed=0, device='cpu',
               pgc_layout='uniform', pgc_module_weights=None,
               pgc_shift_rule='const', pgc_shift_frac=0.5, cell_type='grid',
               pgc_shift_dist='rule', pgc_shift_dist_args=None, pgc_shift_angle=0.0,
               pgc_shift_angle_alt=False,
               spacing_sd=0.0, orient_sd_deg=0.0, lattice_ellip=1.0,
               lattice_shear=0.0, field_jitter=0.0):
    """pgc_layout: how the PGCs are distributed over modules --
         'uniform' (equal share per module, stratified random), 'fine' (fill
         the smallest-spacing modules first), 'coarse' (largest first),
         'middle' (modules 1 and 2 first), or 'weights' (counts proportional
         to pgc_module_weights, a list with one weight per module).
       pgc_shift_rule: 'const' (every PGC shifted by pgc_shift metres) or
         'spacing' (shift = pgc_shift_frac * module spacing: a constant PHASE
         shift across scales).
       cell_type: 'grid' (three-cosine hexagonal cells) or 'band' (one cosine
         along one of the module's three lattice directions, i.e. stripe /
         band cells; predictive band cells are shifted the same way).
       pgc_shift_dist: 'rule' (the const / spacing rule above), or a per-cell
         distribution with parameters pgc_shift_dist_args:
           'tnorm' [mu, sd, lo, hi]  shift = f x module spacing, f from a normal
                                     (mu, sd) truncated to [lo, hi]
           'emp'   [f1, f2, ...]     shift = f x module spacing, f drawn with
                                     replacement from the list
           'empm'  [d1, d2, ...]     shift in metres, drawn with replacement
                                     from the list (recorded shifts as they are)
           'unif'  [lo, hi]          shift in metres, uniform
           'set'   [d1, d2, ...]     metres; the PGCs of each module take the
                                     values in turn (equal shares)
       pgc_shift_angle: degrees to the left of the direction of travel.
       pgc_shift_angle_alt: the PGCs of each module take +angle and -angle in
         turn (half look ahead-left, half ahead-right).
       spacing_sd: SD of the per-cell spacing, as a fraction of the module
         spacing; orient_sd_deg: SD of the per-cell orientation (degrees).
       lattice_ellip: ellipticity of every module's lattice (1 = hexagonal);
         lattice_shear: shear of every module's lattice along a wall axis.
       field_jitter: RMS displacement of the firing fields, as a fraction of
         the cell's spacing."""
    if pgc_shift_mode not in ('space', 'time'):
      raise ValueError(f'pgc_shift_mode must be space|time, got {pgc_shift_mode!r}')
    if pgc_layout not in ('uniform', 'fine', 'coarse', 'middle', 'weights'):
      raise ValueError(f'unknown pgc_layout {pgc_layout!r}')
    if pgc_shift_rule not in ('const', 'spacing'):
      raise ValueError(f'unknown pgc_shift_rule {pgc_shift_rule!r}')
    if cell_type not in ('grid', 'band'):
      raise ValueError(f'unknown cell_type {cell_type!r}')
    if pgc_shift_dist not in ('rule', 'tnorm', 'emp', 'empm', 'unif', 'set'):
      raise ValueError(f'unknown pgc_shift_dist {pgc_shift_dist!r}')
    if pgc_shift_dist != 'rule' and pgc_shift_mode != 'space':
      raise ValueError('pgc_shift_dist needs pgc_shift_mode space')
    self.pgc_shift_dist = pgc_shift_dist
    self.pgc_shift_dist_args = ([float(v) for v in pgc_shift_dist_args]
                                if pgc_shift_dist_args is not None else None)
    self.pgc_shift_angle = float(pgc_shift_angle)
    self.pgc_shift_angle_alt = bool(pgc_shift_angle_alt)
    self.spacing_sd, self.orient_sd_deg = float(spacing_sd), float(orient_sd_deg)
    self.lattice_ellip, self.lattice_shear = float(lattice_ellip), float(lattice_shear)
    self.field_jitter = float(field_jitter)
    if self.lattice_ellip < 1.0:
      raise ValueError('lattice_ellip is long / short axis and must be >= 1')
    self.pgc_layout, self.pgc_shift_rule = pgc_layout, pgc_shift_rule
    self.pgc_shift_frac, self.cell_type = float(pgc_shift_frac), cell_type
    self.pgc_module_weights = (list(map(float, pgc_module_weights))
                               if pgc_module_weights is not None else None)
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
    lam_module = lam                 # look-aheads are fractions of the MODULE spacing
    if self.spacing_sd > 0 or self.orient_sd_deg > 0:
      # Within-module spread: its own stream, one normal draw per cell and property.
      rv = np.random.default_rng([self.seed, 202])
      z = rv.normal(0.0, 1.0, (self.n_cells, 2))
      lam = lam * np.clip(1.0 + self.spacing_sd * z[:, 0], 0.5, 1.5)
      th = th + np.radians(self.orient_sd_deg) * z[:, 1]
    self.cell_spacing, self.cell_orientation = lam, th

    # Wave vectors k_i (i = 0, 1, 2) at 60-degree increments.
    kmag = 4.0 * np.pi / (np.sqrt(3.0) * lam)
    ang = th[:, None] + np.arange(3)[None, :] * np.pi / 3.0                  # [N,3]
    K = kmag[:, None, None] * np.stack([np.cos(ang), np.sin(ang)], -1)       # [N,3,2]

    # Phases uniform over the unit cell of the lattice of maxima.
    a1 = lam[:, None] * np.stack([np.cos(th + np.pi / 6), np.sin(th + np.pi / 6)], -1)
    a2 = lam[:, None] * np.stack([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)], -1)
    uv = rng.uniform(0.0, 1.0, (self.n_cells, 2))
    phi = uv[:, :1] * a1 + uv[:, 1:] * a2                                     # [N,2]

    # Lattice deformation, one matrix per module: x_deformed = D x, so the
    # wave vectors become k D^-1 and the phases D phi.
    self.deform = np.tile(np.eye(2), (self.n_modules, 1, 1))                  # [M,2,2]
    if self.lattice_ellip != 1.0 or self.lattice_shear != 0.0:
      rd = np.random.default_rng([self.seed, 303])
      axis = rd.uniform(0.0, np.pi, self.n_modules)        # long axis of the ellipse
      wall = rd.integers(0, 2, self.n_modules)             # shear along x (0) or y (1)
      sign = rd.choice([-1.0, 1.0], self.n_modules)
      e = np.sqrt(self.lattice_ellip)
      for m in range(self.n_modules):
        c, s_ = np.cos(axis[m]), np.sin(axis[m])
        R = np.array([[c, -s_], [s_, c]])
        D = R @ np.diag([e, 1.0 / e]) @ R.T
        S = np.eye(2)
        S[wall[m], 1 - wall[m]] = sign[m] * self.lattice_shear
        self.deform[m] = S @ D
      Dc = self.deform[self.module]                                           # [N,2,2]
      K = np.einsum('nid,nde->nie', K, np.linalg.inv(Dc))
      phi = np.einsum('nde,ne->nd', Dc, phi)

    # Field jitter: a smooth random displacement field per cell,
    # W(x) = sum_k A_k sin(q_k . x + psi_k), added to the position before the
    # lattice is read. RMS |W| = field_jitter x spacing.
    self.n_warp = 4
    warp_A = np.zeros((self.n_cells, self.n_warp, 2))
    warp_Q = np.zeros((self.n_cells, self.n_warp, 2))
    warp_psi = np.zeros((self.n_cells, self.n_warp))
    if self.field_jitter > 0:
      rj = np.random.default_rng([self.seed, 404])
      wl = rj.uniform(1.5, 2.5, (self.n_cells, self.n_warp)) * lam[:, None]   # wavelengths
      qa = rj.uniform(0.0, 2 * np.pi, (self.n_cells, self.n_warp))
      aa = rj.uniform(0.0, 2 * np.pi, (self.n_cells, self.n_warp))
      warp_psi = rj.uniform(0.0, 2 * np.pi, (self.n_cells, self.n_warp))
      warp_Q = (2 * np.pi / wl)[..., None] * np.stack([np.cos(qa), np.sin(qa)], -1)
      amp = self.field_jitter * lam * np.sqrt(2.0 / self.n_warp)
      warp_A = amp[:, None, None] * np.stack([np.cos(aa), np.sin(aa)], -1)

    # PGC assignment over modules according to pgc_layout.
    is_pgc = np.zeros(self.n_cells, dtype=bool)
    k_total = int(round(self.pgc_frac * self.n_cells))
    sizes = [int((self.module == m).sum()) for m in range(self.n_modules)]
    per = self._module_counts(k_total, sizes)
    self.pgc_per_module_target = per
    for m in range(self.n_modules):
      idx = np.where(self.module == m)[0]
      if per[m] > len(idx):
        raise ValueError('pgc_frac too large for module size')
      chosen = rng.choice(idx, size=per[m], replace=False)
      is_pgc[chosen] = True
    if pgc_shift_mode == 'space':
      if self.pgc_shift_rule == 'spacing':
        base = self.pgc_shift_frac * lam_module     # per cell: fraction of its module spacing
      else:
        base = np.full(self.n_cells, self.pgc_shift)
    else:
      base = np.full(self.n_cells, self.pgc_tau)
    if self.pgc_shift_jitter > 0:
      jit = np.exp(rng.normal(0.0, self.pgc_shift_jitter, self.n_cells))
    else:
      jit = np.ones(self.n_cells)
    if self.pgc_shift_dist != 'rule':
      base, jit = self._draw_shifts(is_pgc, lam_module), np.ones(self.n_cells)
    shifts = np.where(is_pgc, base * jit, 0.0)
    # Direction of the look-ahead relative to the direction of travel.
    ang = np.radians(self.pgc_shift_angle)
    shifts_lat = np.zeros(self.n_cells)
    if self.pgc_shift_angle != 0.0:
      shifts, shifts_lat = shifts * np.cos(ang), shifts * np.sin(ang)
      if self.pgc_shift_angle_alt:
        for m in range(self.n_modules):
          idx = np.where((self.module == m) & is_pgc)[0]
          shifts_lat[idx[1::2]] *= -1.0
    # Band cells: one lattice direction per cell (3 orientations per module).
    self.band_dir = np.arange(self.n_cells) % 3

    f32 = dict(dtype=torch.float32, device=self.device)
    self.K = torch.as_tensor(K, **f32)                 # [N,3,2]
    self.phi = torch.as_tensor(phi, **f32)             # [N,2]
    self.shift = torch.as_tensor(shifts, **f32)        # [N] (m or s; 0 for grid cells)
    self.shift_lat = torch.as_tensor(shifts_lat, **f32)  # [N] to the left of travel (m)
    self.has_lat = bool(np.any(shifts_lat != 0.0))
    self.has_warp = self.field_jitter > 0
    self.warp_A = torch.as_tensor(warp_A, **f32)       # [N,k,2]
    self.warp_Q = torch.as_tensor(warp_Q, **f32)       # [N,k,2]
    self.warp_psi = torch.as_tensor(warp_psi, **f32)   # [N,k]
    self.learned_shift = False
    self.is_pgc = torch.as_tensor(is_pgc, device=self.device)
    self._band_dir_t = torch.as_tensor(self.band_dir, dtype=torch.long, device=self.device)
    self.pgc_idx = np.where(is_pgc)[0]
    self.gc_idx = np.where(~is_pgc)[0]
    self.n_pgc = int(is_pgc.sum())
    self.has_pgc = self.n_pgc > 0
    self._np = dict(K=K, phi=phi, shifts=shifts, is_pgc=is_pgc, shifts_lat=shifts_lat,
                    warp_A=warp_A, warp_Q=warp_Q, warp_psi=warp_psi)

  def _draw_shifts(self, is_pgc, lam_module):
    """Per-cell look-ahead (m) under pgc_shift_dist; its own random stream,
    one draw per cell so the values do not depend on which cells are PGCs."""
    a, N = self.pgc_shift_dist_args, self.n_cells
    rs = np.random.default_rng([self.seed, 101])
    if self.pgc_shift_dist == 'tnorm':
      mu, sd, lo, hi = a
      f = rs.normal(mu, sd, N)
      for _ in range(1000):                       # redraw the values outside [lo, hi]
        bad = (f < lo) | (f > hi)
        if not bad.any():
          break
        f[bad] = rs.normal(mu, sd, int(bad.sum()))
      return f * lam_module
    if self.pgc_shift_dist == 'emp':
      return rs.choice(np.asarray(a, float), N, replace=True) * lam_module
    if self.pgc_shift_dist == 'empm':
      return rs.choice(np.asarray(a, float), N, replace=True)
    if self.pgc_shift_dist == 'unif':
      return rs.uniform(a[0], a[1], N)
    # 'set': the PGCs of each module take the listed distances in turn
    out = np.zeros(N)
    for m in range(self.n_modules):
      idx = np.where((self.module == m) & is_pgc)[0]
      out[idx] = np.asarray(a, float)[np.arange(len(idx)) % len(a)]
    return out

  def set_shift(self, fwd, lat=None):
    """Replace every cell's look-ahead vector (metres ahead, metres to the
    left of the direction of travel). Used for learned shifts."""
    self.shift = torch.as_tensor(fwd, dtype=torch.float32, device=self.device).detach().clone()
    self.shift_lat = (torch.zeros_like(self.shift) if lat is None else
                      torch.as_tensor(lat, dtype=torch.float32, device=self.device).detach().clone())
    self.has_lat = bool((self.shift_lat != 0).any())
    self.learned_shift = True

  def _module_counts(self, k_total, sizes):
    """How many PGCs each module gets under pgc_layout (sum == k_total)."""
    M = len(sizes)
    if k_total == 0:
      return [0] * M
    if self.pgc_layout == 'uniform':
      per = [k_total // M + (1 if m < k_total % M else 0) for m in range(M)]
    elif self.pgc_layout in ('fine', 'coarse', 'middle'):
      order = {'fine': list(range(M)), 'coarse': list(range(M))[::-1],
               'middle': sorted(range(M), key=lambda m: abs(m - (M - 1) / 2))}[self.pgc_layout]
      per, left = [0] * M, k_total
      for m in order:
        take = min(sizes[m], left)
        per[m], left = take, left - take
      if left:
        raise ValueError('pgc_frac too large')
    else:  # weights
      w = np.asarray(self.pgc_module_weights or [1.0] * M, float)
      if len(w) != M or w.sum() <= 0:
        raise ValueError('pgc_module_weights must have one non-negative weight per module')
      raw = k_total * w / w.sum()
      per = [int(np.floor(v)) for v in raw]
      # distribute the remainder by largest fractional part, respecting sizes
      rem = k_total - sum(per)
      for m in sorted(range(M), key=lambda m: -(raw[m] - np.floor(raw[m]))):
        if rem <= 0:
          break
        if per[m] < sizes[m]:
          per[m] += 1
          rem -= 1
      for m in range(M):            # cap at module size, push overflow to the next module
        if per[m] > sizes[m]:
          over, per[m] = per[m] - sizes[m], sizes[m]
          for mm in range(M):
            room = sizes[mm] - per[mm]
            take = min(room, over)
            per[mm], over = per[mm] + take, over - take
            if over == 0:
              break
    return per

  # ------------------------------------------------------------------ core
  def activity(self, pos, dir=None, vel=None, shift_scale=1.0, shift=None):
    """Population activity [B, N] in [0, 1].

    pos [B,2] allocentric position (m); dir [B,2] unit heading-of-motion
    (space mode); vel [B,2] allocentric velocity (m/s, time mode).
    shift_scale multiplies every PGC's shift (0 turns PGCs into plain grid
    cells -- the `pgc_as_gc` evaluation probe).
    shift: optional [N,2] tensor of (forward, leftward) look-aheads that
    replaces the stored ones (it may require grad: learnable shifts).
    """
    pos = pos.to(self.device, torch.float32)
    B = pos.shape[0]
    x = pos[:, None, :].expand(B, self.n_cells, 2)
    if shift is not None:
      if shift_scale != 0.0:
        src = dir if self.pgc_shift_mode == 'space' else vel
        if src is None:
          raise ValueError('shifted population needs dir (space mode) or vel (time mode)')
        src = src.to(self.device, torch.float32)
        left = torch.stack([-src[:, 1], src[:, 0]], 1)
        x = (x + (shift_scale * shift[:, 0])[None, :, None] * src[:, None, :]
             + (shift_scale * shift[:, 1])[None, :, None] * left[:, None, :])
    elif (self.has_pgc or self.learned_shift) and shift_scale != 0.0:
      src = dir if self.pgc_shift_mode == 'space' else vel
      if src is None:
        raise ValueError('PGC population needs dir (space mode) or vel (time mode)')
      src = src.to(self.device, torch.float32)
      x = x + (shift_scale * self.shift)[None, :, None] * src[:, None, :]
      if self.has_lat:
        left = torch.stack([-src[:, 1], src[:, 0]], 1)
        x = x + (shift_scale * self.shift_lat)[None, :, None] * left[:, None, :]
    if self.has_warp:
      ph = torch.einsum('bnd,nkd->bnk', x, self.warp_Q) + self.warp_psi[None]
      x = x + torch.einsum('bnk,nkd->bnd', torch.sin(ph), self.warp_A)
    proj = torch.einsum('bnd,nid->bni', x - self.phi[None], self.K)        # [B,N,3]
    if self.cell_type == 'band':
      c = torch.cos(proj)                                                  # [B,N,3]
      s1 = c.gather(2, self._band_dir_t[None, :, None].expand(B, -1, 1)).squeeze(2)
      r = (0.5 * (s1 + 1.0)).clamp(0.0, 1.0)                               # one cosine
    else:
      s = torch.cos(proj).sum(-1)                                          # [B,N] in [-1.5, 3]
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
        pgc_layout=self.pgc_layout, pgc_shift_rule=self.pgc_shift_rule,
        pgc_shift_frac=self.pgc_shift_frac, cell_type=self.cell_type,
        pgc_shift_m_per_module=[float(np.mean(self._np['shifts'][(self.module == m) & self._np['is_pgc']]))
                                if ((self.module == m) & self._np['is_pgc']).any() else 0.0
                                for m in range(self.n_modules)],
        pgc_idx=[int(i) for i in self.pgc_idx],
        pgc_per_module=[int(((self.module == m) & self._np['is_pgc']).sum())
                        for m in range(self.n_modules)],
        pgc_shift_dist=self.pgc_shift_dist, pgc_shift_dist_args=self.pgc_shift_dist_args,
        pgc_shift_angle=self.pgc_shift_angle, pgc_shift_angle_alt=self.pgc_shift_angle_alt,
        spacing_sd=self.spacing_sd,
        orient_sd_deg=self.orient_sd_deg, lattice_ellip=self.lattice_ellip,
        lattice_shear=self.lattice_shear, field_jitter=self.field_jitter,
        lattice_hash=self.lattice_hash(),
    )

  def lattice_hash(self):
    """Fingerprint of everything that defines the cells as built (wave
    vectors, phases, look-ahead vectors, warp); learned shifts are not
    included. A resumed run must reproduce it."""
    h = hashlib.sha256()
    for k in ('K', 'phi', 'shifts', 'shifts_lat', 'warp_A', 'warp_Q', 'warp_psi'):
      h.update(np.round(np.asarray(self._np[k], dtype=np.float64), 9).tobytes())
    return h.hexdigest()[:16]

  def to(self, device):
    self.device = torch.device(device)
    for k in ('K', 'phi', 'shift', 'shift_lat', 'is_pgc', '_band_dir_t',
              'warp_A', 'warp_Q', 'warp_psi'):
      setattr(self, k, getattr(self, k).to(self.device))
    return self
