"""Tests for the control conditions gc67 and pgc50, the class-half lesion
probes, the tolerant resume-config check and the weights-only extension of a
finished run."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from code.common import (CONDITIONS, PROBES, CodeSource, build_population,  # noqa: E402
                         half_indices, resolve_cond, sham_indices)
from code.train import check_resume_config  # noqa: E402

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Reference values for the seed-1 pgc33 population, recorded with the code as
# it was before gc67 / pgc50 / the half probes were added: the sham draws and
# the PGC index set of every existing run must not move.
SHAM0_FIRST20 = [272, 480, 472, 388, 352, 332, 400, 56, 500, 444,
                 336, 156, 124, 508, 280, 380, 132, 268, 232, 424]
SHAM_SHA256 = {
    0: 'ee8532a5617c894cc5283925180ac5c345f451af796d3eb707c0ff1f3a03f329',
    1: '7f1fd7eff955fc67cbde58fddcd910a877d1995d60dc762b5df7f0c2a9ea966d',
    2: '617b4b1a2963417e48f25bf1fecd4add4dc066fb799f6151667a7e69aadd6cbb',
    3: '31c619886961337f305940e652105916fa21d9b020f8220a75209d113a3195c6',
    4: 'c174c5fc91e10ddd084d0c7d75a79b5742cb95b9cc4171128679e9c990cda4ed'}
PGC_IDX_FIRST20 = [1, 7, 8, 9, 11, 12, 14, 17, 18, 19, 23, 28, 30, 31, 33, 34, 53, 58, 60, 61]
PGC_IDX_SHA256 = '356a98b16170fe2d14ccec777a20bcb6f69b2fd4a55e8b9fba8fe4280bf8e810'


def _cfg(cond, seed=1, **over):
  """Resolved config for the standard 512-cell population (code args only)."""
  cfg = dict(cond=cond, pgc_frac=None, code_mode=None, code_seed=None, seed=seed, n_cells=512,
             n_modules=4, spacing_min=0.40, spacing_ratio=1.42, pgc_shift=0.20,
             pgc_shift_mode='space', pgc_tau=0.5, pgc_shift_jitter=0.0, sharpness=2.0,
             code_noise=0.0, pgc_layout='uniform', pgc_module_weights=None,
             pgc_shift_rule='const', pgc_shift_frac=0.5, cell_type='grid', arena_half=1.25,
             v_max=0.35, goal_code_shift='intact', silence_pgc=0)
  cfg.update(over)
  return resolve_cond(cfg)


def _train_cfg(out, **over):
  """A full training config (every EXPERIMENT_KEY) as train.py would store it."""
  base = dict(out=str(out), frames=1000, episode_len=1350, action_repeat=4, rot_max=2.2,
              goal_radius=0.3, vision='heading_only', img_noise=0.02, randomize_textures=1,
              policy='ff', conv_arch='si', nh=256, encoder_norm='layernorm', forward_bias=0.0,
              extra_input='none', n_envs=32, rollout=100, gamma=0.99, lr=3e-4, rms_alpha=0.99,
              rms_eps=1e-5, entropy=8e-5, value_coef=0.5, grad_clip=40.0, reward_clip=1.0,
              log_every=20)
  base.update(over)
  return _cfg(base.pop('cond', 'pgc33'), **base)


def _sha(idx):
  return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _per_module(pop, idx):
  return [int((pop.module[np.asarray(idx, dtype=int)] == m).sum()) for m in range(pop.n_modules)]


def _states(n=16, seed=0, device='cpu'):
  g = torch.Generator().manual_seed(seed)
  pos = (torch.rand(n, 2, generator=g) * 2 - 1) * 1.17
  ang = torch.rand(n, generator=g) * 2 * np.pi
  d = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
  return pos.to(device), d.to(device), (d * 0.35).to(device)


def _write(path, cfg):
  with open(path, 'w') as f:
    json.dump(cfg, f)
  return str(path)


def test_gc67_is_pgc33_with_its_pgcs_silenced():
  dev = torch.device(DEV)
  c67, c33 = _cfg('gc67'), _cfg('pgc33')
  assert CONDITIONS['gc67'] == (0.33, 'full')
  assert c67['cond'] == 'gc67' and c67['silence_pgc'] == 1 and c67['pgc_frac'] == 0.33
  assert c33['cond'] == 'pgc33' and c33['silence_pgc'] == 0
  p67, p33 = build_population(c67, dev), build_population(c33, dev)
  assert np.array_equal(p67.pgc_idx, p33.pgc_idx) and len(p67.pgc_idx) == 169
  assert torch.equal(p67.phi, p33.phi) and torch.equal(p67.K, p33.K)
  pos, d, v = _states(device=dev)
  g67 = CodeSource(c67, p67, dev)(pos, d, v)
  g33 = CodeSource(c33, p33, dev)(pos, d, v)
  # gc67 intact == pgc33 no_pgc == pgc33 intact with the 169 PGC entries zeroed
  assert torch.equal(g67, CodeSource(c33, p33, dev, 'no_pgc')(pos, d, v))
  expect = g33.clone()
  expect[:, torch.as_tensor(p33.pgc_idx, device=dev)] = 0.0
  assert torch.equal(g67, expect)
  assert float(g67[:, torch.as_tensor(p67.gc_idx, device=dev)].abs().sum()) > 0
  # the PGC slots stay zero under every probe, in the current and in the goal code
  pgc = torch.as_tensor(p67.pgc_idx, device=dev)
  for probe in PROBES:
    if probe == 'untrained':
      continue
    cs = CodeSource(c67, p67, dev, probe)
    assert cs.n_masked >= 169, probe
    for out in (cs(pos, d, v), cs.goal(pos, d, v)):
      assert float(out[:, pgc].abs().sum()) == 0.0, probe
  # intact == no_pgc == pgc_as_gc == pgc_half<k> exactly; no_gc silences everything
  for probe in ('no_pgc', 'pgc_as_gc', 'pgc_half0', 'pgc_half1', 'pgc_half2'):
    cs = CodeSource(c67, p67, dev, probe)
    assert cs.n_masked == 169, probe
    assert torch.equal(cs(pos, d, v), g67) and torch.equal(cs.goal(pos, d, v), g67), probe
  no_gc = CodeSource(c67, p67, dev, 'no_gc')
  assert no_gc.n_masked == 512 and float(no_gc(pos, d, v).abs().sum()) == 0.0
  # sham draws come from the 343 live grid cells, exactly as in pgc33
  for k in range(5):
    assert np.array_equal(sham_indices(p67, k), sham_indices(p33, k))
  assert CodeSource(c67, p67, dev, 'sham0').n_masked == 169 + 169
  assert CodeSource(c67, p67, dev, 'gc_half0').n_masked == 169 + 171
  # an explicit --silence_pgc on another preset is labelled honestly; gc67 is not renamed
  assert _cfg('pgc33', silence_pgc=1)['cond'] == 'pgc33+silenced'
  assert _cfg('gc67', silence_pgc=0)['silence_pgc'] == 1
  assert _cfg('gc67', pgc_frac=0.5)['cond'] == 'gc67+frac0.5+full'
  # a config without the key (every run before gc67 existed) is not silenced
  old = dict(c33)
  del old['silence_pgc']
  assert torch.equal(CodeSource(old, p33, dev)(pos, d, v), g33)


def test_sham_indices_and_pgc_indices_unchanged_for_seed1():
  pop = build_population(_cfg('pgc33'), 'cpu')
  assert pop.n_pgc == 169 and len(pop.gc_idx) == 343
  assert pop.pgc_idx[:20].tolist() == PGC_IDX_FIRST20 and _sha(pop.pgc_idx) == PGC_IDX_SHA256
  assert pop.describe()['pgc_per_module'] == [43, 42, 42, 42]
  for k, sha in SHAM_SHA256.items():
    idx = sham_indices(pop, k)
    assert len(idx) == 169 and _sha(idx) == sha, k
  assert sham_indices(pop, 0)[:20].tolist() == SHAM0_FIRST20
  # the same draws for gc67 (same live grid cells)
  assert _sha(sham_indices(build_population(_cfg('gc67'), 'cpu'), 0)) == SHAM_SHA256[0]


def test_pgc50_counts_and_half_probes():
  c = _cfg('pgc50')
  assert CONDITIONS['pgc50'] == (0.5, 'full') and c['cond'] == 'pgc50' and c['pgc_frac'] == 0.5
  pop = build_population(c, 'cpu')
  assert pop.n_pgc == 256 and _per_module(pop, pop.pgc_idx) == [64, 64, 64, 64]
  cw = _cfg('pgc50', pgc_layout='weights', pgc_module_weights='1,2,2,1')
  popw = build_population(cw, 'cpu')
  assert popw.n_pgc == 256 and _per_module(popw, popw.pgc_idx) == [43, 85, 85, 43]
  # the cells themselves are the pgc33 / gc100 cells of the same seed
  c33 = _cfg('pgc33')
  p33 = build_population(c33, 'cpu')
  assert torch.equal(pop.phi, p33.phi) and torch.equal(pop.K, p33.K)
  for p in (pop, popw):
    for cls, pool in (('pgc', p.pgc_idx), ('gc', p.gc_idx)):
      sizes = _per_module(p, pool)
      draws = [half_indices(p, cls, k) for k in range(3)]
      for k, idx in enumerate(draws):
        assert len(idx) == 128 == len(set(idx.tolist())), (cls, k)
        assert set(idx.tolist()) <= set(pool.tolist()), (cls, k)       # right class
        per = _per_module(p, idx)                                      # module-stratified
        assert all(s // 2 <= x <= (s + 1) // 2 for s, x in zip(sizes, per)), (cls, k, per)
      assert len({tuple(sorted(x.tolist())) for x in draws}) == 3, cls  # draws differ
      assert half_indices(p, cls, 1).tolist() == draws[1].tolist()      # reproducible
    for k in range(3):                                                   # disjoint classes
      assert not set(half_indices(p, 'pgc', k).tolist()) & set(half_indices(p, 'gc', k).tolist())
  # masked counts through CodeSource
  for probe, n in (('no_pgc', 256), ('no_gc', 256), ('pgc_half0', 128), ('gc_half0', 128),
                   ('pgc_half2', 128), ('gc_half2', 128), ('sham0', 169), ('intact', 0),
                   ('pgc_as_gc', 0)):
    assert CodeSource(c, pop, 'cpu', probe).n_masked == n, probe
  for probe, n in (('pgc_half0', 84), ('pgc_half2', 84), ('gc_half0', 171), ('gc_half1', 171),
                   ('sham0', 169), ('no_pgc', 169), ('no_gc', 343)):
    assert CodeSource(c33, p33, 'cpu', probe).n_masked == n, probe
  assert _per_module(p33, half_indices(p33, 'pgc', 0)) == [21, 21, 21, 21]
  assert _per_module(p33, half_indices(p33, 'gc', 0)) == [42, 43, 43, 43]
  # a half probe zeroes exactly the drawn cells, in g_t and g*, and nothing else
  pos, d, v = _states()
  g = CodeSource(c, pop, 'cpu')(pos, d, v)
  for probe, idx in (('pgc_half1', half_indices(pop, 'pgc', 1)),
                     ('gc_half2', half_indices(pop, 'gc', 2))):
    cs = CodeSource(c, pop, 'cpu', probe)
    keep = torch.as_tensor(np.setdiff1d(np.arange(512), idx))
    for out in (cs(pos, d, v), cs.goal(pos, d, v)):
      assert float(out[:, torch.as_tensor(idx)].abs().sum()) == 0.0, probe
      assert torch.equal(out[:, keep], g[:, keep]), probe
  # an empty class silences nothing (the probe equals intact)
  c100 = _cfg('gc100')
  g100 = build_population(c100, 'cpu')
  assert len(half_indices(g100, 'pgc', 0)) == 0 and len(half_indices(g100, 'gc', 0)) == 256
  assert CodeSource(c100, g100, 'cpu', 'pgc_half0').n_masked == 0
  assert torch.equal(CodeSource(c100, g100, 'cpu', 'pgc_half0')(pos, d, v),
                     CodeSource(c100, g100, 'cpu')(pos, d, v))
  p100 = build_population(_cfg('pgc100'), 'cpu')
  assert len(half_indices(p100, 'gc', 0)) == 0 and len(half_indices(p100, 'pgc', 0)) == 256


def test_check_resume_config_tolerates_stored_none_but_not_mismatch(tmp_path):
  cfg = _train_cfg(tmp_path)
  stored = dict(cfg)
  for k in ('pgc_layout', 'pgc_module_weights', 'pgc_shift_rule', 'pgc_shift_frac',
            'cell_type', 'silence_pgc'):
    stored[k] = None
  check_resume_config(cfg, _write(tmp_path / 'a.json', stored))            # None = not set
  missing = {k: v for k, v in cfg.items() if k not in ('pgc_layout', 'cell_type', 'silence_pgc')}
  check_resume_config(cfg, _write(tmp_path / 'b.json', missing))           # absent = not set
  check_resume_config(cfg, _write(tmp_path / 'c.json', dict(cfg, frames=5000, log_every=3)))
  for bad in (dict(pgc_shift=0.4), dict(cond='gc100', pgc_frac=0.0), dict(lr=1e-4),
              dict(pgc_shift_rule='spacing'), dict(cell_type='band'), dict(silence_pgc=1),
              dict(pgc_module_weights='1,2,2,1')):
    with pytest.raises(SystemExit, match='refused'):
      check_resume_config(cfg, _write(tmp_path / 'd.json', dict(cfg, **bad)))
  # the honest label of an overridden preset resolves to the stored one
  c10 = _train_cfg(tmp_path, pgc_frac=0.1)
  assert c10['cond'] == 'pgc33+frac0.1+full'
  check_resume_config(c10, _write(tmp_path / 'e.json', dict(c10)))
  with pytest.raises(SystemExit, match='refused'):
    check_resume_config(cfg, _write(tmp_path / 'f.json', dict(c10)))
  # a stored population description must be reproduced by the requested arguments
  same = json.loads(json.dumps(build_population(cfg, 'cpu').describe()))
  check_resume_config(cfg, _write(tmp_path / 'g.json', dict(cfg, population=same)))
  other = build_population(_cfg('pgc33', seed=2), 'cpu').describe()
  with pytest.raises(SystemExit, match='population'):
    check_resume_config(cfg, _write(tmp_path / 'h.json', dict(cfg, population=other)))


def _run(cmd):
  r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
  if r.returncode != 0:
    raise AssertionError(f'rc={r.returncode}: {" ".join(cmd)}\n{r.stdout}\n{r.stderr}')
  return r


def _ckpt_checksum(path):
  st = torch.load(path, map_location='cpu')['policy']
  return float(sum(v.double().abs().sum().item() for v in st.values()))


def test_weights_only_extension_smoke(tmp_path):
  out = str(tmp_path / 'ext')
  base = [sys.executable, '-m', 'code.train', '--cond', 'pgc33', '--pgc_frac', '0.1',
          '--policy', 'ff', '--vision', 'heading_only', '--seed', '3', '--n_envs', '4',
          '--rollout', '10', '--episode_len', '15', '--log_every', '1',
          '--encoder_norm', 'layernorm', '--lr', '3e-4', '--rms_eps', '1e-5',
          '--device', DEV, '--out', out]
  per_update = 4 * 10 * 4
  _run(base + ['--frames', '2000'])
  cfg = json.load(open(os.path.join(out, 'config.json')))
  assert cfg['cond'] == 'pgc33+frac0.1+full' and 'extensions' not in cfg
  ck0 = torch.load(os.path.join(out, 'ckpt_final.pt'), map_location='cpu')
  f0 = ck0['frames']
  assert f0 == 13 * per_update == 2080 and ck0['update'] == 13
  n_rows0 = len(open(os.path.join(out, 'episodes.csv')).read().splitlines()) - 1
  n_metrics0 = len(open(os.path.join(out, 'metrics.jsonl')).read().splitlines())
  assert n_rows0 > 0 and n_metrics0 == 13
  # a canonical-looking evaluation and a checkpoint evaluation; no resume.pt
  os.makedirs(os.path.join(out, 'eval'), exist_ok=True)
  _write(os.path.join(out, 'eval', 'intact.json'), dict(mean_score=1.0, canonical=True))
  _write(os.path.join(out, 'eval', f'no_code@{f0}f_64ep.json'), dict(mean_score=2.0, canonical=False))
  os.remove(os.path.join(out, 'resume.pt'))
  r = _run(base + ['--frames', '4000', '--resume', '--warmup_updates', '3'])
  assert f'extending run from {f0} to 4000 frames (weights_only)' in r.stdout
  assert 'weights-only extension from ckpt_0000002080.pt' in r.stdout
  # demotion: old final checkpoint kept under its frame count, canonical eval renamed
  demoted = os.path.join(out, f'ckpt_{f0:010d}.pt')
  assert torch.load(demoted, map_location='cpu')['frames'] == f0
  assert not os.path.exists(os.path.join(out, 'eval', 'intact.json'))
  dem = json.load(open(os.path.join(out, 'eval', f'intact@{f0}f_200ep.json')))
  assert dem['mean_score'] == 1.0 and dem['canonical'] is False
  assert os.path.exists(os.path.join(out, 'eval', f'no_code@{f0}f_64ep.json'))
  # the run continued to the new budget; provenance in config.json
  ck1 = torch.load(os.path.join(out, 'ckpt_final.pt'), map_location='cpu')
  assert ck1['frames'] == 4000 and ck1['update'] == 13 + 12 and ck1['episodes_seen'] > n_rows0
  cfg = json.load(open(os.path.join(out, 'config.json')))
  assert cfg['frames'] == 4000 and cfg['pgc_frac'] == 0.1 and cfg['cond'] == 'pgc33+frac0.1+full'
  assert cfg['extensions'] == [dict(from_frames=f0, to_frames=4000, mode='weights_only')]
  # metrics: warm-up start / end records; frames and update advance only afterwards
  recs = [json.loads(l) for l in open(os.path.join(out, 'metrics.jsonl')).read().splitlines()]
  new = recs[n_metrics0:]
  warm = [x for x in new if x.get('warmup')]
  assert [x['event'] for x in warm] == ['warmup_start', 'warmup_end']
  assert all(x['frames'] == f0 and x['update'] == 13 and x['episodes'] == n_rows0 for x in warm)
  assert warm[0]['lr'] == 0.0 and warm[1]['lr'] == 3e-4
  learn = [x for x in new if not x.get('warmup')]
  assert new.index(warm[1]) < new.index(learn[0])
  assert [x['frames'] for x in learn] == [f0 + per_update * (i + 1) for i in range(12)]
  assert learn[0]['update'] == 14 and learn[0]['episodes'] >= n_rows0
  # weights unchanged through the warm-up (checksum of the demoted checkpoint), changed after
  c_before = _ckpt_checksum(demoted)
  assert abs(warm[0]['param_checksum'] - c_before) < 1e-3
  assert abs(warm[1]['param_checksum'] - c_before) < 1e-3
  assert abs(_ckpt_checksum(os.path.join(out, 'ckpt_final.pt')) - c_before) > 1e-3
  # no episode rows during the warm-up: every new row lies beyond the old budget
  rows = open(os.path.join(out, 'episodes.csv')).read().splitlines()[1:]
  assert len(rows) > n_rows0 and all(int(x.split(',')[0]) > f0 for x in rows[n_rows0:])
  # resume.pt is back: a finished run is a no-op, a further extension takes the resume path
  assert os.path.exists(os.path.join(out, 'resume.pt'))
  assert 'already finished' in _run(base + ['--frames', '4000', '--resume']).stdout
  r = _run(base + ['--frames', '4800', '--resume', '--warmup_updates', '3'])
  assert 'extending run from 4000 to 4800 frames (resume)' in r.stdout
  assert 'resumed at frames=4000 update=25' in r.stdout
  cfg = json.load(open(os.path.join(out, 'config.json')))
  assert cfg['frames'] == 4800 and [e['mode'] for e in cfg['extensions']] == ['weights_only', 'resume']
  recs = [json.loads(l) for l in open(os.path.join(out, 'metrics.jsonl')).read().splitlines()]
  assert sum(1 for x in recs if x.get('warmup')) == 2          # no warm-up on the resume path
  assert torch.load(os.path.join(out, 'ckpt_final.pt'), map_location='cpu')['frames'] == 4800
  assert os.path.exists(os.path.join(out, f'ckpt_{4000:010d}.pt'))
  assert os.path.exists(os.path.join(out, 'eval', f'intact@{f0}f_200ep.json'))
  # a directory left with neither resume.pt nor ckpt_final.pt (an extension
  # killed during its warm-up) continues from its latest checkpoint's weights,
  # with the warm-up, instead of restarting from zero
  os.remove(os.path.join(out, 'resume.pt'))
  os.remove(os.path.join(out, 'ckpt_final.pt'))
  r = _run(base + ['--frames', '5600', '--resume', '--warmup_updates', '2'])
  assert 'continuing from ckpt_0000004000.pt (4000 frames)' in r.stdout
  cfg = json.load(open(os.path.join(out, 'config.json')))
  assert cfg['frames'] == 5600 and cfg['cond'] == 'pgc33+frac0.1+full'
  assert cfg['extensions'][-1] == dict(from_frames=4000, to_frames=5600, mode='recover_weights_only')
  ck2 = torch.load(os.path.join(out, 'ckpt_final.pt'), map_location='cpu')
  assert ck2['frames'] == 5600 and ck2['update'] == 25 + 10
  assert os.path.exists(os.path.join(out, 'resume.pt'))
  recs = [json.loads(l) for l in open(os.path.join(out, 'metrics.jsonl')).read().splitlines()]
  assert [x['event'] for x in recs if x.get('warmup')][-2:] == ['warmup_start', 'warmup_end']
