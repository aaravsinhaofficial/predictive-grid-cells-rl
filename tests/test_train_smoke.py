import json
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _run(cmd, check=True):
  return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True)


def test_train_and_eval_smoke(tmp_path):
  out = str(tmp_path / 'run')
  base = [sys.executable, '-m', 'code.train', '--cond', 'pgc33', '--policy', 'lstm',
          '--seed', '1', '--n_envs', '8', '--rollout', '30', '--episode_len', '40',
          '--log_every', '1', '--out', out, '--device', DEV, '--encoder_norm', 'layernorm',
          '--lr', '3e-4', '--rms_eps', '1e-5']
  _run(base + ['--frames', str(4 * 8 * 30 * 2)])
  cfg = json.load(open(os.path.join(out, 'config.json')))
  assert cfg['population']['n_pgc'] == 169 and cfg['pgc_frac'] == 0.33
  assert os.path.exists(os.path.join(out, 'ckpt_final.pt'))
  lines = open(os.path.join(out, 'metrics.jsonl')).read().strip().splitlines()
  assert len(lines) == 2
  rec = json.loads(lines[-1])
  assert rec['frames'] == 4 * 8 * 30 * 2
  assert all(v is not None for v in (rec['pol_loss'], rec['val_loss'], rec['grad_norm']))
  assert 'frac_saturated' in rec and rec['frac_saturated'] == 0.0
  eps = open(os.path.join(out, 'episodes.csv')).read().strip().splitlines()
  assert eps[0].startswith('frames,update,env,ret,')
  assert len(eps) >= 1 + 1   # staggered clocks: only complete episodes are logged
  # resume is a no-op once finished
  _run(base + ['--frames', str(4 * 8 * 30 * 2), '--resume'])
  # a changed experiment definition is refused on resume of an unfinished run
  out2 = str(tmp_path / 'run2')
  b2 = [a if a != out else out2 for a in base]
  _run(b2 + ['--frames', str(4 * 8 * 30 * 3), '--resume_every', '1', '--ckpt_every', '10'])
  os.remove(os.path.join(out2, 'ckpt_final.pt'))
  r = _run([a if a != 'pgc33' else 'gc100' for a in b2] + ['--frames', '1', '--resume'], check=False)
  assert r.returncode != 0 and 'refused' in (r.stdout + r.stderr)
  # evaluation probes run end to end; provenance naming for non-canonical evals
  ev = [sys.executable, '-m', 'code.evaluate', '--run', out, '--episodes', '4',
        '--n_envs', '4', '--probes', 'intact,no_code,no_pgc,pgc_as_gc,no_goal,untrained',
        '--device', DEV]
  _run(ev)
  files = os.listdir(os.path.join(out, 'eval'))
  for p in ('intact', 'no_code', 'no_pgc', 'pgc_as_gc', 'no_goal', 'untrained'):
    f = [x for x in files if x.startswith(p + '@')]
    assert len(f) == 1, (p, files)
    r = json.load(open(os.path.join(out, 'eval', f[0])))
    assert r['episodes'] == 4 and 'mean_score' in r and r['canonical'] is False
  bad = _run(ev[:-2] + ['--probes', 'intact,nocode'], check=False)
  assert bad.returncode != 0 and 'unknown probe' in (bad.stdout + bad.stderr)


def test_ff_none_and_time_mode_variants(tmp_path):
  for i, extra in enumerate((['--policy', 'ff', '--cond', 'none'],
                             ['--policy', 'ff', '--cond', 'pgc100', '--pgc_shift_mode', 'time',
                              '--extra_input', 'vel', '--goal_code_shift', 'zero'])):
    out = str(tmp_path / f'v{i}')
    cmd = [sys.executable, '-m', 'code.train', '--seed', '2', '--n_envs', '4',
           '--rollout', '10', '--episode_len', '15', '--frames', str(4 * 4 * 10 * 2),
           '--log_every', '1', '--out', out, '--device', DEV] + extra
    _run(cmd)
    assert os.path.exists(os.path.join(out, 'ckpt_final.pt'))
