"""The extension report maps evaluation files to milestones by probe and frame
count: canonical finals (eval/<probe>.json) by the frames stored in the file,
tagged files (eval/<probe>@<frames>f_<episodes>ep.json) by their tag."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def _write(path, obj):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump(obj, f)


def _ev(probe, frames, score, canonical):
  return dict(probe=probe, frames=frames, mean_score=score, canonical=canonical, episodes=200)


def make_run(root, name, final_frames, scores):
  d = os.path.join(root, name)
  _write(os.path.join(d, 'config.json'), dict(frames=final_frames, episode_len=1350))
  for probe, sc in scores.items():
    _write(os.path.join(d, 'eval', f'{probe}.json'), _ev(probe, final_frames, sc, True))
  _write(os.path.join(d, 'eval', 'untrained.json'), _ev('untrained', 0, 1.0, True))
  # demoted 300M finals, a 64-episode checkpoint probe and a 200-episode milestone
  _write(os.path.join(d, 'eval', 'intact@300006400f_200ep.json'),
         _ev('intact', 300006400, scores['intact'] - 10, False))
  _write(os.path.join(d, 'eval', 'no_code@300006400f_200ep.json'),
         _ev('no_code', 300006400, scores['no_code'] - 5, False))
  _write(os.path.join(d, 'eval', 'intact@325004800f_64ep.json'),
         _ev('intact', 325004800, 42.0, False))
  return d


def test_canonical_finals_keep_their_probe(tmp_path):
  import extension_report as er
  make_run(tmp_path, 'gc100_ffb_s1', 600000000, dict(intact=37.0, no_code=4.0, no_goal=30.0))
  make_run(tmp_path, 'pgc33_ffb_s1', 600000000, dict(intact=49.0, no_code=12.0, no_goal=40.0))
  runs = er.load_runs(str(tmp_path))
  assert [r['name'] for r in runs] == ['gc100_ffb_s1', 'pgc33_ffb_s1']
  g = runs[0]
  assert g['final_label'] == '600M'
  assert er.score(g, '600M') == 37.0
  assert er.score(g, '600M', 'no_code') == 4.0
  assert er.score(g, '600M', 'no_goal') == 30.0
  assert er.score(g, '300M') == 27.0 and er.score(g, '300M', 'no_code') == -1.0
  assert er.score(g, '600M', 'untrained') == 1.0
  assert g['ckpt_probes']['intact'] == {325004800: 42.0}
  gaps = er.gap_analysis(runs)
  per = gaps['ffb']['per_label']
  assert per['600M']['pairs'][1]['gap'] == 12.0
  assert per['300M']['pairs'][1]['gap'] == 12.0 - 0.0 + 0.0  # (49-10) - (37-10)


def test_report_runs_end_to_end(tmp_path):
  make_run(tmp_path / 'runs_ext', 'gc100_ff_s2', 600000000, dict(intact=80.0, no_code=40.0))
  make_run(tmp_path / 'runs_ext', 'pgc33_ff_s2', 600000000, dict(intact=95.0, no_code=45.0))
  out = tmp_path / 'results_ext'
  figs = tmp_path / 'figures'
  r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'extension_report.py'),
                      '--runs_dir', str(tmp_path / 'runs_ext'), '--out', str(out),
                      '--figures', str(figs)], capture_output=True, text=True)
  assert r.returncode == 0, r.stderr
  md = (out / 'EXTENSION.md').read_text()
  assert '| gc100_ff_s2 | 600 | 0 | 70.00 | 80.00 * |' in md
  assert '| ff | 600M | 1 | 80.00 | 95.00 |' in md
  summary = json.loads((out / 'extension.json').read_text())
  assert summary['runs']['pgc33_ff_s2']['milestones']['600M']['intact'] == 95.0
  assert (figs / 'fig11_training_length.png').exists()


def test_milestone_labels_round_to_two_decimals():
  import extension_report as er
  assert er.milestone_label(300006400) == '300M'
  assert er.milestone_label(600000000) == '600M'
  assert er.milestone_label(1000000000) == '1B'
  assert er.milestone_label(1500006400) == '1.5B'
  assert er.label_frames('1.5B') == 1.5e9
