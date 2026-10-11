"""One-screen progress summary of every run under runs/."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'runs')
rows = []
for d in sorted(d for d in glob.glob(os.path.join(runs_dir, '*')) if os.path.isdir(d)):
  cfg_p = os.path.join(d, 'config.json')
  if not os.path.exists(cfg_p):
    continue
  cfg = json.load(open(cfg_p))
  last = None
  mp = os.path.join(d, 'metrics.jsonl')
  if os.path.exists(mp):
    with open(mp) as f:
      for line in f:
        if line.strip():
          last = json.loads(line)
  final = os.path.exists(os.path.join(d, 'ckpt_final.pt'))
  ev = os.path.join(d, 'eval', 'intact.json')
  score = json.load(open(ev))['mean_score'] if os.path.exists(ev) else None
  frames = last['frames'] if last else 0
  fps = last['fps'] if last else 0
  eta_h = (cfg['frames'] - frames) / fps / 3600 if fps and not final else 0.0
  rows.append((os.path.basename(d), frames / 1e6, cfg['frames'] / 1e6, fps,
               last['avg_return_100'] if last else None,
               last['avg_goals_100'] if last else None,
               last['path_eff_100'] if last else None,
               last['entropy'] if last else None,
               'final' if final else ('running' if fps else 'idle'), eta_h, score))
print(f'{"run":18s} {"Mframes":>8s}/{"budget":>6s} {"fps":>6s} {"ret100":>7s} {"goals":>6s} '
      f'{"patheff":>7s} {"ent":>5s} {"state":>8s} {"eta_h":>6s} {"evalscore":>9s}')
for r in rows:
  fmt = lambda v, w, p=2: (f'{v:{w}.{p}f}' if isinstance(v, (int, float)) and v is not None else f'{"-":>{w}s}')
  print(f'{r[0]:18s} {r[1]:8.1f}/{r[2]:6.0f} {r[3]:6.0f} {fmt(r[4],7)} {fmt(r[5],6)} '
        f'{fmt(r[6],7,3)} {fmt(r[7],5)} {r[8]:>8s} {r[9]:6.1f} {fmt(r[10],9)}')
