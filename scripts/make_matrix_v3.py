"""Run matrices for the control wave and for extending finished runs.

  python scripts/make_matrix_v3.py controls --seeds 8 --arms ffb --frames 300000000 \
      --out /tmp/waveD_controls.json [--layouts]
  python scripts/make_matrix_v3.py extend --src runs --dst runs_ext --frames 600000000 \
      --only '^(gc100|pgc33)_(ffb|ff|lstm)_s[1-8]$' --out /tmp/waveD_ext.json [--append]

controls
  Class-size and grid-count controls for the 33 % PGC result: cond gc67 (the
  343 grid cells of the pgc33 population with the 169 PGC slots silent) and
  cond pgc50 (256 grid cells + 256 PGCs), one run per arm and seed, in the
  order gc67, pgc50 (prio 0, seeds ascending). --layouts appends the pgc50
  arrangements pgc50phase27 (shift = 0.27 x module spacing), pgc50mec
  (MEC-weighted modules 1:2:2:1) and pgc50retro (20 cm behind) at prio 1.

extend
  For every finished run in --src whose name matches --only (ckpt_final.pt
  present and below --frames): creates --dst/<name>/ with a copy of
  config.json, ckpt_final.pt, metrics.jsonl, episodes.csv and eval/ (not the
  intermediate checkpoints, not the optimizer state) and emits a scheduler
  entry that continues the run to --frames. cond is the preset part of the
  stored cond (before any '+'), arm follows from (policy, vision), and extra
  holds the population / goal / input settings whose stored value differs
  from the command-line default. --check (default) parses the exact command
  the scheduler would run for the entry (scripts/launch_matrix.py, arm
  arguments and --train_args included) and compares every EXPERIMENT_KEYS
  entry with the stored config; an entry that fails is refused and nothing is
  copied for it. --src is never written to. Repeat the subcommand with other
  --frames / --only into the same file via --append (e.g. 2B, 1B and 600M
  targets).

Every entry is {name, cond, arm, seed, frames, prio, extra}, as consumed by
scripts/launch_matrix.py --matrix_json. --append merges into an existing
file: one entry per run name, the larger frame budget wins.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import launch_matrix as lm  # noqa: E402

# Must match the --train_args default of scripts/launch_matrix.py.
TRAIN_ARGS = '--encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5'
CONTROL_CONDS = ('gc67', 'pgc50')
LAYOUTS = {
    'phase27': ['--pgc_shift_rule', 'spacing', '--pgc_shift_frac', '0.27'],
    'mec': ['--pgc_layout', 'weights', '--pgc_module_weights', '1,2,2,1'],
    'retro': ['--pgc_shift', '-0.20'],
}
# Settings reproduced from a stored config when they differ from the CLI default
# (pgc_frac: from the preset's fraction), in command-line order.
EXTRA_KEYS = ('pgc_frac', 'pgc_shift', 'pgc_layout', 'pgc_module_weights', 'pgc_shift_rule',
              'pgc_shift_frac', 'pgc_shift_jitter', 'cell_type', 'goal_code_shift',
              'extra_input', 'silence_pgc')
COPY_FILES = ('config.json', 'ckpt_final.pt', 'metrics.jsonl', 'episodes.csv')


def spec(name, cond, arm, seed, frames, prio, extra=()):
  return dict(name=name, cond=cond, arm=arm, seed=int(seed), frames=int(frames),
              prio=int(prio), extra=list(extra))


def int_frames(s):
  return int(float(s))


# ---------------------------------------------------------------- controls
def controls(args):
  arms = [a for a in args.arms.split(',') if a]
  bad = [a for a in arms if a not in lm.ARMS]
  if bad:
    raise SystemExit(f'unknown arm(s) {bad}; choose from {sorted(lm.ARMS)}')
  seeds = range(1, args.seeds + 1)
  runs = []
  for s in seeds:
    for cond in CONTROL_CONDS:
      for arm in arms:
        runs.append(spec(f'{cond}_{arm}_s{s}', cond, arm, s, args.frames, 0))
  if args.layouts:
    for s in seeds:
      for sfx, extra in LAYOUTS.items():
        for arm in arms:
          runs.append(spec(f'pgc50{sfx}_{arm}_s{s}', 'pgc50', arm, s, args.frames, 1, extra))
  return runs


# ------------------------------------------------------------------ extend
def shared_defaults():
  """(defaults, actions) of code.common.add_shared_args: dest -> default value,
  dest -> argparse action (for the option string and arity)."""
  from code.common import add_shared_args
  ap = argparse.ArgumentParser(add_help=False)
  add_shared_args(ap)
  defaults = vars(ap.parse_args([]))
  actions = {a.dest: a for a in ap._actions if a.option_strings}
  return defaults, actions


def arm_of(cfg):
  for arm, a in lm.ARMS.items():
    d = dict(zip(a[0::2], a[1::2]))
    if cfg.get('policy') == d['--policy'] and cfg.get('vision') == d['--vision']:
      return arm
  return None


def cli_value(v):
  if isinstance(v, bool):
    return str(int(v))
  return str(v)


def entry_from_config(name, cfg, frames, prio, defaults, actions, extra_args=()):
  """Scheduler entry reproducing the stored config, or (None, reason)."""
  from code.common import CONDITIONS
  preset = str(cfg.get('cond', '')).split('+')[0]
  if preset not in CONDITIONS:
    return None, f'stored cond {cfg.get("cond")!r}: preset {preset!r} is not in CONDITIONS'
  arm = arm_of(cfg)
  if arm is None:
    return None, f'no arm for policy={cfg.get("policy")!r} vision={cfg.get("vision")!r}'
  extra = []
  for k in EXTRA_KEYS:
    if k not in actions or cfg.get(k) is None:
      continue
    v = cfg[k]
    ref = CONDITIONS[preset][0] if k == 'pgc_frac' else defaults.get(k)
    if k == 'silence_pgc' and not v:
      continue
    if v == ref:
      continue
    act = actions[k]
    if act.nargs == 0:          # bare flag (store_true / store_const)
      extra.append(act.option_strings[0])
    else:
      extra += [act.option_strings[0], cli_value(v)]
  extra += list(extra_args)
  return spec(name, preset, arm, cfg['seed'], frames, prio, extra), None


def check_entry(entry, stored, out, train_args):
  """Parse the command launch_matrix.py would run for `entry` and compare the
  resulting config with the stored one on every EXPERIMENT_KEYS entry (a
  stored None or missing key counts as unset). Returns a list of problems."""
  from code.common import EXPERIMENT_KEYS, resolve_cond
  from code.train import parse_args
  argv = lm.train_cmd(entry, out, train_args.split())[3:]     # drop "python -m code.train"
  try:
    ns = parse_args(argv)
  except SystemExit:
    return [f'the scheduler command does not parse: {" ".join(argv)}']
  cfg = resolve_cond(vars(ns))
  bad = []
  for k in EXPERIMENT_KEYS:
    s = stored.get(k)
    if s is None:
      continue
    if cfg.get(k) != s:
      bad.append(f'{k}: stored {s!r}, command gives {cfg.get(k)!r}')
  return bad


def final_frames(run_dir):
  import torch
  p = os.path.join(run_dir, 'ckpt_final.pt')
  if not os.path.exists(p):
    return None
  try:
    return int(torch.load(p, map_location='cpu').get('frames', 0))
  except Exception:
    return None


def copy_run(src, dst):
  os.makedirs(dst)
  for f in COPY_FILES:
    p = os.path.join(src, f)
    if os.path.exists(p):
      shutil.copy2(p, os.path.join(dst, f))
  ev = os.path.join(src, 'eval')
  if os.path.isdir(ev):
    shutil.copytree(ev, os.path.join(dst, 'eval'))


def extend(args):
  src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
  if src == dst:
    raise SystemExit('--src and --dst must be different directories')
  if not os.path.isdir(src):
    raise SystemExit(f'--src {args.src} is not a directory')
  rx = re.compile(args.only)
  defaults, actions = shared_defaults()
  extra_args = args.extra_args.split()
  runs, refused = [], []
  for d in sorted(glob.glob(os.path.join(src, '*'))):
    name = os.path.basename(d)
    if not os.path.isdir(d) or not rx.search(name):
      continue
    if not os.path.exists(os.path.join(d, 'config.json')):
      continue
    done = final_frames(d)
    if done is None:
      print(f'skip {name}: not finished (no readable ckpt_final.pt)')
      continue
    if done >= args.frames:
      print(f'skip {name}: ckpt_final.pt already has {done} >= {args.frames} frames')
      continue
    cfg = json.load(open(os.path.join(d, 'config.json')))
    entry, why = entry_from_config(name, cfg, args.frames, args.prio, defaults, actions, extra_args)
    if entry is None:
      refused.append((name, [why]))
      print(f'REFUSED {name}: {why}')
      continue
    out = os.path.join(dst, name)
    if args.check:
      bad = check_entry(entry, cfg, out, args.train_args)
      if bad:
        refused.append((name, bad))
        print(f'REFUSED {name}: the scheduler command would not reproduce the stored config:')
        for b in bad:
          print(f'    {b}')
        continue
    if os.path.exists(out):
      print(f'{out} exists: nothing copied (entry still emitted)')
    else:
      copy_run(d, out)
      print(f'copied {name} -> {out} ({done} frames; eval/ included)')
    runs.append(entry)
  if refused:
    print(f'{len(refused)} run(s) refused: ' + ', '.join(n for n, _ in refused))
  return runs


# ------------------------------------------------------------------ output
def write_matrix(runs, out, append):
  existing = []
  if append and os.path.exists(out):
    existing = json.load(open(out))
  by_name = {}
  for r in existing + runs:
    if r['name'] in by_name:
      prev = by_name[r['name']]
      if r['frames'] != prev['frames']:
        print(f'{r["name"]}: listed twice, keeping the larger budget '
              f'{max(prev["frames"], r["frames"])}')
      prev['frames'] = max(prev['frames'], r['frames'])
    else:
      by_name[r['name']] = dict(r)
  merged = list(by_name.values())
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  with open(out, 'w') as f:
    json.dump(merged, f, indent=1)
  print(f'{len(merged)} entries -> {out} ({len(runs)} new, {len(existing)} kept)')
  groups = {}
  for r in merged:
    key = (r['cond'], r['arm'], r['frames'], ' '.join(r['extra']))
    groups.setdefault(key, []).append(r['seed'])
  for (cond, arm, frames, extra), seeds in sorted(groups.items()):
    print(f'  {cond:>8s} {arm:<4s} {frames / 1e6:7.0f}M  seeds {sorted(seeds)}'
          + (f'  extra: {extra}' if extra else ''))
  return merged


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest='kind', required=True)
  c = sub.add_parser('controls', help='gc67 / pgc50 controls (+ pgc50 layouts)')
  c.add_argument('--seeds', type=int, default=8)
  c.add_argument('--arms', default='ffb', help='comma list of ffb, ff, lstm')
  c.add_argument('--frames', type=int_frames, default=300_000_000)
  c.add_argument('--layouts', action='store_true',
                 help='also pgc50phase27, pgc50mec, pgc50retro (prio 1)')
  e = sub.add_parser('extend', help='copy finished runs and raise their budget')
  e.add_argument('--src', default='runs')
  e.add_argument('--dst', default='runs_ext')
  e.add_argument('--frames', type=int_frames, required=True, help='new budget')
  e.add_argument('--only', required=True, help='regex on the run name')
  e.add_argument('--prio', type=int, default=0)
  e.add_argument('--extra_args', default='',
                 help='appended to every entry\'s extra (e.g. "--warmup_updates 200")')
  e.add_argument('--train_args', default=TRAIN_ARGS,
                 help='the scheduler\'s --train_args (used by --check)')
  e.add_argument('--check', dest='check', action='store_true', default=True)
  e.add_argument('--no_check', dest='check', action='store_false')
  for p in (c, e):
    p.add_argument('--out', required=True)
    p.add_argument('--append', action='store_true', help='merge into an existing --out')
  args = ap.parse_args()
  runs = controls(args) if args.kind == 'controls' else extend(args)
  write_matrix(runs, args.out, args.append)


if __name__ == '__main__':
  main()
