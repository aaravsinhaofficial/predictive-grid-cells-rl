"""Final tests that a stopped spot VM left empty are run again here from the saved final network.

A spot VM can be switched off while a test result is being written. The file then exists but is empty, and
the VM's scheduler does not run that test again after the restart. This script finds such files in the
collected runs of VMs that have finished (runs_codes/<block>/<run>/eval/<probe>.json: empty, unreadable, or
from a network that was not trained for the whole budget), fetches the run's final network from the bucket
and runs the same test (code/evaluate.py, 200 episodes, the same test seed).

  python -m scripts.codes.fill_tests              # list what is missing
  python -m scripts.codes.fill_tests --run        # run the missing tests; record them in data/refilled_tests.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, 'runs_codes')
BUCKET = 'gs://connectome-ai-research-popcode/codes'
GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
M = 1_000_000


def state(path, budget):
  """'ok' | 'missing' | 'empty' | 'earlier network'"""
  if not os.path.exists(path):
    return 'missing'
  try:
    j = json.load(open(path))
  except ValueError:
    return 'empty'
  return 'ok' if j.get('frames') is None or j['frames'] >= budget - 1.5 * M else 'earlier network'


def find(tags):
  holes = []
  for tag in tags:
    plan_path = os.path.join(BASE, tag, 'plan.json')
    probes_path = os.path.join(BASE, tag, 'probes.txt')
    if not (os.path.exists(plan_path) and os.path.exists(probes_path)):
      continue
    plan = json.load(open(plan_path))
    vm_of = {r: vm for vm, v in plan.items() for r in v['runs']}
    # Only a VM that has finished will never run the test itself (its DONE marker is in the bucket).
    ls = subprocess.run(f'{GSUTIL} ls "{BUCKET}/{tag}/runs/*/DONE"', shell=True, capture_output=True, text=True)
    done_vms = {u.split('/')[-2] for u in ls.stdout.split()} if ls.returncode == 0 else set()
    probes = open(probes_path).read().strip().split(',')
    for cfg_path in sorted(glob.glob(os.path.join(BASE, tag, '*', 'config.json'))):
      d = os.path.dirname(cfg_path)
      run = os.path.basename(d)
      if vm_of.get(run) not in done_vms:
        continue
      budget = int(json.load(open(cfg_path)).get('frames', 0))
      st = {p: state(os.path.join(d, 'eval', f'{p}.json'), budget) for p in probes}
      if all(s == 'ok' for s in st.values()) or all(s == 'missing' for s in st.values()):
        continue          # complete, or not tested yet (still training)
      holes.append(dict(tag=tag, run=run, dir=d, vm=vm_of[run], budget=budget,
                        probes={p: s for p, s in st.items() if s != 'ok'}))
  return holes


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tags', default='oc,os,dr,tr,lstm,trl,ox,dx,lx')
  ap.add_argument('--run', action='store_true')
  ap.add_argument('--python', default=sys.executable)
  args = ap.parse_args()
  holes = find(args.tags.split(','))
  for h in holes:
    print(f'{h["tag"]}/{h["run"]} ({h["budget"] // M} M): ' + ', '.join(f'{p} {s}' for p, s in h['probes'].items()))
  print(f'{len(holes)} runs with incomplete final tests')
  if not args.run or not holes:
    return
  import torch
  rec_path = os.path.join(ROOT, 'results', 'code_variations', 'data', 'refilled_tests.json')
  rec = json.load(open(rec_path)) if os.path.exists(rec_path) else []
  for h in holes:
    ck = os.path.join(h['dir'], 'ckpt_final.pt')
    have = os.path.exists(ck) and int(torch.load(ck, map_location='cpu').get('frames', 0)) >= h['budget'] - 1.5 * M
    if not have:
      r = subprocess.run([GSUTIL, '-q', 'cp', f'{BUCKET}/{h["tag"]}/runs/{h["vm"]}/{h["run"]}/ckpt_final.pt', ck],
                         capture_output=True, text=True)
      have = (r.returncode == 0 and
              int(torch.load(ck, map_location='cpu').get('frames', 0)) >= h['budget'] - 1.5 * M)
    if not have:
      print(f'  {h["tag"]}/{h["run"]}: no final network of {h["budget"] // M} M frames yet, skipped')
      continue
    for p in h['probes']:          # an empty file would otherwise stay if the test fails
      f = os.path.join(h['dir'], 'eval', f'{p}.json')
      if os.path.exists(f):
        os.remove(f)
    r = subprocess.run([args.python, '-m', 'code.evaluate', '--run', h['dir'], '--probes', ','.join(h['probes']),
                        '--episodes', '200', '--ckpt', 'final'], cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0 and all(state(os.path.join(h['dir'], 'eval', f'{p}.json'), h['budget']) == 'ok'
                                   for p in h['probes'])
    print(f'  {h["tag"]}/{h["run"]}: {"tests run again: " + ", ".join(h["probes"]) if ok else "FAILED " + r.stderr[-300:]}')
    if ok:
      rec = [x for x in rec if not (x['block'] == h['tag'] and x['run'] == h['run'])]
      rec.append(dict(block=h['tag'], run=h['run'], trained_M=h['budget'] // M, tests=h['probes']))
  os.makedirs(os.path.dirname(rec_path), exist_ok=True)
  json.dump(sorted(rec, key=lambda x: (x['block'], x['run'])), open(rec_path, 'w'), indent=1)
  print(f'{len(rec)} runs recorded in {rec_path}')


if __name__ == '__main__':
  main()
