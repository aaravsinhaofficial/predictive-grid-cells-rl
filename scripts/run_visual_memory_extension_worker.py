"""Run two separately frozen reset diagnostics, then the unchanged fixed extension."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {'gc100': 'ring-20261105-vision-grid-01', 'pgc33': 'ring-20261105-vision-pgc-01'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition', choices=tuple(PARENTS), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parent = ROOT / 'results/ring_cloud_artifacts' / PARENTS[args.condition]
    original = ROOT / f'models/{args.condition}/{args.condition}_lstm_s1_1000M.pt'
    frozen = ROOT / 'results/lstm_endogenous_ring/every_blind_update_reset/protocol.json'
    records = []
    for label, checkpoint, reference in [('original', original, parent / 'baseline.json'),
                                          ('v1final', parent / 'final.pt', parent / 'final_evaluation.json')]:
        folder = args.out / f'every_blind_reset_{label}'
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen, folder / 'protocol.json')
        command = [sys.executable, str(ROOT / 'scripts/lstm_endogenous_every_blind_reset.py'),
                   '--checkpoint', str(checkpoint), '--reference', str(reference),
                   '--device', 'cuda', '--out', str(folder)]
        started = time.monotonic()
        with (folder / 'worker.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=150)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        records.append(dict(label=label, command=command, exit_code=exit_code,
                            elapsed_seconds=time.monotonic() - started,
                            finished_at=dt.datetime.now(dt.timezone.utc).isoformat()))
        (args.out / 'diagnostic_status.json').write_text(json.dumps(records, indent=2) + '\n')
    command = [sys.executable, str(ROOT / 'scripts/rl_visual_memory_extend.py'),
               '--condition', args.condition, '--out', str(args.out), '--device', 'cuda', '--max-seconds', '6600']
    # Diagnostic failure is retained separately; it cannot select or change training.
    result = subprocess.run(command)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
