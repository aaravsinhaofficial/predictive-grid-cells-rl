"""Run two separately frozen final-checkpoint diagnostics on one bounded worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ('matched_noise', 'scripts/eval_visual_memory_matched_noise.py', args.out, 600),
        ('every_blind_reset', 'scripts/lstm_final75_reset_runtime_adapter.py',
         args.out / 'final75_every_blind_reset', 240),
    ]
    reports = []
    for name, source, destination, seconds in jobs:
        command = [sys.executable, source, '--out', str(destination),
                   '--max-seconds', str(seconds)]
        started = time.monotonic()
        try:
            result = subprocess.run(command, check=False, timeout=seconds + 10)
            status = dict(exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            status = dict(exit_code=124, timeout=True)
        reports.append(dict(name=name, command=command,
                            elapsed_seconds=time.monotonic() - started, **status))
        (args.out / 'diagnostic_status.json').write_text(json.dumps(reports, indent=2) + '\n')
    if any(item['exit_code'] != 0 for item in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
