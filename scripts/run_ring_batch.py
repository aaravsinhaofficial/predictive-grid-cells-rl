"""Run an explicit, finite list of independent training jobs on one worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-seconds', type=float, default=6000)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 86400:
        raise ValueError('A finite positive job deadline is required')
    jobs = json.loads(args.manifest.read_text())['jobs']
    if not 1 <= len(jobs) <= 128:
        raise ValueError('Manifest must contain 1..128 explicit jobs')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'manifest.json').write_text(args.manifest.read_text())
    started, records = time.monotonic(), []
    for index, job in enumerate(jobs):
        elapsed = time.monotonic()-started
        remaining = args.max_seconds-elapsed
        if remaining < 60:
            break
        name = job['name']
        if not name.replace('_','').replace('-','').isalnum():
            raise ValueError('Unsafe job name')
        output = args.out/f'{name}.pt'
        if output.exists():
            raise FileExistsError(output)
        script=job.get('script','scripts/train_ring_task.py')
        if script not in ('scripts/train_ring_task.py','scripts/train_ring_heading.py'):
            raise ValueError('Only audited ring trainers are supported')
        command = [sys.executable, script, '--device', args.device,
                   '--out', str(output), '--max-seconds', str(max(1, remaining-30))]
        for key, value in job['args'].items():
            if key in ('device','out','max-seconds'):
                raise ValueError('Manifest cannot override worker bounds')
            command += ['--'+key, str(value)]
        print(json.dumps(dict(event='job_start', index=index, name=name, command=command)), flush=True)
        with (args.out/f'{name}.stdout').open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        record = dict(name=name, exit_code=proc.returncode, elapsed_seconds=time.monotonic()-started)
        if output.with_suffix('.json').exists():
            summary = json.loads(output.with_suffix('.json').read_text())
            record['validation'] = summary['validation']
            record['steps_completed'] = summary['steps_completed']
        records.append(record)
        (args.out/'batch_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if proc.returncode:
            raise RuntimeError(f'Job {name} failed: see {name}.stdout')
    (args.out/'batch_result.json').write_text(json.dumps(dict(
        planned_jobs=len(jobs), completed_jobs=len(records), jobs=records,
        elapsed_seconds=time.monotonic()-started), indent=2)+'\n')


if __name__ == '__main__':
    main()
