"""Provision the missing plotting dependency, then run the unchanged reset assay."""
import argparse
import importlib.metadata
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
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'torch')}
    constraints = args.out / 'preserved_numerical_dependencies.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()))
    install = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
               '--constraint', str(constraints), 'matplotlib==3.9.2']
    started = time.monotonic()
    subprocess.run(install, check=True, timeout=180)
    installed = json.loads(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'list', '--format=json', '--disable-pip-version-check'], text=True))
    after = {item['name'].lower(): item['version'] for item in installed}
    assert all(after[name] == version for name, version in versions.items())
    assert after['matplotlib'] == '3.9.2'
    smoke = """import sys,json; from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path.cwd()/'scripts'))
import matplotlib;import lstm_final75_reset_runtime_adapter as adapter
saved=json.loads(Path('docs/research/final75_reset_runtime_adapter_protocol.json').read_text());saved.pop('frozen_at')
assert adapter.design()==saved
print('Clean worker import and exact adapter protocol passed; matplotlib='+matplotlib.__version__)
"""
    result = subprocess.run([sys.executable, '-c', smoke], text=True, capture_output=True, timeout=60)
    report = dict(python=sys.version, executable=sys.executable, install_command=install,
                  preserved_numerical_versions=versions, installed_packages=installed,
                  smoke_returncode=result.returncode, smoke_stdout=result.stdout,
                  smoke_stderr=result.stderr, elapsed_seconds=time.monotonic()-started)
    (args.out / 'environment_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(result.stdout, result.stderr, flush=True)
    result.check_returncode()
    subprocess.run([sys.executable, 'scripts/lstm_final75_reset_runtime_adapter.py',
                    '--out', str(args.out), '--device', 'cuda', '--max-seconds', '240'],
                   check=True, timeout=250)


if __name__ == '__main__':
    main()
