"""Prepare exact frozen acquisition inputs and four bounded launch commands; never launch."""
from __future__ import annotations
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/research/local_return_feedback_portable_protocol.json'
PROTOCOL_SHA='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
COMMIT='f1e5e1725c4449ee6c8bd39cac022b9e94b0139f'
OUT=ROOT/'results/local_return_feedback_design/portable/cloud_package'
ROUTING=ROOT/'docs/research/local_return_feedback_portable_launch.json'

STARTUP = r"""import sys,json,hashlib,platform,datetime
from pathlib import Path
root=Path.cwd();sys.path[:0]=[str(root),str(root/'scripts')]
import torch,numpy
import local_return_feedback_portable_run as runner
import local_return_feedback_portable_evaluate as evaluator
protocol_path=root/'docs/research/local_return_feedback_portable_protocol.json'
protocol_sha='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==protocol_sha
torch.set_num_threads(2)
before=evaluator.global_rng_snapshot()
frozen=json.loads(protocol_path.read_text());assert frozen.pop('frozen_utc');assert runner.protocol()==frozen
plan,bank,_=evaluator.load_bank()
assert runner.tree_equal(before,evaluator.global_rng_snapshot())
args=sys.argv[1:];out=Path(args[args.index('--out')+1]).parent
record=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=protocol_sha,
 bank_sha256=bank['sha256'],bank_tensor_schema_exact=True,source_and_input_hashes_exact=True,
 origin_validation_sha256=evaluator.ORIGIN_SHA256,global_cpu_cuda_mps_rng_unchanged=True,
 policy_initialized=False,policy_forward=False,environment_step=False,optimizer_step=False,
 python=sys.version,torch=str(torch.__version__),numpy=numpy.__version__,machine=platform.machine(),system=platform.system())
(out/'portable_bank_runtime_check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
del plan,before
sys.argv=['scripts/local_return_feedback_portable_run.py']+args
runner.main()
"""

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    assert not ROUTING.exists(), 'Preserve existing launch routing'
    design=json.loads(PROTOCOL.read_text())
    entries={**design['source_sha256'],**design['input_assets'],str(PROTOCOL.relative_to(ROOT)):PROTOCOL_SHA}
    manifest=[]
    for name,digest in sorted(entries.items()):
        p=ROOT/name;assert p.is_file() and not p.is_symlink() and sha(p)==digest,name
        manifest.append(dict(path=name,sha256=digest,bytes=p.stat().st_size))
    OUT.mkdir(parents=True,exist_ok=True)
    archive=OUT/'feedback_acquisition_portable_inputs.tar.gz';assert not archive.exists()
    with archive.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0,compresslevel=1) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for item in manifest:
            data=(ROOT/item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
            info=tarfile.TarInfo(item['path']);info.mode=0o644;info.size=len(data);tar.addfile(info,io.BytesIO(data))
    with tempfile.TemporaryDirectory(prefix='feedback-final-package-') as temp:
        path=Path(temp)
        with tarfile.open(archive) as tar:tar.extractall(path,filter='data')
        preflight="""import sys,json,hashlib\nfrom pathlib import Path\nr=Path.cwd();sys.path[:0]=[str(r),str(r/'scripts')]\nimport local_return_feedback_portable_run as runner\nimport local_return_feedback_portable_evaluate as evaluator\nimport audit_local_return_feedback\np=r/'docs/research/local_return_feedback_portable_protocol.json'\nf=json.loads(p.read_text());f.pop('frozen_utc');assert runner.protocol()==f\nfor name,h in {**f['source_sha256'],**f['input_assets']}.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h\nassert all(not getattr(m,'__file__',None) or '/Users/aaravsinha/predictive-grid-cells-rl/' not in str(m.__file__) for m in sys.modules.values())\nprint(json.dumps({'passed':True,'imports':['runner','evaluator','auditor'],'source_and_asset_hashes':True,'policy_forward':False,'environment_step':False,'optimizer_step':False}))\n"""
        check=subprocess.run([sys.executable,'-I','-c',preflight],cwd=path,text=True,capture_output=True,check=True)
    ledger=json.loads((ROOT/'docs/research/ring_cloud_budget.json').read_text())
    committed=ledger['ancillary_reserve_usd']+sum(r.get('actual_upper_bound_usd',r['reserved_usd']) for r in ledger['runs'])
    assert committed+21.25<=500
    workers=[]
    for condition,pop in [('gc100','grid'),('pgc33','pgc')]:
        for mode,short in [('retained_actor','retain'),('reset_actor','reset')]:
            name=f'ring-20261105-feedback-{pop}-{short}-02';arm=f'{condition}_{mode}'
            output=f'results/ring_feedback_acquisition_portable/{arm}'
            command=['python','-c',STARTUP,'--condition',condition,'--actor-mode',mode,
                     '--device','cuda','--out',output+'/experiment','--max-seconds','12600']
            import shlex
            argv=[sys.executable,'scripts/ring_cloud.py','launch','--name',name,'--zone','us-central1-a',
                  '--machine','n1-standard-8','--hours','4','--output-dir',output,'--command',shlex.join(command),'--include-only']
            for item in manifest:argv.extend(['--include',item['path']])
            workers.append(dict(condition=condition,actor_mode=mode,name=name,zone='us-central1-a',
                artifact_subdir='experiment',local_artifact_root=f'results/ring_cloud_artifacts/{name}',
                output_dir=output,command=command,launch_argv=argv,status='prepared',attempt=1,
                audit_path=f'results/local_return_feedback_design/portable/cloud_audit/{arm}.json'))
    record=dict(status='prepared_not_reserved_or_launched',prepared_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        protocol_sha256=PROTOCOL_SHA,source_commit=COMMIT,project='connectome-ai-research',
        machine='n1-standard-8',accelerator='nvidia-tesla-t4',provider_delete_seconds=14400,process_max_seconds=12600,
        original_budget_usd=500,maximum_authorized_usd=25,planned_reservation_usd=21.25,
        cumulative_before_usd=round(committed,4),prior_failed_attempt_cost_usd=.2468,combined_pilot_max_reservation_usd=21.4968,remaining_after_reservation_usd=round(500-committed-21.25,4),
        no_automatic_retry_or_extension=True,include_only=True,include=[m['path'] for m in manifest],
        package_manifest=manifest,canonical_archive=dict(path=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size),
        isolated_preflight=json.loads(check.stdout),workers=workers)
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    ROUTING.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(files=len(manifest),uncompressed_bytes=sum(m['bytes'] for m in manifest),archive=record['canonical_archive'],workers=[w['name'] for w in workers],preflight=record['isolated_preflight'])))

if __name__=='__main__':main()
