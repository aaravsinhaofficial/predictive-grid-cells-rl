import hashlib,json,sys,torch
from pathlib import Path
root=Path.cwd().resolve();sys.path[:0]=[str(root),str(root/'scripts')]
import local_return_feedback_portable_run as runner
before=torch.get_rng_state().clone()
path=root/'docs/research/local_return_feedback_portable_protocol.json';assert hashlib.sha256(path.read_bytes()).hexdigest()=='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
frozen=json.loads(path.read_text());frozen.pop('frozen_utc');assert runner.protocol()==frozen
pins={**frozen['source_sha256'],**frozen['input_assets']}
for name,expected in pins.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
assert torch.equal(before,torch.get_rng_state())
(root/'reproduced/pure_protocol_check.json').write_text(json.dumps(dict(passed=True,exact_protocol_and_bank_checks=True,all_pinned_assets_checked=len(pins),global_cpu_rng_preserved=True,no_model_or_environment_execution=True,torch_version=torch.__version__),indent=2)+'\n')
print('Pure protocol, saved-bank and input checks passed.')
