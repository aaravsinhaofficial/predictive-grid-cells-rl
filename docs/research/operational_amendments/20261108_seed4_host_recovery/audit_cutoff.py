"""Authenticate the deliberate seed4 host cutoff; CPU reads only, no model runs."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from code import train_memory_v2 as producer

HERE = Path(__file__).resolve().parent
NAME = 'ring-20261105-simplemem-gc100-native-s4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        assert not value.is_floating_point() or bool(torch.isfinite(value).all())
    elif isinstance(value, np.ndarray):
        assert not np.issubdtype(value.dtype, np.inexact) or bool(np.isfinite(value).all())
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value)


def main():
    torch.set_num_threads(1)
    read = lambda p: json.loads(Path(p).read_text())
    folder = ROOT/'results/ring_cloud_artifacts'/NAME/'seed4'
    plan = read(ROOT/'docs/research/simple_memory_recovery_runtime_plan.json')
    row = next(x for x in plan['models'] if x.get('name') == NAME)
    ledger = read(ROOT/'docs/research/ring_cloud_budget.json')
    worker = next(x for x in ledger['runs'] if x['name'] == NAME)
    assert str(worker['instance_id']) == '2749035052516857068'
    done = read(folder.parent/'WORKER_RESULT.json')
    assert done == worker['worker_result'] and done['exit_code'] == 143
    assert done['package_sha256'] == worker['package_manifest_sha256']
    assert done['command'][1:] == row['train_argv'][1:]
    assert not (folder/'final.pt').exists() and not (folder/'summary.json').exists()
    assert not list(folder.glob('rejected*'))
    assert sorted(p.name for p in folder.glob('attempt*')) == ['attempt_000.json']
    record, interrupted, lineage, proof = [read(folder/n) for n in
        ('attempt_000.json', 'interrupted_000.json', 'lineage.json', 'provenance.json')]
    assert record == interrupted and record['exit_code'] == 143 and record['complete'] is False
    assert record['attempt'] == 0 and record['restored_frames'] == record['restored_updates'] == 0
    assert record['lineage_sha256'] == proof['lineage_sha256'] == sha(folder/'lineage.json')
    assert lineage['kind'] == 'v2_fresh' and lineage['logical_run_id'] == 'gc100/native/seed4'
    assert lineage['restored_frames'] == lineage['restored_updates'] == 0
    for path, digest in proof['source_sha256'].items(): assert sha(ROOT/path) == digest
    initial, resume, first_input = [torch.load(folder/n, map_location='cpu', weights_only=False)
        for n in ('initial.pt', 'resume.pt', 'resume_input_000.pt')]
    assert initial['frames'] == initial['updates'] == 0 and not initial['optimizer']['state']
    assert initial['policy_sha256'] == '4b6b55fc98a9dcc916b45271674299a620b9f689abc462d8b046d99f6c5be8ae'
    assert producer.payload_digest(initial) == producer.payload_digest(first_input) == lineage['initial_payload_sha256']
    assert sha(folder/'resume_input_000.pt') == record['resume_sha256']
    assert sha(folder/'resume.pt') == record['resume_sha256_after']
    assert resume['provenance'] == initial['provenance'] == proof
    assert resume['frames'] == record['final_frames'] == 9_996_800
    assert resume['updates'] == record['final_updates'] == 781
    assert resume['runtime']['counters']['frames'] == resume['frames']
    assert resume['runtime']['counters']['updates'] == resume['updates']
    assert resume['runtime']['counters']['decisions'] == resume['updates']*3200
    finite(resume)
    raw = (folder/'metrics_000.jsonl').read_text(); assert raw.endswith('\n')
    metrics = [json.loads(s) for s in raw.splitlines()]
    assert len(metrics) == resume['updates']
    for update, metric in enumerate(metrics, 1):
        assert metric['attempt'] == 0 and metric['updates'] == update
        assert metric['frames'] == update*12800 and metric['decisions'] == update*3200
        for k in ('loss','grad_norm','fps','seconds','replay_logit_max_error','replay_value_max_error'):
            assert math.isfinite(metric[k])
        assert metric['replay_guard']['accepted'] is True
        assert math.isclose(metric['fps'], metric['frames']/metric['seconds'], rel_tol=1e-9)
    files = {p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    result = dict(complete=True, audited_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_worker_name=NAME, source_instance_id=str(worker['instance_id']),
        source_run=str(folder.relative_to(ROOT)), source_files_sha256=files,
        source_worker_result=dict(path=str((folder.parent/'WORKER_RESULT.json').relative_to(ROOT)),
                                  sha256=sha(folder.parent/'WORKER_RESULT.json')),
        resume_sha256=files['resume.pt'], resume_payload_sha256=producer.payload_digest(resume),
        source_attempt=0, attempt=1, start_frames=resume['frames'], start_updates=resume['updates'],
        all_updates_contiguous=True, full_checkpoint_finite=True, original_initialization_preserved=True,
        accepted_guard_fallbacks=sum(x['replay_guard']['fallback'] for x in metrics),
        terminal_fps=metrics[-1]['fps'], scientific_source_hashes_unchanged=True)
    output = HERE/'cutoff_audit.json'; assert not output.exists()
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
