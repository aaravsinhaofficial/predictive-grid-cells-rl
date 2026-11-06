"""Compare saved baseline first-choice representations; no model or rollout execution."""
from __future__ import annotations
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'docs/research/local_return_feedback_portable_protocol.json'
PROTOCOL_SHA = '4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
BASELINES = 'results/local_return_feedback_design/portable/cloud_audit/all_baselines_independent_review.json'
BASELINES_SHA = '73e7d0b3ccded5b155b6728608a4e3a9924a3523fb0ed1a19501a88e290da698'
SAME = ('state', 'input', 'position', 'heading', 'motors', 'value')
OUTPUT = ('probabilities', 'logits')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def tensor_record(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest(), bytes=array.nbytes)


def run(root=ROOT):
    root = Path(root).resolve(); inputs = {}

    def checked(relative, digest, size=None):
        path = root/relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert sha(path) == digest, relative
        if size is not None: assert path.stat().st_size == size
        inputs[str(relative)] = digest
        return path

    design = read(checked(PROTOCOL, PROTOCOL_SHA))
    sealed = read(checked(BASELINES, BASELINES_SHA))
    assert sealed['passed'] and sealed['complete_all_four_baselines']
    init_path = 'docs/research/local_return_feedback_initialization.json'
    initial = read(checked(init_path, design['input_assets'][init_path]))
    assets = {(r['condition'], r['actor_mode']):r for r in initial['policies']}
    arms = {}; policies = {}
    for row in sealed['rows']:
        condition, mode = row['condition'], row['actor_mode']; key = (condition, mode)
        proof = row['baseline_audit']; audit = read(checked(proof['path'], proof['sha256'], proof.get('bytes')))
        assert audit['passed'] and audit['protocol_sha256'] == PROTOCOL_SHA
        assert (audit['condition'], audit['actor_mode']) == key
        assert audit['checks']['evaluations']['baseline']['complete']
        folder = Path('results/ring_cloud_artifacts')/row['worker']/'experiment'
        assert Path(audit['folder']).parts[-len(folder.parts):] == folder.parts
        record = audit['artifacts']['incoming.pt']
        incoming = torch.load(checked(folder/'incoming.pt', record['sha256'], record['bytes']), map_location='cpu', weights_only=True)
        asset = assets[key]
        assert asset['sha256'] == design['input_assets'][asset['path']]
        saved = torch.load(checked(asset['path'], asset['sha256'], asset['bytes']), map_location='cpu', weights_only=True)
        assert set(saved['policy']) == set(incoming['policy'])
        assert all(tensor_record(v) == tensor_record(incoming['policy'][k]) for k,v in saved['policy'].items())
        assert incoming['counters']['updates'] == 0
        policies[key] = {k:tensor_record(v) for k,v in saved['policy'].items()}
        arms[key] = dict(folder=folder, audit=audit, totals=row['baseline_totals'])
    assert set(arms) == {(c,m) for c in ('gc100','pgc33') for m in ('retained_actor','reset_actor')}
    models = {}
    for condition in ('gc100','pgc33'):
        left, right = [(condition,mode) for mode in ('retained_actor','reset_actor')]
        changed = sorted(k for k in policies[left] if policies[left][k] != policies[right][k])
        assert changed == ['pi.bias','pi.weight'], changed
        rows = []; mismatches = []; first_residual = None
        for context in range(16):
            records = []
            for key in (left,right):
                arm = arms[key]; relative = f'baseline_evaluation/context{context}/result.json'
                proof = arm['audit']['artifacts'][relative]
                record = read(checked(arm['folder']/relative, proof['sha256'], proof['bytes']))
                assert record['valid'] and record['context'] == context and len(record['rows']) == 16
                raw = record['raw']; raw_key = f'baseline_evaluation/context{context}/raw.pt.gz'
                assert raw['file'] == 'raw.pt.gz'
                assert arm['audit']['artifacts'][raw_key] == dict(sha256=raw['sha256'],bytes=raw['bytes'])
                records.append(record)
            for heading in range(16):
                prefix = f'/first_free/{heading}/'
                a,b = [r['raw']['tensor_schema'] for r in records]
                same = {field:a[prefix+field] == b[prefix+field] for field in SAME}
                different = {field:a[prefix+field] != b[prefix+field] for field in OUTPUT}
                rows.append(dict(context=context,heading_index=heading,identical=same,different=different))
                if not all(same.values()): mismatches.append(dict(context=context,heading_index=heading,fields=[k for k,v in same.items() if not v]))
            # A mismatch triggers only one context-pair decompression per population.
            if mismatches and first_residual is None:
                raw_values = []
                for key,record in zip((left,right),records):
                    p = arms[key]['folder']/f'baseline_evaluation/context{context}/raw.pt.gz'; proof=record['raw']
                    with gzip.open(checked(p,proof['sha256'],proof['bytes']),'rb') as stream:
                        raw_values.append(torch.load(stream,map_location='cpu',weights_only=True)['first_free'])
                first_residual = dict(context=context, max_abs_by_field={field:max(float((raw_values[0][i][field].double()-raw_values[1][i][field].double()).abs().max()) for i in range(16)) for field in SAME})
        models[condition] = dict(changed_parameter_names=changed,identical_other_parameter_tensors=len(policies[left])-2,
            cases=256,identical_counts={f:sum(r['identical'][f] for r in rows) for f in SAME},
            different_counts={f:sum(r['different'][f] for r in rows) for f in OUTPUT},
            baseline_successes={m:arms[(condition,m)]['totals']['successes'] for m in ('retained_actor','reset_actor')},
            all_first_choice_nonactor_fields_identical=not mismatches,mismatches=mismatches,
            first_mismatch_context_numeric_residual=first_residual,rows=rows)
    return dict(version='saved_feedback_baseline_actor_isolation_v1',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        complete=True,source_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,input_sha256=inputs,models=models,
        all_first_choice_nonactor_fields_identical=all(r['all_first_choice_nonactor_fields_identical'] for r in models.values()),
        timing='Saved first_free state is concatenated h,c AFTER the first FREE LSTM input and BEFORE sampling or applying its action (local_return_feedback.py collect).',
        scope='Within-population baseline comparison only. Exact incoming tensor bytes are independently compared; saved first_free tensor schema hashes are authenticated through immutable passing baseline audits and result-file hashes. No new rollout, training, recurrent replay, or cross-population state comparison.',
        interpretation='At the first free choice, any verified identical input/h,c/physical state with different logits isolates the changed action readout. Subsequent actions can produce different physical observations and recurrent trajectories; their states are not asserted identical.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/local_return_feedback_design/portable/cloud_audit/baseline_actor_isolation.json')
    args=p.parse_args(); result=run(); args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({c:{k:v for k,v in r.items() if k not in ('rows','mismatches')} for c,r in result['models'].items()}))


if __name__=='__main__': main()
