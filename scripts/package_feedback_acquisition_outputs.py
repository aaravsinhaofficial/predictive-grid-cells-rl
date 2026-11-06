"""Package terminal feedback-acquisition evidence; never train, collect or publish.

Three deterministic, independently manifested assets stay below 2 GiB each.
The exact 49-file portable input asset is a separate required prerequisite.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'f1e5e1725c4449ee6c8bd39cac022b9e94b0139f'
PROTOCOL = 'docs/research/local_return_feedback_portable_protocol.json'
PROTOCOL_SHA = '4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
OLD_SHA = 'ad969e782ec2943de2e59b2d72629002dea537f7c9c0fcf2ad3905ce051038c7'
ROUTING = 'docs/research/local_return_feedback_portable_launch.json'
OLD_ROUTING = 'docs/research/local_return_feedback_launch.json'
DESIGN = 'results/local_return_feedback_design'
AUDIT = DESIGN+'/portable/cloud_audit'
OLD_AUDIT = DESIGN+'/cloud_audit'
REPORT = 'results/local_return_feedback_acquisition/portable/report'
REFERENCES = 'results/local_return_feedback_acquisition/references'
INPUT = DESIGN+'/portable/cloud_package/feedback_acquisition_portable_inputs.tar.gz'
INPUT_SHA = '797981f1e1792588ee0facf3d82125dc508e3d607d02f6aaf7f0a857514c77ae'
INPUT_BYTES = 96310897
ARMS = {(c, a) for c in ('gc100', 'pgc33') for a in ('retained_actor', 'reset_actor')}
LIMIT = 2*1024**3
COMPLETE = 'complete_collected_audited_deleted'
FAILED = {'failed_collected_deleted', 'launch_failed', 'cancelled_before_reservation_or_launch'}
CHECKPOINTS = ('incoming.pt', 'baseline_before_evaluation.pt', 'baseline.pt',
               'latest.pt', 'final_before_evaluation.pt', 'final.pt')
REPORT_FILES = ('summary.json', 'README.md', 'all_arms.csv', 'feedback_return.png',
                'feedback_return.pdf', 'feedback_learning.png', 'feedback_learning.pdf')
TOOLS = ('scripts/package_feedback_acquisition_outputs.py',
         'tests/test_package_feedback_acquisition_outputs.py',
         'scripts/compare_feedback_training_streams.py', 'tests/test_compare_feedback_training_streams.py',
         'scripts/plot_local_return_feedback.py', 'tests/test_plot_local_return_feedback.py',
         'scripts/monitor_local_return_feedback.py', 'scripts/monitor_local_return_feedback_portable.py',
         'scripts/prepare_local_return_feedback_cloud.py', 'scripts/prepare_local_return_feedback_portable_cloud.py',
         'scripts/finalize_feedback_acquisition_cloud.py',
         'scripts/check_feedback_baseline_actor_isolation.py',
         'scripts/check_feedback_completed_finals.py',
         'scripts/analyze_feedback_restricted_actions.py', 'scripts/write_feedback_restricted_actions.py',
         'tests/test_feedback_restricted_actions.py', 'tests/test_write_feedback_restricted_actions.py')
SECRET_NAME = re.compile(r'(^\.env($|\.)|credential|client_secret|service.account|id_rsa|\.pem$|\.key$)', re.I)
SECRET_TEXT = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
    r'"(?:private_key|client_secret|refresh_token|access_token)"\s*:\s*"[^"\s]+', re.I)
ALLOWED = {'.pt', '.gz', '.json', '.jsonl', '.log', '.txt', '.md', '.py', '.png', '.pdf', '.csv'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def checked_path(root, value):
    relative = Path(value); root = Path(root).resolve()
    assert str(value) not in ('', '.') and relative.parts and not relative.is_absolute() and '..' not in relative.parts
    result = root
    for part in relative.parts:
        result /= part
        assert not result.is_symlink(), f'Symlink is not archived: {result}'
    assert result.resolve().is_relative_to(root)
    return result


def inspect_file(path):
    assert path.is_file() and not path.is_symlink(), path
    assert path.suffix in ALLOWED and not SECRET_NAME.search(path.name), path
    assert not path.name.endswith('.tmp'), f'Incomplete writer: {path}'
    if path.suffix in {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.csv'}:
        assert not SECRET_TEXT.search(path.read_text()), f'Potential credential material: {path}'


def tree(root, relative):
    folder = checked_path(root, relative); assert folder.is_dir(), folder
    paths = []
    for current, dirs, files in os.walk(folder, followlinks=False):
        assert all(not (Path(current)/d).is_symlink() for d in dirs), folder
        for name in files:
            path = Path(current)/name; inspect_file(path); paths.append(path)
    assert paths, f'Empty evidence tree: {folder}'
    return sorted(paths)


def file_record(root, path):
    path = Path(path); inspect_file(path)
    return dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)


def verify_record(root, item, key='path'):
    path = checked_path(root, item[key]); inspect_file(path)
    assert sha(path) == item['sha256'], path
    if 'bytes' in item: assert path.stat().st_size == item['bytes'], path
    return path


def routes(routing, protocol_hash, terminal=True):
    assert routing['protocol_sha256'] == protocol_hash
    rows = routing['workers']
    assert len(rows) == 4 and {(r['condition'], r['actor_mode']) for r in rows} == ARMS
    assert len({r['name'] for r in rows}) == 4
    for r in rows:
        assert re.fullmatch(r'ring-20261105-feedback-(grid|pgc)-(retain|reset)-[0-9]+', r['name'])
        assert r['local_artifact_root'] == 'results/ring_cloud_artifacts/'+r['name']
        assert r['artifact_subdir'] == 'experiment'
        if terminal: assert r['status'] == COMPLETE or r['status'] in FAILED, 'Every worker must be terminal'
    return rows


def verify_cleanup(root, routing, relative, protocol_hash):
    proof = read(checked_path(root, relative)); rows = routes(routing, protocol_hash)
    assert proof['complete'] is True and proof['protocol_sha256'] == protocol_hash and proof['verified_utc']
    expected = {r['name']:r for r in rows}
    assert len(proof['workers']) == 4 and {r['name'] for r in proof['workers']} == set(expected)
    for item in proof.get('global_evidence', []):
        path = verify_record(root, item)
        assert read(path) == [], 'Global absence proof must be an empty resource list'
    for key in ('prior_failed_attempt_cleanup', 'budget'):
        if key in proof: verify_record(root, proof[key])
    for r in proof['workers']:
        route = expected[r['name']]
        assert all(r[k] == route[k] for k in ('condition', 'actor_mode', 'zone'))
        assert r['instance_absent'] is True and r['disk_absent'] is True
        kinds = set()
        for item in r['evidence']:
            path = verify_record(root, item)
            for kind in ('instances', 'disks'):
                if path.name == f"{r['name']}_{kind}_absence.log":
                    text = path.read_text()
                    assert r['name'] in text and kind in text and ('was not found' in text or 'notFound' in text)
                    kinds.add(kind)
        if r.get('created', True): assert kinds == {'instances', 'disks'}
        else:
            assert route['status'] in {'launch_failed', 'cancelled_before_reservation_or_launch'}
            assert len(proof.get('global_evidence', [])) >= 2
    return proof


def verify_audit(root, worker, terminal_only=False):
    path = checked_path(root, worker['audit_path']); report = read(path)
    assert report['protocol_sha256'] == PROTOCOL_SHA
    assert (report['condition'], report['actor_mode']) == (worker['condition'], worker['actor_mode'])
    assert report['auditor_sha256'] == sha(root/'scripts/audit_local_return_feedback.py')
    relative = Path(worker['local_artifact_root'])/worker['artifact_subdir']; folder = checked_path(root, relative)
    assert Path(report['folder']).parts[-len(relative.parts):] == relative.parts
    for name, record in report['artifacts'].items(): verify_record(folder, dict(path=name, **record))
    complete = worker['status'] == COMPLETE
    if not complete:
        assert terminal_only, 'An incomplete arm cannot be labeled a completed study'
        assert report['status'] in ('pending', 'failed') and not report['complete']
        assert report['failures'] or report['pending']
        return report
    assert report['passed'] and report['complete'] and report['status'] == 'complete'
    assert not report['failures'] and not report['pending']
    assert sha(path) == worker['audit_sha256']
    required = set(CHECKPOINTS)|{'protocol.json', 'training_config.json', 'initialization.json', 'metrics.jsonl', 'summary.json'}
    for endpoint in ('baseline', 'final'):
        required.update({f'{endpoint}_evaluation/summary.json', f'{endpoint}_evaluation_integrity.json'})
        evaluation = report['checks']['evaluations'][endpoint]
        assert evaluation['complete'] and evaluation['totals']['trials'] == 256
        contexts = evaluation['contexts']; assert len(contexts) == 16 and {r['context'] for r in contexts} == set(range(16))
        for context in contexts:
            prefix = f"{endpoint}_evaluation/context{context['context']}/"
            required.update(prefix+s for s in ('input.json', 'result.json'))
            required.add(prefix+('raw.pt.gz' if context['valid'] else 'failed_plan.pt'))
    assert required <= set(report['artifacts']), 'Missing fixed endpoint/checkpoint/raw evidence'
    summary = read(folder/'summary.json')
    assert summary['completed'] and summary['protocol_sha256'] == PROTOCOL_SHA
    assert (summary['counters']['updates'], summary['counters']['trials']) == (1024, 32768)
    assert set(summary['counters']['evaluation_records']) == {'baseline', 'final'}
    assert sha(folder/'final.pt') == summary['final_checkpoint_sha256'] == worker['final_checkpoint_sha256']
    return report


def verify_private_streams(root, routing):
    """Authenticate and compare saved data; instantiate no policy or RNG."""
    import torch
    path = root/AUDIT/'paired_final_training_validation.json'; report = read(path)
    assert report['complete'] and report['passed'] and report['protocol_sha256'] == PROTOCOL_SHA
    assert report['routing_sha256'] == sha(root/ROUTING)
    assert report['producer_sha256'] == sha(root/'scripts/compare_feedback_training_streams.py')
    assert report['checks'] == {k:True for k in ('plan', 'motor', 'action', 'state')}
    raw_path = verify_record(root, report['raw'])
    assert report['raw']['path'] == AUDIT+'/paired_final_private_streams.pt'
    snapshots = torch.load(raw_path, map_location='cpu', weights_only=True)
    workers = {r['name']:r for r in routing['workers']}
    assert set(snapshots) == set(workers) and len(report['inputs']) == 4
    assert {r['name'] for r in report['inputs']} == set(workers)
    first = next(iter(snapshots.values()))
    for item in report['inputs']:
        worker = workers[item['name']]
        assert item['checkpoint'] == worker['local_artifact_root']+'/experiment/final.pt'
        assert item['audit'] == worker['audit_path']
        checkpoint = checked_path(root, item['checkpoint']); audit = checked_path(root, item['audit'])
        assert sha(checkpoint) == item['checkpoint_sha256'] and sha(audit) == item['audit_sha256']
        actual = torch.load(checkpoint, map_location='cpu', weights_only=True)
        saved = snapshots[item['name']]
        for state in (actual['streams'], saved):
            assert set(state) == {'seed', 'batches', 'contexts', 'generators'}
            assert (state['seed'], state['batches'], state['contexts']) == (494801, 1024, 32768)
            assert set(state['generators']) == {'plan', 'motor', 'action', 'state'}
        for name, tensor in saved['generators'].items():
            assert tensor.dtype == torch.uint8 and tensor.ndim == 1
            assert torch.equal(tensor, actual['streams']['generators'][name]), 'Raw private stream differs from checkpoint'
            assert torch.equal(tensor, first['generators'][name]), 'Final private streams differ across arms'
        assert actual['counters'] == item['counters'] and actual['active_physics_frames'] == item['active_physics_frames']
        del actual
    return dict(path=str(path.relative_to(root)), sha256=sha(path), raw=report['raw'], scope=report['scope'])


def verify_baselines(root, routing):
    report = read(root/AUDIT/'all_baselines_independent_review.json')
    assert report['passed'] and report['complete_all_four_baselines'] and report['protocol_sha256'] == PROTOCOL_SHA
    assert report['total_prescribed_baseline_trials'] == 1024 and not report['baseline_score_used_for_run_selection']
    assert len(report['rows']) == 4
    workers = {r['name']:r for r in routing['workers']}
    assert {r['worker'] for r in report['rows']} == set(workers)
    for row in report['rows']:
        audit = read(verify_record(root, row['baseline_audit']))
        assert audit['passed'] and audit['protocol_sha256'] == PROTOCOL_SHA
        assert audit['checks']['evaluations']['baseline']['totals'] == row['baseline_totals']
        prefix = verify_record(root, row['exact_metrics_prefix'])
        worker = workers[row['worker']]; folder = root/worker['local_artifact_root']/'experiment'
        # Historical mutable audit entries retain upload-time scope. Check their
        # preserved metric prefix and immutable baseline files, not later latest.pt.
        assert (folder/'metrics.jsonl').read_bytes().startswith(prefix.read_bytes())
        for name, record in audit['artifacts'].items():
            if name.startswith('baseline') or name in ('incoming.pt', 'initialization.json', 'training_config.json'):
                verify_record(folder, dict(path=name, **record))
        verify_record(root, row['prelude'])
    first = read(root/AUDIT/'first_baseline_evidence/manifest.json')
    for item in first['files']: verify_record(root, item, key='preserved_path')
    return dict(all_four_immutable_baselines=True, historical_mutable_files='Preserved historical hashes, not required to equal later uploads.')


def verify_references(root):
    audit = read(root/DESIGN/'reference_audit.json')
    assert audit['passed'] and audit['complete'] and not audit['failures'] and audit['protocol_sha256'] == OLD_SHA
    assert audit['auditor_sha256'] == sha(root/'scripts/audit_local_return_feedback.py')
    folder = root/REFERENCES
    for name, record in audit['artifacts'].items(): verify_record(folder, dict(path=name, **record))
    top = read(folder/'summary.json'); assert top['complete'] and top['no_policy_or_optimization'] and top['reference_trials'] == 512
    for controller in ('uniform', 'planner'):
        assert audit['checks'][controller]['totals']['trials'] == 256
        for i in range(16):
            assert all(f'{controller}/context{i}/{s}' in audit['artifacts'] for s in ('input.json', 'result.json', 'raw.pt.gz'))
    return audit


def verify_input_archive(root, routing):
    assert routing['source_commit'] == SOURCE_COMMIT
    assert routing['canonical_archive'] == dict(path=INPUT, sha256=INPUT_SHA, bytes=INPUT_BYTES)
    path = verify_record(root, routing['canonical_archive'])
    rows = routing['package_manifest']; assert len(rows) == 49 and len({r['path'] for r in rows}) == 49
    verify_archive(path, rows)
    for r in rows: verify_record(root, r)
    return dict(path=INPUT, sha256=INPUT_SHA, bytes=INPUT_BYTES, files=49, source_commit=SOURCE_COMMIT)


def inventory(root=ROOT, tooling_commit=None, terminal_only=False):
    root = Path(root).resolve(); assert tooling_commit and re.fullmatch(r'[0-9a-f]{40}', tooling_commit)
    assert sha(root/PROTOCOL) == PROTOCOL_SHA
    protocol = read(root/PROTOCOL); routing = read(root/ROUTING); old = read(root/OLD_ROUTING)
    workers = routes(routing, PROTOCOL_SHA); previous = routes(old, OLD_SHA)
    complete = all(w['status'] == COMPLETE for w in workers)
    assert complete or terminal_only, 'Completed-study packaging requires all four audited fixed finals'
    prerequisite = verify_input_archive(root, routing)
    cleanup = verify_cleanup(root, routing, AUDIT+'/cleanup_verification.json', PROTOCOL_SHA)
    assert cleanup['scientific_completed'] == complete
    assert routing['scientific_completed'] == complete
    assert routing['cleanup']['path'] == AUDIT+'/cleanup_verification.json'
    verify_record(root, routing['cleanup'])
    verify_cleanup(root, old, OLD_AUDIT+'/cleanup_verification.json', OLD_SHA)
    audits = {w['name']:verify_audit(root, w, terminal_only) for w in workers}
    baselines = verify_baselines(root, routing)
    streams = verify_private_streams(root, routing) if complete else dict(status='not_available_incomplete_fixed_endpoints', complete=False)
    verify_references(root)
    first = read(root/AUDIT/'paired_first_plan_validation.json'); assert first['passed'] and first['all_global_rng_preserved']
    for r in first['records']: verify_record(root, r)
    verify_record(root, first['producer'])
    terminal = read(root/OLD_AUDIT/'terminal_attempt_record.json')
    assert terminal['protocol_sha256'] == OLD_SHA and terminal['no_scientific_outcomes']
    assert terminal['policy_evaluation_trials'] == terminal['optimizer_updates'] == 0
    groups = {k:set() for k in ('grid', 'pgc', 'common')}
    for w in workers+previous:
        folder = checked_path(root, w['local_artifact_root'])
        if w['status'] == 'cancelled_before_reservation_or_launch':
            assert not folder.exists(), 'Unexpected output for a never-launched attempt'
            continue
        found = tree(root, w['local_artifact_root'])
        assert (folder/'worker.log').is_file() and (folder/'WORKER_RESULT.json').is_file()
        result = read(folder/'WORKER_RESULT.json')
        assert (result['exit_code'] == 0) == (w['status'] == COMPLETE)
        groups['grid' if w['condition'] == 'gc100' else 'pgc'].update(found)
    common = groups['common']
    for relative in (AUDIT, OLD_AUDIT, DESIGN+'/source_history', REFERENCES, REPORT,
                     DESIGN+'/portable/restricted_action_bound'): common.update(tree(root, relative))
    for parent in (root/'docs/research', root/DESIGN, root/DESIGN/'portable', root/DESIGN/'cloud_package', root/DESIGN/'portable/cloud_package'):
        for path in parent.iterdir():
            if path.is_file() and (parent != root/'docs/research' or path.name.startswith('local_return_')):
                if path.suffix in {'.json', '.md'}: inspect_file(path); common.add(path)
    for path in (ROUTING, OLD_ROUTING, PROTOCOL, DESIGN+'/PORTABLE_REPRODUCE.md', DESIGN+'/REPRODUCE.md', DESIGN+'/PORTABILITY.md',
                 'docs/research/ring_assay_positive_controls.md',
                 'docs/research/feedback_restricted_action_bound.md',
                 'docs/research/feedback_restricted_action_bound_protocol.json'):
        common.add(checked_path(root, path)); inspect_file(root/path)
    report = read(root/REPORT/'summary.json')
    assert not report['synthetic'] and report['protocol_sha256'] == PROTOCOL_SHA
    assert report['all_arms_complete'] == complete and len(report['arms']) == 4
    assert {(r['condition'],r['actor_mode']) for r in report['arms']} == ARMS
    assert report['references']['status'] == 'complete'
    for name in REPORT_FILES: inspect_file(root/REPORT/name)
    for value, digest in report['input_sha256'].items():
        path = Path(value); assert path.is_absolute() and path.resolve().is_relative_to(root)
        assert sha(path) == digest; common.add(path)
    sources = sorted(set(protocol['source_sha256'])|set(TOOLS))
    for name in sources:
        path = checked_path(root, name); inspect_file(path)
        expected_commit = SOURCE_COMMIT if name in protocol['source_sha256'] else tooling_commit
        committed = subprocess.check_output(['git', 'show', f'{expected_commit}:{name}'], cwd=root)
        assert hashlib.sha256(committed).hexdigest() == sha(path), f'Exact source commit mismatch: {name}'
        if name in protocol['source_sha256']: assert sha(path) == protocol['source_sha256'][name]
        common.add(path)
    # Per-arm data stay in their population asset even when referenced by a report.
    common.difference_update(groups['grid']|groups['pgc'])
    ledger = read(root/'docs/research/ring_cloud_budget.json'); names={w['name'] for w in workers+previous}
    selected = [r for r in ledger['runs'] if r['name'] in names]
    assert {r['name'] for r in selected} == {w['name'] for w in workers+previous if w['status'] != 'cancelled_before_reservation_or_launch'}
    budget = dict(source_ledger_sha256=sha(root/'docs/research/ring_cloud_budget.json'),
                  scope='Only the explicitly routed feedback attempts; unrelated resource rows omitted.', runs=selected)
    manifests = {}
    for group, paths in groups.items():
        rows = [file_record(root, p) for p in sorted(paths)]
        manifests[group] = dict(version='feedback_acquisition_outputs_v1', group=group,
            source_commit=SOURCE_COMMIT, tooling_commit=tooling_commit, protocol_sha256=PROTOCOL_SHA,
            complete_fixed_study=complete, all_workers_terminal_and_absent=True,
            mode='complete' if complete else 'terminal_evidence_only', required_input_asset=prerequisite,
            routing_sha256=sha(root/ROUTING), prior_routing_sha256=sha(root/OLD_ROUTING),
            files=rows, file_count=len(rows), uncompressed_bytes=sum(r['bytes'] for r in rows),
            all_worker_trees_retained=True, baseline_evidence=baselines, private_streams=streams,
            budget_snapshot=budget if group=='common' else None,
            source_sha256={p:sha(root/p) for p in sources},
            extraction_reproduction='Not performed by packager; requires subsequent isolated verification.')
    return manifests


def verify_archive(path, rows):
    expected = {r['path']:r for r in rows}; assert len(expected) == len(rows)
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers(); assert len(members) == len(rows) and {m.name for m in members} == set(expected)
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            row = expected[member.name]; assert member.size == row['bytes']
            h = hashlib.sha256()
            with archive.extractfile(member) as f:
                for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
            assert h.hexdigest() == row['sha256'], member.name


def write_archive(root, destination, manifest, max_bytes=LIMIT):
    """Synthetic inventories supported; production callers must use inventory()."""
    root=Path(root).resolve(); destination=Path(destination)
    assert not destination.exists(), 'Never overwrite an existing asset'
    rows=manifest['files']; assert len({r['path'] for r in rows}) == len(rows)
    name=f"results/local_return_feedback_acquisition/portable/{manifest['group']}_outputs_manifest.json"
    payload=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    embedded=dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    assert name not in {r['path'] for r in rows}
    temporary=destination.with_name(destination.name+'.tmp'); assert not temporary.exists()
    try:
        with temporary.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as archive:
                for row in sorted(rows+[embedded],key=lambda r:r['path']):
                    info=tarfile.TarInfo(row['path']); info.size=row['bytes']; info.mode=0o644; info.mtime=0
                    if row is embedded: archive.addfile(info,io.BytesIO(payload))
                    else:
                        path=verify_record(root,row)
                        with path.open('rb') as f: archive.addfile(info,f)
                    assert raw.tell() <= max_bytes, 'GitHub asset size limit exceeded'
        assert temporary.stat().st_size <= max_bytes, 'GitHub asset size limit exceeded'
        verify_archive(temporary,rows+[embedded])
        for row in rows: verify_record(root,row)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True); raise
    return dict(path=str(destination),name=destination.name,sha256=sha(destination),bytes=destination.stat().st_size,
                members=len(rows)+1,manifest=embedded,all_member_bytes_verified=True)


def synthetic_preflight():
    with tempfile.TemporaryDirectory(prefix='feedback-package-synthetic-') as temporary:
        root=Path(temporary); (root/'synthetic.json').write_text('{"synthetic":true}\n')
        manifest=dict(group='common',synthetic=True,files=[file_record(root,root/'synthetic.json')])
        first=write_archive(root,root/'one.tar.gz',manifest); second=write_archive(root,root/'two.tar.gz',manifest)
        assert first['sha256']==second['sha256']
        return dict(synthetic=True,passed=True,deterministic_archive=True,member_bytes_verified=True,no_study_inputs_read=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tooling-commit'); parser.add_argument('--terminal-evidence-only',action='store_true')
    parser.add_argument('--synthetic-preflight',action='store_true')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/ring_public_assets')
    parser.add_argument('--record',type=Path,default=ROOT/'docs/research/feedback_acquisition_outputs_archive.json')
    args=parser.parse_args()
    if args.synthetic_preflight: print(json.dumps(synthetic_preflight())); return
    manifests=inventory(tooling_commit=args.tooling_commit,terminal_only=args.terminal_evidence_only)
    assert not args.record.exists(), 'Preserve an earlier archive record'
    paths={g:args.out_dir/f'feedback-acquisition-{g}-outputs-v1.tar.gz' for g in manifests}
    assert all(not p.exists() for p in paths.values())
    args.out_dir.mkdir(parents=True,exist_ok=True)
    assets={g:write_archive(ROOT,paths[g],m) for g,m in manifests.items()}
    assert inventory(tooling_commit=args.tooling_commit,terminal_only=args.terminal_evidence_only)==manifests, 'Inputs changed while packaging'
    record=dict(version='feedback_acquisition_output_assets_v1',source_commit=SOURCE_COMMIT,
        tooling_commit=args.tooling_commit,protocol_sha256=PROTOCOL_SHA,assets=assets,
        complete_fixed_study=manifests['common']['complete_fixed_study'],
        required_input_asset=manifests['common']['required_input_asset'],
        generator_sha256=sha(__file__),owner_cleanup_verified=True,all_archived_member_bytes_verified=True,
        isolated_extraction_reproduction_verified=False,published=False)
    args.record.parent.mkdir(parents=True,exist_ok=True); args.record.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__': main()
