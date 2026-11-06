"""Inventory and deterministic release packaging; never run scientific models.

Default/--plan lists missing completion metadata without building archives.
--inventory-only and --build require a full, matching reproduction commit and
all terminal results, independently of whether any scientific test is positive.
The source trees are included completely to preserve local import dependencies.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

from package_goal_cue_memory_outputs import checked, tree_files, sha, partition
from package_goal_cue_memory_outputs import digest, json_bytes, read, prove, commit_sha

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'results/nonlinear_ring_release'
MANIFEST_NAME = f'{RELEASE}/manifest.json'
MAX_ASSET_BYTES = 1_500_000_000
PAYLOAD_BYTES = 1_450_000_000
MODELS = tuple((c, s) for c in ('gc100', 'pgc33') for s in (1, 2))
TREES = tuple('results/' + name for name in (
    'ring_nonlinear_design', 'lstm_module_phase_ring',
    'lstm_module_phase_observable', 'lstm_module_phase_normal_audit',
    'lstm_module_phase_torus', 'lstm_module_all_phase_control', 'native_movement_cycles',
    'native_movement_pulses', 'native_movement_bounded_pulses'))
DOCS = (
    'README.md', 'mechanisms/learned-ring-in-navigation.md',
    'requirements-ring-analysis.txt',
    'docs/research/nonlinear_ring_search_protocol.md',
    'docs/research/alternative_ring_methods_literature_2026.md',
    'docs/research/ring_local_environment.json',
    *('docs/research/' + name for name in (
        'native_movement_cycles_protocol.json',
        'native_movement_cycles_views_addendum.json',
        'native_movement_cycles_source_manifest.json',
        'native_movement_pulse_protocol.json',
        'native_movement_pulse_source_manifest.json',
        'native_movement_bounded_pulse_protocol.json',
        'native_movement_bounded_pulse_source_manifest.json')),
    f'{RELEASE}/REPRODUCE.md')
NEW_SCRIPTS = tuple('scripts/' + name + '.py' for name in (
    'published_ssm_ring_benchmark', 'published_ssm_ring_report',
    'native_lstm_equilibrium_branches', 'lstm_module_phase_ring',
    'lstm_module_phase_ring_report', 'lstm_module_phase_observable',
    'lstm_module_phase_observable_report', 'lstm_module_phase_normal_audit',
    'lstm_module_phase_torus', 'lstm_module_phase_torus_report',
    'lstm_module_all_phase_control', 'lstm_module_all_phase_control_report',
    'native_movement_cycles', 'report_native_movement_cycles',
    'native_movement_pulses', 'report_native_movement_pulses',
    'native_movement_bounded_pulses', 'report_native_movement_bounded_pulses',
    'plot_nonlinear_ring_search_summary', 'package_nonlinear_ring_outputs',
    'package_goal_cue_memory_outputs'))
NEW_TESTS = tuple('tests/test_' + name + '.py' for name in (
    'published_ssm_ring_benchmark', 'native_lstm_equilibrium_branches',
    'lstm_module_phase_ring', 'lstm_module_phase_observable',
    'lstm_module_phase_normal_audit', 'lstm_module_phase_torus',
    'lstm_module_all_phase_control',
    'lstm_module_phase_torus_review', 'native_movement_cycles',
    'native_movement_pulses', 'report_native_movement_pulses',
    'native_movement_bounded_pulses', 'report_native_movement_bounded_pulses',
    'package_goal_cue_memory_outputs'))
INPUTS = tuple(
    [f'models/{c}/{c}_lstm_s{s}_1000M.{ext}'
     for c, s in MODELS for ext in ('pt', 'json')] +
    [f'results/original_goal_bearing_manifold/pilot/{c}_s{s}/context0_states.npz'
     for c, s in MODELS] +
    [f'results/lstm_endogenous_ring/{stage}/{c}_s{s}/context0_{regime}.npz'
     for stage in ('fixed_points', 'equilibrium_elimination')
     for c, s in MODELS for regime in ('goal_withheld', 'actual_goal_held')])
TEXT_EXTENSIONS = {'.py', '.sh', '.txt', '.md', '.json', '.log', '.toml', '.svg', '.ipynb'}


def source_files(root):
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', 'code', 'scripts', 'tests'], cwd=root).decode().split('\0')
    return set(filter(None, names)) | set(NEW_SCRIPTS) | set(NEW_TESTS)


def check_name_and_content(root, name):
    """An explicit scope plus conservative credential checks, without printing secrets."""
    path = checked(root, name)
    parts = PurePosixPath(name).parts
    assert not any(p.startswith('.') or p in ('__pycache__', 'node_modules') for p in parts), name
    assert path.suffix not in ('.pem', '.key', '.p12', '.pfx', '.pyc'), name
    assert not re.search(r'(^|[/_.-])(credentials?|secrets?|service.account)([/_.-]|$)', name, re.I), name
    if path.suffix in TEXT_EXTENSIONS:
        body = path.read_bytes()
        patterns = (rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                    rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{50,}',
                    rb'GOCSPX-[A-Za-z0-9_-]{20,}')
        assert not any(re.search(p, body) for p in patterns), f'Possible credential: {name}'
    return path


def completion(root):
    """Collect all missing/stale metadata rather than hiding later gaps at first failure."""
    issues, checks = [], []

    def check(label, action):
        try:
            action()
            checks.append(label)
        except (AssertionError, OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(check=label, reason=str(error) or type(error).__name__))

    def require(name):
        return checked(root, name)

    def document(name):
        return read(require(name))

    def equal(value, expected, label):
        assert value == expected, f'{label}: got {value!r}, expected {expected!r}'

    def source_mapping(name, key):
        for path, expected in document(name)[key].items():
            prove(root, path, dict(sha256=expected))

    for stem in ('native_movement_cycles', 'native_movement_pulse', 'native_movement_bounded_pulse'):
        name = f'docs/research/{stem}_source_manifest.json'
        check(name, lambda n=name: source_mapping(n, 'sha256'))
    for name in ('lstm_module_phase_ring', 'lstm_module_phase_observable',
                 'lstm_module_phase_normal_audit', 'lstm_module_phase_torus',
                 'lstm_module_all_phase_control'):
        path = f'results/{name}/protocol.json'
        check(path, lambda n=path: source_mapping(n, 'source_sha256'))

    def all_phase_parent():
        value = document('results/lstm_module_all_phase_control/protocol.json')
        equal(value['cases'], 768, 'all-phase frozen cohort')
        for key, name in (('parent_protocol_sha256', 'protocol.json'),
                          ('parent_summary_sha256', 'summary.json')):
            prove(root, f'results/lstm_module_phase_torus/{name}', dict(sha256=value[key]))
        equal(set(value['parent_archives']), {f'{c}_s{s}' for c, s in MODELS}, 'all-phase parent models')
        for tag, expected in value['parent_archives'].items():
            prove(root, f'results/lstm_module_phase_torus/{tag}/states_and_tangents.npz', dict(sha256=expected))
    check('local all-phase frozen parent inputs', all_phase_parent)

    def vendor():
        base = 'results/ring_nonlinear_design/vendor/marraffa2026'
        metadata = document(f'{base}/SOURCE_MANIFEST.json')
        equal(metadata['license'], 'MIT', 'vendor license')
        require(f'{base}/LICENSE')
        for row in metadata['files']:
            prove(root, f'{base}/{row["path"]}', row)
    check('MIT vendor inventory', vendor)

    for suffix in ('published_benchmark', 'published_benchmark_1000s'):
        def benchmark(suffix=suffix):
            base = f'results/ring_nonlinear_design/{suffix}'
            document(f'{base}/summary.json')
            require(f'{base}/ode_trajectories.npz'); require(f'{base}/map_trajectories.npz')
            hashes = document(f'{base}/source_hashes.json')
            for path, expected in hashes.items():
                # Preserve the exact historical source, even when current source has evolved.
                target = f'{base}/executed_source.py' if path == 'scripts/published_ssm_ring_benchmark.py' else path
                prove(root, target, dict(sha256=expected))
            if suffix == 'published_benchmark':
                recovered = document(f'{base}/source_reconstruction.json')
                equal(recovered['expected_sha256'], hashes['scripts/published_ssm_ring_benchmark.py'], base)
                equal(recovered['reconstructed_sha256'], recovered['expected_sha256'], base)
        check(suffix + ' complete with exact executed source', benchmark)
    for name in ('protocol.json', 'executed_source.py', 'corrected_cubic_validation.json'):
        path = f'results/ring_nonlinear_design/published_benchmark_corrected_rom/{name}'
        check(path, lambda n=path: require(n))

    for c, s in MODELS:
        tag = f'{c}_s{s}'

        def cycle(c=c, s=s, tag=tag):
            base = f'results/native_movement_cycles/{tag}'
            value = document(f'{base}/summary.json')
            assert value['complete'] and value['policy_unchanged']
            equal((value['condition'], value['seed']), (c, s), base)
            equal(value['episode_summary']['episodes'], 32, base)
            prove(root, f'{base}/native_trace.pt', dict(sha256=value['trace_sha256'], bytes=value['trace_bytes']))
            prove(root, f'{base}/census.json', dict(sha256=value['census_sha256']))
        check('native census ' + tag, cycle)

        for study in ('native_movement_pulses', 'native_movement_bounded_pulses'):
            def pulses(c=c, s=s, tag=tag, study=study):
                base = f'results/{study}/{tag}'
                value = document(f'{base}/summary.json')
                assert value['complete'] and value['policy_unchanged']
                equal((value['condition'], value['seed']), (c, s), base)
                equal(value['episodes'], 32, base)
                equal(value['eligible'] + len(value['missing']), 32, base)
                equal(len(value['raw_assets']), value['eligible'], base)
                equal(len(value['fork_checks']), value['eligible'], base + ' fork checks')
                stem = 'native_movement_bounded_pulse' if study.endswith('bounded_pulses') else 'native_movement_pulse'
                prove(root, f'docs/research/{stem}_protocol.json', dict(sha256=value['protocol_sha256']))
                prove(root, f'docs/research/{stem}_source_manifest.json', dict(sha256=value['manifest_sha256']))
                selected = document(f'{base}/selection.json')['selected']
                slots = [r['slot'] for r in selected] + [r['slot'] for r in value['missing']]
                equal(sorted(slots), list(range(32)), base + ' selection coverage')
                prechecks = document(f'{base}/native_only_prechecks.json')
                assert prechecks['complete'] and prechecks['before_any_counterfactual']
                equal(len(prechecks['cases']), value['eligible'], base + ' prechecks')
                for path, record in value['raw_assets'].items():
                    prove(root, f'{base}/{path}', record)
                equal({p.name for p in (Path(root)/base).glob('slot*.pt')}, set(value['raw_assets']), base + ' raw inventory')
                if study.endswith('bounded_pulses'):
                    old = f'results/native_movement_pulses/{tag}/summary.json'
                    prove(root, old, dict(sha256=value['original_summary_sha256']))
                    equal(value['eligible'], document(old)['eligible'], base + ' original cases')
            check(study + ' ' + tag, pulses)

        for regime in ('goal_withheld', 'actual_goal_held'):
            def roots(c=c, s=s, tag=tag, regime=regime):
                base = f'results/ring_nonlinear_design/native_equilibria/{tag}/context0_{regime}'
                value = document(f'{base}/summary.json')
                equal(value['starts'], 64, base)
                equal(len(document(f'{base}/outcomes.json')), 64, base)
                for name in ('inputs.npz', 'all_starts.npz', 'roots.npz'):
                    require(f'{base}/{name}')
                for key in ('checkpoint', 'initializer', 'prior_solution'):
                    prove(root, value[key], dict(sha256=value[key + '_sha256']))
                prove(root, 'scripts/native_lstm_equilibrium_branches.py', dict(sha256=value['source_sha256']))
            check('equilibrium ' + tag + ' ' + regime, roots)

        for study, terminal, raw, count_key, count in (
            ('lstm_module_phase_ring', f'{tag}/result.json', f'{tag}/states.npz', 'families', 16),
            ('lstm_module_phase_observable', f'{tag}.json', None, 'rows', 192),
            ('lstm_module_phase_torus', f'{tag}/result.json', f'{tag}/states_and_tangents.npz', 'records', 192),
            ('lstm_module_all_phase_control', f'{tag}/result.json', f'{tag}/states_and_phase_derivatives.npz', 'records', 192)):
            def phase(c=c, s=s, study=study, terminal=terminal, raw=raw, count_key=count_key, count=count):
                value = document(f'results/{study}/{terminal}')
                equal((value['condition'], value['seed']), (c, s), terminal)
                equal(len(value[count_key]), count, terminal)
                if raw: require(f'results/{study}/{raw}')
                prove(root, f'models/{c}/{c}_lstm_s{s}_1000M.pt', dict(sha256=value['checkpoint_sha256']))
                if study == 'lstm_module_phase_observable':
                    prove(root, f'results/lstm_module_phase_ring/{c}_s{s}/states.npz', dict(sha256=value['archive_sha256']))
                else:
                    history = f'results/original_goal_bearing_manifold/pilot/{c}_s{s}/context0_states.npz'
                    prove(root, history, dict(sha256=value['history_sha256']))
                if study == 'lstm_module_all_phase_control':
                    parent = f'results/lstm_module_phase_torus/{c}_s{s}/states_and_tangents.npz'
                    prove(root, parent, dict(sha256=value['parent_archive_sha256']))
            check(study + ' ' + tag, phase)

    for study in ('native_movement_pulses', 'native_movement_bounded_pulses'):
        def execution(study=study):
            base = f'results/{study}'
            value = document(f'{base}/execution.json')
            assert value['complete']
            equal(sorted((r['condition'], r['seed']) for r in value['models']), sorted(MODELS), base)
            for row in value['models']:
                equal(row['exit_code'], 0, base)
                expected = f'{base}/{row["condition"]}_s{row["seed"]}/summary.json'
                equal(row['summary_path'], expected, base)
                for key in ('summary', 'log'):
                    prove(root, row[key + '_path'], dict(sha256=row[key + '_sha256']))
                pre = f'{base}/{row["condition"]}_s{row["seed"]}/native_only_prechecks.json'
                prove(root, pre, dict(sha256=row['native_only_prechecks_sha256']))
            validation = document(f'{base}/report/validation.json')
            assert validation['complete']
            summary_key, source_key = ('summary_sha256', 'source_sha256') if study.endswith('bounded_pulses') else ('report_summary_sha256', 'report_source_sha256')
            prove(root, f'{base}/report/summary.json', dict(sha256=validation[summary_key]))
            prove(root, f'scripts/report_{study}.py', dict(sha256=validation[source_key]))
            for path, record in validation['figures'].items():
                prove(root, f'{base}/report/{path}', record)
            document(f'{base}/report/summary.json'); require(f'{base}/report/README.md')
            assert list((Path(root)/base/'report').glob('*.png')), base + ' figure missing'
        check(study + ' terminal execution and report', execution)

    for study, key, count in (('lstm_module_phase_ring', 'total_family_regime_anchors', 192),
                              ('lstm_module_phase_observable', 'cases', 768),
                              ('lstm_module_phase_torus', 'cases', 768),
                              ('lstm_module_all_phase_control', 'cases', 768)):
        def aggregate(study=study, key=key, count=count):
            value = document(f'results/{study}/summary.json')
            equal(value[key], count, study)
            equal(sorted((r['condition'], r['seed']) for r in value['results']), sorted(MODELS), study)
            for row in value['results']:
                tag = f'{row["condition"]}_s{row["seed"]}'
                path = f'{tag}.json' if study == 'lstm_module_phase_observable' else f'{tag}/result.json'
                equal(row, document(f'results/{study}/{path}'), study + ' aggregate/model agreement')
        check(study + ' complete cohort', aggregate)

    def normals():
        base = 'results/lstm_module_phase_normal_audit'
        protocol = document(f'{base}/protocol.json')
        prove(root, 'results/lstm_module_phase_observable/summary.json', dict(sha256=protocol['parent_sha256']))
        value = document(f'{base}/summary.json')
        equal(value['case_count'], len(protocol['candidate_ids']), base)
        equal(len(value['results']), value['case_count'], base)
        for case in protocol['candidate_ids']:
            document(f'{base}/{case}/result.json'); require(f'{base}/{case}/all_perturbations.npz')
    check('all nominated normal-audit cases retained', normals)
    for name in ('results/ring_nonlinear_design/README.md',
                 'results/lstm_module_phase_torus/README.md',
                 'results/lstm_module_phase_torus/module_phase_torus_screen.png',
                 'results/lstm_module_phase_torus/module_phase_torus_screen.pdf',
                 'results/lstm_module_all_phase_control/README.md',
                 'results/lstm_module_all_phase_control/all_phase_memory_control.png',
                 'results/lstm_module_all_phase_control/all_phase_memory_control.pdf'):
        check(name, lambda n=name: require(n))
    def final_tests():
        value = document('results/ring_nonlinear_design/final_validation.json')
        equal(value['exit_code'], 0, 'final test execution')
        equal(value['failed'], 0, 'final test failures')
        assert value['passed'] >= 36, 'Final completed-study validation is incomplete'
    check('final completed-study numerical validation', final_tests)
    return dict(complete=not issues, passed_checks=checks, issues=issues,
                scientific_success_required=False, scientific_experiments_rerun=False)


def git_candidates(root):
    """Exact proposed source/report paths; large data and unfinished work stay out."""
    names = set(NEW_SCRIPTS) | set(NEW_TESTS) | set(DOCS)
    report_dirs = set(TREES) | {
        'results/ring_nonlinear_design/' + sub for sub in (
            'published_benchmark', 'published_benchmark_1000s',
            'published_benchmark_corrected_rom', 'native_equilibria')}
    report_dirs |= {'results/native_movement_cycles/fork_reference_audit',
                    'results/native_movement_pulses/report',
                    'results/native_movement_bounded_pulses/report'}
    for directory in sorted(report_dirs):
        for path in (Path(root)/directory).glob('*'):
            if path.is_file() and (path.suffix in ('.md', '.py', '.png', '.pdf', '.svg') or
                                   path.suffix == '.json' and path.stat().st_size <= 250_000):
                names.add(path.relative_to(root).as_posix())
    vendor = 'results/ring_nonlinear_design/vendor/marraffa2026'
    names |= {f'{vendor}/{name}' for name in ('LICENSE', 'SOURCE_MANIFEST.json', 'ARRAY_INSPECTION.json')}
    assert not any('distill' in name for name in names)
    return sorted(name for name in names if (Path(root)/name).is_file())


def inventory(root, source_commit=None, *, strict=False):
    root = Path(root).resolve()
    status = completion(root)
    sources = source_files(root)
    names = sources | set(DOCS) | set(INPUTS)
    for tree in TREES:
        try:
            names.update(tree_files(root, tree))
        except (AssertionError, OSError) as error:
            status['issues'].append(dict(check=tree, reason=str(error)))
    entries = []
    for name in sorted(names):
        try:
            path = check_name_and_content(root, name)
            entries.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
        except (AssertionError, OSError) as error:
            status['issues'].append(dict(check=name, reason=str(error)))
    if not source_commit:
        status['issues'].append(dict(check='source_commit', reason='Final full reproduction commit has not been supplied.'))
    else:
        assert re.fullmatch('[0-9a-f]{40}', source_commit), 'Use a full commit SHA'
        # Raw results and model bytes are separate release assets, not git requirements.
        for name in sorted(sources | {d for d in DOCS if not d.startswith('results/')}):
            try:
                assert sha(checked(root, name)) == commit_sha(root, source_commit, name), name
            except (AssertionError, OSError, subprocess.CalledProcessError) as error:
                status['issues'].append(dict(check='source_commit:' + name, reason=str(error)))
    proposed_parts = []
    try:
        proposed_parts = [dict(part=i, files=len(rows), uncompressed_bytes=sum(r['bytes'] for r in rows))
                          for i, rows in enumerate(partition(entries, PAYLOAD_BYTES), 1)]
    except AssertionError as error:
        status['issues'].append(dict(check='multipart payload bounds', reason=str(error)))
    status['complete'] = not status['issues']
    manifest = dict(version='nonlinear_ring_outputs_v1', source_commit=source_commit,
        completion=status, scientific_success_required_for_packaging=False,
        result_trees=list(TREES), input_assets=list(INPUTS), source_files=sorted(sources),
        max_asset_bytes=MAX_ASSET_BYTES, payload_limit_bytes=PAYLOAD_BYTES,
        proposed_parts=proposed_parts,
        source_provenance_notes=[
            'The initial 400s benchmark is retained, including its unsuccessful closure result.',
            'Its executed source was reconstructed later and matches the contemporaneous SHA256 exactly; see source_reconstruction.json.',
            'The 1000s benchmark has its original executed-source copy; corrected reduced-model validation is a separate tree.',
            'No archived source is silently replaced by the latest script. Vendor code is MIT licensed and was inspected, not executed.'],
        git_recommended_files=git_candidates(root),
        git_exclusions='All checkpoint/tensor arrays, dense raw JSON, logs, vendor notebooks/data, archives, operational packaging snapshot, unrelated files, and unfinished distillation work. Complete bytes remain in the release manifest.',
        file_count=len(entries), uncompressed_bytes=sum(e['bytes'] for e in entries), files=entries)
    if strict:
        assert status['complete'], json.dumps(status['issues'], indent=2)
        partition(entries, PAYLOAD_BYTES)
    return manifest


def safe_archive_name(name):
    relative = PurePosixPath(name)
    assert name and '\\' not in name and '\0' not in name
    assert relative.parts and not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name


def verify_archive(path, entries, manifest_bytes):
    expected = {e['path']: e for e in entries}
    assert len(expected) == len(entries) and MANIFEST_NAME not in expected
    expected[MANIFEST_NAME] = dict(bytes=len(manifest_bytes), sha256=hashlib.sha256(manifest_bytes).hexdigest())
    seen = []
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            safe_archive_name(member.name)
            assert member.isfile() and member.name in expected and member.name not in seen
            assert (member.uid, member.gid, member.mtime, member.mode) == (0, 0, 0, 0o644)
            assert not member.uname and not member.gname
            row = expected[member.name]
            assert member.size == row['bytes']
            with archive.extractfile(member) as stream:
                assert digest(stream) == row['sha256'], f'Archived bytes differ: {member.name}'
            seen.append(member.name)
    assert seen == sorted(expected), 'Archive member set/order differs'


def build(root, manifest, out, *, payload_limit=PAYLOAD_BYTES, asset_limit=MAX_ASSET_BYTES):
    root, out = Path(root).resolve(), Path(out).resolve()
    assert manifest['completion']['complete'] and manifest['source_commit']
    for tree in TREES:
        assert not out.is_relative_to(root/tree), 'Archive destination overlaps input tree'
    assert not out.exists() or not any(out.iterdir()), 'Use a fresh archive directory'
    entries = manifest['files']
    assert len({e['path'] for e in entries}) == len(entries)
    for entry in entries:
        prove(root, entry['path'], entry)
    parts = partition(entries, payload_limit)
    assert parts
    body = json_bytes(manifest)
    out.mkdir(parents=True, exist_ok=True)
    index = dict(version=manifest['version'], source_commit=manifest['source_commit'],
        manifest_sha256=hashlib.sha256(body).hexdigest(), max_asset_bytes=asset_limit, parts=[])
    for number, part in enumerate(parts, 1):
        path = out/f'nonlinear-ring-outputs-part{number:03d}-of{len(parts):03d}.tar.gz'
        temporary = path.with_suffix('.gz.tmp')
        with temporary.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                contents = {e['path']: e for e in part}
                for name in sorted([*contents, MANIFEST_NAME]):
                    entry = contents.get(name)
                    info = tarfile.TarInfo(name)
                    info.size = len(body) if entry is None else entry['bytes']
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    if entry is None:
                        archive.addfile(info, io.BytesIO(body))
                    else:
                        with checked(root, name).open('rb') as stream:
                            archive.addfile(info, stream)
        assert temporary.stat().st_size < asset_limit, 'Compressed asset must be below size limit'
        verify_archive(temporary, part, body)
        temporary.replace(path)
        index['parts'].append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path),
            members=[e['path'] for e in part], all_archived_bytes_verified=True))
    # Also reject changed input bytes after the last member has been streamed.
    for entry in entries:
        prove(root, entry['path'], entry)
    (out/'manifest.json').write_bytes(body)
    (out/'index.json').write_bytes(json_bytes(index))
    return index


def verify_release(out):
    out = Path(out).resolve()
    manifest_path = checked(out, 'manifest.json')
    body = manifest_path.read_bytes()
    manifest, index = json.loads(body), read(checked(out, 'index.json'))
    assert sha(manifest_path) == index['manifest_sha256']
    assert manifest['completion']['complete'] and manifest['source_commit'] == index['source_commit']
    expected = {e['path']: e for e in manifest['files']}
    assert len(expected) == len(manifest['files'])
    seen = []
    for part in index['parts']:
        assert PurePosixPath(part['path']).name == part['path']
        path = prove(out, part['path'], part)
        assert path.stat().st_size < index['max_asset_bytes'] <= MAX_ASSET_BYTES
        entries = [expected[name] for name in part['members']]
        verify_archive(path, entries, body)
        seen.extend(part['members'])
    assert sorted(seen) == sorted(expected) and len(seen) == len(expected)
    return dict(verified=True, parts=len(index['parts']), files=len(expected), source_commit=index['source_commit'])


def self_test():
    """Tiny temporary byte fixtures only; never read study data or create release assets."""
    with tempfile.TemporaryDirectory(prefix='nonlinear-package-test-') as directory:
        root = Path(directory)
        rows = []
        for name, data in [('z.txt', b'z'), ('a.txt', b'a'), ('nested/b.txt', b'b')]:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            rows.append(dict(path=name, bytes=len(data), sha256=sha(path)))
        manifest = dict(version='synthetic', source_commit='a'*40,
                        completion=dict(complete=True), files=rows)
        first = build(root, manifest, root/'one', payload_limit=1, asset_limit=10000)
        second = build(root, manifest, root/'two', payload_limit=1, asset_limit=10000)
        assert first == second and len(first['parts']) == 3
        assert verify_release(root/'one')['files'] == 3
        unfinished = dict(manifest, completion=dict(complete=False))
        try:
            build(root, unfinished, root/'unfinished', payload_limit=1, asset_limit=10000)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete release accepted')
        linked = root/'linked.tar.gz'
        with tarfile.open(linked, 'w:gz') as archive:
            member = tarfile.TarInfo('a.txt')
            member.type = tarfile.SYMTYPE
            member.linkname = '../escape'
            archive.addfile(member)
        try:
            verify_archive(linked, rows, json_bytes(manifest))
        except AssertionError:
            pass
        else:
            raise AssertionError('Archive link accepted')
        (root/'a.txt').write_bytes(b'changed')
        try:
            build(root, manifest, root/'changed', payload_limit=1, asset_limit=10000)
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed input accepted')
        for unsafe in ('../escape', '/absolute', 'x//y', 'x\\y', '.'):
            try:
                safe_archive_name(unsafe)
            except AssertionError:
                pass
            else:
                raise AssertionError('Unsafe member accepted')
        part = root/'one'/first['parts'][0]['path']
        part.write_bytes(b'corrupt archive')
        try:
            verify_release(root/'one')
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupt archive accepted')
    return dict(passed=True, synthetic_only=True, checks=[
        'deterministic multipart bytes', 'every member verified', 'source mutation rejected',
        'incomplete release rejected', 'archive links rejected',
        'unsafe paths rejected', 'archive corruption rejected'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan', action='store_true')
    actions.add_argument('--inventory-only', action='store_true')
    actions.add_argument('--build', action='store_true')
    actions.add_argument('--verify-release', action='store_true')
    actions.add_argument('--self-test', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--out', type=Path, default=ROOT/RELEASE/'archives')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.verify_release:
        result = verify_release(args.out)
    else:
        manifest = inventory(ROOT, args.source_commit, strict=args.build or args.inventory_only)
        result = build(ROOT, manifest, args.out) if args.build else manifest
    print(json_bytes(result).decode(), end='')


if __name__ == '__main__':
    main()
