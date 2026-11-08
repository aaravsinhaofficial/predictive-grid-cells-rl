"""Authenticate the explicit replay-guard amendment without changing assay math.

Original checkpoints keep their original provenance. Version-two endpoints must
also retain the exact parent evidence, migration state and registered lineage.
No policy forward pass, optimizer step or outcome selection occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = 'docs/research/simple_memory_guard_amendment.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def checked_file(folder, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Nonlocal evidence path')
    folder = Path(folder).resolve()
    path = folder / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder),
            'Missing or nonlocal evidence file')
    return path


def validate_v2_provenance(run, cfg, proof, *, root=ROOT, amendment_path=None):
    """Validate the complete producer/parent chain; final counters are checked by callers."""
    import torch
    from code import train_memory_v2 as producer

    root, run = Path(root).resolve(), Path(run).resolve()
    path = Path(amendment_path) if amendment_path else root / AMENDMENT
    amendment = read(path)
    require(amendment.get('schema') == 1 and amendment.get('status') == 'frozen' and
            amendment.get('kind') == 'simple_memory_replay_guard_amendment', 'Unfrozen guard amendment')
    require(proof.get('provenance_version') == 2 and proof.get('producer_module') == 'code.train_memory_v2',
            'Unexpected amended producer')
    require(proof.get('amendment_sha256') == sha(path), 'Guard amendment binding changed')
    require(isinstance(producer.GUARD_ID, str) and bool(producer.GUARD_ID) and
            amendment['guard']['status'] == 'frozen' and
            proof.get('guard_id') == amendment['guard']['id'] == producer.GUARD_ID,
            'Guard identity changed')
    training_path = root / 'docs/research/simple_memory_training_protocol.json'
    training = read(training_path)
    require(training['status'] == 'frozen' and
            amendment['parent_protocol_sha256'] == proof['protocol_sha256'] == sha(training_path),
            'Original scientific protocol changed')
    sources = amendment['source_sha256']
    require(set(sources) == set(producer.SOURCES) and proof['source_sha256'] == sources,
            'Incomplete amended producer source union')
    for name, digest in sources.items():
        require(sha(checked_file(root, name)) == digest, f'Changed amended source: {name}')
    base = proof['base_provenance']
    require(base['source_sha256'] == training['source_sha256'], 'Parent source closure changed')
    for name, digest in training['source_sha256'].items():
        require(sources.get(name) == digest, 'Original training source changed')
    for key, value in base.items():
        if key != 'source_sha256':
            require(proof.get(key) == value, f'Amendment changed original provenance: {key}')
    require('provenance_version' not in base and 'base_provenance' not in base, 'Nested parent producer')
    require(bool(amendment.get('numerical_runtime')) and
            all(base.get(key) == value for key, value in amendment['numerical_runtime'].items()),
            'Amended numerical runtime changed')
    scientific = {k: v for k, v in cfg.items() if k != 'population'}
    digest = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    require(base['config_sha256'] == digest, 'Amendment changed scientific configuration')
    logical = producer.logical_run_id(cfg)
    expected = {f'{condition}/{task}/seed{seed}' for condition in ('gc100', 'pgc33')
                for task in ('native', 'blackout') for seed in (1, 2, 3, 4)}
    bindings = amendment['run_bindings']
    require(set(bindings) == expected, 'Amendment changed sixteen-model cohort')
    kinds = [b['kind'] for b in bindings.values()]
    require({k: kinds.count(k) for k in set(kinds)} ==
            {'v1_completed': 4, 'v1_resume': 4, 'v2_fresh': 8}, 'Amendment changed producer assignment counts')
    binding = bindings[logical]
    require(binding['kind'] in ('v1_resume', 'v2_fresh') and proof['logical_run_id'] == logical,
            'Endpoint producer does not match registered logical model')
    lineage = read(checked_file(run, 'lineage.json'))
    require(sha(run / 'lineage.json') == proof['lineage_sha256'], 'Lineage bytes changed')
    require(lineage['schema'] == 1 and lineage['logical_run_id'] == logical and
            lineage['kind'] == binding['kind'] and lineage['amendment_sha256'] == sha(path) and
            lineage['parent_protocol_sha256'] == sha(training_path), 'Lineage identity changed')
    terminal = read(checked_file(run, 'summary.json'))
    require(terminal['lineage_sha256'] == proof['lineage_sha256'], 'Terminal lineage binding changed')
    initial_path = checked_file(run, 'initial.pt')
    require(sha(initial_path) == terminal['initial_sha256'], 'Original initialization changed')
    initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    require(initial['frames'] == initial['updates'] == 0 and initial['config'] == scientific,
            'Invalid initial checkpoint')
    require(producer.payload_digest(initial) == lineage['initial_payload_sha256'],
            'Initialization payload changed')
    if binding['kind'] == 'v1_resume':
        parent = run / 'parent_evidence'
        parent_cfg = dict(scientific, protocol=str(training_path))
        parent_saved, parent_details = producer.authenticate_parent(parent, binding, parent_cfg, base)
        require(initial['provenance'] == base and
                sha(initial_path) == binding['parent_evidence_sha256']['initial.pt'],
                'Resumed model changed original initialization')
        for key, value in parent_details.items():
            require(lineage.get(key) == value, f'Parent lineage mismatch: {key}')
        parent_digest = producer.payload_digest(parent_saved)
        require(lineage['full_state_exact'] is True and
                lineage['parent_payload_sha256'] == lineage['migrated_payload_sha256'] == parent_digest,
                'Parent migration payload mismatch')
        migrated = torch.load(checked_file(run, 'migration_boundary.pt'), map_location='cpu', weights_only=False)
        require(migrated['provenance'] == proof and producer.payload_digest(migrated) == parent_digest,
                'Migration changed full training state')
    else:
        require(initial['provenance'] == proof and
                lineage['restored_frames'] == lineage['restored_updates'] == lineage['accepted_parent_update_cutoff'] == 0,
                'Fresh model has a resumed history')
        require('parent_checkpoint_sha256' not in lineage and 'parent_evidence_sha256' not in lineage,
                'Unexpected parent for fresh model')
    return dict(producer_version=2, logical_run_id=logical, amendment_sha256=sha(path),
                guard_id=proof['guard_id'], lineage_sha256=proof['lineage_sha256'],
                producer_assignment=binding['kind'], restored_frames=lineage['restored_frames'],
                restored_updates=lineage['restored_updates'])


def authenticate_run(run, *, root=ROOT, amendment_path=None, analysis_protocol=None,
                     expected_analysis_sha256=None):
    """Operational entry point shared with the bounded analysis runner."""
    from scripts.run_simple_memory_analysis import authenticate_run as authenticate
    options = dict(root=root, amendment_path=amendment_path)
    if analysis_protocol is not None:
        options['analysis_protocol'] = analysis_protocol
    if expected_analysis_sha256 is not None:
        options['expected_analysis_sha256'] = expected_analysis_sha256
    return authenticate(run, **options)


def load_final(run, device='cpu', threads=2):
    from scripts.collect_simple_memory_states import load_final as load
    return load(run, device, threads)
