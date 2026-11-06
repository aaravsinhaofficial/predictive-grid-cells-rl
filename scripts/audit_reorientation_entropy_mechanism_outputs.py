"""Independent artifact/opportunity audit of all four genuine fixed90M models.

The original saved-path and native-physics checks are called unchanged. Only
the new producer/routing metadata and the causal protocol envelope differ.
No checkpoint is relabeled, no neural history is recomputed, and all defined
interventions and zero-hit windows are retained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import audit_reorientation_mechanism_outputs as original
import lstm_reorientation_entropy_mechanism as producer

BASE=producer.OUT
ORIGINAL_AUDITOR_SHA='5285b44f16347d3d78a5cd62ab38485f17b60e328c314ea0c67b48d826c5d67e'
PRODUCER_PROTOCOL_SHA='18728e4b6b3d6f324f180dfced93ad980bcb462916ec971ef58cc5d030ec6562'
CAUSAL_PROTOCOL=ROOT/'results/lstm_reorientation_mechanism/causal_navigation/protocol.json'
CAUSAL_PROTOCOL_SHA='d4c555909fdbebb0142cf241e40a532931b7259c9eb9a189f99e1cda21d5c9a3'
sha=original.sha
read=original.read


def verify_sources():
    assert sha(Path(original.__file__))==ORIGINAL_AUDITOR_SHA
    assert sha(BASE/'protocol.json')==PRODUCER_PROTOCOL_SHA
    assert read(BASE/'protocol.json')==producer.protocol()
    assert sha(CAUSAL_PROTOCOL)==CAUSAL_PROTOCOL_SHA


def check_completed_training_audit(path,folder,condition,arm,checkpoint_hash):
    audit=read(path)
    assert audit['passed'] and audit['complete'] and audit['required_complete']
    assert not audit['failures'] and not audit['pending']
    assert (audit['condition'],audit['arm'])==(condition,arm)
    assert Path(audit['folder']).resolve()==Path(folder).resolve()
    assert audit['auditor_sha256']==sha(ROOT/'scripts/audit_rl_blind_reorientation_entropy.py')
    assert audit['artifacts']['final.pt']['sha256']==checkpoint_hash
    for name,item in audit['artifacts'].items():
        artifact=Path(folder)/name
        assert artifact.stat().st_size==item['bytes'] and sha(artifact)==item['sha256'],name
    return dict(path=str(path),sha256=sha(path),all_training_artifact_hashes_verified=True)


def check_native_collection(folder,condition,arm,checkpoint_hash):
    producer.validate_producer(folder,checkpoint_hash,condition,arm)
    collection=read(folder/'collection.json')
    assert collection['contexts']==2 and collection['checkpoint_sha256']==checkpoint_hash
    assert collection['state_noise'] is False and collection['native_environment_noise'] is True
    assert 2700<=collection['decisions']<=6000
    coverage=read(folder/'native_phase_coverage.json')
    assert coverage['observation_only'] and coverage['decisions']==collection['decisions']
    hashes={}
    for context in (0,1):
        path=folder/'contexts'/f'context{context}.npz'
        with np.load(path) as native:
            assert str(native['checkpoint_sha256'].item())==checkpoint_hash
        hashes[f'context{context}']=sha(path)
    return dict(collection_sha256=sha(folder/'collection.json'),
        native_phase_coverage_sha256=sha(folder/'native_phase_coverage.json'),
        native_context_sha256=hashes,unchanged_native_collection=True)


def audit_causal_envelope(folder,primary_folder,context,checkpoint_hash,condition,arm):
    """Unwrap ONLY the inherited numerical source manifest for the old auditor.

    Genuine90M metadata is checked first and remains unchanged on disk and in
    memory. The old path checker only needs the inherited source-hash mapping;
    it never sees a fabricated40M checkpoint or rewritten endpoint fields.
    """
    protocol_path=folder/'protocol.json'
    envelope=read(protocol_path);inputs=read(folder/'inputs.json')
    assert envelope['adapter']==producer.protocol()
    inherited=read(CAUSAL_PROTOCOL)
    assert envelope['inherited_numerical_protocol']==inherited
    assert envelope['inherited_cohort_override']=='Numerical protocol unchanged; actual prospective cohort is four fixed90M endpoints, not40M'
    assert (inputs['condition'],inputs['arm'])==(condition,arm)
    assert inputs['genuine_extra_frames']==90_132_480
    assert inputs['checkpoint_sha256']==checkpoint_hash
    assert Path(inputs['state_archive']).resolve()==(primary_folder/f'context{context}_states.npz').resolve()
    before={path:sha(path) for path in (protocol_path,folder/'inputs.json')}
    original_read=original.read

    def read_inherited_manifest(path):
        return inherited if Path(path).resolve()==protocol_path.resolve() else original_read(path)

    with patch.object(original,'read',read_inherited_manifest):
        report=original.audit_causal(folder,primary_folder,context,checkpoint_hash,condition)
    assert all(sha(path)==digest for path,digest in before.items())
    return dict(report,arm=arm,condition=condition,genuine_extra_frames=90_132_480,
        causal_envelope_sha256=before[protocol_path],
        inherited_numerical_auditor_sha256=ORIGINAL_AUDITOR_SHA,
        metadata_adapter='Only unwrapped the verified inherited numerical source manifest; original90M metadata untouched')


def run(require_complete=False,routing=producer.ROUTING):
    verify_sources();reports={};pending=[]
    for arm in producer.ARMS:
        reports[arm]={}
        for condition in producer.CONDITIONS:
            label=f'{arm}/{condition}'
            checkpoint=producer.resolve_checkpoint(condition,arm,routing)
            folder=BASE/arm/condition
            if not checkpoint.exists():pending.append(label+'/final.pt');continue
            data,cfg=producer.preflight(condition,arm,checkpoint,routing)
            checksum=sha(checkpoint)
            audit_path=ROOT/f'results/rl_blind_reorientation_entropy/cloud_audit/{condition}_{arm}.json'
            if not audit_path.exists() or not read(audit_path).get('complete'):
                pending.append(label+'/complete_training_audit');continue
            training=check_completed_training_audit(audit_path,checkpoint.parent,condition,arm,checksum)
            if not (folder/'summary.json').exists():pending.append(label+'/primary');continue
            native=check_native_collection(folder,condition,arm,checksum)
            summary=read(folder/'summary.json')
            assert (summary['condition'],summary['arm'])==(condition,arm)
            assert summary['extra_frames']==data['extra_frames']==90_132_480
            assert summary['checkpoint_sha256']==checksum and len(summary['contexts'])==2
            report=dict(arm=arm,condition=condition,genuine_extra_frames=data['extra_frames'],
                checkpoint_sha256=checksum,training_audit=training,native_collection=native,
                primary=[],causal=[])
            for context in (0,1):
                report['primary'].append(original.audit_primary(folder,condition,checksum,context))
                causal=folder/'causal'/f'context{context}'
                if not (causal/'result.json').exists() or not (causal/'navigation_paths.npz').exists():
                    pending.append(f'{label}/causal/context{context}')
                else:report['causal'].append(audit_causal_envelope(causal,folder,context,checksum,condition,arm))
            assert sha(checkpoint)==checksum
            reports[arm][condition]=report
    complete=not pending and sum(len(rows) for rows in reports.values())==4
    if require_complete:assert complete,pending
    return dict(passed=True,complete=complete,pending=pending,source_sha256=sha(__file__),
        original_numerical_auditor_sha256=ORIGINAL_AUDITOR_SHA,
        producer_protocol_sha256=PRODUCER_PROTOCOL_SHA,
        routing_sha256=sha(routing),routing_snapshot=read(routing),models=reports,
        scope='Independent saved-artifact/path/opportunity reconciliation of all4fixed90Mmodels and8contexts. Genuine final/provenance validation; unchanged40M numerical path/physics checks. All defined interventions,192phases and zero-hit windows retained; no neural outcomes recomputed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routing',type=Path,default=producer.ROUTING)
    parser.add_argument('--out',type=Path,default=BASE/'independent_artifact_audit.json')
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    try:output=run(args.require_complete,args.routing)
    except Exception as exc:
        output=dict(passed=False,complete=False,source_sha256=sha(__file__),error=f'{type(exc).__name__}: {exc}')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({key:output.get(key) for key in ('passed','complete','pending','error')}))
    if not output['passed']:raise SystemExit(1)


if __name__=='__main__':main()
