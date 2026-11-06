"""Verify worker-added runtime provenance, then call the unchanged frozen assay."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import lstm_final75_memory as frozen
from lstm_heading_memory_probe import sha,save

BASE_VALIDATOR=frozen.validate_checkpoint


def normalize_metadata(data,arm,condition):
    """Return a shallow checkpoint copy after verifying/removing runtime metadata."""
    provenance=copy.deepcopy(data['provenance'])
    runtime=provenance.pop('runtime')
    assert set(runtime)=={'torch_version','condition','device','protocol_sha256'}
    assert runtime['condition']==condition and runtime['device']=='cuda'
    assert isinstance(runtime['torch_version'],str) and runtime['torch_version']
    assert runtime['protocol_sha256']==sha(frozen.TRAINING_PROTOCOLS[arm])
    return dict(data,provenance=provenance)


def checked_metadata(data,arm,condition,design):
    normalized=normalize_metadata(data,arm,condition)
    BASE_VALIDATOR(normalized,arm,condition,design)
    return copy.deepcopy(data['provenance']['runtime'])


def main():
    path=ROOT/'results/lstm_final75_memory/runtime_adapter_protocol.json'
    protocol=dict(reason='First final-v2 attempts stopped before collection because worker appends runtime metadata to scientific provenance. All scientific fields matched. Frozen geometry wrapper is unchanged.',
        validation='Require exact runtime keys, correct condition/CUDA backend, nonempty torch version and exact frozen training-protocol SHA; remove only this verified metadata field before unchanged original validator.',
        source_sha256=sha(__file__),unchanged_wrapper_sha256=sha(ROOT/'scripts/lstm_final75_memory.py'))
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    def validator(data,arm,condition,design):
        runtime=checked_metadata(data,arm,condition,design)
        out=Path(sys.argv[sys.argv.index('--out')+1])
        save(out/'runtime_metadata_audit.json',dict(runtime=runtime,scientific_provenance_exact=True,adapter_source_sha256=sha(__file__)))
    with patch.object(frozen,'validate_checkpoint',validator):frozen.main()


if __name__=='__main__':main()
