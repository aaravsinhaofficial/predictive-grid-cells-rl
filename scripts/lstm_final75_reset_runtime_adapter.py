"""Verify additive runtime metadata, then run the frozen75M reset evaluator."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import lstm_final75_every_blind_reset as frozen
from lstm_final75_memory_runtime_adapter import normalize_metadata
from lstm_endogenous_curve_recovery import sha,save

PROTOCOL=ROOT/'docs/research/final75_reset_runtime_adapter_protocol.json'
ORIGINAL_VALIDATE=frozen.validate


def validate(data,key):
    arm,condition=key.split('_',1)
    scientific=normalize_metadata(data,arm,condition) if 'runtime' in data['provenance'] else data
    ORIGINAL_VALIDATE(scientific,key)


def design():
    sources=('scripts/lstm_final75_reset_runtime_adapter.py',
             'scripts/lstm_final75_memory_runtime_adapter.py')
    return dict(version='final75_reset_runtime_metadata_adapter_v1',
        reason='Noiseless final checkpoints append verified runtime information to frozen scientific provenance. Noise-arm checkpoints contain the exact scientific dictionary without this additive field.',
        operation='Validate exact runtime keys, CUDA device, condition, torch version and frozen protocol hash; pass a shallow copy without this verified additive field to the unchanged scientific validator. Original checkpoint, scientific design, reset evaluator, model hashes and all outcomes remain unchanged.',
        source_sha256={p:sha(ROOT/p) for p in sources},
        frozen_reset_protocol_sha256=sha(ROOT/'docs/research/final75_every_blind_reset_protocol.json'))


def main():
    current=design()
    if sys.argv[1:]==['--freeze-adapter']:
        assert not PROTOCOL.exists()
        save(PROTOCOL,dict(current,frozen_at=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(sha(PROTOCOL));return
    saved=json.loads(PROTOCOL.read_text());saved.pop('frozen_at')
    assert current==saved
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,required=True)
    args,_=parser.parse_known_args()
    with patch.object(frozen,'validate',validate):
        frozen.main()
    save(args.out/'metadata_adapter_protocol.json',current)


if __name__=='__main__':main()
