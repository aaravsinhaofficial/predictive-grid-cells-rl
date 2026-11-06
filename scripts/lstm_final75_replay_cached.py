"""Reproduce fixed75M primary assays from published native context inputs."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import lstm_final75_memory as primary
from lstm_final75_memory_runtime_adapter import normalize_metadata
from lstm_endogenous_curve_recovery import sha,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['v2','v3'],required=True)
    parser.add_argument('--condition',choices=['gc100','pgc33'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='mps')
    args=parser.parse_args()
    assert args.out.resolve()!=args.source_root.resolve(),'Reproduce into a separate output folder'
    loaded=[];original_validator=primary.validate_checkpoint

    def validate(data,arm,condition,design):
        scientific=normalize_metadata(data,arm,condition) if 'runtime' in data['provenance'] else data
        original_validator(scientific,arm,condition,design)

    def cached(net,cfg,checkpoint,parameters):
        kind=Path(parameters.out).name
        assert kind in ('heading','travel')
        source=args.source_root/kind
        windows=[]
        for context in (0,1):
            path=source/f'contexts/context{context}.npz'
            with np.load(path) as archive:
                window={k:archive[k] for k in archive.files}
            assert str(window['checkpoint_sha256'].item())==sha(checkpoint)
            windows.append(window)
            loaded.append(dict(kind=kind,context=context,path=str(path),sha256=sha(path)))
        Path(parameters.out).mkdir(parents=True,exist_ok=True)
        (Path(parameters.out)/'collection.json').write_bytes((source/'collection.json').read_bytes())
        return windows

    argv=['lstm_final75_memory.py','--arm',args.arm,'--condition',args.condition,
          '--checkpoint',str(args.checkpoint),'--device',args.device,'--out',str(args.out)]
    with patch.object(primary,'validate_checkpoint',validate),\
         patch.object(primary.heading,'collect_contexts',cached),\
         patch.object(sys,'argv',argv):
        primary.main()
    save(args.out/'cached_input_reproduction.json',dict(checkpoint_sha256=sha(args.checkpoint),
        native_inputs=loaded,wrapper_sha256=sha(__file__),
        scope='Published native collector dictionaries reused exactly; frozen primary analyses/gates unchanged. Input backend can still affect floating-point details; compare numerical archives.'))


if __name__=='__main__':main()
