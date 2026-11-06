"""Reproduce the frozen40M geometry from archived native contexts.

Only native context collection is replaced by its exact saved inputs. Final
checkpoint validation, physical replay, neural flow and every gate are unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import lstm_reorientation_mechanism as primary
from lstm_endogenous_curve_recovery import sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True,help='Archived contexts directory')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert args.out.resolve()!=args.inputs.parent.resolve(),'Use a separate output directory'
    expected=json.loads((ROOT/'results/lstm_reorientation_mechanism/implementation_protocol.json').read_text())
    assert expected['design_sha256']==sha(primary.PROTOCOL)
    for path,digest in expected['source_sha256'].items():assert sha(ROOT/path)==digest,path
    contexts=[];assets=[]
    for index in (0,1):
        path=args.inputs/f'context{index}.npz'
        with np.load(path) as a:window={key:a[key] for key in a.files}
        assert str(window['checkpoint_sha256'])==sha(args.checkpoint)
        contexts.append(window);assets.append(dict(path=str(path),sha256=sha(path)))

    def cached_collection(net,cfg,checkpoint,out):
        assert sha(checkpoint)==sha(args.checkpoint)
        save(out/'cached_collection.json',dict(checkpoint_sha256=sha(checkpoint),inputs=assets,
            source_sha256=sha(__file__),original_collector_unchanged=True,
            replacement='Exact archived native context inputs only; no new context selection'))
        return contexts

    argv=['lstm_reorientation_mechanism.py','--checkpoint',str(args.checkpoint),'--out',str(args.out)]
    with patch.object(sys,'argv',argv),patch.object(primary,'collect_contexts',cached_collection):
        primary.main()


if __name__=='__main__':main()
