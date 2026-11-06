"""Reproduce fixed90M geometry using the exact saved native context inputs.

Only context collection is replaced. Genuine endpoint validation and every
frozen physical/neural helper remain unchanged. Run after finals are available.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import lstm_reorientation_entropy_mechanism as producer
from lstm_endogenous_curve_recovery import sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=producer.CONDITIONS,required=True)
    p.add_argument('--arm',choices=tuple(producer.ARMS),required=True)
    p.add_argument('--inputs',type=Path,required=True,help='Archived contexts directory')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--routing',type=Path,default=producer.ROUTING);args=p.parse_args()
    destination=args.out/args.arm/args.condition
    assert destination.resolve()!=args.inputs.parent.resolve(),'Use a separate output directory'
    frozen=json.loads((producer.OUT/'protocol.json').read_text())
    assert frozen==producer.protocol()
    checkpoint=producer.resolve_checkpoint(args.condition,args.arm,args.routing)
    producer.preflight(args.condition,args.arm,checkpoint,args.routing)
    contexts=[];assets=[]
    for index in (0,1):
        path=args.inputs/f'context{index}.npz'
        with np.load(path) as a:window={key:a[key] for key in a.files}
        assert str(window['checkpoint_sha256'])==sha(checkpoint)
        contexts.append(window);assets.append(dict(path=str(path),sha256=sha(path)))

    def cached_collection(net,cfg,actual_checkpoint,out):
        assert sha(actual_checkpoint)==sha(checkpoint)
        save(out/'cached_collection.json',dict(checkpoint_sha256=sha(checkpoint),inputs=assets,
            source_sha256=sha(__file__),genuine_extra_frames=90_132_480,
            replacement='Exact archived native contexts only; no new context selection or metadata rewriting'))
        return contexts

    argv=['lstm_reorientation_entropy_mechanism.py','--condition',args.condition,'--arm',args.arm,
          '--routing',str(args.routing),'--out',str(args.out),'--mode','primary']
    with patch.object(sys,'argv',argv),patch.object(producer.primary,'collect_contexts',cached_collection):
        producer.main()


if __name__=='__main__':main()
