"""Reproduce frozen primary analyses using published native context inputs.

Only the collector is replaced by loading its previously saved dictionaries.
The original main functions, numerical analyses, and candidate gates are used.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import lstm_continued_heading_memory as heading
import lstm_travel_memory_probe as travel
import lstm_travel_memory_heldout as heldout
from lstm_endogenous_curve_recovery import sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['continued_heading','continued_travel','heldout_travel'],required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='mps');args=p.parse_args();out=args.out or args.source_root;loaded=[]
    def cached(net,cfg,checkpoint,parameters):
        folder=args.source_root/Path(parameters.out).name if args.task=='heldout_travel' else args.source_root
        windows=[]
        for context in (0,1):
            path=folder/f'contexts/context{context}.npz';a=np.load(path)
            window={key:a[key]for key in a.files}
            assert str(window['checkpoint_sha256'].item())==sha(checkpoint),'Native context checkpoint mismatch'
            windows.append(window);loaded.append(dict(path=str(path),sha256=sha(path),checkpoint_sha256=sha(checkpoint)))
        # The original held-out caller reads this metadata after collection.
        target=Path(parameters.out);target.mkdir(parents=True,exist_ok=True)
        metadata=folder/'collection.json'
        if target.resolve()!=folder.resolve():(target/'collection.json').write_bytes(metadata.read_bytes())
        return windows
    if args.task=='continued_heading':
        if args.checkpoint is None:p.error('--checkpoint required')
        argv=['lstm_continued_heading_memory.py','--checkpoint',str(args.checkpoint),'--out',str(out),'--device',args.device]
        entry=heading.main
    elif args.task=='continued_travel':
        if args.checkpoint is None:p.error('--checkpoint required')
        argv=['lstm_travel_memory_probe.py','--cohort','continued','--checkpoint',str(args.checkpoint),'--out',str(out),'--device',args.device]
        entry=travel.main
    else:
        argv=['lstm_travel_memory_heldout.py','--out',str(out),'--device',args.device];entry=heldout.main
    with patch.object(heading,'collect_contexts',cached),patch.object(sys,'argv',argv):entry()
    save(out/'cached_input_reproduction.json',dict(task=args.task,wrapper_sha256=sha(__file__),native_inputs=loaded,
        interpretation='Published collector dictionaries reused exactly; no new context selection. Frozen analysis main functions and gates are unchanged. Rendering/BLAS numerical backends can still affect floating-point details.'))


if __name__=='__main__':main()
