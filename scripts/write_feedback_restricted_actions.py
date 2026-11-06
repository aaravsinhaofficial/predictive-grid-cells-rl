"""Serialize the unchanged frozen restricted-action calculation's NumPy scalars."""
import argparse
import json
from pathlib import Path
import numpy as np
import analyze_feedback_restricted_actions as frozen


def json_values(value):
    if isinstance(value,dict):return {k:json_values(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_values(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=frozen.ROOT/'results/local_return_feedback_design/portable/restricted_action_bound')
    args=parser.parse_args();assert not args.out.exists() or not any(args.out.iterdir()),'Preserve previous outputs'
    result=json_values(frozen.analyze())
    result['writer_sha256']=frozen.sha(__file__)
    result['serialization_note']='Original writer rejected numpy.bool_ after calculation; no numerical functions changed. This wrapper converts NumPy scalars to JSON-native scalar types.'
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Turning right and strafing right limits the available routes','',
        'Secondary analysis of the fixed **pretraining baseline**; no model or environment was run.','',
        '| Population | Potentially reachable headings | Analytical upper bound | Unresolved headings |',
        '|---|---:|---:|---:|']
    for r in result['summaries']:
        lines.append(f"| {r['condition']} | {r['possible_trials']}/256 | {r['bound_percent']:.1f}% | {r['unknown_trials']} |")
    lines += ['', 'The calculation bounds **any controller restricted to right turns and right strafes**, using the same saved starting poses and body-frame motor-noise tapes. All 16 contexts and 16 headings remain. Passing the necessary angle condition means potentially reachable, not that a route has been found.', '',
        'The Grid baseline sampled only these two actions; the PGC baseline also sampled nine backward actions. The full softmax networks retain other actions, so this is not their unrestricted performance ceiling. It suggests why acquiring a broader movement strategy may matter for fast local return, independently of memory or ring geometry.', '',
        'This is a bound for the native equations with declared 0.1 mm and 0.0001 rad numerical margins, not formal hardware verification. [Derivation](../../../../docs/research/feedback_restricted_action_bound.md) · [All cases and input hashes](summary.json).', '']
    (args.out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(result['summaries']))


if __name__=='__main__':main()
