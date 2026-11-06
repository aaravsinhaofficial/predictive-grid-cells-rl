"""Compare saved private draw streams; never execute models or generators.

Explicit partial mode validates already audited matching stages. Final mode
requires both complete canonical audits and compares all four stages plus final.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/research/local_goal_return_training_protocol.json'
PROTOCOL_SHA='0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
ROUTING=ROOT/'docs/research/local_goal_return_training_launch.json'
OUT=ROOT/'results/local_goal_return_design/cloud_audit'
CONDITIONS=('gc100','pgc33')
GENERATORS={'plan','motor','action','state'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def compare_snapshots(first,second,updates):
    assert set(first)==set(second)=={'seed','batches','contexts','generators'}
    assert set(first['generators'])==set(second['generators'])==GENERATORS
    fields={key:dict(grid=first[key],pgc=second[key],equal=first[key]==second[key],
        expected=expected,expected_values=first[key]==second[key]==expected)
        for key,expected in (('seed',294801),('batches',updates),('contexts',32*updates))}
    generators={}
    for name in sorted(GENERATORS):
        a,b=first['generators'][name],second['generators'][name]
        assert a.dtype==b.dtype==torch.uint8 and a.ndim==b.ndim==1
        assert a.device.type==b.device.type=='cpu'
        generators[name]=dict(equal=torch.equal(a,b),grid_shape=list(a.shape),pgc_shape=list(b.shape),
            grid_sha256=hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest(),
            pgc_sha256=hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest())
    return dict(passed=all(v['equal'] and v['expected_values'] for v in fields.values())
                and all(v['equal'] for v in generators.values()),
        metadata=fields,generators=generators,global_worker_rng_compared=False)


def audit_for(folder,condition,stage,final_mode):
    path=OUT/(f'{condition}.json' if final_mode else f'stage{stage}_{condition}.json')
    audit=read(path)
    assert audit['passed'] and not audit['failures']
    if final_mode:assert audit['complete'] and audit['status']=='complete' and not audit['pending']
    assert audit['condition']==condition and audit['protocol_sha256']==PROTOCOL_SHA
    rel=folder.relative_to(ROOT)
    assert Path(audit['folder']).parts[-len(rel.parts):]==rel.parts
    assert audit['auditor_sha256']==sha(ROOT/'scripts/audit_local_goal_return_training.py')
    for name,digest in audit['audit_dependency_sha256'].items():assert sha(ROOT/name)==digest
    return path,audit


def collect(partial_through_stage=None):
    assert partial_through_stage is None or partial_through_stage in (1,2,3)
    assert sha(PROTOCOL)==PROTOCOL_SHA
    protocol=read(PROTOCOL)
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest
    routes=read(ROUTING)['workers']
    assert len(routes)==2 and {r['condition'] for r in routes}==set(CONDITIONS)
    final_mode=partial_through_stage is None
    maximum=4 if final_mode else partial_through_stage
    checkpoints=[(f'stage{s}.pt',s,s*1024) for s in range(1,maximum+1)]
    if final_mode:checkpoints.append(('final.pt',4,4096))
    snapshots={};results={};inputs={}
    for name,stage,updates in checkpoints:
        paired={};references={}
        for condition in CONDITIONS:
            route=next(r for r in routes if r['condition']==condition)
            relative=Path(route['local_artifact_root'])/route['artifact_subdir']
            assert not relative.is_absolute() and '..' not in relative.parts
            folder=ROOT/relative;path=folder/name
            audit_path,audit=audit_for(folder,condition,stage,final_mode)
            expected=audit['artifacts'][name]
            assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
            value=torch.load(path,map_location='cpu',weights_only=True)
            assert value['provenance']['condition']==condition and value['provenance']['protocol_sha256']==PROTOCOL_SHA
            assert value['counters']['updates']==updates and value['counters']['trials']==updates*32
            assert value['counters']['completed_stages']==stage
            paired[condition]=value['streams']
            references[condition]=dict(path=str(path.relative_to(ROOT)),sha256=expected['sha256'],
                audit_path=str(audit_path.relative_to(ROOT)),audit_sha256=sha(audit_path))
        snapshots[name]=paired;inputs[name]=references
        results[name]=compare_snapshots(paired['gc100'],paired['pgc33'],updates)
    complete=final_mode and list(results)==['stage1.pt','stage2.pt','stage3.pt','stage4.pt','final.pt']
    report=dict(complete=complete,passed=all(r['passed'] for r in results.values()),
        partial_through_stage=partial_through_stage,protocol_sha256=PROTOCOL_SHA,
        reporting_source_sha256=sha(__file__),observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        inputs=inputs,comparisons=results,global_worker_rng_compared=False,
        scope='Exact saved private seeds/counters/generator-state comparison. Draws, neural states, policies, environments and learning are not executed. Global worker RNG remains separately preserved in each original checkpoint.',
        inference='Equal states and the frozen deterministic draw schedule verify equal private draw consumption; different success times do not change the prescribed later trial/tape stream.')
    return report,dict(protocol_sha256=PROTOCOL_SHA,inputs=inputs,snapshots=snapshots,
        complete=complete,global_worker_rng_included=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial-through-stage',type=int,choices=(1,2,3))
    p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    report,snapshots=collect(args.partial_through_stage)
    if args.partial_through_stage is None:
        raw_name='paired_private_streams.pt';json_name='paired_final_training_validation.json'
    else:
        raw_name=f'paired_private_streams_stage{args.partial_through_stage}.pt'
        json_name=f'paired_training_streams_stage{args.partial_through_stage}.json'
    args.out.mkdir(parents=True,exist_ok=True);raw=args.out/raw_name;path=args.out/json_name
    assert not raw.exists() and not path.exists(),'Preserve earlier comparison records; use a fresh output folder'
    temporary=raw.with_suffix('.pt.tmp');torch.save(snapshots,temporary);temporary.replace(raw)
    report['raw_private_snapshots']=dict(file=raw.name,sha256=sha(raw),bytes=raw.stat().st_size)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=report['complete'],passed=report['passed'],checkpoints=list(report['comparisons']),report=str(path))))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
