"""Read-only final private-stream comparison for the four fixed portable arms."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SHA='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
NAMES={'plan','motor','action','state'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare(streams):
    assert len(streams)==4
    first=next(iter(streams.values()))
    for state in streams.values():
        assert set(state)=={'seed','batches','contexts','generators'}
        assert state['seed']==494801 and state['batches']==1024 and state['contexts']==32768
        assert set(state['generators'])==NAMES
        for t in state['generators'].values():assert t.dtype==torch.uint8 and t.ndim==1
    return {key:all(torch.equal(state['generators'][key],first['generators'][key]) for state in streams.values()) for key in sorted(NAMES)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routing',type=Path,default=ROOT/'docs/research/local_return_feedback_portable_launch.json')
    p.add_argument('--out',type=Path,default=ROOT/'results/local_return_feedback_design/portable/cloud_audit')
    a=p.parse_args();routing=json.loads(a.routing.read_text());assert routing['protocol_sha256']==PROTOCOL_SHA
    assert {(w['condition'],w['actor_mode']) for w in routing['workers']}=={(c,m) for c in ('gc100','pgc33') for m in ('retained_actor','reset_actor')}
    target=a.out/'paired_final_training_validation.json';raw=a.out/'paired_final_private_streams.pt'
    assert not target.exists() and not raw.exists(), 'Preserve previous validation artifacts'
    streams={};inputs=[]
    for worker in routing['workers']:
        assert worker['status']=='complete_collected_audited_deleted'
        audit_path=ROOT/worker['audit_path'];audit=json.loads(audit_path.read_text())
        assert audit['passed'] and audit['complete'] and not audit['failures'] and not audit['pending']
        assert audit['protocol_sha256']==PROTOCOL_SHA
        checkpoint=ROOT/worker['local_artifact_root']/worker['artifact_subdir']/'final.pt'
        assert audit['artifacts']['final.pt']==dict(sha256=sha(checkpoint),bytes=checkpoint.stat().st_size)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['completed'] and saved['counters']['updates']==1024 and saved['counters']['trials']==32768
        assert saved['provenance']['protocol_sha256']==PROTOCOL_SHA
        assert saved['provenance']['condition']==worker['condition'] and saved['provenance']['actor_mode']==worker['actor_mode']
        streams[worker['name']]=saved['streams']
        inputs.append(dict(name=worker['name'],condition=worker['condition'],actor_mode=worker['actor_mode'],
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
            audit=str(audit_path.relative_to(ROOT)),audit_sha256=sha(audit_path),
            active_physics_frames=saved['active_physics_frames'],counters=saved['counters']))
    checks=compare(streams);a.out.mkdir(parents=True,exist_ok=True);torch.save(streams,raw)
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),complete=True,passed=all(checks.values()),
        scope='All4 fixed-final named private generator states and seed/batch/context counters. This checks matching draw consumption; first-batch tensor equality is separately archived. It does not independently save or replay every intervening Gaussian array. Global network RNG and active physical exposure may differ and are not equalized.',
        protocol_sha256=PROTOCOL_SHA,routing_sha256=sha(a.routing),producer_sha256=sha(__file__),
        checks=checks,inputs=inputs,raw=dict(path=str(raw.relative_to(ROOT)),sha256=sha(raw),bytes=raw.stat().st_size),
        first_batch_record='results/local_return_feedback_design/portable/cloud_audit/paired_first_plan_validation.json')
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],checks=checks)))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
