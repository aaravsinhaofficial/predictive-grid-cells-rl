"""Verify terminal cloud absence and freeze routing/accounting; never change a VM."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
ROUTING=ROOT/'docs/research/local_return_feedback_portable_launch.json'
OUT=ROOT/'results/local_return_feedback_design/portable/cloud_audit'
PROTOCOL_SHA='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'

def evidence(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)

def main():
    target=OUT/'cleanup_verification.json';assert not target.exists(), 'Preserve completed cleanup evidence'
    routing=json.loads(ROUTING.read_text());assert routing['protocol_sha256']==PROTOCOL_SHA
    ledger_path=ROOT/'docs/research/ring_cloud_budget.json';ledger=json.loads(ledger_path.read_text());records={r['name']:r for r in ledger['runs']}
    assert len(routing['workers'])==4
    rows=[];success=[]
    for worker in routing['workers']:
        name=worker['name'];run=records[name]
        assert worker['status'] in ('complete_collected_audited_deleted','failed_collected_deleted')
        assert run['status']=='DELETED' and 'actual_upper_bound_usd' in run
        items=[]
        for kind in ('instances','disks'):
            path=OUT/f'{name}_{kind}_absence.log';text=path.read_text();assert 'was not found' in text or 'notFound' in text
            items.append(evidence(path))
        audit_path=ROOT/worker['audit_path'];audit=json.loads(audit_path.read_text())
        complete=worker['status']=='complete_collected_audited_deleted';success.append(complete)
        if complete:assert audit['passed'] and audit['complete'] and not audit['failures'] and not audit['pending']
        items.append(evidence(audit_path))
        folder=ROOT/worker['local_artifact_root'];items.extend(evidence(folder/n) for n in ('WORKER_RESULT.json','worker.log'))
        rows.append(dict(condition=worker['condition'],actor_mode=worker['actor_mode'],name=name,zone=worker['zone'],
            instance_absent=True,disk_absent=True,scientific_completed=complete,evidence=items,
            actual_upper_bound_usd=run['actual_upper_bound_usd']))
        worker['actual_upper_bound_usd']=run['actual_upper_bound_usd'];worker['absence_verified_utc']=run['absence_verified_at']
    global_evidence=[]
    for kind in ('instances','disks'):
        result=subprocess.run(['gcloud','compute',kind,'list','--project',routing['project'],
            '--filter=name~^ring-20261105-feedback-','--format=json'],text=True,capture_output=True,check=True,timeout=60)
        assert json.loads(result.stdout)==[]
        path=OUT/f'all_feedback_{kind}_absent.json';path.write_text(result.stdout);global_evidence.append(evidence(path))
    prior=ROOT/'results/local_return_feedback_design/cloud_audit/cleanup_verification.json';old=json.loads(prior.read_text());assert old['complete']
    replacement=sum(x['actual_upper_bound_usd'] for x in rows);combined=replacement+old['attempt_upper_bound_usd'];assert combined<=25
    committed=ledger['ancillary_reserve_usd']+sum(r.get('actual_upper_bound_usd',r['reserved_usd']) for r in ledger['runs'])
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    budget=dict(utc=now,source_ledger_sha256=evidence(ledger_path)['sha256'],original_budget_usd=ledger['budget_usd'],
        ancillary_reserve_usd=ledger['ancillary_reserve_usd'],runs=[{k:r[k] for k in ('name','status','reserved_usd','actual_upper_bound_usd') if k in r} for r in ledger['runs']],
        committed_upper_bound_usd=round(committed,4),remaining_original_budget_usd=round(500-committed,4))
    budget_path=OUT/'budget_at_completion.json';budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    report=dict(complete=True,scientific_completed=all(success),verified_utc=now,protocol_sha256=PROTOCOL_SHA,
        workers=rows,global_evidence=global_evidence,prior_failed_attempt_cleanup=evidence(prior),budget=evidence(budget_path),
        replacement_upper_bound_usd=round(replacement,4),prior_failed_attempt_upper_bound_usd=old['attempt_upper_bound_usd'],
        combined_pilot_upper_bound_usd=round(combined,4),original_budget_usd=500,
        cumulative_committed_upper_bound_usd=round(committed,4),remaining_original_budget_usd=round(500-committed,4))
    target.write_text(json.dumps(report,indent=2)+'\n')
    routing.update(status='complete_collected_audited_deleted' if all(success) else 'terminal_with_failure_collected_deleted',
        finished_utc=now,scientific_completed=all(success),cleanup=evidence(target),combined_pilot_upper_bound_usd=round(combined,4))
    temporary=ROUTING.with_suffix('.tmp');temporary.write_text(json.dumps(routing,indent=2)+'\n');temporary.replace(ROUTING)
    if all(success):subprocess.run([sys.executable,'scripts/compare_feedback_training_streams.py'],cwd=ROOT,check=True)
    print(json.dumps(dict(cleanup_complete=True,scientific_completed=all(success),combined_upper_bound_usd=round(combined,4))))

if __name__=='__main__':main()
