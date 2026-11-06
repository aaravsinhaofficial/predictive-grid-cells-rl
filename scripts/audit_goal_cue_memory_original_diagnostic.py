"""Independent saved-output adapter for the exploratory original-only test.

The frozen goal-memory audit supplies all numerical checks unchanged. This file
adds honest exploratory provenance, unchanged original-policy snapshots and the
still-failed paired training qualification. No new trials or training run here.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import audit_goal_cue_memory as base
import goal_cue_memory_bank as banks
from rl_goal_memory import atomic_json

VERSION='goal_cue_memory_original_diagnostic_v1'
PARENT_SHA='0dcd21301c9d9e569bbea7e938a3d387b0be6d5164e2d1da9781bed4ace0cd10'
PROTOCOL=ROOT/'docs/research/goal_cue_memory_original_diagnostic_protocol.json'
sha,read,equal=base.sha,base.read,base.equal


def validate_protocol(path):
    design=read(path);assert design.pop('frozen_utc')
    assert design['version']==VERSION and design['conditions']==['gc100','pgc33'] and design['original_seed']==1
    for name,expected in design['source_sha256'].items():assert sha(base.checked_path(ROOT,name))==expected,name
    for name,expected in design['input_assets'].items():assert sha(base.checked_path(ROOT,name))==expected,name
    stopped=design['stopped_parent_protocol'];parent_path=base.checked_path(ROOT,stopped['path'])
    assert sha(parent_path)==stopped['sha256']==PARENT_SHA
    parent=base.validate_protocol(parent_path)
    assert stopped['paired_training_qualification_passed']is False and stopped['training_remains_prohibited']is True
    assert len(stopped['known_qualification_outcomes'])==2
    for row,(condition,hits)in zip(stopped['known_qualification_outcomes'],[('gc100',181),('pgc33',236)]):
        assert (row['condition'],row['successes'],row['trials'],row['qualified'])==(condition,hits,256,hits>=192)
        audit_path=base.checked_path(ROOT,row['audit_path']);summary_path=base.checked_path(ROOT,row['summary_path'])
        assert str(audit_path.relative_to(ROOT))in design['input_assets'] and str(summary_path.relative_to(ROOT))in design['input_assets']
        record=read(audit_path);summary=read(summary_path)
        assert record['phase']==summary['phase']=='qualification' and record['condition']==summary['condition']==condition
        assert record['passed'] and record['complete'] and not record['failures'] and not record['pending']
        assert record['auditor_sha256']==sha(base.__file__) and record['protocol_sha256']==summary['protocol_sha256']==PARENT_SHA
        totals=summary['totals'];assert (totals['successes'],totals['trials'],totals['valid_trials'],totals['invalid_trials'])==(hits,256,256,0)
        assert summary['qualified']==row['qualified'] and summary['no_optimization'] and summary['complete']
    e=design['evaluation'];p=parent['evaluation']
    for key,value in dict(bank_name='memory',manifest_path=p['manifest_path'],manifest_sha256=p['manifest_sha256'],bank=p['banks']['memory'],
            variants=list(base.VARIANTS),contexts=16,angles_per_context=16,trials_per_variant=256,cue_updates=15,stationary_delay_updates=45,
            free_updates=90,goal_radius_m=.3,goal_distance_m=.75,img_noise=0.,state_noise=0.,native_free_motor_noise_v=.03,native_free_motor_noise_r=.15,
            same_saved_bank_both_models=True,no_variant_or_trial_selection=True).items():assert e[key]==value,key
    execution=design['execution']
    for key,value in dict(device='cpu',threads=2,global_seed=694801,optimizer_created=False,learning_updates=0,cloud_resources=0,
            cloud_reservation_usd=0,preserve_entire_policy_global_rng_and_mode=True).items():assert execution[key]==value,key
    interpretation=design['interpretation']
    for key in ('no_50_percent_certificate','no_automatic_training','no_automatic_neural_nomination','single_original_seed_pair','report_all_controls_and_donor_directional_outcomes'):
        assert interpretation[key]is True,key
    assert interpretation['descriptive_memory_screen']==dict(native_successes=167,erasure_loss=39,trials=256)
    return design


def audit(folder,condition,protocol=PROTOCOL):
    folder=Path(folder);protocol=Path(protocol);checks={};failures=[];pending=[]
    result=dict(version='goal_cue_memory_original_diagnostic_independent_audit_v1',phase='original_diagnostic',condition=condition,
        folder=str(folder.resolve()),protocol_sha256=sha(protocol),auditor_sha256=sha(__file__),
        frozen_numerical_auditor_sha256=sha(base.__file__),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope=base.SCOPE,
        exploratory_after_qualification=True,parent_training_gate_still_failed=True,no_automatic_neural_nomination=True)
    rng=torch.get_rng_state().clone()
    try:
        design=validate_protocol(protocol);assert condition in design['conditions']
        provenance=read(folder/'protocol.json')
        assert provenance['design']==design and provenance['protocol_sha256']==sha(protocol)
        assert provenance['phase']=='original_diagnostic' and provenance['condition']==condition and provenance['original_seed']==1
        assert provenance['device']=='cpu' and provenance['threads']==2 and provenance['parent_training_gate_still_failed']is True
        incoming=torch.load(folder/'incoming.pt',map_location='cpu',weights_only=True)
        assert set(incoming)=={'version','phase','policy','config','initialization','provenance','global_rng','model_training','no_optimization','no_optimizer_created'}
        assert incoming['version']==VERSION and incoming['phase']=='original_diagnostic' and incoming['provenance']==provenance
        assert incoming['no_optimization'] and incoming['no_optimizer_created'] and type(incoming['model_training'])is bool
        initial,checks['initialization']=base.audit_initial(incoming,condition)
        assert equal(incoming['policy'],initial['policy']),'Exploration must use the unchanged original policy'
        assert read(folder/'initialization.json')==incoming['initialization'] and read(folder/'config.json')==incoming['config']
        required=[folder/'after.pt',folder/'evaluation/summary.json',folder/'evaluation_integrity.json',folder/'summary.json']
        pending.extend(str(path)for path in required if not path.exists())
        if not pending:
            after=torch.load(folder/'after.pt',map_location='cpu',weights_only=True)
            assert equal(incoming,after),'Exploratory evaluation changed policy, runtime, RNG or mode'
            e=design['evaluation'];manifest=base.checked_path(ROOT,e['manifest_path'])
            plan,bank,_=banks.load_bank(manifest,'memory');assert bank==e['bank']
            reference=dict(manifest_path=e['manifest_path'],manifest_sha256=e['manifest_sha256'],bank_name='memory',bank_path=bank['path'],bank_sha256=bank['sha256'])
            checked=base.audit_evaluation(folder/'evaluation',incoming,plan,reference,base.VARIANTS)
            checks['evaluation']=checked
            evaluation=read(folder/'evaluation/summary.json')
            integrity=read(folder/'evaluation_integrity.json')
            assert integrity==dict(entire_snapshot_unchanged=True,no_optimizer_created=True,no_optimization=True,
                incoming_sha256=sha(folder/'incoming.pt'),after_sha256=sha(folder/'after.pt'),evaluation_summary_sha256=sha(folder/'evaluation/summary.json'))
            expected=dict(version=VERSION,phase='original_diagnostic',complete=True,condition=condition,protocol_sha256=sha(protocol),
                policy_hash=base.policy_digest(initial['policy']),all_variants_valid=checked['valid'],variant_order=list(base.VARIANTS),
                variants=checked['variants'],evaluator_descriptive_screens=evaluation['gates'],no_automatic_neural_nomination=True,
                parent_training_gate_still_failed=True,no_optimization=True,no_optimizer_created=True,evaluation_summary_sha256=integrity['evaluation_summary_sha256'])
            assert read(folder/'summary.json')==expected,'Exploratory summary or fixed branch set differs'
            checks['descriptive_screens']=checked['gates']
            checks['entire_original_snapshot_unchanged']=True
        if(folder/'incomplete.json').exists():failures.append('Unresolved exploratory execution failure: incomplete.json')
        assert torch.equal(rng,torch.get_rng_state()),'Saved audit consumed global RNG'
    except Exception as error:failures.append(f'{type(error).__name__}: {error}')
    result.update(checks=checks,failures=failures,pending=sorted(set(pending)),passed=not failures,complete=not failures and not pending,
        status='failed'if failures else 'pending'if pending else 'complete',artifacts=base.artifact_manifest(folder))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--condition',choices=['gc100','pgc33'],required=True)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    result=audit(args.folder,args.condition,args.protocol);args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,result)
    print(json.dumps({key:result[key]for key in ('phase','condition','passed','complete','status','failures','pending')}))
    if not result['passed']or(args.require_complete and not result['complete']):raise SystemExit(2)


if __name__=='__main__':main()
