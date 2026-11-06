"""Separately registered exploratory memory tests of unchanged original policies.

Registered after the failed paired training qualification, before any memory-bank
policy outcome. There is no optimizer, learning, cloud launch, training-gate
override or automatic neural nomination in this adapter.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import torch
import goal_cue_memory_run as original
import goal_cue_memory_evaluate as evaluator
from local_return_feedback_evaluate import global_rng_snapshot
from rl_goal_memory import atomic_json, atomic_torch, state_digest
from rl_visual_memory_extend import tree_equal

PROTOCOL = ROOT/'docs/research/goal_cue_memory_original_diagnostic_protocol.json'
PARENT_SHA = '0dcd21301c9d9e569bbea7e938a3d387b0be6d5164e2d1da9781bed4ace0cd10'
QUALIFICATION = ROOT/'results/goal_cue_memory/qualification'
VERSION = 'goal_cue_memory_original_diagnostic_v1'
NEW_SOURCES = ('scripts/goal_cue_memory_original_diagnostic.py',
    'scripts/audit_goal_cue_memory_original_diagnostic.py',
    'tests/test_goal_cue_memory_original_diagnostic.py',
    'tests/test_audit_goal_cue_memory_original_diagnostic.py')
sha = original.sha


def protocol():
    assert sha(original.PROTOCOL) == PARENT_SHA
    parent = json.loads(original.PROTOCOL.read_text()); assert parent.pop('frozen_utc')
    assert parent == original.protocol(), 'Previously frozen computation or assets changed'
    assets = dict(parent['input_assets'])
    assets[str(original.PROTOCOL.relative_to(ROOT))] = PARENT_SHA
    outcomes = []
    for condition, successes in (('gc100',181), ('pgc33',236)):
        audit_path = QUALIFICATION/'audits'/f'{condition}.json'
        summary_path = QUALIFICATION/condition/'summary.json'
        audit = json.loads(audit_path.read_text()); summary = json.loads(summary_path.read_text())
        assert audit['passed'] and audit['complete'] and not audit['failures'] and not audit['pending']
        assert audit['phase'] == summary['phase'] == 'qualification'
        assert audit['condition'] == summary['condition'] == condition
        assert audit['auditor_sha256'] == sha(ROOT/'scripts/audit_goal_cue_memory.py')
        assert audit['protocol_sha256'] == summary['protocol_sha256'] == PARENT_SHA
        assert summary['no_optimization'] and summary['complete']
        totals = summary['totals']
        assert (totals['successes'],totals['trials'],totals['valid_trials'],totals['invalid_trials']) == (successes,256,256,0)
        assert summary['qualified'] == (successes >= 192)
        for path in (audit_path, summary_path): assets[str(path.relative_to(ROOT))] = sha(path)
        outcomes.append(dict(condition=condition,successes=successes,trials=256,
            qualified=summary['qualified'],audit_path=str(audit_path.relative_to(ROOT)),
            summary_path=str(summary_path.relative_to(ROOT))))
    sources = dict(parent['source_sha256'])
    sources.update({name:sha(ROOT/name) for name in NEW_SOURCES})
    return dict(version=VERSION,conditions=['gc100','pgc33'],original_seed=1,
        registration='Exploratory follow-up registered after qualification outcomes181/236, before memory-bank policy outcomes.',
        stopped_parent_protocol=dict(path=str(original.PROTOCOL.relative_to(ROOT)),sha256=PARENT_SHA,
            paired_training_qualification_passed=False,training_remains_prohibited=True,
            known_qualification_outcomes=outcomes),
        initialization='Exact original1B actor/CNN/LSTM plus the same archived fresh critic; critic cannot change policy actions.',
        evaluation=dict(bank_name='memory',manifest_path=parent['evaluation']['manifest_path'],
            manifest_sha256=parent['evaluation']['manifest_sha256'],bank=parent['evaluation']['banks']['memory'],
            variants=list(evaluator.FULL_VARIANTS),contexts=16,angles_per_context=16,trials_per_variant=256,
            cue_updates=15,stationary_delay_updates=45,free_updates=90,
            goal_radius_m=.3,goal_distance_m=.75,img_noise=0.,state_noise=0.,
            native_free_motor_noise_v=.03,native_free_motor_noise_r=.15,
            donor_timing='Genuine same-context incoming h/c swapped immediately before first FREE input; ±90 goal donors; physical recipient goal unchanged.',
            same_saved_bank_both_models=True,no_variant_or_trial_selection=True),
        interpretation=dict(descriptive_memory_screen=dict(native_successes=167,erasure_loss=39,trials=256),
            no_50_percent_certificate=True,no_automatic_training=True,no_automatic_neural_nomination=True,
            single_original_seed_pair=True,report_all_controls_and_donor_directional_outcomes=True),
        execution=dict(device='cpu',threads=2,global_seed=original.TRAIN_SEED,
            optimizer_created=False,learning_updates=0,cloud_resources=0,cloud_reservation_usd=0,
            preserve_entire_policy_global_rng_and_mode=True),
        source_sha256=sources,input_assets=assets)


def snapshot(net,cfg,initialization,provenance):
    return dict(version=VERSION,phase='original_diagnostic',policy=copy.deepcopy(net.state_dict()),
        config=copy.deepcopy(cfg),initialization=copy.deepcopy(initialization),
        provenance=copy.deepcopy(provenance),global_rng=global_rng_snapshot(),
        model_training=net.training,no_optimization=True,no_optimizer_created=True)


def run(net,cfg,initialization,provenance,out,*,manifest):
    out=Path(out)
    before=snapshot(net,cfg,initialization,provenance); atomic_torch(out/'incoming.pt',before)
    try:
        result=evaluator.evaluate_bank(net,cfg,out/'evaluation',bank_name='memory',manifest=manifest,
            device='cpu',variants=evaluator.FULL_VARIANTS)
        after=snapshot(net,cfg,initialization,provenance)
        assert tree_equal(before,after), 'Diagnostic changed original weights, global RNG or mode'
        atomic_torch(out/'after.pt',after)
        assert result['variant_order']==list(evaluator.FULL_VARIANTS)
        assert all(v['totals']['trials']==256 for v in result['variants'].values())
        integrity=dict(entire_snapshot_unchanged=True,no_optimizer_created=True,no_optimization=True,
            incoming_sha256=sha(out/'incoming.pt'),after_sha256=sha(out/'after.pt'),
            evaluation_summary_sha256=sha(out/'evaluation/summary.json'))
        atomic_json(out/'evaluation_integrity.json',integrity)
        summary=dict(version=VERSION,phase='original_diagnostic',complete=True,condition=cfg['cond'],
            protocol_sha256=provenance['protocol_sha256'],policy_hash=state_digest(net),
            all_variants_valid=result['all_variants_valid'],variant_order=result['variant_order'],
            variants={name:v['totals'] for name,v in result['variants'].items()},
            evaluator_descriptive_screens=result['gates'],no_automatic_neural_nomination=True,
            parent_training_gate_still_failed=True,no_optimization=True,no_optimizer_created=True,
            evaluation_summary_sha256=integrity['evaluation_summary_sha256'])
        atomic_json(out/'summary.json',summary)
        return summary
    except Exception as error:
        atomic_json(out/'incomplete.json',dict(phase='original_diagnostic',complete=False,
            error_type=type(error).__name__,error=str(error),no_automatic_retry=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=('gc100','pgc33'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(2); design=protocol()
    if args.freeze_only:
        assert not args.protocol.exists(), 'Never overwrite prospective registration'
        atomic_json(args.protocol,dict(design,frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(json.dumps(dict(path=str(args.protocol),sha256=sha(args.protocol))));return
    frozen=json.loads(args.protocol.read_text()); assert frozen.pop('frozen_utc')
    assert frozen==design, 'Source, inputs or prior-outcome provenance changed after freeze'
    assert args.condition and args.out
    assert not args.out.exists() or not any(args.out.iterdir()), 'Preserve prior attempts; use a fresh output'
    net,cfg,initialization=original.load_initial(args.condition,'cpu')
    torch.manual_seed(original.TRAIN_SEED)
    provenance=dict(phase='original_diagnostic',condition=args.condition,original_seed=1,
        protocol_sha256=sha(args.protocol),design=design,device='cpu',torch_version=str(torch.__version__),
        threads=torch.get_num_threads(),parent_training_gate_still_failed=True)
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/'protocol.json',provenance)
    atomic_json(args.out/'initialization.json',initialization)
    atomic_json(args.out/'config.json',cfg)
    result=run(net,cfg,initialization,provenance,args.out,
        manifest=ROOT/design['evaluation']['manifest_path'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
