"""Implementation-only full-architecture state-swap smoke.

Uses exact stored untrained local-task initializers and one private physical
context, never sealed evaluation contexts or learned endpoints. Reports no
success, phase-decoding, geometry or candidate results.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import local_goal_return_training as task
from local_goal_return_training_run import load_baseline_policy
from lstm_local_return_family import collect_family,slice_plan,_rng,_same_rng
from lstm_local_return_causal import collect_intervention,CUES,TIMES
from rl_goal_memory import state_digest
from lstm_endogenous_curve_recovery import sha

SOURCES=('scripts/check_local_return_native_causal.py','scripts/lstm_local_return_family.py',
    'scripts/lstm_local_return_causal.py','scripts/local_goal_return_training.py',
    'scripts/local_goal_return_training_run.py','code/env.py','code/nets.py','code/common.py','code/codes.py')


@torch.inference_mode()
def check_condition(condition):
    net,cfg,initialization=load_baseline_policy(condition,'cpu');net.eval().requires_grad_(False)
    checkpoint=ROOT/f'results/local_goal_return_baseline/{condition}/initialized_policy.pt';digest=sha(checkpoint)
    policy_digest=state_digest(net);rng_before=_rng()
    plan=task.TrialStreams(781931).draw(1,task.STAGES[3],cfg,headings=96)
    original_delay=int(plan['delay'][0]);plan['delay'].fill_(60)
    family=collect_family(net,cfg,plan,chunk_size=48)
    schedule=task.schedule(plan,'cpu');free=int(schedule['free'][0])
    records=[]
    for begin in range(0,96,32):
        selected=torch.arange(begin,begin+32);donors=(selected+8)%96
        chunk=slice_plan(plan,selected);expected=family['pre_cue_state'][selected]
        edited=family['pre_cue_state'][donors]
        cues={cue:family['cue_inputs'][cue][selected] for cue in CUES[:-1]}
        _,sham_checks,sham=collect_intervention(net,cfg,chunk,expected,expected_cues=cues)
        _,edit_checks,edit=collect_intervention(net,cfg,chunk,expected,edited,expected_cues=cues)
        assert torch.equal(sham['cue_inputs'],edit['cue_inputs'])
        for field in ('position','action','raw_reward','active','phase','pa','pr'):
            assert torch.equal(sham[field][:free],edit[field][:free]),field
        assert torch.equal(sham['goal'],edit['goal'])
        for t in TIMES:assert torch.equal(sham['snapshots'][t][-32:],edit['snapshots'][t][-32:])
        for field in ('action','position','raw_reward','active','terminal'):
            assert torch.equal(sham[field][:,-32:],edit[field][:,-32:]),field
        for i in range(160):
            for field in ('input','position','heading','motors'):
                assert torch.equal(sham['first_free'][i][field],edit['first_free'][i][field]),field
            assert torch.equal(sham['snapshots'][60][i],sham['first_free'][i]['state'])
            assert torch.equal(edit['snapshots'][60][i],edit['first_free'][i]['state'])
        actor_errors={}
        for label,raw in (('none',sham),('plus30',edit)):
            h=raw['snapshots'][60][:,:256].contiguous()
            expected_logits=net.pi(h)
            actual_logits=torch.stack([raw['first_free'][i]['logits'] for i in range(160)])
            actor_errors[label]=float((expected_logits-actual_logits).abs().max())
            assert torch.equal(expected_logits,actual_logits), 'Actor did not consume captured firstFREE state'
        replay={}
        for group,cue in enumerate(CUES[:-1]):
            actual=sham['snapshots'][60][group*32:(group+1)*32]
            reference=family['first_free_states'][cue][selected]
            difference=(actual-reference).abs()
            assert bool((difference<=3e-4+2e-6*reference.abs()).all())
            replay[cue]=float(difference.max())
        delta=edited-expected
        assert torch.equal(edit['intended_precue_edit'],delta.repeat(5,1))
        assert not net.lstm._forward_hooks and not net.lstm._forward_pre_hooks
        records.append(dict(phase_ids=selected.tolist(),donor_phase_ids=donors.tolist(),
            collected_shape=[160,512],cue_input_shape=list(sham['cue_inputs'].shape),
            precue_history_replay_max_abs_error={kind:record['precue_history_replay_max_abs_error']
                for kind,record in (('none',sham_checks),('plus30',edit_checks))},
            cue_input_replay_max_abs_error={kind:record['native_cue_input_replay_max_abs_error']
                for kind,record in (('none',sham_checks),('plus30',edit_checks))},
            first_free_family_batch_shape_replay_max_abs_error=replay,
            first_free_actor_reconstruction_max_abs_error=actor_errors,
            intended_plus30_donor_state_substitution_exact=True,
            recipient_cue_goal_and_pre_free_physics_unchanged=True,
            postcue_reset_states_and_actions_identical_after_swap=True,
            first_free_single_processing_exact=True,
            no_lingering_hooks=True))
        del sham,edit
        print(json.dumps(dict(condition=condition,chunk_start=begin,implementation_checks_passed=True)),flush=True)
    assert state_digest(net)==policy_digest==initialization['full_initialized_policy_hash']
    assert sha(checkpoint)==digest and _same_rng(rng_before,_rng())
    return dict(condition=condition,checkpoint_sha256=digest,full_initialized_policy_hash=policy_digest,
        actual_native_architecture=True,phases=96,family_chunk_size=48,causal_chunk_size=32,
        causal_variants=5,original_private_delay=original_delay,delay_override=60,
        cue_clock=int(schedule['glimpse'][0]),first_free_clock=free,
        full_state_shape=list(family['pre_cue_state'].shape),checks=records,
        checkpoint_policy_and_global_rng_unchanged=True,clean_state_noise_sigma=0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/local_goal_return_design/native_causal_validation.json')
    args=parser.parse_args();assert not args.out.exists();torch.set_num_threads(2)
    sources={name:sha(ROOT/name) for name in SOURCES};rows=[]
    try:
        for condition in ('gc100','pgc33'):rows.append(check_condition(condition))
        assert sources=={name:sha(ROOT/name) for name in SOURCES}
        result=dict(passed=True,rows=rows,private_physical_seed=781931,no_sealed_bank=True,
            optimizer_updates=0,trained_endpoint_runs=0,geometry_or_decoder_tests=0,
            success_metrics_reported=False,source_sha256=sources,torch_version=str(torch.__version__),
            checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scope='Implementation validation only: nativeCNN/LSTM, actual task timing and physical separation, two numerical batch shapes, observer and actual+30 donorstate substitution. No learned-model or ring outcomes.')
    except Exception as error:
        result=dict(passed=False,rows=rows,source_sha256=sources,error_type=type(error).__name__,error=str(error))
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');raise
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,models=len(rows),output=str(args.out))),flush=True)


if __name__=='__main__':main()
