"""Seal prospective local-return behavior inputs/gates without policy execution."""
from __future__ import annotations
import datetime as dt
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import local_goal_return_training as task

OUT=ROOT/'results/local_goal_return_design'
CONFIG=ROOT/'models/gc100/gc100_lstm_s1_1000M.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha(value):return hashlib.sha256(value.encode()).hexdigest()


def validate_plan(plan,stage):
    assert plan['n_contexts']==16 and plan['headings']==16 and plan['batch']==0
    assert plan['stage']==dict(low=stage.low,high=stage.high,delay=stage.delay)
    for key in ('endpoint','goal','radius','pickup_angle','duration_a','duration_b',
                'turn_a','turn_b','delay','land_pos','stripe_phase','context'):
        rows=plan[key].reshape(16,16,*plan[key].shape[1:])
        assert torch.equal(rows,rows[:,:1].expand_as(rows)),key
    assert torch.equal(plan['heading_index'],torch.arange(16).repeat(16))
    head=plan['heading'].reshape(16,16)
    assert torch.max(torch.abs(head[:,8:]-head[:,:8]-math.pi))<1e-5
    for key,axis in (('motor',2),('action',1),('state',2)):
        value=plan[key].movedim(axis,0).reshape(16,16,*plan[key].movedim(axis,0).shape[1:])
        assert torch.equal(value,value[:,:1].expand_as(value)),key
    assert plan['motor'].shape==(160,3,256)
    assert plan['action'].shape==(160,256)
    assert plan['state'].shape==(160,2,256,256)
    assert all(torch.isfinite(value).all() for value in plan.values() if isinstance(value,torch.Tensor))
    assert ((plan['action']>=0)&(plan['action']<1)).all()


def tensor_manifest(plan):
    return {key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=task.tensor_hash(value))
            for key,value in plan.items() if isinstance(value,torch.Tensor)}


def main():
    OUT.mkdir(parents=True,exist_ok=True);bankdir=OUT/'banks';bankdir.mkdir(exist_ok=True)
    assert not (OUT/'behavior_protocol.json').exists(),'Preserve the frozen protocol'
    cfg=json.loads(CONFIG.read_text());initial_global_rng=torch.get_rng_state().clone()
    source_before=sha(Path(task.__file__))
    expected_stages=[dict(low=.36,high=.40,delay=4),dict(low=.44,high=.48,delay=15),
                     dict(low=.52,high=.56,delay=30),dict(low=.58,high=.62,delay=60)]
    assert [dict(low=s.low,high=s.high,delay=s.delay) for s in task.STAGES]==expected_stages
    assert task.HEAD_SEED==291801
    banks={}
    for name,seed,index in [('near_baseline',391801,0),('final_hard',391901,3)]:
        target=bankdir/f'{name}.pt';assert not target.exists()
        streams=task.TrialStreams(seed);plan=streams.draw(16,task.STAGES[index],cfg,headings=16)
        validate_plan(plan,task.STAGES[index]);torch.save(plan,target)
        loaded=torch.load(target,map_location='cpu',weights_only=True)
        assert tensor_manifest(loaded)==tensor_manifest(plan)
        banks[name]=dict(path=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size,
            seed=seed,stage_index=index,stage=plan['stage'],contexts=16,headings_per_context=16,
            trials=256,antipodal_pairs=128,tensors=tensor_manifest(plan),
            non_tensor_fields={key:value for key,value in plan.items() if not isinstance(value,torch.Tensor)})
    assert torch.equal(torch.get_rng_state(),initial_global_rng)
    assert sha(Path(task.__file__))==source_before,'Generator module changed during draw'
    protocol=dict(version='prospective_local_return_behavior_banks_v1',
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        no_policy_execution=True,no_environment_execution=True,no_training=True,
        torch_version=str(torch.__version__),generator_backend='cpu',
        source_sha256=sha(__file__),generator_file_at_freeze_sha256=source_before,
        generator_scope='Whole task module is still under implementation review; its historical hash is provenance. Class source, Stage source, explicit stage values and sealed tensor bytes below are immutable bank definitions.',
        generator_class_sha256=text_sha(inspect.getsource(task.TrialStreams)),
        stage_class_sha256=text_sha(inspect.getsource(task.Stage)),stages=expected_stages,
        config_path=str(CONFIG.relative_to(ROOT)),config_sha256=sha(CONFIG),
        native_physics={key:cfg[key] for key in ('arena_half','action_repeat','v_max','rot_max','goal_radius','nh')},
        physical_defaults=dict(tau=.25,noise_v=.03,noise_r=.15,wall_margin=.08),
        banks=banks,clean_evaluation_state_noise_sigma=0.,
        tapes='Full160-update clocks; motor/action/state draws repeated identically across16headings withincontext and reused across cue/state/population branches. Store state draws but clean evaluation sigma iszero. Draws do not depend on earlier hits. Contexts remain distinct independent physical draws.',
        initialization=dict(source='Original1B checkpoint for each population/source seed',
            head_seed_for_pilot=291801,head_seed_formula='291800 + original_model_seed',
            actor_orthogonal_gain=.01,critic_orthogonal_gain=1.,biases_zero=True,
            exact_nonhead_tensor_preservation=True,all_parameters_trainable=True,
            terminology='Transfer originalCNN/LSTM/spatial features; action/value heads explicitly reinitialized; not unchanged-policy continuation'),
        baseline_gate=dict(bank='near_baseline',models=['gc100_s1','pgc33_s1'],variant='blind_native',
            denominator_per_model=256,minimum_success_fraction=.05,minimum_successes_per_model=13,
            every_population_required=True,maximum_forced_rewards=0,all_physical_checks_required=True,
            incomplete_or_invalid_trial='Retain in256 denominator asfailure; do not replace after seeing actions',
            controls=['uniform_six_actions','heading_aware_analysis_planner'],
            recording='Allfirsthits/failures,actualinitialFREEentropy/sixactionprobabilities,freeactions/path/progress,setup/forcedrewards,endpointandangularcoverage'),
        final_behavior_gate=dict(bank='final_hard',denominator_per_model=256,
            intact_variant='blind_native',erasure_variant='blind_reset_after_glimpse',
            minimum_intact_success_fraction=.65,minimum_intact_successes=167,
            minimum_erasure_cost_fraction=.15,minimum_paired_success_count_difference=39,
            qualifier='Descriptivepilot gate,not a significance test. Atleastonepopulation must qualify before expensive dynamics; bothmodels/allfixedendpoints retained. IfonlyGridqualifies,retain this outcome.',
            no_qualifying_model='Stop automatic geometry/extension; inspect learning and report fixedpilot failure'),
        certificate=dict(helper='scripts/certify_local_goal_return.py::certify',
            source_sha256=sha(ROOT/'scripts/certify_local_goal_return.py'),
            contexts=16,pairs_per_context=8,all_pairs_required_for_special_50_percent_claim=True,
            aggregate_bound='.5 * certified_pair_fraction + 1 * uncertified_pair_fraction',
            failure='Retain everyfailedpair and its reason; no outcome-based replacement; report bound rather than claiming50percent foruncertified data',
            behavioral_gate_scope='Special certified-memory gate additionally requires all128pairs certified. Numeric65percent/15pp indicators remain reported regardless.',
            producer_checks='ActualsameFREEinputs/goal/erasedcontrollerstate/bodymotors/position; antipodalheadings; commonBODYmotor/actiontapes; nativeparameters; heldcurrent/blackRGB/no-prehitfeedback'),
        variants=list(task.VARIANTS),
        training_accounting=dict(stages=4,batches_per_stage=1024,trials_per_batch=32,
            total_trials_per_model=131072,physical_frames='4 * activephysicalupdates; exclude padding/setup/warmup',
            primary_endpoint='Fixedlastbatch,allintermediatecheckpointsretained'),
        causal_next_step='Afterbehaviorgate,±30actualhistory-state substitutions immediatelyBEFORE endpointcue,recipientphysicalstatepreserved;native/ordinary/wrong±60cueonly169slots; regenerateexact30degreehistories orfreezeadensefamily,not22.5degreeneighborrelabeling; persistentrepairaftercuewithdrawal plusmovement,radial/randomcontrols',
        replication='Seed1pilot;seed2development;freezemethodsbeforeconfirmationseeds3–8;allconfirmationmodelsretained,trialscollapsedwithinmodel',
        review='docs/research/local_goal_return_study_review.md')
    (OUT/'behavior_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(protocol_sha256=sha(OUT/'behavior_protocol.json'),banks={key:value['sha256'] for key,value in banks.items()},no_policy_execution=True)))


if __name__=='__main__':main()
