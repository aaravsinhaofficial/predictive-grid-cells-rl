"""Conditional ring tests for the exact fixed local-return training endpoints.

The task, learner, criterion functions and stored policies are never modified.
The entire pilot pair must be complete and audited; its predeclared behavioral
memory gate decides whether these expensive neural tests execute at all.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from code.common import build_policy
from lstm_endogenous_curve_recovery import advance, sha, save
import lstm_continued_heading_memory as geometry
from lstm_endogenous_observable_flow import observable_probe
from lstm_actor_logit_observable import centered_actor_logits
import audit_local_goal_return_training as audit

OUT = ROOT/'results/lstm_local_return_mechanism'
ROUTING = ROOT/'docs/research/local_goal_return_training_launch.json'
CONDITIONS = ('gc100', 'pgc33')
CUES = ('native', 'ordinary', 'plus60', 'minus60')
PHASES = 768
CONTEXTS = (0, 1)
TRAINING_HASH = '0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'


def resolve_checkpoint(condition, routing=ROUTING):
    record = json.loads(Path(routing).read_text())
    assert record['protocol_sha256'] == TRAINING_HASH
    matches = [r for r in record['workers'] if r['condition'] == condition]
    assert len(matches) == 1
    row = matches[0]
    root = Path(row['local_artifact_root']); child = Path(row['artifact_subdir'])
    assert not root.is_absolute() and '..' not in root.parts
    assert not child.is_absolute() and '..' not in child.parts
    assert root.parts[:2] == ('results', 'ring_cloud_artifacts')
    return ROOT/root/child/'final.pt'


def behavioral_gate(reports):
    """A training-integrity pass and a useful-memory pass are distinct."""
    assert set(reports) == set(CONDITIONS)
    for condition, report in reports.items():
        assert report['passed'] and report['complete'] and report['condition'] == condition
        assert report['checks']['final_exact_stage4'] is True
        assert report['checks']['final_summary'] is True
    rows = {c: reports[c]['checks']['stage4_raw_evaluation']['gates'] for c in CONDITIONS}
    return dict(passed=any(row['hard_certified_memory_pass'] for row in rows.values()),
                by_condition=rows, selection='Both fixed final models retained if either passes; no stage selection')


def pair_preflight(routing=ROUTING):
    reports = {}; checkpoints = {}
    for condition in CONDITIONS:
        checkpoint = resolve_checkpoint(condition, routing)
        assert checkpoint.exists(), f'Fixed final endpoint pending: {condition}'
        reports[condition] = audit.run(checkpoint.parent, condition, require_complete=True)
        checkpoints[condition] = dict(path=str(checkpoint), sha256=sha(checkpoint))
    return behavioral_gate(reports), reports, checkpoints


def source_files():
    return tuple(dict.fromkeys((
        'scripts/lstm_local_return_mechanism.py',
        'scripts/lstm_local_return_family.py',
        'scripts/lstm_local_return_causal.py',
        'scripts/audit_local_goal_return_training.py',
        'scripts/audit_local_goal_return_baseline.py',
        'scripts/local_goal_return_training.py',
        'scripts/local_goal_return_training_evaluate.py',
        'scripts/local_goal_return_training_certificate.py',
        'scripts/certify_local_goal_return.py',
        'scripts/lstm_continued_heading_memory.py',
        'scripts/lstm_heading_curve_flow.py',
        'scripts/lstm_endogenous_curve_recovery.py',
        'scripts/lstm_endogenous_observable_flow.py',
        'scripts/lstm_actor_logit_observable.py',
        'scripts/lstm_reorientation_causal_navigation.py',
        'scripts/lstm_travel_cue_pulse.py',
        'scripts/lstm_heading_memory_probe.py',
        'scripts/plot_lstm_heading_memory.py',
        'code/common.py', 'code/env.py', 'code/nets.py', 'code/codes.py',
        'tests/test_lstm_local_return_family.py',
        'tests/test_lstm_local_return_causal.py',
        'tests/test_lstm_local_return_mechanism.py')))


def protocol():
    return dict(version='local_return_task_native_mechanism_v1',
        training_protocol_sha256=TRAINING_HASH,
        design_sha256={name:sha(ROOT/name) for name in (
            'docs/research/local_return_mechanism_design.md',
            'docs/research/local_return_mechanism_design.json')},
        behavior_protocol_sha256=sha(ROOT/'results/local_goal_return_design/behavior_protocol.json'),
        cohort='Original seed1 Grid/PGC feature transfer with identical new heads; both fixed4096-update endpoints and all4stage evaluations required. Not original1B unchanged policies.',
        gate='Before neural outcomes, require the frozen final hard bank at least167/256 intact hits and39/256 postcue erasure cost, all128paired-history certificates, in at leastone population. Retain both models; if neither passes, record that decision without running neural assays.',
        contexts='Sealed hard physical contexts0and1, selected by index before outcomes. Same endpoint/goal/room/pickup/forcedactions/tapes across models. Densify to768actual endpointheading histories; override delay to60within trained finalstage support. Preserve every context including collectionfailure; no replacement.',
        collection='Call unchanged localtask collector with clean state noise0 and read-only LSTM hooks. Start h/czero and exact native goal setup. Capture state immediately before cue, actual input/cuestate, common subsequent input and firstFREEstate. Physical brains remain float32; independent recurrent assays float64. Preserve stored histories, sourcechecks and replayerrors.',
        primary='Unchanged full512(h,c) geometry.analyze_branch in raw and h/cRMS metrics; all4cue branches. ExistingTIMES0,1,5,15,30,45,60; fixedcurve and movingtrajectory/transverse recovery reported distinctly; every original criterion retained.',
        observables='Unchanged observable_probe fromstate15, evolve30thenobserve15; both fixedallhidden1024coordinate and fixedcenteredactor-score24coordinate maps, raw/RMSfullstate kicks and centered/noncentered results. No fit, PCA, selectedneuron subset, whitening, or newthreshold.',
        causal='192fixedsubset headings, actual precue±30deg native-history h/csubstitution, plus±2%radius transverse andisotropic controls inraw/RMSconstructions. Correct/ordinary/±60cue andpostcueerase branches; recipientphysicalstate andallotherinputs unchanged. Nativefloat32 taskcollector controls30FREEactions usingexactsharedbanktapes; fulltrajectory,firsthit,progress,actioncounts retained. Native-navigation stateedits areprecue; later radialrecovery ismeasuredbyunchangedgeometry. Uniform/plannercontrols andnew192casepostcue-resetcertificate useactual60delay/tapeindex; noimplicitreuseofsealed16certificate.',
        timing='Endpointcue atfirstDELAYinput; pre-cue snapshot precedes itsLSTMcall. State0is aftercue. State60has processed firstFREEinput; itsactor output producesfirstFREEaction. Cuefeaturesuse actualprobevelocity beforebrake. Futurecommoninput isblack+ordinaryendpoint+savedgoal+previousLEFT+reward0.',
        repair='Comparecorrectversusordinarylatephaseerror after±30edit, subtractingeachcue ownunperturbederror; pairednavigationrescue likewise subtractsunperturbedcuecontrast. Bothsigns,wrongcue direction,radial/randomcontrols andallcontexts retained. Acuteinputrepair andlearnedrecurrentattraction areseparate measuredcomponents.',
        conditional_observable_repair='If fixedfuture-hidden orfixedactor-scoreobservablequalifies whilefullstatefails, additionallymeasurecue/state-editrepair inTHATSAMEfixedmap at15/30/45anchors withfutureobservationsendingby60. Preserveallfullstate/navresults, allcues/contexts, raw/RMSandcentered/noncentered sensitivity. No newfittedmap orcriterion; thisconditionalfollowupmustbeimplementedandtestedbeforeitsoutcomesareexecuted.',
        numerical_validation='Crossdtypeconsistency usespointwiseatol3e-4+rtol2e-6*abs(nativefloat32state), anengineeringtolerance notanerrorbound. Retainoldabsolute3e-4pass, h/cerrorsandamplitudes, anderrorsnormalizedbydirectionfamilyradius/editmagnitude. Everypositivecandidate mustreproduceunchangedcriteriausingactualnativefloat32LSTMtransitions beforemechanisticclaim. Numericalvalidationfailure isunresolved, nevera negativeringresult; contextretainedwithoutreplacement.',
        interpretation='Pilotnestedcontextsarenotindependenttrainedseeds. Physicalcontextsarematchedacrossmodels; model-specificembeddings/goalcodeandtrainedweights differ. Same-input causalclaims referwithinmodel. Confirmationrequiresfrozenfreshbankandindependentsource-seedpairs beforepopulationclaims.',
        source_sha256={name: sha(ROOT/name) for name in source_files()})


def numerical_consistency(native, double, updates):
    native = np.asarray(native); double = np.asarray(double)
    assert native.shape == double.shape and native.shape[1] == 512
    assert 1 <= updates <= 100
    delta = np.abs(native-double)
    report = dict(pointwise_atol=3e-4, pointwise_rtol=2e-6,
        pointwise_pass=bool(np.all(delta <= 3e-4+2e-6*np.abs(native))),
        absolute_only_3e_minus4_pass=bool(np.max(delta)<3e-4), max_abs_error=float(delta.max()),
        clean_native_updates=updates, c_amplitude_bound_from_zero=float(updates), blocks={})
    for name, part in (('hidden', slice(0, 256)), ('cell', slice(256, 512))):
        report['blocks'][name] = dict(max_abs_error=float(delta[:, part].max()),
            mean_abs_error=float(delta[:, part].mean()), native_max_abs_activity=float(np.abs(native[:,part]).max()))
    assert report['blocks']['hidden']['native_max_abs_activity'] <= 1+1e-6
    assert report['blocks']['cell']['native_max_abs_activity'] <= updates+1e-4
    return report


def analyze_geometry(net, family):
    """Unchanged existing criteria; this adapter only supplies task-native states."""
    net.double()
    starts = family['starts']; dark = family['dark_input'].double().reshape(1, -1)
    theta = family['theta'].numpy().astype(np.float64)
    native = starts['native'].double()
    assert native.shape == (PHASES, 512) and dark.shape == (1, 1287)
    scale = np.repeat([float(native[:, :256].square().mean().sqrt()),
                       float(native[:, 256:].square().mean().sqrt())], 256).clip(1e-8)
    movement = family['movement']
    results = {}; arrays = dict(times=np.array(geometry.TIMES), theta=theta, scale=scale,
        pre_cue_state=family['pre_cue_state'].double().numpy(), ordinary_input=dark.numpy())
    for cue in CUES:
        results[cue], states = geometry.analyze_branch(net, starts[cue].double(), dark, theta, movement, scale)
        arrays[cue+'_states'] = np.stack([states[t] for t in geometry.TIMES])
        arrays[cue+'_glimpse_inputs'] = family['cue_inputs'][cue].double().numpy()
        native_free = family['first_free_states'][cue].double().numpy()
        precision = numerical_consistency(native_free, states[60], family['audit']['first_free_clock']+1)
        results[cue]['native_float32_first_free_replay'] = precision
        results[cue]['native_float32_replay_relative_to_direction_family'] = {}
        for metric, units in (('raw', np.ones(512)), ('relative_rms', scale)):
            reference = states[60]/units
            radius = float(np.sqrt(np.mean(np.sum((reference-reference.mean(0))**2, axis=1))))
            difference = np.linalg.norm((native_free-states[60])/units, axis=1)
            results[cue]['native_float32_replay_relative_to_direction_family'][metric] = dict(
                radius=radius, mean_vector_error=float(difference.mean()), max_vector_error=float(difference.max()),
                mean_error_over_radius=None if radius < 1e-12 else float(difference.mean()/radius),
                max_error_over_radius=None if radius < 1e-12 else float(difference.max()/radius),
                interpretation='Numerical discrepancy relative to the measured directional signal; retain alongside unchanged ring criteria')
        if not precision['pointwise_pass']:
            error = ValueError('Cross-dtype recurrence validation unresolved; retain context, do not count as negative ring evidence')
            error.precision_report = dict(cue=cue, **precision)
            raise error
    if family['condition'] == 'gc100':
        for cue in CUES[1:]:
            assert np.array_equal(arrays['native_states'], arrays[cue+'_states'])
    return results, arrays


def analyze_observables(net, arrays):
    net.double()
    dark = torch.tensor(arrays['ordinary_input'], dtype=torch.float64)
    drive = torch.nn.functional.linear(dark, net.lstm.weight_ih, net.lstm.bias_ih)
    transition = lambda z: advance(net.lstm, drive, z)
    maps = dict(all_hidden=lambda z: z[:, :256],
        learned_action_scores=lambda z: centered_actor_logits(z, net.pi.weight, net.pi.bias))
    report = {}
    for cue in CUES:
        values = arrays[cue+'_states']; start = values[0]
        scale = np.repeat(np.sqrt(np.mean(start.reshape(-1, 2, 256)**2, axis=(0, 2))), 256).clip(1e-8)
        report[cue] = {name: {metric: observable_probe(transition, values[geometry.TIMES.index(15)], units,
            output, lambda z: net.pi(z[:, :256]).softmax(1))
            for metric, units in (('raw_full_state_kicks', np.ones(512)), ('h_c_RMS_full_state_kicks', scale))}
            for name, output in maps.items()}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT); p.add_argument('--routing', type=Path, default=ROUTING)
    p.add_argument('--freeze-only', action='store_true'); p.add_argument('--gate-only', action='store_true')
    p.add_argument('--condition', choices=CONDITIONS); p.add_argument('--context', type=int, choices=CONTEXTS)
    p.add_argument('--mode', choices=('primary', 'observable', 'causal'), default='primary')
    args = p.parse_args(); torch.set_num_threads(2); args.out.mkdir(parents=True, exist_ok=True)
    design = protocol(); path = args.out/'protocol.json'
    if args.freeze_only:
        assert not path.exists(); save(path, dict(design, frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat())); return
    saved = json.loads(path.read_text()); assert saved.pop('frozen_utc'); assert saved == design
    gate, audits, checkpoints = pair_preflight(args.routing)
    save(args.out/'behavior_gate.json', dict(gate=gate, checkpoints=checkpoints,
        audit_summary={c: dict(passed=r['passed'], complete=r['complete'], auditor_sha256=r['auditor_sha256']) for c,r in audits.items()}))
    if args.gate_only or not gate['passed']:
        print(json.dumps(gate), flush=True); return
    assert args.condition is not None and args.context is not None
    checkpoint = Path(checkpoints[args.condition]['path']); digest = checkpoints[args.condition]['sha256']
    data = torch.load(checkpoint, map_location='cpu', weights_only=True); cfg = data['config']
    net = build_policy(cfg, 'cpu').eval().requires_grad_(False); net.load_state_dict(data['policy'])
    folder = args.out/args.condition/f'context{args.context}'; folder.mkdir(parents=True, exist_ok=True)
    from lstm_local_return_family import dense_plan, collect_family
    with threadpool_limits(limits=2):
        if args.mode == 'primary':
            assert not (folder/'primary.json').exists() and not (folder/'family.pt').exists()
            plan, metadata = dense_plan(cfg, args.context)
            torch.save(plan, folder/'input_plan.pt')
            save(folder/'collection_attempt.json', dict(checkpoint_sha256=digest, protocol_sha256=sha(path),
                input_plan_sha256=sha(folder/'input_plan.pt'), plan_metadata=metadata))
            try:
                family = collect_family(net, cfg, plan)
            except Exception as error:
                save(folder/'collection_failed.json', dict(error_type=type(error).__name__, error=str(error),
                    checkpoint_sha256=digest, context=args.context, preserve_context=True, replacement_context=False))
                raise
            family.update(condition=args.condition, checkpoint_sha256=digest, plan_metadata=metadata)
            torch.save(family, folder/'family.pt')
            try:
                result, arrays = analyze_geometry(net, family)
            except Exception as error:
                save(folder/'analysis_failed.json', dict(error_type=type(error).__name__, error=str(error),
                    precision_report=getattr(error, 'precision_report', None), preserve_context=True,
                    counted_as_negative_ring_evidence=False, family_sha256=sha(folder/'family.pt')))
                raise
            arrays.update(checkpoint_sha256=np.array(digest), endpoint=plan['endpoint'][0].numpy(),
                          goal_center=plan['goal'][0].numpy())
            np.savez_compressed(folder/'states.npz', **arrays)
            save(folder/'primary.json', dict(checkpoint_sha256=digest, protocol_sha256=sha(path),
                family_sha256=sha(folder/'family.pt'), states_sha256=sha(folder/'states.npz'),
                context=args.context, condition=args.condition, branches=result,
                collection_audit=family['audit'], plan_metadata=metadata))
        else:
            producer = json.loads((folder/'primary.json').read_text())
            assert producer['checkpoint_sha256'] == digest and producer['protocol_sha256'] == sha(path)
            assert producer['family_sha256'] == sha(folder/'family.pt') and producer['states_sha256'] == sha(folder/'states.npz')
            with np.load(folder/'states.npz') as raw: arrays = {k:raw[k] for k in raw.files}
            target = folder/(args.mode+'.json'); assert not target.exists()
            if args.mode == 'observable':
                save(target, dict(checkpoint_sha256=digest, protocol_sha256=sha(path),
                    states_sha256=producer['states_sha256'], branches=analyze_observables(net, arrays)))
            else:
                from lstm_local_return_causal import analyze_causal
                family = torch.load(folder/'family.pt', map_location='cpu', weights_only=True)
                plan, metadata = dense_plan(cfg, args.context)
                result = analyze_causal(net, cfg, plan, family, arrays, folder/'causal_arrays')
                save(target, dict(checkpoint_sha256=digest, protocol_sha256=sha(path),
                    family_sha256=producer['family_sha256'], states_sha256=producer['states_sha256'], **result))
    assert sha(checkpoint) == digest


if __name__ == '__main__': main()
