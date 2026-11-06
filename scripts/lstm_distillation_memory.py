"""Prespecified initial/final direction-memory assays for native distillation.

Reuse frozen physical families and full-state/fixed-output analysis functions.
Only collection curriculum, checkpoint validation, and the additional 2 s timing
are adapted. No stage checkpoint or behavioral outcome selects an assay case.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from code.common import CodeSource, build_policy
from rl_visual_memory import VisualClock
from rl_goal_memory import state_digest
from native_policy_distillation import frozen_digest
from lstm_heading_memory_probe import sha, save
from lstm_endogenous_curve_recovery import advance
import lstm_continued_heading_memory as heading
import lstm_travel_memory_probe as travel
import lstm_endogenous_observable_flow as observable

OUT = ROOT / 'results/lstm_distillation_memory'
TRAINING = ROOT / 'results/native_policy_distillation'
TRAINING_PROTOCOL = ROOT / 'docs/research/native_policy_distillation_protocol.json'
CONDITIONS = ('gc100', 'pgc33')
ENDPOINTS = ('initial', 'final')
WINDOWS = {
    'within_trained_2s': dict(reference=5, evolution=15, offsets=(0, 3, 6, 10)),
    'legacy_4s_extrapolation': dict(reference=15, evolution=30, offsets=(0, 5, 10, 15)),
}
SOURCES = tuple(dict.fromkeys((*heading.SOURCE_FILES,
    'scripts/lstm_distillation_memory.py', 'scripts/lstm_travel_memory_probe.py',
    'scripts/lstm_endogenous_observable_flow.py', 'scripts/native_policy_distillation.py',
    'scripts/rl_goal_memory.py', 'tests/test_lstm_distillation_memory.py',
    'docs/research/native_policy_distillation_memory_commands.md',
    'tests/test_lstm_continued_heading_memory.py', 'tests/test_lstm_endogenous_observable_flow.py')))
ORIGINAL_ELIGIBLE = heading.eligible_window


class CollectionClock(VisualClock):
    """Actual final curriculum, with an independent current-code interval draw."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The unchanged collector uses seed 98201, and clock seed +2.
        self.substitution_rng = torch.Generator().manual_seed(98205)
        self.swap = torch.zeros_like(self.blind)
        self.swap_after_step = []
        self.blind_decisions = 0
        self.ordinary_decisions = 0

    def advance(self, hit, done, max_gap):
        self.blind_decisions += int(self.mask().sum())
        self.ordinary_decisions += int((self.mask() & self.swap).sum())
        entering = super().advance(hit, done, 30)
        draws = torch.rand(self.n, generator=self.substitution_rng) < .5
        self.swap = torch.where(entering, draws.repeat(self.groups), self.swap) & self.mask()
        self.swap_after_step.append(self.swap.clone())
        return entering


def collection_adapters():
    holder = {}

    class Clock(CollectionClock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder['clock'] = self

    class Source(CodeSource):
        def __call__(self, pos, direction, velocity):
            native = super().__call__(pos, direction, velocity)
            clock = holder['clock']
            ordinary = self._code(pos, direction, velocity, 0.)
            return torch.where((clock.mask() & clock.swap)[:, None], ordinary, native)
        # Inherited .goal() preserves the native pickup code, including PGCs.

    return holder, Clock, Source


def cases():
    return [dict(condition=c, endpoint=e,
                 checkpoint=f'results/native_policy_distillation/{c}_s1/{e}.pt')
            for c in CONDITIONS for e in ENDPOINTS]


def protocol():
    training = json.loads(TRAINING_PROTOCOL.read_text())
    assert training['total_updates'] == 180 and training['gaps'][-1] == 30
    return dict(version='native_distillation_direction_memory_v1', cases=cases(),
        timing='Frozen before final pilot outcomes. Run all four fixed initial/final cases irrespective of behavioral results. No stage selection.',
        collection='Unchanged native collector, seed 98201, 32 environments, first two eligible 15-visible-update windows in distinct slots, maximum 6000 decisions. Final 15-visible/15–30-blind clock; half of each blackout interval uses ordinary current code with private seed 98205. Saved goal remains native. All 15 selected visible inputs are native; record the upcoming interval substitution and cumulative exposure. Teacher is absent during collection.',
        contexts='Collect two heading contexts and separately two travel contexts using the unchanged all-angle wall/goal-clearance eligibility. Fresh actor-specific h/c histories for every checkpoint; never transfer original hidden states into a continued network.',
        paired_weight_comparison='The two INITIAL contexts per family additionally supply complete recorded input prefixes, with verified reset-before masks. Replay each prefix independently through initial and final recurrent weights before identical physical phase preparation and common future inputs. Initial own-context results are the paired baseline; final matched-initial-context results isolate the weight change at those fixed histories. Own-policy final contexts describe its operating regime separately. Prefix/source/config/mask hashes are saved; no state transfer.',
        prefix_numerics='Preserve exact native batch32 contexts. Audit single-slot complete-prefix replay against all15 native h/c rows (max error<3e-4), then consistently use the replayed own-weight context in both initial and final geometry. Hash every one of the15 actual768-phase preparation inputs and require exact paired equality, along with future inputs and physical phase angles.',
        families='Unchanged 768-phase physical heading/turn and 14-forward-update travel preparations; 192 held-out-angle readout samples; 384 uniform full-state arc probes. Heading native/ordinary common inputs. Travel common ordinary-current release, plus phase-specific native-cue trajectory labeled input-driven.',
        full_state='Same frozen analyze_branch/invariance_probe and every threshold. Primary reference15+evolution15=30 updates (2 s); retain original reference30+evolution30=60 (4 s) as extrapolation. Raw and initial h/c RMS metrics both retained.',
        fixed_outputs='Same frozen observable_probe, actual full-state ±2% radial/tangent kicks and every guard. Observe all256 hidden outputs or all6 action-centered learned logits at four fixed future offsets; no fitted projection. Both phase-family-centered and noncentered results retained.',
        output_windows=WINDOWS,
        controls='Known attracting ring with common drift must pass centered output; shrinking plane and silent output fail under the new short timing. General full-state threshold function must reproduce the old30+30 gates exactly. Native collector clock caps at30, interval swaps remain blackout-only, and goal encoding stays native.',
        interpretation='Conditional candidates are screening results. Radial recovery alone does not cover all transverse directions; positive cases require independent broader normal validation and predictive-cue repair before a stabilization claim. Eight contexts per family and repeated windows/metrics are not independent model seeds.',
        source_sha256={name: sha(ROOT / name) for name in SOURCES},
        training_protocol_sha256=sha(TRAINING_PROTOCOL),
        original_asset_sha256={f'models/{c}/{c}_lstm_s1_1000M{suffix}':
            sha(ROOT / f'models/{c}/{c}_lstm_s1_1000M{suffix}')
            for c in CONDITIONS for suffix in ('.pt', '.json')})


def validate_endpoint(data, condition, endpoint, summary, original_policy):
    assert data['config']['cond'] == condition and data['config']['seed'] == 1
    assert data['original_frames'] == 1_000_000_000
    assert data['protocol_sha256'] == sha(TRAINING_PROTOCOL)
    assert summary['complete'] is True and summary['updates'] == 180
    assert summary['frozen_tensors_unchanged'] is True and summary['teacher_unchanged'] is True
    assert data['updates'] == (0 if endpoint == 'initial' else 180)
    assert data.get('extra_frames', 0) == (0 if endpoint == 'initial' else 1_036_800)
    if endpoint == 'initial':
        assert set(data['policy']) == set(original_policy)
        assert all(torch.equal(value, original_policy[key]) for key, value in data['policy'].items())
    else:
        assert data['teacher_policy_hash'] == summary['teacher_policy_hash']
        assert data['source_sha256'] == json.loads(TRAINING_PROTOCOL.read_text())['source_sha256']
    for key, value in data['policy'].items():
        if not key.startswith('lstm.'):
            assert torch.equal(value, original_policy[key]), key


def full_state_gates(result, reference, horizon):
    """The frozen nine tests, with only their predeclared time indices varying."""
    checks = {}
    for metric in ('raw', 'relative_rms'):
        q = result['invariance'][str(reference)][metric]
        g = q['geometry']; r = q['horizons'][str(horizon)]; p = r['phase_transport']
        checks[metric] = dict(
            radius_retains5percent=result['trajectory'][str(reference)][metric]['radius_retention'] >= .05,
            largest_edge_below10percent_radius=g['max_edge_over_radius'] <= .1,
            nonlocal_separation_above5percent_radius=g['nonlocal_distance_over_radius']['p10'] >= .05,
            fixed_curve_residual_p90_below5percent_radius=r['residual_to_fixed_curve_over_radius']['p90'] <= .05,
            phase_coverage80percent=p['circular_coverage'] >= .8,
            phase_map_degree_one=abs(p['winding_degree'] - 1) < .1,
            phase_order_preserved=p['backward_edge_fraction'] <= .05,
            radial_gain_below_half=r['radial_normal_gain']['median'] <= .5,
            radial_over_tangent_below_half=r['radial_over_tangent_gain']['median'] <= .5)
    return checks


def centered_actor(state, net):
    logits = net.pi(state[:, :256])
    return logits - logits.mean(1, keepdim=True)


def prefix_hash(inputs, reset_before):
    digest = hashlib.sha256()
    for array in (inputs, reset_before):
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode()); digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def replay_prefix(net, inputs, reset_before, device):
    assert len(inputs) == len(reset_before)
    state = net.zero_state(1, device)
    for x, reset in zip(inputs, reset_before):
        if bool(reset):
            state = net.zero_state(1, device)
        state = net.lstm(torch.as_tensor(x[None], device=device), state)
    return state


@torch.no_grad()
def replay_context(net, source, checkpoint, device):
    context = {key: np.array(value, copy=True) for key, value in source.items()}
    assert prefix_hash(context['prefix_inputs'], context['prefix_reset_before']) == str(context['prefix_sha256'])
    state = replay_prefix(net, context['prefix_inputs'], context['prefix_reset_before'], device)
    before = []; after = []
    for x in context['history_x']:
        before.append(torch.cat(state, 1)[0].cpu().numpy().copy())
        state = net.lstm(torch.as_tensor(x[None], device=device), state)
        after.append(torch.cat(state, 1)[0].cpu().numpy().copy())
    before = np.stack(before); after = np.stack(after)
    context['history_h_before'], context['history_c_before'] = before[:, :256], before[:, 256:]
    context['history_h_after'], context['history_c_after'] = after[:, :256], after[:, 256:]
    context['prefix_source_checkpoint_sha256'] = source['checkpoint_sha256']
    context['checkpoint_sha256'] = np.array(sha(checkpoint))
    context['own_weights_replayed_full_prefix'] = np.array(True)
    return context


def observable_windows(net, archive, variants):
    results = {}
    for name, state_key, input_key in variants:
        states = archive[state_key].astype(np.float64)
        scale = np.repeat(np.sqrt(np.mean(states[0].reshape(-1, 2, 256)**2,
                                         axis=(0, 2))), 256).clip(1e-8)
        x = torch.tensor(archive[input_key], dtype=torch.float64)
        assert x.shape == (1, 1287), 'Require one identical future input for every phase'
        drive = torch.nn.functional.linear(x, net.lstm.weight_ih, net.lstm.bias_ih)
        transition = lambda z: advance(net.lstm, drive, z)
        # Check the entire saved common-input trajectory before perturbations.
        z = torch.tensor(states[0], dtype=torch.float64); errors = {}
        for update in range(61):
            if update in archive['times']:
                index = list(archive['times']).index(update)
                errors[str(update)] = float(np.max(np.abs(z.numpy() - states[index])))
            if update < 60:
                z = transition(z)
        assert max(errors.values()) < 1e-10
        branch = dict(replay_max_errors=errors, windows={})
        for label, timing in WINDOWS.items():
            curve = states[list(archive['times']).index(timing['reference'])]
            with patch.object(observable, 'OBSERVATIONS', timing['offsets']), \
                 patch.object(observable, 'EVOLUTION', timing['evolution']):
                branch['windows'][label] = {
                    output_name: {metric: observable.observable_probe(
                        transition, curve, units, output,
                        lambda s: net.pi(s[:, :256]).softmax(1))
                        for metric, units in [('raw', np.ones(512)), ('relative_rms', scale)]}
                    for output_name, output in [('all_hidden', lambda s: s[:, :256]),
                                                ('centered_actor_logits', lambda s: centered_actor(s, net))]}
        results[name] = branch
    return results


def collect(net, cfg, checkpoint, args, family):
    holder, Clock, Source = collection_adapters()
    recorded_inputs = []; recorded_resets = []; recorded_blind = []; recorded_swap = []
    def record_prefix(module, arguments):
        x, state = arguments
        # The frozen collector also does batch-one post-collection replay audits.
        if len(x) != 32:
            return
        reset = (x[:, 1280:1286] == 0).all(1)
        assert all(bool((z[reset] == 0).all()) for z in state)
        recorded_inputs.append(x.detach().cpu().numpy().copy())
        recorded_resets.append(reset.cpu().numpy().copy())
        recorded_blind.append(holder['clock'].mask().cpu().numpy().copy())
        recorded_swap.append((holder['clock'].mask() & holder['clock'].swap).cpu().numpy().copy())
    def eligible(window, slot, radius):
        valid, why = ORIGINAL_ELIGIBLE(window, slot, radius)
        if not valid or family == 'heading':
            return valid, why
        clear = travel.path_clearance(window[0]['pos'][slot], window[0]['goal_center'][slot], cfg)
        return (True, 'accepted') if clear['eligible'] else (False, 'all_angle_forward_path_clearance')
    hook = net.lstm.register_forward_pre_hook(record_prefix)
    try:
        with patch.object(heading, 'VisualClock', Clock), patch.object(heading, 'CodeSource', Source), \
             patch.object(heading, 'eligible_window', eligible):
            contexts = heading.collect_contexts(net, cfg, checkpoint, args)
    finally:
        hook.remove()
    clock = holder['clock']
    for index, context in enumerate(contexts):
        slot = int(context['history_slot']); step = int(context['collector_step'])
        context['upcoming_blackout_ordinary_current'] = np.array(bool(clock.swap_after_step[step][slot]))
        context['visible_history_current_substitution'] = np.zeros(15, dtype=bool)
        first_visible = step - 14
        assert np.array_equal(np.stack(recorded_inputs[first_visible:step + 1])[:, slot], context['history_x'])
        context['prefix_inputs'] = np.stack(recorded_inputs[:first_visible])[:, slot]
        context['prefix_reset_before'] = np.stack(recorded_resets[:first_visible])[:, slot]
        context['prefix_blind'] = np.stack(recorded_blind[:first_visible])[:, slot]
        context['prefix_current_ordinary'] = np.stack(recorded_swap[:first_visible])[:, slot]
        context['prefix_sha256'] = np.array(prefix_hash(context['prefix_inputs'], context['prefix_reset_before']))
        context['prefix_mask_sha256'] = np.array(prefix_hash(context['prefix_blind'], context['prefix_current_ordinary']))
        context['config_sha256'] = np.array(sha(ROOT / f'models/{cfg["cond"]}/{cfg["cond"]}_lstm_s1_1000M.json'))
        replayed = replay_context(net, context, checkpoint, args.device)
        error = max(float(np.max(np.abs(replayed[key] - context[key]))) for key in (
            'history_h_before', 'history_c_before', 'history_h_after', 'history_c_after'))
        assert error < 3e-4, error
        context['complete_prefix_replay_max_error'] = np.array(error)
        np.savez_compressed(args.out / 'contexts' / f'context{index}.npz', **context)
    metadata_path = args.out / 'collection.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(schedule='15 visible updates,15–30 blind updates after actual pickup; half of blackout intervals use ordinary CURRENT code. Native goal preserved. Actual actor actions; no teacher.',
                    substitution_seed=98205, ordinary_current_decisions=clock.ordinary_decisions,
                    black_decisions=clock.blind_decisions,
                    selected_upcoming_substitutions=[bool(c['upcoming_blackout_ordinary_current']) for c in contexts],
                    adapter_source_sha256=sha(__file__))
    save(metadata_path, metadata)
    return contexts


def run(condition, endpoint, out, device):
    checkpoint = TRAINING / f'{condition}_s1' / f'{endpoint}.pt'
    # Both complete endpoints must exist before any initial/final assay starts.
    for c in CONDITIONS:
        assert json.loads((TRAINING / f'{c}_s1' / 'summary.json').read_text())['complete'] is True
    summary = json.loads(checkpoint.with_name('summary.json').read_text())
    assert sha(checkpoint) == summary['artifacts'][checkpoint.name]['sha256']
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    original = torch.load(ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.pt',
                          map_location='cpu', weights_only=True)
    validate_endpoint(data, condition, endpoint, summary, original['policy'])
    cfg = data['config']; folder = out / condition / endpoint
    assert cfg == json.loads((ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    assert cfg['action_repeat'] == 4 and cfg['nh'] == 256 and cfg['extra_input'] == 'none'
    if endpoint == 'final':
        initial_summary = json.loads((out / condition / 'initial' / 'summary.json').read_text())
        assert initial_summary['complete'] and initial_summary['contexts'] == 4
        assert initial_summary['protocol_sha256'] == sha(out / 'protocol.json')
    assert not folder.exists(), 'Keep completed and interrupted evidence; do not overwrite'
    folder.mkdir(parents=True)
    net = build_policy(cfg, device).eval().requires_grad_(False)
    net.load_state_dict(data['policy'])
    digest = state_digest(net)
    assert digest == summary['teacher_policy_hash' if endpoint == 'initial' else 'student_final_policy_hash']
    assert frozen_digest(net) == summary['frozen_tensor_hash']
    save(folder / 'checkpoint_audit.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        summary_sha256=sha(checkpoint.with_name('summary.json')), protocol_sha256=sha(out / 'protocol.json'),
        fixed_tensors_exact=True, policy_hash=digest, condition=condition, endpoint=endpoint))
    results = []
    regimes = ('own_policy',) if endpoint == 'initial' else ('own_policy', 'matched_initial_contexts')
    for regime, family in ((r, f) for r in regimes for f in ('heading', 'travel')):
        destination = folder / family if regime == 'own_policy' else folder / regime / family
        args = SimpleNamespace(device=device, out=destination, smoke_only=False,
                               cohort='native_distillation_' + endpoint)
        args.out.mkdir(parents=True)
        if regime == 'own_policy':
            sources = collect(net, cfg, checkpoint, args, family)
            source_paths = [args.out / 'contexts' / f'context{i}.npz' for i in range(2)]
        else:
            source_paths = [out / condition / 'initial' / family / 'contexts' / f'context{i}.npz' for i in range(2)]
            sources = []
            for path in source_paths:
                with np.load(path) as a:
                    sources.append({key: a[key] for key in a.files})
        replay_folder = args.out / 'replayed_contexts'; replay_folder.mkdir()
        for index, source in enumerate(sources):
            assert str(source['config_sha256']) == sha(ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.json')
            assert str(source['prefix_mask_sha256']) == prefix_hash(source['prefix_blind'], source['prefix_current_ordinary'])
            context = replay_context(net, source, checkpoint, device)
            np.savez_compressed(replay_folder / f'context{index}.npz', **context)
            preparation_hashes = []
            def capture_factory(*factory_args, **factory_kwargs):
                prepared = build_policy(*factory_args, **factory_kwargs)
                def capture(module, arguments):
                    x = arguments[0]
                    if len(x) == 768:
                        preparation_hashes.append(hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                prepared.lstm.register_forward_pre_hook(capture)
                return prepared
            if family == 'heading':
                with patch.object(heading, 'build_policy', capture_factory):
                    result = heading.analyze_context(data, cfg, context, index, args)
                path = args.out / f'context{index}_states.npz'
                variants = [('native_current', 'native_states', 'native_input'),
                            ('ordinary_current', 'ordinary_states', 'ordinary_input')]
                full_branches = result['branches']
            else:
                with patch.object(travel, 'build_policy', capture_factory):
                    result = travel.analyze(dict(checkpoint=str(checkpoint)), context, index, args)
                path = args.out / f'{condition}_s1' / f'context{index}_states.npz'
                variants = [('ordinary_current_release', 'states', 'ordinary_input')]
                full_branches = {'ordinary_current_release': result['ordinary_release']}
            with np.load(path) as a:
                archive = {key: a[key] for key in a.files}
            assert len(preparation_hashes) == 15
            paired_audit = None
            if regime == 'matched_initial_contexts':
                reference = out / condition / 'initial' / family
                reference = reference / (f'context{index}_states.npz' if family == 'heading' else f'{condition}_s1/context{index}_states.npz')
                with np.load(reference) as a:
                    equal_keys = ('theta', 'native_input', 'ordinary_input') if family == 'heading' else ('theta', 'native_inputs', 'ordinary_input')
                    assert all(np.array_equal(archive[key], a[key]) for key in equal_keys)
                    assert preparation_hashes == list(a['preparation_input_sha256'])
                initial_record = json.loads((out / condition / 'initial' / family / f'context{index}_adapted.json').read_text())
                assert initial_record['archive_sha256'] == sha(reference)
                assert initial_record['context_source_sha256'] == sha(source_paths[index])
                assert initial_record['prefix_sha256'] == str(context['prefix_sha256'])
                paired_audit = dict(initial_archive=str(reference), initial_archive_sha256=sha(reference),
                    inputs_and_angles_byte_identical=True, all15_preparation_inputs_byte_identical=True,
                    identical_prefix_sha256=str(context['prefix_sha256']),
                    initial_hc_never_transferred=True)
            archive.update(checkpoint_sha256=np.array(sha(checkpoint)),
                           endpoint=np.array(endpoint), regime=np.array(regime),
                           prefix_sha256=context['prefix_sha256'], prefix_mask_sha256=context['prefix_mask_sha256'],
                           context_source_sha256=np.array(sha(source_paths[index])),
                           preparation_input_sha256=np.array(preparation_hashes),
                           wrapper_protocol_sha256=np.array(sha(out / 'protocol.json')))
            np.savez_compressed(path, **archive)
            analysis_net = build_policy(cfg, 'cpu').eval().requires_grad_(False)
            analysis_net.load_state_dict(data['policy']); analysis_net.double()
            with torch.no_grad():
                outputs = observable_windows(analysis_net, archive, variants)
            full = {}
            for name, branch in full_branches.items():
                full[name] = {label: full_state_gates(branch, reference, horizon)
                              for label, reference, horizon in [
                                  ('within_trained_2s', 15, 15), ('legacy_4s_extrapolation', 30, 30)]}
            record = dict(condition=condition, endpoint=endpoint, family=family, context=index, regime=regime,
                checkpoint_sha256=sha(checkpoint), archive_sha256=sha(path),
                context_source_sha256=sha(source_paths[index]), prefix_sha256=str(context['prefix_sha256']),
                prefix_mask_sha256=str(context['prefix_mask_sha256']),
                source_native_prefix_replay_max_error=float(source['complete_prefix_replay_max_error']),
                preparation_input_sha256=preparation_hashes,
                paired_comparison=paired_audit,
                upcoming_context_substitution=bool(context['upcoming_blackout_ordinary_current']),
                all_visible_preparation_inputs_native=True,
                full_state_candidate_gates=full, fixed_observable=outputs,
                original_function_result_role='Original saved full-state fields use the legacy4s timing; wrapper explicitly separates the2s primary screen.',
                source_sha256=sha(__file__), protocol_sha256=sha(out / 'protocol.json'))
            save(args.out / f'context{index}_adapted.json', record)
            results.append(dict(family=family, context=index, regime=regime,
                full_state_candidates={name: {w: all(all(g.values()) for g in metrics.values())
                    for w, metrics in windows.items()} for name, windows in full.items()},
                observable_candidates={name: {w: {o: {m: dict(
                    centered=r['centered_conditional_factor_candidate'],
                    noncentered=r['noncentered_observable_candidate']) for m, r in metrics.items()}
                    for o, metrics in output.items()} for w, output in branch['windows'].items()}
                    for name, branch in outputs.items()}))
    assert state_digest(net) == digest
    save(folder / 'summary.json', dict(complete=True, condition=condition, endpoint=endpoint,
        contexts=len(results), results=results, checkpoint_sha256=sha(checkpoint),
        protocol_sha256=sha(out / 'protocol.json'), no_cloud=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--endpoint', choices=ENDPOINTS)
    parser.add_argument('--device', default='mps', choices=('cpu', 'mps'))
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    design = protocol(); path = args.out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(design)), 'Frozen source/protocol changed'
    elif args.freeze_only:
        args.out.mkdir(parents=True, exist_ok=True); save(path, design)
    else:
        raise RuntimeError('Review and freeze before any new analysis outcomes')
    if args.freeze_only:
        return
    assert args.condition and args.endpoint
    with threadpool_limits(limits=2):
        run(args.condition, args.endpoint, args.out, args.device)


if __name__ == '__main__':
    main()
