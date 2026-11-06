"""Task-native state substitutions before the predictive endpoint cue.

All physical steps, action sampling, goal termination and available inputs use
the unchanged local-return collector. A scoped LSTM pre-hook performs the sole
causal change. Direction estimates are offline and never enter the policy.
"""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
import local_goal_return_training as task
from lstm_endogenous_curve_recovery import sha
from lstm_heading_curve_flow import curve_geometry
import lstm_reorientation_causal_navigation as controls

CUES = ('native', 'ordinary', 'plus60', 'minus60', 'native_reset')
VARIANTS = ('blind_native', 'blind_ordinary', 'blind_plus60', 'blind_minus60', 'blind_reset_after_glimpse')
TIMES = (0, 1, 5, 15, 30, 45, 60)
PHASES = 192


@torch.no_grad()
def collect_intervention(net, cfg, plan, expected_pre, edited_pre=None, expected_cues=None, controller='policy'):
    """One fixed phase chunk. None is an exact observer-only sham."""
    n = len(plan['heading']); nh = cfg['nh']; groups = len(VARIANTS)
    assert controller in ('policy', 'uniform', 'planner')
    assert controller == 'policy' or edited_pre is None
    sched = task.schedule(plan, 'cpu'); cue_step = int(sched['glimpse'][0])
    assert bool((sched['glimpse'] == cue_step).all()) and bool((plan['delay'] == 60).all())
    assert expected_pre.shape == (n, 2*nh)
    if edited_pre is not None:
        assert edited_pre.shape == expected_pre.shape and torch.isfinite(edited_pre).all()
    expected = expected_pre.repeat(groups, 1).float()
    replacement = None if edited_pre is None else edited_pre.repeat(groups, 1).float()
    counter = [0]; snapshots = {}; checks = {}; cue_inputs = None; intervention_arrays = {}

    def before(module, inputs):
        nonlocal cue_inputs
        if counter[0] != cue_step: return None
        x, state = inputs; actual = torch.cat(state, 1)
        error = float((actual-expected).abs().max())
        assert error < 3e-4, ('Native physical history changed before intervention', error)
        checks['precue_history_replay_max_abs_error'] = error
        intervention_arrays['precue_replay_difference'] = (actual-expected).detach().cpu().clone()
        intervention_arrays['intended_precue_edit'] = (torch.zeros_like(expected) if replacement is None else replacement-expected).detach().cpu().clone()
        checks['all_variants_same_precue_state'] = all(torch.equal(actual[:n], actual[j*n:(j+1)*n]) for j in range(groups))
        assert checks['all_variants_same_precue_state']
        cue_inputs = x.detach().cpu().clone()
        if expected_cues is not None:
            target = torch.cat([expected_cues[cue.replace('_reset', '')] for cue in CUES]).float()
            input_error = float((x-target).abs().max()); assert input_error < 3e-5
            checks['native_cue_input_replay_max_abs_error'] = input_error
        if replacement is None: return None
        checks['applied_state_max_change'] = float((replacement-actual).abs().max())
        # The original input tensor is returned unchanged; only h/c is replaced.
        return x, tuple(replacement.chunk(2, 1))

    def after(module, inputs, output):
        step = counter[0]-cue_step
        if step in TIMES:
            z = torch.cat(output, 1).detach().cpu().clone()
            # The collector erases the reset branch after its cue LSTM output.
            # Its actor is not used during this forced step.
            if step == 0: z[-n:] = 0.
            snapshots[step] = z
        counter[0] += 1

    before_handle = net.lstm.register_forward_pre_hook(before)
    after_handle = net.lstm.register_forward_hook(after)
    try:
        roll, rows, physical, first = task.collect(net, cfg, task.make_source(cfg, 'cpu'), plan,
            variants=VARIANTS, sigma=0., device='cpu', retain_inputs=False, controller=controller)
    finally:
        before_handle.remove(); after_handle.remove()
    assert counter[0] == task.MAX_STEPS and set(snapshots) == set(TIMES)
    assert cue_inputs is not None and physical['forced_hits'] == 0
    for i in range(n*groups):
        assert torch.equal(snapshots[60][i], first[i]['state'])
    if cfg['cond'] == 'gc100':
        for j in range(1, 4):
            assert torch.equal(cue_inputs[:n], cue_inputs[j*n:(j+1)*n])
            for t in TIMES: assert torch.equal(snapshots[t][:n], snapshots[t][j*n:(j+1)*n])
            native = [{k:v for k,v in r.items() if k!='variant'} for r in rows[:n]]
            sham = [{k:v for k,v in r.items() if k!='variant'} for r in rows[j*n:(j+1)*n]]
            assert native == sham
    checks.update(physical=physical, unchanged_task_collector=True, input_not_modified=True,
                  perturbation_time='Immediately before endpoint cue LSTM call',
                  first_free_state_exact=True, state_noise_sigma=0.)
    raw = {k: torch.stack(v).cpu() for k,v in roll.items() if isinstance(v, list) and v}
    raw.update(goal=roll['goal'].cpu(), first_free=first, snapshots=snapshots, cue_inputs=cue_inputs, **intervention_arrays)
    return rows, checks, raw


def phase_summary(net, states, refs, theta, subset, scales):
    result = {}
    for t in TIMES:
        result[str(t)] = {}
        for cue in CUES:
            correct = refs['native'][t]; own = refs[cue][t]
            result[str(t)][cue] = {}
            for metric, scale in scales.items():
                radius = curve_geometry(correct/scale)['radius']
                result[str(t)][cue][metric] = controls.phase_metrics(
                    states[cue][t], correct, own, theta, theta[subset], scale, radius)
            baseline_prob = net.pi(torch.tensor(own[subset, :256], dtype=torch.float32)).softmax(1).numpy()
            edited_prob = net.pi(torch.tensor(states[cue][t][:, :256], dtype=torch.float32)).softmax(1).numpy()
            result[str(t)][cue]['actor_total_variation_from_own_unperturbed'] = controls.describe(
                .5*np.abs(edited_prob-baseline_prob).sum(1))
    return result


def enrich_navigation_rows(rows, raw, plan):
    start = int(task.schedule(plan, 'cpu')['free'][0]); n = len(plan['heading'])
    result = []
    for i, row in enumerate(rows):
        goal = plan['goal'][i % n]
        before = torch.linalg.vector_norm(raw['first_free'][i]['position']-goal)
        after = torch.linalg.vector_norm(raw['position'][start, i]-goal)
        result.append(dict(row, first_action_goalward_progress_m=float(before-after)))
    return result


def navigation_summary(rows):
    result = {}
    for cue, variant in zip(CUES, VARIANTS):
        values = [r for r in rows if r['variant'] == variant]
        assert len(values) == PHASES
        result[cue] = dict(trials=len(values), successes=sum(r['hit'] for r in values),
            success_fraction=float(np.mean([r['hit'] for r in values])),
            mean_goal_progress_m=float(np.mean([r['progress_m'] for r in values])),
            mean_first_action_goalward_progress_m=float(np.mean([r['first_action_goalward_progress_m'] for r in values])),
            mean_path_length_m=float(np.mean([r['free_path_m'] for r in values])),
            rows=values)
    return result


def rescue_contrasts(results):
    """A cue's ordinary unperturbed benefit is subtracted from its edit rescue."""
    contrasts = {}
    for construction, edits in results.items():
        sham = edits['none']; rows = {}
        for edit in ('plus30', 'minus30'):
            if edit not in edits: continue
            tested = edits[edit]; phase = {}; direct_phase = {}
            for metric in ('raw', 'relative_rms'):
                scores = {(kind, cue): result['phase']['60'][cue][metric]['phase_mae_deg']
                    for kind, result in (('none', sham), ('edit', tested)) for cue in ('native', 'ordinary')}
                phase[metric] = None if any(x is None for x in scores.values()) else (
                    scores[('edit', 'ordinary')]-scores[('none', 'ordinary')]
                    -scores[('edit', 'native')]+scores[('none', 'native')])
                direct_phase[metric] = None if any(x is None for x in scores.values()) else (
                    scores[('edit', 'ordinary')]-scores[('edit', 'native')])
            behavior = {}; direct_behavior = {}
            for key in ('success_fraction', 'mean_goal_progress_m'):
                behavior[key] = (tested['navigation']['native'][key]-sham['navigation']['native'][key]
                    -tested['navigation']['ordinary'][key]+sham['navigation']['ordinary'][key])
                direct_behavior[key] = tested['navigation']['native'][key]-tested['navigation']['ordinary'][key]
            rows[edit] = dict(late_phase_error_reduction_deg=phase,
                incremental_native_cue_navigation_rescue=behavior,
                direct_native_vs_ordinary_phase_error_reduction_deg=direct_phase,
                direct_native_vs_ordinary_navigation_gain=direct_behavior,
                positive_direction='Positive means correct cue reduces the state-edit cost beyond its unperturbed cue benefit')
        contrasts[construction] = rows
    return contrasts


def controlled_certificate(cfg, plan, chunks):
    """Recheck changed delay/tape indexing; never borrow the sealed16 certificate."""
    import certify_local_goal_return as certificate
    snapshots = []; goal = []
    for raw in chunks:
        width = len(raw['first_free'])//len(VARIANTS)
        snapshots.extend(raw['first_free'][4*width+i] for i in range(width))
        goal.append(raw['goal'][4*width:5*width])
    arrays = {new:torch.stack([r[old] for r in snapshots]).numpy() for new,old in (
        ('first_free_input','input'),('erased_state','state'),('terminal_pos','position'),
        ('terminal_hd','heading'),('terminal_motors','motors'))}
    start = int(task.schedule(plan, 'cpu')['free'][0])
    arrays.update(goal_snapshot=torch.cat(goal).numpy(), motor=plan['motor'][start:start+30].numpy(),
                  uniform=plan['action'][start:start+30].numpy())
    result = certificate.certify(arrays, cfg, plan['goal'][0].numpy())
    result.update(delay_override=60, free_start_update=start, source_sha256=sha(certificate.__file__),
        scope='Actual dense controlled-family erased snapshots and newly indexed tapes; not the untouched sealed16-heading evaluation bank')
    return result


@torch.no_grad()
def analyze_causal(net, cfg, plan, family, arrays, out, chunk_size=32):
    from lstm_local_return_family import slice_plan
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Preserve completed or interrupted causal evidence'
    net.float(); n = len(plan['heading']); assert n == 768 and n % PHASES == 0
    subset = np.arange(0, n, n//PHASES); theta = arrays['theta']
    scales = dict(raw=np.ones(512), relative_rms=arrays['scale'])
    # All reference families use the same frozen network and common input.
    refs = {cue: {t:arrays[cue+'_states'][TIMES.index(t)] for t in TIMES} for cue in CUES[:-1]}
    net.double()
    reference_reset = controls.follow(net, np.zeros((n, 512)),
        torch.tensor(arrays['ordinary_input'], dtype=torch.float64), anchor=0)
    refs['native_reset'] = reference_reset
    net.float(); results = {}; files = []; dense_certificate = None
    for construction, scale in scales.items():
        edits, edit_audit = controls.state_edits(arrays['pre_cue_state'], scale, subset)
        results[construction] = {}
        for edit, values in edits.items():
            all_rows = []; checks = []; raw_chunks = []; chunks = {cue:{t:[] for t in TIMES} for cue in CUES}
            for begin in range(0, PHASES, chunk_size):
                stop = min(begin+chunk_size, PHASES); selected = subset[begin:stop]
                small = slice_plan(plan, selected)
                pre = family['pre_cue_state'][selected]
                replaced = None if edit == 'none' else torch.tensor(values[begin:stop], dtype=torch.float32)
                cues = {cue:family['cue_inputs'][cue][selected] for cue in CUES[:-1]}
                rows, verification, raw = collect_intervention(net, cfg, small, pre, replaced, cues)
                rows = enrich_navigation_rows(rows, raw, small)
                verification['precue_replay_relative_accuracy'] = {}
                for metric, units in scales.items():
                    full = arrays['pre_cue_state']/units
                    radius = float(np.sqrt(np.mean(np.sum((full-full.mean(0))**2, axis=1))))
                    difference = np.linalg.norm(raw['precue_replay_difference'].numpy()/units, axis=1)
                    kick = np.linalg.norm(raw['intended_precue_edit'].numpy()/units, axis=1)
                    verification['precue_replay_relative_accuracy'][metric] = dict(
                        family_radius=radius, maximum_vector_error=float(difference.max()),
                        max_error_over_radius=None if radius < 1e-12 else float(difference.max()/radius),
                        mean_intended_edit_size=float(kick.mean()),
                        max_error_over_intended_edit=None if bool((kick < 1e-12).any()) else float((difference/kick).max()))
                path = out/f'{construction}__{edit}__chunk{begin:03d}.pt'
                torch.save(dict(plan=small, edit=edit, construction=construction, raw=raw,
                    rows=rows, checks=verification), path)
                files.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
                all_rows.extend(rows); checks.append(verification)
                if construction == 'raw' and edit == 'none': raw_chunks.append(raw)
                width = stop-begin
                for j, cue in enumerate(CUES):
                    for t in TIMES: chunks[cue][t].append(raw['snapshots'][t][j*width:(j+1)*width].numpy())
            states = {cue:{t:np.concatenate(chunks[cue][t]) for t in TIMES} for cue in CUES}
            result = dict(edit_definition=edit_audit, phase=phase_summary(net, states, refs, theta, subset, scales),
                          navigation=navigation_summary(all_rows), collection_checks=checks)
            results[construction][edit] = result
            if raw_chunks:
                dense_certificate = controlled_certificate(cfg, slice_plan(plan, subset), raw_chunks)
    physical_controls = {}
    for controller in ('uniform', 'planner'):
        rows_all = []; checks = []
        for begin in range(0, PHASES, chunk_size):
            selected = subset[begin:begin+chunk_size]; small = slice_plan(plan, selected)
            rows, verification, raw = collect_intervention(net, cfg, small,
                family['pre_cue_state'][selected], controller=controller)
            rows = enrich_navigation_rows(rows, raw, small)
            path = out/f'control__{controller}__chunk{begin:03d}.pt'
            torch.save(dict(plan=small, raw=raw, rows=rows, checks=verification), path)
            files.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
            rows_all.extend(rows); checks.append(verification)
        physical_controls[controller] = dict(navigation=navigation_summary(rows_all), checks=checks)
    return dict(results=results, rescue_contrasts=rescue_contrasts(results), raw_files=files,
        controlled_family_certificate=dense_certificate, physical_controls=physical_controls,
        phases=PHASES, clean_native_float32=True, frozen_task_sha256=sha(task.__file__),
        dynamics_vs_navigation='Phase references evolve under common LEFT input; native navigation then receives its own actual previous actions. Full native state snapshots and raw trajectory evidence are retained.',
        inference='One trained pair; all phases and contexts are within-model repeated measures. No trial-count p value for population claims.')
