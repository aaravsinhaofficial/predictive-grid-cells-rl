"""Display the complete, audited local-return mechanism cohort.

This reads saved measurements only: no policy execution, fitting, new criteria,
or context selection. A failed collection/numerical validation is unresolved;
it cannot be silently omitted to produce a partial mechanism figure.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
BASE = ROOT/'results/lstm_local_return_mechanism'
PROTOCOL_HASH = '012825b8278ab571a3a2aa30dee544985c537031d5d113269229ac9d8af4f72a'
CONDITIONS = ('gc100', 'pgc33')
CONTEXTS = (0, 1)
CUES = ('native', 'ordinary', 'plus60', 'minus60')
CAUSAL_CUES = CUES + ('native_reset',)
METRICS = ('raw', 'relative_rms')
MAPS = ('all_hidden', 'learned_action_scores')
OBS_METRICS = ('raw_full_state_kicks', 'h_c_RMS_full_state_kicks')
TIMES = (0, 1, 5, 15, 30, 45, 60)
EDITS = ('none', 'plus30', 'minus30', 'radial_plus', 'radial_minus', 'isotropic_plus', 'isotropic_minus')
LABELS = {'gc100':'Grid', 'pgc33':'PGC + grid'}
NAV_KEYS = {'mean_goal_progress_m':'progress_m', 'mean_first_action_goalward_progress_m':'first_action_goalward_progress_m',
            'mean_path_length_m':'free_path_m'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_file(path, digest=None):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), f'Missing or symlinked input: {path}'
    actual = sha(path)
    assert digest is None or actual == digest, f'Changed input: {path}'
    return actual


def checked_navigation(record, context):
    """Reconcile saved means on the exact 192 paired physical phase IDs."""
    assert record['trials'] == 192 and len(record['rows']) == 192
    rows = sorted(record['rows'], key=lambda r:r['heading_index'])
    assert [(r['context'], r['heading_index']) for r in rows] == [(context, i) for i in range(0, 768, 4)]
    hits = sum(bool(r['hit']) for r in rows)
    assert record['successes'] == hits and record['success_fraction'] == hits/192
    for key, field in NAV_KEYS.items():
        assert math.isclose(record[key], float(np.mean([r[field] for r in rows])), rel_tol=1e-10, abs_tol=1e-12), key
    return rows


def flatten_context(condition, context, primary, observable, causal):
    """Retain every stored measurement; extract existing values for presentation."""
    assert condition in CONDITIONS and context in CONTEXTS
    assert primary['condition'] == condition and primary['context'] == context
    assert set(primary['branches']) == set(observable['branches']) == set(CUES)
    assert causal['phases'] == 192 and causal['clean_native_float32'] is True
    assert set(causal['results']) == set(METRICS)
    tags = dict(condition=condition, context=context)
    geometry_rows = []; observable_rows = []; phase_rows = []; navigation_rows = []
    contrasts = []; invalid = []; precision = []
    for cue in CUES:
        b = primary['branches'][cue]
        assert b['native_float32_first_free_replay']['pointwise_pass'] is True
        assert set(b['candidate_gates']) == set(METRICS)
        assert b['passes_both_metric_candidate_screen'] == all(all(g.values()) for g in b['candidate_gates'].values())
        precision.append(dict(**tags, cue=cue, validation=b['native_float32_first_free_replay'],
            relative_to_direction_family=b['native_float32_replay_relative_to_direction_family']))
        for metric in METRICS:
            r = b['invariance']['30'][metric]['horizons']['30']; gates = b['candidate_gates'][metric]
            geometry_rows.append(dict(**tags, cue=cue, metric=metric,
                radial_normal_gain=r['radial_normal_gain']['median'],
                radial_over_tangent_gain=r['radial_over_tangent_gain']['median'],
                fixed_curve_residual_p90=r['residual_to_fixed_curve_over_radius']['p90'],
                radius_retention=b['trajectory']['30'][metric]['radius_retention'],
                candidate=all(gates.values()), candidate_gates=gates,
                source_anchor_updates=30, recovery_updates=30))
        assert set(observable['branches'][cue]) == set(MAPS)
        for name in MAPS:
            assert set(observable['branches'][cue][name]) == set(OBS_METRICS)
            for metric in OBS_METRICS:
                r = observable['branches'][cue][name][metric]
                assert r['observation_offsets'] == [0, 5, 10, 15] and r['evolution_updates'] == 30
                for centering, key in (('centered','centered_conditional_factor_candidate'),
                                       ('noncentered','noncentered_observable_candidate')):
                    gates = r['candidate_gates'][centering]
                    assert r[key] == all(gates.values())
                    observable_rows.append(dict(**tags, cue=cue, map=name, metric=metric, centering=centering,
                        radial_normal_gain=r['radial_observable_normal_gain']['median'],
                        radial_over_tangent_gain=r['radial_over_tangent_observable_gain']['median'],
                        candidate=r[key], candidate_gates=gates,
                        invariance=r['invariance'][centering], source_anchor_updates=15,
                        evolution_updates=30, observation_offsets=[0, 5, 10, 15]))
    for construction, edits in causal['results'].items():
        assert 'none' in edits
        declared = edits['none']['edit_definition']; unavailable = declared['invalid_edits']
        assert set(edits).isdisjoint(unavailable) and set(edits)|set(unavailable) == set(EDITS)
        invalid += [dict(**tags, construction=construction, edit=k, reason=v) for k,v in unavailable.items()]
        for edit, result in edits.items():
            assert result['edit_definition'] == declared
            assert set(result['phase']) == set(map(str, TIMES)) and set(result['navigation']) == set(CAUSAL_CUES)
            for cue, nav in result['navigation'].items():
                checked_navigation(nav, context)
                navigation_rows.append(dict(**tags, construction=construction, edit=edit, cue=cue,
                    **{k:v for k,v in nav.items() if k!='rows'}))
            for time, cues in result['phase'].items():
                assert set(cues) == set(CAUSAL_CUES)
                for cue, metrics in cues.items():
                    assert set(metrics) == set(METRICS)|{'actor_total_variation_from_own_unperturbed'}
                    for metric in METRICS:
                        phase_rows.append(dict(**tags, construction=construction, edit=edit, cue=cue,
                            time=int(time), metric=metric, **metrics[metric],
                            actor_total_variation_from_own_unperturbed=metrics['actor_total_variation_from_own_unperturbed']))
        for edit, row in causal['rescue_contrasts'][construction].items():
            assert edit in ('plus30', 'minus30') and edit in edits
            contrasts.append(dict(**tags, construction=construction, edit=edit, **row))
    assert set(causal['physical_controls']) == {'uniform', 'planner'}
    for control in causal['physical_controls'].values():
        assert set(control['navigation']) == set(CAUSAL_CUES)
        for nav in control['navigation'].values(): checked_navigation(nav, context)
    assert causal['controlled_family_certificate']['heading_trials'] == 192
    # Scale is irrelevant to the literal history substitutions; both are retained.
    for edit in ('none', 'plus30', 'minus30'):
        a, b = (causal['results'][m] for m in METRICS)
        assert (edit in a) == (edit in b)
        if edit in a: assert a[edit]['navigation'] == b[edit]['navigation']
    return dict(geometry=geometry_rows, observables=observable_rows, phase=phase_rows,
        navigation=navigation_rows, rescue_contrasts=contrasts, undefined_edits=invalid,
        numerical_validity=precision)


def check_independent_audit(record, base, expected_inputs):
    assert record['passed'] is True and record['complete'] is True
    assert record['status'] == 'complete' and record['neural_artifacts_complete'] is True
    assert record['protocol_sha256'] == PROTOCOL_HASH
    assert set(record['models']) == set(CONDITIONS)
    for condition in CONDITIONS:
        assert set(map(str, record['models'][condition])) == {'0', '1'}
    assert record['auditor_sha256'] == checked_file(ROOT/'scripts/audit_local_return_mechanism.py')
    for name, digest in expected_inputs.items():
        assert record['input_sha256'].get(name) == digest, f'Independent audit does not cover {name}'


def load_complete(base=BASE):
    """All guards precede any output. This performs hash/audit reads only."""
    import lstm_local_return_mechanism as producer
    from package_local_goal_return_outputs import verify_final_audit
    base = Path(base).resolve(); hashes = {}

    def track(path, digest=None):
        path = Path(path)
        actual = checked_file(path, digest)
        path = path.resolve()
        hashes[str(path.relative_to(ROOT))] = actual
        return path

    protocol = read(track(base/'protocol.json', PROTOCOL_HASH)); frozen = dict(protocol)
    assert frozen.pop('frozen_utc') and frozen == producer.protocol()
    for name, digest in {**protocol['source_sha256'], **protocol['design_sha256']}.items(): track(ROOT/name, digest)
    track(ROOT/'results/local_goal_return_design/behavior_protocol.json', protocol['behavior_protocol_sha256'])
    track(ROOT/'docs/research/local_goal_return_training_launch.json')
    reports = {}; checkpoints = {}
    for condition in CONDITIONS:
        checkpoint = producer.resolve_checkpoint(condition)
        report = read(track(ROOT/f'results/local_goal_return_design/cloud_audit/{condition}.json'))
        verify_final_audit(ROOT, checkpoint.parent, condition, report)
        reports[condition] = report; checkpoints[condition] = track(checkpoint)
        for name, item in report['artifacts'].items(): track(checkpoint.parent/name, item['sha256'])
    gate = producer.behavioral_gate(reports)
    assert gate['passed'], 'Frozen useful-memory gate did not pass; no automatic mechanism display'
    saved_gate = read(track(base/'behavior_gate.json')); assert saved_gate['gate'] == gate
    for condition in CONDITIONS:
        assert saved_gate['checkpoints'][condition]['sha256'] == sha(checkpoints[condition])
    tables = {k:[] for k in ('geometry','observables','phase','navigation','rescue_contrasts','undefined_edits','numerical_validity')}
    contexts = []; audit_inputs = {}
    for condition in CONDITIONS:
        for context in CONTEXTS:
            folder = base/condition/f'context{context}'
            assert not (folder/'collection_failed.json').exists() and not (folder/'analysis_failed.json').exists(), 'Unresolved context; never drop it'
            records = {name:read(track(folder/f'{name}.json')) for name in ('primary','observable','causal')}
            primary, observable, causal = (records[k] for k in ('primary','observable','causal'))
            for name, record in records.items():
                assert record['checkpoint_sha256'] == sha(checkpoints[condition]) and record['protocol_sha256'] == PROTOCOL_HASH
                assert record['states_sha256'] == primary['states_sha256']
                audit_inputs[str((folder/f'{name}.json').relative_to(ROOT))] = sha(folder/f'{name}.json')
            assert causal['family_sha256'] == primary['family_sha256']
            for name, digest in (('family.pt',primary['family_sha256']),('states.npz',primary['states_sha256'])):
                track(folder/name, digest); audit_inputs[str((folder/name).relative_to(ROOT))] = digest
            attempt = read(track(folder/'collection_attempt.json'))
            assert attempt['protocol_sha256'] == PROTOCOL_HASH and attempt['checkpoint_sha256'] == primary['checkpoint_sha256']
            assert attempt['plan_metadata'] == primary['plan_metadata']
            track(folder/'input_plan.pt', attempt['input_plan_sha256'])
            audit_inputs[str((folder/'input_plan.pt').relative_to(ROOT))] = attempt['input_plan_sha256']
            assert causal['frozen_task_sha256'] == protocol['source_sha256']['scripts/local_goal_return_training.py']
            expected = {f'{m}__{e}__chunk{i:03d}.pt' for m,edits in causal['results'].items() for e in edits for i in range(0,192,32)}
            expected |= {f'control__{c}__chunk{i:03d}.pt' for c in ('uniform','planner') for i in range(0,192,32)}
            assert {Path(item['path']).name for item in causal['raw_files']} == expected
            assert len(causal['raw_files']) == len(expected)
            assert {p.name for p in (folder/'causal_arrays').iterdir()} == expected
            for item in causal['raw_files']:
                path = folder/'causal_arrays'/Path(item['path']).name
                track(path, item['sha256']); assert path.stat().st_size == item['bytes']
                audit_inputs[str(path.relative_to(ROOT))] = item['sha256']
            with np.load(folder/'states.npz', allow_pickle=False) as arrays:
                assert arrays['checkpoint_sha256'].item() == primary['checkpoint_sha256']
                assert np.array_equal(arrays['times'], TIMES)
                assert arrays['theta'].shape == (768,) and arrays['pre_cue_state'].shape == (768,512)
                assert arrays['ordinary_input'].shape == (1,1287)
                for cue in CUES: assert arrays[cue+'_states'].shape == (7,768,512)
            flat = flatten_context(condition, context, **records)
            for key, values in flat.items(): tables[key].extend(values)
            contexts.append(dict(condition=condition,context=context,
                files={**{name:str((folder/f'{name}.json').relative_to(ROOT)) for name in records},
                    **{name:str((folder/file).relative_to(ROOT)) for name,file in (
                        ('input plan','input_plan.pt'),('native family','family.pt'),
                        ('full state archive','states.npz'),('raw causal chunks','causal_arrays'))}},
                physical_controls=causal['physical_controls'], certificate=causal['controlled_family_certificate'],
                records=records))
    independent_path = track(base/'independent_artifact_audit.json')
    check_independent_audit(read(independent_path), base, audit_inputs)
    track(ROOT/'scripts/audit_local_return_mechanism.py')
    track(ROOT/'scripts/package_local_goal_return_outputs.py')
    track(Path(__file__)); track(ROOT/'tests/test_plot_local_return_mechanism.py')
    counts = dict(full_state=sum(r['candidate'] for r in tables['geometry']),
        full_state_checks=len(tables['geometry']), observable=sum(r['candidate'] for r in tables['observables']),
        observable_checks=len(tables['observables']),
        full_state_both_metric_candidates=sum(all(r['candidate'] for r in tables['geometry']
            if r['condition']==c and r['context']==k and r['cue']==cue)
            for c in CONDITIONS for k in CONTEXTS for cue in CUES),full_state_branches=16)
    return dict(version='local_return_mechanism_display_v1',complete=True,synthetic=False,
        behavior_gate=gate,protocol_sha256=PROTOCOL_HASH,input_sha256=hashes,
        independent_audit=str(independent_path.relative_to(ROOT)),contexts=contexts,tables=tables,
        candidate_counts=counts,
        positive_candidate_validation=dict(native_float32_criterion_replication='pending' if counts['full_state']+counts['observable'] else 'not triggered',
            conditional_same_observable_rescue='Consult per-cue full-state/observable candidates; required for a qualifying observable-only factor, pending execution.',
            interpretation='Candidate screens nominate structures; numerical agreement, acute phase repair and navigation are separate measurements. No automatic proof label.'),
        scope='Both fixed final models, contexts0/1, every cue/sign/metric/map/time. One original seed pair with new readouts; nested phases are not independent trained models.')


def select(rows, **keys):
    selected = [r for r in rows if all(r.get(k)==v for k,v in keys.items())]
    assert len(selected) <= 1, keys
    return selected[0] if selected else None


def finite(value):
    return value is not None and math.isfinite(value)


def render(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    table = summary['tables']; groups = [(c,k) for c in CONDITIONS for k in CONTEXTS]
    colors = {'native':'#db702f','ordinary':'#357eb0'}
    labels = [f'{LABELS[c]}\nContext {k+1}' for c,k in groups]
    layout = {'phase_ylim':{},'synthetic':summary.get('synthetic',False)}

    def style(ax):
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_xlim(-.48,3.48)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)

    def finish(fig, name):
        if summary.get('synthetic'):
            fig.text(.5,.5,'SYNTHETIC LAYOUT — NOT RESULTS',ha='center',va='center',rotation=23,
                fontsize=27,color='#a21b1b',alpha=.27)
        fig.supxlabel('One original seed pair · two contexts per model · all prescribed directions',fontsize=9)
        for suffix in ('png','pdf'): fig.savefig(out/f'{name}.{suffix}',dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(10,7.1),layout='constrained',sharey='row')
    all_errors=[]
    legend = [Line2D([],[],marker='o',ls='',color=colors[q],label=f'{q.title()} cue') for q in ('native','ordinary')]
    legend += [Line2D([],[],marker=m,ls='',color='.4',label=label,markerfacecolor='none' if m=='o' else '.4')
               for m,label in (('^','+30° state'),('v','−30° state'),('o','Unperturbed'))]
    for col,metric in enumerate(METRICS):
        top, bottom = axes[:,col]; errors=[]
        for i,(condition,context) in enumerate(groups):
            for cue, offset in (('native',-.12),('ordinary',.12)):
                row=select(table['geometry'],condition=condition,context=context,cue=cue,metric=metric)
                value=row['radial_over_tangent_gain']
                if finite(value): top.scatter(i+offset,value,c=colors[cue],s=34)
                else: top.text(i+offset,.02,'?',ha='center',transform=top.get_xaxis_transform())
                if finite(value) and not row['candidate_gates']['radius_retains5percent']:
                    top.annotate('Small signal',(i+offset,value),xytext=(0,7),textcoords='offset points',fontsize=6,ha='center')
                for edit,marker,shift in (('none','o',0),('plus30','^',-.035),('minus30','v',.035)):
                    r=select(table['phase'],condition=condition,context=context,construction='raw',edit=edit,
                        cue=cue,time=60,metric=metric)
                    value=None if r is None else r['phase_mae_deg']
                    if finite(value):
                        errors.append(value);bottom.scatter(i+offset+shift,value,marker=marker,s=38,
                            facecolors='none' if edit=='none' else colors[cue],edgecolors=colors[cue],linewidths=1)
                    else: bottom.text(i+offset,.02,'?',ha='center',transform=bottom.get_xaxis_transform(),color=colors[cue])
        top.axhline(.5,color='.5',ls='--',lw=1);top.set_ylim(bottom=0)
        top.set(title='Raw state' if metric=='raw' else 'h/c amplitude-scaled state',ylabel='Radial / tangent persistence\n2 s → 4 s (lower = more radial recovery)')
        all_errors.extend(errors)
        bottom.axhline(30,color='.65',ls=':',lw=1)
        bottom.set(ylabel='Phase error at 4 s (°)\nrelative to correct-cue family')
        style(top);style(bottom)
    for metric,ax in zip(METRICS,axes[1]):
        ax.set_ylim(0,max(33.,1.1*max([30.]+all_errors)))
        layout['phase_ylim'][metric]=list(ax.get_ylim())
    axes[0,0].legend(handles=legend[:2],frameon=False,fontsize=8,loc='best')
    axes[1,0].legend(handles=legend[2:],frameon=False,fontsize=8,loc='best')
    fig.suptitle('Recovery and predictive-cue repair',fontsize=14)
    finish(fig,'recovery_and_phase_repair')

    fig, axes = plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for ax,key,factor,title,ylabel in zip(axes,('success_fraction','mean_goal_progress_m'),(100,100),
            ('Goals reached','Movement toward the goal'),('Success in 30 free actions (%)','Mean goalward progress (cm)')):
        for i,(condition,context) in enumerate(groups):
            for edit,marker,dx in (('none','o',0),('plus30','^',-.035),('minus30','v',.035)):
                pair=[]
                for cue,offset in (('native',-.12),('ordinary',.12)):
                    row=select(table['navigation'],condition=condition,context=context,construction='raw',edit=edit,cue=cue)
                    if row is None:
                        ax.text(i+offset,.02,'?',ha='center',transform=ax.get_xaxis_transform());continue
                    point=(i+offset+dx,factor*row[key]);pair.append(point)
                    ax.scatter(*point,marker=marker,s=40,facecolors='none' if edit=='none' else colors[cue],edgecolors=colors[cue],zorder=3)
                if len(pair)==2:ax.plot(*zip(*pair),color='.65',lw=.7,zorder=1)
        style(ax);ax.set(title=title,ylabel=ylabel)
    axes[0].set_ylim(0,100);axes[1].axhline(0,color='.65',lw=.8)
    axes[0].legend(handles=legend[:2],frameon=False,fontsize=8);axes[1].legend(handles=legend[2:],frameon=False,fontsize=8)
    finish(fig,'paired_navigation')
    return layout


def write_tables(tables, out):
    for name, rows in tables.items():
        (out/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with (out/f'{name}.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def write_readme(summary, out):
    counts=summary['candidate_counts'];relative=lambda name:os.path.relpath(ROOT/name,out)
    lines=['# Predictive cues and local-return memory','',
        '**Both fixed final models; one original seed pair, with new action/value readouts.** All two contexts per model and every prescribed cue branch are retained. The complete artifact audit passed, and the frozen behavior gate permitted the mechanism analyses.','',
        '![Recovery and cue repair](recovery_and_phase_repair.png)','',
        'Top: the existing median radial/tangent gain ratio from 2 s to 4 s, in raw and amplitude-scaled full h/c state. Radial recovery uses both signs of a 2%-radius kick at 384 arc points, measured relative to the evolving curve; tangent persistence uses the local curve-length change. The cue changes the starting state; subsequent ordinary inputs and the recurrent transition rule are identical across cue branches within each model. The dashed 0.5 line is one existing screen threshold. A small ratio must be read with curve persistence, amplitude, invariance and ordered phase transport. It is not the complete ring test. Bottom: both signs of a 30° history-state swap across 192 directions, measured after the predictive cue has been absent for four seconds. Open circles show the unperturbed cue baselines. The raw/scaled panels share axes. Question marks retain undefined phase estimates or interventions.','',
        '![Matched navigation](paired_navigation.png)','',
        'Lines connect native and ordinary cue results on the same 192 physical directions and random tapes, for each state condition. Only the input cue differs within each connected pair. The policy then has 30 free actions; success and goalward progress use every prescribed trial, with first-hit termination. These controlled two-context measurements are separate from the 256-trial behavioral gate.','',
        '| Model / context | Full-state metric passes (of 8) | Observable candidates (of 32) | Dense certified pairs | Planner successes | Uniform successes |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['contexts']:
        c,k=row['condition'],row['context'];cert=row['certificate']
        selected=lambda name:[r for r in summary['tables'][name] if r['condition']==c and r['context']==k]
        nav=lambda kind:row['physical_controls'][kind]['navigation']['native']['successes']
        lines.append(f"| {LABELS[c]} / {k+1} | {sum(r['candidate'] for r in selected('geometry'))} | {sum(r['candidate'] for r in selected('observables'))} | {cert['certified_pairs']}/{cert['equally_weighted_pairs']} | {nav('planner')}/192 | {nav('uniform')}/192 |")
    lines += ['',f"The frozen full-state screen nominates **{counts['full_state_both_metric_candidates']} / {counts['full_state_branches']} cue branches in both metrics**; **{counts['full_state']} / {counts['full_state_checks']} individual metric checks** pass. There are **{counts['observable']} / {counts['observable_checks']} fixed-observable candidates**. Counts include all four cues, both metrics, both fixed observation maps and both centering conventions; they are nested checks, not independent trained seeds.",'',
        '**A candidate is a measured pattern requiring the specified follow-through.** Any positive candidate still requires the unchanged criteria to reproduce under native float32 transitions. A qualifying observable-only factor also requires phase repair in that same fixed observation map. These follow-ups are not marked complete by this display. Acute correction by a pulse, autonomous transverse attraction, and navigation recovery are separate results.','',
        'The double-versus-native first-free recurrence checks passed for all displayed branches. Their absolute errors, full-state errors relative to directional signal, and original numerical flags remain in [numerical validity](numerical_validity.json). Agreement of that replay does not substitute for native float32 replication of positive geometric candidates.','',
        'The main figures show native/ordinary cue comparisons and both ±30° state swaps. The complete tables also retain +60° and −60° cues, post-cue erasure, radial/random state edits, both constructions, every time, both phase-reference conventions, off-curve errors, actor effects and per-trial navigation. Baseline-adjusted cue/state interactions stay separate from direct cue benefits.','',
        '[Every geometry gate](geometry.json) · [Both fixed observable maps](observables.json) · [Every phase measurement](phase.json) · [Navigation](navigation.json) · [Direct and baseline-adjusted rescue](rescue_contrasts.json) · [Undefined edits](undefined_edits.json) · [Full source records, rows and file hashes](summary.json)','',
        f"[Independent artifact audit]({relative(summary['independent_audit'])}) · [Frozen protocol]({relative('results/lstm_local_return_mechanism/protocol.json')}) · [Cue-information interpretation]({relative('docs/research/local_return_cue_information.md')})",'',
        'Original input, family/state archives and every causal raw chunk are linked in the following per-context records. Paths in the full input hash manifest are repository-relative.','']
    for row in summary['contexts']:
        links=' · '.join(f'[{name}]({relative(path)})' for name,path in row['files'].items())
        lines.append(f"- {LABELS[row['condition']]} / context {row['context']+1}: {links}")
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=BASE)
    parser.add_argument('--out',type=Path,required=True,help='Fresh report directory; existing files are never replaced')
    args=parser.parse_args();assert not args.out.exists() or not any(args.out.iterdir())
    summary=load_complete(args.input)
    args.out.mkdir(parents=True,exist_ok=True)
    summary['display_layout']=render(summary,args.out)
    write_tables(summary['tables'],args.out);write_readme(summary,args.out)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),complete=True,candidates=summary['candidate_counts'])))


if __name__=='__main__':main()
