"""Synthetic schema/layout checks; no trained checkpoints or results are read."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import plot_local_return_mechanism as report


def nav(context, cue, edit):
    count=144 if edit=='none' else (120 if cue=='native' else 96)
    progress=.2 if cue=='native' else .15
    rows=[dict(context=context,heading_index=i*4,hit=i<count,progress_m=progress,
        first_action_goalward_progress_m=.003,free_path_m=.6,
        variant='blind_'+cue) for i in range(192)]
    return dict(trials=192,successes=count,success_fraction=count/192,
        mean_goal_progress_m=progress,mean_first_action_goalward_progress_m=.003,
        mean_path_length_m=.6,rows=rows)


def fixture_context(condition='gc100',context=0):
    branches={};observables={}
    for cue in report.CUES:
        gates={m:dict(radius_retains5percent=True,radial_over_tangent_below_half=False) for m in report.METRICS}
        radial=.8 if cue=='native' else 1.1
        branches[cue]=dict(candidate_gates=gates,passes_both_metric_candidate_screen=False,
            native_float32_first_free_replay=dict(pointwise_pass=True,max_abs_error=1e-8),
            native_float32_replay_relative_to_direction_family={},
            trajectory={'30':{m:dict(radius_retention=.8) for m in report.METRICS}},
            invariance={'30':{m:dict(horizons={'30':dict(radial_normal_gain={'median':radial},
                radial_over_tangent_gain={'median':radial},residual_to_fixed_curve_over_radius={'p90':.1})})
                for m in report.METRICS}})
        observables[cue]={name:{m:dict(observation_offsets=[0,5,10,15],evolution_updates=30,
            candidate_gates={c:dict(test=False) for c in ('centered','noncentered')},
            centered_conditional_factor_candidate=False,noncentered_observable_candidate=False,
            radial_observable_normal_gain={'median':radial},radial_over_tangent_observable_gain={'median':radial},
            invariance={c:dict(test=1.) for c in ('centered','noncentered')})
            for m in report.OBS_METRICS} for name in report.MAPS}
    results={}
    for construction in report.METRICS:
        results[construction]={}
        for edit in report.EDITS:
            phase={}
            for time in report.TIMES:
                phase[str(time)]={}
                for cue in report.CAUSAL_CUES:
                    error=(1. if cue=='native' else 2.) if edit=='none' else (20. if cue=='native' else 30.)
                    phase[str(time)][cue]={m:dict(phase_estimate_valid=True,phase_mae_deg=error,
                        own_branch_phase_mae_deg=error-1,off_correct_curve_over_radius={'median':.02}) for m in report.METRICS}
                    phase[str(time)][cue]['actor_total_variation_from_own_unperturbed']={'mean':.1}
            results[construction][edit]=dict(edit_definition={'invalid_edits':{}},phase=phase,
                navigation={c:nav(context,c,edit) for c in report.CAUSAL_CUES})
    causal=dict(phases=192,clean_native_float32=True,results=results,
        rescue_contrasts={m:{e:dict(late_phase_error_reduction_deg={m:9. for m in report.METRICS})
            for e in ('plus30','minus30')} for m in report.METRICS},
        physical_controls={name:dict(navigation={c:nav(context,c,'none') for c in report.CAUSAL_CUES})
            for name in ('uniform','planner')},
        controlled_family_certificate=dict(heading_trials=192,equally_weighted_pairs=96,certified_pairs=96))
    return dict(primary=dict(condition=condition,context=context,branches=branches),
        observable=dict(branches=observables),causal=causal)


def synthetic_summary():
    tables={name:[] for name in ('geometry','observables','phase','navigation','rescue_contrasts','undefined_edits','numerical_validity')}
    for condition in report.CONDITIONS:
        for context in report.CONTEXTS:
            for key,rows in report.flatten_context(condition,context,**fixture_context(condition,context)).items():tables[key]+=rows
    return dict(synthetic=True,tables=tables)


def test_all_fixed_contexts_cues_metrics_maps_times_retained():
    summary=synthetic_summary();tables=summary['tables']
    assert len(tables['geometry'])==32 and len(tables['observables'])==128
    assert len(tables['navigation'])==4*2*7*5
    assert len(tables['phase'])==4*2*7*5*7*2
    assert {(r['condition'],r['context']) for r in tables['geometry']} == {(c,k) for c in report.CONDITIONS for k in report.CONTEXTS}
    assert {r['cue'] for r in tables['phase']}==set(report.CAUSAL_CUES)
    assert {r['time'] for r in tables['phase']}==set(report.TIMES)


def test_missing_cue_or_false_numerical_completion_rejected():
    for field in ('primary','observable'):
        fixture=fixture_context();del fixture[field]['branches']['minus60']
        with pytest.raises(AssertionError):report.flatten_context('gc100',0,**fixture)
    fixture=fixture_context()
    fixture['primary']['branches']['native']['native_float32_first_free_replay']['pointwise_pass']=False
    with pytest.raises(AssertionError):report.flatten_context('gc100',0,**fixture)


def test_duplicate_phase_and_changed_summary_rejected():
    original=nav(1,'native','none')
    duplicated=deepcopy(original);duplicated['rows'][1]=deepcopy(duplicated['rows'][0])
    with pytest.raises(AssertionError):report.checked_navigation(duplicated,1)
    wrong=deepcopy(original);wrong['mean_goal_progress_m']+=.01
    with pytest.raises(AssertionError):report.checked_navigation(wrong,1)
    wrong=deepcopy(original);wrong['successes']-=1
    with pytest.raises(AssertionError):report.checked_navigation(wrong,1)


def test_undefined_edits_and_null_phase_are_retained():
    fixture=fixture_context()
    for construction in report.METRICS:
        edits=fixture['causal']['results'][construction]
        del edits['radial_plus'];del edits['radial_minus']
        for result in edits.values():
            result['edit_definition']['invalid_edits']={k:'Degenerate radial direction' for k in ('radial_plus','radial_minus')}
            for time in result['phase'].values():
                for cue in time.values():
                    for metric in report.METRICS:
                        cue[metric].update(phase_estimate_valid=False,phase_mae_deg=None)
    flat=report.flatten_context('gc100',0,**fixture)
    assert len(flat['undefined_edits'])==4
    assert len(flat['navigation'])==2*5*5
    assert all(row['phase_mae_deg'] is None for row in flat['phase'])


def test_phase_substitutions_cannot_change_with_metric_construction():
    fixture=fixture_context()
    fixture['causal']['results']['relative_rms']['plus30']['navigation']['native']=nav(0,'ordinary','plus30')
    with pytest.raises(AssertionError):report.flatten_context('gc100',0,**fixture)


def test_hash_and_incomplete_independent_audit_guards(tmp_path,monkeypatch):
    path=tmp_path/'input.json';path.write_text('{}')
    digest=hashlib.sha256(b'{}').hexdigest();assert report.checked_file(path,digest)==digest
    path.write_text('{"changed": true}')
    with pytest.raises(AssertionError):report.checked_file(path,digest)
    audit=dict(passed=True,complete=True,status='not_run_behavior_gate',neural_artifacts_complete=False)
    with pytest.raises(AssertionError):report.check_independent_audit(audit,tmp_path,{})
    source=tmp_path/'scripts/audit_local_return_mechanism.py';source.parent.mkdir();source.write_text('# synthetic')
    monkeypatch.setattr(report,'ROOT',tmp_path)
    audit.update(status='complete',neural_artifacts_complete=True,protocol_sha256=report.PROTOCOL_HASH,
        auditor_sha256=report.sha(source),models={c:{'0':{},'1':{}} for c in report.CONDITIONS},
        input_sha256={'input.json':'fake'})
    report.check_independent_audit(audit,tmp_path,{'input.json':'fake'})
    with pytest.raises(AssertionError):report.check_independent_audit(audit,tmp_path,{'input.json':'changed'})
    del audit['models']['pgc33']['1']
    with pytest.raises(AssertionError):report.check_independent_audit(audit,tmp_path,{})


def test_synthetic_layout_keeps_full_shift_scale_and_watermark(tmp_path,monkeypatch):
    import matplotlib.figure
    captured=[];original=matplotlib.figure.Figure.text
    def record(self,*args,**kwargs):
        if len(args)>2:captured.append(args[2])
        return original(self,*args,**kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure,'text',record)
    summary=synthetic_summary()
    for row in summary['tables']['phase']:
        if row['edit']!='none':row['phase_mae_deg']=.005
    layout=report.render(summary,tmp_path)
    assert all(v[0]==0 and v[1]>=30 for v in layout['phase_ylim'].values())
    assert layout['phase_ylim']['raw']==layout['phase_ylim']['relative_rms']
    assert captured.count('SYNTHETIC LAYOUT — NOT RESULTS')==2
    assert {p.name for p in tmp_path.iterdir()}=={'recovery_and_phase_repair.png','recovery_and_phase_repair.pdf','paired_navigation.png','paired_navigation.pdf'}
