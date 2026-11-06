"""Synthetic accounting checks only; never read live model outcomes."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import plot_local_goal_return_training as report


def fake_evaluation(native=167, erased=128, certified=128):
    rows=[];variants={}
    for variant in report.VARIANTS:
        count=erased if variant=='blind_reset_after_glimpse' else native
        local=[dict(context=i//16,heading_index=i%16,variant=variant,hit=i<count) for i in range(256)]
        rows.extend(local)
        variants[variant]=dict(trials=256,successes=count,success_fraction=count/256,invalid_trials=0)
    numeric=native>=167 and native-erased>=39
    return dict(complete=True,bank='final_hard',variants=variants,rows=rows,all_physical_checks=True,
        gates=dict(near_baseline_pass=False,hard_numeric_pass=numeric,
                   hard_certified_memory_pass=numeric and certified==128),
        certificate=dict(equally_weighted_pairs=128,heading_trials=256,certified_pairs=certified,
            uncertified_pairs=128-certified,all_pairs_certified=certified==128,
            common_information_success_upper_bound=(.5*certified+128-certified)/128))


def test_all_contexts_retained_and_gate_exact_integer_boundaries():
    result=report.aggregate_evaluation(fake_evaluation())
    assert result['postcue_erasure_cost_successes']==39
    assert result['gates']['hard_certified_memory_pass']
    for native,erased in ((166,127),(167,129)):
        assert not report.aggregate_evaluation(fake_evaluation(native,erased))['gates']['hard_numeric_pass']
    incomplete=report.aggregate_evaluation(fake_evaluation(certified=127))
    assert incomplete['gates']['hard_numeric_pass'] and not incomplete['gates']['hard_certified_memory_pass']
    assert incomplete['common_information_success_upper_bound']>.5
    assert all(v['contexts']==16 and v['trials']==256 for v in result['variants'].values())


def test_duplicate_trial_and_wrong_reported_count_are_rejected():
    duplicate=fake_evaluation();duplicate['rows'][0]=dict(duplicate['rows'][1])
    with pytest.raises(AssertionError):report.aggregate_evaluation(duplicate)
    wrong=fake_evaluation();wrong['variants']['blind_native']['successes']+=1
    with pytest.raises(AssertionError):report.aggregate_evaluation(wrong)


def test_invalid_trials_stay_in_denominator_and_require_failure():
    rows=[dict(context=i//16,heading_index=i%16,hit=False,invalid=i<16) for i in range(256)]
    stats=report.summarize_rows(rows)
    assert stats['trials']==256 and stats['contexts']==16 and stats['valid_contexts']==15
    assert stats['invalid_trials']==16 and stats['success_fraction']==0
    rows[0]['hit']=True
    with pytest.raises(AssertionError):report.summarize_rows(rows)


def test_training_windows_do_not_mix_stages_and_weight_actual_decisions():
    rows=[]
    for i in range(4096):
        stage=i//1024+1;odd=i%2
        rows.append(dict(update=i+1,stage=stage,completed_trials=32,free_hits=stage,
            free_updates=32 if odd else 64,entropy_per_free_decision=1. if odd else 0.))
    result=report.training_windows(rows)
    for stage,window in enumerate(result,1):
        assert window['updates'][0]==(stage-1)*1024+32 and window['updates'][-1]==stage*1024
        assert len(window['updates'])==993
        assert all(v==stage/32 for v in window['success_fraction'])
        assert all(abs(v-1/3)<1e-12 for v in window['entropy_per_decision'])
    missing=deepcopy(rows);missing[3]['update']=3
    with pytest.raises(AssertionError):report.training_windows(missing)
