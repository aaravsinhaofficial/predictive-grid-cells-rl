"""Synthetic corruption checks for the unfrozen saved-data report."""
import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import report_native_turn_distillation as report


def fixture():
    rows=[dict(slot=i,score=10.,n_goals=1,first_goal_step=10,path_eff=0.,path_eff_n=0)for i in range(16)]
    names=report.evaluation.LIVE_VARIANTS;phase=[20000,1000,100,100,100,100,200]
    return dict(complete=True,episodes_per_variant=16,
        variants={v:report.episode_summary(copy.deepcopy(rows),1350)for v in names},
        exposure={v:dict(choice=200,other_control=21000,forced=400)for v in names},
        mean_teacher_kl={v:dict(choice=.5,other_control=.2,forced=1.)for v in names},
        teacher_kl_sums={v:dict(choice=100.,other_control=4200.,forced=400.)for v in names},
        event_clock=dict(phase_update_counts=[phase.copy()for _ in names],admitted=[10]*6),
        input_audit=dict(scheduled_blind_updates=[600.]*6,pulses=[10.]*6),
        choice_erasure_counts={v:200 if'reset'in v else 0 for v in names},
        reward_by_phase={v:[160.,0.,0.,0.,0.,0.,0.]for v in names})


def test_valid_live_denominators_and_shams():
    result=report.live_summary(fixture(),'gc100')
    assert result['native_blind_erasure_cost']==result['visible_erasure_cost']==0


def test_redistributed_phase_counts_cannot_pass_total_only_check():
    result=fixture();result['event_clock']['phase_update_counts'][0][0]-=1
    result['event_clock']['phase_update_counts'][0][6]+=1
    with pytest.raises(AssertionError):report.live_summary(result,'pgc33')


def test_discovery_reward_and_erasure_corruptions_are_rejected():
    result=fixture();value=result['variants']['blind_native'];value['rows'][0]['first_goal_step']=11
    result['variants']['blind_native']=report.episode_summary(value['rows'],1350)
    with pytest.raises(AssertionError):report.live_summary(result,'pgc33')
    result=fixture();result['reward_by_phase']['blind_native'][0]+=10
    with pytest.raises(AssertionError):report.live_summary(result,'pgc33')
    result=fixture();result['choice_erasure_counts']['seen_native']=1
    with pytest.raises(AssertionError):report.live_summary(result,'pgc33')


def test_grid_sham_kl_corruption_even_with_consistent_means_is_rejected():
    result=fixture();result['mean_teacher_kl']['blind_ordinary']['choice']=.6
    result['teacher_kl_sums']['blind_ordinary']['choice']=120.
    report.live_summary(result,'pgc33')  # No universal cross-population identity requirement.
    with pytest.raises(AssertionError):report.live_summary(result,'gc100')
    result=fixture();result['input_audit']['pulses'][3]+=1
    with pytest.raises(AssertionError):report.live_summary(result,'gc100')


def test_only_complete_successful_matching_execution_record_is_accepted(tmp_path,monkeypatch):
    monkeypatch.setattr(report.train,'BASE',tmp_path)
    # Use a path under the actual repository to exercise portable relative links.
    monkeypatch.setattr(report,'ROOT',tmp_path)
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    monkeypatch.setattr(report.train,'PROTOCOL',protocol)
    path=tmp_path/'gc100_clean_execution.json'
    value=dict(condition='gc100',arm='clean',complete=True,exit_code=0,no_cloud=True,
        protocol_sha256=report.train.dist.sha(protocol))
    path.write_text(json.dumps(value));assert report.complete_execution('gc100','clean')['sha256']
    for change in ({'complete':False},{'exit_code':1},{'protocol_sha256':'wrong'},{'arm':'noise'}):
        path.write_text(json.dumps({**value,**change}))
        with pytest.raises(AssertionError):report.complete_execution('gc100','clean')
