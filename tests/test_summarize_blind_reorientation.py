import copy

import numpy as np
import pytest

from scripts.summarize_blind_reorientation import episode_values, clustered_contrast


def fixture():
    episodes = [dict(slot=s, episode=e) for s in range(3) for e in range(2)]
    events = [dict(slot=s, episode=0, goal_hit=True, progress_m=.4,
                   path_m=.7, free_updates=20) for s in range(3)]
    return dict(episodes_per_variant=6,
        variants={v:dict(rows=copy.deepcopy(episodes)) for v in ('native', 'ordinary')},
        free_window=dict(
            native=dict(rows=events, free_hits=3, free_progress_m=1.2,
                        free_path_m=2.1, free_updates=60),
            ordinary=dict(rows=[], free_hits=0, free_progress_m=0.,
                          free_path_m=0., free_updates=0)))


def test_zero_window_episodes_remain_in_denominator_and_pair_by_key():
    result = fixture()
    result['variants']['ordinary']['rows'].reverse()
    keys, values = episode_values(result, 'native')
    assert len(keys) == 6 and np.count_nonzero(values[:, 0]) == 3
    estimate = clustered_contrast(result, 'native', 'ordinary')
    assert estimate['mean'] == .5 and estimate['ci95'] == [.5, .5]
    assert estimate['paired_slots'] == 3


def test_window_partition_does_not_create_more_replicates_or_change_hit_effect():
    result = fixture()
    expected = clustered_contrast(result, 'native', 'ordinary')
    split = []
    for row in result['free_window']['native']['rows']:
        for hit in (True, False):
            split.append(dict(row, goal_hit=hit, progress_m=.2, path_m=.35, free_updates=10))
    result['free_window']['native']['rows'] = split
    assert clustered_contrast(result, 'native', 'ordinary') == expected


def test_unmatched_or_duplicated_episode_is_rejected():
    result = fixture()
    result['variants']['ordinary']['rows'][0] = dict(slot=9, episode=0)
    with pytest.raises(AssertionError):
        clustered_contrast(result, 'native', 'ordinary')
    result = fixture()
    result['variants']['ordinary']['rows'][0] = result['variants']['ordinary']['rows'][1]
    with pytest.raises(AssertionError):
        episode_values(result, 'ordinary')
