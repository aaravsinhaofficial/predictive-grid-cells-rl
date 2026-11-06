import copy
import numpy as np
import pytest
from scripts import summarize_reorientation_reward_allocation as audit


def example():
    variant='blind_native';original=[dict(slot=i,episode=0,n_goals=goals) for i,goals in enumerate([4,0,2])]
    free=[dict(slot=0,episode=0,event=1,goal_hit=True,free_updates=25)]
    windows=[dict(slot=i,episode=0,event=j,end_phase=phase,goal_hit=True,episode_end=False)
        for i,j,phase in [(0,1,'free_blind'),(0,2,'visible'),(2,1,'turn_a')]]
    schedule={k:[value]*12 for k,value in dict(admitted=3,completed=0,interrupted_reward=3,interrupted_done=0,coverage_violations=0).items()}
    schedule['phase_update_counts']=[[3900,40,20,15,10,40,25]]*12
    return dict(episodes_per_variant=3,variants={variant:dict(rows=original)},
        free_window={variant:dict(rows=free,free_hits=1,forced_hits=1,free_updates=25)},
        window_rows={variant:windows},schedule=schedule)


def test_goal_accounting_preserves_zero_event_episodes_and_all_rewards():
    result=audit.summarize_variant(example(),'blind_native')
    assert len(result['episode_rows'])==3 and result['episodes_without_event']==1
    assert result['totals']['waiting_goals']==3
    assert result['totals']['visible_goals']==result['totals']['forced_goals']==result['totals']['free_goals']==1
    assert sum(result['goals_per_original_episode'].values())==2
    assert result['goals_per_100_corresponding_updates']['free']==4


def test_inconsistent_free_counter_is_rejected():
    data=example();data['free_window']['blind_native']['free_hits']=2
    with pytest.raises(AssertionError):audit.summarize_variant(data,'blind_native')


def test_entire_zero_discovery_evaluation_is_retained_with_undefined_free_rate():
    data=example()
    for row in data['variants']['blind_native']['rows']:row['n_goals']=0
    data['free_window']['blind_native']=dict(rows=[],free_hits=0,forced_hits=0,free_updates=0)
    data['window_rows']['blind_native']=[]
    for key in data['schedule']:
        data['schedule'][key]=([[4050,0,0,0,0,0,0]]*12 if key=='phase_update_counts' else [0]*12)
    result=audit.summarize_variant(data,'blind_native')
    assert result['episodes']==result['episodes_without_event']==3
    assert result['total_goals_per_original_episode']==0
    assert result['goals_per_100_corresponding_updates']['free'] is None


def test_training_differences_exclude_warmup_and_use_matching_phase_denominators():
    before=dict(phase_counts=np.full(7,100),free_hits=50,forced_hits=30,schedule={'admitted':10})
    after=copy.deepcopy(before);after['phase_counts']+=np.array([20,10,5,5,5,5,50]);after['free_hits']+=2;after['forced_hits']+=1;after['schedule']['admitted']+=2
    result=audit.training_difference(before,after,100)
    assert result['free_goal_hits']==2 and result['free_goals_per_100_free_updates']==4
    assert result['forced_goals_per_100_forced_updates']==5
    assert result['total_training_goals'] is None
