import copy
import pytest
from scripts.summarize_reorientation_entropy import paired_components, worker_folder


def fixture():
    episodes=[dict(slot=s,episode=e) for s in range(3) for e in range(2)]
    events=[dict(slot=s,episode=0,goal_hit=True,progress_m=.4,path_m=.7,free_updates=20) for s in range(3)]
    return dict(episodes_per_variant=6,
        variants={v:dict(rows=copy.deepcopy(episodes)) for v in ('native','reset')},
        free_window=dict(native=dict(rows=events,free_hits=3,free_progress_m=1.2,free_path_m=2.1,free_updates=60),
                         reset=dict(rows=[],free_hits=0,free_progress_m=0.,free_path_m=0.,free_updates=0)))


def test_cross_policy_contrast_retains_zero_episodes_and_paired_components():
    a=fixture();b=fixture()
    result=paired_components([(1,a,'native'),(-1,b,'reset')],'free_hits')
    assert result['mean']==.5 and result['ci95']==[.5,.5]
    assert result['episodes']==6 and result['paired_slots']==3
    result=paired_components([(1,a,'native'),(-1,a,'reset'),(-1,b,'native'),(1,b,'reset')],'free_hits')
    assert result['mean']==0 and result['ci95']==[0,0]
    b['variants']['reset']['rows'][0]=dict(slot=99,episode=0)
    with pytest.raises(AssertionError):paired_components([(1,a,'native'),(-1,b,'reset')],'free_hits')


def test_explicit_actual_worker_child_routing():
    row=dict(name='ring-20261105-example-02',artifact_subdir='experiment')
    assert str(worker_folder(row)).endswith('/ring-20261105-example-02/experiment')
    for child in ('../other','/absolute','unrecorded'):
        with pytest.raises(AssertionError):worker_folder(dict(row,artifact_subdir=child))
