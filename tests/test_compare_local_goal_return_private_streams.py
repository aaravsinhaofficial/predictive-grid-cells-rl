from copy import deepcopy
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_local_goal_return_private_streams import compare_snapshots


def snapshot():
    return dict(seed=294801,batches=1024,contexts=32768,
        generators={k:torch.arange(32,dtype=torch.uint8) for k in ('plan','motor','action','state')})


def test_exact_named_streams_and_counters():
    a=snapshot();result=compare_snapshots(a,deepcopy(a),1024)
    assert result['passed'] and not result['global_worker_rng_compared']
    assert set(result['generators'])=={'plan','motor','action','state'}
    assert all(v['equal'] for v in result['generators'].values())


def test_changed_draw_state_or_counter_cannot_pass():
    a=snapshot();b=deepcopy(a);b['generators']['motor'][3]+=1
    result=compare_snapshots(a,b,1024)
    assert not result['passed'] and not result['generators']['motor']['equal']
    b=deepcopy(a);b['contexts']-=1
    assert not compare_snapshots(a,b,1024)['passed']
    assert not compare_snapshots(a,a,2048)['passed']
