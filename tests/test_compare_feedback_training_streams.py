from pathlib import Path
import copy,sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_feedback_training_streams import compare

def fixture():
    state=dict(seed=494801,batches=1024,contexts=32768,generators={k:torch.tensor([1,2,3],dtype=torch.uint8) for k in ('plan','motor','action','state')})
    return {str(i):copy.deepcopy(state) for i in range(4)}

def test_exact_streams_and_single_byte_corruption():
    streams=fixture();assert all(compare(streams).values())
    streams['3']['generators']['motor'][1]=9
    checks=compare(streams);assert checks==dict(action=True,motor=False,plan=True,state=True)

def test_wrong_draw_count_rejected():
    streams=fixture();streams['1']['contexts']-=32
    with pytest.raises(AssertionError):compare(streams)
