import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from native_movement_bounded_pulses import bounded_case


def test_clip_only_h_then_shrink_all_pulses_preserves_norm_and_bounds():
    gen=torch.Generator().manual_seed(8213);incoming=torch.cat((torch.full((256,),.999),torch.arange(256,dtype=torch.float32)/10))
    pulses=torch.cat((torch.zeros(1,512),torch.randn(6,512,generator=gen)*.02))
    scale=torch.ones(512);case=dict(pulses=pulses,scale=scale,metadata={})
    result=bounded_case(case,incoming);record=result['metadata']['bound_robustness']
    assert torch.equal(case['pulses'],pulses)and torch.equal(result['pulses'][0],torch.zeros(512))
    assert all(x<=1 for x in record['shrink_factors'])
    assert record['h_bound_exact']and record['c_displacement_unchanged_during_clipping']
    assert record['realized_norm_max_relative_error']<2e-5
    for j in range(1,7):
        torch.testing.assert_close(result['pulses'][j,256:],pulses[j,256:]*record['shrink_factors'][j],rtol=1e-6,atol=1e-8)
    assert any(v<.99 for v in record['cosine_with_original'])
