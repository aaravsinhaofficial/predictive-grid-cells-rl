"""Identity and information-boundary checks for the explicitly designed ring."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import original_rl_goal_ring as m
from code.common import build_population
import json


def population():
    cfg=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    return build_population(cfg,'cpu')


def test_ring_has_one_neutral_phase_and_one_contracting_radial_direction():
    torch.set_num_threads(2)
    theta=torch.tensor(.7,dtype=torch.float64)
    z=torch.stack((theta.cos(),theta.sin()))
    jac=torch.autograd.functional.jacobian(lambda y:m.ring_step(y[None])[0],z)
    radial=z;tangent=torch.stack((-z[1],z[0]))
    assert torch.allclose(jac@radial,.9*radial,atol=1e-12)
    assert torch.allclose(jac@tangent,tangent,atol=1e-12)
    m.validate_ring()


def test_map_inverse_recovers_independent_positions_without_position_initialization():
    pop=population();ordinary=m.MapInverse(pop,(~pop.is_pgc).nonzero().flatten());predictive=m.MapInverse(pop,pop.is_pgc.nonzero().flatten())
    result=m.validate_inverse(pop,ordinary,predictive)
    assert all(r['max_position_error_m']<2e-5 for r in result.values())


def test_predictive_bank_decodes_ahead_point_not_pickup_or_heading_label():
    pop=population();ordinary=m.MapInverse(pop,(~pop.is_pgc).nonzero().flatten());predictive=m.MapInverse(pop,pop.is_pgc.nonzero().flatten())
    pos=torch.tensor([[.65,-.29],[-.1,.58],[-.72,-.12]])
    angle=torch.tensor([torch.pi/2,-.4,1.8]);direction=torch.stack((angle.cos(),angle.sin()),1)
    observed=pop.activity(pos,direction,direction*.35)
    pickup,_=ordinary.decode(observed[:,ordinary.indices]);ahead,_=predictive.decode(observed[:,predictive.indices])
    assert torch.allclose(pickup,pos,atol=2e-5)
    assert torch.allclose(ahead,pos+.2*direction,atol=2e-5)
    assert not torch.allclose(ahead,pos,atol=.01)


def test_phase_pulse_changes_only_angle_and_has_inverse():
    z=torch.tensor([[.6,.8],[-.8,.6]],dtype=torch.float64)
    pulse=torch.tensor([torch.pi/4,-torch.pi/4],dtype=torch.float64)
    altered=m.phase_rotate(z,pulse)
    assert torch.allclose(altered.norm(dim=1),z.norm(dim=1),atol=1e-13)
    assert torch.allclose(m.phase_rotate(altered,-pulse),z,atol=1e-13)
    assert torch.allclose(m.ring_step(z),z,atol=1e-13)
