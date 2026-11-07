"""Private random-policy conditional-prefix checks; no trained model outcomes."""
import math

import numpy as np
import pytest
import torch

from code.common import CodeSource, build_population
from scripts import simple_memory_family as family
from scripts import collect_simple_memory_states as collector
from tests.test_collect_simple_memory_states import config, AuditedPickupArena


def fixture(tmp_path,condition='pgc33'):
    torch.set_num_threads(1)
    cfg = config(tmp_path,condition)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(741099)
        policy = collector.build_policy(cfg,'cpu')
        _, _, contexts = collector.collect(policy,cfg,episodes=4,n_envs=4,seed=93101,
            max_gap=3,stride=3,env_class=AuditedPickupArena)
    assert len(contexts)==4
    return cfg,policy,contexts[0]


def test_actual_rotated_history_phi0_closure_common_dark_and_allowed_cue_slots(tmp_path):
    cfg,policy,context = fixture(tmp_path)
    rng = torch.get_rng_state().clone()
    result = family.build_family(policy,cfg,context,'cpu',phases=8)
    assert torch.equal(rng,torch.get_rng_state())
    assert result['states'].shape == (8,512) and result['states'].dtype == torch.float32
    torch.testing.assert_close(result['theta'],torch.arange(8,dtype=torch.float64)*(2*math.pi/8),rtol=0,atol=0)
    meta = result['metadata']
    assert meta['qualified'] and meta['all_angle_physical']['qualified']
    assert meta['replay_phi0']['max_absolute_error'] < 1e-5
    assert meta['closure_2pi']['max_absolute_error'] < 1e-5
    slot=context['selected_slot'];rt=context['runtime']
    expected=torch.cat(rt['state'],1)[slot]
    torch.testing.assert_close(result['states'][0],expected,rtol=1e-4,atol=1e-5)
    dark=result['dark_inputs']
    assert dark.shape==(8,1287) and torch.equal(dark,dark[:1].expand_as(dark))
    assert not dark[:,256:768].any() and torch.equal(dark[:,768:1280],rt['goal'][slot].repeat(8,1))
    assert not dark[:,-1].any()
    pop=build_population(cfg,'cpu');source=CodeSource(cfg,pop,'cpu')
    native=source(rt['obs']['pos'],rt['obs']['dir'],rt['obs']['vel'])[slot]
    assert torch.equal(result['current_codes']['correct'][0],native)
    for name,pulse in result['pulse_inputs'].items():
        assert torch.equal(pulse[:,:256],dark[:,:256]) and torch.equal(pulse[:,768:],dark[:,768:])
        assert torch.equal(pulse[:,256:768],result['current_codes'][name])
        assert torch.equal(result['current_codes'][name][:,pop.gc_idx],result['current_codes']['ordinary'][:,pop.gc_idx])
    assert not torch.equal(result['current_codes']['wrong_plus'],result['current_codes']['correct'])
    for block in range(2):
        expected_scale=result['states'][:,256*block:256*(block+1)].square().mean().sqrt().clamp_min(1e-8)
        assert torch.equal(result['scale'][256*block:256*(block+1)],expected_scale.repeat(256))


def test_grid_pulses_are_exact_shams_and_family_has_no_blind_double_update(tmp_path):
    cfg,policy,context = fixture(tmp_path,'gc100')
    result=family.build_family(policy,cfg,context,'cpu',phases=4)
    assert result['metadata']['qualified'] and result['metadata']['grid_pulse_shams_exact']
    values=list(result['pulse_inputs'].values())
    assert all(torch.equal(values[0],value) for value in values[1:])
    h,c=result['states'][0].split(256)
    with torch.no_grad():
        after=torch.cat(policy.lstm(result['dark_inputs'][:1],(h[None],c[None])),1)[0]
    assert not torch.equal(after,result['states'][0])


def test_all_768_physical_rotations_preserve_lengths_and_exact_identity():
    theta=torch.arange(768,dtype=torch.float64)*2*math.pi/768
    endpoint=torch.tensor([.2,-.1]);point=torch.tensor([.15,-.16]).repeat(768,1)
    result=family.rotate_positions(point,endpoint,theta)
    assert torch.equal(result[0],point[0])
    torch.testing.assert_close((result-endpoint).norm(dim=1),(point-endpoint).norm(dim=1),rtol=0,atol=1e-7)
    closed=family.rotate_positions(point[:1],endpoint,torch.tensor([2*math.pi],dtype=torch.float64))
    assert torch.equal(closed,point[:1])


def test_ineligible_physical_context_cannot_be_repaired_by_neural_family(tmp_path):
    cfg,policy,context=fixture(tmp_path)
    slot=context['selected_slot']
    context['runtime']['obs']['pos']=context['runtime']['obs']['pos'].clone()
    context['runtime']['obs']['pos'][slot]=torch.tensor([1.17,0.])
    with pytest.raises(ValueError,match='physical_context_unqualified'):
        family.build_family(policy,cfg,context,'cpu',phases=4)


def test_actual_native_family_to_behavior_forks_private_grid_integration(tmp_path):
    import random
    from scripts.simple_memory_interventions import behavioral_forks
    cfg,policy,context=fixture(tmp_path,'gc100')
    result=family.build_family(policy,cfg,context,'cpu',phases=24)
    assert result['metadata']['qualified']
    torch_before=torch.get_rng_state().clone(); numpy_before=np.random.get_state(); python_before=random.getstate()
    forks=behavioral_forks(policy,cfg,result,device='cpu',tapes=2,horizon=3)
    assert forks['complete'] and forks['validation_qualified']
    assert forks['validations']['no_op_native_continuation']['qualified']
    assert forks['validations']['grid_pulses_exact_sham']
    assert len(forks['records'])==50
    for row in forks['records']:
        if row['qualified']:
            assert row['other_slots_identical_for_compared_steps']==3
    assert torch.equal(torch_before,torch.get_rng_state())
    numpy_after=np.random.get_state()
    assert numpy_before[0]==numpy_after[0] and np.array_equal(numpy_before[1],numpy_after[1]) and numpy_before[2:]==numpy_after[2:]
    assert python_before==random.getstate()
