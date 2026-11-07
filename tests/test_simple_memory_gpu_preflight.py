"""Private fixture checks; these do not claim CUDA has been validated locally."""
import copy
import time

import numpy as np
import pytest
import torch

from code.memory_task import MemoryClock
from scripts.simple_memory_gpu_preflight import categorical_equivalence, exact_tree, force_known, resume_fixture


def test_native_shaped_categorical_equivalence_preserves_cpu_rng():
    before = torch.get_rng_state().clone()
    result = categorical_equivalence('cpu')
    assert result['complete'] and result['global_rng_restored']
    assert [case['shape'] for case in result['cases']] == [[32, 6], [100, 32, 6]]
    assert all(case['bitwise_equal'] for case in result['cases'])
    assert torch.equal(before, torch.get_rng_state())


def test_exact_tree_rejects_changed_state_dtype_or_rng():
    original = {'state': (torch.tensor([1., 2.]), np.array([3, 4], dtype=np.uint32)), 'config': {'seed': 90731}}
    assert exact_tree(original, copy.deepcopy(original)) == 3
    changed = copy.deepcopy(original); changed['state'][0][0] += .001
    with pytest.raises(AssertionError, match='tensor values differ'):
        exact_tree(original, changed)
    changed = copy.deepcopy(original); changed['state'][1][0] += 1
    with pytest.raises(AssertionError):
        exact_tree(original, changed)
    changed = copy.deepcopy(original); changed['state'] = (changed['state'][0].double(), changed['state'][1])
    with pytest.raises(AssertionError):
        exact_tree(original, changed)


def test_forced_known_benchmark_preserves_other_clock_state():
    clock = MemoryClock(2, 'cpu', 90731)
    clock.remaining[:] = torch.tensor([5, 8]); clock.blind[1] = True
    rng = clock.generator.get_state().clone()
    assert force_known({'clock': clock}) == 2
    assert clock.known.all() and clock.remaining.tolist() == [5, 8]
    assert clock.blind.tolist() == [False, True]
    assert torch.equal(rng, clock.generator.get_state())
    assert force_known({'clock': clock}) == 0


def test_private_disk_resume_fixture_cpu_path(tmp_path):
    result = resume_fixture(tmp_path, 'gc100', 'cpu', time.monotonic()+60)
    assert result['complete'] and result['bitwise_resume_equal']
    assert not result['research_model']
    assert result['changed_parameter_names']
    assert result['private_seed'] == 90731
    assert (tmp_path/'technical_resume_fixture.pt').exists()
