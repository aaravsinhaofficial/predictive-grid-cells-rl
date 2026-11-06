"""The sampling audit must leave the frozen sampler and all other RNGs unchanged."""
import torch
from scripts.eval_visual_memory_matched_noise import AuditedNoise
from scripts.rl_visual_memory_noise import StateNoise


def test_raw_noise_audit_is_exact_and_independent_of_masks_or_state_values():
    before = torch.get_rng_state().clone()
    original = StateNoise('cpu', seed=128201)
    first = AuditedNoise('cpu', seed=128201)
    second = AuditedNoise('cpu', seed=128201)
    state = (torch.zeros(6, 5), torch.ones(6, 5))
    mask = torch.tensor([True, False, True, False, True, False])
    for step in range(7):
        a = original.draw(state, mask, groups=3)
        b = first.draw(state, mask, groups=3)
        second.draw(tuple(value + step for value in state), ~mask, groups=3)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert torch.equal(original.generator.get_state(), first.generator.get_state())
    assert first.audit() == second.audit()
    assert first.audit()['draws'] == 7
    assert torch.equal(before, torch.get_rng_state())
