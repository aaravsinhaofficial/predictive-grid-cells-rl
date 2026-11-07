import torch
from code.memory_task import MemoryClock, gate_observation, max_gap_at


def test_curriculum_exact_boundaries():
    assert max_gap_at(0) == max_gap_at(3_333_333) == 0
    assert max_gap_at(3_333_334) == 1
    assert max_gap_at(100_000_000) == 30
    assert max_gap_at(200_000_000) == max_gap_at(1_000_000_000) == 60
    assert max_gap_at(1_000_000_000, max_gap=0) == 0


def test_fifteen_visible_then_exact_gap_and_native_hit_reset():
    clock = MemoryClock(1, 'cpu', 9)
    no = torch.tensor([False]); yes = ~no
    for _ in range(20):
        clock.advance(no, no, 4); assert not clock.mask().item()
    clock.advance(yes, no, 4)
    for i in range(15):
        assert not clock.mask().item()
        clock.advance(no, no, 4)
    assert clock.mask().item()
    duration = clock.remaining.item(); assert 2 <= duration <= 4
    for _ in range(duration):
        assert clock.mask().item(); clock.advance(no, no, 4)
    assert not clock.mask().item() and clock.remaining.item() == 15
    clock.advance(yes, yes, 4)
    assert not clock.known.item() and not clock.mask().item()


def test_gate_masks_all_current_slots_without_touching_native_observation():
    rgb = torch.rand(2, 3, 84, 84); code = torch.rand(2, 512)
    before = (rgb.clone(), code.clone())
    a, b = gate_observation(rgb, code, torch.tensor([False, True]))
    assert torch.equal(a[0], rgb[0]) and torch.equal(b[0], code[0])
    assert a[1].count_nonzero() == b[1].count_nonzero() == 0
    assert torch.equal(rgb, before[0]) and torch.equal(code, before[1])


def test_private_clock_restore_and_branch_draws():
    clock = MemoryClock(3, 'cpu', 47, groups=2)
    global_before = torch.get_rng_state().clone()
    events = torch.tensor([True, False, True] * 2); no = torch.zeros(6, dtype=torch.bool)
    clock.advance(events, no, 60)
    saved = clock.state_dict()
    expected = []
    for _ in range(45):
        clock.advance(no, no, 60); expected.append(clock.mask().clone())
    restored = MemoryClock(3, 'cpu', 999, groups=2); restored.load_state_dict(saved)
    for mask in expected:
        restored.advance(no, no, 60)
        assert torch.equal(mask, restored.mask()) and torch.equal(mask[:3], mask[3:])
    assert torch.equal(global_before, torch.get_rng_state())
    assert torch.equal(clock.generator.get_state(), restored.generator.get_state())


def test_native_control_never_blinds_or_consumes_duration_rng():
    clock = MemoryClock(2, 'cpu', 73)
    rng = clock.generator.get_state().clone()
    for t in range(100):
        clock.advance(torch.tensor([t % 11 == 0, t % 13 == 0]), torch.tensor([t % 23 == 0, False]), 0)
        assert not clock.mask().any()
    assert clock.draw_steps == 0 and torch.equal(clock.generator.get_state(), rng)
