import torch
from scripts.lstm_travel_memory_heldout import VisibleWindowClock


def test_original_context_window_clock_never_removes_vision():
    clock=VisibleWindowClock(1,'cpu',17)
    yes=torch.tensor([True]);no=~yes
    assert not bool(clock.mask().any())
    assert not bool(clock.advance(yes,no,60).any())
    for t in range(1,46):
        assert not bool(clock.mask().any())
        completed=clock.advance(no,no,60)
        assert bool(completed[0]) == (t%15==0)
    clock.advance(no,yes,60)
    assert not bool(clock.known.any()) and not bool(clock.mask().any())
