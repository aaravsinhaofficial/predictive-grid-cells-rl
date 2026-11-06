import torch
from scripts.rl_visual_memory import VisualClock
from scripts.lstm_endogenous_every_blind_reset import EveryBlindUpdateClock


def test_every_blind_reset_matches_next_input_schedule_and_preserves_clock_rng():
    original=VisualClock(3,'cpu',31415)
    repeated=EveryBlindUpdateClock(3,'cpu',31415)
    before_next=torch.zeros(3,dtype=torch.bool)
    total_reset=0;total_blind=0
    for t in range(180):
        # The evaluator resets after the previous action, before this input.
        assert torch.equal(before_next,repeated.mask())
        total_blind+=int(repeated.mask().sum())
        hit=torch.tensor([t in (5,70),t==12,False])
        done=torch.tensor([t==120,t==155,False])
        entering=original.advance(hit,done,60)
        before_next=repeated.advance(hit,done,60)
        assert not before_next[2]  # This slot never discovers its goal.
        assert torch.equal(before_next,repeated.mask())
        assert bool((entering<=before_next).all())
        for name in ('known','blind','remaining'):
            assert torch.equal(getattr(original,name),getattr(repeated,name))
        assert torch.equal(original.generator.get_state(),repeated.generator.get_state())
        total_reset+=int(before_next.sum())
    assert total_blind>30
    assert abs(total_reset-total_blind)<=3
