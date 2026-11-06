from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from original_goal_ring_history_control import arrival_from_decoded_positions


def test_direction_retains_last_moving_displacement_without_heading_oracle():
    points=np.array([[0.,0.],[.01,0.],[.01,.02],[.0101,.0201]])
    direction,last=arrival_from_decoded_positions(points)
    np.testing.assert_allclose(direction,[0,1])
    assert last==1


def test_unavailable_history_is_explicit_not_replaced_by_truth():
    direction,last=arrival_from_decoded_positions(np.array([[0.,0.],[.0001,.0002]]))
    assert direction is None and last is None
