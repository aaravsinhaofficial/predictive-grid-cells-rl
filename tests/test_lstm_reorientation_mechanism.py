import numpy as np
import torch
from scripts.lstm_reorientation_mechanism import motion_path,physical_clearance
from scripts import rl_blind_reorientation as task


def example():
    cfg=dict(arena_half=1.25,action_repeat=4,v_max=.35,rot_max=2.2,goal_radius=.3)
    w=dict(initial_vf=np.array(.1),initial_vs=np.array(-.03),initial_vr=np.array(.2),
        phases=np.array([task.VISIBLE]*15+[task.TURN_A]*7+[task.TURN_B]*9+[task.PROBE]*4),
        actions=np.array([5]*15+[0]*7+[1]*9+[2]*4),endpoint=np.array([0.,0.]),goal_center=np.array([.9,.9]))
    return cfg,w


def test_path_uses_actual_actions_and_brakes_with_safe_rotated_clearance():
    cfg,w=example();path,heading=motion_path(cfg,w)
    assert path.shape==(36,2) and np.isfinite(heading)
    # Native first-turn brake means rotations cause no translational motion.
    np.testing.assert_array_equal(path[16:32],np.repeat(path[15:16],16,axis=0))
    assert .04<np.linalg.norm(path[-1]-path[-5])<.05
    assert physical_clearance(cfg,w)['eligible']


def test_all_angle_envelope_rejects_goal_and_wall_intersections():
    cfg,w=example();w['endpoint']=np.array([1.1,0.]);assert not physical_clearance(cfg,w)['eligible']
    w['endpoint']=np.array([.8,.8]);assert not physical_clearance(cfg,w)['eligible']
