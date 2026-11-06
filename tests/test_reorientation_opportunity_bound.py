import itertools
import math
import numpy as np
from scripts.audit_reorientation_mechanism_outputs import path_upper_bound


def test_noiseless_straight_motion_attains_speed_bound():
    n=30;dt=4/60;tau=.2;v=.35
    upper=path_upper_bound(np.zeros((n,3,2)),v,.03,dt,tau)
    exact=sum(dt*v*(1-math.exp(-dt/tau)**(t+1)) for t in range(n))
    np.testing.assert_allclose(upper,exact,atol=1e-14)


def test_bound_covers_every_short_action_sequence_with_fixed_noise():
    motor=np.random.default_rng(12).normal(size=(4,3,1));v=.35;dt=4/60;tau=.2;k=1-math.exp(-dt/tau)
    upper=path_upper_bound(motor,v,.03,dt,tau)[0]
    targets=np.array([[0,0],[0,0],[v,0],[-v,0],[0,v],[0,-v]])
    for actions in itertools.product(range(6),repeat=4):
        velocity=np.zeros(2);distance=0.
        for t,action in enumerate(actions):
            velocity=(1-k)*velocity+k*(targets[action]+.03*motor[t,:2,0])
            distance+=dt*np.linalg.norm(velocity)
        assert distance<=upper+1e-14
