import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from report_native_movement_pulses import joint_availability


def test_joint_cohort_censors_both_streams_and_direction_comparisons():
    active=torch.ones(30,21,dtype=torch.bool)
    active[10:,8]=False  # ordinary turn-plus ended early
    assert joint_availability(active,30,15,1)==(False,False,False)
    assert joint_availability(active,30,15,5)==(True,True,False)
    assert joint_availability(active,30,5,1)==(True,True,True)
    assert joint_availability(active,4,5,1)==(True,False,False)
