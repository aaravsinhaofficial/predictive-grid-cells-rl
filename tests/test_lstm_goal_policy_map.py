"""Frame and alternative-model checks for the original-policy steering map."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lstm_goal_policy_map import ACTION_OFFSET, VARIANTS, fit_field_models


def test_physical_action_offsets_recover_common_world_bearing():
    headings=np.arange(64)*2*np.pi/64-np.pi
    for offset in ACTION_OFFSET.values():
        bearing=.73
        probability=.5+.3*np.cos(headings+offset-bearing)
        moment=np.sum(probability*np.exp(1j*(headings+offset)))/probability.sum()
        assert abs(np.angle(moment)-bearing)<1e-12


def _positions():
    coordinates=np.linspace(-.95,.95,7)
    return np.array([[x,y] for x in coordinates for y in coordinates if np.hypot(x,y)>.4])


def test_displaced_targets_recovered_despite_shared_frame_offset():
    position=_positions(); train=np.arange(len(position))%2==0
    targets=np.array([[-.29,0],[-.09,0],[-.29,.2],[-.49,0],[-.29,-.2]])
    direction={}
    for name,point in zip(VARIANTS,targets):
        angle=np.arctan2(point[1]-position[:,1],point[0]-position[:,0])+.21
        direction[name]=np.column_stack((np.cos(angle),np.sin(angle)))
    fit=fit_field_models(position,direction,np.ones(len(position)),train)
    actual=np.array(list(fit['vary_target_shared_offset']['aimpoints_xy'].values()))
    np.testing.assert_allclose(actual,targets,atol=1e-6)
    assert fit['target_model_heldout_improvement_deg']>0


def test_global_rotation_alternative_is_distinguished_from_target_shift():
    position=_positions(); train=np.arange(len(position))%2==0
    direction={}
    for name,offset in zip(VARIANTS,[-.2,0,.1,.3,-.4]):
        angle=np.arctan2(-position[:,1],-.2-position[:,0])+offset
        direction[name]=np.column_stack((np.cos(angle),np.sin(angle)))
    fit=fit_field_models(position,direction,np.ones(len(position)),train)
    assert fit['target_model_heldout_improvement_deg']<0
    assert fit['shared_target_vary_offset']['mean_heldout_angle_error_deg']<1e-6
