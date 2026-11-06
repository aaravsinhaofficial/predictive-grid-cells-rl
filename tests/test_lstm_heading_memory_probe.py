import numpy as np
import torch

from scripts.lstm_heading_memory_probe import (
    candidate_gates, fit_readout, make_env, periodic_tangent, readout_metrics,
)
from code.common import build_population


def test_fixed_readout_tracks_common_physical_rotation_on_heldout_angles():
    angle=np.arange(96)*2*np.pi/96
    initial=np.column_stack((np.cos(angle),np.sin(angle),np.zeros_like(angle)))
    decoder=fit_readout(initial,angle)
    turn=1.7
    later=np.column_stack((np.cos(angle+turn),np.sin(angle+turn),np.zeros_like(angle)))
    corrected=readout_metrics(later,angle,decoder,turn)
    uncorrected=readout_metrics(later,angle,decoder)
    assert corrected['heldout_mae_deg'] < 1e-8
    assert uncorrected['heldout_mae_deg'] > 90
    assert corrected['decoded_phase_coverage_fraction'] > .97


def test_information_refit_cannot_substitute_for_remaining_state_amplitude():
    angle=np.arange(96)*2*np.pi/96
    initial=np.column_stack((np.cos(angle),np.sin(angle)))
    collapsed=initial*1e-5
    fixed=readout_metrics(collapsed,angle,fit_readout(initial,angle))
    refit=readout_metrics(collapsed,angle,fit_readout(collapsed,angle))
    assert refit['heldout_mae_deg'] < 1e-8
    assert fixed['heldout_amplitude_at_least_point1_fraction'] == 0
    row=dict(scaled_radius_retention=1e-5,fixed_readout_movement_corrected=fixed)
    rec={'latent_tangent_total_gain':{'median':1.},'normal_transverse_gain':{'median':.1},'tangent_over_transverse':10.}
    gates=candidate_gates({'75':row},{'75':{'horizons':{'15':rec}}})
    assert not gates['radius_after_5s_at_least_20_percent']
    assert not gates['fixed_readout_amplitude_valid_in_80_percent']


def test_periodic_curve_tangent_wraps_without_endpoint_discontinuity():
    angle=torch.arange(192,dtype=torch.float64)*2*np.pi/192
    state=torch.stack((angle.cos(),angle.sin()),1)
    tangent=periodic_tangent(state,2*np.pi/192)
    expected=torch.stack((-angle.sin(),angle.cos()),1)
    assert torch.nn.functional.cosine_similarity(tangent,expected).min() > .999999999


def test_stationary_native_turn_changes_heading_but_not_pgc_code():
    cfg=dict(arena_half=1.25,episode_len=1350,action_repeat=4,v_max=.35,rot_max=2.2,
             goal_radius=.30,vision='raycast',img_noise=.02,randomize_textures=1,
             n_cells=512,n_modules=4,spacing_min=.4,spacing_ratio=1.42,pgc_frac=.33,
             pgc_shift=.2,pgc_shift_mode='space',pgc_tau=.5,pgc_shift_jitter=0.,
             sharpness=2.,code_noise=0.,code_seed=1)
    archive=dict(history_pos=np.array([[.1,.1]],dtype=np.float32),
                 history_goal_center=np.array([[-.7,.2]],dtype=np.float32),
                 history_direction=np.array([[0.,1.]],dtype=np.float32),
                 history_landmark_pos=np.array([-.6,.3,.8,-.2],dtype=np.float32),
                 history_stripe_phase=np.array([0.,.25,.5,.75],dtype=np.float32),
                 history_hd=np.array([.2],dtype=np.float32))
    env=make_env(cfg,archive,16,0);env.vision='none'
    pop=build_population(cfg,'cpu')
    pos=env.pos.clone();mdir=env.mdir.clone();hd=env.hd.clone()
    initial=pop.activity(pos,mdir,torch.zeros_like(pos))
    for _ in range(30):
        _,reward,done,info=env.step(torch.zeros(16,dtype=torch.long))
        assert not bool(done.any() or info['hit'].any() or reward.any())
    torch.testing.assert_close(env.pos,pos,atol=0,rtol=0)
    torch.testing.assert_close(env.mdir,mdir,atol=0,rtol=0)
    torch.testing.assert_close(pop.activity(env.pos,env.mdir,torch.zeros_like(pos)),initial,atol=0,rtol=0)
    assert float((env.hd-hd).abs().max()) > .5
