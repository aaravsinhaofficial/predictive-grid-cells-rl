import torch
from scripts.preflight_local_goal_return_v2 import common_draws,UniformPolicy


def test_body_frame_noise_and_action_uniforms_are_shared_across_heading_pairs():
    motor,uniform=common_draws(0)
    assert motor.shape==(30,3,16) and uniform.shape==(30,16)
    assert torch.equal(motor,motor[:,:,:1].expand_as(motor))
    assert torch.equal(uniform,uniform[:,:1].expand_as(uniform))
    assert not torch.equal(motor,common_draws(1)[0])


def test_uniform_control_has_no_heading_or_history_dependency():
    net=UniformPolicy();h=torch.randn(16,256);c=torch.randn_like(h)
    assert torch.equal(net.pi(h),torch.zeros(16,6))
    assert net.lstm(torch.randn(16,1287),(h,c))[0] is h
