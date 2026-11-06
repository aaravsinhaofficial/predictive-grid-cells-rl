"""A saved continuation must reproduce subsequent simulator and optimizer updates."""
import copy
import io

import numpy as np
import torch

from code.common import CodeSource, build_population
from scripts import rl_visual_memory_extend as extension
from tests.test_rl_goal_memory import TinyPolicy, config


def test_full_snapshot_reproduces_uninterrupted_training_exactly():
    torch.set_num_threads(2)
    torch.manual_seed(821)
    np.random.seed(821)
    cfg = config()
    cfg.update(vision='raycast', img_noise=.02, episode_len=45)
    net = TinyPolicy()
    opt = torch.optim.RMSprop(net.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    runtime = extension.fresh_runtime(net, cfg, 'cpu')
    runtime['env'].t.zero_()
    runtime['env'].partial.zero_()
    runtime['env'].pos[:] = runtime['env'].goal
    runtime['obs'] = runtime['env'].observe()
    source = CodeSource(cfg, build_population(cfg, 'cpu'), 'cpu')
    for _ in range(4):
        extension.update(net, cfg, opt, source, runtime)
    assert runtime['clock'].mask().any()
    assert runtime['goal'].saved.any()
    checkpoint = dict(policy=copy.deepcopy(net.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                      snapshot=extension.capture_runtime(runtime, 'cpu'))
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    expected_metrics = [extension.update(net, cfg, opt, source, runtime) for _ in range(5)]
    expected_runtime = extension.capture_runtime(runtime, 'cpu')

    restored_net = TinyPolicy()
    restored_net.load_state_dict(checkpoint['policy'], strict=True)
    restored_opt = extension.restored_optimizer(restored_net, cfg, checkpoint)
    restored_runtime = extension.fresh_runtime(restored_net, cfg, 'cpu')
    extension.restore_runtime(restored_runtime, checkpoint['snapshot'], 'cpu')
    actual_metrics = [extension.update(restored_net, cfg, restored_opt, source, restored_runtime) for _ in range(5)]
    assert actual_metrics == expected_metrics
    assert extension.tree_equal(restored_net.state_dict(), net.state_dict())
    assert extension.tree_equal(restored_opt.state_dict(), opt.state_dict())
    assert extension.tree_equal(extension.capture_runtime(restored_runtime, 'cpu'), expected_runtime)
    # The split passes through episode ends, noise draws and native goal/clock resets.
    assert restored_runtime['env'].episodes_done >= 2
