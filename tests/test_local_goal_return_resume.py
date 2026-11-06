import copy
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'tests'))
import local_goal_return_training as task
import local_goal_return_training_run as runner
from test_local_goal_return_training import TinyPolicy, config
from rl_visual_memory_extend import tree_equal


def test_exact_cpu_optimizer_rng_resume_reproduces_next_completed_update(tmp_path):
    torch.set_num_threads(2); cfg = config(); source = task.make_source(cfg, 'cpu')
    net = TinyPolicy(); initialization = task.reset_native_heads(net, 1827)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    streams = task.TrialStreams(1828); counters = runner.fresh_counters(); provenance = {'test': 'exact_resume'}
    plan = streams.draw(2, task.STAGES[0], cfg)
    metrics, _ = task.update(net, cfg, optimizer, source, plan)
    counters.update(updates=1, trials=2)
    for key in ('active_physics_updates', 'forced_updates', 'free_updates', 'padded_slots', 'setup_pickups', 'free_hits'):
        counters[key] = metrics[key]
    checkpoint = tmp_path/'resume.pt'
    torch.save(runner.snapshot(net, optimizer, streams, cfg, initialization, counters, provenance), checkpoint)
    expected_plan = streams.draw(2, task.STAGES[0], cfg)
    expected_metrics, expected_rows = task.update(net, cfg, optimizer, source, expected_plan)
    expected_policy = copy.deepcopy(net.state_dict()); expected_optimizer = copy.deepcopy(optimizer.state_dict())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored_net = TinyPolicy()
    restored_optimizer = torch.optim.RMSprop(restored_net.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    restored_streams = task.TrialStreams(1828)
    restored_counters = runner.restore(restored_net, restored_optimizer, restored_streams, saved, cfg, provenance)
    assert restored_counters == counters
    actual_plan = restored_streams.draw(2, task.STAGES[0], cfg)
    assert tree_equal(actual_plan, expected_plan)
    actual_metrics, actual_rows = task.update(restored_net, cfg, restored_optimizer, source, actual_plan)
    assert tree_equal(expected_policy, restored_net.state_dict())
    assert tree_equal(expected_optimizer, restored_optimizer.state_dict())
    assert actual_rows == expected_rows and actual_metrics == expected_metrics
    assert tree_equal(streams.snapshot(), restored_streams.snapshot())
