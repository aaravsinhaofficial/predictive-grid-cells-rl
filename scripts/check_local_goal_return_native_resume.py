"""Reproduce the disposable full-CNN/LSTM checkpoint-resume implementation check.

Uses a separate unit-check seed, never the sealed scientific evaluation banks.
Run with --out /tmp/native-resume-check.json to preserve published results.
"""
import argparse
import copy
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
import local_goal_return_training as task
import local_goal_return_training_run as runner
from local_goal_return_training_evaluate import initial_policy
from rl_visual_memory_extend import tree_equal
from rl_goal_memory import state_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Preserve completed validation records'
    functions = (task.update, task.collect, task.replay, task.returns, task.losses,
                 runner.snapshot, runner.restore, runner.pending_stage, initial_policy)
    code_hashes = lambda: {f.__module__ + '.' + f.__name__:
        hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions}
    before_sources = code_hashes()
    torch.set_num_threads(2)
    results = []
    for condition in ('gc100', 'pgc33'):
        net, cfg, initialization = initial_policy(condition, 1, 'cpu')
        source = task.make_source(cfg, 'cpu')
        optimizer = torch.optim.RMSprop(net.parameters(), lr=cfg['lr'],
            alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
        streams = task.TrialStreams(531801)
        counters = runner.fresh_counters()
        provenance = {'purpose': 'Full native architecture implementation check; not scientific pilot training'}
        first_plan = streams.draw(2, task.STAGES[0], cfg)
        first_metrics, _ = task.update(net, cfg, optimizer, source, first_plan)
        counters.update(updates=1, trials=2)
        for key in ('active_physics_updates', 'forced_updates', 'free_updates',
                    'padded_slots', 'setup_pickups', 'free_hits'):
            counters[key] = first_metrics[key]
        with tempfile.TemporaryDirectory(prefix='local-native-resume-') as tmp:
            checkpoint = Path(tmp) / 'checkpoint.pt'
            torch.save(runner.snapshot(net, optimizer, streams, cfg,
                initialization, counters, provenance), checkpoint)
            expected_plan = streams.draw(2, task.STAGES[0], cfg)
            expected_metrics, expected_rows = task.update(net, cfg, optimizer, source, expected_plan)
            expected_policy = copy.deepcopy(net.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_rng = runner.global_rng_snapshot()
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            restored, restored_cfg, _ = initial_policy(condition, 1, 'cpu')
            assert restored_cfg == cfg
            restored_optimizer = torch.optim.RMSprop(restored.parameters(), lr=cfg['lr'],
                alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
            restored_streams = task.TrialStreams(531801)
            assert runner.restore(restored, restored_optimizer, restored_streams,
                saved, cfg, provenance) == counters
            actual_plan = restored_streams.draw(2, task.STAGES[0], cfg)
            assert tree_equal(actual_plan, expected_plan)
            actual_metrics, actual_rows = task.update(restored, cfg, restored_optimizer, source, actual_plan)
            assert tree_equal(expected_policy, restored.state_dict())
            assert tree_equal(expected_optimizer, restored_optimizer.state_dict())
            assert actual_metrics == expected_metrics and actual_rows == expected_rows
            assert tree_equal(streams.snapshot(), restored_streams.snapshot())
            assert tree_equal(expected_rng, runner.global_rng_snapshot())
            results.append(dict(condition=condition,
                original_checkpoint_sha256=initialization['parent_checkpoint_sha256'],
                initializer_policy_hash=initialization['full_initialized_policy_hash'],
                replayed_second_policy_hash=state_digest(restored),
                full_native_cnn_and_lstm=True, trials_per_batch=2,
                uninterrupted_updates=2, restored_updates_repeated=1,
                policy_optimizer_streams_rows_metrics_global_rng_all_exact=True,
                first_metrics=first_metrics, second_metrics=actual_metrics))
        assert sha(ROOT / initialization['parent_checkpoint']) == initialization['parent_checkpoint_sha256']
    assert before_sources == code_hashes()
    report = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
        source_sha256=sha(__file__), function_source_sha256=before_sources,
        environment=dict(torch=str(torch.__version__), device='cpu', threads=2), rows=results,
        scope='Disposable copies of both full native1B architectures; private unit seed531801. '
              'Serialize after one complete two-trial update and exactly reproduce the next update. '
              'Torch CPU/CUDA global RNG, all private streams, optimizer, policy, metrics and rows checked. '
              'No sealed bank used, source checkpoint changed, scientific training or cloud job.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, output=str(args.out), models=len(results))))


if __name__ == '__main__':
    main()
