"""Safety checks for the isolated cloud launcher. No cloud requests are made."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("ring_cloud", Path(__file__).parents[1] / "scripts/ring_cloud.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


class RingCloudSafety(unittest.TestCase):
    def args(self, **kwargs):
        data = dict(name="ring-20261105-simplemem-unit-test", command="python scripts/example.py",
                    output_dir="results/ring_attractor", hours=2, machine="g2-standard-4",
                    zone="us-central1-a", include=[])
        data.update(kwargs)
        return argparse.Namespace(**data)

    def test_worker_source_compiles(self):
        compile(cloud.WORKER, "ring_worker.py", "exec")

    def test_storage_serial_environment_is_scoped_to_storage_subprocess(self):
        original = dict(os.environ)
        result = argparse.Namespace(returncode=0, stderr='', stdout='')
        with mock.patch.object(cloud.subprocess, 'run', return_value=result) as run:
            cloud.run(['gcloud', 'storage', 'rsync', 'gs://private-test/source', '/tmp/private-test'])
            env = run.call_args.kwargs['env']
            self.assertEqual(env['CLOUDSDK_STORAGE_PROCESS_COUNT'], '1')
            self.assertEqual(env['CLOUDSDK_STORAGE_THREAD_COUNT'], '1')
            self.assertEqual(dict(os.environ), original)
            cloud.run(['gcloud', 'compute', 'instances', 'describe', 'private-test'])
            self.assertNotIn('env', run.call_args.kwargs)

    def test_absolute_deadline_accounts_for_delayed_boot(self):
        timing = {}
        exec(cloud.WORKER_TIMING, timing)
        start = datetime.datetime(2026, 11, 7, tzinfo=datetime.timezone.utc)
        deadline = (start + datetime.timedelta(hours=2)).isoformat()
        budget = timing['training_seconds']
        self.assertEqual(budget(deadline, start.timestamp()), 7200 - 420)
        self.assertEqual(budget(deadline, start.timestamp() + 600), 7200 - 420 - 600)
        self.assertEqual(budget(deadline, start.timestamp() + 7000), 0)
        self.assertEqual(budget(deadline, start.timestamp() + 7300), 0)
        with self.assertRaisesRegex(ValueError, 'UTC offset'):
            budget('2026-11-07T02:00:00', start.timestamp())

    def test_plan_deadline_is_bounded_and_new_names_are_isolated(self):
        planned = cloud.plan(self.args(hours=25.5))
        before = datetime.datetime.fromisoformat(planned['reserved_at'])
        after = datetime.datetime.fromisoformat(planned['provider_deadline_utc'])
        self.assertEqual((after - before).total_seconds(), 91800)
        self.assertEqual(planned['reserved_usd'], 32.1875)
        self.assertEqual(16 * planned['reserved_usd'], 515.0)
        with self.assertRaisesRegex(ValueError, '25.5 hour'):
            cloud.plan(self.args(hours=25.51))
        with self.assertRaisesRegex(ValueError, 'namespace'):
            cloud.plan(self.args(name='ring-20261105-old-style'))

    def test_cpu_startup_has_no_gpu_dependency(self):
        cpu = cloud.plan(self.args(machine="e2-standard-8", command="python scripts/example.py --device cpu"))
        script = cloud.startup_script(cpu, "0" * 64)
        self.assertNotIn("nvidia-smi", script)
        self.assertNotIn("torch.cuda", script)
        self.assertIn("CPU worker", script)
        self.assertIn("sha256sum --check", script)
        self.assertEqual(cpu["reserved_usd"], 2.8125)
        gpu = cloud.plan(self.args())
        self.assertIn("assert torch.cuda.is_available()", cloud.startup_script(gpu, "0" * 64))

    def test_startup_is_locked_and_one_shot_before_package_changes(self):
        script = cloud.startup_script(cloud.plan(self.args()), '0' * 64)
        self.assertLess(script.index('flock -n 9'), script.index('gsutil cp'))
        self.assertLess(script.index('[ -e /work/.ring-startup-attempted ]'), script.index('gsutil cp'))
        self.assertLess(script.index('> /work/.ring-startup-attempted'), script.index('gsutil cp'))
        self.assertIn('explicit checkpoint recovery on a new worker required', script)
        self.assertIn('pass_fds=(9,)', cloud.WORKER)

    def test_orphan_trainer_keeps_inherited_flock(self):
        # Isolate FD9 from pytest. The child is a tiny process, not a model job.
        program = r'''
import fcntl, os, subprocess, sys, tempfile
with tempfile.TemporaryDirectory() as folder:
    path = folder + '/worker.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    if fd != 9:
        os.dup2(fd, 9); os.close(fd)
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen([sys.executable, '-c',
        "import os,time;os.fstat(9);print('ready',flush=True);time.sleep(10)"],
        stdout=subprocess.PIPE, text=True, pass_fds=(9,))
    try:
        assert child.stdout.readline().strip() == 'ready'
        os.close(9)  # Simulate the worker disappearing while its trainer lives.
        contender = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('Orphan trainer failed to retain worker lock')
        child.terminate(); child.wait(timeout=5)
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(contender)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)
'''
        subprocess.run([sys.executable, '-c', program], check=True, timeout=15,
                       capture_output=True, text=True)

    def test_unsafe_names_and_paths(self):
        for name in ("existing-user-vm", "ring-20261105-a;echo", "ring-20261105-"):
            with self.assertRaises(ValueError):
                cloud.check_name(name)
        with self.assertRaises(ValueError):
            cloud.relative_path("../outside")
        with self.assertRaises(ValueError):
            cloud.plan(self.args(output_dir="results/lstm_variable_1B"))

    def test_budget_denial_before_cloud_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "ledger.json"
            data = dict(budget_usd=600., reserved_rate_usd_per_hour=1.25,
                        ancillary_reserve_usd=25., max_active_workers=1,
                        runs=[dict(name="old", reserved_usd=574., actual_upper_bound_usd=574.)])
            ledger.write_text(json.dumps(data))
            with mock.patch.object(cloud, "LEDGER", ledger), mock.patch.object(cloud, "run") as run:
                with self.assertRaisesRegex(ValueError, "exceed"):
                    cloud.launch(self.args())
                run.assert_not_called()
            self.assertEqual(json.loads(ledger.read_text()), data)

    def test_active_pilot_denial_before_cloud_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "ledger.json"
            data = dict(budget_usd=600., reserved_rate_usd_per_hour=1.25,
                        ancillary_reserve_usd=25., max_active_workers=1,
                        runs=[dict(name="old", reserved_usd=2.8125)])
            ledger.write_text(json.dumps(data))
            with mock.patch.object(cloud, "LEDGER", ledger), mock.patch.object(cloud, "run") as run:
                with self.assertRaisesRegex(ValueError, "Active-worker"):
                    cloud.launch(self.args())
                run.assert_not_called()

    def test_locked_ceiling_update_preserves_old_records(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'ledger.json'
            old = dict(budget_usd=500., reserved_rate_usd_per_hour=1.25,
                       ancillary_reserve_usd=25., max_active_workers=8,
                       runs=[dict(name=f'old-{i}', reserved_usd=2.8125,
                                  actual_upper_bound_usd=.1, status='DELETED')
                             for i in range(51)])
            ledger.write_text(json.dumps(old))
            with mock.patch.object(cloud, 'LEDGER', ledger), mock.patch.object(cloud, 'BUDGET', 500.):
                with cloud.locked_ledger() as data:
                    data['budget_usd'] = 600.
            current = json.loads(ledger.read_text())
            self.assertEqual(current, dict(old, budget_usd=600.))
            self.assertEqual(cloud.committed(current), cloud.committed(old))
            with mock.patch.object(cloud, 'LEDGER', ledger):
                with cloud.locked_ledger() as checked:
                    self.assertEqual(len(checked['runs']), 51)

    def test_replaced_vm_cannot_be_deleted(self):
        record = dict(instance_id="123")
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            cloud.validate_identity(record, dict(id="456", labels=dict(study=cloud.PREFIX)))
        with self.assertRaisesRegex(RuntimeError, "label"):
            cloud.validate_identity(record, dict(id="123", labels=dict(study="another-study")))

    def test_closed_cost_is_not_recomputed_days_later(self):
        r = dict(name="ring-20261105-test", zone="us-central1-a", gcs_prefix="gs://test/owned",
                 reserved_usd=2.8125, actual_upper_bound_usd=.10, created_at="2026-01-01T00:00:00+00:00")
        with mock.patch.object(cloud, "describe", return_value=None), mock.patch.object(
            cloud, "run", return_value=argparse.Namespace(returncode=1)):
            cloud.refresh(r)
        self.assertEqual(r["actual_upper_bound_usd"], .10)


if __name__ == "__main__":
    unittest.main()
