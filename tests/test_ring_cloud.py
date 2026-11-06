"""Safety checks for the isolated cloud launcher. No cloud requests are made."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("ring_cloud", Path(__file__).parents[1] / "scripts/ring_cloud.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


class RingCloudSafety(unittest.TestCase):
    def args(self, **kwargs):
        data = dict(name="ring-20261105-unit-test", command="python scripts/example.py",
                    output_dir="results/ring_attractor", hours=2, machine="g2-standard-4",
                    zone="us-central1-a", include=[])
        data.update(kwargs)
        return argparse.Namespace(**data)

    def test_worker_source_compiles(self):
        compile(cloud.WORKER, "ring_worker.py", "exec")

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
            data = dict(budget_usd=500., reserved_rate_usd_per_hour=1.25,
                        ancillary_reserve_usd=25., max_active_workers=1,
                        runs=[dict(name="old", reserved_usd=474., actual_upper_bound_usd=474.)])
            ledger.write_text(json.dumps(data))
            with mock.patch.object(cloud, "LEDGER", ledger), mock.patch.object(cloud, "run") as run:
                with self.assertRaisesRegex(ValueError, "exceed"):
                    cloud.launch(self.args())
                run.assert_not_called()
            self.assertEqual(json.loads(ledger.read_text()), data)

    def test_active_pilot_denial_before_cloud_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "ledger.json"
            data = dict(budget_usd=500., reserved_rate_usd_per_hour=1.25,
                        ancillary_reserve_usd=25., max_active_workers=1,
                        runs=[dict(name="old", reserved_usd=2.8125)])
            ledger.write_text(json.dumps(data))
            with mock.patch.object(cloud, "LEDGER", ledger), mock.patch.object(cloud, "run") as run:
                with self.assertRaisesRegex(ValueError, "Active-worker"):
                    cloud.launch(self.args())
                run.assert_not_called()

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
