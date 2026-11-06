"""Refresh descriptive summaries and finalize figures after all frozen runs finish."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path("results/ring_heading/mechanistic_confirmation")
    started = time.monotonic()
    phase_done = population_done = False
    last_counts = None
    while time.monotonic() - started < 4 * 3600:
        pgc_count = sum((root / f"phase_reduction_predictive_s{s}.json").exists()
                        and (root / f"memory_pulse_predictive_s{s}.json").exists() for s in range(2001, 2017))
        population_count = sum(all((root / f"memory_pulse_{c}_s{s}.json").exists()
                                   for c in ("grid", "predictive", "grid_heading")) for s in range(2001, 2017))
        if (pgc_count, population_count) != last_counts:
            if pgc_count:
                subprocess.run([sys.executable, "scripts/summarize_heading_mechanistic_secondary.py"], check=True)
            if population_count:
                subprocess.run([sys.executable, "scripts/summarize_heading_pulse_populations.py"], check=True)
            last_counts = (pgc_count, population_count)
        if pgc_count == 16 and not phase_done:
            subprocess.run([sys.executable, "scripts/plot_heading_mechanistic_confirmation.py"], check=True)
            phase_done = True
            print(json.dumps({"all16_pgc_summaries_and_figures_complete": True}), flush=True)
        if population_count == 16 and not population_done:
            subprocess.run([sys.executable, "scripts/plot_heading_pulse_populations.py"], check=True)
            population_done = True
            print(json.dumps({"all16_population_triplet_summaries_and_figures_complete": True}), flush=True)
        (root / "finalization_status.json").write_text(json.dumps({"pgc_models_complete": pgc_count,
            "population_triplets_complete": population_count, "pgc_figures_final": phase_done,
            "population_figures_final": population_done}, indent=2) + "\n")
        if phase_done and population_done:
            return
        time.sleep(20)
    raise TimeoutError("Mechanistic secondary replication did not finish within4hours; inspect watcher logs")


if __name__ == "__main__":
    main()
