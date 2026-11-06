# Reproduce the four fixed 75M endpoints

The release archive `final75-memory-inputs-v1.tar.gz` contains the four exact final checkpoints, their clean/noisy evaluation references, and all sixteen saved native context inputs. Exact matched-noise and every-step-reset episode rows, environment records and success/failure statuses are included under the original diagnostic worker folders. It preserves repository-relative paths. Compare its SHA256 with [the committed archive record](../../docs/research/final75_inputs_archive.json) before extracting at the repository root.

The frozen wrapper also checks the two 25M parent checkpoints. Those are already distributed in `endogenous-ring-inputs-v1.tar.gz` on the [earlier endogenous-ring release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/endogenous-ring-tests-2026-11-06). Obtain that archive first if the parent files are absent.

Download the new inputs from the [75M continuation release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/visual-memory-continuations-2026-11-06). The [input manifest](reproduction_inputs_manifest.json) gives every path, size and hash. No intermediate checkpoint was selected.

For example, reproduce both heading and both travel contexts of the noiseless Grid endpoint into a separate directory:

```bash
python scripts/lstm_final75_replay_cached.py \
  --arm v2 --condition gc100 \
  --checkpoint results/ring_cloud_artifacts/ring-20261105-vision-long-grid-01/final.pt \
  --source-root results/lstm_final75_memory/v2/gc100 \
  --out /tmp/final75-grid-reproduction --device mps
```

Use `cpu` when MPS is unavailable. Numerical backends can affect floating-point results. On the original MPS backend, all numerical fields in the four output state archives for this noiseless Grid example reproduced **bit-for-bit** in a separate folder; [validation record](cached_reproduction_validation.json). The wrapper replaces only context collection with loading the published dictionaries and validates additive runtime metadata. The original analysis functions and gates remain unchanged.

| Arm | Population | Worker checkpoint folder |
|---|---|---|
| v2 | gc100 | ring-20261105-vision-long-grid-01 |
| v2 | pgc33 | ring-20261105-vision-long-pgc-01 |
| v3 | gc100 | ring-20261105-vision-noise-grid-01 |
| v3 | pgc33 | ring-20261105-vision-noise-pgc-01 |

Each folder contains `final.pt`. Use the matching `results/lstm_final75_memory/ARM/CONDITION` source folder. The reproduced `heading/context{0,1}_states.npz` and `travel/CONDITION_s1/context{0,1}_states.npz` feed the separate frozen observable, cue-pulse and finite-time spectral analyses. Their protocols retain every model/context and record source hashes. Large derived arrays can be regenerated from these compact native inputs.

- [Future-hidden-activity protocol](../lstm_endogenous_ring/observable_history/final75/protocol.json)
- [Cue-pulse cohort and exact input/output paths](../lstm_heading_memory/travel_cue_pulse/final75/cohort_protocol.json)
- [Finite-time spectral protocol](../lstm_final75_finite_spectra/protocol.json)
- [Final checkpoint/cloud audit](../ring_visual_memory/continuation_cloud_audit/artifact_verification.json)

The two new blind-reorientation training runs are a separate experiment and are not included in this archive.
