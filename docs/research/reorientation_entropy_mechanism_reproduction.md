# Fixed 90M entropy continuations: mechanism reproduction

This is the prospective adapter for **all four fixed final checkpoints**, with 90,132,480 learning frames beyond the original 1B model. The added entropy continuation contributes 50,073,600 frames after the fixed 40,058,880-frame parent. The two entropy coefficients are 0.00008 and 0.01; there is no stage selection.

The [adapter](../../scripts/lstm_reorientation_entropy_mechanism.py) calls the frozen 40M context collector, physical replay, full-state geometry, and causal-navigation functions unchanged. It validates the real 90M checkpoint metadata directly. The old 40M validators and numerical sources remain untouched.

The exact parent checkpoints, shared by both entropy arms within a population, are:

| Population | Parent source | SHA256 |
|---|---|---|
| Grid | `results/ring_cloud_artifacts/ring-20261105-reorient-grid-01/final.pt` | `ee0635c73a94c0f19ead27f6772495cece8c18ee19666d32d9765f5646de4651` |
| PGC + grid | `results/ring_cloud_artifacts/ring-20261105-reorient-pgc-02/final.pt` | `654f756dac3f13eed5fe4e5c04ae64769eef21c520fa531de05136d2b1a95817` |

These parent files and their `final_evaluation.json` companions are checked against the [frozen continuation protocol](rl_blind_reorientation_entropy_protocol.json). The numerical sources are [the primary collector/geometry](../../scripts/lstm_reorientation_mechanism.py) and [the causal-navigation assay](../../scripts/lstm_reorientation_causal_navigation.py); their original frozen manifests remain under `results/lstm_reorientation_mechanism/` and `results/lstm_reorientation_mechanism/causal_navigation/`.

Before final checkpoint outcomes, freeze the adapter and source hashes:

```sh
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --freeze-only
```

For each population `gc100` and `pgc33`, and each entropy arm `original` and `explore`, run the following three commands, substituting those two arguments consistently:

```sh
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --condition gc100 --arm original --mode primary
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --condition gc100 --arm original --mode causal --context 0
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --condition gc100 --arm original --mode causal --context 1
```

The checkpoint is resolved from the matching condition/arm row's explicit worker `name` and `artifact_subdir` in [the launch routing](rl_blind_reorientation_entropy_launch.json). This accommodates the documented infrastructure retries and `experiment/` child directory. It never guesses an `01` worker name or selects `latest.pt`. The routing snapshot, actual checkpoint hash, real provenance and analysis backend are recorded in each `extension_adapter.json`.

Output folders are `results/lstm_reorientation_entropy_mechanism/{original,explore}/{gc100,pgc33}`. The primary command writes both context archives, all native angular coverage, the inherited numerical manifest, the separate extension metadata, and all original gate results. The causal commands write each context's complete perturbation/cue/navigation results and matched noise/action tapes under `causal/context0` and `causal/context1`. Run causal analysis for every context regardless of geometry outcomes.

```text
results/lstm_reorientation_entropy_mechanism/
  protocol.json, review.json
  original/gc100/                  # same layout for original/pgc33,
    extension_adapter.json        # explore/gc100 and explore/pgc33
    implementation_protocol.json
    collection.json, native_phase_coverage.json
    contexts/context0.npz, contexts/context1.npz
    context0.json, context0_states.npz
    context1.json, context1_states.npz
    summary.json
    causal/context0/result.json, navigation_paths.npz, inputs.json, protocol.json
    causal/context1/result.json, navigation_paths.npz, inputs.json, protocol.json
```

Collection uses fresh clean CPU histories under the trained final 30–60-update delay, with the same fixed seed, two-context physical eligibility rule and collection budget as the 40M assay. Training state noise is off during collection and geometry. The controlled forced delay is noiseless; actual free navigation retains the original motor noise. The native noisy whole-episode behavioral evaluation is separate.

The original [phase/action indexing](reorientation_mechanism_reproduction.md) is unchanged: saved time 0 is **after** the single endpoint cue's recurrent update, before its first forced LEFT action. At saved time 60, the first free-action input has already been processed. The first free action therefore uses `pi(h60)` directly, followed by 29 further recurrent updates for exactly 30 actual free actions. No extra brake is applied at free onset. Goal/current-code and previous-action rules remain as frozen.

The full-state gates and the separate future-hidden-activity gates retain their original thresholds. A correct pulse's repair effect within one frozen model is separate from its recurrent stability under common ordinary input. Population comparisons use each agent's own selected physical history and saved goal code; they are descriptive comparisons of one original seed pair, with repeated contexts and phase samples.

One follow-up is conditional on a future-hidden-activity ring factor meeting its frozen criteria while the full-state test fails. In that case, also measure cue repair in the **same fixed observation map**: all 256 hidden units at offsets 0, 5, 10 and 15 under the common ordinary input. Use the prescribed saved anchors 15, 30 and 45, so the latest state remains 60. Compare with the dense correct-cue and own-cue reference curves in all 1,024 observable coordinates, retaining both original and mean-subtracted geometry and every cue/context of that model. There is no learned representation, selected neuron set or decoder plane. Existing full-state causal measurements stay primary, and the existing actor/navigation outcomes supply the functional comparison. This follow-up is explicitly planned, not presently implemented or added as a new gate.

The metadata tests reject old 40M checkpoints, incomplete/restarted endpoints, changed configurations, incorrect entropy arms, unexpected runtime fields and invalid routing paths. They also verify every inherited numerical source against its original manifest. They do not construct or report synthetic trained-network outcomes.
