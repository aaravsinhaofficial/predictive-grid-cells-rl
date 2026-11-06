# Direction-memory tests for the distillation pilot

Run every fixed initial and final model, regardless of the behavioral result. Both training summaries must be complete. The final-model commands require the corresponding initial-model analysis to be complete first.

The primary test ends 30 recurrent updates after blackout onset: **2 seconds**. The existing 60-update test is retained as **4-second extrapolation**. Full-state geometry, all hidden outputs and all centered actor scores use the existing criteria. No angle decoder defines a ring.

Two comparisons are saved:

- **Matched initial contexts:** replay the same complete input prefix through each checkpoint, reconstructing its own hidden and cell states. Apply the same physical heading/travel preparation and future inputs. This isolates the change in recurrent weights at the fixed histories.
- **Own-policy contexts:** collect contexts from each checkpoint's own navigation under the final training schedule. This describes the network in its own operating regime.

Raw collector contexts are preserved. Single-slot prefix replay must agree with their recorded states within `3e-4`; both endpoints consistently use replayed states for geometry. Every paired preparation input, future input and phase angle must match exactly. The saved goal code remains native; current-code substitutions during collection are recorded. These are controlled physical diagnostic families, not a claim that the actor chose the imposed turn/forward sequence.

Freeze after review, before any model geometry outcomes:

```sh
/opt/anaconda3/bin/python -m pytest -q tests/test_lstm_distillation_memory.py tests/test_lstm_continued_heading_memory.py tests/test_lstm_endogenous_observable_flow.py
/opt/anaconda3/bin/python scripts/lstm_distillation_memory.py --freeze-only
```

Run the fixed cohort from the repository root:

```sh
/opt/anaconda3/bin/python scripts/lstm_distillation_memory.py --condition gc100 --endpoint initial --device mps
/opt/anaconda3/bin/python scripts/lstm_distillation_memory.py --condition pgc33 --endpoint initial --device mps
/opt/anaconda3/bin/python scripts/lstm_distillation_memory.py --condition gc100 --endpoint final --device mps
/opt/anaconda3/bin/python scripts/lstm_distillation_memory.py --condition pgc33 --endpoint final --device mps
```

Outputs are under `results/lstm_distillation_memory/{population}/{endpoint}/`. Each initial endpoint has four contexts: two heading and two travel. Each final endpoint has those four own-policy contexts plus four matched-initial-context tests. The complete cohort therefore contains **24 context evaluations**, with all input branches, coordinate metrics, time windows and positive/negative outcomes retained. The protocol records source, training-protocol and original-checkpoint hashes; each run adds exact endpoint, context, prefix, mask and preparation-input hashes. Existing frozen reports and checkpoints are not overwritten.

Any candidate is followed by broader normal-direction testing and predictive-cue repair before a stabilization interpretation. No candidate is inferred merely from a closed input family or successful angle decoding.
