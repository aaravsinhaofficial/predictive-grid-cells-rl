# Matched goal-only gate rescue job

The primary experiment evaluates eight frozen PGC agents under four conditions: intact goal prediction, ordinary goal code, ordinary goal code with predictive influence restored to the forget gate, and ordinary goal code with predictive influence restored to the candidate update and output gate. Current predictive code remains intact throughout. No recurrent-state noise is added.

Every seed runs 64 matched environments for two episodes per condition: 128 episodes per seed and condition, 4,096 episodes in total. Episode layouts, physical noise and action uniforms are paired across conditions; recurrent states and trajectories evolve independently after behavior diverges. Goal snapshots update on each branch's actual pickups.

The job was initially prepared for CUDA. Before any primary seed ran, the root agent selected local MPS execution after cloud GPU stockouts. This change is recorded in `results/ring_attractor/goal_gate_behavior/execution_plan.json`. The primary command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/lstm_goal_gate_behavior.py \
  --device mps --seeds 1 2 3 4 5 6 7 8 \
  --n-envs 64 --episodes 128 \
  --out results/ring_attractor/goal_gate_behavior
```

Completed seed files are reused automatically after validating their evaluation settings. Seed results and protocol files are written atomically. The process prints progress about every 25 seconds and saves each seed independently.

For a separate CUDA replication, use `--device cuda:0` and a new output directory. Required worker assets are:

- `scripts/lstm_goal_gate_behavior.py` and `scripts/lstm_code_swap.py`.
- The `code/` package, including environment, network and code-generation modules.
- Eight `models/pgc33/pgc33_lstm_s{1..8}_1000M.pt` checkpoints and their JSON configs.
- Eight `models/gc100/gc100_lstm_s{1..8}_1000M.json` configs for independent grid-code identity checks. Grid checkpoints are not required.
- Python packages: PyTorch with CUDA, NumPy, SciPy and Matplotlib.
- Optionally `results/ring_attractor/gate_stability/summary.json` to generate the combined memory/navigation figure on the worker. Otherwise regenerate the summary locally after collecting results.

Four score contrasts are specified in the saved primary protocol: intact minus ordinary; forget restoration minus ordinary; candidate/output restoration minus ordinary; candidate/output restoration minus forget restoration. Training seeds are the independent units. Holm correction is applied separately to paired-t and exhaustive paired sign-flip tests. Other behavior metrics are descriptive.

The local MPS benchmark ran 16 environments for one episode per condition, all four conditions together, in **11.12 seconds** (about 7,772 aggregate environment decisions per second). It is stored under `results/ring_attractor/goal_gate_behavior/benchmark_mps` and excluded from primary analysis. GPU runtime should be measured on the chosen worker rather than inferred directly from MPS throughput.

Benchmark and independent snapshot checks found zero error for the explicit LSTM reconstruction and for selecting gate drives before versus after their nonlinearities. The actual intact branch equals full native restoration; the actual ordinary branch equals no restoration. Before first discovery the four branches have identical first-goal times. Non-PGC goal channels and the entire current input are unchanged, and checkpoint hashes are checked again after evaluation.

Historical MPS baseline episode rows are compared when locally available, but the comparison is diagnostic: a CUDA run can draw different physical-noise/action streams and have different floating-point trajectories. The causal comparisons are paired within the same evaluation run and backend.

After transferring all eight JSON results back to this repository:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/lstm_goal_gate_behavior.py \
  --summarize-only --device mps --seeds 1 2 3 4 5 6 7 8 \
  --n-envs 64 --episodes 128 \
  --out results/ring_attractor/goal_gate_behavior
```

This produces a simple two-panel figure with the exact same four conditions: cell-state error after four seconds in the existing targeted perturbation assay, and navigation reward in this new noiseless behavioral assay. The experiment compares pathway functions; it does not itself test whether suppressing the targeted memory disturbance improves navigation. That would require a separate closed-loop perturbation experiment.
