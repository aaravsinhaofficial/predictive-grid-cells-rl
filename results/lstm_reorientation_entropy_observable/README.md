# Future-activity tests for the four entropy continuations

**All 64 prescribed checks are complete: zero centered and zero noncentered candidates.** Every saved trajectory replayed exactly before the probes. This reuses the 40M future-activity test without changing its observation map, perturbations or criteria; the adapter was frozen before these outcomes.

Each model contributes two native contexts, four cue conditions and two full-state perturbation metrics: **64 nominal checks across four models**. Grid cue shams are duplicates, and all contexts/checks are nested in one original training-seed pair.

First run the [90M primary collector](../../docs/research/reorientation_entropy_mechanism_reproduction.md) for every model. It validates genuine completed 90,132,480-frame metadata and resolves actual worker names and artifact directories from the launch record. Then run:

```sh
export OPENBLAS_NUM_THREADS=2
python scripts/lstm_endogenous_entropy_observable.py --arm original --condition gc100
python scripts/lstm_endogenous_entropy_observable.py --arm original --condition pgc33
python scripts/lstm_endogenous_entropy_observable.py --arm explore --condition gc100
python scripts/lstm_endogenous_entropy_observable.py --arm explore --condition pgc33 --require-complete
```

The adapter reads `results/lstm_reorientation_entropy_mechanism/ARM/CONDITION/context{0,1}_states.npz`. It validates producer/checkpoint hashes and replays each saved trajectory before testing recovery. Every branch receives the same ordinary delay input after its single cue.

The test observes all 256 hidden outputs at four future times. It begins at update 15, compares activity after 30 further updates, and ends at update 60, before the first free action. Centered and noncentered results remain separate. A candidate requires every existing geometry, observable-amplitude and selective-recovery criterion; any later cue-repair follow-up uses the same fixed observation map.

One partial result deserves emphasis: the higher-entropy Grid model’s second context has a raw radial/angular ratio of 0.042 and centered departure of 0.0053 radii. Its curve fails the separation and phase-correspondence guards. Thus some conditions recover from a disturbance while still failing the full geometric test. Failing correspondence to an earlier curve does not mean the actual direction information disappears.

[Complete numerical results and every gate](summary.json) · [Frozen protocol](protocol.json) · [Review, hashes and tests](review.json) · [Completion status](status.json) · [Adapter source](../../scripts/lstm_endogenous_entropy_observable.py) · [Separate learned-action-score test](../lstm_actor_logit_observable/RESULTS.md)
