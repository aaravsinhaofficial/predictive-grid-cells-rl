# Dynamics in the learned action scores

**All 152 prescribed checks are complete: none meets every frozen ring criterion.** [See the simple figure and results](RESULTS.md). The method was frozen before any new results in this observation map. The full-state and all-hidden-activity tests remain unchanged. Original and 40M geometry results, and the 40M action-readout bottleneck, were already known when this follow-up was designed.

This test observes the exact six learned action scores after subtracting the score shared by all actions. It observes those scores at four future times: 0, 5, 10 and 15 updates. All 24 coordinates are retained; the action-common subtraction leaves at most 20 independent coordinates. No decoder, PCA projection, score normalization or new recurrent model is fitted.

The existing probe still makes the same full-state radial/angular perturbations and uses every original criterion. Raw and hidden/cell-scaled kicks, and both original and family-mean-subtracted score curves, remain separate. The unchanged amplitude guard is explicitly measured in centered-logit units here.

| Existing cohort | Models | Contexts | Branch/metric checks |
|---|---:|---:|---:|
| Original 1B, pilot seeds 1–2 | 4 | 4 | 8 |
| Original 1B, earlier held-out seeds 3–8 | 12 | 24 | 48 |
| Fixed 40M task endpoints | 2 | 4 | 32 |
| Fixed 90M entropy endpoints | 4 | 8 | 64 |
| Total | 22 endpoints | 40 | 152 |

The 40M and 90M endpoints continue one original seed pair. Contexts, cues and metrics are nested measurements; repeated Grid cue shams are not independent models. “Earlier held-out” describes the original collection cohort, not newly untouched data for this follow-up. No additional contexts are collected.

A positive result would nominate a circular factor in the learned action scores under the tested common input. It would trigger cue-repair tests in this same map and comparison with actual action probabilities/navigation. Output saturation and other hidden dynamics can remain; a projected factor alone is not a functional ring mechanism.

After downloading the checkpoint and exact state-archive dependencies, run from the repository root. Use a fresh output directory because the frozen runner protects completed case files from overwrite:

```sh
export OPENBLAS_NUM_THREADS=2
ACTOR_SCORE_OUT=$(mktemp -d "${TMPDIR:-/tmp}/actor-score-reproduce.XXXXXX")
cp results/lstm_actor_logit_observable/protocol.json "$ACTOR_SCORE_OUT/protocol.json"
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --cohort original1B
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --cohort forty
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --cohort ninety --arm original --condition gc100
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --cohort ninety --arm original --condition pgc33
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --cohort ninety --arm explore --condition gc100
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --cohort ninety --arm explore --condition pgc33
python scripts/lstm_actor_logit_observable.py --out "$ACTOR_SCORE_OUT" --summarize-only --require-complete
```

The 90M commands require genuine completed checkpoints and the corresponding primary archives. The summary is written to `$ACTOR_SCORE_OUT/summary.json`. Optional `--seed` and `--context` arguments permit scheduling the prescribed original cases individually; they do not change the complete cohort. The reporting-only figure can be rebuilt from the published completed summary with `python scripts/plot_actor_logit_observable.py`.

[Frozen protocol](protocol.json) · [Review and freeze record](review.json) · [Source](../../scripts/lstm_actor_logit_observable.py) · [Tests](../../tests/test_lstm_actor_logit_observable.py) · [Metadata-only preflight](metadata_preflight.json)
