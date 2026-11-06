# Releasing the fixed current predictive cue

We repeated the original PGC flow test after changing **only the 169 current predictive-grid slots into ordinary-grid values**. The saved goal code, black-image input, previous turn action, reward, network weights and starting recurrent states were identical to the native-input run.

This tests a concrete alternative: perhaps the common predictive direction was holding a genuine ring at one phase.

| Original PGC model | Native radial/tangential gain | After current-cue release |
| --- | ---: | ---: |
| Seed 1 | 1.034 | 1.000 |
| Seed 2 | 0.993 | 0.998 |

These are raw-state gains over five seconds from the thirty-second reference. Ratios near one mean radial mistakes persist about as strongly as angular differences. Releasing the current cue preserved this behavior. The released families also moved 16–17 radii away from the earlier fixed curves. Both released cases fail the separately frozen candidate screen.

The input audit confirms exactly 169 changed values and **zero changes to the saved goal or any other input**. The [protocol](protocol.json) and [summary](summary.json) retain the matched comparison. An initial adapter argument error occurred before any dynamics ran; its earlier protocol is retained as `protocol_before_adapter_fix.json`. The corrected adapter injects the exact saved native states into the unchanged flow implementation.

```bash
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_heading_current_release.py
```
