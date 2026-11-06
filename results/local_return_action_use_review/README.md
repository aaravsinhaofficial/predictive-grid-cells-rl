Independent review passed for all **38,912 records**, including every fixed endpoint and control. No policy or environment was executed, and the frozen analysis outputs were not modified.

All 801 input hashes, 233,472 counterfactual action endpoints, 152 pooled summaries and 2,432 context summaries reconciled. Every condition retained 256 trials; none were undefined. The largest discrepancy from the recorded first-step position was **1.59 × 10⁻⁸ m**, below the fixed 2 × 10⁻⁶ m tolerance. Independently calculated metric differences were at most 1.11 × 10⁻¹⁶.

| Native hard-bank condition | Mean path | Mean goal progress |
|---|---:|---:|
| Grid, stage3 | 33.92 cm | −4.41 cm |
| PGC, stage3 | 33.09 cm | −4.56 cm |
| Grid, final | 19.50 cm | −0.77 cm |
| PGC, final | 19.60 cm | −0.66 cm |

The agents moved, but their final positions were farther from the goal on average. Native first-step expected goal progress ranged from −0.033 to +0.0047 mm across the five endpoints and two models: little **net immediate goalward alignment** across the tested headings. This signed average does not imply that every individual action was uninformative. First-step movement also does not assess the long-term value of turning or identify a ring mechanism.

The best-of-six reference uses the realized motor noise and remains a hindsight physical bound. These records are repeated measurements of one original seed pair, not independent trained models. Body-angle and observed first-progress fields retain the producer's native float32 arithmetic; the six-action weighted metrics use float64.

[Complete audit and hashes](completed_output_audit.json) · [Independent checker](../../scripts/audit_local_return_action_use.py) · [Frozen action-use report](../local_return_action_use/README.md)
