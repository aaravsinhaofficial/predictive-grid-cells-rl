# Turning right and strafing right limits the available routes

Secondary analysis of the fixed **pretraining baseline**; no model or environment was run.

| Population | Potentially reachable headings | Analytical upper bound | Unresolved headings |
|---|---:|---:|---:|
| gc100 | 139/256 | 54.3% | 0 |
| pgc33 | 139/256 | 54.3% | 0 |

The calculation bounds **any controller restricted to right turns and right strafes**, using the same saved starting poses and body-frame motor-noise tapes. All 16 contexts and 16 headings remain. Passing the necessary angle condition means potentially reachable, not that a route has been found.

The Grid baseline sampled only these two actions; the PGC baseline also sampled nine backward actions. The full softmax networks retain other actions, so this is not their unrestricted performance ceiling. It suggests why acquiring a broader movement strategy may matter for fast local return, independently of memory or ring geometry.

This is a bound for the native equations with declared 0.1 mm and 0.0001 rad numerical margins, not formal hardware verification. [Derivation](../../../../docs/research/feedback_restricted_action_bound.md) · [All cases and input hashes](summary.json).
