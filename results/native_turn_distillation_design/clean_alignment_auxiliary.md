# Clean endpoint: turning and action-separation check

This saved-data check includes all16 contexts and128 opposite-heading pairs per mode. It supplements the first-step movement measure; useful pure turns can have zero translation.

| Model | History | Policy | Turn-weighted facing improvement (degrees/update) | Turn probability | Pair action TV | Largest pair TV |
|---|---|---|---:|---:|---:|---:|
| gc100 | turn_only | teacher/full_native | -0.4548279 | 0.6193 | 0.516259 | 0.999976 |
| gc100 | turn_only | original/native | 0.0009643 | 0.8692 | 0.001399 | 0.021940 |
| gc100 | turn_only | clean/native | 0.0007257 | 0.8309 | 0.001582 | 0.008239 |
| gc100 | turn_only | clean/ordinary | 0.0007257 | 0.8309 | 0.001582 | 0.008239 |
| gc100 | turn_only | clean/pre_cue_reset_native | -0.0000024 | 0.8643 | 0.000000 | 0.000000 |
| gc100 | probe_glimpse | teacher/full_native | -0.4457166 | 0.6237 | 0.520850 | 0.999978 |
| gc100 | probe_glimpse | original/native | 0.0095322 | 0.8645 | 0.011686 | 0.423263 |
| gc100 | probe_glimpse | clean/native | 0.0028207 | 0.8280 | 0.009161 | 0.164502 |
| gc100 | probe_glimpse | clean/ordinary | 0.0028207 | 0.8280 | 0.009161 | 0.164502 |
| gc100 | probe_glimpse | clean/pre_cue_reset_native | -0.0000024 | 0.8635 | 0.000000 | 0.000000 |
| pgc33 | turn_only | teacher/full_native | -0.4471155 | 0.7309 | 0.533198 | 0.999986 |
| pgc33 | turn_only | original/native | 0.0000140 | 0.9984 | 0.000038 | 0.000392 |
| pgc33 | turn_only | clean/native | 0.0012313 | 0.9526 | 0.001609 | 0.036916 |
| pgc33 | turn_only | clean/ordinary | 0.0012313 | 0.9526 | 0.001609 | 0.036916 |
| pgc33 | turn_only | clean/pre_cue_reset_native | -0.0000006 | 0.9666 | 0.000000 | 0.000000 |
| pgc33 | probe_glimpse | teacher/full_native | -0.4422348 | 0.7275 | 0.534744 | 0.999967 |
| pgc33 | probe_glimpse | original/native | -0.0001176 | 0.9984 | 0.000228 | 0.002840 |
| pgc33 | probe_glimpse | clean/native | 0.0003832 | 0.9525 | 0.005136 | 0.061819 |
| pgc33 | probe_glimpse | clean/ordinary | 0.0005183 | 0.9526 | 0.005091 | 0.061870 |
| pgc33 | probe_glimpse | clean/pre_cue_reset_native | 0.0001825 | 0.9659 | 0.000735 | 0.007083 |

Facing the goal is descriptive, not an oracle utility measure: the native agent can strafe. The JSON retains every context, individual case alignment/probability, and pair TV for all8 interventions. The earlier clean inspection is preserved unchanged and linked by hash.

**There is no broadly expressed hidden turn-choice effect in the clean endpoints.** Mean paired action TV is 0.92% for Grid and 0.51% for PGC+grid, compared with 52.08% and 53.47% for their teachers. The largest individual pair differences (16.45% and 6.18%) are retained, so this is an aggregate statement rather than an assertion that every pair is identical.

The teachers' mean turn-weighted facing changes are -0.446° and -0.442° per update despite their positive goalward movement. This illustrates why facing alignment must remain descriptive, rather than a success criterion.
