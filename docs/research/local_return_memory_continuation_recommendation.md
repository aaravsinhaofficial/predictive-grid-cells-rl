# After fast steering: a bounded native-memory continuation

**Prospective recommendation only.** The acquisition protocol remains frozen at `ad969e782ec2943de2e59b2d72629002dea537f7c9c0fcf2ad3905ce051038c7` (commit `016ec248`). No new training, bank, policy outcome, implementation or freeze accompanies this note.

**Keep acquired navigation; remove feedback only after a clear encoding period.** If no acquisition final meets its fixed 192/256 fast-steering criterion, stop this proposed sequence. If competence emerges, retain all four prescribed final arms and their failed counterparts; do not select the best intermediate checkpoint or actor mode. Continue each exact final policy and RMSProp state, without resetting heads. This is a development comparison within one original seed pair.

## One change in what the agent must remember

Keep fresh native RGB and current spatial code throughout the preparatory turns and four forward probe actions, as in acquisition. **At the endpoint input, black out RGB; give one native current-code observation, then hold the ordinary endpoint code through the remaining LEFT delay and all 30 FREE actions.** Retain the actual saved goal, previous actions, brakes, physics and real FREE reward. The endpoint input is the first delay input, not an extra step. No new angle, clock, target action or ring loss enters learning.

This avoids reinstating the earlier blind-preparation distribution change. The animal-like sequence is encoding with feedback, a delay without it, then a rewarded action. Recurrent policies have learned reward-based physical memory control, providing a relevant learning precedent rather than a guarantee about geometry. [Heess et al., 2015](https://arxiv.org/abs/1512.04455).

| Continuation setting | Fixed proposal for later owner review |
|---|---|
| Training | Three blocks, each 1,024 batches × 32 trials; maximum LEFT delays 4, 15, 60, sampled uniformly from half-maximum through maximum |
| Geometry/response | Same .58–.62 m goal-center distance, .30 m goal radius, 30 FREE actions; no near-goal restart curriculum |
| Learning | All weights trainable; retain acquisition optimizer; same native reward, complete-trial credit, entropy 8e−5 and other settings |
| State noise | Zero for this first memory-acquisition test; native FREE motor noise unchanged |
| Budget/stop | 98,304 additional trials per arm, at most 393,216 total across four arms; retain every block, judge the prescribed last endpoint, no automatic extension |

Choose and seal the new training streams and final 256-trial memory bank before execution. The first short-delay block already requires roughly two seconds of blind response, so it tests learned steering from remembered direction; the later blocks extend retention. Record each fixed block without using it to retune the schedule. A separate infrastructure review must fit this finite proposal within the remaining compute ledger.

## Which predictive input is relevant?

The **current-code pulse** evaluates 169 predictive maps ahead of the endpoint along the actual last travel direction. The straight probe aligns travel and body heading at that moment. Subsequent LEFT turns change body heading while that last travel direction stays fixed. Thus the cue can refresh the probe direction; the recurrent computation must retain it and account for later turns. Track initial travel direction, physically updated body heading and egocentric goal bearing separately.

The **saved PGC goal code** is different: it comes from the earlier actual pickup and includes its independent arrival direction. Hold it fixed across heading histories, state edits and cue replacements. It can aid goal representation but cannot identify the heading variant within such a matched family. Replacing the endpoint predictive CURRENT slots does not ablate that goal input or earlier predictive observations. This distinction connects the experiment to the repository's actual code rather than treating every PGC feature as a compass.

At a fixed endpoint and goal, heading and egocentric goal bearing differ by a known sign/offset, so a single family cannot identify their coding frame. Use the two prespecified goal contexts and actual actor effects to interpret the variable. A correct pulse might repair a disturbed directional state while adding little to an already accurate memory; intact cue-removal equality alone is not evidence against that repair mechanism.

## Behavior first; transfer the rule, not old evidence

At each fixed endpoint report the matched native/ordinary/±60° endpoint cues, post-cue h/c erasure and every-blind-step erasure. Also retain a **fully refreshed RGB-plus-code** control, clearly distinguished from the old RGB-only “seen” condition.

The old **167/256 native successes and at least 39 fewer after post-cue erasure** can be reused unchanged as a descriptive useful-memory screen on the new, fully specified hard bank. These counts are not a trained-seed significance test or a test of PGC necessity. The change in encoding history is disclosed, so this is a new experiment, not a replay of the failed old task.

Recompute all 128 opposite-heading pair certificates from the new saved states, inputs and tapes. The old bank's 50% common-information ceiling does not transfer automatically. Require the unchanged valid-certificate condition before conditional geometry; retain failed pairs and their conservative aggregate bound. A failure of the final gate stops automatic ring analysis and further continuation. It identifies unsuccessful memory acquisition under this recipe, rather than the geometry of a successful memory.

## Then distinguish attraction, cue repair and useful action

Replay **actual histories from the new collector**, not the old blind-preparation histories. Reuse the existing [dense-family methods and unchanged criteria](local_return_mechanism_design.md): first two sealed contexts, 768 headings, common ordinary delay input, full h/c and the fixed observable maps, targeted radial and random disturbances, physically corrected phase transport and native float32 validation. A constant LEFT command permits phase motion; it is not a demand for a stationary state.

For the same recipient states, swap genuine ±30° pre-cue histories, cross with native/ordinary/±60° one-step cues, withdraw the pulse, and score both persistent state correction and actual 30-action navigation. Keep matched radial/random controls and post-cue erasure. A pulse changes the state; it does not change the subsequent fixed-input transition law. Report autonomous transverse recovery and acute phase repair separately before linking them to behavior.

Do not require an exact continuum of fixed points: trained LSTMs can have a useful slow closed manifold with preferred points. Conversely, continuous angle information alone can live in a nonattracting plane, and coarse reward tolerance can support discrete memories. Ságodi et al.'s noise-trained, supervised angular tasks motivate the stability tests, **not importing their sine/cosine target or promising a reward-trained ring**. [Primary paper, especially S7.7](https://arxiv.org/html/2408.00109v2). If useful memory emerges without the fixed ring criteria, report that measured mechanism; do not automatically add noise, narrow the goal or change thresholds to seek a positive ring.

A successful within-PGC cue/state intervention would establish correction by the actual predictive current input in an adapted native navigator. Between-population stability also depends on their different learned embeddings and saved-goal codes; use the already proposed common-input sensitivity if making a trained-weight-only claim. Explaining the original navigation advantage specifically through this new angular mechanism would still require its causal use in the original task. Neither the acquisition result nor a newly learned ring replaces that distinction.
