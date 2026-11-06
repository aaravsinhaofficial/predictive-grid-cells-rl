# Pre-outcome clarification of the reorientation mechanism comparison

Recorded 6 November 2026 UTC after independent protocol review. This addendum preserves the original frozen design and implementation hashes. No training outcome motivated the clarification, and no frozen training or geometry code is changed.

## What is held identical

In the primary native-context assay, the complete ordinary-delay input is identical **across angular histories and cue branches within one model and one context**. Native, ordinary and rotated endpoint pulses begin from the same pre-cue states; all their subsequent inputs are shared.

Grid and PGC models independently supply their first two eligible native contexts. Their goals, histories, saved-goal codes and learned CNN embeddings can differ. Comparing their native-context geometry describes the two trained agents in their own task contexts. It does not isolate a change in trained LSTM weights from those differences in drive and initialization.

The acute predictive-pulse comparison within one frozen PGC network has a cleaner intervention: only the 169 predictive current values change at the actual endpoint glimpse. Its phase-repair and radial-recovery effects remain distinct outcomes.

## Separate prospective common-input control

A stronger trained-LSTM comparison can cross-replay all four prespecified physical contexts in both networks. Preserve each source context's room, true goal, actual action sequence and controlled all-angle paths. Do not transfer one model's recurrent states into the other's learned coordinates.

For this explicitly additional intervention, initialize both full recurrent states at zero and feed the same complete input histories. Use a fixed common visual encoder (the original seed-1 Grid checkpoint's encoder), ordinary current maps throughout preparation and cue, and one ordinary goal code evaluated at the source context's true goal center. The two populations share the same seed-1 ordinary maps; verify all 1,287 inputs byte-for-byte, including the common black-image embedding, previous actions and rewards. The true-goal-center code replaces the native pickup snapshot and must be labelled as such.

Reuse all primary full-state and observable criteria unchanged. This control compares trained LSTM computations, including learned input and recurrent weights, under common drive and zero initialization. It is secondary to native-context behavior and does not replace a failed native candidate. Its cross-encoder, ordinary-input, goal-snapshot and state-reset interventions are reported explicitly.

## Physical and neural time indexing

Saved `*_states[0]` follows processing the endpoint-glimpse input, before the first imposed LEFT action. Each subsequent common-input update corresponds to the observation after one more actual LEFT action. Thus `*_states` at update 60 has already processed the first free-choice input, following 60 imposed LEFT actions.

Controlled navigation must take its **first free action directly from `pi(h)` at that saved state**. It must not call another recurrent update before that action. Advance the real environment, then process its next held-input observation for the second free action. Record exactly 30 free actions, with actual goal pickup interrupting the event as in training.

The physical heading advances according to the native acceleration and turn equations. A state rotating around a closed curve can satisfy the geometric test; stationary decoded phase is not required. Report updated body-heading and egocentric-goal-bearing error separately from retention of the original cue angle.
