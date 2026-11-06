# Can one predictive cue repair useful memory?

This analysis is frozen before the new blind-reorientation training outcomes. It follows the [prospective mechanism protocol](../../../docs/research/reorientation_mechanism_protocol.json); results are pending.

The test changes the agent's recurrent state just before its one spatial glimpse. It then supplies the correct predictive cue, ordinary grid values, or a cue pointing 60° in the wrong direction. Every later input during the forced turn is identical. The cue lasts one update.

- **Change the remembered direction:** substitute the full state elicited by a direction 30° clockwise or counterclockwise.
- **Push away from the activity curve:** apply a small radial disturbance, with equal-sized random disturbances as controls. Repeat after one second without the cue.
- **Erase the state after the cue:** check whether any lasting benefit travels through recurrent memory.
- **Let the original actor navigate:** after four seconds of turning, let it choose thirty actual actions. Measure initial goalward progress, path, and goal hits, retaining every unsuccessful window.

The full 512-dimensional state is compared with both the unperturbed correct-cue family and its own cue condition. All four fixed model/context combinations and both distance scalings are retained. A collapsed family receives no invented direction: undefined perturbations or phase estimates are marked, and its unperturbed navigation still runs. Angular samples are repeated measurements of one trained seed pair.

The controlled forced delay uses noiseless native physics to match the saved neural trajectories. Free navigation uses native noisy physics, the real goal, the held ordinary endpoint code, and the original actor. All branches receive identical pre-drawn motor noise and action-sampling uniforms. The first free action uses the saved state immediately; there is no extra recurrent update. Each window stops at its first actual goal hit, before respawn. This controlled test is separate from evaluation under the task's usual noisy forced turns.

Independent code review and four tests passed. A shape-only check on an original PGC checkpoint verified full-state cue trajectories and the thirty-action interface; it was not used to select a model or claim a mechanism.

[Frozen implementation manifest](protocol.json) · [Analysis source](../../../scripts/lstm_reorientation_causal_navigation.py) · [Tests](../../../tests/test_lstm_reorientation_causal_navigation.py) · [Shape check](original_shape_smoke.json)
