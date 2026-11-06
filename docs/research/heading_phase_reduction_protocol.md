# Exploratory reduction of the learned heading dynamics

This secondary analysis was specified after observing pilot anchoring and before fitting the reduced model. It does not alter the locked confirmation protocol or training. All populations receive the same analysis, and a poor fit is retained.

The proposed one-dimensional equation is

`d phi / dt = f0(phi) + k sin(theta_cue - phi)`.

Here `phi` is the RNN's decoded remembered direction. `f0` is measured independently with all inputs zero and fitted with three Fourier harmonics plus an intercept. The cue coefficient `k` is then fitted to the additional phase velocity induced by native input. No sign constraint forces a restoring effect. The reversal prediction uses the same coefficient and a cue rotated by pi; it is not refitted. Ordinary-input and blank-input predictions use `f0` alone.

Initialization supplies the trained four-step heading/spatial cue, followed by 40 blank steps. One-step fields start from these actual 64-dimensional states. The fit uses positions (-0.8,-0.8) and (0.8,0.8), and even-indexed state and cue angles from a 72-angle grid. Held-out one-step fields use odd-indexed angles, and separate positions (0,0), (0.8,-0.8), and (10,10). Recovery trajectories use these same held-out positions with new state/cue combinations; their state angles include both even and odd indices. Neither held-out phase velocities nor held-out recovery trajectories are fitted.

Report held-out variance explained and phase-velocity error for the single-sine model, a constant-phase baseline, and the independently fitted no-anchor `f0` baseline. A three-harmonic cue term is a labeled diagnostic of deviations from a simple sine. A coefficient fitted only within 60 degrees is also diagnostic and must not replace the global fit after inspecting its results.

For the more demanding test, initialize new state/cue combinations at the held-out positions, then compare 200-step full-RNN recovery with the reduced equation integrated at the trained 50 ms step. Include initial phase substitutions of 0, +/-10, +/-45, +/-90, and +/-150 degrees, rounding to the measured five-degree state grid. Report trajectory and final angular prediction error separately for each physical context and input intervention. This deliberately tests whether a phase-only approximation survives the initial excursion away from the blank-input state curve.

For PGC, reversed input changes the learned displacement. For explicit-heading controls, the corresponding reversal changes the supplied heading components while keeping ordinary grid code fixed. For Grid, no directional cue exists and the reversal is a no-op. In addition, test self-generated directional input, whose cue follows the model's own current estimate: a genuine external-reference restoring term should disappear.

This is a reduced description of a learned network, not a ring architecture imposed during training. Blank memory persistence, normal attraction, heading-specific input anchoring, and predictive accuracy of this reduction remain separate measurements.
