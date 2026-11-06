# Secondary cue-dependent linearization

Specified after exploratory seed1006 and before evaluating any confirmation spectrum. Evaluate all16 PGC training seeds2001–2016, without excluding seeds by gain, accuracy or ring quality. This is a mechanistic secondary family, separate from the four locked primary endpoints.

Initialize72 evenly spaced directions using the shared four-step cue and40 blank steps at the five prescribed in-region positions and the separate(10,10) position. At these same full recurrent states, compare Jacobians under zero input, native PGC input and ordinary replacement. Also report the same comparisons after200 native cue steps as a second operating-point analysis.

The primary comparison within this secondary analysis is the in-region mean of the five median one-step phase gains under native versus ordinary input at the cue-free states. Also compare native versus blank gain and native versus ordinary spectral radius. Report all16 paired effects,95%t intervals and exact seed sign-flip tests with Holm correction over these three contrasts. The native-anchored operating points and distant position are descriptive sensitivity checks.

For the leaky tanh update, `J=(1−leak)I+leak·diag(1−tanh(preactivation)^2)W_rec`. Prediction changes the input-dependent gain matrix while recurrent weights remain fixed. The phase gain is `gradφ(h_next)·J·t / [gradφ(h)·t]`, with the empirical full-state tangent `t` and the network's trained sine/cosine readout. This gain is a coordinate derivative, not an eigenvalue. Report eigenvalues, tangent alignment, singular gains and phase-readout amplitudes separately. Mark numerically undefined phase gains; never treat them as zero recovery. The current-normal projection omits leakage into the tangent and is only a diagnostic, not a full transverse-stability result.

Independent checks use autograd for full Jacobians and centered finite differences for phase gains. Full-state finite-time recovery, true-heading accuracy and retained circular coverage are evaluated by the separate memory/anchoring experiments, so collapse cannot be interpreted as successful phase correction.
