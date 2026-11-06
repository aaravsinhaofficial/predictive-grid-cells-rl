# Matched population extension of the frozen memory-pulse analysis

Added after the three population pilot results and before evaluating these pulses in confirmation Grid or Grid+explicit-heading checkpoints. This extends the existing secondary analysis; it does not change the locked four primary tests, checkpoint selection, memory pulse, controller, noise, endpoints, or source files.

Run the same frozen `evaluate_heading_memory_pulse.py` on all 16 Grid and all 16 Grid+explicit-heading final checkpoints as well as the already scheduled 16 PGC checkpoints. Keep the same four evaluation seeds and 128 trials per model. Retain every training seed.

The useful paired comparisons are the effects of the phase pulse relative to each population's unperturbed memory and equal-norm normal pulse. Compare early heading error and induced path deviation between populations, and report the arrival/dwell cost of the pulse. The explicit-heading control tests whether an independent directional reference can reproduce the protection. Descriptive uncertainty must use the 16 paired training seeds, not individual frames or trials as independent replicates.

This extension was motivated by the pilot result that native Grid agents retain a larger phase disturbance, whereas both PGC and explicit-heading agents repair it. It avoids relying only on the within-PGC ordinary-code replacement branch, which already impairs baseline behavior substantially. Treat these as secondary replicated results, not retroactively added primary endpoints.
