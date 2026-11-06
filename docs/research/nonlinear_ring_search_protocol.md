# Further ring searches in the original navigation agents

Started 2026-11-06 after the completed stationary goal-memory diagnostic. This is a new exploratory study. Earlier results and their criteria remain unchanged.

The target mechanism is a circular organization of recurrent activity that helps navigation and becomes more stable with predictive grid input. A closed drawing alone is insufficient: the actual network dynamics must preserve the circular organization, recover from transverse perturbations, and use the circular coordinate to influence behavior.

## Complementary tests

1. **Calibrate on a published learned ring.** Use the original 256-unit Memory-Pro RNN supplied by Marraffa et al. (2026). Independently reconstruct its full dynamics, qualify its fixed points, and trace the two unstable branches of a saddle. Test closure, invariance and recovery in all native state coordinates. Compare the continuous differential equation with the authors' discrete Euler implementation. Validate the supplied nonlinear reduced model separately. This is a method check, not evidence about the navigation LSTMs.

2. **Search for missed unstable states in the original LSTMs.** Use the same recorded fixed-input cases as the earlier equilibrium analysis: Grid and PGC, original seeds 1–2, context 0, saved goal held or withheld. Solve the exact eliminated-cell equilibrium equation with analytic Newton derivatives, random starts and deflation of previously known roots. Retain all 64 attempts per case. Validate any new root in the full 512-dimensional recurrent map, calculate its full spectrum, and compare its cell-memory scale and distance with native trajectories. If a relevant saddle is found, trace its branches in the actual LSTM before fitting a reduced model.

3. **Test recurrent movement during real navigation.** Record 32 native episodes from each of the four discovery models. Preserve the original maps, six actions, goal messages, rewards and noise. Census full heading revolutions and turn–strafe transitions using physical criteria fixed before collection. Where a usable repeated movement exists, perturb recurrent state along and across its local trajectory. Compare real closed-loop behavior with replay of the same recorded inputs. Replace only the current predictive-grid channels by their ordinary-grid counterparts, then restore predictive influence at the candidate-update and output gates. The saved goal message remains native throughout this particular comparison.

4. **Check a different circular coordinate: grid-module phase.** Added after the bounded equilibrium search recovered no new branches, and before this assay's outcomes. Each grid module has two periodic lattice coordinates. Sweep one module's saved goal code through one lattice period, keeping its predictive offsets and all other input components fixed. This is an explicitly synthetic code intervention; except at the common reference point, the modules need not describe a single physical location. It isolates possible recurrent modes that a heading or goal-bearing sweep could miss. Elicit recurrent states, restore one common input, and test the same full-state persistence, invariance and recovery criteria. Any positive result requires a subsequent check using physically consistent native histories and held-out models.

## Follow-ups added during discovery

These are explicit stages of an exploratory search, not a claim that the complete search was specified before its first result.

- After the complete module-phase state screen, apply the existing fixed future-hidden and future-actor observable assays to every family. Their definitions and thresholds stay unchanged; the circular coordinate is new.
- Test every resulting candidate with additional full-state, hidden-only and cell-only disturbances. Preserve the original selected-direction results. Measure both signs, several horizons and both original state metrics.
- Test both module phases together. A disturbance perpendicular to one ring slice can lie along the second phase of a torus. The two-phase assay removes both advected phase tangents and independently checks surface geometry, invariance, rank and recovery. Periodic input coordinates alone do not qualify a torus.
- After the two-phase test, account locally for all eight known grid-module phase directions. Compare the same disturbances with two versus eight tangent directions removed. Bound all implied phase changes, including compensating changes in the selected module, to 0.1 lattice period. Report projected residuals and locality failures together. This checks a nuisance-direction explanation; it is not a global eight-dimensional attractor assay.
- After the native movement pulses revealed hidden-output bound overshoots, repeat the same cases with bounded pulses. Clip only the proposed hidden coordinates to their legal interval, then shrink every nonzero pulse in a case to a common realized norm. Preserve cell components during clipping, record the direction changes, and retain both the original and bounded results. This is a numerical-domain robustness check, with no new case selection.

## Interpretation and confirmation

- Recovery only when actions change subsequent inputs suggests stabilization through agent–environment feedback.
- Convergence to a unique state sequence under common input suggests an input-driven response.
- A circular family that preserves distinct phases under common input, preferentially removes transverse errors and changes behavior through its phase is a recurrent circular-manifold candidate.
- An oscillatory cycle and a ring of angular memories are reported separately.
- Original seeds 1–2 are discovery cases. A positive candidate requires a frozen follow-up on held-out original seeds 3–8; exploratory episode counts are not independent trained-model replications.

The previous work already tested moving tangents, time-evolving reference curves, hidden activity, future-output observables, saturated cell-state spectators and constant-input slow curves. Repeating those calculations is not counted as a new method.

All current work is local. The original $500 cloud authorization is cumulative: the last completed ledger reserved $71.1726 including ancillary costs, leaving $428.8274. This study does not reset that authorization.

See [primary literature and model provenance](alternative_ring_methods_literature_2026.md). Each executable experiment records its own numerical settings and source hashes. Failed attempts and subsequent methodological corrections remain distinguishable.
