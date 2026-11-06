# Why the predictive cue stabilizes ring phase

This derivation describes the **explicitly designed ring model**, not an inferred exact equation for the unconstrained trained RNNs. Its sensory readouts and cue gain are learned; its continuous-attractor geometry and correction law are imposed. The [implementation](../../scripts/designed_predictive_ring.py), [eight-seed experiment](../../mechanisms/designed-predictive-ring.md), and [exact-map checks](../../tests/test_designed_predictive_ring.py) accompany the derivation.

## 1. The spatial code contains a direction difference

For one ideal Fourier component with wave vector `k`, ordinary and predictive spatial phases are

`ψ_current = k·x`,

`ψ_predictive = k·(x + d u)`,

where `u = (cos θ, sin θ)` is measured travel direction and `d = 0.20 m`. Consequently,

`ψ_predictive − ψ_current = d k·u`.

The common position cancels. Stacking non-collinear components gives `Δψ = G u`, with rows of `G` equal to `d kᵀ`; a known-geometry pseudoinverse recovers the two-component direction vector. Using a wrapped phase requires `|d k·u| < π` for the selected components. The constructed decoder therefore uses the coarser modules 1–3. The finest module can violate this condition and create a direction alias.

The actual rate channels are the repository's squared three-cosine grid functions. We fit linear readouts of the complex phase components from independent calibration examples; no phase is directly supplied at inference. The fitted current and predictive complex readouts are multiplied as `z_predictive × conjugate(z_current)`, and their arguments provide the relative phases. The original position or direction labels do not enter the inference call.

This cancellation explains an available computation. It does not identify literal phase subtraction as the unique algorithm learned by the separate unconstrained encoder. Nor does it imply that ordinary grids lack direction information in their temporal history. The constructed Grid control uses the same instantaneous phase-readout architecture.

## 2. A genuine ring in a 64-dimensional state

Let `U` be a 64×2 matrix with orthonormal columns. Write any state with a nonzero phase-plane component as

`h = U q + v`, with `Uᵀv = 0`, `q = r u(φ)`, and `r > 0`.

There are two coordinates in `q` and 62 in `v`. The ring is the set

`M = { U u(φ) : φ ∈ [0, 2π) }`.

The exact update first predicts the completed angular movement,

`φ_prior = φ + Δt ω_measured`,

then assimilates the current decoded sensory vector `z`:

`φ_next = φ_prior + Δt κ det(u(φ_prior), z)`,

`r_next = 1 + ρ(r − 1)`,

`v_next = ρv`,

`h_next = U r_next u(φ_next) + v_next`.

Here `ρ = 0.9`, `Δt = 0.05 s`, and learned `κ` has units per second. The decoded cue vector is bounded in magnitude at 1.5 and retains its original magnitude near zero. An ordinary-code difference close to zero therefore remains a weak signal. The cue is zero when unavailable. Recurrent noise is added separately in the noisy navigation experiment.

With both cue and angular-motion input absent, `φ_next = φ`. Every point on `M` is fixed, while radial error and all 62 orthogonal coordinates shrink by `ρ`. The ring is therefore transversely attracting and neutral along its phase, on the domain `r > 0`.

## 3. The cue changes exactly one local stability direction

Consider a stationary cue `z = m u(θ)` and a state on the ring aligned with it. For phase error `e = φ − θ`, the exact angular map is

`e_next = e − α sin e`, where `α = Δt κ m`.

Linearizing at `e = 0` gives the phase multiplier `λ_phase = 1 − α`. The radial multiplier and the 62 other multipliers remain `ρ`.

For the unit full-state tangent `t = U u_perp(θ)`, the aligned Jacobian is

`J_cue = ρI + (1 − α − ρ) t tᵀ`.

Without the cue,

`J_blank = ρI + (1 − ρ) t tᵀ`.

Thus the cue changes the Jacobian by the rank-one term

`J_cue − J_blank = −α t tᵀ`.

This is the mechanism in full neural coordinates: **the cue supplies restorative feedback along the memory direction while leaving transverse recovery unchanged**. The eight trained interfaces select `κ = 12` or `16`, so an aligned unit cue gives phase multipliers 0.4 or 0.2. The blank spectrum is `{1, 0.9 ×63}`.

Alignment matters. At a phase error `e`, the angular-coordinate derivative is `1 − α cos e`; the full-state Jacobian away from a fixed point is not obtained by simply reusing the aligned eigendecomposition. A reversed cue makes the direction opposite the true heading stable. Contracting errors toward that wrong cue is stable but inaccurate memory. This reversal uses the frozen trained decoder: a decoder trained from the outset on a backward-shifted code could learn its opposite sign. The result demonstrates dependence on the trained cue relationship, not an exclusive advantage of forward over backward codes.

For a correct constant cue and `0 < α < 2`, every angular error strictly between `−π` and `π`, except zero, decreases in absolute value under the deterministic map: `|e − α sin e| < |e|`. The exactly opposite state is an unstable exception. This elementary convergence result applies to the designed map; it is not a theorem about the unconstrained 64-unit RNNs.

## 4. Why an independent cue controls accumulated error

For small errors, a unit directional cue and angular measurement noise give the approximate update

`e_next ≈ (1 − a)(e + Δt b + Δt ε_ω) + a ε_θ + ε_recurrent`,

where `a = α` on visible-cue steps and zero otherwise, `b` is angular-sensor bias, and the three `ε` terms represent independent errors. If availability is independent with probability `p`, the mean error multiplier is `1 − pα`. With a constant bias, the corresponding steady mean is

`mean(e) ≈ [(1 − pα)/(pα)] Δt b`.

Without a reference (`pα = 0`), bias accumulates instead of approaching this finite mean. In the zero-bias, small-error case, mean-square stability requires `E[(1 − a)²] < 1`, giving the same local bound `0 < α < 2` when `p > 0`.

This approximation explains the tradeoff controlled by the learned gain: stronger correction reduces integrated drift but passes more cue noise into memory. It is an analytical interpretation, not a fitted quantitative prediction of every noisy navigation trajectory. A cue generated solely from the network's own mistaken estimate does not provide the independent information assumed here.

## 5. Numerical precision and connection to the learned RNNs

The spectrum audit uses an explicitly re-orthonormalized double-precision basis and matches the analytic spectrum to approximately 2×10⁻¹⁵. It also uses cue magnitudes decoded from actual spatial-rate inputs. The operational float32 model is checked separately: its noise-free circle drifts by less than 8×10⁻⁷ in full-state distance over ten seconds. This separates the mathematical design from finite-precision arithmetic.

The unconstrained learned RNNs have no imposed `U`, radial law or rank-one Jacobian change. Their measured Jacobians change through input-dependent activation slopes, and their geometry can include sharp hidden-state transitions. A learned direction-carrying input plane, same-state Jacobian comparisons, held-out phase-field predictions and behavioral interventions provide empirical links to the simple mechanism. The designed model supplies a controlled existence proof; it does not replace those empirical checks or turn every learned network into the same ring.
