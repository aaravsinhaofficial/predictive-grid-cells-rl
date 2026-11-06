# Original RL goal-state pulse: fixed exploratory replication

The seed1 pilots were inspected before this protocol. The remaining original seeds2–8 are a fixed replication; all eight seeds and both trained populations are retained. This experiment is exploratory, not a prospective confirmation of a PGC advantage. It uses the original 1B-frame A2C checkpoints, original square arena, original visual observations, six actions, original reward and respawn rules, and original stored pickup-code semantics. All policy parameters remain frozen. No heading-supervised network or external controller is used.

## Intervention

Collect64 native episodes per model with evaluation seed872601. Select the first known-goal return state in each episode that has at least20 steps since pickup, goal distance0.9–1.8m, a complete300-step continuation, and valid real/rotated pickup and goal coordinates. The nominal angular displacement is15 degrees, alternating sign by episode slot. Both rotation signs must satisfy map bounds. No earlier observation in the15-step conditioning window may fall inside the fictitious goal disk. Episodes without an eligible state are reported, not replaced.

Replay the actual preceding15 observations, previous actions, and rewards from their actual original LSTM state. In the counterfactual replay, rotate only the supplied saved goal snapshot around the endpoint observer: rotate pickup position and pickup arrival direction/velocity together. This preserves the original hexagonal map and predictive displacement rule. The true physical goal and observations never change. Replaying the unchanged message must reconstruct the native endpoint state within4e-4 maximum absolute error.

At the same physical endpoint compare four states: native baseline; the full counterfactual `(h,c)`; counterfactual `c` with native `h`; counterfactual `h` with native `c`. The full state is elicited by a valid alternative goal input. The two hybrids are explicit component interventions; they are not claimed to be naturally occupied states or manifold tangents. The `c`-only intervention must leave the first action unchanged.

Restore the actual real goal snapshot on subsequent updates. Compare native inputs with ordinary replacement of the predictive slots in the saved goal code. The frozen implementation also retains current-code ordinary replacement as a secondary diagnostic; the principal report focuses on native and goal-ordinary inputs. Ordinary replacement evaluates the same cell phases at zero spatial shift, not zero activity. A Grid-only network must therefore give bitwise-identical results under the three input labels.

## Measurements

1. Under identical future observations/actions/rewards, measure total-variation distance of action probabilities and full/c/h state differences from the corresponding unperturbed input-condition baseline at0,1,5,15,30,60 steps. Report the0–1s mean action effect and its successive decline. A remaining effect of `c` on the original actor demonstrates a recurrent contribution, without presuming an angular attractor.
2. Measure the first decision's expected translational-action change projected toward the fictitious goal rotation, in world coordinates. This checks whether the state substitution has the intended functional direction.
3. Let the original actor control cloned original environments for300 decisions (20s), with matched physical/image noise and action-sampling uniforms. Report first-second path displacement and signed transverse movement, first goal-hit probability and capped first-goal time. Preserve timeouts. Outcomes after a hit are not interpreted as a single continuous return path.

The original goal-center coordinates are used only for experimental stimulus construction, trial eligibility and scoring. They are never supplied to the policy. Actual goal snapshots remain boundary pickup snapshots, not goal-center codes.

## Aggregation and claim boundary

Each trained seed is the population-level unit. Average eligible trials within a model, then report all eight paired seed means, descriptive95% Student-t intervals and exact two-sided sign-flip statistics where a contrast is useful. Do not count individual trajectories as independent trained networks. Report missing/failed runs and every eligibility count; no seed is excluded. The source manifest freezes the implementation before seeds2–8 are run. Existing seed1 outputs retain their original exploratory provenance.

This assay distinguishes a short actor-facing `h` effect from a continuing goal-conditioned `c` effect and tests whether predictive goal input changes recovery. A goal-dependent state family or lingering cell-state effect does not imply a ring attractor. The pilot did not show superior PGC repair; subsequent results must retain that possibility.

The earlier linear-decoder edit is preserved separately: a nominal15-degree linear readout shift produced only1.2 degrees in an independent nonlinear checker and negligible behavioral effects. It is not used as evidence for a15-degree functional memory displacement.
