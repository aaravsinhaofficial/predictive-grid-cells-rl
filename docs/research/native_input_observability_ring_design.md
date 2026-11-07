# Conditional follow-up: read the same memory through different actions

**Provisional design only.** This is a separate possible follow-up after the current 1,024-angle resolution study. It changes no current assay, threshold, cohort or result. No implementation, model run or new outcome is authorized by this note.

The useful question is whether an LSTM state carries a stable angular memory that becomes readable under several ordinary action contexts, even when the single held-input observation misses it. The smallest useful addition is **all six one-action response probes from independent copies of the same saved state**, with all 36 two-action probes as a separately declared finite extension. Start with actual actions; goal-code swaps add unnecessary interpretive complications.

## Why this is a different measurement

Albertini and Sontag define nonlinear state observability through output differences caused by common input sequences, in both discrete and continuous time. Their examples show that observability can depend on input coupling. This supports testing multiple fixed input sequences; their theorem for a particular sigmoid RNN class is not a theorem about this trained LSTM. [Primary paper, 1994](https://www.sontaglab.org/FTPDIR/obs-sigma.pdf).

Gated-network theory shows that inputs can change memory timescales and output participation. Here the standard LSTM equation makes the proposed effect concrete: holding incoming h fixed, the direct cell-to-next-hidden derivative is `o * (1 − tanh(c_next)^2) * f`. Different ordinary inputs can change this sensitivity. A large saturated cell can remain unreadable, and a query that opens a gate can also overwrite memory; both possibilities must remain visible. [Krishnamurthy, Can and Schwab, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9762509/).

A complementary empirical precedent is an uninformative “ping”: Wolff and colleagues revealed working-memory information through a common stimulus response. That motivates identical probes across headings. Their human EEG experiment used decoding and does not provide an LSTM ring-attractor result. Our proposed measurement would use complete fixed response vectors directly. [Primary experiment, 2017](https://www.nature.com/articles/nn.4546).

## Smallest truthful probe bank

Take an archived post-update state `z=(h,c)`, before its next environment action. Clone it separately for each native action: left, right, forward, backward, strafe left and strafe right. Execute that actual action in the matching environment clone; only then deliver its truthful previous-action one-hot to the next recurrent update. Merely relabeling the last action at the old timestamp is a synthetic gate intervention and cannot be described as a native action probe.

Keep RGB black, the ordinary current-code snapshot held, and the native goal-code snapshot fixed. Each complete probe input must be byte-identical across the entire heading family, including reward, action label, episode clock and termination handling. Actual zero reward is permissible when every branch truly receives zero; overwriting a heading-dependent hit or reward is not. Physical noise is paired across headings. Check physics validity before neural screening and retain ineligible contexts/actions with reasons. A common action conveys an angular increment, not an absolute starting heading. The construction follows `code/env.py` and the previous-action concatenation in `code/nets.py`.

Define a fixed response profile `Phi(z)` by concatenating **all six actor probabilities from every probe**, in a fixed order and with equal probe weights. Keep all-hidden outputs and centered actor logits as explicitly separate secondary profiles; they can reveal neural sensitivity that barely affects action probabilities. No selected neurons, fitted angle decoder, output whitening, probe selection or response normalization to a unit circle. A curve is indexed by the original physical headings, never by the query-bank index.

For a declared 36-probe extension, execute every ordered pair of actual actions from its own copy and append the final response. These sequences are not asserted to return the body to its starting position. Fresh images/current-position codes stay withheld, and truthful reward/termination must still be common. Keep every response, not only the action that reveals the clearest ring.

**The memory dynamics stay unchanged.** Evolve the original state under the existing LEFT or F/B retention map, without any query. Apply the exact same response bank independently to the original and evolved states. Query updates are a measurement cost and are reported separately from retention time. Repeated probing must not become the mechanism that maintains the supposed memory.

## Screening and normal directions

First reuse the existing coherent history families, full-dimensional curve checks and bounded h/c perturbations. Preserve all original raw/RMS kick types and both signs. For each finite kick report its response displacement, off-curve distance, phase component, visibility and later leakage.

A direction normal to the full h/c curve need not remain normal after a nonlinear observer. It may map to:

- **An observable normal:** it visibly moves the response off its curve; test its return relative to the identically evolved reference.
- **Phase only:** it moves along the response curve; measure phase persistence/transport, not transverse recovery.
- **An invisible fiber:** its initial response scarcely changes; challenge its future effects rather than divide by a negligible error or call it recovered.

All three categories remain in the denominator and raw tables. An original kick that is phase-only or invisible makes that transverse test inconclusive. It cannot be silently dropped to obtain an all-directions pass, and does not itself refute an observable angular factor. Conversely, a previously invisible error becoming visible later is explicit failure of the proposed functional equivalence for that probe set.

A **separately registered functional-quotient validation** can construct physical kicks normal in response space. Let `J=D Phi` and let `t` be the response-curve tangent. The phase covector is `J^T t`; projecting a predefined physical kick off that covector removes its linear response-phase component. Apply the projection within the h or c block for block-pure controls. Report the rank and conditioning of `(I−tt^T/||t||²)J`, and cover its observable normal directions with fixed deterministic directions or its complete retained singular basis. This is a local derivative of a fixed observer, not an angle regression. Check the **actual finite** direction after bounding/clipping; zero, invisible or reoriented kicks remain inconclusive. If there are no observable transverse directions, transverse attraction has not been measured.

The one-action actor bank has at most 30 independent response dimensions; the combined one/two-action bank at most 210. Its invisible fibers can therefore be substantial in a 512-dimensional LSTM state. All-hidden profiles can instead be locally injective. A well-conditioned injective observer cannot turn a genuinely non-attracting full-state system into an attracting one by a coordinate change; severe conditioning must not hide that distinction.

## Held-out validation and causal meaning

Freeze a depth-three action-sequence test set before screening. Every such sequence is a fresh physical counterfactual with truthful labels and common input tapes. Require that states judged equivalent by the discovery profile retain equivalent responses under these unseen compositions and after the declared retention windows. Include bounded h-only/c-only perturbations that initially lie near invisible fibers; do not test only easy visible normals. Profile agreement that breaks under held-out queries rejects the finite-bank quotient interpretation. A finite bank provides an empirical domain of validity, not all-input equivalence.

A surviving candidate then needs an intervention in the **stored state before query delivery**: small signed phase shifts along the genuine history curve, followed by held-out native action choices and pre-hit navigation. The behavioral error should change with the imposed phase; matched transverse and sham pulses supply controls. Restore a correct predictive cue versus ordinary and rotated predictive cues at the same physical location and true measured travel direction, then withdraw it before measuring repair. Preserve the legitimate ordinary-code position-history route. Input responses or angle-dependent clouds alone are not recovery or useful memory.

Positive synthetic controls should include a genuine circular memory gated silent during the delay but readable by the common probes. Negative controls should include a contracting plane/point, a query-created circular response without persistent memory, and a nuisance state invisible to the discovery bank but revealed by held-out probes. No new candidate is qualified if the assay mistakes any of these for selective, behaviorally relevant recovery.

A strong positive would support an **input-conditioned functional ring factor inside the LSTM**: the stored phase persists, observable errors return, held-out probes preserve the factor, and phase interventions alter navigation. Calling it an isolated full-state ring additionally requires return of the complete h/c state to a closed invariant curve. Persistent behaviorally inert fibers instead describe a larger recurrent system with a circular functional factor. For periodically driven F/B retention, retain that driving qualification; do not relabel a drive-locked loop as an autonomous ring.
