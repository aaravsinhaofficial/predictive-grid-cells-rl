# Resolution refinement: bounded proposal

This is a proposal only. No new inputs, recurrent trajectories, or outcomes have been generated. The original 128-phase study and its zero nominations remain unchanged. Root owns authorization, freezing, scheduling, and launches; no refinement should compete with the current training/evaluation queue.

The completed original study has **536 rows that pass every necessary geometry check except spacing**. Accounting for exact shams, these represent 324 numerical rows, 48 physical input families, 62 branch families, and 121 branch/regime families. The spacing guard limits each neighbouring-state distance to 10% of the angular-family radius. A steep but smooth angular response can fail that guard at 128 samples.

| Original cohort | All rows, including shams | Numerical rows | Physical input families | Branch/regime families |
| --- | ---: | ---: | ---: | ---: |
| Grid, spacing only | 324 | 162 | 26 | 52 |
| PGC + grid, spacing only | 212 | 162 | 22 | 69 |
| Grid, spacing and/or departure | 2,060 | 1,030 | 64 | 190 |
| PGC + grid, spacing and/or departure | 2,000 | 1,571 | 63 | 342 |

These are computational denominators, not independent trained networks. The narrow cohort contains fixed hidden-output and actor-output rows, including a few noncentered Grid rows. **No full-state row passes every other check.** All windows and conditions enter the eligibility rule, not just the primary window or favorable contexts.

## Recommended first batch

1. Seal the exact IDs of every spacing-only row in the completed original summaries. Apply the identical eligibility rule separately to both fixed clean and both fixed noise endpoints when their original-resolution summaries finish. No behaviour-based filtering. Keep alias records and excluded-row reasons. Retain the other metric as a mandatory comparison if only one metric is eligible; this adds no rows to the original cohort and ensures that future raw/RMS qualification always has its required counterpart.
2. Generate **1,024 actual physical headings once per required input family**, using the unchanged `prepare_family(..., phases=1024)` helper. It already supports this parameter. Each heading receives the genuine rendering, grid-code calculation, actual control tape, and fixed goal. Reuse prepared inputs across checkpoints because their CNN/input maps are fixed. Replay each checkpoint's complete native prefix to obtain its own incoming h/c.
3. Derive the nested 256-, 512-, and 1,024-sample analyses by selecting genuinely simulated headings from that dense batch. No interpolation or fitting of hidden states. Require exact shared physical trajectories, headings, nonvisual codes, goals, action labels, rewards, and common future inputs. Allow only the CNN columns an absolute 1e-5 tolerance for differing float32 convolution batch sizes, with every error reported. Shared h/c errors must be below 1e-8 + 1e-5 times that coordinate's maximum magnitude across phases. Separately within hidden and cell blocks, the maximum centered row-RMS error must be below 1e-8 + 0.1% of that block's angular RMS. Record RMS/max discrepancies, common-mean shifts and ratios. A large common cell baseline cannot license a changed hidden pattern. A source/history mismatch stops qualification.
4. Reuse the original geometry criteria, clocks, physical phase comparison, output definitions, raw/RMS metrics, and two separate F/B sections. Save all three densities for every eligible row, even if a finer density loses a gate that passed earlier.
5. Require agreement at **512 and 1,024** before a positive refinement result. As an additional numerical-resolution check, compare each 1,024-point initial/evolved curve with the corresponding 512-segment polygon: its maximum discrepancy must be below 0.005 of the initial radius. Report all gate values and physical-phase differences at each density. A nonconverged case is unresolved, not positive; this check strengthens rather than relaxes the original criteria.
6. Only converged geometry-qualified rows receive the existing finite-normal tests, at both 512 and 1,024 samples. Preserve the eight fixed physical anchor angles across the circle, all 20 normal directions, both signs, tangent controls, and bounded-kick rules. Retain per-anchor outcomes. Require the original recovery criteria at both densities and report tangent changes caused by denser sampling. These remain screening candidates: eight anchors can miss poor recovery in steep portions. Any positive requires subsequent dense-phase/Jacobian confirmation, held-out tests, and behavioural cue repair.

Before native outcomes, add a resolution control: a genuine attracting circle with strongly nonuniform but smooth phase parameterization must fail the coarse spacing guard and become correctly resolved at higher density. A contracting plane and an action-locked point must remain negative. Use the existing attractor controls, not a learned angle projection.

## Why consider the broader cohort separately?

The current distance calculation projects onto every polygon **segment**, with interpolation, rather than only to sampled vertices. Nevertheless, a coarse polygon can miss a curved portion between vertices. Refining rows that fail only spacing and/or departure is therefore a principled **polygon-discretization diagnostic**. It covers 127 of 128 physical input families and substantially increases cost. It must be a separately declared cohort with the same convergence and recovery rules; omission of the departure gate is only an eligibility rule, never a changed pass criterion.

Some other phase-order and separation statistics also depend on the polygon. Neither targeted cohort can support a statement that every possible angular family has been excluded. The narrower cohort is the clearest first test because all its non-spacing checks already pass.

## Local cost estimate and stopping rule

The actual 128-phase preparations took about 377 seconds per population for 64 families, running concurrently. They stored 41.7 MB per input family. Original geometry took 474 seconds for Grid and 680 seconds for PGC. Linear preparation scaling gives about **21 minutes wall time and 16 GB input storage** for the original narrow cohort at 1,024 phases, with two population workers. Allow roughly **45–60 minutes total** for dense trajectories, three-density geometry, saved-data audits, and short-window finite probes; this estimate needs a measured first-family throughput check. Most eligible rows have a two-update evolution plus a one-update output observation; a small Grid subset has 7-, 14-, or 30-update evolution.

The broader original cohort would require about **50 minutes for preparation**, roughly 42 GB of input storage, and potentially **90–150 minutes total**. Quadratic polygon work and the number of recovery-qualified rows make that estimate less certain. Across all six endpoints, dense inputs can be reused; the union is bounded by 128 physical families, but recurrent trajectories and normal tests must be evaluated for each checkpoint.

A dense input family is approximately 334 MB. Temporary lists plus stacked inputs, the current renderer/CNN batch and numerical workspaces suggest **up to roughly 3 GiB peak resident memory per worker**; this is an estimate, not a measured native run. Families are prepared and processed one at a time, and peak resident memory is recorded. No full 16 GB input bank is loaded into memory at once.

Proposed cap: **60 minutes local wall time per population/endpoint cohort**, including any missing input preparation and own-prefix replay, no cloud. Two population workers can run concurrently under root scheduling. Progress and incomplete cases are preserved if the cap is reached. A cap hit requests a scheduling decision; it does not silently drop rows or change density, and automatic selective retry is refused. Before authorizing later endpoints or the broader cohort, replace the estimate with measured throughput and its exact eligible count. Root subsequently authorized implementation of the narrow study only; review, sealing, and native launches are still pending.
