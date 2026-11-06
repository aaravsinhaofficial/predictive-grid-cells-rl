# Complete learned-heading results

All 16 planned training seeds are included. Intervals are nominal 95% paired-t or group-mean intervals across seeds, not simultaneous intervals. Exact two-sided sign-flip tests are corrected within the declared families. Individual trajectories and phases are not independent training replicates.

## Four locked primary endpoints

| Network | Native heading error (degrees) | Error at new places (degrees) |
| --- | ---: | ---: |
| Grid | 28.143 [27.675, 28.612] | 28.132 [27.652, 28.611] |
| PGC + grid | 7.418 [7.180, 7.655] | 7.977 [7.602, 8.351] |
| Grid + explicit direction | 4.801 [4.711, 4.891] | 4.802 [4.712, 4.892] |

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| PGC − Grid: native heading error (degrees) | -20.726 [-21.191, -20.260] | 0.00012 |
| Ordinary replacement − native PGC: error (degrees) | 66.032 [64.581, 67.484] | 0.00012 |
| Native − ordinary: retained local phase gain at 5 s | -0.009 [-0.018, -0.00025] | 0.00012 |
| PGC − Grid: error at new places (degrees) | -20.155 [-20.731, -19.579] | 0.00012 |

The accurate-anchoring criterion is met by 15/16 seeds. This criterion combines decoded coverage with lower true-heading error than ordinary replacement and is never used to exclude a model. Ordinary replacement can contract disturbances by collapsing toward a wrong state; phase gain alone is therefore not evidence of useful anchoring or a globally smooth ring.

## Direction-plane intervention at new places

| Frozen-PGC intervention | Heading error (degrees) | Goal reached (%) | Sustained goal holding (%) |
| --- | ---: | ---: | ---: |
| Native | 8.225 [7.821, 8.630] | 77.393 [75.135, 79.650] | 77.148 [74.851, 79.446] |
| Ordinary replacement | 75.426 [73.743, 77.109] | 6.787 [5.152, 8.422] | 3.955 [2.626, 5.284] |
| Remove direction plane | 71.292 [69.062, 73.521] | 8.154 [6.349, 9.959] | 5.176 [3.590, 6.762] |
| Restore direction plane | 10.047 [9.383, 10.711] | 69.775 [65.700, 73.851] | 68.994 [64.910, 73.078] |
| Restore random planes (within-model mean) | 75.900 [74.296, 77.505] | 7.161 [6.424, 7.899] | 3.939 [3.333, 4.544] |

Heading improvements are measured in degrees; positive contrasts mean improvement or a removal cost, as specified in the frozen protocol. Navigation contrasts are percentage points.

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| native benefit | 67.201 [65.568, 68.834] | 0.00012 |
| removal cost | 63.067 [60.833, 65.300] | 0.00012 |
| restoration benefit | 65.379 [63.591, 67.167] | 0.00012 |
| specificity over random | 65.853 [63.998, 67.708] | 0.00012 |

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| native benefit | 70.605 [68.533, 72.678] | 0.00012 |
| removal cost | 69.238 [66.375, 72.102] | 0.00012 |
| restoration benefit | 62.988 [58.996, 66.981] | 0.00012 |
| specificity over random | 62.614 [58.622, 66.606] | 0.00012 |

## Additional fixed-mean diagnostic

This follow-up was specified after the pilot and after six models had their original subspace analyses computed. Its three contrasts per endpoint form separate exploratory correction families. It does not replace the original four-plus-four subspace families.

| Intervention | Heading error (degrees) | Goal reached (%) |
| --- | ---: | ---: |
| Restore only fixed mean | 75.215 [73.567, 76.863] | 6.592 [5.561, 7.622] |
| Remove variation, preserve fixed mean | 71.042 [68.772, 73.311] | 8.545 [7.115, 9.975] |

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| mean only benefit | 0.211 [-0.242, 0.664] | 0.339 |
| full plane over mean only | 65.168 [63.396, 66.941] | 9.2e-05 |
| mean preserving removal cost | 62.816 [60.534, 65.099] | 9.2e-05 |

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| mean only benefit | -0.195 [-1.681, 1.290] | 0.839 |
| full plane over mean only | 63.184 [59.341, 67.026] | 9.2e-05 |
| mean preserving removal cost | 68.848 [66.528, 71.167] | 9.2e-05 |

## Same-state Jacobian measurements

The state and recurrent weights are fixed while the input changes. Phase gain is a derivative in the decoded-angle coordinate, not an eigenvalue. These median local measurements do not certify global hidden-state topology; the branch audit is reported separately.

| Input at cue-free states | One-step phase gain | Spectral radius |
| --- | ---: | ---: |
| Input off | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] |
| Ordinary code | 0.993 [0.988, 0.997] | 0.938 [0.934, 0.943] |
| Predictive code | 0.813 [0.802, 0.823] | 0.899 [0.895, 0.903] |

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| phase gain native minus ordinary | -0.180 [-0.191, -0.169] | 9.2e-05 |
| phase gain native minus blank | -0.187 [-0.198, -0.176] | 9.2e-05 |
| spectral radius native minus ordinary | -0.039 [-0.043, -0.035] | 9.2e-05 |

## Separate raw-input architecture cohort

Eight fresh seed triplets, 24 models. Removing the nonlinear encoder also reduces parameter count; this is architecture/capacity sensitivity at a fixed training budget. The three tests have their own correction family. [Tracking figure and group means](raw_sensitivity_evaluation/README.md); [memory and anchoring diagnostics](raw_memory_evaluation/README.md).

| Contrast | Mean difference [95% paired interval] | Holm p |
| --- | ---: | ---: |
| PGC − Grid: native heading error (degrees) | -7.694 [-8.570, -6.818] | 0.023 |
| Ordinary replacement − native PGC: error (degrees) | 9.735 [8.534, 10.937] | 0.023 |
| PGC − Grid: error at new places (degrees) | -7.585 [-8.488, -6.681] | 0.023 |

## Independent geometry confirmation

Sixteen fresh Grid/PGC pairs, 32 models. The single prespecified endpoint is log10 of estimated peak-to-median angular sensitivity of the full initialization-to-memory map. This is not a test of global ring topology or transverse stability.

| Contrast | Mean difference [95% paired interval] | Exact two-sided p |
| --- | ---: | ---: |
| PGC − Grid log10 sharpness | -0.448 [-1.013, 0.117] | 0.110138 |

PGC has lower estimated sharpness in 10/16 pairs. The primary test does not confirm a reliable population difference. All models are retained. [All-seed figure and controls](geometry_confirmation/README.md).

## Source data

[Primary summary](confirmation_summary.json) · [Subspace summary](subspace_confirmation/summary.json) · [Fixed-mean summary](tonic_direction_confirmation/summary.json) · [Jacobian summary](cue_spectra_confirmation/summary.json) · [Geometry audits](geometry_audit/) · [Independent geometry summary](geometry_confirmation/summary.json) · [Raw-input summary](raw_sensitivity_evaluation/summary.json) · [Full-state memory](confirmation_evaluation/memory_summary.json) · [Reduced phase model and memory pulses](mechanistic_confirmation/summary.json)
