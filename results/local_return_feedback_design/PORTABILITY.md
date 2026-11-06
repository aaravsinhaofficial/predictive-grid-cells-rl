# Cloud input check: failure and narrow repair

**The first cloud attempt stopped before model initialization, baseline evaluation or learning.** Three workers reached the same input check; the fourth was cancelled before reservation or creation. All three machines and boot disks have been deleted. The conservative cost bound is **$0.2468**. [Terminal record](cloud_audit/terminal_attempt_record.json) · [Cleanup](cloud_audit/cleanup_verification.json).

The workers received the correct sealed test bank: its exact file checksum, size and every tensor checksum passed. An additional check regenerated the bank from its private seed and demanded identical floating-point values. That assumption failed across the local Mac/PyTorch 2.10 and cloud Linux/PyTorch 2.9.1 environments.

| Regenerated values | Largest absolute difference |
|---|---:|
| Goal coordinates | 0.000000060 m |
| Motor Gaussian tape | 0.00000185 |
| Unused state Gaussian tape | 0.00000209 |

All four random-generator states were identical before and after generation, and the action/uniform-derived arrays matched exactly. The evidence points to differences in the numerical transformation of the same random draws. PyTorch documents that identical seeds do not guarantee identical results across releases and platforms. [Diagnostic](cloud_audit/bank_regeneration_failure_diagnostic.json) · [Generator comparison](cloud_audit/bank_regeneration_rng_comparison.json) · [PyTorch documentation](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

The reviewed repair is to **authenticate and use the saved values directly**. A separate loader retains exact file/tensor checks, schema, seed, geometry, counts and generator-source verification. The successful exact regeneration performed when the bank was created remains pinned as origin evidence. All existing frozen sources, protocols and this failed attempt remain preserved. The same sealed bank, four incoming policies, training settings and endpoint criterion are retained. No approximate match or replacement test bank is introduced.

Implementation and a new prospective protocol must precede a manually reviewed replacement launch. The first replacement worker will check the package on Linux before evaluating a policy; its completed baseline and execution health will be checked before the remaining arms launch. Baseline scores will not select which arms receive training. The original four-arm comparison and fixed endpoint remain intact.
