# Final cloud cleanup and conservative cost bound

Verified at **2026-11-06T02:35:51.304266+00:00**: **zero study VMs and zero study boot disks remain** in Compute Engine. The provider inventory was queried independently after the identity-checked collectors closed all study reservations. This audit made no changes to unrelated resources.

All **32 geometry-confirmation** checkpoints (16 Grid/PGC pairs, seeds 5001–5016) and **24 raw-architecture** checkpoints (8 seeds ×3 conditions, seeds 3001–3008) were collected. Every final checkpoint reports 6,000 completed updates and normal step completion; all eight workers report successful completion. Exact checkpoint hashes are retained in the [audit record](ring_cloud_final_cost_bound.json).

| Conservative accounting component | USD |
|---|---:|
| Closed VM/boot-disk upper bounds, all study runs | 18.7364 |
| Storage and transfer allowance | 25.00 |
| **Total conservative envelope** | **43.7364** |
| User-authorized ceiling | 500.00 |

**This is not an invoice.** The ledger uses a deliberately conservative $1.25 per VM-hour, including boot-disk costs, with a 15-minute reservation allowance per launch. Closed-run bounds use verified absence and the reserved provider-enforced lifetime. The separate $25 allowance covers ancillary storage/transfer. Source archives and model artifacts remain in the study's isolated storage prefix; any future storage after this snapshot depends on retention duration.

Evidence: [final audit record](ring_cloud_final_cost_bound.json), [live budget ledger](ring_cloud_budget.json), [pricing and lifecycle controls](ring_cloud_plan.md), and [read-only audit implementation](../../scripts/finalize_ring_cloud_audit.py). The complete ledger snapshot is recorded by SHA256 `65510e3f6a14fceba01366dbcc708074a3c8d2f3250f785846e47e48114f8400` and retained locally at `results/ring_cloud_artifacts/final_cleanup/ledger_snapshot.json`.
