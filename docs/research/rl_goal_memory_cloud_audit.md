# RL goal-memory pilot: cleanup and cumulative cost

Verified at **2026-11-06T04:13:40.251172+00:00**: **no study VM or boot disk remains**. The completed pilot was collected before its identity-checked deletion; unrelated resources were not modified.

| Conservative accounting | USD |
|---|---:|
| Previous study, retained in full | 43.7364 |
| This completed pilot, including boot disk | 0.4549 |
| **Cumulative envelope** | **44.1913** |
| Original authorization | 500.00 |
| Remaining under this envelope | 455.8087 |

This is **not an invoice**. The original $25 storage/transfer allowance remains included once. Both L4 stockout attempts created no VM and carry zero compute charge. The T4 worker completed both fixed-budget models successfully and was deleted. Retained cloud artifacts can incur future storage costs depending on retention duration.

Both final model files match their GCS object size and MD5 checksum. Their SHA256 values also match the cloud-generated completion summary, the local archive manifest, and the actual files inside the release archive. The complete archive SHA256 is `c0b6727bc6782602f97e56fdc467fe3ada2389758c50578275e39393e9f26313`.

[Machine-readable audit and per-model hashes](rl_goal_memory_cloud_audit.json) · [Live cumulative ledger](ring_cloud_budget.json) · [Pilot results](../../results/ring_rl_goal_memory/README.md) · [Previous immutable accounting snapshot](ring_cloud_final_cost_bound.md).
