# Qualification and exploratory original-memory archive plan

Preparation only. The original-memory writers and their final audits must finish before an output archive is built. The failed paired training qualification and every exploratory result will be retained regardless of behavioral screens. No learning or cloud instance belongs to this study.

The frozen diagnostic protocol is `docs/research/goal_cue_memory_original_diagnostic_protocol.json`, SHA256 `2d9386fad85f9774d80ced37d7d6220d6bb75c682879684511a0da4d4bb04125`, committed before outcomes in `fbb34e60`. It explicitly preserves the parent training protocol and known qualification results. The eventual reproduction checkout must be a concrete later commit containing the final reports and archive tool; its numerical sources must still match this protocol exactly.

## Exact inclusion list

Start from the diagnostic protocol's complete `source_sha256` and `input_assets` maps, verifying each literal file hash. These supply the inherited source closure, original model/config assets, exact archived initialization/value head, parent protocols, both banks and their creation/authentication records. Include the diagnostic protocol itself. Do not regenerate bank values from seeds at load time.

| Sealed bank | Bytes | SHA256 |
|---|---:|---|
| `results/goal_cue_memory_design/banks/qualification.pt` | 682862 | `186a50f1e10c63e513d7ed6bb29432ea0ccdb9f6cd4752162e60a69fe94ef6fd` |
| `results/goal_cue_memory_design/banks/memory.pt` | 681833 | `3f111946c316219c58d411ee90cef8b25f8eb99dfc9f731cf63e0e20fda7f3ba` |

Add every regular file under these two explicit output roots:

- `results/goal_cue_memory/qualification/`
- `results/goal_cue_memory/original_diagnostic/`

This preserves both populations, all qualification and diagnostic incoming/after checkpoints, configurations, provenance, integrity records, context plans, all variant inputs/results/raw tensors, reports and audits. The diagnostic root is `{gc100,pgc33}` with `audits/{condition}.json`; each model has all 16 contexts and all 11 variants. Preserve any failed/incomplete-attempt evidence rather than replacing it. A valid failed scientific outcome is an archived result, not a packaging error.

Add these small explicit records when finalized: the behavioral/training/diagnostic protocols; applicable initialization, bank, implementation and independent review records; `results/goal_cue_memory_design/original_diagnostic_execution.json`; the final plain-language report/figures; the prospective interpretation note; the archive tool, its tests and the reproduction guide. Determine this final small-file allowlist before building. Avoid broad inclusion of unrelated results, credentials, temporary directories or prior cloud output trees. Large `.pt` and `.pt.gz` files remain release assets, outside Git.

## Completion and integrity checks

Require complete, passing artifact audits for each population in both studies, with no pending records, correct phase/condition/protocol/source hashes and all prescribed trial denominators. The failed qualification gate is expected and must remain failed. No success count or neural-candidate gate is a packaging requirement. Check every artifact path/hash named by these audits and integrity records. Require all context plans and each prescribed variant's input/result plus its recorded raw or explicit failed-plan evidence. An omitted member cannot count as a complete archive.

Require both diagnostic `incoming.pt`/`after.pt` snapshot checks and no-optimization records. Preserve exact execution logs and runtime metadata, without fabricating GPU, provider-cleanup or training records for this local CPU evaluation. The current qualification tree contains 153 files and 329312707 bytes; these are planning measurements, not a final inventory.

## Small deterministic packager

A generic manifest-and-tar tool is sufficient; it need not reproduce the previous large study-specific packagers. Resolve only the allowlisted files, reject symlinks, absolute archive paths and `..`, deduplicate by repository-relative path, and sort paths lexicographically. Record every file's exact byte count and SHA256, its source/protocol pin, and the final reproduction commit.

Partition that sorted sequence deterministically into parts with at most 1750000000 payload bytes each. This leaves room below a strict **1800000000-byte compressed asset limit** for tar/gzip framing and manifests; assert the final bound after writing. If a single member exceeds the bound, fail explicitly rather than omit or alter it. Use normalized tar metadata, regular files only, fixed gzip time zero and stable JSON serialization. Verify each streamed member against its recorded hash, then reopen the completed tar and verify its complete member set and bytes. Store a part index with archive sizes/hashes and the shared full inventory. No production archive is built while a writer remains active.

## Isolated reproduction after final assets exist

Create a detached checkout at the recorded reproduction commit. Verify archive hashes before safe extraction; reject duplicate conflicting members or traversal. Verify every extracted file against the shared inventory and both exact bank manifests. Re-run both qualification audits and both original-diagnostic audits into fresh output paths, then regenerate reporting from saved results into a fresh directory. This reuses the archived actual evaluations; it does not require training or new policy rollouts.

If a fresh policy-evaluation reproduction is separately requested, run the original diagnostic CLI for **both** populations into new empty directories with the same frozen diagnostic protocol, then audit them. Keep that more expensive step separate from byte validation and saved-data audit replay. Compare all deterministic tensors/counts against the archived run, reporting any platform-dependent tolerance explicitly; timestamps and absolute checkout prefixes are the only routine metadata exclusions. Record commands, exit statuses, source/asset hashes and comparison results only after those checks actually run.
