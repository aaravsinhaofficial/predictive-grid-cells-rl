# Standalone report reproduction

The release archives were extracted into a fresh detached checkout of `2d2a63ca398e045523f5ab8027cdb9a95eee45fc`. Every one of the 699 archived files passed streamed and extracted-byte SHA256 checks; all 494 source files and 45 other tracked files matched the pinned commit. No training, model evaluation or environment simulation was repeated.

Only these saved-data reports were regenerated:

```sh
/opt/anaconda3/bin/python scripts/report_native_policy_distillation.py
/opt/anaconda3/bin/python scripts/lstm_distillation_memory_report.py
```

The actual verification ran these commands through the archived proof's file-access guard, which rejected original-workspace reads and confirmed every checkout input came from the extracted archive. Both report runs passed. The two numerical JSON files, two Markdown reports and six PNG figures reproduced byte for byte. The six PDFs differed only in explicit creation/modification date metadata. No other PDF bytes were normalized. Original archive inputs remained unchanged.

[Full reproduction proof](reproduction_proof.json) records member/source checks, exact guarded commands, all compared hashes, report read sets and the verification procedure. [Frozen reproduction instructions](REPRODUCE.md) remain unchanged. This is an archive and reporting completeness check, not an independent scientific replication.
