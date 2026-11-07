# Exploratory counterbalanced readout: incomplete equivalence

This separate supplement preserves **all 1,536 families and 196,608 readout cases**, including the **12 failed input-equivalence checks and six downstream cell-state discrepancies**. The original scientific run remains incomplete and its equivalence criterion remains rejected. The completed descriptive report analyzes the entire saved cohort; it does not turn the failed criterion into a pass or qualify a ring mechanism.

The single 237,339,857-byte archive contains 1,757 files (256,197,640 uncompressed bytes): raw NPZ/PT data, diagnostics, both failed execution attempts and logs, the descriptive figures/audits, and exact source/protocol files. It is an **overlay**, requiring the separately verified primary native-turn128 archives at `aa7ee742bd8aeb2e664471226c286fa0ed1495c2`. Its manifest binds that primary manifest and index exactly. Shared checkpoints, input banks and original paired results are reused.

Every supplement member was independently verified during extraction into the restored clean primary checkout. The descriptive reporter then recomputed all saved-array arithmetic with original-workspace reads blocked. All 11 compared outputs reproduced: eight byte-identically (including all numerical JSON, CSV, PNG and README files), and three PDFs differing only in explicit creation/modification dates. Scientific status and all 12/6 failures were retained. No environment, policy rollout, recurrent transition or optimization was executed.

[Reproduction proof](reproduction_proof.json) records the exact guarded command, read set, member/source checks and output comparisons. Primary archives remain unchanged.

To reproduce, first reconstruct the primary archive checkout using its frozen instructions. Verify this overlay with the existing safe archive verifier before extracting its regular files into that checkout:

```sh
cd /path/to/the/verified/primary/checkout
export PYTHONPATH="$PWD/scripts:$PWD"
python - <<'PYCODE'
from pathlib import Path
import package_nonlinear_ring_outputs as package
package.MANIFEST_NAME = 'results/native_turn_counterbalanced_release/manifest.json'
print(package.verify_release(Path('/path/to/overlay/archives')))
PYCODE
```

Preserve the archived `results/native_turn_counterbalanced_readout/report` directory outside the disposable checkout, including its independent reviews. Then run only the saved-data reporter:

```sh
python scripts/report_native_turn_counterbalanced_readout.py --report
```

Compare its complete scientific JSON, CSV, PNG and README bytes with the preserved originals. Only PDF `/CreationDate` and `/ModDate` metadata may differ. Archived independent reviews describe the archived outputs; retain them separately from newly generated files. Both execution attempts and the scientific `incomplete.json` remain evidence, not missing files to repair.
