"""Package every original model used in the task-linked study and both RL pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    files = []
    for condition in ("gc100", "pgc33"):
        for seed in range(1, 9):
            stem = ROOT / f"models/{condition}/{condition}_lstm_s{seed}_1000M"
            files.extend([stem.with_suffix(".pt"), stem.with_suffix(".json")])
        pilot = ROOT / "results/ring_cloud_artifacts/ring-20261105-rl-goal-memory-t4-01" / condition
        files.extend(pilot / name for name in ("final.pt", "original_config.json", "training.json"))
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Complete cohort required; missing {missing}")
    records = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                "sha256": digest(p)} for p in files]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    # Check the packaged bytes, not only the input filenames.
    with tarfile.open(args.out, "r:gz") as archive:
        assert set(archive.getnames()) == {r["path"] for r in records}
        for row in records:
            stream = archive.extractfile(row["path"])
            assert hashlib.sha256(stream.read()).hexdigest() == row["sha256"]
    result = {"archive": args.out.name, "archive_bytes": args.out.stat().st_size,
              "archive_sha256": digest(args.out), "original_models": 16,
              "additional_RL_pilot_models": 2, "files": records,
              "scope": "All original Grid/PGC 1B LSTMs used by this study, plus both completed goal-flash RL pilots. The designed ring adds no trained weights. Large replay arrays are regenerated from scripts."}
    manifest = ROOT / "docs/research/rl_map_checkpoint_archive.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
