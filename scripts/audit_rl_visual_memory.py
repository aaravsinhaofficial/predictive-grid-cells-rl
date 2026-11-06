"""Check the fixed pilot's local artifacts independently of its training process."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'gc100': 'ring-20261105-vision-grid-01',
    'pgc33': 'ring-20261105-vision-pgc-01',
}
GAPS = (4, 8, 15, 30, 60)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_policy(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def audit(condition, name):
    folder = ROOT / 'results/ring_cloud_artifacts' / name
    summary = json.loads((folder / 'summary.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    assert summary['completed'] and summary['training']['completed']
    assert summary['training']['extra_frames'] == 25_024_000
    assert summary['training']['warmup_exposure_frames'] == 2_560_000
    for path, expected in protocol['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in protocol['original_asset_sha256'].items():
        assert sha(ROOT / path) == expected, path
    original = torch.load(ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.pt',
                          map_location='cpu', weights_only=True)['policy']
    warmup = torch.load(folder / 'after_warmup.pt', map_location='cpu', weights_only=True)
    assert same_policy(original, warmup['policy'])
    assert warmup['extra_frames'] == 0
    previous = warmup['policy']
    checkpoints = []
    for stage, gap in enumerate(GAPS, 1):
        path = folder / f'stage{stage}_gap{gap}.pt'
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        assert ckpt['extra_frames'] == stage * 5_004_800
        assert ckpt['policy'].keys() == original.keys()
        assert all(ckpt['policy'][key].shape == original[key].shape for key in original)
        assert all(torch.isfinite(x).all() for x in ckpt['policy'].values())
        assert not same_policy(previous, ckpt['policy'])
        previous = ckpt['policy']
        checkpoints.append(dict(stage=stage, gap=gap, learning_frames=ckpt['extra_frames'],
                                sha256=sha(path), bytes=path.stat().st_size))
    final_path = folder / 'final.pt'
    final = torch.load(final_path, map_location='cpu', weights_only=True)
    assert same_policy(previous, final['policy'])
    assert final['completed'] and final['extra_frames'] == 25_024_000
    assert sha(final_path) == summary['final_checkpoint_sha256']
    assert not (folder / 'incomplete.json').exists()
    final_evaluation = json.loads((folder / 'final_evaluation.json').read_text())
    assert final_evaluation == json.loads((folder / 'stage5_evaluation.json').read_text())
    return dict(condition=condition, completed=True, learning_frames=25_024_000,
                warmup_exposure_frames=2_560_000, warmup_weights_exactly_unchanged=True,
                all_frozen_sources_and_original_assets_unchanged=True,
                all_stage_weights_finite_and_changed=True,
                final_weights_equal_fixed_last_stage=True,
                final_sha256=sha(final_path), checkpoints=checkpoints)


def main():
    torch.set_num_threads(2)
    report = {condition: audit(condition, name) for condition, name in NAMES.items()}
    out = ROOT / 'results/ring_visual_memory/cloud_audit/artifact_verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value['completed'] for key, value in report.items()}))


if __name__ == '__main__':
    main()
