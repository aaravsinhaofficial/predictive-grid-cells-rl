"""Independently verify both fixed continuations and their common incoming states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from rl_visual_memory_extend import tree_equal

NAMES = {
    'noiseless': {'gc100': 'ring-20261105-vision-long-grid-01', 'pgc33': 'ring-20261105-vision-long-pgc-01'},
    'noise': {'gc100': 'ring-20261105-vision-noise-grid-01', 'pgc33': 'ring-20261105-vision-noise-pgc-01'},
}
PARENTS = {'gc100': 'ring-20261105-vision-grid-01', 'pgc33': 'ring-20261105-vision-pgc-01'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(arm, condition, incoming_only=False):
    folder = ROOT / 'results/ring_cloud_artifacts' / NAMES[arm][condition]
    parent_folder = ROOT / 'results/ring_cloud_artifacts' / PARENTS[condition]
    parent = torch.load(parent_folder / 'final.pt', map_location='cpu', weights_only=True)
    incoming = torch.load(folder / 'incoming.pt', map_location='cpu', weights_only=True)
    assert tree_equal(incoming['policy'], parent['policy'])
    assert tree_equal(incoming['optimizer'], parent['optimizer'])
    assert incoming['extension_frames'] == 0
    snapshot = incoming['runtime_snapshot']
    assert all(not value.any() for value in snapshot['recurrent_state'])
    assert not snapshot['goal'].any() and not snapshot['clock']['known'].any()
    assert not snapshot['clock']['blind'].any()
    assert not snapshot['previous_reward'].any() and (snapshot['previous_action'] == -1).all()
    protocol = json.loads((folder / 'protocol.json').read_text())
    for key in ('source_sha256', 'original_asset_sha256'):
        for path, expected in protocol[key].items():
            assert sha(ROOT / path) == expected, path
    for name, expected in protocol['parent_sha256'][condition].items():
        assert sha(parent_folder / name) == expected, name
    incoming_evaluation = json.loads((folder / 'baseline.json').read_text())
    parent_evaluation = json.loads((parent_folder / 'final_evaluation.json').read_text())
    for variant in incoming_evaluation['variants']:
        assert incoming_evaluation['variants'][variant]['rows'] == parent_evaluation['variants'][variant]['rows']
    report = dict(arm=arm, condition=condition, incoming_policy_and_optimizer_exact=True,
                  fresh_episode_boundary_verified=True, incoming_behavior_matches_parent=True,
                  original_sources_assets_and_parent_hashes_match=True)
    if incoming_only:
        return report
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['completed'] and summary['training']['completed']
    assert summary['training']['extension_frames'] == 50_048_000
    assert summary['training']['extra_frames'] == 75_072_000
    assert summary['training']['warmup_updates'] == 0
    previous = incoming['policy']
    blocks = []
    for block in range(1, 6):
        checkpoint_path = folder / f'block{block}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['extension_frames'] == block * 10_009_600
        assert checkpoint['extra_frames'] == 25_024_000 + block * 10_009_600
        assert checkpoint['policy'].keys() == parent['policy'].keys()
        assert all(torch.isfinite(value).all() for value in checkpoint['policy'].values())
        assert not tree_equal(previous, checkpoint['policy'])
        assert {float(value['step']) for value in checkpoint['optimizer']['state'].values()} == {2155. + 782 * block}
        assert 'runtime_snapshot' in checkpoint
        evaluation = json.loads((folder / f'block{block}_evaluation.json').read_text())
        assert evaluation['all_first_discoveries_identical']
        assert evaluation['episodes_per_variant'] == 64 and evaluation['max_blind_gap'] == 60
        if condition == 'gc100':
            for a, b in [('blind', 'blind_current_grid'), ('seen', 'seen_current_grid')]:
                assert evaluation['variants'][a]['rows'] == evaluation['variants'][b]['rows']
        blocks.append(dict(block=block, sha256=sha(checkpoint_path), extension_frames=checkpoint['extension_frames']))
        previous = checkpoint['policy']
    final = torch.load(folder / 'final.pt', map_location='cpu', weights_only=True)
    assert final['completed'] and tree_equal(previous, final['policy'])
    assert sha(folder / 'final.pt') == summary['final_sha256']
    assert not (folder / 'incomplete.json').exists()
    if arm == 'noise':
        assert final['runtime_snapshot']['recurrent_noise_sigma'] == .02
        assert summary['recurrent_noise']['replayed_updates'] == 3910
        assert summary['recurrent_noise']['max_rollout_replay_error'] < 3e-4
        noisy = json.loads((folder / 'final_noisy_evaluation.json').read_text())
        assert noisy['all_first_discoveries_identical']
        assert noisy['recurrent_noise']['matched_noise_across_branches']
        report['final_noisy_evaluation_verified'] = True
        report['max_rollout_replay_error'] = summary['recurrent_noise']['max_rollout_replay_error']
    report.update(completed=True, extra_frames=75_072_000, extension_frames=50_048_000,
                  final_sha256=sha(folder / 'final.pt'), blocks=blocks)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--incoming-only', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(2)
    reports = {arm: {condition: audit(arm, condition, args.incoming_only) for condition in pairs}
               for arm, pairs in NAMES.items()}
    out = ROOT / 'results/ring_visual_memory/continuation_cloud_audit'
    out.mkdir(parents=True, exist_ok=True)
    name = 'incoming_verification.json' if args.incoming_only else 'artifact_verification.json'
    (out / name).write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps({arm: list(pairs) for arm, pairs in reports.items()}))


if __name__ == '__main__':
    main()
