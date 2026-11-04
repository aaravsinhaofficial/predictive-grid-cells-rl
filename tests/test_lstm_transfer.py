import hashlib
import json
import sys

import numpy as np
import pytest
import torch

from code.common import build_policy
from scripts import lstm_transfer as transfer


def test_spectrum_is_centered():
    values = transfer.spectrum([[101, 5], [102, 5], [103, 5]])
    assert values['pc3'] == values['participation_ratio'] == 1
    assert values['d90'] == values['total_variance'] == 1
    assert transfer.spectrum(np.ones((3, 4)))['pc3'] is None


def test_seed_level_association():
    def record(s, goal, pc3):
        return dict(seed=s, arena='large', models={
            'gc100': dict(navigation=dict(mean_goals=5), matched_hidden=dict(pc3=.4, pc10=.7, d90=5),
                          current_code=dict(d90=9), full_input=dict(d90=10)),
            'pgc33': dict(navigation=dict(mean_goals=goal), matched_hidden=dict(pc3=pc3, pc10=pc3+.3, d90=4),
                          current_code=dict(d90=11), full_input=dict(d90=13))})
    out = transfer.summarize([record(1, 4, .3), record(2, 8, .6), record(3, 5, .4)])
    assert out['large']['spearman_delta_pc3_vs_goals'] == 1
    assert out['large']['spearman_delta_pc10_vs_goals'] == 1


def test_paired_cli(tmp_path, monkeypatch):
    # Local manifest of throwaway random networks; never use real source weights.
    models = tmp_path/'models'
    manifest = []
    for c in transfer.CONDITIONS:
        stem = f'{c}_lstm_s1_1000M'
        path = models/c/(stem+'.pt')
        path.parent.mkdir(parents=True)
        cfg = json.loads((transfer.ROOT/'models'/c/(stem+'.json')).read_text())
        cfg.update(episode_len=6, nh=8, n_envs=2, rollout=2)
        path.with_suffix('.json').write_text(json.dumps(cfg))
        torch.manual_seed(101)
        torch.save(dict(frames=transfer.FRAMES, policy=build_policy(cfg,'cpu').state_dict()), path)
        manifest.append(dict(seed=1, condition=c, checkpoint=str(path.relative_to(tmp_path)),
                             checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    ref = tmp_path/'results/lstm_pca_1B/summary.json'
    ref.parent.mkdir(parents=True)
    ref.write_text(json.dumps(dict(models=manifest)))
    monkeypatch.setattr(transfer, 'ROOT', tmp_path)
    monkeypatch.setattr(transfer, 'ARENAS', {'reference':1.25, 'large':2.0})
    monkeypatch.setattr(transfer, 'ADAPT_FRAMES', 16)
    out = tmp_path/'out'
    def run(mode):
        monkeypatch.setattr(sys, 'argv', ['transfer', mode, '--models', str(models),
                            '--out', str(out), '--seeds','1','--slots','2','--episodes','4','--stride','2'])
        transfer.main()

    run('plan')
    assert not out.exists()
    run('frozen')
    run('finetune')
    for mode in ('frozen', 'finetune'):
        assert (out/mode/'pc10_vs_navigation.png').stat().st_size > 0
        row = json.loads((out/mode/'large_s1.json').read_text())
        for c, src in zip(transfer.CONDITIONS, manifest):
            metrics = row['models'][c]
            assert metrics['navigation']['episodes'] == 4
            assert metrics['native_hidden']['samples'] == 6
            assert metrics['matched_hidden']['samples'] == 12
            assert metrics['current_code']['width'] == 512
            assert metrics['full_input']['width'] == 1287
            hashes = row['checkpoints'][c]
            assert hashes['parent_sha256'] == src['checkpoint_sha256']
            assert (hashes['parent_sha256'] == hashes['evaluated_sha256']) == (mode == 'frozen')
    assert 'pgc_minus_grid_adaptation_gain' in row
    for c in transfer.CONDITIONS:
        cfg = json.loads((out/'finetune'/f'large_{c}_s1'/'config.json').read_text())
        assert cfg['code_seed'] == 1 and cfg['seed'] == 9001 and cfg['paired_layouts']
    assert not (out/'finetune'/'reference_s1.json').exists()
    with pytest.raises(FileExistsError):
        run('frozen')
    path = tmp_path/manifest[0]['checkpoint']
    path.write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='hash'):
        run('frozen')
    path.unlink()
    with pytest.raises(SystemExit) as exc:
        run('frozen')
    assert exc.value.code == 2
