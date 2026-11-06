"""Repeat the unchanged every-black-update reset at all four fixed75M endpoints."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
from code.common import build_policy
import rl_visual_memory as parent
from lstm_endogenous_every_blind_reset import EveryBlindUpdateClock
from lstm_endogenous_curve_recovery import sha, save

MODELS = {
    'v2_gc100': 'ring-20261105-vision-long-grid-01',
    'v2_pgc33': 'ring-20261105-vision-long-pgc-01',
    'v3_gc100': 'ring-20261105-vision-noise-grid-01',
    'v3_pgc33': 'ring-20261105-vision-noise-pgc-01'}
PROTOCOLS = {'v2': 'docs/research/rl_visual_memory_extension_protocol.json',
             'v3': 'docs/research/rl_visual_memory_noise_protocol.json'}


def protocol():
    files = ('scripts/lstm_final75_every_blind_reset.py',
        'scripts/lstm_endogenous_every_blind_reset.py', 'scripts/rl_visual_memory.py',
        'scripts/rl_goal_memory.py', 'scripts/lstm_code_swap.py',
        'code/nets.py', 'code/env.py', 'code/common.py', 'code/codes.py')
    return dict(version='final75M_every_blind_update_reset_v1', models=MODELS,
        selection='All four fixed75.072M final checkpoints; no favorable stage selection',
        timing='Followup frozen after noiseless final onset-reset scores, before any75M every-step reset outcomes',
        intervention='Unchanged EveryBlindUpdateClock and frozen parent evaluator. Only h/c is reset before every scheduled black-input update in reset branches; matched seen-reset branches follow the same clock.',
        retained='Actual previous action/reward, current spatial input, external saved goal, physics, rewards, original actor and weights. This tests recurrent carryover, not all forms of history.',
        seed=98201, episodes=64, n_envs=32, max_gap=60, training_noise_during_evaluation=False,
        validation='Every nonreset raw episode row must exactly equal the corresponding fixed final clean CUDA evaluation. First discoveries remain paired. Saved final checkpoint must match frozen training provenance and budgets.',
        training_protocol_sha256={a:sha(ROOT/p) for a,p in PROTOCOLS.items()},
        source_sha256={p:sha(ROOT/p) for p in files})


def validate(data, key):
    arm, condition = key.split('_', 1)
    design = json.loads((ROOT/PROTOCOLS[arm]).read_text()); design.pop('frozen_at', None)
    assert data.get('completed') is True
    assert data.get('original_frames') == 1_000_000_000
    assert data.get('extra_frames') == 75_072_000 and data.get('extension_frames') == 50_048_000
    assert data['config']['cond'] == condition and data['config']['seed'] == 1
    assert data['provenance'] == design


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=ROOT/'docs/research/final75_every_blind_reset_protocol.json')
    ap.add_argument('--freeze-only', action='store_true')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max-seconds', type=int, default=240)
    args = ap.parse_args(); torch.set_num_threads(2); design = protocol()
    if args.freeze_only:
        assert not args.protocol.exists()
        save(args.protocol, dict(design, frozen_at=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(sha(args.protocol)); return
    frozen = json.loads(args.protocol.read_text()); frozen.pop('frozen_at')
    assert frozen == design
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out/'summary.json').exists()
    save(args.out/'protocol.json', design)
    deadline = time.monotonic()+args.max_seconds; reports = {}
    for key, worker in MODELS.items():
        parent.require_time(deadline)
        folder = ROOT/'results/ring_cloud_artifacts'/worker
        checkpoint = folder/'final.pt'; reference_path = folder/'final_evaluation.json'
        data = torch.load(checkpoint, map_location='cpu', weights_only=True); validate(data, key)
        cfg = data['config']; net = build_policy(cfg, args.device).eval().requires_grad_(False)
        net.load_state_dict(data['policy'], strict=True)
        assert not net.lstm._forward_hooks
        reference = json.loads(reference_path.read_text())
        assert (reference['seed'],reference['episodes_per_variant'],reference['max_blind_gap']) == (98201,64,60)
        with patch.object(parent, 'VisualClock', EveryBlindUpdateClock):
            result = parent.evaluate(net,cfg,device=args.device,seed=98201,episodes=64,
                                     n_envs=32,max_gap=60,deadline=deadline)
        identity = {v:result['variants'][v]['rows']==reference['variants'][v]['rows']
                    for v in parent.VARIANTS if not v.endswith('_reset')}
        result.update(model=key,checkpoint_sha256=sha(checkpoint),reference_sha256=sha(reference_path),
                      exact_nonreset_row_identity=identity,device=args.device,torch_version=torch.__version__,
                      original_onset_reset_contrasts=reference['descriptive_contrasts'])
        save(args.out/f'{key}.json',result)
        assert all(identity.values()), 'Retained mismatch; isolate backend/input discrepancy before interpreting'
        reports[key]=dict(scores={v:r['mean_score'] for v,r in result['variants'].items()},
            contrasts=result['descriptive_contrasts'],exact_nonreset_row_identity=identity,
            state_reset_counts=result['state_reset_counts'])
        print(json.dumps(dict(model=key,**reports[key])),flush=True)
    save(args.out/'summary.json',dict(completed=True,models=reports,source_sha256=sha(__file__)))


if __name__ == '__main__': main()
