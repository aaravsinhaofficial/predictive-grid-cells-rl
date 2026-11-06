"""Compare all fixed final models using the frozen CUDA noise evaluator and verified shared draws."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from code.common import build_policy
import rl_visual_memory_noise as study
from rl_goal_memory import atomic_json, state_digest

MODELS = {
    'noiseless_gc100': 'ring-20261105-vision-long-grid-01',
    'noiseless_pgc33': 'ring-20261105-vision-long-pgc-01',
    'noise_gc100': 'ring-20261105-vision-noise-grid-01',
    'noise_pgc33': 'ring-20261105-vision-noise-pgc-01',
}


class AuditedNoise(study.StateNoise):
    """Call the unchanged sampler; independently reproduce and hash its raw draws."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_hasher = hashlib.sha256()
        self.draws = 0
        self.shapes = set()

    def draw(self, state, mask, groups=1):
        verifier = torch.Generator(device=self.device)
        verifier.set_state(self.generator.get_state())
        shape = (2, state[0].shape[0] // groups, state[0].shape[1])
        raw = torch.randn(*shape, generator=verifier, device=state[0].device, dtype=state[0].dtype)
        result = super().draw(state, mask, groups)
        expected = raw.repeat(1, groups, 1) * self.sigma * mask[None, :, None]
        assert torch.equal(result[0], expected[0]) and torch.equal(result[1], expected[1])
        assert torch.equal(self.generator.get_state(), verifier.get_state())
        self.raw_hasher.update(raw.detach().cpu().contiguous().numpy().tobytes())
        self.draws += 1; self.shapes.add((shape, str(raw.dtype)))
        return result

    def audit(self):
        return dict(raw_draw_sha256=self.raw_hasher.hexdigest(), draws=self.draws,
                    shapes=[dict(shape=list(shape), dtype=dtype) for shape, dtype in sorted(self.shapes)],
                    actual_sampled_arrays_match_independent_reproduction=True)


def protocol():
    frozen = json.loads((ROOT / 'docs/research/rl_visual_memory_noise_protocol.json').read_text())
    for path, expected in frozen['source_sha256'].items():
        assert study.pilot.sha(ROOT / path) == expected, path
    extra = ('scripts/eval_visual_memory_matched_noise.py', 'tests/test_visual_memory_matched_noise_eval.py')
    return dict(experiment='Common-backend noisy evaluation of all four fixed continuation endpoints',
                models=MODELS, required_learning_frames_beyond_1B=75_072_000, device='cuda',
                evaluator='Unchanged rl_visual_memory_noise.evaluate_noisy', noise_seed=128201,
                sigma_h=.02, sigma_c=.02, evaluation_seed=98201, episodes=64, n_envs=32,
                variants=list(study.pilot.VARIANTS), model_selection='All fixed final endpoints; no checkpoint selection',
                sampling_audit='Call the original sampler; reproduce each raw draw with a cloned private generator, require exact array and RNG equality, and hash unmasked raw arrays across all four models. Masks follow each model reward history.',
                behavioral_control='For both noise-trained models, require every raw episode row to equal the existing CUDA final_noisy_evaluation on the same checkpoint.',
                frozen_noise_protocol_sha256=study.pilot.sha(ROOT / 'docs/research/rl_visual_memory_noise_protocol.json'),
                source_sha256={**frozen['source_sha256'], **{path: study.pilot.sha(ROOT / path) for path in extra}})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/ring_visual_memory/matched_noisy_endpoints')
    ap.add_argument('--max-seconds', type=int, default=600)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'docs/research/visual_memory_matched_noise_eval_protocol.json')
    ap.add_argument('--freeze-only', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(2)
    design = protocol()
    if args.freeze_only:
        if args.protocol.exists(): raise FileExistsError('Preserve frozen matched-noise evaluation protocol')
        design['frozen_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(args.protocol, design); print(study.pilot.sha(args.protocol)); return
    frozen = json.loads(args.protocol.read_text()); frozen.pop('frozen_at')
    assert design == frozen
    assert torch.cuda.is_available()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'protocol.json').exists(): raise FileExistsError('Preserve previous evaluation output')
    atomic_json(args.out / 'protocol.json', design)
    deadline = time.monotonic() + args.max_seconds
    reports = {}
    for name, worker in MODELS.items():
        folder = ROOT / 'results/ring_cloud_artifacts' / worker
        checkpoint_path = folder / 'final.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['completed'] and checkpoint['extra_frames'] == 75_072_000
        cfg = checkpoint['config']
        net = build_policy(cfg, 'cuda').eval(); net.load_state_dict(checkpoint['policy'], strict=True)
        initial_hash = state_digest(net); instances = []
        class TracedNoise(AuditedNoise):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs); instances.append(self)
        with patch.object(study, 'StateNoise', TracedNoise):
            result = study.evaluate_noisy(net, cfg, device='cuda', deadline=deadline)
        assert len(instances) == 1 and state_digest(net) == initial_hash
        trace = instances[0].audit()
        assert trace['draws'] == 2700
        assert trace['shapes'] == [dict(shape=[2, 32, 256], dtype='torch.float32')]
        if name.startswith('noise_'):
            reference = json.loads((folder / 'final_noisy_evaluation.json').read_text())
            assert all(result['variants'][key]['rows'] == reference['variants'][key]['rows'] for key in study.pilot.VARIANTS)
            result['all_rows_match_existing_cuda_evaluation'] = True
        result.update(model=name, checkpoint_sha256=study.pilot.sha(checkpoint_path), policy_unchanged=True,
                      torch_version=str(torch.__version__), cuda_version=str(torch.version.cuda), noise_draw_audit=trace)
        atomic_json(args.out / f'{name}.json', result)
        reports[name] = dict(checkpoint_sha256=result['checkpoint_sha256'], noise_draw_audit=trace,
                             scores={key: value['mean_score'] for key, value in result['variants'].items()},
                             contrasts=result['descriptive_contrasts'])
        atomic_json(args.out / 'progress.json', reports)
        del net
    hashes = {row['noise_draw_audit']['raw_draw_sha256'] for row in reports.values()}
    assert len(hashes) == 1
    atomic_json(args.out / 'summary.json', dict(completed=True, all_models_received_identical_raw_noise=True,
                common_raw_noise_sha256=next(iter(hashes)), models=reports))
    print(json.dumps({name: report['scores'] for name, report in reports.items()}), flush=True)


if __name__ == '__main__':
    main()
