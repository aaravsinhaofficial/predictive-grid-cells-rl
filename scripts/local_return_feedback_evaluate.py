"""Sealed-bank feedback acquisition evaluation; no study execution on import.

Full float32 policy inputs are archived losslessly, one physical context at a
time. Physical references are a separate, once-only evaluation, never training.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
from code.nets import PolicyNet
from rl_goal_memory import atomic_json, state_digest
from rl_visual_memory_extend import tree_equal
import local_return_feedback as feedback
from local_goal_return_training_evaluate import context_plan

BEHAVIOR = ROOT/'docs/research/local_return_feedback_behavior_protocol.json'
BANK_SEED, CONTEXTS, HEADINGS, SUCCESS_THRESHOLD = 491801, 16, 16, 192
sha = feedback.sha


def global_rng_snapshot():
    return dict(cpu=torch.random.get_rng_state().clone(),
        cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        mps=torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None)


def restore_global_rng(saved):
    assert set(saved) == {'cpu', 'cuda', 'mps'}
    torch.random.set_rng_state(saved['cpu'].cpu())
    if saved['cuda']:
        assert torch.cuda.is_available() and len(saved['cuda']) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all(saved['cuda'])
    if saved['mps'] is not None:
        assert torch.backends.mps.is_available()
        torch.mps.set_rng_state(saved['mps'].cpu())


@contextmanager
def preserved_rng():
    before = global_rng_snapshot()
    try: yield
    finally:
        restore_global_rng(before)
        assert tree_equal(before, global_rng_snapshot())


def tensor_schema(value, prefix=''):
    """Exact uncompressed tensor identity; no dtype conversion or quantization."""
    result = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=feedback.task.tensor_hash(value), bytes=value.numel()*value.element_size())
    elif isinstance(value, dict):
        for key, child in value.items(): result.update(tensor_schema(child, f'{prefix}/{key}'))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value): result.update(tensor_schema(child, f'{prefix}/{index}'))
    return result


def save_raw(path, value):
    path = Path(path); assert not path.exists()
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0, compresslevel=1) as zipped:
            # Legacy serialization streams sequentially through gzip and preserves tensors exactly.
            torch.save(value, zipped, _use_new_zipfile_serialization=False)
    temporary.replace(path)
    return dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size,
        compression='gzip_level1_lossless', tensor_schema=tensor_schema(value))


def load_raw(path, record=None):
    path = Path(path)
    if record:
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
        assert record['compression'] == 'gzip_level1_lossless'
    with gzip.open(path, 'rb') as stream:
        value = torch.load(stream, map_location='cpu', weights_only=True)
    if record: assert tensor_schema(value) == record['tensor_schema']
    return value


def load_bank(manifest=BEHAVIOR):
    manifest = Path(manifest); design = json.loads(manifest.read_text())
    assert design['version'] == 'feedback_acquisition_behavior_v2'
    assert design['seed'] == BANK_SEED and design['stage'] == asdict(feedback.STAGE)
    assert design['n_contexts'] == CONTEXTS and design['headings'] == HEADINGS
    assert design['success_threshold'] == SUCCESS_THRESHOLD
    for name, cls in [('generator', feedback.TrialStreams), ('stage', feedback.Stage)]:
        assert design[name+'_class_sha256'] == hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
    record = design['bank']; path = ROOT/record['path']
    assert not Path(record['path']).is_absolute() and '..' not in Path(record['path']).parts
    assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    plan = torch.load(path, map_location='cpu', weights_only=True)
    assert tensor_schema(plan) == record['tensor_schema']
    assert plan['stage'] == asdict(feedback.STAGE) and plan['n_contexts'] == CONTEXTS and plan['headings'] == HEADINGS
    assert plan['batch'] == 0
    cfg = json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    regenerated = feedback.TrialStreams(BANK_SEED).draw(CONTEXTS, feedback.STAGE, cfg, headings=HEADINGS)
    assert tree_equal(plan, regenerated), 'Sealed bank must match its declared private seed exactly'
    return plan, record, design


def summarize(rows, n_contexts, headings):
    expected = {(c, h) for c in range(n_contexts) for h in range(headings)}
    assert len(rows) == n_contexts*headings
    assert {(r['context'], r['heading_index']) for r in rows} == expected
    assert all(r['variant'] == 'feedback_native' and isinstance(r['hit'], bool) for r in rows)
    valid = [r for r in rows if not r.get('invalid', False)]
    assert all(not r['hit'] for r in rows if r.get('invalid', False))
    mean = lambda key: sum(r.get(key, 0.) for r in rows)/len(rows)
    return dict(trials=len(rows), valid_trials=len(valid), successes=sum(r['hit'] for r in rows), invalid_trials=len(rows)-len(valid),
        descriptive_mean_convention='Means use all prescribed rows; invalid trials contribute zero. Invalid/valid counts are explicit; invalid trials remain failures.',
        success_fraction=sum(r['hit'] for r in rows)/len(rows), mean_path_m=mean('free_path_m'),
        mean_progress_m=mean('progress_m'), mean_free_entropy=mean('mean_free_entropy'),
        mean_initial_free_entropy=mean('first_free_entropy'),
        action_counts=[sum(r.get('action_counts', [0]*6)[i] for r in rows) for i in range(6)],
        free_updates=sum(r.get('free_steps', 0) for r in rows))


@torch.inference_mode()
def evaluate_plan(net, cfg, plan, reference, out, *, device='cpu', controller='policy', actor_mode=None):
    """Small synthetic fixtures may call this; production first validates the sealed bank."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); assert not any(out.iterdir())
    before = state_digest(net); was_training = net.training
    assert actor_mode is None or actor_mode in feedback.ACTOR_MODES
    rows = []; contexts = []; started = time.monotonic()
    with preserved_rng():
        source = feedback.make_source(cfg, device)
        net.eval()
        try:
            for index in range(plan['n_contexts']):
                chunk = context_plan(plan, index); folder = out/f'context{index}'; folder.mkdir()
                input_ref = dict(reference, context_index=index, context_ids=chunk['context'].tolist())
                atomic_json(folder/'input.json', input_ref)
                try:
                    roll, local, checks, first = feedback.collect(net, cfg, source, chunk,
                        device=device, controller=controller, retain_inputs=controller == 'policy')
                    raw = {key: torch.stack(value).cpu() for key, value in roll.items() if isinstance(value, list) and value}
                    raw.update(goal=roll['goal'].cpu(), bootstrap=roll['bootstrap'].cpu(),
                        goal_world=chunk['goal'].clone(), first_free=first, plan_reference=input_ref)
                    record = save_raw(folder/'raw.pt.gz', raw)
                    result = dict(context=index, valid=True, rows=local, audit=checks, raw=record)
                except Exception as error:
                    failed = save_raw(folder/'failed_plan.pt.gz', chunk)
                    local = [dict(variant='feedback_native', context=index, heading_index=h,
                        hit=False, invalid=True) for h in range(plan['headings'])]
                    result = dict(context=index, valid=False, rows=local, failure_category='collection',
                        error_type=type(error).__name__, error=str(error), failed_plan=failed)
                rows.extend(local); contexts.append(result); atomic_json(folder/'result.json', result)
                print(json.dumps(dict(context=index, controller=controller, valid=result['valid'],
                    successes=sum(r['hit'] for r in local))), flush=True)
        finally: net.train(was_training)
    assert state_digest(net) == before
    totals = summarize(rows, plan['n_contexts'], plan['headings'])
    physical = all(r['valid'] and r['audit']['forced_hits'] == 0 and r['audit']['setup_learning_reward'] == 0
                   and r['audit']['fresh_feedback_on_every_active_input'] and r['audit']['state_noise_sigma'] == 0
                   and r['audit']['midtrial_resets'] == 0 and not r['audit']['unique_endpoint_cue'] for r in contexts)
    report = dict(version='feedback_acquisition_evaluation_v2', complete=True, condition=cfg['cond'],
        controller=controller, actor_mode=actor_mode, checkpoint_policy_hash=before, source_sha256=sha(__file__),
        collector_sha256=sha(feedback.__file__), reference=reference, rows=rows, contexts=contexts,
        totals=totals, all_physical_checks=physical, no_optimization=True, state_noise_sigma=0.,
        common_information_certificate_applies=False, success_threshold=SUCCESS_THRESHOLD,
        criterion_pass=controller == 'policy' and totals['trials'] == 256 and physical and totals['successes'] >= SUCCESS_THRESHOLD,
        seconds=time.monotonic()-started, environment=dict(device=device, torch_version=str(torch.__version__)))
    atomic_json(out/'summary.json', report)
    return report


def evaluate_bank(net, cfg, out, *, device='cpu', manifest=BEHAVIOR, controller='policy', actor_mode=None):
    plan, bank, _ = load_bank(manifest)
    reference = dict(manifest_path=str(Path(manifest).resolve().relative_to(ROOT)),
        manifest_sha256=sha(manifest), bank_path=bank['path'], bank_sha256=bank['sha256'])
    return evaluate_plan(net, cfg, plan, reference, out, device=device, controller=controller, actor_mode=actor_mode)


class ReferencePolicy(torch.nn.Module):
    """Zero-cost placeholders; reference actions come only from uniform/planner rules."""
    n_actions, extra_in = 6, 0
    _concat = PolicyNet._concat
    def __init__(self, nh): super().__init__(); self.nh = nh
    def embed(self, rgb): return torch.zeros(len(rgb), 256, device=rgb.device)
    def zero_state(self, n, device): return (torch.zeros(n, self.nh, device=device), torch.zeros(n, self.nh, device=device))
    def lstm(self, x, state): return state
    def pi(self, h): return torch.zeros(len(h), 6, device=h.device)
    def v(self, h): return torch.zeros(len(h), 1, device=h.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', action='store_true', required=True)
    parser.add_argument('--manifest', type=Path, default=BEHAVIOR)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--device', default='cpu')
    args = parser.parse_args(); torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True); assert not any(args.out.iterdir())
    # Config only, with no trained policy loaded or evaluated for these references.
    cfg = json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    cfg.update(n_envs=32, rollout=160, entropy=8e-5)
    net = ReferencePolicy(cfg['nh'])
    results = {name:evaluate_bank(net, cfg, args.out/name, device=args.device,
        manifest=args.manifest, controller=name) for name in ('uniform', 'planner')}
    atomic_json(args.out/'summary.json', dict(complete=True, manifest_sha256=sha(args.manifest),
        reference_trials=512, no_policy_or_optimization=True,
        results={name:dict(summary_sha256=sha(args.out/name/'summary.json'), totals=r['totals']) for name,r in results.items()}))


if __name__ == '__main__': main()
