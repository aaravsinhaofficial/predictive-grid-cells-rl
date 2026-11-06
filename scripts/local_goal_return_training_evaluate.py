"""Clean, sealed-bank local-return evaluation and reset-head baseline.

No optimization or adaptive trial selection. Every physical context is retained,
including preparation failures. The separately frozen behavioral gates predate
all baseline outcomes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from code.common import build_policy
from rl_goal_memory import atomic_json, atomic_torch, state_digest
import local_goal_return_training as task

BEHAVIOR = ROOT / 'results/local_goal_return_design/behavior_protocol.json'
BEHAVIOR_HASH = '4c7ba90dd79c8c0b5f8ac13ba89e4a7d6ed5bf0287556982ade07f911fc16e48'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def load_bank(name):
    assert sha(BEHAVIOR) == BEHAVIOR_HASH
    design = json.loads(BEHAVIOR.read_text()); item = design['banks'][name]
    assert hashlib.sha256(inspect.getsource(task.TrialStreams).encode()).hexdigest() == design['generator_class_sha256']
    assert hashlib.sha256(inspect.getsource(task.Stage).encode()).hexdigest() == design['stage_class_sha256']
    path = ROOT / item['path']; assert sha(path) == item['sha256']
    plan = torch.load(path, map_location='cpu', weights_only=True)
    for key, record in item['tensors'].items():
        assert list(plan[key].shape) == record['shape'] and str(plan[key].dtype) == record['dtype']
        assert task.tensor_hash(plan[key]) == record['sha256'], key
    return plan, item, design


def context_plan(plan, index):
    assert 0 <= index < plan['n_contexts']; headings = plan['headings']
    interval = slice(index*headings, (index+1)*headings); result = {}
    for key, value in plan.items():
        if isinstance(value, torch.Tensor):
            if key in ('motor', 'state'): result[key] = value[:, :, interval].clone()
            elif key == 'action': result[key] = value[:, interval].clone()
            else: result[key] = value[interval].clone()
        else: result[key] = value
    result['n_contexts'] = 1
    return result


def initial_policy(condition, seed=1, device='cpu'):
    stem = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    original = json.loads(stem.with_suffix('.json').read_text())
    parent = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert parent['frames'] == 1_000_000_000
    # Architecture and code remain exactly native; only explicit task/RL fields change.
    cfg = dict(original, n_envs=32, rollout=160, entropy=.01)
    with torch.random.fork_rng(devices=[]):
        net = build_policy(cfg, 'cpu')
    net.load_state_dict(parent['policy'], strict=True)
    initialization = task.reset_native_heads(net, task.HEAD_SEED + seed - 1)
    initialization.update(parent_checkpoint=str(stem.with_suffix('.pt').relative_to(ROOT)),
        parent_checkpoint_sha256=sha(stem.with_suffix('.pt')),
        parent_config_sha256=sha(stem.with_suffix('.json')), original_frames=parent['frames'],
        full_initialized_policy_hash=state_digest(net))
    return net.to(device), cfg, initialization


def summarize_rows(rows, variants, contexts):
    result = {}
    for variant in variants:
        selected = [r for r in rows if r['variant'] == variant]
        assert len(selected) == contexts * 16
        valid = [r for r in selected if not r.get('invalid', False)]
        mean = lambda key: sum(r.get(key, 0.) for r in selected) / len(selected)
        result[variant] = dict(trials=len(selected), successes=sum(r['hit'] for r in selected),
            success_fraction=sum(r['hit'] for r in selected)/len(selected), invalid_trials=len(selected)-len(valid),
            mean_path_m=mean('free_path_m'), mean_progress_m=mean('progress_m'),
            mean_free_entropy=mean('mean_free_entropy'), mean_initial_free_entropy=mean('first_free_entropy'),
            mean_max_action_probability=mean('mean_max_action_probability'),
            mean_free_value=mean('mean_value'), max_abs_logit=max((r.get('max_abs_logit', 0.) for r in selected)))
    return result


@torch.inference_mode()
def evaluate_bank(net, cfg, bank_name, out, *, device='cpu', controls=False, variants=task.VARIANTS):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'summary.json').exists(), 'Preserve every completed bank result'
    plan, bank, behavior = load_bank(bank_name); source = task.make_source(cfg, device)
    before = state_digest(net); was_training = net.training; net.eval()
    rows, context_reports = [], []; control_rows = {key: [] for key in ('uniform', 'planner')}
    started = time.monotonic()
    for index in range(plan['n_contexts']):
        chunk = context_plan(plan, index); folder = out / f'context{index}'; folder.mkdir()
        reference = dict(bank=bank_name, path=bank['path'], sha256=bank['sha256'], context_index=index,
            context_ids=chunk['context'].tolist(), plan_generator_class_sha256=behavior['generator_class_sha256'])
        atomic_json(folder/'input.json', reference)
        try:
            roll, local_rows, audit, first_free = task.collect(net, cfg, source, chunk,
                variants=variants, sigma=0., device=device, retain_inputs=False)
            raw = {k: torch.stack(v).cpu() for k, v in roll.items() if isinstance(v, list) and v}
            raw.update(goal=roll['goal'].cpu(), first_free=first_free, plan_reference=reference)
            atomic_torch(folder/'raw.pt', raw)
            context_result = dict(context=index, rows=local_rows, audit=audit, certificate=None,
                raw_sha256=sha(folder/'raw.pt'), valid=True)
        except Exception as error:
            # Never replace a failed context after inspecting its actions.
            atomic_torch(folder/'failed_plan.pt', chunk)
            context_result = dict(context=index, valid=False, failure_category='policy_collection',
                error_type=type(error).__name__, error=str(error))
            local_rows = [dict(variant=v, context=index, heading_index=h, hit=False, invalid=True)
                          for v in variants for h in range(16)]
            context_result['rows'] = local_rows
        if context_result['valid'] and 'blind_reset_after_glimpse' in variants:
            try:
                from local_goal_return_training_certificate import certify_context
                context_result['certificate'] = certify_context(chunk, cfg, roll, first_free, variants)
            except Exception as error:
                atomic_json(folder/'certificate_provenance_failure.json', dict(
                    error_type=type(error).__name__, error=str(error), policy_rows_preserved=local_rows))
                raise  # A broken provenance validator is not an ordinary failed physical trial.
        if context_result['valid'] and cfg['cond'] == 'gc100':
            for view in ('blind', 'seen'):
                if f'{view}_native' not in variants: continue
                native_rows = [{k:v for k,v in row.items() if k != 'variant'} for row in local_rows
                               if row['variant'] == f'{view}_native']
                for cue in ('ordinary', 'plus60', 'minus60'):
                    if f'{view}_{cue}' in variants:
                        sham_rows = [{k:v for k,v in row.items() if k != 'variant'} for row in local_rows
                                     if row['variant'] == f'{view}_{cue}']
                        assert sham_rows == native_rows, 'Grid cue replacement must be an exact identity'
            context_result['grid_shams_exact'] = True
        if controls:
            context_result['controls'] = {}
            for name in control_rows:
                try:
                    rr, outcomes, checks, _ = task.collect(net, cfg, source, chunk,
                        variants=('blind_native',), sigma=0., device=device, retain_inputs=False, controller=name)
                    control_result = dict(rows=outcomes, audit=checks, valid=True)
                    atomic_torch(folder/f'{name}_raw.pt', {k: torch.stack(v).cpu() for k, v in rr.items() if isinstance(v, list) and v})
                except Exception as error:
                    outcomes = [dict(variant='blind_native', context=index, heading_index=h, hit=False, invalid=True)
                                for h in range(16)]
                    atomic_torch(folder/f'{name}_failed_plan.pt', chunk)
                    control_result = dict(rows=outcomes, valid=False, failure_category='control_collection',
                        error_type=type(error).__name__, error=str(error))
                control_rows[name].extend(outcomes); context_result['controls'][name] = control_result
        rows.extend(local_rows); context_reports.append(context_result)
        atomic_json(folder/'result.json', context_result)
        print(json.dumps(dict(bank=bank_name, condition=cfg['cond'], context=index,
            valid=context_result['valid'], native_hits=sum(r['hit'] for r in local_rows if r['variant']=='blind_native'))), flush=True)
    summaries = summarize_rows(rows, variants, plan['n_contexts'])
    all_physical = all(r['valid'] and r['audit']['forced_hits'] == 0 for r in context_reports)
    if controls:
        all_physical = all_physical and all(control['valid'] and control['audit']['forced_hits'] == 0
            for result in context_reports for control in result['controls'].values())
    certificates = [r['certificate'] for r in context_reports if r['valid'] and r['certificate'] is not None]
    from local_goal_return_training_certificate import aggregate_certificates
    certification = aggregate_certificates(certificates, expected_contexts=16)
    success = summaries['blind_native']['successes']; erase = summaries.get('blind_reset_after_glimpse', {}).get('successes')
    gates = dict(near_baseline_pass=bank_name=='near_baseline' and all_physical and success >= 13,
        hard_numeric_pass=bank_name=='final_hard' and all_physical and success >= 167 and erase is not None and success-erase >= 39,
        hard_certified_memory_pass=False)
    gates['hard_certified_memory_pass'] = gates['hard_numeric_pass'] and certification['all_pairs_certified']
    assert state_digest(net) == before
    report = dict(bank=bank_name, bank_sha256=bank['sha256'], behavior_protocol_sha256=BEHAVIOR_HASH,
        source_sha256=sha(__file__), core_sha256=sha(Path(task.__file__)), condition=cfg['cond'], model_seed=cfg['seed'],
        checkpoint_policy_hash=before, variants=summaries, rows=rows, contexts=context_reports,
        all_physical_checks=all_physical, certificate=certification, gates=gates,
        complete=True, no_optimization=True, clean_state_noise_sigma=0., seconds=time.monotonic()-started,
        controls={name: summarize_rows(outcomes, ('blind_native',), 16)['blind_native']
                  for name, outcomes in control_rows.items()} if controls else {},
        environment=dict(torch_version=str(torch.__version__), device=device, torch_threads=torch.get_num_threads()))
    atomic_json(out/'summary.json', report); net.train(was_training)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('gc100', 'pgc33'), required=True)
    parser.add_argument('--seed', type=int, default=1); parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True); assert not any(args.out.iterdir()), 'Use an empty baseline output folder'
    net, cfg, initialization = initial_policy(args.condition, args.seed, args.device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        assert saved['config'] == cfg; net.load_state_dict(saved['policy'])
        initialization = saved['initialization']
    atomic_json(args.out/'execution.json', dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=sha(__file__), core_sha256=sha(Path(task.__file__)), behavior_protocol_sha256=sha(BEHAVIOR),
        initialization=initialization, checkpoint=None if args.checkpoint is None else str(args.checkpoint),
        checkpoint_sha256=None if args.checkpoint is None else sha(args.checkpoint)))
    if args.checkpoint is None:
        atomic_torch(args.out/'initialized_policy.pt', dict(policy=net.state_dict(), config=cfg,
            initialization=initialization, no_optimization=True, behavior_protocol_sha256=BEHAVIOR_HASH,
            source_sha256=sha(__file__), core_sha256=sha(Path(task.__file__)), torch_version=str(torch.__version__)))
        atomic_json(args.out/'initialization.json', dict(initialization,
            initialized_policy_sha256=sha(args.out/'initialized_policy.pt')))
    source_hashes = (sha(__file__), sha(Path(task.__file__)))
    results = {name: evaluate_bank(net, cfg, name, args.out/name, device=args.device, controls=args.controls)
               for name in ('near_baseline', 'final_hard')}
    assert source_hashes == (sha(__file__), sha(Path(task.__file__))), 'Evaluation source changed during baseline'
    atomic_json(args.out/'summary.json', dict(complete=True, initialization=initialization,
        initialized_policy_sha256=None if args.checkpoint else sha(args.out/'initialized_policy.pt'),
        evaluation_source_sha256=source_hashes[0], core_source_sha256=source_hashes[1],
        bank_gates={name:r['gates'] for name,r in results.items()},
        outputs={name:str(args.out/name/'summary.json') for name in results}))


if __name__ == '__main__': main()
