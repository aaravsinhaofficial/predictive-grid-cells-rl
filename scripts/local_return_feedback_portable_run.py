"""Portable bank-authentication entrypoint; frozen v2 learning computations unchanged.

Preparation only until a sealed bank and prospective protocol exist. No bank,
policy outcome, optimization or cloud operation is executed on import.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import datetime as dt
import inspect
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import torch
from rl_goal_memory import atomic_json, atomic_torch, state_digest
from rl_visual_memory_extend import tree_equal
import local_return_feedback as feedback
import local_return_feedback_portable_evaluate as evaluator

PROTOCOL = ROOT/'docs/research/local_return_feedback_portable_protocol.json'
FROZEN_V2_PROTOCOL = ROOT/'docs/research/local_return_feedback_training_protocol.json'
FROZEN_V2_SHA256 = 'ad969e782ec2943de2e59b2d72629002dea537f7c9c0fcf2ad3905ce051038c7'
INITIALIZATION = ROOT/'docs/research/local_return_feedback_initialization.json'
INITIAL_POLICIES = ROOT/'results/local_return_feedback_design/initialization'
UPDATES, TRIALS_PER_BATCH, TRAIN_SEED = 1024, 32, 494801
MAX_SECONDS, PROVIDER_SECONDS = 12600, 14400
VERSION = 'feedback_acquisition_exact_resume_v2'
COUNTERS = ('active_physics_updates', 'forced_updates', 'free_updates', 'padded_slots', 'setup_pickups', 'free_hits')
NEW_SOURCES = ('scripts/local_return_feedback.py', 'scripts/local_return_feedback_evaluate.py',
    'scripts/local_return_feedback_run.py', 'scripts/audit_local_return_feedback.py',
    'scripts/audit_local_goal_return_baseline.py', 'scripts/audit_local_goal_return_training.py',
    'tests/test_local_return_feedback.py', 'tests/test_local_return_feedback_run.py',
    'tests/test_local_return_feedback_evaluate.py', 'tests/test_audit_local_return_feedback.py')
sha = feedback.sha
global_rng_snapshot, restore_global_rng = evaluator.global_rng_snapshot, evaluator.restore_global_rng


def assert_config(cfg):
    for key, expected in dict(n_envs=32, rollout=160, nh=256, n_cells=512, action_repeat=4,
            entropy=8e-5, lr=3e-4, rms_alpha=.99, rms_eps=1e-5, gamma=.99,
            value_coef=.5, grad_clip=40., reward_clip=1., policy='lstm', conv_arch='si',
            encoder_norm='layernorm', extra_input='none', code_noise=0., goal_radius=.3).items():
        assert cfg[key] == expected, (key, cfg[key])
    assert cfg['cond'] in ('gc100', 'pgc33') and cfg['seed'] == 1


def normalized_initialization(initialization):
    result = copy.deepcopy(initialization)
    for key in ('parent_checkpoint_path', 'archived_heads_path'):
        result[key] = str(Path(result[key]).resolve().relative_to(ROOT))
    return result


def prepare_initializations():
    """Authorized provenance-only composition: no forward pass, bank or optimizer."""
    assert not INITIALIZATION.exists() and not INITIAL_POLICIES.exists(), 'Preserve earlier exact inputs'
    INITIAL_POLICIES.mkdir(parents=True); entries = []; heads = []
    with evaluator.preserved_rng():
        for condition in ('gc100', 'pgc33'):
            for mode in feedback.ACTOR_MODES:
                net, cfg, initialization = feedback.initialize(condition, mode, device='cpu')
                assert_config(cfg); initialization = normalized_initialization(initialization)
                name = f'{condition}_{mode}.pt'; path = INITIAL_POLICIES/name
                saved = dict(policy=net.state_dict(), config=cfg, initialization=initialization,
                    no_forward_pass=True, no_optimization=True, collector_source_sha256=sha(feedback.__file__))
                atomic_torch(path, saved)
                entries.append(dict(condition=condition, actor_mode=mode, path=str(path.relative_to(ROOT)),
                    sha256=sha(path), bytes=path.stat().st_size, full_initial_policy_hash=state_digest(net),
                    incoming_head_hashes=initialization['incoming_head_hashes']))
                heads.append(initialization['incoming_head_hashes'])
    for key in ('v.weight', 'v.bias'): assert len({h[key] for h in heads}) == 1
    for key in ('pi.weight', 'pi.bias'): assert heads[1][key] == heads[3][key]
    report = dict(version='feedback_acquisition_exact_incoming_inputs_v2', complete=True,
        no_forward_pass=True, no_optimization=True, no_bank_generated=True, policies=entries,
        collector_source_sha256=sha(feedback.__file__),
        normalization_function_sha256=hashlib.sha256(inspect.getsource(normalized_initialization).encode()).hexdigest(),
        original_nonhead_bytes_exact=True, critic_bytes_identical_all_four_arms=True,
        reset_actor_bytes_identical_across_populations=True, created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(INITIALIZATION, report)
    return report


def initial_assets():
    report = json.loads(INITIALIZATION.read_text())
    assert report['version'] == 'feedback_acquisition_exact_incoming_inputs_v2' and report['complete']
    assert report['no_forward_pass'] and report['no_optimization'] and report['no_bank_generated']
    assert report['collector_source_sha256'] == sha(feedback.__file__)
    assert report['normalization_function_sha256'] == hashlib.sha256(inspect.getsource(normalized_initialization).encode()).hexdigest()
    assert {(r['condition'], r['actor_mode']) for r in report['policies']} == {
        (c, a) for c in ('gc100', 'pgc33') for a in feedback.ACTOR_MODES}
    assert len(report['policies']) == 4
    assets = {str(INITIALIZATION.relative_to(ROOT)):sha(INITIALIZATION)}
    for row in report['policies']:
        assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
        path = ROOT/row['path']; assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        assets[row['path']] = row['sha256']
    return assets


def load_initial_policy(condition, mode, device):
    initial_assets()
    row = next(r for r in json.loads(INITIALIZATION.read_text())['policies']
               if r['condition'] == condition and r['actor_mode'] == mode)
    saved = torch.load(ROOT/row['path'], map_location='cpu', weights_only=True)
    net, cfg, expected = feedback.initialize(condition, mode, device='cpu')
    expected = normalized_initialization(expected)
    assert saved['config'] == cfg and saved['initialization'] == expected
    assert saved['no_forward_pass'] and saved['no_optimization']
    assert saved['collector_source_sha256'] == sha(feedback.__file__)
    assert tree_equal(saved['policy'], net.state_dict())
    net.load_state_dict(saved['policy'], strict=True)
    assert state_digest(net) == row['full_initial_policy_hash'] == expected['full_initial_policy_hash']
    return net.to(device), cfg, expected


def protocol(manifest=evaluator.BEHAVIOR):
    """Copy the frozen numerical design; add only portable authentication provenance."""
    assert sha(FROZEN_V2_PROTOCOL)==FROZEN_V2_SHA256
    design=json.loads(FROZEN_V2_PROTOCOL.read_text());assert design.pop('frozen_utc')
    for name,expected in design['source_sha256'].items():assert sha(ROOT/name)==expected,name
    for name,expected in design['input_assets'].items():assert sha(ROOT/name)==expected,name
    _,bank,_=evaluator.load_bank(manifest)
    assert bank==design['evaluation']['bank']
    assert str(Path(manifest).resolve().relative_to(ROOT))==design['evaluation']['manifest_path']
    assert sha(manifest)==design['evaluation']['manifest_sha256']
    for name in ('scripts/local_return_feedback_portable_evaluate.py',
                 'scripts/local_return_feedback_portable_run.py',
                 'tests/test_local_return_feedback_portable.py'):
        design['source_sha256'][name]=sha(ROOT/name)
    design['input_assets'][str(FROZEN_V2_PROTOCOL.relative_to(ROOT))]=FROZEN_V2_SHA256
    design['input_assets'][evaluator.ORIGIN_PATH]=evaluator.ORIGIN_SHA256
    design['portability']=dict(parent_protocol_path=str(FROZEN_V2_PROTOCOL.relative_to(ROOT)),
        parent_protocol_sha256=FROZEN_V2_SHA256,
        origin_validation_path=evaluator.ORIGIN_PATH,origin_validation_sha256=evaluator.ORIGIN_SHA256,
        bank_authentication='Exact original file SHA/bytes, uncompressed tensor schema/SHA, manifest seed/stage/class pins and hash-pinned creation-time exact seed-regeneration evidence. No worker regeneration or tolerance-based acceptance.',
        same_saved_bank=True,same_training_computation=True,same_evaluate_plan=True,
        raw_evaluator_source='scripts/local_return_feedback_evaluate.py',
        infrastructure_correction_only=True)
    return design


def fresh_counters():
    return dict(updates=0, trials=0, **{key:0 for key in COUNTERS}, evaluation_records={})


def pending_evaluation(counters):
    updates = counters['updates']; records = counters['evaluation_records']
    assert 0 <= updates <= UPDATES and set(records) <= {'baseline', 'final'}
    if 'baseline' not in records:
        assert updates == 0 and not records, 'Cannot learn before the prescribed baseline evaluation'
        return 'baseline'
    if 'final' in records:
        assert updates == UPDATES
        return None
    return 'final' if updates == UPDATES else None


def snapshot(net, opt, streams, cfg, initialization, counters, provenance):
    assert streams.batches == counters['updates'] and streams.contexts == counters['trials']
    assert counters['trials'] == counters['updates']*cfg['n_envs']
    pending_evaluation(counters)
    return dict(version=VERSION, policy=copy.deepcopy(net.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
        streams=streams.snapshot(), global_rng=global_rng_snapshot(), config=copy.deepcopy(cfg),
        initialization=copy.deepcopy(initialization), counters=copy.deepcopy(counters), provenance=copy.deepcopy(provenance),
        no_inflight_trial=True, boundary_state='Every trial terminal; next trial h/c zero, pa=-1, pr=0',
        completed=counters['updates'] == UPDATES and 'final' in counters['evaluation_records'],
        active_physics_frames=counters['active_physics_updates']*cfg['action_repeat'])


def restore(net, opt, streams, saved, cfg, provenance):
    assert saved['version'] == VERSION and saved['no_inflight_trial']
    assert saved['config'] == cfg and saved['provenance'] == provenance
    counters = copy.deepcopy(saved['counters']); pending_evaluation(counters)
    assert saved['streams']['batches'] == counters['updates']
    assert saved['streams']['contexts'] == counters['trials'] == counters['updates']*cfg['n_envs']
    assert saved['active_physics_frames'] == counters['active_physics_updates']*cfg['action_repeat']
    net.load_state_dict(saved['policy'], strict=True); feedback.configure_trainable(net)
    opt.load_state_dict(saved['optimizer']); streams.restore(saved['streams']); restore_global_rng(saved['global_rng'])
    return counters


def advance_counters(counters, metrics, cfg):
    assert metrics['completed_trials'] == cfg['n_envs'] and metrics['setup_learning_reward'] == metrics['forced_hits'] == 0
    counters['updates'] += 1; counters['trials'] += cfg['n_envs']
    for key in COUNTERS: counters[key] += metrics[key]


def inherit_resume_artifacts(checkpoint, out, counters):
    """Preserve exact restart bytes, all old accepted evidence and rolled-back attempts."""
    checkpoint, out = Path(checkpoint), Path(out); parent = checkpoint.parent
    evidence = out/'resume_parent_evidence'; evidence.mkdir(); copied = []
    def copy_item(source, target):
        if source.is_dir(): shutil.copytree(source, target)
        else: shutil.copy2(source, target)
        files = sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
        for path in files:
            dest = target/path.relative_to(source) if source.is_dir() else target
            assert sha(path) == sha(dest)
            copied.append(dict(source=str(path.resolve()), copied=str(dest.relative_to(out)), sha256=sha(path)))
    assert (parent/'metrics.jsonl').exists()
    copy_item(checkpoint, evidence/'checkpoint.pt')
    # A resume at update1024 has no further update that would create latest.pt.
    # Preserve the exact supplied completed-trial boundary under that role too.
    if counters['updates'] == UPDATES: copy_item(checkpoint, out/'latest.pt')
    for name in ('incoming.pt', 'metrics.jsonl', 'protocol.json', 'initialization.json', 'training_config.json',
                 'incomplete.json', 'summary.json', 'resume_lineage.json', 'failed_plan.pt', 'failed_predraw_streams.pt'):
        if (parent/name).exists(): copy_item(parent/name, evidence/name)
    if (parent/'resume_parent_evidence').exists():
        copy_item(parent/'resume_parent_evidence', evidence/'ancestor_evidence')
    for endpoint, record in counters['evaluation_records'].items():
        assert sha(parent/f'{endpoint}_before_evaluation.pt') == record['checkpoint_before_sha256']
        assert sha(parent/f'{endpoint}_evaluation/summary.json') == record['summary_sha256']
        for name in (f'{endpoint}_before_evaluation.pt', f'{endpoint}.pt',
                     f'{endpoint}_evaluation', f'{endpoint}_evaluation_integrity.json'):
            copy_item(parent/name, out/name)
    pending = pending_evaluation(counters)
    if pending and (parent/f'{pending}_evaluation').exists():
        copy_item(parent/f'{pending}_evaluation', out/f'parent_incomplete_{pending}_evaluation')
    lineage = dict(resume_checkpoint=str(checkpoint.resolve()), resume_checkpoint_sha256=sha(checkpoint),
        restored_updates=counters['updates'], pending_evaluation=pending, copied_artifacts=copied,
        parent_lineage=json.loads((parent/'resume_lineage.json').read_text()) if (parent/'resume_lineage.json').exists() else None,
        parent_metrics_scope='All raw attempt rows preserved; only rows up to restored_updates are accepted. Later rows are rolled-back evidence, never double counted.')
    atomic_json(out/'resume_lineage.json', lineage)


def checked_evaluation(net, cfg, opt, streams, out, *, device, manifest=evaluator.BEHAVIOR, actor_mode=None):
    before = dict(policy=copy.deepcopy(net.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
        streams=streams.snapshot(), rng=global_rng_snapshot(), training=net.training)
    result = evaluator.evaluate_bank(net, cfg, out, device=device, manifest=manifest, actor_mode=actor_mode)
    assert tree_equal(before['policy'], net.state_dict()) and tree_equal(before['optimizer'], opt.state_dict())
    assert tree_equal(before['streams'], streams.snapshot()) and tree_equal(before['rng'], global_rng_snapshot())
    assert net.training == before['training'] and result['complete'] and result['all_physical_checks']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('gc100', 'pgc33'))
    parser.add_argument('--actor-mode', choices=feedback.ACTOR_MODES)
    parser.add_argument('--out', type=Path); parser.add_argument('--device', default='cuda')
    parser.add_argument('--manifest', type=Path, default=evaluator.BEHAVIOR)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--resume', type=Path); parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--prepare-initializations', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(2)
    if args.prepare_initializations:
        assert not args.freeze_only and not args.resume and not args.condition and not args.actor_mode
        report = prepare_initializations(); print(json.dumps(dict(complete=report['complete'], path=str(INITIALIZATION))))
        return
    design = protocol(args.manifest)
    if args.freeze_only:
        assert not args.protocol.exists(), 'Never overwrite prospective frozen source/assets'
        atomic_json(args.protocol, dict(design, frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(sha(args.protocol)); return
    assert args.condition and args.actor_mode and args.out and 0 < args.max_seconds <= MAX_SECONDS
    frozen = json.loads(args.protocol.read_text()); assert frozen.pop('frozen_utc'); assert frozen == design
    args.out.mkdir(parents=True, exist_ok=True); assert not any(args.out.iterdir()), 'Use a fresh experiment child folder'
    net, cfg, initialization = load_initial_policy(args.condition, args.actor_mode, args.device)
    assert_config(cfg); source = feedback.make_source(cfg, args.device); opt = feedback.optimizer(net, cfg)
    streams = feedback.TrialStreams(TRAIN_SEED); counters = fresh_counters()
    provenance = dict(design=design, protocol_sha256=sha(args.protocol), condition=args.condition,
        actor_mode=args.actor_mode, original_seed=1, device=args.device, torch_version=str(torch.__version__))
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        assert not saved['completed'], 'A completed fixed endpoint needs no resumed training'
        assert saved['initialization'] == initialization
        counters = restore(net, opt, streams, saved, cfg, provenance)
        inherit_resume_artifacts(args.resume, args.out, counters)
    atomic_json(args.out/'protocol.json', provenance); atomic_json(args.out/'initialization.json', initialization)
    atomic_json(args.out/'training_config.json', cfg)
    save = lambda name: atomic_torch(args.out/name, snapshot(net, opt, streams, cfg, initialization, counters, provenance))
    save('incoming.pt'); started = time.monotonic(); deadline = started+args.max_seconds
    (args.out/'metrics.jsonl').touch()
    def evaluate(endpoint):
        assert pending_evaluation(counters) == endpoint
        save(f'{endpoint}_before_evaluation.pt')
        result = checked_evaluation(net, cfg, opt, streams, args.out/f'{endpoint}_evaluation',
            device=args.device, manifest=args.manifest, actor_mode=args.actor_mode)
        record = dict(endpoint=endpoint, updates=counters['updates'],
            checkpoint_before_sha256=sha(args.out/f'{endpoint}_before_evaluation.pt'),
            summary_sha256=sha(args.out/f'{endpoint}_evaluation/summary.json'),
            policy_optimizer_private_global_rng_unchanged=True, successes=result['totals']['successes'])
        atomic_json(args.out/f'{endpoint}_evaluation_integrity.json', record)
        counters['evaluation_records'][endpoint] = record; save(f'{endpoint}.pt')
    try:
        pending = pending_evaluation(counters)
        if pending: evaluate(pending)
        with (args.out/'metrics.jsonl').open('w') as log:
            for _ in range(counters['updates'], UPDATES):
                if time.monotonic() > deadline-600:
                    save('incomplete_boundary.pt'); raise TimeoutError('Fixed schedule incomplete; no automatic extension')
                predraw = streams.snapshot(); plan = streams.draw(TRIALS_PER_BATCH, feedback.STAGE, cfg)
                try: metrics, rows = feedback.update(net, cfg, opt, source, plan, device=args.device)
                except Exception:
                    atomic_torch(args.out/'failed_plan.pt', plan); atomic_torch(args.out/'failed_predraw_streams.pt', predraw)
                    raise
                advance_counters(counters, metrics, cfg)
                item = dict(update=counters['updates'], seconds=time.monotonic()-started,
                    active_physics_frames=counters['active_physics_updates']*cfg['action_repeat'], **metrics,
                    action_counts=[sum(r['action_counts'][a] for r in rows) for a in range(6)],
                    mean_path_m=sum(r['free_path_m'] for r in rows)/len(rows),
                    mean_progress_m=sum(r['progress_m'] for r in rows)/len(rows))
                log.write(json.dumps(item)+'\n'); log.flush()
                if counters['updates'] % 20 == 0: print(json.dumps(item), flush=True)
                if counters['updates'] % 128 == 0: save('latest.pt')
        if pending_evaluation(counters): evaluate('final')
        assert counters['updates'] == UPDATES and counters['trials'] == UPDATES*TRIALS_PER_BATCH
        assert set(counters['evaluation_records']) == {'baseline', 'final'}
        save('final.pt')
        atomic_json(args.out/'summary.json', dict(completed=True, counters=counters,
            active_physics_frames=counters['active_physics_updates']*cfg['action_repeat'],
            final_checkpoint_sha256=sha(args.out/'final.pt'), final_evaluation='final_evaluation/summary.json',
            protocol_sha256=sha(args.protocol), seconds=time.monotonic()-started, initialization=initialization))
    except Exception as error:
        atomic_json(args.out/'incomplete.json', dict(error_type=type(error).__name__, error=str(error), counters=counters))
        raise


if __name__ == '__main__': main()
