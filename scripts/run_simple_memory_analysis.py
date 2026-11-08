"""Bounded operational wrapper for the frozen five-stage simple-memory analysis.

No download, VM creation, training, policy construction or scientific selection
lives here. Existing valid outputs may be authenticated and skipped; partial or
invalid outputs are retained and never silently overwritten/retried.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PROTOCOL = ROOT/'docs/research/simple_memory_analysis_protocol.json'
ANALYSIS_SHA256 = '156e8798fa2d94bc5e4478a0f4958335ddea51088907f1c307434624a51a3d7e'
STAGES = ('behavior', 'states', 'activity', 'dynamics', 'interventions')
PRODUCERS = dict(zip(STAGES, ('scripts/evaluate_simple_memory.py',
    'scripts/collect_simple_memory_states.py', 'scripts/analyze_simple_memory_activity.py',
    'scripts/analyze_simple_memory_dynamics.py', 'scripts/simple_memory_interventions.py')))
OUTPUTS = dict(zip(STAGES, ('memory_eval_gap60.json', 'memory_states', 'activity_analysis',
                            'dynamics_analysis', 'intervention_analysis')))
VARIANTS = ('blind', 'blind_reset_onset', 'blind_reset_each', 'seen', 'seen_reset_onset',
            'seen_reset_each', 'blind_current_grid', 'seen_current_grid')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(value, message):
    if not value: raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def local_path(base, relative):
    """Artifact/source manifests cannot escape their declared root."""
    base = Path(base).resolve()
    target = (base/relative).resolve()
    require(not Path(relative).is_absolute() and target.is_relative_to(base), f'Nonlocal artifact: {relative}')
    return target


def check_sources(manifest, root, required=()):
    require(isinstance(manifest, dict) and manifest and set(required) <= set(manifest), 'Missing source manifest entries')
    for name, digest in manifest.items():
        require(sha(local_path(root, name)) == digest, f'Changed source: {name}')


def authenticate_run(run, *, root=ROOT, analysis_protocol=None, expected_analysis_sha256=ANALYSIS_SHA256,
                     amendment_path=None):
    """CPU-only endpoint authentication; returns portable, JSON-safe evidence.

    Alternate root/protocol/hash arguments support isolated private fixtures and
    exact extracted packages. The production CLI always uses the frozen hash.
    """
    import torch
    root, run = Path(root).resolve(), Path(run).resolve()
    protocol_path = Path(analysis_protocol) if analysis_protocol else root/'docs/research/simple_memory_analysis_protocol.json'
    require(sha(protocol_path) == expected_analysis_sha256, 'Analysis protocol hash changed')
    analysis = read(protocol_path)
    require(analysis['status'] == 'frozen', 'Analysis protocol not frozen')
    check_sources(analysis['source_sha256'], root, PRODUCERS.values())
    training_path = local_path(root, analysis['training_protocol'])
    training = read(training_path)
    require(training['status'] == 'frozen', 'Training protocol not frozen')
    check_sources(training['source_sha256'], root, ('code/train_memory.py',))
    cfg, terminal, proof = (read(run/name) for name in ('config.json', 'summary.json', 'provenance.json'))
    require(terminal.get('complete') is True, 'Training endpoint incomplete')
    require(cfg['frames'] == terminal['frames'] == 1_000_000_000 and terminal['updates'] == 78125, 'Not the fixed 1B endpoint')
    for key, value in training['training_config'].items():
        require(cfg.get(key) == value, f'Training invariant changed: {key}')
    cohort = training['allowed_cohort']
    require(cfg['cond'] in cohort['conditions'] and cfg['seed'] in cohort['seeds'] and
            cfg['task_max_gap'] in cohort['task_max_gaps'], 'Model outside cohort')
    require(cfg['pgc_frac'] == cohort['pgc_fractions'][cfg['cond']], 'Population fraction changed')
    require(not cohort['code_seed_equals_seed'] or cfg['code_seed'] == cfg['seed'], 'Map seed changed')
    require(cfg['clock_seed'] == cfg['seed']*cohort['clock_seed_multiplier']+cohort['clock_seed_offset'], 'Clock seed changed')
    require(cfg['tf32'] == 0, 'FP32 configuration changed')
    require(proof['protocol_sha256'] == sha(training_path), 'Training protocol binding changed')
    if proof.get('provenance_version') == 2:
        from scripts.simple_memory_endpoint import validate_v2_provenance
        producer_evidence = validate_v2_provenance(run, cfg, proof, root=root, amendment_path=amendment_path)
    else:
        require('provenance_version' not in proof, 'Unknown training producer version')
        require(proof['source_sha256'] == training['source_sha256'], 'Training source closure changed')
        producer_evidence = dict(producer_version=1)
    require(proof['deterministic_algorithms'] is True and proof['cudnn_benchmark'] is False and
            proof['cudnn_tf32'] is False and proof['matmul_tf32'] is False, 'Training precision changed')
    final_hash = sha(run/'final.pt')
    require(final_hash == terminal['final_sha256'], 'Final checkpoint bytes changed')
    saved = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
    require(saved['schema'] == 1 and saved['frames'] == 1_000_000_000 and saved['updates'] == 78125, 'Final checkpoint counters/schema')
    require(saved['runtime']['counters']['frames'] == saved['frames'] and
            saved['runtime']['counters']['updates'] == saved['updates'], 'Runtime counters changed')
    require(saved['runtime']['counters']['decisions'] == terminal['decisions'] == 250_000_000,
            'Endpoint decision count changed')
    require(saved['provenance'] == terminal['provenance'] == proof, 'Endpoint provenance mismatch')
    scientific = {k: v for k, v in cfg.items() if k != 'population'}
    require(saved['config'] == scientific, 'Endpoint config key/value mismatch')
    require(proof['config_sha256'] == hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest(), 'Config digest mismatch')
    h = hashlib.sha256()
    require(bool(saved['policy']), 'Empty policy')
    for name, tensor in sorted(saved['policy'].items()):
        require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), f'Invalid policy tensor: {name}')
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    require(h.hexdigest() == saved['policy_sha256'] == terminal['final_policy_sha256'], 'Policy tensor digest mismatch')
    return dict(condition=cfg['cond'], training_seed=cfg['seed'], training_task_max_gap=cfg['task_max_gap'],
        frames=1_000_000_000, updates=78125, decisions=250_000_000, checkpoint_sha256=final_hash,
        policy_sha256=h.hexdigest(), config_sha256=sha(run/'config.json'),
        terminal_summary_sha256=sha(run/'summary.json'), training_provenance_sha256=sha(run/'provenance.json'),
        training_protocol_sha256=sha(training_path), analysis_protocol_sha256=sha(protocol_path),
        **producer_evidence,
        training_runtime={key:proof[key] for key in ('torch_version','cuda_version','cudnn_version',
                                                    'cuda_device_name','cuda_capability')})


def validate_runtime_versions(actual, endpoints):
    require(actual['cuda_available'] and actual['cuda_device_name'] == 'Tesla T4', 'Analysis requires CUDA Tesla T4')
    for endpoint in endpoints:
        require(all(actual[key] == value for key,value in endpoint['training_runtime'].items()),
                'Analysis GPU/Torch/CUDA/cuDNN must match the training provenance')


def probe_runtime(endpoints):
    """Fail before long stages if a worker lacks numerical/figure dependencies."""
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import platform
    import numpy as np
    import scipy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import threadpoolctl
    import torch
    torch.set_num_threads(2)
    require(torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback')
    actual=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,threadpoolctl=threadpoolctl.__version__,
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),cuda_available=True,
        cuda_device_name=torch.cuda.get_device_name(0),cuda_capability=list(torch.cuda.get_device_capability(0)),
        device='cuda:0',torch_threads=torch.get_num_threads(),threadpools=threadpoolctl.threadpool_info())
    validate_runtime_versions(actual,endpoints)
    # Importing pyplot here is intentional: a late renderer import failure
    # must not occur only after costly collection. No figure/outcome is made.
    del plt
    return actual


def artifact(base, item):
    target = local_path(base, item['path'])
    require(target.is_file() and sha(target) == item['sha256'], f'Artifact changed: {target}')
    if 'bytes' in item: require(target.stat().st_size == item['bytes'], f'Artifact size changed: {target}')
    return target


def stage_path(stage, run):
    return Path(run)/OUTPUTS[stage]


def validate_stage(stage, run, endpoint, *, root=ROOT):
    """Authenticate one producer's complete artifacts, independent of cloud paths.

    Failed/unqualified scientific tests remain complete records. This checks
    execution, cohort and provenance, never favorable outcomes or ring gates.
    """
    import numpy as np
    root, run = Path(root), Path(run)
    base = stage_path(stage, run)
    summary_path = base if stage == 'behavior' else base/'summary.json'
    s = read(summary_path)
    require(s.get('complete') is True, f'{stage}: incomplete producer')
    for key in ('condition', 'training_seed', 'training_task_max_gap'):
        require(s[key] == endpoint[key], f'{stage}: model identity changed')
    if stage in ('activity', 'interventions'):
        p = s['provenance']
        require(p['final_checkpoint_sha256'] == endpoint['checkpoint_sha256'] and
                p['final_policy_sha256'] == endpoint['policy_sha256'], f'{stage}: endpoint binding')
        require(p['analysis_protocol_sha256'] == endpoint['analysis_protocol_sha256'] and
                p['training_protocol_sha256'] == endpoint['training_protocol_sha256'], f'{stage}: protocol binding')
        require(p['collection_summary_sha256'] == sha(run/'memory_states/summary.json') and
                p['states_sha256'] == sha(run/'memory_states/states.npz'), f'{stage}: collection binding')
        require(p['producer_sha256'] == sha(root/PRODUCERS['activity']), f'{stage}: authentication helper binding')
    else:
        require(s['checkpoint_sha256'] == endpoint['checkpoint_sha256'], f'{stage}: endpoint binding')
        require(s['config_sha256'] == endpoint['config_sha256'], f'{stage}: config binding')
        require(s['checkpoint_frames'] == endpoint['frames'], f'{stage}: frame binding')
    if stage != 'behavior':
        if stage != 'activity': require(s['analysis_protocol_sha256'] == endpoint['analysis_protocol_sha256'], f'{stage}: analysis binding')
        require(s['policy_unchanged'] is True if stage != 'activity' else True, f'{stage}: changed policy')
    if stage != 'activity':
        check_sources(s['source_sha256'], root, (PRODUCERS[stage],))
    if stage == 'behavior':
        for key, value in dict(seed=741021, episodes_per_variant=128, n_envs=32,max_gap=60,min_gap=30,visible_steps=15,frames=5_529_600).items():
            require(s[key] == value, f'Behavior setting/count: {key}')
        require(math.isfinite(s['seconds']) and s['seconds'] > 0 and s['all_first_discoveries_identical'] is True, 'Behavior timing/discovery')
        require(set(s['variants']) == set(VARIANTS), 'Behavior variants incomplete')
        for name, row in s['variants'].items():
            require(len(row['rows']) == 128, f'Behavior episode count: {name}')
            require(len({(r['episode'], r['slot']) for r in row['rows']}) == 128, 'Duplicate behavior episodes')
            require(len(row['scheduled_action_counts']) == 6 and isinstance(row['totals'], dict), 'Missing behavior exposure')
        n = s['evaluation_numerics']
        require(n['device'] == 'cuda:0' and n['deterministic_algorithms'] is True and not n['cudnn_tf32'] and not n['matmul_tf32'], 'Behavior numerical settings')
        if endpoint['condition'] == 'gc100': require(s['grid_swap_is_exact_sham'] is True, 'Grid sham failed')
    elif stage == 'states':
        for key, value in dict(seed=741022,episodes=128,n_envs=32,max_gap=60,stride=3,samples=57600,frames=691200).items():
            require(s[key] == value, f'Collection setting/count: {key}')
        for key in ('terminal_summary_sha256','training_provenance_sha256','training_protocol_sha256','policy_sha256'):
            require(s[key] == endpoint[key], f'Collection binding: {key}')
        a = artifact(base, s['array_file']); artifact(base, s['contexts_file'])
        require(s['array_file']['path'] == 'states.npz' and s['contexts_file']['path'] == 'onset_contexts.pt', 'Collection artifact names')
        with np.load(a, allow_pickle=False) as z:
            require(set(z.files) == set(s['array_schema']), 'Collection array key mismatch')
            for key, spec in s['array_schema'].items():
                x = z[key]
                require(list(x.shape) == spec['shape'] and str(x.dtype) == spec['dtype'] and x.nbytes == spec['bytes'] and
                        hashlib.sha256(x.tobytes()).hexdigest() == spec['sha256'], f'Collection array changed: {key}')
            require(z['h'].shape == z['c'].shape == (57600,256), 'Collection state shape')
        require(0 <= s['onset_contexts']['count'] <= 4, 'Context count')
        import torch
        payload=torch.load(base/'onset_contexts.pt',map_location='cpu',weights_only=False)
        require(payload['schema']==1 and payload['checkpoint_sha256']==endpoint['checkpoint_sha256'], 'Context archive provenance')
        require(len(payload['contexts'])==s['onset_contexts']['count']==len(s['onset_contexts']['rows']), 'Context archive count')
        for context,row in zip(payload['contexts'],s['onset_contexts']['rows']):
            require(all(context[key]==value for key,value in row.items()), 'Context identity mismatch')
            require(0 < len(context['native_continuation']) <= 150, 'Context continuation count')
        require(s['numerics']['device'] == 'cuda:0' and s['numerics']['threads'] == 2, 'Collection device/threads')
    elif stage == 'activity':
        artifact(base, dict(path='activity_arrays.npz',sha256=s['numeric_artifact_sha256']))
        require(s['numerics']['threads'] == 2 and s['numerics']['accumulation'] == 'float64', 'Activity numerics')
        require(all((base/name).is_file() and (base/name).stat().st_size > 0 for name in ('activity_summary.png','activity_summary.pdf')), 'Activity figures incomplete')
    else:
        collection = read(run/'memory_states/summary.json')
        require(len(s['contexts']) == collection['onset_contexts']['count'], f'{stage}: dropped contexts')
        require([row['index'] for row in s['contexts']] == list(range(len(s['contexts']))), f'{stage}: context indices')
        if stage == 'dynamics':
            require(s['state_summary_sha256'] == sha(run/'memory_states/summary.json'), 'Dynamics collection summary')
            require(s['numerical_settings']['family_device'] == 'cuda:0' and s['numerical_settings']['threads'] == 2, 'Dynamics numerics')
        else:
            require(s['activity_summary_sha256'] == sha(run/'activity_analysis/summary.json'), 'Intervention activity binding')
        for row in s['contexts']:
            selected=collection['onset_contexts']['rows'][row['index']]
            require(row['episode_id']==selected['episode_id'] and row['global_step']==selected['global_step'], f'{stage}: context identity changed')
            require(read(base/f"context{row['index']}.json") == row, f'{stage}: context summary mismatch')
            if stage == 'dynamics': artifact(base, row['arrays'])
            elif row['family']['qualified']:
                require(row['cue_repair']['complete'] and row['behavior']['complete'], 'Partial interventions')
                behavior=row['behavior']
                require(behavior['tapes']==4 and behavior['horizon']==150 and len(behavior['records'])==100,
                        'Intervention branch/budget count changed')
                require(row['qualified']==behavior['validation_qualified']==behavior['validations']['no_op_native_continuation']['qualified'],
                        'No-op failure must remain explicitly unqualified')
                require(len(row['artifacts']) == 2, 'Missing intervention raw archives')
                for name, item in row['artifacts'].items(): artifact(base, dict(path=name, **item))
    files = [base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file())
    result = dict(complete=True, summary_path=summary_path.relative_to(run).as_posix(), summary_sha256=sha(summary_path),
                 artifacts={p.relative_to(run).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in files})
    if stage == 'behavior': result.update(frames=s['frames'], seconds=s['seconds'], variants=list(VARIANTS), episodes_per_variant=128)
    if stage == 'states': result.update(frames=s['frames'],seconds=s['seconds'],samples=s['samples'],contexts=s['onset_contexts']['count'])
    return result


def command(stage, run, remaining, *, root=ROOT, python=sys.executable):
    require(remaining > 0, 'Deadline exhausted')
    run, root = Path(run), Path(root)
    args = [python, str(root/PRODUCERS[stage]), '--run', str(run), '--out', str(stage_path(stage, run))]
    if stage == 'behavior': args += ['--checkpoint','final.pt','--episodes','128','--n-envs','32','--max-gap','60','--seed','741021']
    elif stage != 'states': args += ['--states',str(run/'memory_states')]
    if stage == 'activity': args += ['--figures','--protocol',str(root/'docs/research/simple_memory_analysis_protocol.json')]
    else: args += ['--device','cuda:0','--max-seconds',format(remaining,'.6f')]
    if stage != 'behavior': args += ['--threads','2']
    if stage == 'states': args += ['--analysis-protocol',str(root/'docs/research/simple_memory_analysis_protocol.json')]
    if stage == 'interventions': args += ['--activity',str(run/'activity_analysis')]
    return args


def run_child(argv, log, timeout, *, root=ROOT):
    """Kill the complete subprocess group on deadline; retain stdout/stderr."""
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    started = time.monotonic()
    with Path(log).open('xb') as output:
        child = subprocess.Popen(argv, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            return dict(exit_code=code, timed_out=False, seconds=time.monotonic()-started)
        except BaseException as exc:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                return dict(exit_code=child.returncode, timed_out=True, seconds=time.monotonic()-started)
            raise


def run_pipeline(runs, out, max_seconds=3000, *, root=ROOT, authenticate=authenticate_run,
                 validate=validate_stage, execute=run_child, clock=time.monotonic, runtime_probe=probe_runtime):
    root, out = Path(root).resolve(), Path(out).resolve()
    runs = [Path(p).resolve() for p in runs]
    require(1 <= len(runs) <= 2 and len(set(runs)) == len(runs), 'Supply one or two distinct run paths')
    require(0 < max_seconds <= 3000, 'Global budget must be in (0,3000] seconds')
    require(all(p.is_relative_to(out.parent) for p in runs), 'Runs must be within the pipeline output parent for portable collection')
    require(not out.exists(), 'Preserve previous pipeline records; choose a new output path')
    started = clock()
    deadline = started+max_seconds
    progress = dict(schema=1,complete=False,started_utc=utc(),max_seconds=max_seconds,
        analysis_protocol_sha256=ANALYSIS_SHA256,runner_sha256=sha(__file__),
        policy='Fixed five stages; no retries, no replacement, no score-based completion gate.',
        runs=[dict(run=p.relative_to(out.parent).as_posix(),status='pending',stages={}) for p in runs])
    logs = out.parent/(out.stem+'_logs'); logs.mkdir(parents=True,exist_ok=True)
    atomic_json(out,progress)
    try:
        # Authenticate every endpoint before the first scientific subprocess.
        for run, record in zip(runs, progress['runs']):
            require(clock() < deadline, 'Deadline during authentication')
            record['endpoint'] = authenticate(run,root=root)
            record['status'] = 'authenticated'
            atomic_json(out,progress)
        progress['runtime_preflight']=runtime_probe([r['endpoint'] for r in progress['runs']])
        atomic_json(out,progress)
        for index,(run,record) in enumerate(zip(runs,progress['runs'])):
            record['status']='running'
            for stage in STAGES:
                require(clock() < deadline, 'Global analysis deadline exhausted')
                target = stage_path(stage,run)
                step = record['stages'][stage] = dict(status='validating_existing' if target.exists() else 'pending')
                atomic_json(out,progress)
                if target.exists() and (target.is_file() or any(target.iterdir())):
                    step.update(status='skipped_valid',validation=validate(stage,run,record['endpoint'],root=root))
                else:
                    remaining = deadline-clock()
                    argv = command(stage,run,remaining,root=root)
                    log = logs/f'{index}_{stage}.log'
                    step.update(status='running',argv=argv,started_utc=utc(),remaining_seconds=remaining,
                                log=log.relative_to(out.parent).as_posix())
                    atomic_json(out,progress)
                    execution = execute(argv,log,remaining,root=root)
                    step.update(execution,finished_utc=utc(),log_sha256=sha(log))
                    require(not execution['timed_out'] and execution['exit_code'] == 0, f'{stage}: subprocess failed or timed out')
                    step.update(validation=validate(stage,run,record['endpoint'],root=root),status='complete')
                atomic_json(out,progress)
            record['status']='complete'
        # Recheck input/source bytes after all five writers; never rely only on exit0.
        for run,record in zip(runs,progress['runs']):
            require(authenticate(run,root=root) == record['endpoint'], 'Inputs changed during analysis')
            for step in record['stages'].values():
                for name,item in step['validation'].get('artifacts',{}).items(): artifact(run,dict(path=name,**item))
        require(clock() <= deadline, 'Global deadline exceeded during final authentication')
        progress['complete']=True
    except BaseException as exc:
        progress['error']=dict(type=type(exc).__name__,message=str(exc))
        for record in progress['runs']:
            if record['status']=='running':
                record['status']='incomplete'
                for step in record['stages'].values():
                    if step['status'] in ('running','pending','validating_existing'): step['status']='incomplete'
    finally:
        progress.update(finished_utc=utc(),seconds=clock()-started)
        atomic_json(out,progress)
    return progress


def verify_pipeline(pair_output_dir, *, root=ROOT, pipeline_name='pipeline.json'):
    """Read-only verification after archive/GCS relocation; no model execution."""
    base=Path(pair_output_dir).resolve(); p=read(base/pipeline_name)
    require(p.get('complete') is True and p['analysis_protocol_sha256']==ANALYSIS_SHA256, 'Pipeline incomplete/protocol changed')
    require(p['runner_sha256']==sha(__file__), 'Pipeline runner changed')
    validate_runtime_versions(p['runtime_preflight'],[r['endpoint'] for r in p['runs']])
    require(0 < p['max_seconds'] <= 3000 and p['seconds'] <= p['max_seconds'], 'Pipeline deadline record')
    require(1 <= len(p['runs']) <= 2 and len({r['run'] for r in p['runs']})==len(p['runs']), 'Pipeline run cohort')
    for record in p['runs']:
        run=local_path(base,record['run']); endpoint=authenticate_run(run,root=root)
        require(endpoint==record['endpoint'] and record['status']=='complete', 'Pipeline endpoint changed')
        require(set(record['stages'])==set(STAGES), 'Missing pipeline stage')
        for stage,step in record['stages'].items():
            require(step['status'] in ('complete','skipped_valid'), 'Stage not complete')
            if step['status']=='complete':
                require(step['exit_code']==0 and not step['timed_out'], 'Stage execution failed')
                require(sha(local_path(base,step['log']))==step['log_sha256'], 'Changed execution log')
            require(validate_stage(stage,run,endpoint,root=root)==step['validation'], f'Relocated {stage} artifacts changed')
    return dict(complete=True,pipeline_sha256=sha(base/pipeline_name),models=len(p['runs']),stages=len(p['runs'])*5)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=3000)
    args=parser.parse_args(argv)
    def interrupted(signum, frame): raise InterruptedError(f'Worker received signal {signum}; partial evidence retained')
    signal.signal(signal.SIGTERM,interrupted)
    result=run_pipeline(args.runs,args.out,args.max_seconds)
    print(json.dumps(dict(complete=result['complete'],out=str(args.out),seconds=result['seconds'],error=result.get('error'))),flush=True)
    return 0 if result['complete'] else 1


if __name__=='__main__': sys.exit(main())
