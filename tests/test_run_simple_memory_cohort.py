"""Mocked supervisor tests: no cloud requests, model execution, or billable jobs."""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shlex
from unittest import mock

import pytest
import torch

from scripts import run_simple_memory_cohort as supervisor


@pytest.fixture(autouse=True)
def forbid_unmocked_cloud_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected cloud call in a mocked test')
    monkeypatch.setattr(supervisor.cloud, 'run', forbidden)


def running_vm():
    return {'id': '123', 'status': 'RUNNING', 'labels': {'study': supervisor.cloud.PREFIX}}


def cohort():
    jobs = []
    for seed in range(1, 5):
        for condition in ('gc100', 'pgc33'):
            for gap in (0, 60):
                name = f'ring-20261105-simplemem-{condition}-g{gap}-s{seed}'
                parent = f'results/ring_simple_memory/{condition}/gap{gap}'
                train = ['python', '-m', 'code.train_memory', '--cond', condition, '--seed', str(seed),
                         '--code_seed', str(seed), '--task_max_gap', str(gap), '--frames', '1000000000',
                         '--out', parent+f'/seed{seed}', '--protocol', 'docs/research/simple_memory_training_protocol.json']
                argv = ['python', 'scripts/ring_cloud.py', 'plan', '--name', name, '--machine', 'g2-standard-4',
                        '--zone', 'us-central1-c', '--hours', '24', '--output-dir', parent,
                        '--command', shlex.join(train), '--include-only', '--include', 'code/train.py']
                jobs.append(dict(name=name, condition=condition, seed=seed, task_max_gap=gap, train_argv=train,
                                 plan_argv=argv, worker_output_dir=parent, trainer_output_dir=parent+f'/seed{seed}'))
    return dict(jobs=jobs, design={'first_jobs': [j['name'] for j in jobs[:4]]},
                package={'current_manifest': [{'path': 'code/train.py', 'sha256': 'source'}]},
                hardware={'minimum_training_fps': 12000})


def record(job):
    return dict(name=job['name'], machine='g2-standard-4', zone='us-central1-c', command=job['train_argv'],
        output_dir=job['worker_output_dir'], package_manifest=[{'path': 'code/train.py', 'sha256': 'source'}],
        package_manifest_sha256='package', instance_id='123', reserved_usd=30.3125,
        provider_deadline_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=24)).isoformat(),
        gcs_prefix='gs://private-test/'+job['name'])


def metric(frames=2_100_000, fps=13000):
    return dict(frames=frames, updates=200, fps=fps, loss=1., grad_norm=2.,
                replay_logit_max_error=.000001, replay_value_max_error=.000001,
                avg_return_100='THIS FIELD MUST NOT AFFECT THE GATE')


def test_technical_metrics_ignore_rewards_and_partial_last_line():
    text = 'partial suffix\n'+json.dumps(metric())+'\n{"frames":'
    value = supervisor.technical_row(text)
    assert value['fps'] == 13000 and 'avg_return_100' not in value
    bad = metric(); bad['grad_norm'] = float('nan')
    with pytest.raises(AssertionError, match='Nonfinite'):
        supervisor.technical_row(json.dumps(bad)+'\n')


def test_auth_failure_is_not_treated_as_missing_object():
    with mock.patch.object(supervisor.cloud, 'run', return_value=argparse.Namespace(returncode=1, stderr='Reauthentication failed', stdout='')):
        with pytest.raises(RuntimeError, match='Cloud read failed'):
            supervisor.cat('gs://private-test/owned/DONE.json')
    with mock.patch.object(supervisor.cloud, 'run', return_value=argparse.Namespace(returncode=1, stderr='No URLs matched', stdout='')):
        assert supervisor.cat('gs://private-test/owned/DONE.json') is None


def test_first_four_gate_restart_and_eight_worker_cap(tmp_path):
    plan = cohort(); state = {'jobs': {}}; ledger = tmp_path/'ledger.json'
    old = dict(name='unrelated-old-run', reserved_usd=1., actual_upper_bound_usd=.1)
    ledger.write_text(json.dumps({'runs': [old], 'ancillary_reserve_usd': 25})); launched = []; rows = {}
    def launch(args):
        job = next(j for j in plan['jobs'] if j['name'] == args.name)
        data = json.loads(ledger.read_text()); data['runs'].append(record(job))
        ledger.write_text(json.dumps(data)); launched.append(args.name)
    def cat(uri, tail=False):
        if not tail: return None
        for name, row in rows.items():
            if f'/{name}/' in uri: return json.dumps(row)+'\n'
        return None
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch', side_effect=launch), \
         mock.patch.object(supervisor.cloud, 'plan', return_value={'package_manifest': plan['package']['current_manifest'], 'reserved_usd': 30.3125}), \
         mock.patch.object(supervisor.cloud, 'describe', return_value=running_vm()), \
         mock.patch.object(supervisor, 'cat', side_effect=cat):
        supervisor.cycle(plan, state, 'protocol')
        assert launched == plan['design']['first_jobs']
        supervisor.cycle(plan, state, 'protocol')
        assert len(launched) == 4  # Restart/adoption never launches an existing name again.
        rows.update({name: metric() for name in plan['design']['first_jobs'][:3]})
        supervisor.cycle(plan, state, 'protocol'); assert len(launched) == 4
        rows[plan['design']['first_jobs'][3]] = metric()
        supervisor.cycle(plan, state, 'protocol'); assert len(launched) == 8
        supervisor.cycle(plan, state, 'protocol'); assert len(launched) == 8
    assert json.loads(ledger.read_text())['runs'][0] == old


def test_slow_first_worker_halts_before_more_launches(tmp_path):
    plan = cohort(); ledger = tmp_path/'ledger.json'
    ledger.write_text(json.dumps({'runs': [record(j) for j in plan['jobs'][:4]]}))
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch') as launch, \
         mock.patch.object(supervisor.cloud, 'describe', return_value=running_vm()), \
         mock.patch.object(supervisor, 'cat', side_effect=lambda uri, tail=False: json.dumps(metric(fps=11000))+'\n' if tail else None):
        with pytest.raises(AssertionError, match='throughput gate failed'):
            supervisor.cycle(plan, {'jobs': {}}, 'protocol')
        launch.assert_not_called()


def test_promoted_worker_can_slow_when_original_deadline_still_fits(tmp_path):
    plan = cohort(); ledger = tmp_path/'ledger.json'; job = plan['jobs'][0]
    data = {'runs': [record(j) for j in plan['jobs'][:4]]}; ledger.write_text(json.dumps(data))
    state = {'jobs': {job['name']: {'technical_gate_passed': True}}}
    row = metric(frames=400_000_000, fps=11000)
    def cat(uri, tail=False):
        return json.dumps(row)+'\n' if tail and '/'+job['name']+'/' in uri else None
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch') as launch, \
         mock.patch.object(supervisor.cloud, 'describe', return_value=running_vm()), mock.patch.object(supervisor, 'cat', side_effect=cat):
        supervisor.cycle(plan, state, 'protocol')
        launch.assert_not_called()
    check = state['jobs'][job['name']]['deadline_feasibility']
    assert check['margin_seconds'] > 0 and check['completion_sync_reserve_seconds'] == 600
    assert state['jobs'][job['name']]['technical_gate_passed']
    assert check['provider_deadline_utc'] == data['runs'][0]['provider_deadline_utc']
    assert json.loads(ledger.read_text()) == data


def test_promoted_worker_that_cannot_fit_deadline_halts_without_launch(tmp_path):
    plan = cohort(); ledger = tmp_path/'ledger.json'; job = plan['jobs'][0]
    records = [record(j) for j in plan['jobs'][:4]]
    records[0]['provider_deadline_utc'] = (dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1)).isoformat()
    ledger.write_text(json.dumps({'runs': records}))
    state = {'jobs': {job['name']: {'technical_gate_passed': True}}}
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch') as launch, \
         mock.patch.object(supervisor.cloud, 'describe', return_value=running_vm()), \
         mock.patch.object(supervisor, 'cat', side_effect=lambda uri, tail=False: json.dumps(metric(frames=400_000_000))+'\n' if tail else None):
        with pytest.raises(AssertionError, match='original provider deadline'):
            supervisor.cycle(plan, state, 'protocol')
        launch.assert_not_called()
    assert json.loads(ledger.read_text()) == {'runs': records}


@pytest.mark.parametrize('vm', [None, dict(running_vm(), status='TERMINATED'), dict(running_vm(), id='456')])
def test_promoted_worker_still_requires_exact_running_instance(tmp_path, vm):
    plan = cohort(); ledger = tmp_path/'ledger.json'; job = plan['jobs'][0]
    ledger.write_text(json.dumps({'runs': [record(j) for j in plan['jobs'][:4]]}))
    state = {'jobs': {job['name']: {'technical_gate_passed': True}}}
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch') as launch, \
         mock.patch.object(supervisor.cloud, 'describe', return_value=vm), \
         mock.patch.object(supervisor, 'cat', side_effect=lambda uri, tail=False: json.dumps(metric(frames=400_000_000, fps=11000))+'\n' if tail else None):
        with pytest.raises((AssertionError, RuntimeError)):
            supervisor.cycle(plan, state, 'protocol')
        launch.assert_not_called()


def test_projection_uses_original_warmup_and_fixed_completion_reserve():
    plan = cohort(); job = plan['jobs'][0]; rec = record(job)
    now = dt.datetime.fromisoformat(rec['provider_deadline_utc']).timestamp()-3600
    parse = lambda row: supervisor.technical_row(json.dumps(row)+'\n')
    early = supervisor.technical_viability(plan, job, rec, {}, parse(metric(frames=12800, fps=100)), now=now)
    assert not early['projection_enforced'] and early['throughput_warmup_frames'] == 2_000_000
    assert early['margin_seconds'] < 0
    # 30M / 10K FPS takes 3,000 seconds, leaving exactly the fixed 600-second reserve.
    check = supervisor.technical_viability(plan, job, rec, {'technical_gate_passed': True},
                                          parse(metric(frames=970_000_000, fps=10000)), now=now)
    assert check['margin_seconds'] == 0 and check['projection_enforced']
    with pytest.raises(AssertionError, match='original provider deadline'):
        supervisor.technical_viability(plan, job, rec, {'technical_gate_passed': True},
                                       parse(metric(frames=970_000_000, fps=10000)), now=now+1)
    bad = parse(metric()); bad['loss'] = float('nan')
    with pytest.raises(AssertionError, match='Nonfinite'):
        supervisor.technical_viability(plan, job, rec, {}, bad, now=now)


@pytest.mark.parametrize('vm', [None, dict(running_vm(), status='TERMINATED'), dict(running_vm(), id='456')])
def test_stale_gcs_metrics_cannot_pass_gate(tmp_path, vm):
    plan = cohort(); ledger = tmp_path/'ledger.json'
    ledger.write_text(json.dumps({'runs': [record(j) for j in plan['jobs'][:4]]}))
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch') as launch, \
         mock.patch.object(supervisor.cloud, 'describe', return_value=vm), \
         mock.patch.object(supervisor, 'cat', side_effect=lambda uri, tail=False: json.dumps(metric())+'\n' if tail else None):
        with pytest.raises((AssertionError, RuntimeError)):
            supervisor.cycle(plan, {'jobs': {}}, 'protocol')
        launch.assert_not_called()


def test_wrong_prepared_identity_is_rejected():
    job = cohort()['jobs'][0]; wrong = copy.deepcopy(job)
    wrong['condition'] = 'pgc33'
    with pytest.raises(AssertionError): supervisor.launch_args(wrong)
    changed = record(job); changed['command'] += ['--resume']
    with pytest.raises(AssertionError): supervisor.matching_record(job, changed, {'code/train.py': 'source'})


def test_unbound_or_changed_instance_requires_explicit_reconciliation():
    job = cohort()['jobs'][0]; saved = record(job); saved.pop('instance_id')
    with pytest.raises(AssertionError, match='Unbound instance ID'):
        supervisor.matching_record(job, saved, {'code/train.py': 'source'})
    with pytest.raises(AssertionError, match='Bound instance ID changed'):
        supervisor.matching_record(job, record(job), {'code/train.py': 'source'}, instance_id='456')


@pytest.mark.parametrize('flag,replacement', [('--out', 'results/ring_other/seed1'), ('--protocol', 'docs/research/other.json')])
def test_out_and_protocol_are_bound_to_frozen_job(flag, replacement):
    job = cohort()['jobs'][0]
    job['train_argv'][job['train_argv'].index(flag)+1] = replacement
    job['plan_argv'][job['plan_argv'].index('--command')+1] = shlex.join(job['train_argv'])
    with pytest.raises(AssertionError): supervisor.launch_args(job)


def test_absolute_protocol_path_is_rejected_even_if_it_resolves_locally():
    job = cohort()['jobs'][0]
    job['train_argv'][job['train_argv'].index('--protocol')+1] = str(supervisor.PROTOCOL)
    job['plan_argv'][job['plan_argv'].index('--command')+1] = shlex.join(job['train_argv'])
    with pytest.raises(AssertionError, match='portable relative paths'): supervisor.launch_args(job)


@pytest.mark.parametrize('remote_metric,local_metric', [(True, False), (False, True), (False, False)])
def test_done_artifact_is_processed_after_provider_deletion_without_fps_gate(tmp_path, remote_metric, local_metric):
    plan = cohort(); ledger = tmp_path/'ledger.json'; job = plan['jobs'][0]
    ledger.write_text(json.dumps({'runs': [record(j) for j in plan['jobs'][:4]]}))
    folder = tmp_path/'collected'; folder.mkdir()
    if local_metric: (folder/'metrics_000.jsonl').write_text(json.dumps(metric())+'\n')
    def cat(uri, tail=False):
        if '/'+job['name']+'/' not in uri: return None
        assert not tail, 'Validated completion must not depend on later metrics reads'
        return '{"exit_code":0}'
    state = {'jobs': {}}
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'describe', return_value=None) as describe, \
         mock.patch.object(supervisor.cloud, 'launch') as launch, mock.patch.object(supervisor, 'cat', side_effect=cat), \
         mock.patch.object(supervisor, 'finish', return_value={'complete': True, 'summary_path': str(folder/'summary.json')}) as finish:
        supervisor.cycle(plan, state, 'protocol')
        assert state['jobs'][job['name']]['complete'] and state['jobs'][job['name']]['technical_gate_passed']
        assert state['jobs'][job['name']]['technical_gate_basis'] == 'validated_1B_completion'
        finish.assert_called_once(); describe.assert_not_called(); launch.assert_not_called()


def test_full_cohort_budget_refuses_even_when_one_launch_fits(tmp_path):
    plan = cohort(); ledger = tmp_path/'ledger.json'
    data = {'ancillary_reserve_usd': 25, 'runs': [dict(name='old', reserved_usd=300., actual_upper_bound_usd=300.)]}
    ledger.write_text(json.dumps(data))
    assert supervisor.cloud.committed(data) + 30.3125 < 600
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'launch') as launch, \
         mock.patch.object(supervisor.cloud, 'plan', return_value={'reserved_usd': 30.3125}):
        with pytest.raises(AssertionError, match='full-cohort reservations exceed'):
            supervisor.cycle(plan, {'jobs': {}}, 'protocol')
        launch.assert_not_called()
    assert json.loads(ledger.read_text()) == data


@pytest.mark.parametrize('frames,runtime_frames,wrong_config,accepted', [
    (1_000_000_000, 1_000_000_000, False, True), (999_987_200, 999_987_200, False, False),
    (1_000_000_000, 999_987_200, False, False), (1_000_000_000, 1_000_000_000, True, False)])
def test_terminal_requires_exact_billion_and_collects_before_delete(tmp_path, frames, runtime_frames, wrong_config, accepted):
    plan = cohort(); job = plan['jobs'][0]; r = record(job)
    ledger = tmp_path/'ledger.json'
    ledger.write_text(json.dumps(dict(budget_usd=600., reserved_rate_usd_per_hour=1.25, runs=[r])))
    folder = tmp_path/'artifacts'/'seed1'; folder.mkdir(parents=True)
    config = dict(cond='pgc33' if wrong_config else job['condition'], seed=1, code_seed=1, task_max_gap=0, frames=1_000_000_000)
    (folder/'config.json').write_text(json.dumps(config))
    saved = dict(frames=frames, updates=78125, config=config, provenance={'protocol_sha256': 'protocol'},
                 runtime={'counters': {'frames': runtime_frames, 'updates': 78125, 'decisions': 250_000_000}})
    torch.save(saved, folder/'final.pt')  # Tiny counter-only fixture; no model is instantiated.
    summary = dict(complete=True, frames=frames, updates=78125, decisions=250_000_000,
                   provenance=saved['provenance'], final_sha256=supervisor.sha(folder/'final.pt'))
    (folder/'summary.json').write_text(json.dumps(summary)); events = []
    def collect(rec): events.append('collect'); rec['local_artifacts'] = str(folder.parent)
    def refresh(rec): rec.update(status='DELETED', actual_upper_bound_usd=.5)
    vm = {'id': '123', 'labels': {'study': supervisor.cloud.PREFIX}}
    with mock.patch.object(supervisor.cloud, 'LEDGER', ledger), mock.patch.object(supervisor.cloud, 'describe', return_value=vm), \
         mock.patch.object(supervisor.cloud, 'collect', side_effect=collect), mock.patch.object(supervisor.cloud, 'refresh', side_effect=refresh), \
         mock.patch.object(supervisor.cloud, 'run', side_effect=lambda *a, **k: events.append('delete')):
        if accepted:
            assert supervisor.finish(job, {'exit_code': 0, 'package_sha256': 'package'}, 'protocol')['complete']
        else:
            with pytest.raises(RuntimeError, match='terminal validation failed'):
                supervisor.finish(job, {'exit_code': 0, 'package_sha256': 'package'}, 'protocol')
    assert events == ['collect', 'delete']
    assert json.loads(ledger.read_text())['runs'][0]['status'] == 'DELETED'


def test_draft_refusal_makes_no_cloud_requests(tmp_path):
    p = tmp_path/'plan.json'; q = tmp_path/'protocol.json'
    p.write_text(json.dumps({'status': 'draft'})); q.write_text(json.dumps({'status': 'draft'}))
    with mock.patch.object(supervisor.cloud, 'run') as run:
        with pytest.raises(AssertionError, match='must be frozen'): supervisor.validate(p, q)
        run.assert_not_called()
