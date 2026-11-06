"""Metamorphic corruption tests of the read-only completed-baseline auditor."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import pytest
import torch
import audit_local_goal_return_baseline as audit


def artifacts():
    folder=ROOT/'results/local_goal_return_baseline/gc100'
    if not (folder/'summary.json').exists():pytest.skip('Requires published completed baseline artifacts')
    saved=torch.load(folder/'initialized_policy.pt',map_location='cpu',weights_only=True)
    raw=torch.load(folder/'near_baseline/context0/raw.pt',map_location='cpu',weights_only=True)
    record=audit.read(folder/'near_baseline/context0/result.json')
    plan,_,_=audit.producer.load_bank('near_baseline')
    return saved['config'],audit.producer.context_plan(plan,0),raw,record['rows']


def test_auditor_rejects_setup_or_forced_reward_in_learning():
    cfg,plan,raw,rows=artifacts();raw['raw_reward'][0,0]=10.
    with pytest.raises(AssertionError):
        audit.verify_trajectory(plan,cfg,raw,rows,raw['first_free'],list(audit.task.VARIANTS))


def test_auditor_rejects_wrong_free_motor_tape():
    cfg,plan,raw,rows=artifacts()
    start=int(audit.task.schedule(plan,'cpu')['free'][0]);plan['motor'][start,0,0]+=10.
    with pytest.raises(AssertionError):
        audit.verify_trajectory(plan,cfg,raw,rows,raw['first_free'],list(audit.task.VARIANTS))


def test_auditor_checks_grid_states_not_just_actions():
    _,_,raw,_=artifacts()
    raw['first_free'][16]['state'][0]+=1.
    with pytest.raises(AssertionError):audit.grid_shams(raw,list(audit.task.VARIANTS))
