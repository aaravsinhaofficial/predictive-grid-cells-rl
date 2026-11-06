"""Sealed-bank timing, leakage and invalid-denominator certificate checks."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import pytest
import torch
from torch import nn
from code.nets import PolicyNet
import local_goal_return_training as task
import local_goal_return_training_evaluate as evaluation
import local_goal_return_training_certificate as certificate
from certify_local_goal_return import all_action_path_bound


def artifacts():
    cfg = json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    bank, _, _ = evaluation.load_bank('final_hard')
    plan = evaluation.context_plan(bank, 0)
    schedule = task.schedule(plan, 'cpu'); start = int(schedule['free'][0])
    actions = schedule['forced_action'].clone()
    previous = torch.cat((torch.full((1,16), -1), actions[:-1]))
    goal = torch.zeros(16,512)
    x = torch.zeros(16,1287); x[:,1280] = 1
    first = {i: dict(input=x[i].clone(), state=torch.zeros(512),
                    position=plan['endpoint'][i].clone(), heading=plan['heading'][i].clone(),
                    motors=torch.zeros(3)) for i in range(16)}
    roll = dict(goal=goal, phase=list(schedule['phase']), action=list(actions), pa=list(previous),
                pr=[torch.zeros(16) for _ in range(160)], active=list(schedule['phase'] != task.PAD))
    return cfg, plan, roll, first, start


def test_certificate_uses_actual_free_onset_tape_and_keeps_all_pairs():
    cfg, plan, roll, first, start = artifacts()
    result = certificate.certify_context(plan,cfg,roll,first,('blind_reset_after_glimpse',))
    assert result['artifact_schema_valid'] and result['certified_all_pairs']
    expected = all_action_path_bound(plan['motor'][start:start+30].numpy(),cfg['v_max'],.03,
                                      cfg['action_repeat']/60,.25,np.zeros((16,2)))
    wrong = all_action_path_bound(plan['motor'][:30].numpy(),cfg['v_max'],.03,
                                   cfg['action_repeat']/60,.25,np.zeros((16,2)))
    assert result['free_start_update'] == start and start > 30
    assert result['pairs'][0]['all_action_path_upper_bound_m'] == float(expected[0])
    assert abs(float(expected[0]-wrong[0])) > 1e-6
    assert len(result['pairs']) == 8 and result['common_information_success_upper_bound'] == .5


def test_input_leak_and_missing_contexts_raise_the_conservative_bound():
    cfg, plan, roll, first, _ = artifacts()
    first[0]['input'][256] = 1.
    result = certificate.certify_context(plan,cfg,roll,first,('blind_reset_after_glimpse',))
    assert not result['certified_all_pairs'] and result['artifact_schema_valid']
    assert 'terminal_input_byte_identical' in result['pairs'][0]['failures']
    assert sum(p['certified'] for p in result['pairs']) == 7
    summary = certificate.aggregate_certificates([result])
    assert summary['heading_trials'] == 256 and summary['equally_weighted_pairs'] == 128
    assert summary['certified_pairs'] == 7 and summary['uncertified_pairs'] == 121
    assert summary['common_information_success_upper_bound'] == (7*.5+121)/128
    with pytest.raises(AssertionError, match='Duplicate'):
        certificate.aggregate_certificates([result,result])


def test_wrong_sealed_plan_rejected_but_missing_snapshot_retained():
    cfg, plan, roll, first, _ = artifacts()
    corrupted = copy.deepcopy(plan); corrupted['motor'][0,0,0] += 1.
    with pytest.raises(AssertionError, match='Sealed plan changed'):
        certificate.certify_context(corrupted,cfg,roll,first,('blind_reset_after_glimpse',))
    del first[0]
    result = certificate.certify_context(plan,cfg,roll,first,('blind_reset_after_glimpse',))
    assert not result['artifact_schema_valid'] and len(result['pairs']) == 8
    assert result['common_information_success_upper_bound'] == 1.


class SmallEncoderPolicy(nn.Module):
    """Untrained test network; no released checkpoint or baseline outcome."""
    def __init__(self):
        super().__init__(); self.nh=256; self.n_actions=6; self.extra_in=0
        self.encoder=nn.Linear(3,256); self.lstm=nn.LSTMCell(1287,256)
        self.pi=nn.Linear(256,6); self.v=nn.Linear(256,1)
    def embed(self,rgb): return self.encoder(rgb.mean((2,3)))
    zero_state=PolicyNet.zero_state
    _concat=PolicyNet._concat


def test_real_collector_erased_snapshots_satisfy_the_same_certificate():
    torch.set_num_threads(2)
    cfg, plan, _, _, _ = artifacts()
    net=SmallEncoderPolicy();source=task.make_source(cfg,'cpu')
    roll, _, audit, first = task.collect(net,cfg,source,plan,
        variants=('blind_reset_after_glimpse',),sigma=0.,retain_inputs=False)
    assert audit['forced_hits'] == 0
    result=certificate.certify_context(plan,cfg,roll,first,('blind_reset_after_glimpse',))
    assert result['artifact_schema_valid'] and result['certified_all_pairs'],result
    assert all(p['conditions']['erased_actions_equal_before_either_hit'] for p in result['pairs'])
