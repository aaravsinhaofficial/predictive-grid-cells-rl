"""Boundary/resume checks on synthetic policies and native-shaped random weights."""
import copy
import json
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
from code.common import build_policy
import local_return_feedback as feedback
import local_return_feedback_run as runner
from test_local_return_feedback import fixture,TinyPolicy
from rl_visual_memory_extend import tree_equal


def roundtrip(net,cfg,tmp_path,seed):
    torch.set_num_threads(2);source=feedback.make_source(cfg,'cpu');opt=feedback.optimizer(net,cfg)
    streams=feedback.TrialStreams(seed);counters=runner.fresh_counters()
    counters['evaluation_records']['baseline']={'synthetic_preflight':True}
    provenance={'synthetic_only':True};initialization={'no_original_policy_loaded':True}
    plan=streams.draw(cfg['n_envs'],feedback.STAGE,cfg)
    metrics,_=feedback.update(net,cfg,opt,source,plan)
    runner.advance_counters(counters,metrics,cfg)
    path=tmp_path/'boundary.pt';torch.save(runner.snapshot(net,opt,streams,cfg,initialization,counters,provenance),path)
    expected_plan=streams.draw(cfg['n_envs'],feedback.STAGE,cfg)
    expected_metrics,expected_rows=feedback.update(net,cfg,opt,source,expected_plan)
    expected_policy=copy.deepcopy(net.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    expected_streams=streams.snapshot();expected_rng=runner.global_rng_snapshot();expected_random=torch.rand(17)
    replacement=copy.deepcopy(net)
    for p in replacement.parameters():p.data.zero_()
    restored_opt=feedback.optimizer(replacement,cfg);restored_streams=feedback.TrialStreams(seed)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    restored=runner.restore(replacement,restored_opt,restored_streams,saved,cfg,provenance)
    assert restored==counters
    actual_plan=restored_streams.draw(cfg['n_envs'],feedback.STAGE,cfg)
    assert tree_equal(expected_plan,actual_plan)
    actual_metrics,actual_rows=feedback.update(replacement,cfg,restored_opt,source,actual_plan)
    assert expected_metrics==actual_metrics and expected_rows==actual_rows
    assert tree_equal(expected_policy,replacement.state_dict()) and tree_equal(expected_opt,restored_opt.state_dict())
    assert tree_equal(expected_streams,restored_streams.snapshot())
    assert tree_equal(expected_rng,runner.global_rng_snapshot()) and torch.equal(expected_random,torch.rand(17))
    assert actual_metrics['replay_error']<3e-4 and actual_metrics['value_replay_error']<3e-4
    return expected_metrics


def test_exact_optimizer_policy_private_global_rng_resume(tmp_path):
    net,cfg,_,_=fixture(741101)
    roundtrip(net,cfg,tmp_path,741102)


@pytest.mark.parametrize('condition',['gc100','pgc33'])
def test_native_cnn_lstm256_shape_exact_resume_with_random_weights(tmp_path,condition):
    """Reads configuration only; never reads a trained policy or study bank."""
    cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    cfg.update(n_envs=1,rollout=160,entropy=8e-5)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(741110)
        net=build_policy(cfg,'cpu')
    assert net.nh==256 and net.lstm.input_size==1287 and net.pi.out_features==6
    roundtrip(net,cfg,tmp_path,741111)


def test_pending_baseline_and_final_must_not_be_skipped():
    counters=runner.fresh_counters();assert runner.pending_evaluation(counters)=='baseline'
    counters['updates']=1
    with pytest.raises(AssertionError,match='before'):runner.pending_evaluation(counters)
    counters['evaluation_records']['baseline']={'present':True}
    assert runner.pending_evaluation(counters) is None
    counters['updates']=1024;assert runner.pending_evaluation(counters)=='final'
    counters['evaluation_records']['final']={'present':True};assert runner.pending_evaluation(counters) is None
    counters['updates']=1023
    with pytest.raises(AssertionError):runner.pending_evaluation(counters)


def test_resume_retains_parent_checkpoint_rolledback_metrics_and_partial_eval(tmp_path):
    parent=tmp_path/'parent';out=tmp_path/'resumed';parent.mkdir();out.mkdir()
    checkpoint=parent/'latest.pt';checkpoint.write_bytes(b'exact restart input')
    raw=b'{"update":1}\n{"update":2}\n';(parent/'metrics.jsonl').write_bytes(raw)
    (parent/'failed_plan.pt').write_bytes(b'exact offending trial tapes')
    (parent/'failed_predraw_streams.pt').write_bytes(b'exact streams before offending draw')
    (parent/'baseline_evaluation').mkdir();(parent/'baseline_evaluation/partial.pt').write_bytes(b'old pending output')
    counters=runner.fresh_counters()
    runner.inherit_resume_artifacts(checkpoint,out,counters)
    assert (out/'resume_parent_evidence/checkpoint.pt').read_bytes()==checkpoint.read_bytes()
    assert (out/'resume_parent_evidence/metrics.jsonl').read_bytes()==raw
    for name in ('failed_plan.pt','failed_predraw_streams.pt'):
        assert (out/'resume_parent_evidence'/name).read_bytes()==(parent/name).read_bytes()
    assert (out/'parent_incomplete_baseline_evaluation/partial.pt').read_bytes()==b'old pending output'
    lineage=json.loads((out/'resume_lineage.json').read_text())
    assert lineage['restored_updates']==0 and lineage['pending_evaluation']=='baseline'
    assert lineage['resume_checkpoint_sha256']==runner.sha(checkpoint)


def test_production_config_rejects_old_entropy_and_architecture():
    cfg=json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text());cfg.update(n_envs=32,rollout=160)
    runner.assert_config(cfg)
    for key,value in [('entropy',.01),('nh',8),('rollout',100),('reward_clip',10)]:
        broken=dict(cfg,**{key:value})
        with pytest.raises(AssertionError):runner.assert_config(broken)


def test_final_boundary_resume_preserves_latest_when_no_learning_remains(tmp_path):
    parent=tmp_path/'parent';out=tmp_path/'resumed';parent.mkdir();out.mkdir()
    checkpoint=parent/'final_before_evaluation.pt';checkpoint.write_bytes(b'exact update1024 boundary')
    (parent/'metrics.jsonl').write_bytes(b'complete raw accepted learning history')
    (parent/'baseline_evaluation').mkdir();(parent/'baseline_evaluation/summary.json').write_bytes(b'baseline result')
    for name in ('baseline_before_evaluation.pt','baseline.pt','baseline_evaluation_integrity.json'):
        (parent/name).write_bytes(name.encode())
    counters=runner.fresh_counters();counters.update(updates=1024,trials=32768)
    counters['evaluation_records']['baseline']={
        'checkpoint_before_sha256':runner.sha(parent/'baseline_before_evaluation.pt'),
        'summary_sha256':runner.sha(parent/'baseline_evaluation/summary.json')}
    assert runner.pending_evaluation(counters)=='final'
    runner.inherit_resume_artifacts(checkpoint,out,counters)
    assert (out/'latest.pt').read_bytes()==checkpoint.read_bytes()
    lineage=json.loads((out/'resume_lineage.json').read_text())
    assert lineage['restored_updates']==1024 and lineage['pending_evaluation']=='final'
    assert any(r['copied']=='latest.pt' and r['sha256']==runner.sha(checkpoint) for r in lineage['copied_artifacts'])


def test_intermediate_resume_does_not_pin_a_later_mutable_latest_role(tmp_path):
    parent=tmp_path/'parent';out=tmp_path/'resumed';parent.mkdir();out.mkdir()
    checkpoint=parent/'latest.pt';checkpoint.write_bytes(b'exact update128 boundary')
    (parent/'metrics.jsonl').write_bytes(b'raw accepted prefix')
    (parent/'baseline_evaluation').mkdir();(parent/'baseline_evaluation/summary.json').write_bytes(b'baseline result')
    for name in ('baseline_before_evaluation.pt','baseline.pt','baseline_evaluation_integrity.json'):
        (parent/name).write_bytes(name.encode())
    counters=runner.fresh_counters();counters.update(updates=128,trials=4096)
    counters['evaluation_records']['baseline']={
        'checkpoint_before_sha256':runner.sha(parent/'baseline_before_evaluation.pt'),
        'summary_sha256':runner.sha(parent/'baseline_evaluation/summary.json')}
    runner.inherit_resume_artifacts(checkpoint,out,counters)
    assert not (out/'latest.pt').exists()
    lineage=json.loads((out/'resume_lineage.json').read_text())
    assert not any(r['copied']=='latest.pt' for r in lineage['copied_artifacts'])
    assert (out/'resume_parent_evidence/checkpoint.pt').read_bytes()==checkpoint.read_bytes()
    # Later periodic saves cannot invalidate any supposedly immutable copied evidence.
    (out/'latest.pt').write_bytes(b'later update256 boundary')
    for item in lineage['copied_artifacts']:assert runner.sha(out/item['copied'])==item['sha256']
