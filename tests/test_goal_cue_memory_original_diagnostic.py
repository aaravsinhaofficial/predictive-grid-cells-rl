"""New adapter tests only; all policy/outcome objects here are synthetic."""
from pathlib import Path
import json
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import goal_cue_memory_original_diagnostic as run


def fixture(tmp_path,monkeypatch,mutation=None):
    net=torch.nn.Linear(2,2); cfg={'cond':'synthetic'}; provenance={'protocol_sha256':'synthetic'}
    calls=[]
    def fake(net,cfg,out,**kwargs):
        calls.append(kwargs); out.mkdir()
        report=dict(variant_order=list(run.evaluator.FULL_VARIANTS),all_variants_valid=True,
            variants={name:{'totals':{'trials':256,'successes':0}}for name in run.evaluator.FULL_VARIANTS},gates={'synthetic':True})
        (out/'summary.json').write_text(json.dumps(report))
        if mutation=='weights':
            with torch.no_grad():net.weight.add_(1)
        elif mutation=='rng':torch.rand(1)
        elif mutation=='mode':net.eval()
        return report
    monkeypatch.setattr(run.evaluator,'evaluate_bank',fake)
    return net,cfg,provenance,calls


def test_all_eleven_fixed_variants_and_honest_snapshot_without_optimizer(tmp_path,monkeypatch):
    net,cfg,provenance,calls=fixture(tmp_path,monkeypatch)
    report=run.run(net,cfg,{'synthetic':True},provenance,tmp_path,manifest=tmp_path/'unused')
    assert report['complete'] and report['parent_training_gate_still_failed']
    assert report['no_optimizer_created'] and report['no_optimization'] and report['no_automatic_neural_nomination']
    assert calls==[dict(bank_name='memory',manifest=tmp_path/'unused',device='cpu',variants=run.evaluator.FULL_VARIANTS)]
    before=torch.load(tmp_path/'incoming.pt',weights_only=True)
    after=torch.load(tmp_path/'after.pt',weights_only=True)
    assert run.tree_equal(before,after) and before['phase']=='original_diagnostic'
    assert 'optimizer'not in before and before['model_training']


@pytest.mark.parametrize('mutation',['weights','rng','mode'])
def test_changed_state_is_rejected_and_failed_attempt_preserved(tmp_path,monkeypatch,mutation):
    net,cfg,provenance,calls=fixture(tmp_path,monkeypatch,mutation)
    with pytest.raises(AssertionError,match='changed original'):
        run.run(net,cfg,{},provenance,tmp_path,manifest=tmp_path/'unused')
    assert(tmp_path/'incoming.pt').exists() and(tmp_path/'evaluation/summary.json').exists()
    assert not(tmp_path/'summary.json').exists() and not(tmp_path/'after.pt').exists()
    failure=json.loads((tmp_path/'incomplete.json').read_text())
    assert not failure['complete'] and failure['no_automatic_retry']
