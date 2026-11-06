"""Exploratory adapter corruption tests; no learned-policy or new-bank outcomes."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import pytest
import torch
import goal_cue_memory_original_diagnostic as producer
import audit_goal_cue_memory_original_diagnostic as audit


def test_prospective_protocol_requires_stopped_training_and_every_fixed_control(tmp_path):
    # This reads only already-known qualification/provenance and hashes assets;
    # protocol() neither loads a learned network nor evaluates a memory bank.
    design=producer.protocol();path=tmp_path/'protocol.json'
    path.write_text(json.dumps(dict(design,frozen_utc='synthetic validation only')))
    assert audit.validate_protocol(path)==design
    changes=[lambda d:d['execution'].__setitem__('learning_updates',1),
        lambda d:d['stopped_parent_protocol'].__setitem__('training_remains_prohibited',False),
        lambda d:d['evaluation']['variants'].pop(),
        lambda d:d['interpretation'].__setitem__('no_automatic_neural_nomination',False)]
    for change in changes:
        damaged=copy.deepcopy(design);change(damaged)
        path.write_text(json.dumps(dict(damaged,frozen_utc='synthetic validation only')))
        with pytest.raises(AssertionError):audit.validate_protocol(path)


def artifact_fixture(tmp_path,monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(795901);net=torch.nn.Linear(2,2)
    cfg={'cond':'gc100'};init={'synthetic':True};folder=tmp_path/'output';folder.mkdir()
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    bank={'path':'synthetic.pt','sha256':'synthetic'}
    design={'conditions':['gc100','pgc33'],'evaluation':{'manifest_path':'synthetic.json','manifest_sha256':'synthetic','bank':bank}}
    provenance=dict(phase='original_diagnostic',condition='gc100',original_seed=1,design=design,protocol_sha256=audit.sha(protocol),
        device='cpu',threads=2,parent_training_gate_still_failed=True)
    for name,value in [('protocol.json',provenance),('config.json',cfg),('initialization.json',init)]:
        (folder/name).write_text(json.dumps(value))
    totals={name:{'trials':256,'successes':0}for name in audit.base.VARIANTS}
    report=dict(variant_order=list(audit.base.VARIANTS),all_variants_valid=True,
        variants={name:{'totals':value}for name,value in totals.items()},gates={'synthetic':True})
    calls=[]
    def evaluate(net,cfg,out,**kwargs):
        out.mkdir();(out/'summary.json').write_text(json.dumps(report));return report
    def checked(folder,incoming,plan,reference,variants):
        calls.append((reference,variants));assert variants==audit.base.VARIANTS
        return dict(complete=True,valid=True,variants=totals,gates={'synthetic':True})
    monkeypatch.setattr(producer.evaluator,'evaluate_bank',evaluate)
    producer.run(net,cfg,init,provenance,folder,manifest=tmp_path/'unused')
    monkeypatch.setattr(audit,'validate_protocol',lambda p:design)
    monkeypatch.setattr(audit.base,'audit_initial',lambda initial,condition:({'policy':net.state_dict(),'config':cfg,'initialization':init},{'synthetic':True}))
    monkeypatch.setattr(audit.banks,'load_bank',lambda manifest,name:({'synthetic':True},bank,{}))
    monkeypatch.setattr(audit.base,'audit_evaluation',checked)
    return folder,protocol,calls


def test_honest_exploratory_snapshot_and_frozen_numerical_audit_reuse(tmp_path,monkeypatch):
    folder,protocol,calls=artifact_fixture(tmp_path,monkeypatch)
    result=audit.audit(folder,'gc100',protocol)
    assert result['passed']and result['complete'],result['failures']
    assert result['parent_training_gate_still_failed']and result['no_automatic_neural_nomination']
    assert len(calls)==1 and calls[0][0]['bank_name']=='memory'and calls[0][1]==audit.base.VARIANTS
    damaged=torch.load(folder/'after.pt',weights_only=True);damaged['policy']['bias']+=.1
    torch.save(damaged,folder/'after.pt')
    result=audit.audit(folder,'gc100',protocol)
    assert not result['passed']and 'changed policy'in result['failures'][0]


@pytest.mark.parametrize('corruption',['optimizer','training_gate','failure_evidence'])
def test_no_hidden_training_role_or_stale_success(tmp_path,monkeypatch,corruption):
    folder,protocol,_=artifact_fixture(tmp_path,monkeypatch)
    if corruption=='optimizer':
        saved=torch.load(folder/'incoming.pt',weights_only=True);saved['optimizer']={};torch.save(saved,folder/'incoming.pt')
    elif corruption=='training_gate':
        path=folder/'summary.json';saved=json.loads(path.read_text());saved['parent_training_gate_still_failed']=False;path.write_text(json.dumps(saved))
    else:(folder/'incomplete.json').write_text('{"synthetic_failed_attempt":true}')
    result=audit.audit(folder,'gc100',protocol)
    assert not result['passed']and not result['complete']
