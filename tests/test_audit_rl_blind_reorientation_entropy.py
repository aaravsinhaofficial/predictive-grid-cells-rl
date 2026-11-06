import copy
import json
import pytest
import torch
from scripts import audit_rl_blind_reorientation_entropy as audit


@pytest.fixture(scope='module')
def incoming():
    design=json.loads(audit.PROTOCOL.read_text());design.pop('frozen_at')
    parent_path=audit.ROOT/design['parent_assets']['gc100']['folder']/'final.pt'
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    cfg=dict(parent['config'],entropy=.01)
    design['runtime']=dict(condition='gc100',entropy_arm='explore',entropy_coefficient=.01,device='cuda',torch_version='2.10',protocol_sha256=audit.PROTOCOL_HASH)
    data=dict(policy=parent['policy'],optimizer=parent['optimizer'],runtime_snapshot=parent['runtime_snapshot'],
        recent_scores=parent['recent_scores'],config=cfg,provenance=design,completed=False,
        original_frames=1_000_000_000,parent_extra_frames=40_058_880,extension_frames=0,extra_frames=40_058_880,
        extension_updates=0,total_updates=2156,warmup_updates=200,extension_warmup_updates=0,stage_evaluations_completed=0)
    return data,cfg,design,parent


def test_exact_real_parent_resume_and_genuine_budgets(incoming):
    data,cfg,design,parent=incoming
    result=audit.audit_checkpoint(data,cfg,design,parent,0,0,False)
    assert result['exact_parent_resume'] and result['total_extra_frames']==40_058_880
    assert audit.checkpoint_specs()[-1]==('final.pt',2445,5,True)
    assert audit.PARENT_FRAMES+2445*audit.FRAMES_PER_UPDATE==90_132_480
    assert audit.PARENT_UPDATES+2445==4601


@pytest.mark.parametrize('corrupt',['frames','warmup','policy','optimizer','hidden','rng','clock_plans','recent_scores'])
def test_resume_corruption_is_rejected(incoming,corrupt):
    original,cfg,design,parent=incoming;data=copy.deepcopy(original)
    if corrupt=='frames':data['extra_frames']=40_058_881
    if corrupt=='warmup':data['extension_warmup_updates']=200
    if corrupt=='policy':next(iter(data['policy'].values())).flatten()[0]+=.1
    if corrupt=='optimizer':next(iter(data['optimizer']['state'].values()))['square_avg'].flatten()[0]+=.1
    if corrupt=='hidden':data['runtime_snapshot']['state'][0][0,0]+=.1
    if corrupt=='rng':data['runtime_snapshot']['action_rng'][0]^=1
    if corrupt=='clock_plans':data['runtime_snapshot']['clock']['plans'][0,0,0]=.123456
    if corrupt=='recent_scores':data['recent_scores'].append(-999.)
    with pytest.raises(AssertionError):audit.audit_checkpoint(data,cfg,design,parent,0,0,False)


def test_runtime_manifest_rejects_wrong_arm_and_metadata_rewriting(incoming):
    _,_,worker,_=incoming;design=copy.deepcopy(worker);design.pop('runtime')
    assert audit.check_runtime_design(worker,design,'gc100','explore')==worker
    bad=copy.deepcopy(worker);bad['runtime']['entropy_arm']='original'
    with pytest.raises(AssertionError):audit.check_runtime_design(bad,design,'gc100','explore')
    bad=copy.deepcopy(worker);bad['learning']['total_learning_frames_beyond_1B']=40_058_880
    with pytest.raises(AssertionError):audit.check_runtime_design(bad,design,'gc100','explore')


def test_missing_cloud_uploads_are_pending_read_only(tmp_path):
    result=audit.run(tmp_path,'gc100','original')
    assert result['passed'] and result['status']=='pending' and not result['complete']
    assert not list(tmp_path.iterdir())
