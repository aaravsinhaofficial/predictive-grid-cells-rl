import copy
import json
from pathlib import Path
import pytest
from scripts import lstm_reorientation_entropy_mechanism as adapter


def metadata(condition='pgc33',arm='explore'):
    design=adapter.training_design()
    design['runtime']=dict(condition=condition,entropy_arm=arm,entropy_coefficient=adapter.ARMS[arm],
        device='cuda',torch_version='test-only',protocol_sha256=adapter.TRAINING_HASH)
    cfg=json.loads((adapter.ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    cfg.update(rollout=160,entropy=adapter.ARMS[arm])
    return dict(provenance=design,completed=True,original_frames=1_000_000_000,
        parent_extra_frames=40_058_880,extension_frames=50_073_600,extra_frames=90_132_480,
        extension_updates=2445,total_updates=4601,warmup_updates=200,extension_warmup_updates=0,
        stage_evaluations_completed=5,config=cfg,optimizer={'state':{0:{'step':4601}}},
        runtime_snapshot={'noise_replays':4601,'noise_max_replay_error':1e-5})


def test_genuine90m_metadata_validation_is_read_only():
    data=metadata();before=copy.deepcopy(data)
    cfg=adapter.validate_final(data,'pgc33','explore')
    assert cfg==data['config'] and data==before
    assert data['extra_frames']==90_132_480


@pytest.mark.parametrize('key,value', [('extra_frames',40_058_880),('completed',False),
    ('total_updates',2156),('stage_evaluations_completed',4),('extension_warmup_updates',1)])
def test_rejects_old_incomplete_or_restarted_endpoint(key,value):
    data=metadata();data[key]=value
    with pytest.raises(AssertionError):adapter.validate_final(data,'pgc33','explore')


def test_wrong_arm_extra_runtime_fields_and_changed_inputs_are_rejected():
    for change in ('arm','extra_runtime','goal_config'):
        data=metadata()
        if change=='arm':data['provenance']['runtime']['entropy_arm']='original'
        elif change=='extra_runtime':data['provenance']['runtime']['new_unreviewed_field']=True
        else:data['config']['goal_radius']=.8
        with pytest.raises(AssertionError):adapter.validate_final(data,'pgc33','explore')


def test_routing_uses_explicit_retry_and_artifact_subdir(tmp_path):
    routing=tmp_path/'routing.json'
    row=dict(condition='pgc33',arm='explore',name='ring-example-retry-03',artifact_subdir='experiment')
    routing.write_text(json.dumps(dict(frozen_protocol_sha256=adapter.TRAINING_HASH,workers=[row])))
    path=adapter.resolve_checkpoint('pgc33','explore',routing)
    assert path==adapter.ROOT/'results/ring_cloud_artifacts/ring-example-retry-03/experiment/final.pt'
    row['artifact_subdir']='../other'
    routing.write_text(json.dumps(dict(frozen_protocol_sha256=adapter.TRAINING_HASH,workers=[row])))
    with pytest.raises(AssertionError):adapter.resolve_checkpoint('pgc33','explore',routing)


def test_inherited_frozen_numerical_sources_match_original_manifests():
    for name in ('implementation_protocol.json','causal_navigation/protocol.json'):
        manifest=json.loads((adapter.ROOT/'results/lstm_reorientation_mechanism'/name).read_text())
        for path,digest in manifest['source_sha256'].items():assert adapter.sha(adapter.ROOT/path)==digest
