"""Small corruption controls for the independent saved-data audit."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_native_turn_distillation_outputs as audit


def update(arm='clean'):
    return dict(condition='gc100', arm=arm, update=1, stage=1, extra_frames=5760,
        loss=.75, grad_norm_before_clip=2., parameter_update_l2=.01,
        replay_max_error=0., teacher_replay_max_error=0.,
        choice_count=12, control_count=1400, forced_count=28,
        phase_counts=[1000,400,7,7,7,7,12], choice_kl=1., control_kl=.5,
        no_choice_fallback=False, counts=dict(noise_scheduled=8, noise_applied=8 if arm=='noise' else 0),
        clipped_hidden_coordinates=0, noise_realized_rms=[0.,0.], noise_realized_mean=[0.,0.])


def test_weighted_loss_and_phase_partition_are_checked():
    audit.audit_update(update(), 'gc100', 'clean', 1)
    for change in ({'loss':.6}, {'phase_counts':[999,400,7,7,7,7,13]}, {'extra_frames':1}):
        bad={**update(), **change}
        with pytest.raises(AssertionError): audit.audit_update(bad,'gc100','clean',1)


def test_noise_masks_and_zero_choice_fallback_cannot_be_hidden():
    row=update('noise');audit.audit_update(row,'gc100','noise',1)
    row['counts']['noise_applied']=7
    with pytest.raises(AssertionError):audit.audit_update(row,'gc100','noise',1)
    row=update();row.update(choice_count=0,control_count=1412,phase_counts=[1012,400,7,7,7,7,0],
        choice_kl=None,loss=.5,no_choice_fallback=True)
    audit.audit_update(row,'gc100','clean',1)
    row['no_choice_fallback']=False
    with pytest.raises(AssertionError):audit.audit_update(row,'gc100','clean',1)


def test_partial_cohort_and_nonzero_exit_are_rejected_before_array_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'BASE',tmp_path)
    for c in audit.CONDITIONS:
        for arm in audit.ARMS:
            folder=tmp_path/c/arm;folder.mkdir(parents=True)
            record=dict(complete=True,condition=c,arm=arm,updates=180,protocol_sha256=audit.EXPECTED)
            (folder/'summary.json').write_text(json.dumps(record))
            (tmp_path/f'{c}_{arm}_execution.json').write_text(json.dumps({**record,'exit_code':0}))
    (tmp_path/'report').mkdir();(tmp_path/'report/summary.json').write_text(json.dumps(dict(complete=True,protocol_sha256=audit.EXPECTED)))
    assert audit.ensure_ready()['complete']
    path=tmp_path/'pgc33_noise_execution.json';d=json.loads(path.read_text());d['exit_code']=1;path.write_text(json.dumps(d))
    with pytest.raises(AssertionError):audit.ensure_ready()
    path.unlink()
    with pytest.raises(FileNotFoundError):audit.ensure_ready()
