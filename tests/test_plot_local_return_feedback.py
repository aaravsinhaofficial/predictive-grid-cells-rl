"""Reporting arithmetic and incomplete-arm coverage; synthetic values only."""
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import plot_local_return_feedback as plot
from test_audit_local_return_feedback import valid_metric


def endpoint(successes=100,invalid=0):
    rows=[]
    for i in range(256):
        row=dict(context=i//16,heading_index=i%16,variant='feedback_native',hit=i<successes)
        if i>=256-invalid:row.update(invalid=True,hit=False)
        else:row.update(progress_m=.1,free_path_m=.4,mean_free_entropy=1.)
        rows.append(row)
    totals=dict(trials=256,successes=successes,invalid_trials=invalid,valid_trials=256-invalid,
        mean_progress_m=sum(r.get('progress_m',0.) for r in rows)/256,
        mean_path_m=sum(r.get('free_path_m',0.) for r in rows)/256)
    return dict(rows=rows,totals=totals,all_physical_checks=invalid==0,criterion_pass=invalid==0 and successes>=192)


def copy_protocol(root,portable=False):
    """Read protocol text only; no actual policy, bank or outcome is accessed."""
    original=root/plot.ORIGINAL_PROTOCOL.relative_to(ROOT);original.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(plot.ORIGINAL_PROTOCOL,original)
    if portable:
        path=root/plot.PORTABLE_PROTOCOL.relative_to(ROOT);shutil.copyfile(plot.PORTABLE_PROTOCOL,path)
    else:path=original
    return path,json.loads(path.read_text())


def test_all_prescribed_trials_and_invalid_denominator_retained():
    result=plot.endpoint_values(endpoint(100,16))
    assert result['successes']==100 and result['trials']==256 and result['invalid_trials']==16
    assert result['mean_goal_progress_cm']==pytest.approx(240/256*10)
    assert len(result['contexts'])==16 and sum(r['successes'] for r in result['contexts'])==100
    assert not result['criterion_pass']
    for broken in (dict(endpoint(),rows=endpoint()['rows'][:-1]),
            dict(endpoint(),rows=endpoint()['rows'][:-1]+[endpoint()['rows'][0]])):
        with pytest.raises(AssertionError):plot.endpoint_values(broken)


def test_criterion_uses_fixed192_threshold_and_physical_guard():
    assert not plot.endpoint_values(endpoint(191))['criterion_pass']
    assert plot.endpoint_values(endpoint(192))['criterion_pass']
    assert not plot.endpoint_values(endpoint(192,1))['criterion_pass']
    wrong=endpoint(191);wrong['criterion_pass']=True
    with pytest.raises(AssertionError):plot.endpoint_values(wrong)


def test_training_blocks_weight_entropy_by_actual_free_decisions():
    rows=[valid_metric(i+1) for i in range(129)]
    rows[0].update(free_updates=32,entropy_sum_per_trial=math.log(6))
    result=plot.training_blocks(rows)
    assert len(result)==2 and result[0]['trials']==4096 and result[1]['trials']==32
    expected=(math.log(6)*32+127*45*32)/(32+127*960)
    assert result[0]['entropy_per_free_decision']==pytest.approx(expected)
    assert result[0]['complete_block'] and not result[1]['complete_block']
    assert result[1]['end_update']==129 and result[1]['cumulative_trials']==4128
    rows[10]['update']=1
    with pytest.raises(AssertionError):plot.training_blocks(rows)


def test_audited_artifact_bytes_must_match(tmp_path):
    path=tmp_path/'summary.json';path.write_text('{}')
    record={'artifacts':{'summary.json':{'sha256':plot.sha(path),'bytes':path.stat().st_size}}}
    assert plot.audited_file(tmp_path,record,'summary.json')==path
    path.write_text('{"modified":true}')
    with pytest.raises(AssertionError):plot.audited_file(tmp_path,record,'summary.json')


def test_missing_and_failed_arms_remain_explicit_without_fake_zeros(tmp_path):
    root=tmp_path/'repo';protocol,design=copy_protocol(root)
    routing=tmp_path/'routing.json';workers=[]
    for c,m in plot.ARMS:workers.append({'condition':c,'actor_mode':m,'name':c+'-'+m,'artifact_subdir':'experiment'})
    routing.write_text(json.dumps({'workers':workers,'protocol_sha256':plot.ORIGINAL_SHA256}));audits=tmp_path/'audits';audits.mkdir()
    c,m=plot.ARMS[0];folder=root/'results/ring_cloud_artifacts'/workers[0]['name']/'experiment'
    record=dict(condition=c,actor_mode=m,folder=str(folder),auditor_sha256=design['source_sha256']['scripts/audit_local_return_feedback.py'],
        status='failed',passed=False,complete=False,failures=['Synthetic damaged evidence'],pending=[])
    (audits/plot.audit_name(c,m)).write_text(json.dumps(record))
    report=plot.load_report(routing,audits,root,protocol_path=protocol)
    assert len(report['arms'])==4 and report['arms'][0]['status']=='failed'
    assert all(r['baseline'] is None and r['final'] is None and r['accepted_updates'] is None for r in report['arms'])
    assert not report['all_arms_complete'] and not any(r['final_qualified'] for r in report['arms'])
    bad=workers[:-1];routing.write_text(json.dumps({'workers':bad,'protocol_sha256':plot.ORIGINAL_SHA256}))
    with pytest.raises(AssertionError,match='Every prescribed arm'):plot.load_report(routing,audits,root,protocol_path=protocol)


def test_synthetic_output_is_watermarked_and_keeps_all_eight_endpoint_rows(tmp_path):
    report=plot.synthetic_preview();assert report['synthetic'] and len(report['arms'])==4
    report['arms'][0].update(status='failed',baseline=None,final=None,training=[],accepted_updates=None)
    plot.write_report(report,tmp_path)
    assert 'SYNTHETIC PREVIEW' in (tmp_path/'README.md').read_text()
    import csv
    rows=list(csv.DictReader((tmp_path/'all_arms.csv').open()))
    assert len(rows)==8 and rows[0]['successes']==rows[1]['successes']==''
    assert rows[0]['status']=='failed'
    for name in ('feedback_return.png','feedback_learning.png'):assert (tmp_path/name).stat().st_size>1000


def test_terminal_worker_failure_keeps_verified_baseline_and_unknown_final(tmp_path):
    root=tmp_path/'repo';protocol,design=copy_protocol(root);sources=design['source_sha256']
    workers=[dict(condition=c,actor_mode=m,name=c+'-'+m,artifact_subdir='experiment',status='planned') for c,m in plot.ARMS]
    workers[0]['status']='launch_failed';workers[1]['status']='failed_collected_deleted'
    routing=tmp_path/'routing.json';routing.write_text(json.dumps({'workers':workers,'protocol_sha256':plot.ORIGINAL_SHA256}))
    audits=tmp_path/'audits';audits.mkdir();worker=workers[1]
    folder=root/'results/ring_cloud_artifacts'/worker['name']/'experiment';child=folder/'baseline_evaluation';child.mkdir(parents=True)
    report=endpoint(100);report.update(condition=worker['condition'],actor_mode=worker['actor_mode'],complete=True,
        collector_sha256=sources['scripts/local_return_feedback.py'],source_sha256=sources['scripts/local_return_feedback_evaluate.py'])
    (child/'summary.json').write_text(json.dumps(report));(folder/'metrics.jsonl').write_text('')
    metrics=dict(accepted_updates=0,rolled_back_rows=0,partial_last_row=False)
    record=dict(condition=worker['condition'],actor_mode=worker['actor_mode'],folder=str(folder),
        auditor_sha256=sources['scripts/audit_local_return_feedback.py'],protocol_sha256=plot.ORIGINAL_SHA256,status='pending',passed=True,complete=False,
        failures=[],pending=['Final not produced'],checks={'metrics':metrics,'evaluations':{'baseline':{
            'complete':True,'totals':report['totals']}}},artifacts={})
    for name in ('baseline_evaluation/summary.json','metrics.jsonl'):
        path=folder/name;record['artifacts'][name]={'bytes':path.stat().st_size,'sha256':plot.sha(path)}
    (audits/plot.audit_name(worker['condition'],worker['actor_mode'])).write_text(json.dumps(record))
    result=plot.load_report(routing,audits,root,protocol_path=protocol);launch,partial=result['arms'][:2]
    assert launch['status']=='failed' and launch['worker_terminal_failure'] and launch['baseline'] is None
    assert partial['status']=='failed' and partial['worker_terminal_failure'] and partial['audit_passed']
    assert partial['baseline']['successes']==100 and partial['accepted_updates']==0
    assert partial['final'] is None and not partial['final_qualified'] and not result['all_arms_complete']
    assert result['arms'][2]['status']==result['arms'][3]['status']=='pending'


@pytest.mark.parametrize('portable',[False,True])
def test_exact_protocol_allowlist_and_parent_reference_identity(tmp_path,portable):
    path,expected=copy_protocol(tmp_path,portable)
    design,parent,digest,kind,inputs=plot.selected_protocol(path,tmp_path)
    assert design==expected and digest==(plot.PORTABLE_SHA256 if portable else plot.ORIGINAL_SHA256)
    assert kind==('portable' if portable else 'original')
    assert parent['evaluation']['bank']==design['evaluation']['bank']
    assert parent['source_sha256']['scripts/local_return_feedback_evaluate.py']==design['source_sha256']['scripts/local_return_feedback_evaluate.py']
    assert plot.ORIGINAL_SHA256 in inputs.values()
    path.write_text(path.read_text()+' ')
    with pytest.raises(AssertionError,match='Unapproved'):plot.selected_protocol(path,tmp_path)


def test_portable_protocol_rejects_wrong_parent_asset(tmp_path):
    path,_=copy_protocol(tmp_path,True)
    parent=tmp_path/plot.ORIGINAL_PROTOCOL.relative_to(ROOT);parent.write_text(parent.read_text()+' ')
    with pytest.raises(AssertionError):plot.selected_protocol(path,tmp_path)


def test_model_audit_must_match_selected_portable_protocol(tmp_path):
    root=tmp_path/'repo';protocol,design=copy_protocol(root,True)
    workers=[dict(condition=c,actor_mode=m,name=c+'-'+m,artifact_subdir='experiment') for c,m in plot.ARMS]
    routing=tmp_path/'routing.json';routing.write_text(json.dumps({'workers':workers,'protocol_sha256':plot.PORTABLE_SHA256}))
    audits=tmp_path/'audits';audits.mkdir();c,m=plot.ARMS[0]
    folder=root/'results/ring_cloud_artifacts'/workers[0]['name']/'experiment'
    record=dict(condition=c,actor_mode=m,folder=str(folder),
        auditor_sha256=design['source_sha256']['scripts/audit_local_return_feedback.py'],
        protocol_sha256=plot.ORIGINAL_SHA256,status='pending',passed=True,complete=False,failures=[],pending=[])
    (audits/plot.audit_name(c,m)).write_text(json.dumps(record))
    with pytest.raises(AssertionError):plot.load_report(routing,audits,root,protocol_path=protocol)


def test_canonical_reference_path_rebases_without_rewriting_published_record(tmp_path):
    original='/old/machine/checkout/results/local_return_feedback_acquisition/references'
    expected=tmp_path/'results/local_return_feedback_acquisition/references'
    assert plot.reference_folder(tmp_path,original)==expected
    assert original=='/old/machine/checkout/results/local_return_feedback_acquisition/references'
    for wrong in ('results/local_return_feedback_acquisition/references','/old/machine/results/other/references'):
        with pytest.raises(AssertionError):plot.reference_folder(tmp_path,wrong)
