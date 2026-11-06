"""Reporting arithmetic and incomplete-arm coverage; synthetic values only."""
import copy
import json
import math
from pathlib import Path
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


def test_missing_and_failed_arms_remain_explicit_without_fake_zeros(tmp_path,monkeypatch):
    root=tmp_path/'repo';protocol=root/plot.PROTOCOL.relative_to(ROOT);protocol.parent.mkdir(parents=True)
    design={'source_sha256':{'scripts/audit_local_return_feedback.py':'synthetic-auditor'}}
    protocol.write_text(json.dumps(design));monkeypatch.setattr(plot,'PROTOCOL_SHA256',plot.sha(protocol))
    routing=tmp_path/'routing.json';workers=[]
    for c,m in plot.ARMS:workers.append({'condition':c,'actor_mode':m,'name':c+'-'+m,'artifact_subdir':'experiment'})
    routing.write_text(json.dumps({'workers':workers}));audits=tmp_path/'audits';audits.mkdir()
    c,m=plot.ARMS[0];folder=root/'results/ring_cloud_artifacts'/workers[0]['name']/'experiment'
    record=dict(condition=c,actor_mode=m,folder=str(folder),auditor_sha256='synthetic-auditor',
        status='failed',passed=False,complete=False,failures=['Synthetic damaged evidence'],pending=[])
    (audits/plot.audit_name(c,m)).write_text(json.dumps(record))
    report=plot.load_report(routing,audits,root)
    assert len(report['arms'])==4 and report['arms'][0]['status']=='failed'
    assert all(r['baseline'] is None and r['final'] is None and r['accepted_updates'] is None for r in report['arms'])
    assert not report['all_arms_complete'] and not any(r['final_qualified'] for r in report['arms'])
    bad=workers[:-1];routing.write_text(json.dumps({'workers':bad}))
    with pytest.raises(AssertionError,match='Every prescribed arm'):plot.load_report(routing,audits,root)


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


def test_terminal_worker_failure_keeps_verified_baseline_and_unknown_final(tmp_path,monkeypatch):
    root=tmp_path/'repo';protocol=root/plot.PROTOCOL.relative_to(ROOT);protocol.parent.mkdir(parents=True)
    sources={'scripts/audit_local_return_feedback.py':'auditor','scripts/local_return_feedback.py':'collector',
        'scripts/local_return_feedback_evaluate.py':'evaluator'}
    protocol.write_text(json.dumps({'source_sha256':sources}));monkeypatch.setattr(plot,'PROTOCOL_SHA256',plot.sha(protocol))
    workers=[dict(condition=c,actor_mode=m,name=c+'-'+m,artifact_subdir='experiment',status='planned') for c,m in plot.ARMS]
    workers[0]['status']='launch_failed';workers[1]['status']='failed_collected_deleted'
    routing=tmp_path/'routing.json';routing.write_text(json.dumps({'workers':workers}))
    audits=tmp_path/'audits';audits.mkdir();worker=workers[1]
    folder=root/'results/ring_cloud_artifacts'/worker['name']/'experiment';child=folder/'baseline_evaluation';child.mkdir(parents=True)
    report=endpoint(100);report.update(condition=worker['condition'],actor_mode=worker['actor_mode'],complete=True,
        collector_sha256='collector',source_sha256='evaluator')
    (child/'summary.json').write_text(json.dumps(report));(folder/'metrics.jsonl').write_text('')
    metrics=dict(accepted_updates=0,rolled_back_rows=0,partial_last_row=False)
    record=dict(condition=worker['condition'],actor_mode=worker['actor_mode'],folder=str(folder),
        auditor_sha256='auditor',protocol_sha256=plot.PROTOCOL_SHA256,status='pending',passed=True,complete=False,
        failures=[],pending=['Final not produced'],checks={'metrics':metrics,'evaluations':{'baseline':{
            'complete':True,'totals':report['totals']}}},artifacts={})
    for name in ('baseline_evaluation/summary.json','metrics.jsonl'):
        path=folder/name;record['artifacts'][name]={'bytes':path.stat().st_size,'sha256':plot.sha(path)}
    (audits/plot.audit_name(worker['condition'],worker['actor_mode'])).write_text(json.dumps(record))
    result=plot.load_report(routing,audits,root);launch,partial=result['arms'][:2]
    assert launch['status']=='failed' and launch['worker_terminal_failure'] and launch['baseline'] is None
    assert partial['status']=='failed' and partial['worker_terminal_failure'] and partial['audit_passed']
    assert partial['baseline']['successes']==100 and partial['accepted_updates']==0
    assert partial['final'] is None and not partial['final_qualified'] and not result['all_arms_complete']
    assert result['arms'][2]['status']==result['arms'][3]['status']=='pending'
