"""Small, status-aware report of all four fixed feedback-acquisition arms.

Presentation only. Reads audited artifacts; never evaluates a policy, changes a
study gate, substitutes zero for missing results, or selects a training endpoint.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import audit_local_return_feedback as auditor

ORIGINAL_PROTOCOL=ROOT/'docs/research/local_return_feedback_training_protocol.json'
PORTABLE_PROTOCOL=ROOT/'docs/research/local_return_feedback_portable_protocol.json'
ORIGINAL_SHA256='ad969e782ec2943de2e59b2d72629002dea537f7c9c0fcf2ad3905ce051038c7'
PORTABLE_SHA256='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
ALLOWED_PROTOCOLS={ORIGINAL_SHA256:'original',PORTABLE_SHA256:'portable'}
REFERENCE_AUDIT_SHA256='340b280b740de8e612828e89fe183b82d6e2e09e4997f2bee35099ec4f3f99aa'
ARMS=tuple((condition,mode) for condition in ('gc100','pgc33') for mode in ('retained_actor','reset_actor'))
LABELS={'gc100':'Grid','pgc33':'PGC + grid'}
MODE_LABELS={'retained_actor':'Original actor','reset_actor':'Reset actor'}
COLORS={'gc100':'#2878b5','pgc33':'#e27932'}
sha=auditor.sha;read=auditor.read


def selected_protocol(path,root=ROOT):
    """Only the two reviewed protocols; portable references retain parent identity."""
    path=Path(path);root=Path(root);digest=sha(path)
    assert digest in ALLOWED_PROTOCOLS,'Unapproved feedback protocol hash'
    design=read(path);kind=ALLOWED_PROTOCOLS[digest]
    inputs={str(path.resolve()):digest};reference_design=design
    if kind=='portable':
        portability=design['portability'];name=portability['parent_protocol_path']
        parent=auditor.checked_path(root,name)
        assert sha(parent)==portability['parent_protocol_sha256']==ORIGINAL_SHA256
        assert design['input_assets'][name]==ORIGINAL_SHA256
        reference_design=read(parent);inputs[str(parent.resolve())]=ORIGINAL_SHA256
        for field in ('manifest_path','manifest_sha256','bank'):
            assert design['evaluation'][field]==reference_design['evaluation'][field],field
        for name in ('scripts/local_return_feedback.py','scripts/local_return_feedback_evaluate.py','scripts/audit_local_return_feedback.py'):
            assert design['source_sha256'][name]==reference_design['source_sha256'][name],name
        assert portability['same_saved_bank'] and portability['same_training_computation'] and portability['same_evaluate_plan']
        assert portability['raw_evaluator_source']=='scripts/local_return_feedback_evaluate.py'
    return design,reference_design,digest,kind,inputs


def audited_file(folder,record,relative):
    path=auditor.checked_path(folder,relative)
    expected=record['artifacts'][relative]
    assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],relative
    return path


def endpoint_values(report):
    rows=report['rows']
    assert len(rows)==256 and {(r['context'],r['heading_index']) for r in rows}=={(c,h) for c in range(16) for h in range(16)}
    assert all(r['variant']=='feedback_native' and isinstance(r['hit'],bool) for r in rows)
    assert all(not r['hit'] for r in rows if r.get('invalid',False))
    successes=sum(r['hit'] for r in rows);invalid=sum(r.get('invalid',False) for r in rows)
    values=dict(trials=256,successes=successes,success_percent=successes/256*100,
        invalid_trials=invalid,valid_trials=256-invalid,
        mean_goal_progress_cm=sum(r.get('progress_m',0.) for r in rows)/256*100,
        mean_path_cm=sum(r.get('free_path_m',0.) for r in rows)/256*100,
        mean_entropy=sum(r.get('mean_free_entropy',0.) for r in rows)/256,
        physical_checks_passed=report['all_physical_checks'],criterion_pass=report['criterion_pass'])
    totals=report['totals'];assert totals['trials']==256 and totals['successes']==successes
    assert totals['invalid_trials']==invalid and totals['valid_trials']==256-invalid
    assert math.isclose(totals['mean_progress_m']*100,values['mean_goal_progress_cm'],abs_tol=1e-12)
    assert math.isclose(totals['mean_path_m']*100,values['mean_path_cm'],abs_tol=1e-12)
    assert report['criterion_pass']==(report['all_physical_checks'] and successes>=192)
    values['contexts']=[dict(context=c,trials=16,successes=sum(r['hit'] for r in rows if r['context']==c),
        mean_goal_progress_cm=sum(r.get('progress_m',0.) for r in rows if r['context']==c)/16*100) for c in range(16)]
    return values


def training_blocks(rows):
    """Fixed consecutive 128-update bins; retain and label a final partial bin."""
    assert [r['update'] for r in rows]==list(range(1,len(rows)+1))
    result=[]
    for start in range(0,len(rows),128):
        block=rows[start:start+128];trials=sum(r['completed_trials'] for r in block)
        decisions=sum(r['free_updates'] for r in block)
        assert trials==len(block)*32 and decisions>0
        successes=sum(r['free_hits'] for r in block)
        entropy=sum(r['entropy_sum_per_trial']*32 for r in block)/decisions
        assert 0<=entropy<=math.log(6)+2e-6
        result.append(dict(start_update=start+1,end_update=start+len(block),updates=len(block),
            complete_block=len(block)==128,cumulative_trials=(start+len(block))*32,
            trials=trials,successes=successes,success_percent=successes/trials*100,
            free_decisions=decisions,entropy_per_free_decision=entropy))
    return result


def audit_name(condition,mode):return f'{condition}_{mode}.json'


def apply_worker_failure(result,worker):
    """Worker completion status is separate from integrity of saved measurements."""
    terminal_failure=worker.get('status') in ('launch_failed','failed_collected_deleted')
    result['worker_terminal_failure']=terminal_failure
    if terminal_failure:
        result['status']='failed'
        message='Worker ended with status '+worker['status']
        result['failures']=list(result['failures'])
        if message not in result['failures']:result['failures'].append(message)
        result['final_qualified']=False


def reference_folder(root,recorded_folder):
    """Rebase hash-pinned published evidence without editing its original path."""
    relative=Path('results/local_return_feedback_acquisition/references')
    original=Path(recorded_folder)
    assert original.is_absolute() and original.parts[-len(relative.parts):]==relative.parts
    return Path(root)/relative


def reference_values(root,design,inputs):
    path=root/'results/local_return_feedback_design/reference_audit.json'
    result=dict(status='pending',uniform=None,planner=None)
    if not path.exists():return result
    assert sha(path)==REFERENCE_AUDIT_SHA256,'Canonical once-only physical-reference audit differs'
    record=read(path);inputs[str(path.resolve())]=sha(path)
    assert record['auditor_sha256']==design['source_sha256']['scripts/audit_local_return_feedback.py']
    if not record['passed'] or not record['complete']:
        result.update(status=record['status'],failures=record['failures']);return result
    assert record['protocol_sha256']==ORIGINAL_SHA256
    folder=reference_folder(root,record['folder'])
    top_path=audited_file(folder,record,'summary.json');top=read(top_path)
    assert top['complete'] and top['no_policy_or_optimization'] and top['reference_trials']==512
    inputs[str(top_path.resolve())]=sha(top_path)
    for controller in ('uniform','planner'):
        path=audited_file(folder,record,f'{controller}/summary.json');report=read(path)
        assert report['controller']==controller and report['actor_mode'] is None
        assert report['complete'] and not report['criterion_pass'] and report['no_optimization']
        assert report['source_sha256']==design['source_sha256']['scripts/local_return_feedback_evaluate.py']
        assert report['collector_sha256']==design['source_sha256']['scripts/local_return_feedback.py']
        assert report['reference']['manifest_sha256']==design['evaluation']['manifest_sha256']
        assert report['reference']['bank_sha256']==design['evaluation']['bank']['sha256']
        assert report['totals']==record['checks'][controller]['totals']==top['results'][controller]['totals']
        assert report['totals']['trials']==256
        result[controller]=dict(successes=report['totals']['successes'],trials=256,
            invalid_trials=report['totals']['invalid_trials'])
        inputs[str(path.resolve())]=sha(path)
    result['status']='complete';return result


def load_report(routing_path,audit_root,root=ROOT,protocol_path=None):
    root=Path(root);audit_root=Path(audit_root)
    protocol=Path(protocol_path) if protocol_path is not None else root/PORTABLE_PROTOCOL.relative_to(ROOT)
    design,reference_design,protocol_hash,protocol_kind,inputs=selected_protocol(protocol,root)
    routing=read(routing_path)
    assert routing['protocol_sha256']==protocol_hash,'Worker routing differs from selected protocol'
    workers=routing['workers'];identities=[(r['condition'],r['actor_mode']) for r in workers]
    assert len(identities)==4 and set(identities)==set(ARMS),'Every prescribed arm must have a routing row'
    rows=[];inputs[str(Path(routing_path).resolve())]=sha(routing_path)
    for condition,mode in ARMS:
        worker=next(r for r in workers if (r['condition'],r['actor_mode'])==(condition,mode))
        name=worker['name'];assert isinstance(name,str) and '/' not in name and name not in ('','.','..')
        child=worker.get('artifact_subdir','experiment');assert child=='experiment'
        folder=root/'results/ring_cloud_artifacts'/name/child
        result=dict(condition=condition,actor_mode=mode,label=f'{LABELS[condition]} · {MODE_LABELS[mode]}',
            worker=name,folder=str(folder.resolve()),status='pending',worker_status=worker.get('status'),
            audit_passed=False,audit_complete=False,baseline=None,final=None,training=[],
            accepted_updates=None,failures=[],pending=[],final_qualified=False)
        apply_worker_failure(result,worker)
        audit_path=audit_root/audit_name(condition,mode)
        if not audit_path.exists():
            result['pending']=['Independent artifact audit unavailable'];rows.append(result);continue
        record=read(audit_path);inputs[str(audit_path.resolve())]=sha(audit_path)
        assert record['condition']==condition and record['actor_mode']==mode
        assert Path(record['folder']).resolve()==folder.resolve()
        assert record['auditor_sha256']==design['source_sha256']['scripts/audit_local_return_feedback.py']
        result.update(status=record['status'],audit_passed=record['passed'],audit_complete=record['complete'],
            failures=record['failures'],pending=record['pending'])
        apply_worker_failure(result,worker)
        if not record['passed']:
            # Failed validation is an explicit arm outcome/status, never a zero score.
            rows.append(result);continue
        assert record.get('protocol_sha256')==protocol_hash
        assert record['status'] in ('pending','complete') and record['complete']==(record['status']=='complete')
        checks=record['checks']
        for endpoint in ('baseline','final'):
            if endpoint not in checks.get('evaluations',{}):continue
            path=audited_file(folder,record,f'{endpoint}_evaluation/summary.json');report=read(path)
            assert report['condition']==condition and report['actor_mode']==mode and report['complete']
            assert report['collector_sha256']==design['source_sha256']['scripts/local_return_feedback.py']
            assert report['source_sha256']==design['source_sha256']['scripts/local_return_feedback_evaluate.py']
            assert report['totals']==checks['evaluations'][endpoint]['totals']
            assert checks['evaluations'][endpoint]['complete']
            result[endpoint]=endpoint_values(report);inputs[str(path.resolve())]=sha(path)
        # Audit every metrics/lineage file used by accepted-prefix reconstruction.
        try:
            for relative in record['artifacts']:
                if Path(relative).name in ('metrics.jsonl','resume_lineage.json'):
                    path=audited_file(folder,record,relative);inputs[str(path.resolve())]=sha(path)
            metrics,metadata=auditor.accepted_metrics(folder)
            assert metadata==checks['metrics']
            result['training']=training_blocks(metrics);result['accepted_updates']=len(metrics)
            result['rolled_back_metric_rows']=metadata['rolled_back_rows']
        except (AssertionError,FileNotFoundError):
            if record['complete']:raise
            # Live training may append rows after a valid partial audit. Keep the
            # audited endpoint values, but never combine unaudited new metrics.
            result['pending']=list(result['pending'])+['Training metrics changed since partial audit; curve awaits a fresh audit']
        if record['complete']:
            assert len(metrics)==1024 and result['baseline'] is not None and result['final'] is not None
            for name in ('final.pt','final_before_evaluation.pt','summary.json'):
                path=audited_file(folder,record,name);inputs[str(path.resolve())]=sha(path)
            result['final_qualified']=result['final']['criterion_pass'] and not result['worker_terminal_failure']
        rows.append(result)
    references=reference_values(root,reference_design,inputs)
    return dict(version='feedback_acquisition_display_v1',synthetic=False,protocol_sha256=protocol_hash,protocol_kind=protocol_kind,
        all_arms_complete=all(r['audit_complete'] and not r['worker_terminal_failure'] for r in rows),arms=rows,input_sha256=inputs,
        references=references,
        criterion=dict(successes=192,trials=256,percent=75,requires_complete_strict_audit=True),
        scope='One original trained seed per population. Four prescribed arms; all network weights trainable, normal feedback throughout. This is a fast-steering acquisition control, not a ring-attractor result.',
        mean_convention='All 256 prescribed evaluation trials; invalid missing progress/path values contribute zero as declared by the producer, with invalid counts retained. Unavailable endpoint means remain null.')


def draw(report,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for row in report['arms']:
        color=COLORS[row['condition']];reset=row['actor_mode']=='reset_actor';style='--' if reset else '-';marker='s' if reset else 'o'
        for ax,key in zip(axes,('success_percent','mean_goal_progress_cm')):
            available=[(i,row[name][key]) for i,name in enumerate(('baseline','final')) if row[name] is not None]
            if available:
                x,y=zip(*available);ax.plot(x,y,color=color,linestyle=style,marker=marker,label=row['label'],alpha=.9)
            else:ax.plot([],[],color=color,linestyle=style,marker=marker,label=row['label']+' (pending/failed)')
    axes[0].axhline(75,color='.45',ls=':',lw=1);axes[0].text(.02,77,'75% criterion',color='.4',fontsize=9)
    axes[0].set(ylabel='Goals reached (%)',ylim=(0,102),title='Return within 2 seconds')
    axes[1].axhline(0,color='.7',lw=.8);axes[1].set(ylabel='Mean progress toward goal (cm)',title='Goal-directed movement')
    for ax in axes:ax.set(xticks=[0,1],xticklabels=['Before training','Fixed final'],xlim=(-.07,1.07));ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8.5)
    fig.suptitle('Feedback-supported goal return'+(' · results incomplete' if not report['all_arms_complete'] else ''),fontsize=14)
    if report['synthetic']:fig.text(.5,.52,'SYNTHETIC PREVIEW — NOT STUDY RESULTS',ha='center',rotation=14,fontsize=19,color='crimson',alpha=.55)
    fig.savefig(out/'feedback_return.png');fig.savefig(out/'feedback_return.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for row in report['arms']:
        color=COLORS[row['condition']];style='--' if row['actor_mode']=='reset_actor' else '-';blocks=row['training']
        for ax,key in zip(axes,('success_percent','entropy_per_free_decision')):
            ax.plot([r['cumulative_trials']/1000 for r in blocks],[r[key] for r in blocks],color=color,ls=style,label=row['label'])
            for r in blocks:
                if not r['complete_block']:ax.plot(r['cumulative_trials']/1000,r[key],'o',mfc='white',mec=color,ms=5)
    axes[0].set(ylabel='Training goals reached (%)',ylim=(0,100),title='Reward discovery')
    axes[1].axhline(math.log(6),color='.5',ls=':',lw=1);axes[1].set(ylabel='Action entropy (nats)',ylim=(0,math.log(6)*1.08),title='Action randomness')
    for ax in axes:ax.set(xlabel='Training trials (thousands)',xlim=(0,32.768));ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8.5)
    fig.suptitle('Training in fixed 128-update blocks',fontsize=14)
    if report['synthetic']:fig.text(.5,.52,'SYNTHETIC PREVIEW — NOT STUDY RESULTS',ha='center',rotation=14,fontsize=19,color='crimson',alpha=.55)
    fig.savefig(out/'feedback_learning.png');fig.savefig(out/'feedback_learning.pdf');plt.close(fig)


def write_report(report,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'all_arms.csv').open('w') as stream:
        fields=['condition','actor_mode','status','audit_passed','audit_complete','accepted_updates','final_qualified',
            'endpoint','successes','trials','invalid_trials','success_percent','mean_goal_progress_cm','mean_path_cm']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for arm in report['arms']:
            for endpoint in ('baseline','final'):
                row={k:arm[k] for k in fields if k in arm};row['endpoint']=endpoint
                row.update({k:v for k,v in (arm[endpoint] or {}).items() if k in fields});writer.writerow(row)
    lines=['# Feedback-supported goal return','',
        '**SYNTHETIC PREVIEW — no study results.**' if report['synthetic'] else ('All four fixed runs are audited.' if report['all_arms_complete'] else 'Results are incomplete. Unknown values are shown as “pending”; failed arms remain listed.'),'',
        'The agents receive fresh images and spatial codes throughout. Each arm is assigned 32,768 training trials and a fixed 256-trial test. All network weights can learn.','',
        '| Population | Actor at start | Status | Before | Fixed final | Final goal progress |','|---|---|---|---:|---:|---:|']
    for row in report['arms']:
        score=lambda endpoint:f"{row[endpoint]['successes']}/256" if row[endpoint] is not None else 'pending'
        progress=f"{row['final']['mean_goal_progress_cm']:.2f} cm" if row['final'] is not None else 'pending'
        lines.append(f"| {LABELS[row['condition']]} | {MODE_LABELS[row['actor_mode']]} | {row['status']} | {score('baseline')} | {score('final')} | {progress} |")
    references=report.get('references',{})
    if references.get('status')=='complete':
        lines+=['',f"Physical controls: uniform actions reached {references['uniform']['successes']}/256 goals; the planner reached {references['planner']['successes']}/256. The planner uses privileged physical state. These are once-only controls, not learned policies."]
    lines+=['','![Return success and goal progress](feedback_return.png)','','![Training reward discovery and action entropy](feedback_learning.png)','',
        'The preset criterion is at least 192/256 returns (75%) at the fixed final endpoint, with a complete passing artifact audit. No intermediate checkpoint is selected.','',
        'Training curves use consecutive 128-update blocks. Entropy is weighted by the number of free decisions; a hollow point marks an unfinished block. Unknown results are never scored as zero. Invalid trials count as failures and contribute zero to the declared all-trial progress/path means.','',
        'This is one original seed pair and a test of learning to steer toward a nearby goal with normal feedback. A qualifying final provides a starting point for the separate memory and ring-perturbation experiments. All four arms and all available endpoint measurements are in [the full table](all_arms.csv) and [the provenance summary](summary.json).','']
    (out/'README.md').write_text('\n'.join(lines));draw(report,out)


def synthetic_preview():
    """Illustrative values exclusively for watermarked layout tests."""
    rows=[]
    for index,(condition,mode) in enumerate(ARMS):
        before=24+index*7;after=104+index*23
        endpoint=lambda n:dict(trials=256,successes=n,success_percent=n/256*100,invalid_trials=0,
            mean_goal_progress_cm=n/10-4,mean_path_cm=40.,criterion_pass=n>=192,physical_checks_passed=True)
        blocks=[dict(start_update=j*128+1,end_update=(j+1)*128,updates=128,complete_block=True,
            cumulative_trials=(j+1)*4096,trials=4096,successes=(j+1)*100,success_percent=(j+1)*(4+index),
            free_decisions=100000,entropy_per_free_decision=1.6-j*.1+index*.02) for j in range(8)]
        rows.append(dict(condition=condition,actor_mode=mode,label=f'{LABELS[condition]} · {MODE_LABELS[mode]}',status='synthetic',
            audit_passed=False,audit_complete=False,accepted_updates=1024,final_qualified=False,
            baseline=endpoint(before),final=endpoint(after),training=blocks))
    return dict(version='feedback_acquisition_display_v1',synthetic=True,all_arms_complete=False,arms=rows,input_sha256={})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=PORTABLE_PROTOCOL)
    p.add_argument('--routing',type=Path,default=ROOT/'docs/research/local_return_feedback_portable_launch.json')
    p.add_argument('--audit-root',type=Path,default=ROOT/'results/local_return_feedback_design/portable/cloud_audit')
    p.add_argument('--out',type=Path,default=ROOT/'results/local_return_feedback_acquisition/portable/report')
    p.add_argument('--synthetic-preview',action='store_true');args=p.parse_args()
    if args.synthetic_preview:
        assert str(args.out.resolve()).startswith('/private/tmp/') or str(args.out.resolve()).startswith('/tmp/'),'Synthetic previews stay outside study results'
        report=synthetic_preview()
    else:
        assert args.routing;report=load_report(args.routing,args.audit_root,protocol_path=args.protocol)
    write_report(report,args.out)
    print(json.dumps(dict(synthetic=report['synthetic'],all_arms_complete=report['all_arms_complete'],out=str(args.out))))


if __name__=='__main__':main()
