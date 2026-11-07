"""Descriptive report of independently audited saved full-circle actor outputs."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import audit_native_turn_counterbalanced_readout as audit
BASE=audit.BASE;OUT=BASE/'report'
EXECUTION=ROOT/'results/native_turn_counterbalanced_readout_execution/attempt02/execution.json'
DEFINITION=ROOT/'results/native_turn_counterbalanced_readout_design/reporting_definition.json'
SCOPE=ROOT/'results/native_turn_counterbalanced_readout_design/descriptive_scope.json'
DIAGNOSIS=ROOT/'results/native_turn_counterbalanced_readout_design/mismatch_diagnosis.json'


def assignment_comparison(left,right):
    left,right=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    assert left.shape==right.shape==(128,)and np.isfinite(left).all()and np.isfinite(right).all()
    balanced=.5*(left.mean()+right.mean());assigned=np.r_[left[:64],right[64:]].mean()
    return dict(left_full_circle_mm=float(left.mean()),right_full_circle_mm=float(right.mean()),
        balanced_mm=float(balanced),old_assignment_dense_mm=float(assigned),
        assignment_minus_balanced_mm=float(assigned-balanced))


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def definition():
    assert not DEFINITION.exists();DEFINITION.parent.mkdir(parents=True,exist_ok=True)
    write(DEFINITION,dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='prospective_saved_data_explanation',
        comparison='Each fixed-prefix full-circle readout; balanced prefix mean; dense old LEFT first-half/RIGHT second-half assignment; original16-case sparse result.',
        all_endpoints_branches_contexts_retained=True,assignment_from_actual_saved_prefix_and_phase_audit=True,
        scope='Proposed and approved before reading completed readout scores. Assignment comparison separates the old history/heading assignment from sparse angle sampling; it is not a new causal intervention.',
        no_new_policy_environment_or_recurrent_execution=True))


def report():
    assert DEFINITION.exists()and SCOPE.exists()and not(OUT/'summary.json').exists()
    execution=audit.read(EXECUTION)
    assert execution['complete']is False and execution['exit_code']==1 and not execution['timeout']
    assert execution['protocol_sha256']==audit.sha(audit.PROTOCOL)
    prior_path=ROOT/execution['prior_execution'];prior=audit.read(prior_path)
    assert prior['complete']is False and prior['exit_code']==1 and prior['source_commit']==execution['source_commit']and prior['protocol_sha256']==execution['protocol_sha256']
    execution_evidence={str(p.relative_to(ROOT)):audit.sha(p)for p in (EXECUTION,EXECUTION.parent/'run.log',prior_path,prior_path.parent/'run.log')}
    OUT.mkdir(parents=True,exist_ok=True)
    validation=audit.audit(OUT/'independent_validation.json')
    summary=audit.read(BASE/'summary.json');assert validation['arithmetic_audit_complete']and validation['summary_sha256']==audit.sha(BASE/'summary.json')
    contexts=[];aggregates=[];source_paths={}
    for condition in audit.CONDITIONS:
        for endpoint in audit.ENDPOINTS:
            folder=ROOT/f'results/native_turn_original_memory/{condition}'if endpoint=='original'else ROOT/f'results/native_turn_distillation/{condition}/{endpoint}/final_paired'
            path=folder/'summary.json';paired=audit.read(path);source_paths[str(path.relative_to(ROOT))]=audit.sha(path)
            for kind in audit.KINDS:
                for branch in audit.BRANCHES:
                    local=[]
                    for ci in range(16):
                        values=[]
                        for prefix in (0,1):
                            p=BASE/f'{condition}/{endpoint}/context{ci:02d}_prefix{prefix}_{kind}_{branch}.npz'
                            with np.load(p)as z:values.append(z['case_progress_mm'].copy())
                        r=assignment_comparison(*values)
                        old=[v for v in paired['rows']if(v['context'],v['kind'],v['variant'])==(ci,kind,branch)]
                        assert len(old)==16
                        r['original_sparse_mm']=1000*float(np.mean([v['expected_progress_m']for v in old]))
                        r['dense_minus_sparse_mm']=r['old_assignment_dense_mm']-r['original_sparse_mm']
                        errors=[v for v in validation['shared_case_precision']if(v['condition'],v['endpoint'],v['context'],v['kind'],v['branch'])==(condition,endpoint,ci,kind,branch)]
                        assert len(errors)==2
                        r['max_shared_score_error_mm']=max(v['max_abs_shared_score_error_mm']for v in errors)
                        r['max_shared_component_bound_mm']=max(v['max_component_bound_mm']for v in errors)
                        r['rejected_old_input_comparisons']=sum(v['old_input_equivalence_rejected']for v in errors)
                        row=dict(condition=condition,endpoint=endpoint,context=ci,kind=kind,branch=branch,**r)
                        contexts.append(row);local.append(row)
                    fields=tuple(r)
                    aggregate={f:float(np.max([x[f]for x in local])if f.startswith('max_')else np.sum([x[f]for x in local])if f=='rejected_old_input_comparisons'else np.mean([x[f]for x in local]))for f in fields}
                    aggregates.append(dict(condition=condition,endpoint=endpoint,kind=kind,branch=branch,contexts=16,**aggregate))
    assert len(contexts)==768 and len(aggregates)==48
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    names={'gc100':'Grid','pgc33':'PGC + grid'};x=np.arange(3);labels=['Original','Clean','State noise']
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),sharex=True)
    for row,kind in enumerate(audit.KINDS):
        for col,condition in enumerate(audit.CONDITIONS):
            ax=axes[row,col]
            rr=[next(r for r in aggregates if(r['condition'],r['endpoint'],r['kind'],r['branch'])==(condition,e,kind,'native'))for e in audit.ENDPOINTS]
            for shift,key,color,label in [(-.15,'left_full_circle_mm','#376b94','Fixed LEFT prefix'),(.15,'right_full_circle_mm','#d48e31','Fixed RIGHT prefix')]:
                ax.bar(x+shift,[r[key]for r in rr],.29,color=color,label=label)
            ax.plot(x,[r['balanced_mm']for r in rr],'kD',markersize=5,label='Balanced mean')
            ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,labels)
            ax.set_title(names[condition]+' · '+('turn only'if kind=='turn_only'else'probe cue'))
            ax.set_ylabel('Expected goalward progress (mm)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Exploratory native-cue readout: each prefix spans the full circle',y=.99)
    fig.text(.5,.01,'All rows retained; 12 old-input comparisons remain rejected.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.96))
    for suffix in ('png','pdf'):fig.savefig(OUT/f'full_circle_progress.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharex=True)
    for ax,condition in zip(axes,audit.CONDITIONS):
        rr=[next(r for r in aggregates if(r['condition'],r['endpoint'],r['kind'],r['branch'])==(condition,e,'probe_glimpse','native'))for e in audit.ENDPOINTS]
        ax.bar(x-.16,[r['old_assignment_dense_mm']for r in rr],.3,color='#d48e31',label='Old assignment, dense128')
        ax.bar(x+.16,[r['balanced_mm']for r in rr],.3,color='#376b94',label='Balanced full-circle prefixes')
        ax.plot(x,[r['original_sparse_mm']for r in rr],'kx',markersize=7,label='Original sparse16')
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,labels);ax.set_title(names[condition]+' · native probe cue')
        ax.set_ylabel('Expected goalward progress (mm)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.035))
    fig.suptitle('Heading–prefix assignment versus balanced readout',y=.99)
    fig.text(.5,.015,'Descriptive saved-data comparison; Grid old16 includes a rejected visible-history match.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.15,1,.94))
    for suffix in ('png','pdf'):fig.savefig(OUT/f'assignment_comparison.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),sharex=True)
    styles=[('native','#315c84','Native cue'),('ordinary','#777777','Ordinary cue'),('pre_cue_reset_native','#168c7f','Pre-cue erase + native'),('pre_cue_reset_ordinary','#ba8531','Pre-cue erase + ordinary')]
    for row,kind in enumerate(audit.KINDS):
        for col,condition in enumerate(audit.CONDITIONS):
            ax=axes[row,col]
            for branch,color,label in styles:
                yy=[next(r['balanced_mm']for r in aggregates if(r['condition'],r['endpoint'],r['kind'],r['branch'])==(condition,e,kind,branch))for e in audit.ENDPOINTS]
                ax.plot(x,yy,'o-',color=color,label=label,linewidth=1.5,markersize=5)
            ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,labels);ax.set_ylabel('Balanced goalward progress (mm)')
            ax.set_title(names[condition]+' · '+('turn only'if kind=='turn_only'else'probe cue'));ax.grid(axis='y',alpha=.18)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Exploratory cue controls with balanced full-circle prefixes',y=.99);fig.tight_layout(rect=(0,.12,1,.96))
    fig.text(.5,.01,'All rows retained; failed old16 equivalence is not promoted to a pass.',ha='center',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'cue_controls.{suffix}',dpi=180)
    plt.close(fig)
    for name,rows in [('all_conditions.csv',aggregates),('all_contexts.csv',contexts)]:
        with (OUT/name).open('w',newline='')as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    precision=max(r['max_shared_score_error_mm']for r in aggregates)
    lines=['# Descriptive full-circle readout from a failed-equivalence run','',
        '**The sealed old-input equivalence check failed and remains rejected.** All1,536 families and196,608 phase rows were saved, including12 failed comparisons. A microscopic pose difference crossed a hard visual stripe boundary in Grid context11’s RIGHT/probe history during encoding. No tolerance, state, input or scientific completion status was changed. This report independently audits the recorded full-circle cohort as a standalone exploratory description.','',
        'Each fixed LEFT and fixed RIGHT prefix now covers every heading. The retained actor reads the already-saved first-choice hidden state; all contexts, modes, four cue/reset branches and six endpoints remain in the analysis. No recurrent trajectory, CNN, policy rollout or optimization was rerun.','',
        '![Full-circle readout](full_circle_progress.png)','',
        'The table below uses the native probe cue. The old assignment chooses LEFT for the first heading half-circle and RIGHT for its antipodes. Comparing that assignment with balanced prefixes on the same dense arrays describes the assignment effect; comparing it with the original sparse16 also involves sampling. Grid’s old16 comparison additionally contains the explicitly rejected visual history and cannot be called input-equivalent. These are descriptive comparisons of saved histories, not new causal interventions.','',
        '| Model | Endpoint | Original sparse (mm) | Dense old assignment (mm) | LEFT full circle (mm) | RIGHT full circle (mm) | Balanced (mm) | Max shared error (mm) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in audit.CONDITIONS:
        for e in audit.ENDPOINTS:
            r=next(r for r in aggregates if(r['condition'],r['endpoint'],r['kind'],r['branch'])==(c,e,'probe_glimpse','native'))
            lines.append('| '+names[c]+' | '+e+' | '+' | '.join(f'{r[k]:.7g}'for k in ('original_sparse_mm','old_assignment_dense_mm','left_full_circle_mm','right_full_circle_mm','balanced_mm','max_shared_score_error_mm'))+' |')
    lines += ['','![Assignment comparison](assignment_comparison.png)','',f'The largest observed shared-angle progress discrepancy was {precision:.3g} mm. This includes the known non-equivalent visible history, not just floating-point actor arithmetic. Every discrepancy and its componentwise bound are retained. These checks cover shared angles; they are not a numerical error bound for every unshared full-circle state. Effects near this scale require that precision context. {validation["downstream_failure_counts"]["cell"]} intact c-state comparisons also fail their frozen bounds; small later action/progress differences do not repair the rejected input or state equivalence.','',
        '![Cue controls](cue_controls.png)','',
        'The first-choice motor state is the declared zero-velocity brake. Signed progress uses the same actual six-action native physics; facing-goal alignment is auxiliary because strafing can also be useful. A phase-averaged policy is a mathematical counterfactual, not an actual recurrent-state erasure. A full-circle actor readout does not establish recurrent attraction or a ring.','',
        'Scope: one original seed per population,16 contexts,2 prefixes,2 modes,4 branches,3 endpoints per population;196,608 phase rows. No p-values, ring qualification or outcome-based exclusions. Arithmetic/report completion is distinct from the failed experiment-equivalence status.','',
        '[All conditions](all_conditions.csv) · [All contexts](all_contexts.csv) · [Independent arithmetic audit](independent_validation.json) · [All retained raw artifacts](../artifacts.json) · [Rendering mismatch diagnosis](../../native_turn_counterbalanced_readout_design/mismatch_diagnosis.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    result=dict(report_complete=True,scientific_run_complete=False,old16_input_equivalence='rejected',failed_equivalence_comparisons=12,
        producer_sha256=audit.sha(__file__),auditor_sha256=audit.sha(audit.__file__),
        protocol_sha256=audit.sha(audit.PROTOCOL),scientific_summary_sha256=audit.sha(BASE/'summary.json'),
        audit_sha256=audit.sha(OUT/'independent_validation.json'),execution_path=str(EXECUTION.relative_to(ROOT)),execution_sha256=audit.sha(EXECUTION),
        execution_evidence_sha256=execution_evidence,
        reporting_definition_sha256=audit.sha(DEFINITION),descriptive_scope_sha256=audit.sha(SCOPE),mismatch_diagnosis_sha256=audit.sha(DIAGNOSIS),original_paired_summary_sha256=source_paths,
        all_endpoints_branches_contexts=True,aggregates=aggregates,contexts=contexts,phase_rows=196608,
        no_new_network_environment_or_recurrent_execution=True)
    write(OUT/'summary.json',result)
    write(OUT/'artifacts.json',{p.name:dict(sha256=audit.sha(p),bytes=p.stat().st_size)for p in OUT.iterdir()if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-definition',action='store_true');p.add_argument('--report',action='store_true');args=p.parse_args()
    assert args.write_definition!=args.report
    if args.write_definition:definition()
    else:report()
