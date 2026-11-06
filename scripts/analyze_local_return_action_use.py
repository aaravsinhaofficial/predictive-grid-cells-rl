"""Secondary saved-data diagnostic of immediate goalward action alignment.

No model forward pass, environment step, random draw, optimization or new gate.
Both final independent audits must pass before any study values are computed.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.env import _TARGETS, ACTIONS
import local_goal_return_training as task
import local_goal_return_training_evaluate as evaluator
from package_local_goal_return_outputs import verify_final_audit, routed_attempts, sha

CONDITIONS=('gc100','pgc33')
VARIANTS=tuple(task.VARIANTS)
PROTOCOL=ROOT/'docs/research/local_return_action_use_protocol.json'
TRAINING=ROOT/'docs/research/local_goal_return_training_protocol.json'
TRAINING_HASH='0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
OUT=ROOT/'results/local_return_action_use'
POSITION_ATOL=2e-6
SOURCES=('scripts/analyze_local_return_action_use.py','tests/test_analyze_local_return_action_use.py',
    'scripts/local_goal_return_training.py','scripts/local_goal_return_training_evaluate.py',
    'scripts/audit_local_goal_return_training.py','scripts/audit_local_goal_return_baseline.py',
    'scripts/package_local_goal_return_outputs.py','code/env.py')


def read(path):return json.loads(Path(path).read_text())


def protocol():
    return dict(version='local_return_saved_first_action_use_v1',training_protocol_sha256=TRAINING_HASH,
        timing='Secondary diagnostic prepared after inspecting stage1–3 PGC success counts, before fixed final outputs. No prespecified primary result is changed.',
        cohort='Both fixed original-seed1 models; initialized new-head baseline and every stage1–4. All12policy variants,16contexts×16headings each; no successful-trial selection. Baseline near/hard banks and existinguniform/plannercontrols retained; mainplot uses the same hard bank throughout.',
        prerequisite='Both canonicalfinalaudits complete/passing, all4stages and exact producer/raw/source hashes; no behavioral gate required or modified.',
        computation='At firstFREE pre-action/post-LSTM snapshot, apply each of6nativeactions to saved position/bodyheading/vf/vs/vr with the SAME saved firstFREE body-motor noise. Native lowpass motor update then updatedheading translation then wallclip/damping. No environment or neural execution.',
        references='Literal saved action probabilities weight six goal-center distance reductions. Uniform is arithmeticmean ofsix; bestsix is a realized-noise hindsight physical bound unavailable to the agent. Recorded baselineuniform/planner controls use their saved controller probability overrides, not mislabeled actor probabilities.',
        validation=dict(chosen_action_position_max_abs_error_m=POSITION_ATOL,
            every_valid_row=True,source='Compare predicted chosenaction endpoint to rawposition at actual firstFREEglobalclock; no respawn. Failedcollection rows retained with undefineddiagnostic, nozero imputation.'),
        outputs='Every six-action progress/probability/endpoint, bodyrelativegoalbearing, actualchosenaction/checkerror, literalprobabilitysum, expected/uniform/best progress, expectedstepdistance, full30actionhit/progress/path. Percontext andpooledsummaries preservevalid/invalidcounts; descriptiveprobabilityvariation across16headings, nofit orseed-levelinference.',
        interpretation='Immediatealignment only. A usefulturn can havezero immediategoalwardprogress; longhorizonvalues/success andringstructure are separate. No adaptive task change ornew learning gate.',
        source_sha256={name:sha(ROOT/name) for name in SOURCES})


def first_action_physics(position,heading,motors,noise,cfg):
    """Vectorized native equations; [trial,action,...], no live environment."""
    position=np.asarray(position,dtype=np.float64);heading=np.asarray(heading,dtype=np.float64)
    motors=np.asarray(motors,dtype=np.float64);noise=np.asarray(noise,dtype=np.float64)
    n=len(position);assert position.shape==(n,2) and heading.shape==(n,)
    assert motors.shape==noise.shape==(n,3)
    assert all(np.isfinite(v).all() for v in (position,heading,motors,noise))
    interval=cfg['action_repeat']/60.;gain=1-math.exp(-interval/.25)
    target=np.asarray(_TARGETS)*[cfg['v_max'],cfg['v_max'],cfg['rot_max']]
    new_motors=motors[:,None,:]+(target[None,:,:]+noise[:,None,:]*[.03,.03,.15]-motors[:,None,:])*gain
    angle=(heading[:,None]+new_motors[:,:,2]*interval+np.pi)%(2*np.pi)-np.pi
    displacement=np.stack((new_motors[:,:,0]*np.cos(angle)-new_motors[:,:,1]*np.sin(angle),
        new_motors[:,:,0]*np.sin(angle)+new_motors[:,:,1]*np.cos(angle)),axis=2)*interval
    target_position=position[:,None,:]+displacement
    endpoint=np.clip(target_position,-cfg['arena_half']+.08,cfg['arena_half']-.08)
    wall=np.any(endpoint!=target_position,axis=2)
    new_motors[:,:,:2]*=np.where(wall,.2,1.)[:,:,None]
    return endpoint,angle,new_motors


def action_metrics(position,goal,endpoints,probabilities):
    p=np.asarray(probabilities,dtype=np.float64);position=np.asarray(position,dtype=np.float64)
    goal=np.asarray(goal,dtype=np.float64);endpoints=np.asarray(endpoints,dtype=np.float64)
    n=len(position);assert p.shape==(n,6) and endpoints.shape==(n,6,2) and goal.shape==(n,2)
    assert np.isfinite(p).all() and (p>=0).all() and np.max(np.abs(p.sum(1)-1))<2e-6
    progress=np.linalg.norm(position-goal,axis=1)[:,None]-np.linalg.norm(endpoints-goal[:,None,:],axis=2)
    distances=np.linalg.norm(endpoints-position[:,None,:],axis=2)
    expected=np.sum(p*progress,axis=1);mass=p.sum(1)
    assert np.all(expected>=progress.min(1)*mass-1e-12) and np.all(expected<=progress.max(1)*mass+1e-12)
    return dict(per_action_goalward_progress_m=progress,per_action_step_distance_m=distances,
        expected_goalward_progress_m=expected,uniform_goalward_progress_m=progress.mean(1),
        best_six_same_noise_progress_m=progress.max(1),expected_step_distance_m=np.sum(p*distances,axis=1),
        worst_six_same_noise_progress_m=progress.min(1),six_action_progress_range_m=np.ptp(progress,axis=1),
        probability_sum=p.sum(1),translation_probability=p[:,2:].sum(1))


def analyze_context(raw,rows,plan,cfg,variants=VARIANTS,first_free=None,controller='policy'):
    """Reconcile every first action, retaining the saved controller probabilities."""
    count=16*len(variants);assert len(rows)==count
    indexed={(r['variant'],r['heading_index']):r for r in rows};assert len(indexed)==count
    first=raw['first_free'] if first_free is None else first_free
    assert set(first)==set(range(count))
    schedule=task.schedule(plan,'cpu');start=int(schedule['free'][0]);assert bool((schedule['free']==start).all())
    assert bool(raw['policy_mask'][start].all()) and bool(raw['active'][start].all())
    position=np.stack([first[i]['position'].numpy() for i in range(count)])
    heading=np.array([float(first[i]['heading']) for i in range(count)])
    motors=np.stack([first[i]['motors'].numpy() for i in range(count)])
    for j in range(1,len(variants)):
        assert np.array_equal(position[:16],position[j*16:(j+1)*16])
        assert np.array_equal(heading[:16],heading[j*16:(j+1)*16])
        assert np.array_equal(motors[:16],motors[j*16:(j+1)*16])
    noise=plan['motor'][start].repeat(1,len(variants)).T.numpy()
    goal=plan['goal'].repeat(len(variants),1).numpy()
    ordered=[indexed[(variants[i//16],i%16)] for i in range(count)]
    assert all(r['context']==int(plan['context'][0]) and not r.get('invalid',False) for r in ordered)
    # Uniform/planner files omit first_free; their stored row probabilities are overrides.
    probabilities=np.asarray([r['first_free_probabilities'] for r in ordered],dtype=np.float64)
    if controller=='policy':
        assert np.array_equal(probabilities,np.stack([first[i]['probabilities'].numpy() for i in range(count)]))
    endpoint,angle,new_motors=first_action_physics(position,heading,motors,noise,cfg)
    values=action_metrics(position,goal,endpoint,probabilities)
    action=raw['action'][start].numpy();assert ((action>=0)&(action<6)).all()
    predicted=endpoint[np.arange(count),action];actual=raw['position'][start].numpy()
    error=np.max(np.abs(predicted-actual),axis=1)
    assert float(error.max())<=POSITION_ATOL, f'First-action native-physics disagreement: {float(error.max())}'
    bearing=np.arctan2(goal[:,1]-position[:,1],goal[:,0]-position[:,0])
    relative=(bearing-heading+np.pi)%(2*np.pi)-np.pi
    result=[]
    for i,row in enumerate(ordered):
        entry=dict(context=row['context'],heading_index=row['heading_index'],variant=row['variant'],controller=controller,
            valid=True,first_free_global_step=start,body_heading_rad=float(heading[i]),
            body_relative_goal_bearing_rad=float(relative[i]),position=position[i].tolist(),goal=goal[i].tolist(),
            motors=motors[i].tolist(),saved_motor_noise=noise[i].tolist(),probabilities=probabilities[i].tolist(),
            counterfactual_endpoints=endpoint[i].tolist(),chosen_action=int(action[i]),
            predicted_chosen_endpoint=predicted[i].tolist(),saved_chosen_endpoint=actual[i].tolist(),
            chosen_position_max_abs_error_m=float(error[i]),
            observed_first_progress_m=float(np.linalg.norm(position[i]-goal[i])-np.linalg.norm(actual[i]-goal[i])),
            recorded_30_action_hit=bool(row['hit']),recorded_free_steps=row['free_steps'],
            recorded_30_action_progress_m=row['progress_m'],recorded_30_action_path_m=row['free_path_m'])
        entry.update({key:value[i].tolist() for key,value in values.items()})
        entry['expected_minus_uniform_progress_m']=entry['expected_goalward_progress_m']-entry['uniform_goalward_progress_m']
        result.append(entry)
    return result


def invalid_rows(rows,controller,reason):
    return [dict(context=r['context'],heading_index=r['heading_index'],variant=r['variant'],controller=controller,
        valid=False,diagnostic_unavailable_reason=reason,recorded_30_action_hit=bool(r['hit']),
        recorded_30_action_progress_m=r.get('progress_m'),recorded_30_action_path_m=r.get('free_path_m')) for r in rows]


MEASURES=('expected_goalward_progress_m','uniform_goalward_progress_m','best_six_same_noise_progress_m',
    'worst_six_same_noise_progress_m','six_action_progress_range_m',
    'expected_minus_uniform_progress_m','expected_step_distance_m','translation_probability',
    'observed_first_progress_m','recorded_30_action_progress_m','recorded_30_action_path_m')


def summarize(rows,expected_contexts):
    assert sorted((r['context'],r['heading_index']) for r in rows)==[(c,h) for c in expected_contexts for h in range(16)]
    valid=[r for r in rows if r['valid']]
    result=dict(trials=len(rows),valid_trials=len(valid),undefined_trials=len(rows)-len(valid),
        recorded_30_action_successes=sum(r['recorded_30_action_hit'] for r in rows),
        means_over_valid_trials={k:float(np.mean([r[k] for r in valid])) if valid else None for k in MEASURES})
    if len(expected_contexts)==1:
        p=np.asarray([r['probabilities'] for r in valid])
        result['probability_phase_vector_RMS']=None if not valid else float(np.sqrt(np.mean(np.sum((p-p.mean(0))**2,axis=1))))
        result['mean_action_probabilities']=None if not valid else p.mean(0).tolist()
    return result


def aggregate(rows):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('condition','bank','stage','controller','variant'))
        groups.setdefault(key,[]).append(row)
    result=[];contexts=[]
    for key,values in groups.items():
        tags=dict(zip(('condition','bank','stage','controller','variant'),key))
        result.append(dict(**tags,**summarize(values,range(16))))
        for c in range(16):contexts.append(dict(**tags,context=c,**summarize([r for r in values if r['context']==c],[c])))
    return result,contexts


def require_complete_pair(reports):
    assert set(reports)==set(CONDITIONS)
    for condition,record in reports.items():
        assert record['condition']==condition and record['passed'] and record['complete']
        assert record['status']=='complete' and not record['failures'] and not record['pending']
        assert record['checks']['final_exact_stage4'] and record['checks']['final_summary']


def load_complete():
    frozen=read(PROTOCOL);assert frozen.pop('frozen_utc') and frozen==protocol()
    assert sha(TRAINING)==TRAINING_HASH
    design=read(TRAINING);routing=read(ROOT/'docs/research/local_goal_return_training_launch.json')
    routes,_=routed_attempts(routing);hashes={};checked={}
    def track(path,digest=None):
        path=Path(path);actual=sha(path);assert digest is None or actual==digest,path
        hashes[str(path.relative_to(ROOT))]=actual;return path
    track(PROTOCOL);track(TRAINING);track(ROOT/'docs/research/local_goal_return_training_launch.json')
    for name,digest in {**design['source_sha256'],**design['original_assets'],**design['passed_baseline_assets']}.items():track(ROOT/name,digest)
    for name in SOURCES:track(ROOT/name)
    # Neither population's outcomes are interpreted until BOTH final audits pass.
    reports={c:read(track(ROOT/f'results/local_goal_return_design/cloud_audit/{c}.json')) for c in CONDITIONS}
    require_complete_pair(reports)
    for route in routes:
        condition=route['condition'];folder=ROOT/route['local_artifact_root']/route['artifact_subdir']
        audit=reports[condition];verify_final_audit(ROOT,folder,condition,audit)
        for name,item in audit['artifacts'].items():track(folder/name,item['sha256'])
        checked[condition]=(folder,read(folder/'training_config.json'))
    baseline_audit=read(track(ROOT/'results/local_goal_return_design/completed_baseline_audit.json'))
    assert baseline_audit['passed'] and baseline_audit['complete'] and baseline_audit['artifact_integrity_passed']
    assert baseline_audit['source_sha256']==sha(ROOT/'scripts/audit_local_goal_return_baseline.py')
    assert baseline_audit['behavior_protocol_sha256']==evaluator.BEHAVIOR_HASH
    track(ROOT/'results/local_goal_return_design/native_resume_validation.json',baseline_audit['native_validation_sha256'])
    banks={name:evaluator.load_bank(name)[0] for name in ('near_baseline','final_hard')}
    behavior=read(evaluator.BEHAVIOR);track(evaluator.BEHAVIOR,evaluator.BEHAVIOR_HASH)
    for item in behavior['banks'].values():track(ROOT/item['path'],item['sha256'])
    rows=[];sources=[]
    for condition in CONDITIONS:
        folder,cfg=checked[condition]
        evaluations=[(0,name,ROOT/f'results/local_goal_return_baseline/{condition}/{name}') for name in banks]
        evaluations += [(stage,'final_hard',folder/f'stage{stage}_evaluation') for stage in range(1,5)]
        for stage,bank,path in evaluations:
            evaluation=read(track(path/'summary.json'))
            assert evaluation['complete'] and evaluation['condition']==condition and evaluation['bank']==bank
            assert set(evaluation['variants'])==set(VARIANTS) and len(evaluation['contexts'])==16
            assert evaluation['source_sha256']==sha(evaluator.__file__) and evaluation['core_sha256']==sha(task.__file__)
            assert evaluation['bank_sha256']==behavior['banks'][bank]['sha256']
            sources.append(dict(condition=condition,stage=stage,bank=bank,path=str((path/'summary.json').relative_to(ROOT))))
            for context in range(16):
                child=path/f'context{context}';record=read(track(child/'result.json'))
                assert record==evaluation['contexts'][context] and record['context']==context
                reference=read(track(child/'input.json'));plan=evaluator.context_plan(banks[bank],context)
                if stage==0:
                    audited=baseline_audit['models'][condition]['banks'][bank]['contexts'][context]
                    assert audited['context']==context and audited['valid']==record['valid']
                    for name,digest in audited['source_hashes'].items():track(child/name,digest)
                tags=dict(condition=condition,bank=bank,stage=stage,source=str((child/'result.json').relative_to(ROOT)))
                raw=None
                if record['valid']:
                    raw=torch.load(track(child/'raw.pt',record['raw_sha256']),map_location='cpu',weights_only=True)
                    assert raw['plan_reference']==reference
                    local=analyze_context(raw,record['rows'],plan,cfg)
                else:
                    track(child/'failed_plan.pt')
                    local=invalid_rows(record['rows'],'policy','Recorded physical/policy collection failed; no first-action value imputed')
                rows += [dict(**tags,**row) for row in local]
                if stage==0:
                    assert set(record['controls'])=={'uniform','planner'}
                    for name,control in record['controls'].items():
                        if control['valid'] and raw is not None:
                            control_raw=torch.load(track(child/f'{name}_raw.pt'),map_location='cpu',weights_only=True)
                            first={i:raw['first_free'][i] for i in range(16)}
                            local=analyze_context(control_raw,control['rows'],plan,cfg,('blind_native',),first,name)
                        else:
                            local=invalid_rows(control['rows'],name,'Recorded control failed or shared first-FREE physical snapshot unavailable')
                        rows += [dict(**tags,**row) for row in local]
    assert len(rows)==38912 and sum(r['controller']=='policy' for r in rows)==36864
    summary,context_rows=aggregate(rows)
    return dict(version='local_return_saved_first_action_use_v1',complete=True,protocol_sha256=sha(PROTOCOL),
        source_sha256=sha(__file__),input_sha256=hashes,rows=rows,summaries=summary,contexts=context_rows,
        evaluation_sources=sources,maximum_chosen_position_error_m=max((r['chosen_position_max_abs_error_m'] for r in rows if r['valid']),default=None),
        scope=protocol()['interpretation'],no_model_or_environment_execution=True,original_seed_pairs=1)


def render(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.9),sharey=True,layout='constrained')
    labels={'gc100':'Grid','pgc33':'PGC + grid'}
    variants=[('blind_native','Native cue','#d87330','o'),('blind_ordinary','Ordinary cue','#357eb0','s'),
              ('blind_reset_after_glimpse','Erase after cue','#74648d','^')]
    for ax,condition in zip(axes,CONDITIONS):
        lookup={(r['stage'],r['variant']):r for r in summary['summaries']
            if r['condition']==condition and r['bank']=='final_hard' and r['controller']=='policy'}
        def values(variant,key):
            return [np.nan if lookup[(stage,variant)]['means_over_valid_trials'][key] is None else
                1000*lookup[(stage,variant)]['means_over_valid_trials'][key] for stage in range(5)]
        for variant,label,color,marker in variants:
            ax.plot(range(5),values(variant,'expected_goalward_progress_m'),color=color,marker=marker,lw=1.4,
                ms=6 if variant=='blind_native' else 4,
                markerfacecolor='none' if variant=='blind_native' else color,
                zorder=3 if variant=='blind_native' else 2,label=label)
        ax.plot(range(5),values('blind_native','uniform_goalward_progress_m'),color='.5',ls='--',lw=1,label='Uniform mean')
        ax.plot(range(5),values('blind_native','best_six_same_noise_progress_m'),color='.35',ls=':',lw=1.2,label='Best of 6 (hindsight)')
        available=[lookup[(stage,variant)]['valid_trials'] for stage in range(5) for variant,_,_,_ in variants]
        coverage=str(min(available)) if min(available)==max(available) else f'{min(available)}–{max(available)}'
        ax.set(title=f'{labels[condition]}\nAvailable trials per point: {coverage} / 256')
        ax.set_xticks(range(5),['New\nheads','Stage 1','Stage 2','Stage 3','Stage 4'])
        ax.axhline(0,color='.7',lw=.6);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13)
    axes[0].set_ylabel('Expected first-step goalward progress (mm)')
    handles,labels_=axes[0].get_legend_handles_labels();fig.legend(handles,labels_,loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.suptitle('First-step goalward progress',fontsize=14)
    if summary.get('synthetic'):fig.text(.5,.5,'SYNTHETIC — NOT RESULTS',ha='center',fontsize=26,rotation=20,color='#a21b1b',alpha=.25)
    for suffix in ('png','pdf'):fig.savefig(out/f'first_action_use.{suffix}',dpi=180)
    plt.close(fig)


def write_outputs(summary,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,values in (('rows',summary['rows']),('contexts',summary['contexts']),('summaries',summary['summaries'])):
        (out/f'{name}.json').write_text(json.dumps(values,indent=2)+'\n')
        fields=list(dict.fromkeys(k for row in values for k in row))
        with (out/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in values:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    (out/'summary.json').write_text(json.dumps({k:v for k,v in summary.items() if k not in ('rows','contexts')},indent=2)+'\n')
    render(summary,out)
    import os
    relative=lambda path:os.path.relpath(ROOT/path,out)
    undefined=sum(not r['valid'] for r in summary['rows'])
    lines=['# First-step action use','',
        'This secondary diagnostic asks whether the saved first free action probabilities favor immediate movement toward the goal. It was prepared after the stage-1–3 success counts were seen. Both prescribed final models and every earlier fixed endpoint are retained.','',
        '![First-step goalward progress](first_action_use.png)','',
        'The curves use the same sealed hard bank at initialization and after each of four training stages: 16 contexts × 16 directions per policy condition. At each saved first-FREE pose, the native motor equations predict all six possible endpoints under the same recorded motor-noise draw. The literal saved probabilities weight their goal-center distance changes. The selected action is checked against the recorded first-step position for every available row. No new policy, environment or random stream is executed.','',
        'Uniform averages the same six physical changes. Best of six knows the realized noise and is a hindsight physical bound, not an option available to the agent. Existing baseline uniform/planner controller runs are also retained in the full tables with their actual overridden probabilities.','',
        '**This measures immediate alignment.** A useful turn can have little immediate progress; first-step alignment is distinct from long-horizon navigation, value, or ring dynamics. Saved 30-action successes, progress and path length remain alongside every row and in the primary training report. A low value alone does not identify a failed heading representation.','',
        'The main figure shows native, ordinary and post-cue-erasure conditions for both models. Complete tables retain all 12 policy variants, near/hard baseline banks, baseline controls, and all contexts and directions. Missing physical collections remain undefined rows with valid/undefined denominators; they are not imputed as zero progress. The descriptive variation of the six probabilities across 16 headings is a vector RMS within each context, not a decoder test or an independent-seed statistic. Body-relative goal angles and every action probability are stored without fitting an angular map.','',
        f"Defined first-action measurements: **{len(summary['rows'])-undefined} / {len(summary['rows'])} trial records**; **{undefined} undefined records** are retained. Progress means use available measurements, with valid/undefined counts in every condition and context. Each figure panel reports its range of available trials per point. These are repeated measurements of one original seed pair.",'',
        f"Largest first-step coordinate reconciliation error: {summary['maximum_chosen_position_error_m']} m (fixed tolerance {POSITION_ATOL} m).",'',
        '[All trials and six actions](rows.json) · [Every context](contexts.json) · [All conditions](summaries.json) · [Hashes and source links](summary.json)', '',
        f"[Primary 30-action behavior report]({relative('results/ring_local_goal_return/report/README.md')}) · [Frozen secondary protocol]({relative(str(PROTOCOL.relative_to(ROOT)))})",'']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    if args.freeze_only:
        assert not PROTOCOL.exists();PROTOCOL.write_text(json.dumps(dict(protocol(),frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat()),indent=2)+'\n');return
    assert not args.out.exists() or not any(args.out.iterdir()),'Use a fresh output directory'
    summary=load_complete();write_outputs(summary,args.out)
    print(json.dumps(dict(complete=True,rows=len(summary['rows']),out=str(args.out))))


if __name__=='__main__':main()
