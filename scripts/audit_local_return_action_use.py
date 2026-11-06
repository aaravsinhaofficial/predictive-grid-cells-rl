"""Independent reconciliation of completed saved first-action diagnostics.

Reads tensors and JSON only. No policy/environment execution or producer math.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

BASE=ROOT/'results/local_return_action_use'
VARIANTS=tuple(f'{view}_{cue}' for view in ('blind','seen') for cue in
    ('native','ordinary','plus60','minus60','reset_after_glimpse','reset_every_blind'))
TARGETS=np.array([[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],dtype=float)
MEASURES=('expected_goalward_progress_m','uniform_goalward_progress_m','best_six_same_noise_progress_m',
    'worst_six_same_noise_progress_m','six_action_progress_range_m','expected_minus_uniform_progress_m',
    'expected_step_distance_m','translation_probability','observed_first_progress_m',
    'recorded_30_action_progress_m','recorded_30_action_path_m')
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def exact(a,b):assert np.array_equal(np.asarray(a),np.asarray(b))


def close(a,b,atol=2e-12):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a,b,atol=atol,rtol=1e-11),f'{a} != {b}'


def audit(base=BASE):
    base=Path(base);summary=read(base/'summary.json');protocol=ROOT/'docs/research/local_return_action_use_protocol.json'
    assert summary['complete'] and summary['no_model_or_environment_execution']
    assert summary['protocol_sha256']==sha(protocol)
    frozen=read(protocol)
    for name,digest in frozen['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert summary['source_sha256']==sha(ROOT/'scripts/analyze_local_return_action_use.py')
    for name,digest in summary['input_sha256'].items():assert sha(ROOT/name)==digest,name
    for condition in ('gc100','pgc33'):
        report=read(ROOT/f'results/local_goal_return_design/cloud_audit/{condition}.json')
        assert report['condition']==condition and report['complete'] and report['passed'] and not report['failures']
    paths=[base/name for name in ('summary.json','rows.json','contexts.json','summaries.json','README.md','first_action_use.png','first_action_use.pdf')]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    rows=read(base/'rows.json');contexts=read(base/'contexts.json');summaries=read(base/'summaries.json')
    assert summaries==summary['summaries'] and len(rows)==38912 and len(summaries)==152 and len(contexts)==2432
    assert all(r['valid'] for r in rows),'Completed producer reports every row valid; no exclusions permitted'
    assert sum(r['controller']=='policy' for r in rows)==36864
    behavior=read(ROOT/'results/local_goal_return_design/behavior_protocol.json')
    banks={name:torch.load(ROOT/item['path'],map_location='cpu',weights_only=True)
        for name,item in behavior['banks'].items()}
    # Saved bank containers carry the exact complete plan under `plan`.
    banks={name:value.get('plan',value) for name,value in banks.items()}
    cfgs={c:read(ROOT/f'models/{c}/{c}_lstm_s1_1000M.json') for c in ('gc100','pgc33')}
    groups={};identities=set()
    for row in rows:
        key=tuple(row[k] for k in ('condition','bank','stage','controller','variant','context','heading_index'))
        assert key not in identities;identities.add(key)
        groups.setdefault((row['source'],row['controller']),[]).append(row)
    maximum_error=maximum_metric_error=0.;checked=0
    for (source,controller),group in groups.items():
        folder=(ROOT/source).parent;record=read(ROOT/source);first_row=group[0]
        condition=first_row['condition'];bank=first_row['bank'];context=first_row['context'];plan=banks[bank];cfg=cfgs[condition]
        assert record['valid'] and record['context']==context
        native=torch.load(folder/'raw.pt',map_location='cpu',weights_only=True)
        if controller=='policy':raw=native;source_rows=record['rows'];variants=VARIANTS
        else:
            assert controller in ('uniform','planner') and first_row['stage']==0
            assert record['controls'][controller]['valid']
            raw=torch.load(folder/f'{controller}_raw.pt',map_location='cpu',weights_only=True)
            source_rows=record['controls'][controller]['rows'];variants=('blind_native',)
        indexed={(r['variant'],r['heading_index']):r for r in source_rows}
        assert len(group)==len(source_rows)==16*len(variants)
        free=15+int(plan['duration_a'][16*context])+int(plan['duration_b'][16*context])+4+int(plan['delay'][16*context])
        for row in group:
            h=row['heading_index'];j=16*variants.index(row['variant'])+h;k=16*context+h
            source_row=indexed[(row['variant'],h)];first=native['first_free'][j if controller=='policy' else h]
            assert row['first_free_global_step']==free and bool(raw['policy_mask'][free,j]) and bool(raw['active'][free,j])
            exact(row['position'],first['position']);exact(row['motors'],first['motors']);exact(row['body_heading_rad'],float(first['heading']))
            exact(row['saved_motor_noise'],plan['motor'][free,:,k]);exact(row['goal'],plan['goal'][k])
            exact(row['probabilities'],source_row['first_free_probabilities'])
            if controller=='policy':exact(row['probabilities'],first['probabilities'])
            position=np.asarray(row['position']);goal=np.asarray(row['goal']);motors=np.asarray(row['motors'])
            noise=np.asarray(row['saved_motor_noise']);p=np.asarray(row['probabilities']);heading=row['body_heading_rad']
            interval=cfg['action_repeat']/60;decay=math.exp(-interval/.25)
            target=TARGETS*np.array([cfg['v_max'],cfg['v_max'],cfg['rot_max']])+noise*[.03,.03,.15]
            velocity=decay*motors+(1-decay)*target
            angle=heading+interval*velocity[:,2]
            displacement=interval*np.column_stack((velocity[:,0]*np.cos(angle)-velocity[:,1]*np.sin(angle),
                velocity[:,0]*np.sin(angle)+velocity[:,1]*np.cos(angle)))
            endpoint=np.minimum(np.maximum(position+displacement,-cfg['arena_half']+.08),cfg['arena_half']-.08)
            close(row['counterfactual_endpoints'],endpoint)
            chosen=int(raw['action'][free,j]);assert row['chosen_action']==chosen
            sampled=min(int((plan['action'][free,k]>torch.tensor(p,dtype=torch.float32).cumsum(0)).sum()),5)
            assert sampled==chosen
            actual=raw['position'][free,j].numpy().astype(float);exact(row['saved_chosen_endpoint'],actual)
            close(row['predicted_chosen_endpoint'],endpoint[chosen])
            error=float(np.abs(actual-endpoint[chosen]).max());assert error<=2e-6
            close(row['chosen_position_max_abs_error_m'],error);maximum_error=max(maximum_error,error)
            distance=np.linalg.norm(position-goal);progress=distance-np.linalg.norm(endpoint-goal,axis=1)
            step=np.linalg.norm(endpoint-position,axis=1);expected=float(np.dot(p,progress));uniform=float(progress.mean())
            metrics=dict(per_action_goalward_progress_m=progress,per_action_step_distance_m=step,
                expected_goalward_progress_m=expected,uniform_goalward_progress_m=uniform,
                best_six_same_noise_progress_m=float(progress.max()),worst_six_same_noise_progress_m=float(progress.min()),
                six_action_progress_range_m=float(progress.max()-progress.min()),expected_minus_uniform_progress_m=expected-uniform,
                expected_step_distance_m=float(np.dot(p,step)),translation_probability=float(p[2:].sum()),probability_sum=float(p.sum()),
                # These two descriptive fields retain native float32 inputs in
                # the producer; the six counterfactual metrics use float64.
                observed_first_progress_m=float(np.linalg.norm(position.astype(np.float32)-goal.astype(np.float32))-
                    np.linalg.norm(actual.astype(np.float32)-goal.astype(np.float32))))
            delta=goal.astype(np.float32)-position.astype(np.float32)
            bearing=float(np.arctan2(delta[1],delta[0]));relative=(bearing-heading+math.pi)%(2*math.pi)-math.pi
            close(row['body_relative_goal_bearing_rad'],relative)
            for name,value in metrics.items():
                close(row[name],value);maximum_metric_error=max(maximum_metric_error,float(np.max(np.abs(np.asarray(row[name])-value))))
            assert (p>=0).all() and abs(p.sum()-1)<2e-6
            assert progress.min()*p.sum()-1e-12<=expected<=progress.max()*p.sum()+1e-12
            for new,old in (('recorded_30_action_hit','hit'),('recorded_free_steps','free_steps'),
                ('recorded_30_action_progress_m','progress_m'),('recorded_30_action_path_m','free_path_m')):assert row[new]==source_row[old]
            checked+=1
    assert checked==38912
    pooled={};bycontext={}
    for row in rows:
        key=tuple(row[k] for k in ('condition','bank','stage','controller','variant'))
        pooled.setdefault(key,[]).append(row);bycontext.setdefault(key+(row['context'],),[]).append(row)
    for records,index,n in ((summaries,pooled,256),(contexts,bycontext,16)):
        seen=set()
        for record in records:
            key=tuple(record[k] for k in ('condition','bank','stage','controller','variant'))
            if n==16:key+=(record['context'],)
            assert key not in seen;seen.add(key);group=index[key]
            assert len(group)==record['trials']==record['valid_trials']==n and record['undefined_trials']==0
            assert len({(r['context'],r['heading_index']) for r in group})==n
            assert record['recorded_30_action_successes']==sum(r['recorded_30_action_hit'] for r in group)
            assert set(record['means_over_valid_trials'])==set(MEASURES)
            for name in MEASURES:close(record['means_over_valid_trials'][name],sum(r[name] for r in group)/n)
            if n==16:
                p=np.array([r['probabilities'] for r in group]);mean=p.mean(0)
                close(record['mean_action_probabilities'],mean)
                close(record['probability_phase_vector_RMS'],np.sqrt(np.sum((p-mean)**2)/16))
        assert seen==set(index)
    close(summary['maximum_chosen_position_error_m'],maximum_error)
    for name,digest in before.items():assert sha(ROOT/name)==digest,'Output changed during read-only audit'
    native=[r for r in summaries if r['bank']=='final_hard' and r['controller']=='policy' and r['variant']=='blind_native']
    return dict(passed=True,complete=True,audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        protocol_sha256=summary['protocol_sha256'],producer_sha256=summary['source_sha256'],output_sha256=before,
        input_files_hash_checked=len(summary['input_sha256']),rows_checked=checked,valid_rows=checked,undefined_rows=0,
        policy_rows=36864,baseline_reference_rows=2048,pooled_summaries_checked=152,context_summaries_checked=2432,
        all_six_action_counterfactuals_checked=6*checked,maximum_chosen_position_error_m=maximum_error,
        maximum_independent_metric_error=maximum_metric_error,native_hard_summaries=native,
        scope='Independent scalar/vector arithmetic and every raw firstFREE/action/pose/noise/goal linkage. No policy/environment execution, new neural probe, optimization, producer numerical helper or output modification. JSON data and figure/report file hashes checked; CSV serialization and plot pixels were not independently recomputed.',
        interpretation='Near-zero signed mean first-step progress indicates little net immediate goalward alignment on the uniformly distributed headings. It is not proof that all individual decisions are uninformative, nor a test of long-horizon planning or ring structure. Full-response path length measures movement; negative mean goal progress means average final position is farther from goal, not that every trajectory moves away. Noise-informed best-of-six is hindsight, and phases/contexts are not independent trained seeds.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--out',type=Path,default=ROOT/'results/local_return_action_use_review/completed_output_audit.json')
    args=p.parse_args();torch.set_num_threads(2);result=audit(args.base)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','complete','rows_checked','maximum_chosen_position_error_m','maximum_independent_metric_error')}))


if __name__=='__main__':main()
