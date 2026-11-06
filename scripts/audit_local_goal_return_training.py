"""Read-only audit of collected, fixed-budget local-goal-return training.

No learner/environment execution. Training noise replay is checked from every
recorded update; clean evaluation physics is independently reconstructed from
saved actions and tapes. Passed integrity is distinct from a behavioral gate.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import audit_local_goal_return_baseline as base
import local_goal_return_training as task
import local_goal_return_training_evaluate as evaluator
import local_goal_return_training_certificate as certificate

PROTOCOL=ROOT/'docs/research/local_goal_return_training_protocol.json'
PROTOCOL_HASH='0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
COUNTERS=('active_physics_updates','forced_updates','free_updates','padded_slots','setup_pickups','free_hits')
sha=base.sha;read=base.read;equal=base.same_tree


def finite(value):
    if isinstance(value,torch.Tensor):
        assert bool(torch.isfinite(value).all()), 'Nonfinite tensor'
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v)
    elif isinstance(value,float):assert math.isfinite(value), 'Nonfinite scalar'


def require_equal(a,b,message):
    assert equal(a,b),message
    return True


def audit_metric(row):
    finite(row);u=row['update'];stage=(u-1)//1024+1
    assert 1<=u<=4096 and row['stage']==stage
    assert row['completed_trials']==row['setup_pickups']==32 and row['setup_raw_reward']==320
    assert row['setup_learning_reward']==row['forced_hits']==0
    assert row['terminal_bootstrap_zero'] and row['controller']=='policy'
    assert row['allocated_slots']==5120
    assert row['active_physics_updates']==row['forced_updates']+row['free_updates']
    assert row['active_physics_updates']+row['padded_slots']==5120
    assert 32<=row['free_updates']<=960 and 0<=row['free_hits']<=32
    delay=task.STAGES[stage-1].delay
    assert 32*(29+(delay+1)//2)<=row['forced_updates']<=32*(39+delay)
    assert row['clearance_min']>.01 and row['wall_clearance_min']>.01 and 0<=row['snap_max_m']<1e-5
    assert 0<=row['replay_error']<3e-4 and row['grad_norm']>=0
    assert 0<=row['entropy_per_free_decision']<=math.log(6)+2e-6
    expected=row['entropy_per_free_decision']*row['free_updates']/32
    assert math.isclose(expected,row['entropy_sum_per_trial'],rel_tol=3e-6,abs_tol=3e-6)
    expected=(100/160)*(row['policy_loss']+.5*row['value_loss']-.01*row['entropy_sum_per_trial'])
    assert math.isclose(row['loss'],expected,rel_tol=2e-5,abs_tol=2e-5)
    assert row['value_loss']>=0


def accepted_metrics(folder,cutoff=None,ancestor='resume_parent_evidence'):
    """Preserve rolled-back attempt rows, count only the exact accepted prefix."""
    folder=Path(folder);prior=[];start=0;discarded=0
    if (folder/'resume_lineage.json').exists():
        lineage=read(folder/'resume_lineage.json');start=lineage['restored_updates']
        prior,meta=accepted_metrics(folder/ancestor,start,'ancestor_evidence')
        discarded+=meta['rolled_back_rows']
    lines=(folder/'metrics.jsonl').read_bytes().splitlines();rows=[];partial=False
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            assert i==len(lines)-1, 'Malformed nonterminal metrics row'
            partial=True;break
        audit_metric(row)
        assert row['update']==start+len(rows)+1, 'Missing, duplicated or reordered update'
        rows.append(row)
    if cutoff is not None:
        assert cutoff>=start and start+len(rows)>=cutoff, 'Missing accepted parent metrics'
        discarded+=sum(r['update']>cutoff for r in rows)
        rows=[r for r in rows if r['update']<=cutoff]
    result=prior+rows
    assert [r['update'] for r in result]==list(range(1,len(result)+1))
    active=0
    for row in result:
        active+=row['active_physics_updates'];assert row['active_physics_frames']==active*4
    return result,dict(accepted_updates=len(result),rolled_back_rows=discarded,partial_last_row=partial)


def audit_rng(state,cuda=False):
    assert isinstance(state,torch.Tensor) and state.dtype==torch.uint8 and state.ndim==1 and state.numel()>0
    if not cuda:
        generator=torch.Generator();generator.set_state(state);assert torch.equal(generator.get_state(),state)


def audit_checkpoint(data,baseline,provenance,updates,stages):
    finite(data);cfg=baseline['config'];c=data['counters']
    assert data['version']=='local_return_exact_boundary_resume_v1' and data['no_inflight_trial'] is True
    assert data['boundary_state']=='Everytrialterminated; nexttrialh/czero,pa−1/pr0'
    assert data['config']==cfg and data['initialization']==baseline['initialization']
    assert data['provenance']==provenance
    assert c['updates']==updates and c['trials']==updates*32 and c['completed_stages']==stages
    assert len(c['evaluation_records'])==stages
    assert c['setup_pickups']==c['trials']
    assert c['active_physics_updates']==c['forced_updates']+c['free_updates']
    assert c['active_physics_updates']+c['padded_slots']==updates*5120
    assert data['active_physics_frames']==c['active_physics_updates']*4
    assert data['completed']==(updates==4096 and stages==4)
    assert data['policy'].keys()==baseline['policy'].keys()
    for key,value in data['policy'].items():
        assert value.shape==baseline['policy'][key].shape and value.dtype==baseline['policy'][key].dtype
    optimizer=data['optimizer'];assert len(optimizer['param_groups'])==1
    group=optimizer['param_groups'][0]
    for key,value in dict(lr=3e-4,alpha=.99,eps=1e-5,weight_decay=0,momentum=0,centered=False,
                          capturable=False,maximize=False,differentiable=False,foreach=None).items():
        assert group[key]==value, f'RMSProp setting changed: {key}'
    assert group['params']==list(range(len(data['policy'])))
    assert set(optimizer['state'])==(set(group['params']) if updates else set())
    for index,record in optimizer['state'].items():
        assert set(record)=={'step','square_avg'} and float(record['step'])==updates
        value=list(data['policy'].values())[index]
        assert record['square_avg'].shape==value.shape and record['square_avg'].dtype==value.dtype
        assert bool((record['square_avg']>=0).all())
    stream=data['streams'];assert stream['seed']==294801 and stream['batches']==updates and stream['contexts']==updates*32
    assert set(stream['generators'])=={'plan','motor','action','state'}
    for state in stream['generators'].values():audit_rng(state)
    rng=data['global_rng'];assert set(rng)=={'cpu','cuda'};audit_rng(rng['cpu'])
    assert len(rng['cuda'])==(1 if provenance['device'].startswith('cuda') else 0)
    for state in rng['cuda']:audit_rng(state,cuda=True)
    if updates==0:
        assert equal(data['policy'],baseline['policy']), 'Incoming policy differs from exact completed baseline'
        assert equal(stream,task.TrialStreams(294801).snapshot()), 'Incoming private streams not fresh'
        assert all(c[k]==0 for k in COUNTERS)
    return dict(updates=updates,trials=c['trials'],completed_stages=stages,
        active_physics_frames=data['active_physics_frames'],policy_sha256=base.policy_hash(data['policy']),
        optimizer_all_parameters_and_steps_valid=True,private_and_global_rng_schema_valid=True,
        exact_baseline_initialization=updates==0)


def compare_evaluation_boundary(before,after,stage):
    for key in before:
        if key not in ('counters','completed'):assert equal(before[key],after[key]),f'Evaluation changed {key}'
    expected=dict(before['counters']);expected['completed_stages']=stage
    expected['evaluation_records']=after['counters']['evaluation_records']
    assert expected==after['counters']
    assert after['counters']['evaluation_records'][:-1]==before['counters']['evaluation_records']
    return True


def verify_actor(raw,policy,variants):
    """Float64 independent readout, with a float32 dot-product error bound."""
    first=raw['first_free'];n=16*len(variants);assert set(first)==set(range(n))
    finite(raw);h=torch.stack([first[i]['state'][:256] for i in range(n)]).double()
    weight=policy['pi.weight'].double();bias=policy['pi.bias'].double()
    expected=h@weight.T+bias;absolute=h.abs()@weight.abs().T+bias.abs()
    tolerance=2e-6+8*torch.finfo(torch.float32).eps*absolute
    logits=torch.stack([first[i]['logits'] for i in range(n)]).double()
    assert bool(((expected-logits).abs()<=tolerance).all()), 'Saved h and actor logits disagree'
    value=h@policy['v.weight'].double().T+policy['v.bias'].double()
    vbound=2e-6+8*torch.finfo(torch.float32).eps*(h.abs()@policy['v.weight'].double().abs().T+policy['v.bias'].double().abs())
    for i,f in first.items():
        assert f['state'].shape==(512,) and f['input'].shape==(1287,)
        assert torch.equal(f['input'][768:1280],raw['goal'][i])
        prob=logits[i].softmax(0)
        assert torch.allclose(prob,f['probabilities'].double(),atol=2e-6,rtol=2e-6)
        entropy=float(-(prob*prob.clamp_min(1e-300).log()).sum())
        assert abs(entropy-float(f['entropy']))<3e-6
        assert abs(float(value[i])-float(f['value']))<=float(vbound[i])
        for key in ('position','heading','motors'):assert torch.equal(f[key],first[i%16][key])
    for view in ('blind','seen'):
        offset=variants.index(view+'_native')*16
        for cue in ('ordinary','plus60','minus60','reset_after_glimpse','reset_every_blind'):
            other=variants.index(view+'_'+cue)*16
            for i in range(16):assert torch.equal(first[offset+i]['input'],first[other+i]['input'])
    return dict(actor_logit_max_abs_error=float((expected-logits).abs().max()),
        actor_float32_dot_product_bound=True,common_physical_start_and_available_free_input=True)


def audit_evaluation(folder,condition,checkpoint):
    report=read(folder/'summary.json');cfg=checkpoint['config'];variants=list(task.VARIANTS)
    plan,bank,_=evaluator.load_bank('final_hard')
    assert report['complete'] and report['condition']==condition and report['model_seed']==1
    assert report['bank']=='final_hard' and report['bank_sha256']==bank['sha256']
    assert report['behavior_protocol_sha256']==evaluator.BEHAVIOR_HASH
    assert report['source_sha256']==sha(evaluator.__file__) and report['core_sha256']==sha(task.__file__)
    assert report['checkpoint_policy_hash']==base.policy_hash(checkpoint['policy'])
    assert report['no_optimization'] and report['clean_state_noise_sigma']==0 and report['controls']=={}
    assert report['environment']['device']==checkpoint['provenance']['device']
    assert report['environment']['torch_version']==checkpoint['provenance']['torch_version']
    assert len(report['contexts'])==16 and set(report['variants'])==set(variants)
    rows=[];certs=[];contexts=[]
    for index in range(16):
        path=folder/f'context{index}';r=read(path/'result.json');chunk=evaluator.context_plan(plan,index)
        assert r==report['contexts'][index] and r['context']==index and 'controls' not in r
        reference=read(path/'input.json')
        assert reference==dict(bank='final_hard',path=bank['path'],sha256=bank['sha256'],context_index=index,
            context_ids=chunk['context'].tolist(),plan_generator_class_sha256=read(evaluator.BEHAVIOR)['generator_class_sha256'])
        checks={}
        if r['valid']:
            assert sha(path/'raw.pt')==r['raw_sha256']
            raw=torch.load(path/'raw.pt',map_location='cpu',weights_only=True);assert raw['plan_reference']==reference
            checks.update(base.verify_trajectory(chunk,cfg,raw,r['rows'],raw['first_free'],variants))
            checks.update(verify_actor(raw,checkpoint['policy'],variants))
            start=int(task.schedule(chunk,'cpu')['free'][0])
            indexed={(row['variant'],row['heading_index']):row for row in r['rows']}
            for j,f in raw['first_free'].items():
                row=indexed[(variants[j//16],j%16)]
                for field in ('probabilities','logits'):
                    assert row['first_free_'+field]==f[field].tolist()
                for field in ('entropy','value'):
                    assert row['first_free_'+field]==float(f[field])
                # Float32 cumulative sums may differ by a few ulps across devices.
                action=int(raw['action'][start,j]);prob=f['probabilities'].double()
                uniform=float(chunk['action'][start,j%16])
                assert uniform>=float(prob[:action].sum())-2e-6
                assert action==5 or uniform<=float(prob[:action+1].sum())+2e-6
            a=r['audit'];assert a['setup_pickups']==192 and a['setup_raw_reward']==1920
            assert a['setup_learning_reward']==a['forced_hits']==0 and a['terminal_bootstrap_zero']
            assert a['active_physics_updates']==int(raw['active'].sum()) and a['free_updates']==int(raw['policy_mask'].sum())
            assert a['active_physics_updates']==a['free_updates']+a['forced_updates']
            assert a['active_physics_updates']+a['padded_slots']==a['allocated_slots']==30720
            assert a['completed_trials']==192 and a['free_hits']==sum(row['hit'] for row in r['rows'])
            assert a['clearance_min']>.01 and a['wall_clearance_min']>.01 and a['snap_max_m']<1e-5
            roll={key:list(raw[key]) for key in ('phase','active','pa','pr','action')};roll['goal']=raw['goal']
            cert=certificate.certify_context(chunk,cfg,roll,raw['first_free'],variants)
            assert cert==r['certificate'];certs.append(cert)
            if condition=='gc100':checks['grid_shams']=base.grid_shams(raw,variants)
        else:
            assert equal(torch.load(path/'failed_plan.pt',map_location='cpu',weights_only=True),chunk)
            assert len(r['rows'])==192 and all(row['invalid'] and not row['hit'] for row in r['rows'])
            checks['failed_context_retained_as_all192_failures']=True
        rows.extend(r['rows']);contexts.append(dict(context=index,valid=r['valid'],checks=checks))
    assert rows==report['rows'] and len(rows)==3072
    for variant in variants:
        subset=[r for r in rows if r['variant']==variant]
        assert {(r['context'],r['heading_index']) for r in subset}=={(c,h) for c in range(16) for h in range(16)}
    assert evaluator.summarize_rows(rows,variants,16)==report['variants']
    aggregate=certificate.aggregate_certificates(certs);assert aggregate==report['certificate']
    physical=all(r['valid'] and r['audit']['forced_hits']==0 for r in report['contexts'])
    assert physical==report['all_physical_checks']
    intact=report['variants']['blind_native']['successes'];erased=report['variants']['blind_reset_after_glimpse']['successes']
    numeric=physical and intact>=167 and intact-erased>=39
    assert report['gates']==dict(near_baseline_pass=False,hard_numeric_pass=numeric,
        hard_certified_memory_pass=numeric and aggregate['all_pairs_certified'])
    return dict(variants=report['variants'],gates=report['gates'],certificate=aggregate,
        all_physical_checks=physical,contexts=contexts)


def run(folder,condition,require_complete=False):
    folder=Path(folder);checks={};failures=[];pending=[];artifacts={}
    assert sha(PROTOCOL)==PROTOCOL_HASH;design=read(PROTOCOL);design.pop('frozen_utc')
    for path,digest in {**design['source_sha256'],**design['original_assets'],**design['passed_baseline_assets']}.items():
        assert sha(ROOT/path)==digest, f'Frozen dependency changed: {path}'
    baseline=torch.load(ROOT/f'results/local_goal_return_baseline/{condition}/initialized_policy.pt',map_location='cpu',weights_only=True)
    def get(name,kind='json'):
        path=folder/name
        if not path.exists():pending.append(name);return None
        artifacts[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        try:return torch.load(path,map_location='cpu',weights_only=True) if kind=='torch' else read(path)
        except Exception as e:failures.append(dict(artifact=name,error=f'{type(e).__name__}: {e}'));return None
    def check(name,fn):
        try:checks[name]=fn()
        except Exception as e:failures.append(dict(artifact=name,error=f'{type(e).__name__}: {e}'));return None
        return checks[name]
    provenance=get('protocol.json');cfg=get('training_config.json');init=get('initialization.json')
    if provenance is not None:
        def verify_provenance():
            assert set(provenance)=={'design','protocol_sha256','condition','model_seed','device','torch_version'}
            assert provenance['design']==design and provenance['protocol_sha256']==PROTOCOL_HASH
            assert provenance['condition']==condition and provenance['model_seed']==1
            assert provenance['device'] in ('cuda','cuda:0') and isinstance(provenance['torch_version'],str)
            return True
        check('frozen_protocol_and_runtime',verify_provenance)
    if cfg is not None:check('training_config',lambda:require_equal(cfg,baseline['config'],'Training config changed'))
    if init is not None:check('initialization',lambda:require_equal(init,baseline['initialization'],'Initialization changed'))
    incoming=get('incoming.pt','torch');snapshots={}
    if incoming is not None and provenance is not None:
        updates=incoming['counters']['updates'];stages=incoming['counters']['completed_stages']
        check('incoming.pt',lambda:audit_checkpoint(incoming,baseline,provenance,updates,stages))
        if updates:
            lineage=get('resume_lineage.json')
            def verify_resume():
                assert lineage['restored_updates']==updates and lineage['completed_stages']==stages
                parent=folder/'resume_parent_evidence/checkpoint.pt'
                assert sha(parent)==lineage['resume_checkpoint_sha256'] and equal(incoming,torch.load(parent,map_location='cpu',weights_only=True))
                for item in lineage['copied_artifacts']:
                    rel=Path(item['copied']);assert not rel.is_absolute() and '..' not in rel.parts
                    assert sha(folder/rel)==item['sha256'],f'Resume evidence changed: {rel}'
                return dict(exact_parent_policy_optimizer_rng_counters=True,restored_updates=updates)
            if lineage is not None:check('resume_evidence',verify_resume)
        else:check('fresh_incoming_has_zero_evaluations',lambda:require_equal(stages,0,'Fresh run has prior evaluations'))
        snapshots['incoming.pt']=incoming
    evaluation_records={}
    for stage in range(1,5):
        names=(f'stage{stage}_before_evaluation.pt',f'stage{stage}.pt');pair=[]
        for name,evals in zip(names,(stage-1,stage)):
            data=get(name,'torch');pair.append(data)
            if data is not None and provenance is not None:
                check(name,lambda d=data,s=stage,e=evals:audit_checkpoint(d,baseline,provenance,s*1024,e))
                snapshots[name]=data
        integrity=get(f'stage{stage}_evaluation_integrity.json')
        if integrity is not None:evaluation_records[stage]=integrity
        summary_path=folder/f'stage{stage}_evaluation/summary.json'
        if not summary_path.exists():pending.append(str(summary_path.relative_to(folder)))
        elif pair[0] is not None:
            check(f'stage{stage}_raw_evaluation',lambda s=stage,d=pair[0]:audit_evaluation(folder/f'stage{s}_evaluation',condition,d))
            for p in sorted(summary_path.parent.rglob('*')):
                if p.is_file():artifacts[str(p.relative_to(folder))]=dict(sha256=sha(p),bytes=p.stat().st_size)
        if all(p is not None for p in pair) and integrity is not None and summary_path.exists():
            def boundary(s=stage,before=pair[0],after=pair[1],record=integrity,path=summary_path):
                compare_evaluation_boundary(before,after,s)
                assert record['stage']==s and record['checkpoint_before_evaluation_sha256']==sha(folder/f'stage{s}_before_evaluation.pt')
                assert record['evaluation_sha256']==sha(path)
                assert all(record[k] is True for k in ('model_policy_unchanged','optimizer_unchanged','private_streams_unchanged','global_rng_unchanged'))
                assert after['counters']['evaluation_records'][-1]==record
                return True
            check(f'stage{stage}_evaluation_preserved_runtime',boundary)
    if (folder/'latest.pt').exists():
        latest=get('latest.pt','torch')
        if latest is not None and provenance is not None:
            def latest_check():
                u=latest['counters']['updates'];assert u>0 and u%128==0 and u<=4096
                return audit_checkpoint(latest,baseline,provenance,u,(u-1)//1024)
            check('latest.pt',latest_check);snapshots['latest.pt']=latest
    final=get('final.pt','torch');summary=get('summary.json')
    if final is not None and provenance is not None:
        check('final.pt',lambda:audit_checkpoint(final,baseline,provenance,4096,4));snapshots['final.pt']=final
        if 'stage4.pt' in snapshots:check('final_exact_stage4',lambda:require_equal(final,snapshots['stage4.pt'],'Final changed after fixed stage4'))
    if summary is not None and final is not None:
        def complete_summary():
            assert summary['completed'] and summary['counters']==final['counters']
            assert summary['active_physics_frames']==final['active_physics_frames']
            assert summary['final_checkpoint_sha256']==sha(folder/'final.pt')
            assert summary['final_evaluation']=='stage4_evaluation/summary.json'
            assert summary['protocol_sha256']==PROTOCOL_HASH and summary['initialization']==baseline['initialization']
            return True
        check('final_summary',complete_summary)
    def evaluation_prefixes():
        for name,snapshot in snapshots.items():
            for i,record in enumerate(snapshot['counters']['evaluation_records'],1):
                if i not in evaluation_records:
                    pending.append(f'evaluation record {i} referenced by {name}');continue
                assert record==evaluation_records[i],f'{name} changed earlier stage evidence'
        return True
    check('checkpoint_evaluation_record_prefixes',evaluation_prefixes)
    metrics_path=folder/'metrics.jsonl'
    if metrics_path.exists():
        artifacts['metrics.jsonl']=dict(bytes=metrics_path.stat().st_size,sha256=sha(metrics_path))
        def reconcile():
            rows,meta=accepted_metrics(folder)
            if require_complete:assert len(rows)==4096 and not meta['partial_last_row']
            for name,snapshot in snapshots.items():
                n=snapshot['counters']['updates']
                if n>len(rows):pending.append(f'metrics through checkpoint {name}');continue
                for key in COUNTERS:assert sum(r[key] for r in rows[:n])==snapshot['counters'][key],f'{name}/{key}'
            return dict(meta,max_noise_replay_error=max((r['replay_error'] for r in rows),default=0),
                no_setup_or_forced_learning_reward=True,all_logged_losses_finite=True,
                scope='Exact saved replay-error assertions and metrics; full training rolls are not retained or rerun.')
        check('all_update_metrics_and_checkpoint_counters',reconcile)
    else:pending.append('metrics.jsonl')
    for name in ('incomplete.json','failed_plan.pt','failed_predraw_streams.pt'):
        if (folder/name).exists():artifacts[name]=dict(bytes=(folder/name).stat().st_size,sha256=sha(folder/name))
    if (folder/'incomplete.json').exists():checks['retained_incomplete_attempt']=read(folder/'incomplete.json')
    complete=not pending and not failures and 'final_summary' in checks
    return dict(passed=not failures and (complete or not require_complete),complete=complete,
        status='complete' if complete else ('failed' if failures else 'pending'),condition=condition,
        folder=str(folder.resolve()),protocol_sha256=PROTOCOL_HASH,auditor_sha256=sha(__file__),
        audit_dependency_sha256={str(Path(p).relative_to(ROOT)):sha(p) for p in (base.__file__,certificate.__file__,evaluator.__file__)},
        audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),checks=checks,failures=failures,pending=pending,
        artifacts=artifacts,scope='Integrity audit of all fixed stages; behavioral gates remain separate. No optimization, neural history regeneration, or outcome selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--condition',choices=('gc100','pgc33'),required=True);p.add_argument('--require-complete',action='store_true')
    p.add_argument('--out',type=Path);args=p.parse_args();torch.set_num_threads(2)
    out=args.out or ROOT/f'results/local_goal_return_design/cloud_audit/{args.condition}.json'
    try:result=run(args.folder,args.condition,args.require_complete)
    except Exception as error:result=dict(passed=False,complete=False,status='failed',condition=args.condition,
        folder=str(args.folder.resolve()),auditor_sha256=sha(__file__),error_type=type(error).__name__,error=str(error))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','complete','status')}))
    if not result['passed']:raise SystemExit(1)
