"""Independent, read-only audit of both complete sealed local-return baselines.

Behavioral gates are reported even when false. No policy/RNN/environment is
executed, no trial is selected, and no scientific input or result is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.env import _TARGETS
import local_goal_return_training as task
import local_goal_return_training_evaluate as producer
import local_goal_return_training_certificate as certificate

BASE=ROOT/'results/local_goal_return_baseline'
OUT=ROOT/'results/local_goal_return_design/completed_baseline_audit.json'
read=lambda path:json.loads(Path(path).read_text())
sha=producer.sha


def policy_hash(policy):
    h=hashlib.sha256()
    for key,value in sorted(policy.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def same_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b


def audit_initialization(folder,condition,native_check):
    top=read(folder/'summary.json');execution=read(folder/'execution.json')
    init=read(folder/'initialization.json');path=folder/'initialized_policy.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    original_path=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
    original=torch.load(original_path,map_location='cpu',weights_only=True)
    cfg=read(original_path.with_suffix('.json'));cfg.update(n_envs=32,rollout=160,entropy=.01)
    assert original['frames']==1_000_000_000 and saved['config']==cfg
    assert saved['no_optimization'] and saved['behavior_protocol_sha256']==producer.BEHAVIOR_HASH
    assert execution['checkpoint'] is None and execution['checkpoint_sha256'] is None
    assert top['initialized_policy_sha256']==init.pop('initialized_policy_sha256')==sha(path)
    assert init==saved['initialization']==top['initialization']==execution['initialization']
    assert init['parent_checkpoint_sha256']==sha(original_path)==native_check['original_checkpoint_sha256']
    assert init['parent_config_sha256']==sha(original_path.with_suffix('.json'))
    digest=policy_hash(saved['policy'])
    assert digest==init['full_initialized_policy_hash']==native_check['initializer_policy_hash']
    heads=('pi.weight','pi.bias','v.weight','v.bias')
    for key,value in saved['policy'].items():
        if key in heads:
            assert task.tensor_hash(value)==init['new_head_hashes'][key]
            assert task.tensor_hash(original['policy'][key])==init['old_head_hashes'][key]
        else:assert torch.equal(value,original['policy'][key]),key
    w=saved['policy']['pi.weight'];v=saved['policy']['v.weight']
    assert torch.allclose(w@w.T,torch.eye(6)*.0001,atol=1e-10)
    assert torch.allclose(v.norm(),torch.tensor(1.),atol=2e-6)
    assert all(bool((saved['policy'][key]==0).all()) for key in ('pi.bias','v.bias'))
    for record in (saved,execution):
        assert record['source_sha256']==sha(producer.__file__)
        assert record['core_sha256']==sha(task.__file__)
    assert top['evaluation_source_sha256']==sha(producer.__file__) and top['core_source_sha256']==sha(task.__file__)
    return saved,cfg,dict(initialized_policy_sha256=sha(path),policy_hash=digest,
        root_native_validation_identity_exact=True,nonhead_tensors_exact=True,
        original_frames=original['frames'],head_hashes=init['new_head_hashes'],
        source_artifacts={name:sha(folder/name) for name in ('summary.json','execution.json','initialization.json','initialized_policy.pt')})


def verify_trajectory(plan,cfg,raw,rows,first_free,variants):
    """Reconcile every row/mask and independently propagate native FREE motors."""
    groups=len(variants);n=16*groups;schedule=task.schedule(plan,'cpu')
    phase=schedule['phase'].repeat(1,groups);forced=schedule['forced_action'].repeat(1,groups)
    assert raw['phase'].shape==(160,n) and torch.equal(raw['phase'],phase)
    active=raw['active'].bool();free=raw['policy_mask'].bool();terminal=raw['terminal'].bool()
    assert torch.equal(free,active&(phase==task.FREE))
    assert bool((raw['pr']==0).all()) and bool((raw['raw_reward'][~free]==0).all())
    assert torch.equal(raw['reward'],raw['raw_reward'].clamp(-cfg['reward_clip'],cfg['reward_clip']))
    assert bool(((raw['raw_reward']==0)|(raw['raw_reward']==10)).all())
    assert bool((raw['action'][active&~free]==forced[active&~free]).all())
    assert bool((terminal.sum(0)==1).all()) and bool((free.sum(0)<=30).all())
    assert bool((raw['pa'][0]==-1).all())
    for t in range(1,160):
        assert torch.equal(raw['pa'][t][active[t]],raw['action'][t-1][active[t]])
    assert bool((raw['pa'][~active]==-1).all())
    start=int(schedule['free'][0]);end=start+30
    assert bool((schedule['free']==start).all())
    indexed={(r['variant'],r['heading_index']):r for r in rows}
    assert len(indexed)==len(rows)==n
    max_path_error=max_progress_error=0.
    for j in range(n):
        row=indexed[(variants[j//16],j%16)];assert row['context']==int(plan['context'][0]) and not row.get('invalid',False)
        final=int(terminal[:,j].nonzero()[0]);expected_active=torch.arange(160)<=final
        assert torch.equal(active[:,j],expected_active)
        hit=bool((raw['raw_reward'][:,j]>0).any())
        assert int((raw['raw_reward'][:,j]>0).sum())==int(hit)
        assert final>=start and (final<end if hit else final==end-1)
        assert terminal[final,j] and (float(raw['raw_reward'][final,j])==10 if hit else True)
        assert row['hit']==hit and row['hit_step']==(final-start+1 if hit else -1)
        assert row['free_steps']==int(free[:,j].sum())==final-start+1
        points=torch.cat((first_free[j]['position'][None],raw['position'][start:final+1,j]),0)
        path=(points[1:]-points[:-1]).norm(dim=1).sum()
        goal=plan['goal'][j%16];initial=(points[0]-goal).norm();last=(points[-1]-goal).norm()
        max_path_error=max(max_path_error,abs(float(path)-row['free_path_m']))
        max_progress_error=max(max_progress_error,abs(float(initial-last)-row['progress_m']))
        assert abs(float(initial)-row['free_initial_distance_m'])<2e-6
        assert (float(last)<cfg['goal_radius']+1e-5) if hit else (float(last)>=cfg['goal_radius']-1e-5)
        assert torch.equal(raw['position'][final:,j],raw['position'][final:final+1,j].expand_as(raw['position'][final:,j]))
        expected_counts=torch.bincount(raw['action'][start:final+1,j],minlength=6).tolist()
        assert row['action_counts']==expected_counts
        blind=(phase[:,j]>=task.TURN_A)&active[:,j]
        expected_before=blind if variants[j//16].endswith('_reset_every_blind') else torch.zeros(160,dtype=torch.bool)
        expected_after=(torch.arange(160)==int(schedule['glimpse'][0])) if variants[j//16].endswith('_reset_after_glimpse') else torch.zeros(160,dtype=torch.bool)
        assert torch.equal(raw['reset_before'][:,j],expected_before)
        assert torch.equal(raw['reset_after'][:,j],expected_after)
    assert max_path_error<3e-6 and max_progress_error<3e-6
    pos=np.stack([first_free[i]['position'].numpy() for i in range(n)]).astype(float)
    hd=np.array([float(first_free[i]['heading']) for i in range(n)])
    motors=np.stack([first_free[i]['motors'].numpy() for i in range(n)]).astype(float)
    tape=plan['motor'].repeat(1,1,groups).numpy().astype(float)
    targets=np.asarray(_TARGETS,dtype=float);k=1-np.exp(-(cfg['action_repeat']/60)/.25)
    max_physics_error=0.
    for t in range(start,end):
        use=free[t].numpy();action=raw['action'][t].numpy();target=targets[action]*[cfg['v_max'],cfg['v_max'],cfg['rot_max']]
        nxt=motors+(target+tape[t].T*[.03,.03,.15]-motors)*k
        angle=(hd+nxt[:,2]*(cfg['action_repeat']/60)+np.pi)%(2*np.pi)-np.pi
        delta=np.column_stack((nxt[:,0]*np.cos(angle)-nxt[:,1]*np.sin(angle),nxt[:,0]*np.sin(angle)+nxt[:,1]*np.cos(angle)))*(cfg['action_repeat']/60)
        before_clip=pos+delta;new=np.clip(before_clip,-cfg['arena_half']+.08,cfg['arena_half']-.08)
        walls=np.any(before_clip!=new,axis=1);nxt[walls,:2]*=.2
        pos[use]=new[use];hd[use]=angle[use];motors[use]=nxt[use]
        max_physics_error=max(max_physics_error,float(np.max(np.abs(pos-raw['position'][t].numpy()))))
    assert max_physics_error<2e-5, max_physics_error
    return dict(trials=n,all_row_counts_and_terminals_exact=True,forced_rewards=0,
        native_free_physics_max_abs_position_error_m=max_physics_error,
        saved_path_reconciliation_max_error_m=max_path_error,saved_progress_reconciliation_max_error_m=max_progress_error,
        zero_cross_trial_bootstrap_and_padding_masks=True)


def verify_first_free(raw,policy,variants):
    first=raw['first_free'];n=16*len(variants);assert set(first)==set(range(n))
    h=torch.stack([first[i]['state'][:256] for i in range(n)])
    logits=torch.nn.functional.linear(h,policy['pi.weight'],policy['pi.bias']);prob=logits.softmax(1)
    error=0.
    for i in range(n):
        f=first[i];assert f['input'].shape==(1287,) and f['state'].shape==(512,)
        assert torch.equal(f['input'][768:1280],raw['goal'][i])
        error=max(error,float((logits[i]-f['logits']).abs().max()))
        assert torch.allclose(logits[i],f['logits'],atol=2e-7,rtol=1e-6)
        assert torch.allclose(prob[i],f['probabilities'],atol=2e-7,rtol=1e-6)
        assert abs(float(-(prob[i]*prob[i].log()).sum())-float(f['entropy']))<2e-6
        assert torch.equal(f['position'],first[i%16]['position'])
        assert torch.equal(f['heading'],first[i%16]['heading']) and torch.equal(f['motors'],first[i%16]['motors'])
    for view in ('blind','seen'):
        base=variants.index(view+'_native')*16
        for cue in ('ordinary','plus60','minus60','reset_after_glimpse','reset_every_blind'):
            offset=variants.index(view+'_'+cue)*16
            for i in range(16):assert torch.equal(first[base+i]['input'],first[offset+i]['input'])
    return dict(first_free_actor_logits_max_abs_error=error,cue_branches_common_free_inputs=True,common_physical_start=True)


def grid_shams(raw,variants):
    for view in ('blind','seen'):
        base=variants.index(view+'_native')*16
        for cue in ('ordinary','plus60','minus60'):
            offset=variants.index(view+'_'+cue)*16
            for key,value in raw.items():
                if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[:2]==(160,192):
                    assert torch.equal(value[:,base:base+16],value[:,offset:offset+16]),key
            assert torch.equal(raw['goal'][base:base+16],raw['goal'][offset:offset+16])
            for i in range(16):assert same_tree(raw['first_free'][base+i],raw['first_free'][offset+i])
    return dict(all_recorded_step_arrays_exact=True,first_free_full_h_c_and_actor_fields_exact=True,
        scope='All160 recorded action/position/mask/reward arrays and first-FREE512state; intermediate recurrent states were not stored.')


def audit_bank(folder,condition,bank_name,saved,cfg):
    report=read(folder/'summary.json');plan,bank,behavior=producer.load_bank(bank_name)
    assert report['complete'] and report['condition']==condition and report['model_seed']==1
    assert report['bank']==bank_name and report['bank_sha256']==bank['sha256']
    assert report['source_sha256']==sha(producer.__file__) and report['core_sha256']==sha(task.__file__)
    assert report['behavior_protocol_sha256']==producer.BEHAVIOR_HASH
    assert report['checkpoint_policy_hash']==policy_hash(saved['policy'])
    assert report['no_optimization'] and report['clean_state_noise_sigma']==0
    variants=list(task.VARIANTS);assert set(report['variants'])==set(variants) and len(report['contexts'])==16
    contexts=[];all_rows=[];control_rows={'uniform':[],'planner':[]};certificates=[]
    for context in range(16):
        path=folder/f'context{context}';r=read(path/'result.json');assert r==report['contexts'][context]
        assert r['context']==context;chunk=producer.context_plan(plan,context)
        input_record=read(path/'input.json');assert input_record['sha256']==bank['sha256'] and input_record['context_index']==context
        checks={};first=None
        if r['valid']:
            assert sha(path/'raw.pt')==r['raw_sha256']
            raw=torch.load(path/'raw.pt',map_location='cpu',weights_only=True)
            assert raw['plan_reference']==input_record
            first=raw['first_free']
            checks.update(verify_trajectory(chunk,cfg,raw,r['rows'],first,variants))
            checks.update(verify_first_free(raw,saved['policy'],variants))
            audit=r['audit'];assert audit['setup_learning_reward']==audit['forced_hits']==0
            assert audit['setup_pickups']==192 and audit['setup_raw_reward']==1920.
            assert audit['free_hits']==sum(row['hit'] for row in r['rows'])
            assert audit['active_physics_updates']==int(raw['active'].sum())
            assert audit['free_updates']==int(raw['policy_mask'].sum())
            assert audit['active_physics_updates']+audit['padded_slots']==160*192
            assert audit['clearance_min']>.01 and audit['wall_clearance_min']>.01 and audit['snap_max_m']<1e-5
            roll={key:list(raw[key]) for key in ('phase','active','pa','pr','action')};roll['goal']=raw['goal']
            cert=certificate.certify_context(chunk,cfg,roll,first,variants)
            assert cert==r['certificate'];certificates.append(cert)
            if condition=='gc100':checks['grid_shams']=grid_shams(raw,variants)
        else:
            assert same_tree(torch.load(path/'failed_plan.pt',map_location='cpu',weights_only=True),chunk)
            assert len(r['rows'])==192 and all(row['invalid'] and not row['hit'] for row in r['rows'])
            checks['invalid_context_retained']=True
        all_rows.extend(r['rows'])
        assert set(r['controls'])==set(control_rows)
        for name,control in r['controls'].items():
            control_rows[name].extend(control['rows'])
            if control['valid'] and first is not None:
                control_raw=torch.load(path/f'{name}_raw.pt',map_location='cpu',weights_only=True)
                checks[name]=verify_trajectory(chunk,cfg,control_raw,control['rows'],
                    {i:first[i] for i in range(16)},('blind_native',))
                assert control['audit']['forced_hits']==control['audit']['setup_learning_reward']==0
            elif not control['valid']:
                assert same_tree(torch.load(path/f'{name}_failed_plan.pt',map_location='cpu',weights_only=True),chunk)
                assert len(control['rows'])==16 and all(row['invalid'] and not row['hit'] for row in control['rows'])
        contexts.append(dict(context=context,valid=r['valid'],raw_checks=checks,
            source_hashes={p.name:sha(p) for p in sorted(path.iterdir()) if p.is_file()}))
    assert all_rows==report['rows'] and len(all_rows)==3072
    summaries={}
    for variant in variants:
        rows=[r for r in all_rows if r['variant']==variant]
        assert len(rows)==256 and {(r['context'],r['heading_index']) for r in rows}=={(c,h) for c in range(16) for h in range(16)}
        successes=sum(r['hit'] for r in rows);s=report['variants'][variant]
        assert s['trials']==256 and s['successes']==successes and s['success_fraction']==successes/256
        assert s['invalid_trials']==sum(r.get('invalid',False) for r in rows)
        summaries[variant]=s
    for name,rows in control_rows.items():
        s=report['controls'][name];assert len(rows)==s['trials']==256
        assert s['successes']==sum(r['hit'] for r in rows) and s['invalid_trials']==sum(r.get('invalid',False) for r in rows)
    aggregate=certificate.aggregate_certificates(certificates)
    assert aggregate==report['certificate']
    physical=all(r['valid'] and r['audit']['forced_hits']==0 and all(c['valid'] and c['audit']['forced_hits']==0 for c in r['controls'].values()) for r in report['contexts'])
    assert report['all_physical_checks']==physical
    successes=summaries['blind_native']['successes'];erased=summaries['blind_reset_after_glimpse']['successes']
    gates=dict(near_baseline_pass=bank_name=='near_baseline' and physical and successes>=13,
        hard_numeric_pass=bank_name=='final_hard' and physical and successes>=167 and successes-erased>=39,
        hard_certified_memory_pass=False)
    gates['hard_certified_memory_pass']=gates['hard_numeric_pass'] and aggregate['all_pairs_certified']
    assert gates==report['gates']
    return dict(variants=summaries,controls=report['controls'],gates=gates,certificate=aggregate,
        all_physical_checks=physical,contexts=contexts,summary_sha256=sha(folder/'summary.json'))


def run(base):
    for condition in ('gc100','pgc33'):
        assert (base/condition/'summary.json').exists(), 'Wait for BOTH full baseline writers to finish'
        assert read(base/condition/'summary.json')['complete']
    native=read(ROOT/'results/local_goal_return_design/native_resume_validation.json');assert native['passed']
    checks={r['condition']:r for r in native['rows']};models={};policies={}
    for condition in ('gc100','pgc33'):
        folder=base/condition;saved,cfg,initial=audit_initialization(folder,condition,checks[condition]);policies[condition]=saved['policy']
        models[condition]=dict(initialization=initial,banks={name:audit_bank(folder/name,condition,name,saved,cfg) for name in ('near_baseline','final_hard')})
    for key in ('pi.weight','pi.bias','v.weight','v.bias'):assert torch.equal(policies['gc100'][key],policies['pgc33'][key])
    control_shams={}
    for bank in ('near_baseline','final_hard'):
        for name in ('uniform','planner'):
            for context in range(16):
                a=torch.load(base/'gc100'/bank/f'context{context}'/f'{name}_raw.pt',map_location='cpu',weights_only=True)
                b=torch.load(base/'pgc33'/bank/f'context{context}'/f'{name}_raw.pt',map_location='cpu',weights_only=True)
                assert same_tree(a,b), (bank,name,context)
            control_shams[f'{bank}/{name}']=True
    return dict(passed=True,complete=True,artifact_integrity_passed=True,
        all_near_learning_gates_passed=all(m['banks']['near_baseline']['gates']['near_baseline_pass'] for m in models.values()),
        hard_memory_gate_by_condition={c:m['banks']['final_hard']['gates']['hard_certified_memory_pass'] for c,m in models.items()},
        matched_new_heads_exact=True,uniform_planner_raw_cross_population_identities=control_shams,
        models=models,source_sha256=sha(__file__),behavior_protocol_sha256=producer.BEHAVIOR_HASH,
        native_validation_sha256=sha(ROOT/'results/local_goal_return_design/native_resume_validation.json'),
        scope='All completed baseline models/banks/contexts/variants/controls retained. Passed means artifact and numerical integrity, not success of a behavioral gate. No policy/recurrent/environment execution or adaptive selection. Existing mathematical certificate reused unchanged; FREE physics independently reconciled from saved actions/tapes.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();torch.set_num_threads(2)
    try:result=run(args.base)
    except Exception as error:
        result=dict(passed=False,complete=False,error_type=type(error).__name__,error=str(error),source_sha256=sha(__file__))
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');raise
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','complete','all_near_learning_gates_passed','hard_memory_gate_by_condition')}))
