"""Independent audit of matched entropy continuations; no training import.

Reuse the original event and runtime validators with the genuine cumulative
update count. Parent metadata is never rewritten to impersonate another task.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import audit_rl_blind_reorientation as original
from audit_rl_blind_reorientation import sha,require,close,finite_tree,tree_equal,policy_digest,audit_runtime,audit_evaluation

PROTOCOL=ROOT/'docs/research/rl_blind_reorientation_entropy_protocol.json'
PROTOCOL_HASH='cb05dc84f44e5cf076a7be29bedb6e1235f5a73ce797eabdfda874d966c1a4bb'
ARMS={'original':8e-5,'explore':.01}
PARENT_FRAMES=40_058_880;PARENT_UPDATES=2156
BLOCK_UPDATES=489;BLOCK_FRAMES=10_014_720;BLOCKS=5
EXTENSION_FRAMES=50_073_600;TOTAL_EXTRA=90_132_480;TOTAL_UPDATES=4601
FRAMES_PER_UPDATE=32*160*4


def checkpoint_specs():
    return [('incoming.pt',0,0,False)]+[(f'block{b}.pt',b*BLOCK_UPDATES,b-1,False) for b in range(1,6)]+[
        ('final.pt',BLOCKS*BLOCK_UPDATES,5,True)]


def audit_checkpoint(data,cfg,design,parent,extension_updates,evaluations,completed):
    finite_tree(data);updates=PARENT_UPDATES+extension_updates;frames=extension_updates*FRAMES_PER_UPDATE
    require(data['provenance']==design,'Extension checkpoint provenance changed')
    require(data['config']==cfg,'Only the prespecified entropy configuration may differ')
    require(data['original_frames']==1_000_000_000 and data['parent_extra_frames']==PARENT_FRAMES,'Wrong original/40M parent budget')
    require(data['extension_frames']==frames and data['extra_frames']==PARENT_FRAMES+frames,'Wrong genuine cumulative learning frames')
    require(data['extension_updates']==extension_updates and data['total_updates']==updates,'Wrong cumulative optimizer updates')
    require(data['warmup_updates']==200 and data['extension_warmup_updates']==0,'Unexpected warmup/restart')
    require(data['completed']==completed and data['stage_evaluations_completed']==evaluations,'Wrong endpoint/completion metadata')
    policy=data['policy'];require(policy.keys()==parent['policy'].keys(),'Policy parameter names changed')
    for key,value in policy.items():require(value.shape==parent['policy'][key].shape and value.dtype==parent['policy'][key].dtype,'Policy architecture/dtype changed')
    optimizer=data['optimizer'];require(len(optimizer['param_groups'])==1,'Wrong RMSProp group count')
    group=optimizer['param_groups'][0]
    require(group==parent['optimizer']['param_groups'][0],'RMSProp settings changed')
    require(len(optimizer['state'])==len(group['params']),'RMSProp accumulators missing')
    for state in optimizer['state'].values():
        require(float(state['step'])==updates,'Optimizer accumulator was reset or update count differs')
        require('square_avg' in state and (state['square_avg']>=0).all(),'Invalid RMSProp accumulator')
    runtime=audit_runtime(data['runtime_snapshot'],cfg,updates)
    if extension_updates==0:
        for key in ('policy','optimizer','runtime_snapshot','recent_scores'):
            require(tree_equal(data[key],parent[key]),f'Incoming {key} differs from exact final40M parent')
    return dict(policy_sha256=policy_digest(policy),extension_frames=frames,total_extra_frames=PARENT_FRAMES+frames,
        total_updates=updates,extension_warmup_updates=0,exact_parent_resume=extension_updates==0,runtime=runtime)


def check_runtime_design(data,design,condition,arm):
    expected=dict(design);runtime=data.get('runtime')
    require(isinstance(runtime,dict),'Worker runtime provenance absent')
    require(set(runtime)=={'condition','entropy_arm','entropy_coefficient','device','torch_version','protocol_sha256'},'Unexpected worker runtime fields')
    require(runtime['condition']==condition and runtime['entropy_arm']==arm and runtime['entropy_coefficient']==ARMS[arm],'Wrong entropy worker routing')
    require(runtime['device'] in ('cuda','cuda:0') and runtime['protocol_sha256']==PROTOCOL_HASH,'Wrong backend/protocol')
    require(isinstance(runtime['torch_version'],str) and bool(runtime['torch_version']),'PyTorch version missing')
    expected['runtime']=runtime;require(data==expected,'Worker scientific provenance changed')
    return expected


def run(folder,condition,arm,require_complete=False):
    folder=Path(folder);checks={};failures=[];pending=[];artifacts={}
    require(sha(PROTOCOL)==PROTOCOL_HASH,'Frozen extension protocol hash changed')
    design=json.loads(PROTOCOL.read_text());design.pop('frozen_at')
    require(design['arms']==ARMS and design['learning']['extension_learning_frames']==EXTENSION_FRAMES,'Frozen experiment changed')
    require(design['learning']['total_learning_frames_beyond_1B']==TOTAL_EXTRA,'Wrong final cumulative budget')
    for path,digest in {**design['source_sha256'],**design['original_asset_sha256']}.items():require(sha(ROOT/path)==digest,f'Frozen source/asset changed:{path}')
    require(sha(original.PROTOCOL)==design['parent_protocol_sha256']==original.EXPECTED_PROTOCOL_HASH,'Original task protocol changed')
    for label,asset in design['parent_assets'].items():
        for name,digest in asset['sha256'].items():require(sha(ROOT/asset['folder']/name)==digest,f'Parent asset changed:{label}/{name}')
    parent_folder=ROOT/design['parent_assets'][condition]['folder']
    parent=torch.load(parent_folder/'final.pt',map_location='cpu',weights_only=True)
    parent_design=json.loads(original.PROTOCOL.read_text());parent_design.pop('frozen_at')
    base_cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text());base_cfg['rollout']=160
    base_model=torch.load(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt',map_location='cpu',weights_only=True)
    checks['exact_completed40M_parent']=original.audit_checkpoint(parent,base_cfg,parent_design,base_model,2156,40_058_880,4,True)
    cfg=dict(base_cfg,entropy=ARMS[arm]);checks['frozen_sources_assets_and_parents']=True

    def get(name,kind='json'):
        path=folder/name
        if not path.exists():pending.append(name);return None
        artifacts[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
        try:return torch.load(path,map_location='cpu',weights_only=True) if kind=='torch' else json.loads(path.read_text())
        except Exception as e:failures.append(dict(artifact=name,error=f'{type(e).__name__}: {e}'));return None

    def check(name,fn):
        try:checks[name]=fn()
        except Exception as e:failures.append(dict(artifact=name,error=f'{type(e).__name__}: {e}'))

    worker=get('protocol.json');worker_design=None
    if worker is not None:
        check('protocol.json',lambda:check_runtime_design(worker,design,condition,arm))
        if 'protocol.json' in checks:worker_design=checks['protocol.json']
    config=get('training_config.json')
    if config is not None:check('training_config.json',lambda:require(config==cfg,'Unexpected training config') or True)
    resume=get('resume_audit.json')
    if resume is not None:
        def audit_resume():
            require(resume['parent_checkpoint_sha256']==sha(parent_folder/'final.pt'),'Wrong resume checkpoint hash')
            require(resume['parent_policy_hash']==policy_digest(parent['policy']),'Wrong parent policy')
            require(resume['optimizer_exact'] and resume['runtime_private_rng_exact'],'Missing exact restoration declaration')
            require(resume['changed_config_fields']==([] if arm=='original' else ['entropy']),'Unapproved config changes')
            require(resume['parent_updates']==2156 and resume['extension_warmup_updates']==0,'Wrong resume count/warmup')
            return True
        check('resume_audit.json',audit_resume)
    hashes={};final_saved={}
    for name,updates,evals,complete in checkpoint_specs():
        data=get(name,'torch')
        if data is not None and worker_design is not None:
            check(name,lambda d=data,u=updates,e=evals,c=complete:audit_checkpoint(d,cfg,worker_design,parent,u,e,c))
            if name in checks:hashes[name]=checks[name]['policy_sha256']
            if name in ('block5.pt','final.pt'):
                final_saved[name]={k:data[k] for k in ('policy','optimizer','runtime_snapshot','recent_scores')}
        del data
    if len(final_saved)==2:
        check('final_matches_fixed_block5',lambda:require(tree_equal(final_saved['block5.pt'],final_saved['final.pt']),'Final policy/optimizer/runtime/RNG changed after fixed block5 evaluation') or True)
    evaluations={}
    for name in ['baseline.json']+[f'block{b}_evaluation.json' for b in range(1,6)]+['final_evaluation.json']:
        data=get(name)
        if data is not None:
            check(name,lambda d=data:audit_evaluation(d,cfg,condition));evaluations[name]=data
    if 'baseline.json' in evaluations:
        reference=json.loads((parent_folder/'final_evaluation.json').read_text())
        check('baseline_exact_parent_all12',lambda:require(tree_equal(evaluations['baseline.json'],reference),'Baseline differs from parent raw rows/events/windows/counters') or True)
    verification=get('baseline_verification.json')
    if verification is not None:
        check('baseline_verification.json',lambda:require(verification==dict(
            all12_complete_raw_episode_event_window_rows_exact=True,all_metadata_counters_exact=True,
            parent_reference_sha256=sha(parent_folder/'final_evaluation.json')),'Baseline verification metadata differs') or True)
    if 'final_evaluation.json' in evaluations and 'block5_evaluation.json' in evaluations:
        check('final_evaluation_fixed_block5',lambda:require(tree_equal(evaluations['final_evaluation.json'],evaluations['block5_evaluation.json']),'Final evaluation is not fixed block5') or True)
    path=folder/'metrics.jsonl'
    if not path.exists():pending.append('metrics.jsonl')
    else:
        def audit_metrics():
            raw=path.read_bytes();lines=raw.splitlines(keepends=True);rows=[]
            for i,line in enumerate(lines):
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if i==len(lines)-1 and not require_complete:return dict(rows=len(rows),partial_last_upload=True)
                    raise
                finite_tree(row);u=row['extension_updates'];require(u==20*(len(rows)+1),'Missing/duplicate log update')
                require(row['total_updates']==PARENT_UPDATES+u,'Log cumulative update mismatch')
                require(row['extension_frames']==u*FRAMES_PER_UPDATE,'Log extension frame mismatch')
                require(row['extra_frames']==PARENT_FRAMES+u*FRAMES_PER_UPDATE,'Log total extra frame mismatch')
                require(row['total_exposure_frames']==4_096_000+row['extra_frames'],'Log warmup exposure mismatch')
                require(row['block']==(u-1)//489+1 and row['max_delay']==60,'Changed block/delay curriculum')
                require(row['entropy_coefficient']==ARMS[arm],'Entropy factor changed')
                require(0<=row['replay_error']<3e-4 and 0<=row['free_policy_fraction']<=1,'Invalid replay or policy fraction')
                require(not any(row['schedule']['coverage_violations']),'Event credit-window coverage failure')
                rows.append(row)
            if require_complete:require(len(rows)==122 and rows[-1]['extension_updates']==2440,'Incomplete final log')
            return dict(rows=len(rows),last_extension_update=rows[-1]['extension_updates'] if rows else 0)
        check('metrics.jsonl',audit_metrics);artifacts['metrics.jsonl']=dict(bytes=path.stat().st_size,sha256=sha(path))
    training=get('training.json');summary=get('summary.json');progress=get('progress.json')
    if progress is not None:
        def audit_progress():
            require(set(progress)=={'stages'} and len(progress['stages'])<=5,'Progress endpoint set changed')
            for b,item in enumerate(progress['stages'],1):
                require(item['block']==b and item['extension_frames']==b*BLOCK_FRAMES and item['extra_frames']==PARENT_FRAMES+b*BLOCK_FRAMES,'Wrong fixed block ordering/frame count')
                if f'block{b}_evaluation.json' in evaluations:require(item['evaluation']==evaluations[f'block{b}_evaluation.json'],'Progress evaluation mismatch')
            if training is not None:require(progress['stages']==training['stages'],'Training/progress evaluations differ')
            return dict(blocks=len(progress['stages']))
        check('progress.json',audit_progress)
    if training is not None:
        def audit_training():
            finite_tree(training)
            require(training['completed'] and training['extension_frames']==EXTENSION_FRAMES and training['extra_frames']==TOTAL_EXTRA,'Wrong completed final learning budget')
            require(training['extension_updates']==2445 and training['total_updates']==4601,'Wrong final update budget')
            require(training['extension_warmup_updates']==0,'Unapproved extension warmup')
            require(training['optimizer_restored_exactly'] and training['runtime_and_private_rng_restored_exactly'] and training['no_boundary_hidden_or_episode_reset'],'Missing continuation invariants')
            require(training['noise_replayed_updates']==4601 and 0<=training['noise_replay_max_error']<3e-4,'Final noise replay mismatch')
            require(len(training['stages'])==5,'Missing fixed block evaluations')
            for b,item in enumerate(training['stages'],1):
                require(item['block']==b and item['extension_frames']==b*BLOCK_FRAMES and item['extra_frames']==PARENT_FRAMES+b*BLOCK_FRAMES,'Final stage budget/order differs')
                if f'block{b}_evaluation.json' in evaluations:require(item['evaluation']==evaluations[f'block{b}_evaluation.json'],'Embedded final stage mismatch')
            return dict(extension_frames=EXTENSION_FRAMES,total_extra_frames=TOTAL_EXTRA,total_updates=4601)
        check('training.json',audit_training)
    if summary is not None:
        def audit_summary():
            require(summary['completed'] and summary['condition']==condition and summary['arm']==arm,'Wrong final worker summary')
            require(summary['baseline_matches_parent_exactly'],'Baseline restoration failed')
            if training is not None:require(summary['training']==training,'Final training/summary mismatch')
            if 'final.pt' in artifacts:require(summary['final_checkpoint_sha256']==artifacts['final.pt']['sha256'],'Final file hash mismatch')
            return True
        check('summary.json',audit_summary)
    if (folder/'incomplete.json').exists():failures.append(dict(artifact='incomplete.json',error=json.loads((folder/'incomplete.json').read_text())))
    complete=not pending and not failures
    return dict(version='matched_entropy_independent_artifact_audit_v1',audited_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        folder=str(folder),condition=condition,arm=arm,status='failed' if failures else 'complete' if complete else 'pending',
        complete=complete,passed=not failures and (complete or not require_complete),required_complete=require_complete,
        protocol_sha256=PROTOCOL_HASH,auditor_sha256=sha(__file__),shared_event_runtime_auditor_sha256=sha(Path(original.__file__)),
        checks=checks,failures=failures,pending=pending,artifacts=artifacts,
        interpretation='Data integrity only. Genuine cumulative frame/update metadata retained; no checkpoint metadata rewritten; all five stages retained, no trained-seed inference.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--condition',choices=tuple(original.CONDITIONS),required=True);p.add_argument('--arm',choices=tuple(ARMS),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--require-complete',action='store_true');args=p.parse_args();torch.set_num_threads(2)
    try:result=run(args.folder,args.condition,args.arm,args.require_complete)
    except Exception as e:result=dict(status='failed',complete=False,passed=False,error=f'{type(e).__name__}: {e}',auditor_sha256=sha(__file__))
    args.out.parent.mkdir(parents=True,exist_ok=True);temp=args.out.with_suffix(args.out.suffix+'.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(args.out)
    print(json.dumps({k:result.get(k) for k in ('status','complete','passed','pending','failures','error')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
