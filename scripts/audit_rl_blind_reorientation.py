"""Independent artifact audit for the frozen blind-reorientation experiment.

Read-only with respect to training artifacts and frozen sources. Missing uploads
are pending; violated invariants are failures. No training code is imported.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

PROTOCOL=ROOT/'docs/research/rl_blind_reorientation_protocol.json'
CONDITIONS=('gc100','pgc33')
DELAYS=(4,15,30,60)
PULSES=('native','ordinary','plus60','minus60','reset_after_glimpse','reset_every_blind')
VARIANTS=tuple(f'{vision}_{pulse}' for vision in ('blind','seen') for pulse in PULSES)
STAGE_UPDATES=489
WARMUP=200
FRAMES_PER_UPDATE=32*160*4
STAGE_FRAMES=10_014_720
TOTAL_FRAMES=40_058_880
WARMUP_FRAMES=4_096_000
EXPECTED_PROTOCOL_HASH='2fa78ec55fe7eefb3dcfd5eacf73e3a703bbf6a2c479fe9a3158f4b6604fa267'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_digest(policy):
    h=hashlib.sha256()
    for name,value in sorted(policy.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def require(value,message):
    if not bool(value):raise AssertionError(message)


def close(actual,expected,label,atol=1e-4,rtol=1e-5):
    require(math.isclose(float(actual),float(expected),abs_tol=atol,rel_tol=rtol),
            f'{label}: {actual} != {expected}')


def finite_tree(value,path='root'):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():require(torch.isfinite(value).all(),f'Nonfinite tensor: {path}')
    elif isinstance(value,float):require(math.isfinite(value),f'Nonfinite number: {path}')
    elif isinstance(value,dict):
        for key,item in value.items():finite_tree(item,f'{path}.{key}')
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):finite_tree(item,f'{path}[{i}]')


def tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def audit_evaluation(data,cfg,condition,episodes=64,n_envs=32):
    finite_tree(data)
    require(data['clean_no_noise'] is True,'Evaluation must be clean')
    require(data['seed']==148201,'Evaluation seed differs from frozen design')
    require(data['max_delay']==60,'Evaluation delay maximum must be60')
    require(data['episodes_per_variant']==episodes,'Episode budget mismatch')
    require(set(data['variants'])==set(VARIANTS),'Missing evaluation branch')
    require(data['all_first_discoveries_identical'] is True,'Missing first-discovery invariant')
    require(data['grid_pulse_controls_exact_noop']==(condition=='gc100'),'Wrong Grid-sham declaration')
    schedule=data['schedule'];probe=data['probe_audit']
    require(len(schedule['admitted'])==12,'Schedule branch dimension')
    expected_keys={(slot,episode) for slot in range(n_envs) for episode in range(episodes//n_envs)}
    total_steps=episodes*cfg['episode_len']
    first_discovery=None;report={};event_plans={}
    for group,variant in enumerate(VARIANTS):
        summary=data['variants'][variant];rows=summary['rows']
        require(len(rows)==episodes and {(r['slot'],r['episode']) for r in rows}==expected_keys,
                f'{variant}: incomplete original episode set')
        by_episode={(r['slot'],r['episode']):r for r in rows}
        first={key:r['first_goal_step'] for key,r in by_episode.items()}
        if first_discovery is None:first_discovery=first
        require(first==first_discovery,f'{variant}: first discoveries diverged before intervention')
        for row in rows:
            require((row['n_goals']==0)==(row['first_goal_step']==-1),'First discovery/goal count inconsistent')
            require(-1<=row['first_goal_step']<cfg['episode_len'],'First goal step out of range')
            close(row['score'],10*row['n_goals'],'Original+10 reward')
        close(summary['mean_score'],sum(r['score'] for r in rows)/episodes,f'{variant}: mean score')
        close(summary['mean_goals'],sum(r['n_goals'] for r in rows)/episodes,f'{variant}: mean goals')
        phases=schedule['phase_update_counts'][group]
        require(len(phases)==7 and sum(phases)==total_steps,'Evaluation phase-update coverage')
        require(all(isinstance(v,int) and v>=0 for v in phases),'Negative/noninteger phase count')
        require(schedule['coverage_violations'][group]==0,'Event crossed a BPTT boundary')
        admitted=schedule['admitted'][group]
        require(admitted==sum(schedule[k][group] for k in ('completed','interrupted_reward','interrupted_done')),
                'Admitted events not completely accounted for')
        windows=data['window_rows'][variant]
        require(len(windows)==admitted,'Missing completed/interrupted event record')
        window_keys={(r['slot'],r['episode'],r['event']) for r in windows}
        require(len(window_keys)==len(windows),'Duplicate event record')
        for row in windows:
            require((row['slot'],row['episode']) in expected_keys,'Event outside original episode set')
            require(by_episode[row['slot'],row['episode']]['first_goal_step']>=0,'Event before known goal')
            require(1<=row['event']<=8,'Per-episode event cap exceeded')
            require(row['turn_a'] in (0,1) and row['turn_b'] in (0,1),'Non-turn forced label')
            require(5<=row['turn_a_updates']<=10 and 5<=row['turn_b_updates']<=10,'Turn duration out of bounds')
            require(30<=row['delay_updates']<=60,'Clean evaluation delay outside30..60')
            planned=15+row['turn_a_updates']+row['turn_b_updates']+4+row['delay_updates']+30
            require(1<=row['elapsed_updates']<=planned<=129,'Window length violates full credit horizon')
            ends=np.cumsum([15,row['turn_a_updates'],row['turn_b_updates'],4,row['delay_updates'],30]).tolist()
            phase_names=['visible','turn_a','turn_b','forward_probe','left_delay','free_blind']
            require(row['end_phase'] in phase_names,'Unknown event end phase')
            phase_index=phase_names.index(row['end_phase'])
            lower=0 if phase_index==0 else ends[phase_index-1]
            require(lower<row['elapsed_updates']<=ends[phase_index],'Elapsed updates disagree with phase timeline')
            require(row['pulse_delivered']==(phase_index>=4),'Pulse timing disagrees with event end phase')
            if row['episode_end']:require(row['elapsed_updates']==planned,'Episode ended before full admitted window fit')
            key=(row['slot'],row['episode'],row['event'])
            plan=tuple(row[k] for k in ('turn_a','turn_b','turn_a_updates','turn_b_updates','delay_updates'))
            if key in event_plans:require(event_plans[key]==plan,'Matched branches have different event-index plans')
            else:event_plans[key]=plan
            require(np.asarray(row['pre_brake_vf_vs_vr']).shape==(3,3),'Missing pre-brake telemetry')
            require(np.asarray(row['post_brake_max_abs']).shape==(3,),'Missing post-brake telemetry')
            require(np.count_nonzero(row['post_brake_max_abs'])==0,'A motor brake did not zero all velocities')
            require(0<=row['probe_wall_updates']<=4,'Probe wall-count bounds')
            if row['pulse_delivered']:
                require(row['probe_displacement_m']>=0 and row['endpoint_speed_m_s']>=0,'Negative probe metric')
                require(abs(row['endpoint_travel_minus_heading_deg'])<=180.0001,'Invalid direction error')
        free=data['free_window'][variant];free_rows=free['rows']
        require(len({(r['slot'],r['episode'],r['event']) for r in free_rows})==len(free_rows),'Duplicate free window')
        for row in free_rows:
            require((row['slot'],row['episode'],row['event']) in window_keys,'Free window without parent event')
            require(1<=row['free_updates']<=30,'Free window update bound')
            require(row['completed30']==(row['free_updates']==30),'Wrong completed30 flag')
            require(row['goal_hit'] or row['episode_end'] or row['completed30'],'Truncated free window unaccounted')
            require(row['path_m']>=0 and row['initial_goal_distance_m']>=0,'Invalid free-window distance')
            require(abs(row['progress_m'])<=row['path_m']+1e-4,'Progress exceeds physical path length')
        close(free['free_updates'],sum(r['free_updates'] for r in free_rows),'Free update accounting',atol=0,rtol=0)
        close(free['free_hits'],sum(r['goal_hit'] for r in free_rows),'Free hit accounting',atol=0,rtol=0)
        close(free['free_progress_m'],sum(r['progress_m'] for r in free_rows),'Free progress accounting')
        close(free['free_path_m'],sum(r['path_m'] for r in free_rows),'Free path accounting')
        require(free['free_updates']==phases[6],'Free updates disagree with phase clock')
        require(free['free_hits']+free['forced_hits']<=sum(r['n_goals'] for r in rows),'More restricted hits than total goals')
        for field,key in [('free_hits','free_goal_hits'),('forced_hits','forced_goal_hits')]:
            close(free[field],probe[key][group],f'{variant}: {field} audit',atol=0,rtol=0)
        close(probe['scheduled_blind_updates'][group],sum(phases[2:]),'Blind update accounting',atol=0,rtol=0)
        pulses=sum(r['pulse_delivered'] for r in windows)
        close(probe['pulses'][group],pulses,'Pulse event coverage',atol=0,rtol=0)
        close(probe['probe_wall_updates'][group],sum(r['probe_wall_updates'] for r in windows),'Probe walls',atol=0,rtol=0)
        close(probe['endpoint_speed_sum'][group],sum(r['endpoint_speed_m_s'] for r in windows if r['pulse_delivered']),'Probe speed')
        close(probe['probe_displacement_sum'][group],sum(r['probe_displacement_m'] for r in windows if r['pulse_delivered']),'Probe displacement')
        close(probe['absolute_heading_error_rad_sum'][group],sum(abs(math.radians(r['endpoint_travel_minus_heading_deg'])) for r in windows if r['pulse_delivered']),'Probe angle')
        expected_resets=pulses if variant.endswith('reset_after_glimpse') else sum(phases[2:]) if variant.endswith('reset_every_blind') else 0
        require(free['resets']==expected_resets,'Wrong reset timing/count')
        report[variant]=dict(episodes=episodes,admitted_events=admitted,free_windows=len(free_rows),
            zero_free_window_episodes=episodes-len({(r['slot'],r['episode']) for r in free_rows}),
            free_hits=free['free_hits'],forced_hits=free['forced_hits'],probe_wall_updates=probe['probe_wall_updates'][group],
            pulses=pulses,all_events_retained=True)
    if condition=='gc100':
        for vision in ('blind','seen'):
            for control in ('ordinary','plus60','minus60'):
                a,b=f'{vision}_native',f'{vision}_{control}'
                require(data['variants'][a]['rows']==data['variants'][b]['rows'],'Grid episode sham differs')
                require(data['free_window'][a]['rows']==data['free_window'][b]['rows'],'Grid free-window sham differs')
                require(data['window_rows'][a]==data['window_rows'][b],'Grid event sham differs')
    return report


def audit_runtime(snapshot,cfg,updates):
    finite_tree(snapshot)
    expected={'environment','episodes_done','environment_rng','observation','state','goal','pa','pr',
              'input_tensors','input_audit','clock','clock_stats','noise_rng','noise_replays',
              'noise_max_replay_error','action_rng'}
    require(set(snapshot)==expected,'Runtime snapshot field set changed or incomplete')
    n=32;env=snapshot['environment'];clock=snapshot['clock'];inputs=snapshot['input_tensors']
    expected_environment={'_ceil','_col_ang','_floor','_gu','_gv','_targets','_v','_wall_col',
        'draw_idx','ep_idx','ep_return','first_goal_t','goal','hd','land_pos','mdir','n_goals','partial',
        'path_eff_n','path_eff_sum','pos','spawn','stripe_phase','t','travelled','vf','vr','vs'}
    expected_clock={'age','duration_a','duration_b','duration_delay','elapsed','episode','event_count',
        'known','phase','phase_counts','plans','remaining','turn_a','turn_b','used_segment'}
    expected_inputs={'brake_after_max','brake_before','held','ordinary','probe_displacement',
        'probe_heading_error','probe_seen','probe_speed','probe_start','probe_walls','reset_after',
        'reset_every','rotation','seen'}
    expected_audit={'absolute_heading_error_rad_sum','endpoint_speed_sum','forced_goal_hits',
        'free_goal_hits','probe_displacement_sum','probe_wall_updates','pulses','scheduled_blind_updates'}
    expected_stats={'admitted','completed','coverage_violations','interrupted_done','interrupted_reward',
                    'wait_cap','wait_episode','wait_segment'}
    require(set(env)==expected_environment,'Environment tensor snapshot incomplete')
    require(set(clock)==expected_clock,'Event-clock tensor snapshot incomplete')
    require(set(inputs)==expected_inputs,'Held-input/probe tensor snapshot incomplete')
    require(set(snapshot['input_audit'])==expected_audit,'Input audit counters incomplete')
    require(set(snapshot['clock_stats'])==expected_stats,'Event audit counters incomplete')
    require(set(snapshot['observation'])=={'rgb','pos','hd','vel','dir'},'Observation snapshot incomplete')
    for key in ('environment_rng','noise_rng','action_rng'):
        require(snapshot[key].dtype==torch.uint8 and snapshot[key].numel()>0,f'Missing private generator:{key}')
    require(len(snapshot['state'])==2 and all(tuple(s.shape)==(n,256) for s in snapshot['state']),'Wrong h/c shape')
    require(tuple(snapshot['goal'].shape)==(n,512),'Wrong saved-goal shape')
    require(tuple(inputs['held'].shape)==(n,512),'Wrong held current snapshot shape')
    require(snapshot['noise_replays']==updates,'Noisy replay count differs from training updates')
    require(0<=snapshot['noise_max_replay_error']<3e-4,'Noisy replay tolerance exceeded')
    require(not clock['phase'].any(),'An event is active at checkpoint/BPTT boundary')
    require(not clock['remaining'].any(),'Inactive clock has remaining event updates')
    require(((clock['event_count']>=0)&(clock['event_count']<=8)).all(),'Episode event cap violated')
    require(((clock['elapsed']>=0)&(clock['elapsed']<=129)).all(),'Event length exceeds129')
    require(tuple(clock['plans'].shape)==(n,8,5) and clock['plans'].dtype==torch.float64,'Missing event plan bank')
    require(((clock['plans']>=0)&(clock['plans']<1)).all(),'Event random quantile out of range')
    counts=clock['phase_counts'];require(tuple(counts.shape)==(n,7),'Phase count shape')
    require((counts.sum(1)==updates*160).all(),'Runtime total update coverage mismatch')
    stats=snapshot['clock_stats']
    require(not stats['coverage_violations'].any(),'Event crossed a BPTT boundary')
    require(torch.equal(stats['admitted'],stats['completed']+stats['interrupted_reward']+stats['interrupted_done']),
            'Checkpoint event accounting incomplete')
    require(torch.equal(clock['known'],env['n_goals']>=1),'Known-goal clock disagrees with environment')
    require(not snapshot['goal'][~clock['known']].any(),'Goal snapshot survives unknown/new episode')
    require((snapshot['goal'][clock['known']].abs().sum(1)>0).all(),'Missing saved goal for known episode')
    require(((snapshot['pa']>=-1)&(snapshot['pa']<6)).all(),'Invalid actual previous action')
    require((snapshot['pr'].abs()<=cfg['reward_clip']).all(),'Previous reward clip changed')
    require(((env['t']>=0)&(env['t']<cfg['episode_len'])).all(),'Environment clock out of range')
    if updates:
        require(torch.equal(snapshot['pa']==-1,env['t']==0),'Previous action reset differs from episode end')
        require(all(not s[env['t']==0].any() for s in snapshot['state']),'h/c not cleared at episode end')
    for key in ('seen','ordinary','reset_after','reset_every','rotation'):
        require(not inputs[key].any(),f'Evaluation intervention leaked into training:{key}')
    require(not inputs['brake_after_max'].any(),'Recorded motor brake failure')
    audit=snapshot['input_audit']
    require(torch.equal(audit['scheduled_blind_updates'],counts[:,2:].sum(1).to(audit['scheduled_blind_updates'].dtype)),
            'Input/clock blind-update counts differ')
    require((audit['pulses']<=stats['admitted']).all(),'More pulses than events')
    require((audit['probe_wall_updates']<=4*stats['admitted']).all(),'Probe wall-count bounds')
    return dict(total_updates=updates,total_environment_updates=updates*160*32,
        admissions=int(stats['admitted'].sum()),completed=int(stats['completed'].sum()),
        coverage_violations=0,noise_replays=snapshot['noise_replays'],max_replay_error=snapshot['noise_max_replay_error'],
        native_goal_scope='Frozen native-goal source/asset path and known/unknown snapshot consistency verified; snapshots do not store every historical pickup for independent code regeneration')


def audit_checkpoint(data,cfg,design,original,updates,frames,stage_evals,completed,policy_unchanged=False):
    finite_tree(data)
    require(data['provenance']==design,'Checkpoint provenance differs from frozen protocol')
    require(data['config']==cfg,'Checkpoint configuration changed')
    require(data['original_frames']==1_000_000_000,'Wrong original parent frame budget')
    require(data['extra_frames']==frames and data['total_updates']==updates,'Checkpoint frame/update budget differs')
    require(data['warmup_updates']==200 and data['completed']==completed,'Warmup/completion metadata differs')
    require(data['stage_evaluations_completed']==stage_evals,'Fixed-stage evaluation count mismatch')
    policy=data['policy'];require(policy.keys()==original['policy'].keys(),'Policy parameter set changed')
    for key,value in policy.items():require(value.shape==original['policy'][key].shape,'Network architecture changed')
    digest=policy_digest(policy)
    if policy_unchanged:require(tree_equal(policy,original['policy']),'Original/zero-LR policy changed')
    optimizer=data['optimizer'];require(len(optimizer['param_groups'])==1,'Unexpected optimizer groups')
    group=optimizer['param_groups'][0]
    close(group['lr'],0. if updates==0 else cfg['lr'],'RMSProp learning rate',atol=0,rtol=0)
    close(group['alpha'],cfg['rms_alpha'],'RMSProp alpha',atol=0,rtol=0)
    close(group['eps'],cfg['rms_eps'],'RMSProp epsilon',atol=0,rtol=0)
    if updates==0:require(not optimizer['state'],'Incoming optimizer must be NEW, not restored from parent')
    else:
        require(len(optimizer['state'])==len(group['params']),'Incomplete RMSProp accumulator state')
        for value in optimizer['state'].values():
            require(float(value['step'])==updates,'RMSProp step count differs')
            require('square_avg' in value and (value['square_avg']>=0).all(),'Invalid RMSProp square accumulator')
    return dict(policy_sha256=digest,optimizer='New original RMSProp; original parent has no optimizer',
        frames=frames,updates=updates,warmup_exposure_frames=WARMUP_FRAMES if updates>=WARMUP else 0,
        runtime=audit_runtime(data['runtime_snapshot'],cfg,updates))


def run(folder,condition,require_complete=False):
    folder=Path(folder);checks={};failures=[];pending=[];artifacts={}
    require(sha(PROTOCOL)==EXPECTED_PROTOCOL_HASH,'Frozen protocol hash changed')
    design=json.loads(PROTOCOL.read_text());design.pop('frozen_at')
    require(design['learning']['total_learning_frames']==TOTAL_FRAMES,'Frozen learning frame budget changed')
    require(design['learning']['warmup_exposure_frames']==WARMUP_FRAMES,'Frozen warmup exposure changed')
    for name,digest in {**design['source_sha256'],**design['original_asset_sha256']}.items():
        require(sha(ROOT/name)==digest,f'Frozen source/asset changed:{name}')
    checks['frozen_source_asset_manifest']=True
    cfgpath=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json'
    original_cfg=json.loads(cfgpath.read_text());cfg=dict(original_cfg,rollout=160)
    original=torch.load(cfgpath.with_suffix('.pt'),map_location='cpu',weights_only=True)
    require('optimizer' not in original,'Original parent unexpectedly contains an optimizer')
    require(original['frames']==1_000_000_000,'Wrong original parent checkpoint')
    checks['original_optimizer_absent']=True

    def get(name,kind='json'):
        path=folder/name
        if not path.exists():pending.append(name);return None
        artifacts[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
        try:return torch.load(path,map_location='cpu',weights_only=True) if kind=='torch' else json.loads(path.read_text())
        except Exception as error:
            failures.append(dict(artifact=name,error=f'{type(error).__name__}: {error}'));return None

    def check(name,fn):
        try:checks[name]=fn()
        except Exception as error:failures.append(dict(artifact=name,error=f'{type(error).__name__}: {error}'))

    for name,expected in [('protocol.json',design),('original_config.json',original_cfg),('training_config.json',cfg)]:
        data=get(name)
        if data is not None:check(name,lambda data=data,expected=expected:require(data==expected,'Metadata differs from frozen design') or True)
    specs=[('incoming.pt',0,0,0,False,True),('after_warmup.pt',200,0,0,False,True)]
    specs += [(f'stage{s}_delay{delay}.pt',200+s*489,s*STAGE_FRAMES,s-1,s==4,False) for s,delay in enumerate(DELAYS,1)]
    specs += [('final.pt',2156,TOTAL_FRAMES,4,True,False)]
    policy_hashes={}
    for name,updates,frames,evals,complete,unchanged in specs:
        data=get(name,'torch')
        if data is not None:
            check(name,lambda data=data,updates=updates,frames=frames,evals=evals,complete=complete,unchanged=unchanged:
                  audit_checkpoint(data,cfg,design,original,updates,frames,evals,complete,unchanged))
            if name in checks:policy_hashes[name]=checks[name]['policy_sha256']
        del data
    if 'final.pt' in policy_hashes and 'stage4_delay60.pt' in policy_hashes:
        check('final_policy_matches_fixed_stage4',lambda:require(policy_hashes['final.pt']==policy_hashes['stage4_delay60.pt'],'Final policy is not fixed last stage') or True)
    evaluations={}
    for name in ['baseline.json']+[f'stage{s}_evaluation.json' for s in range(1,5)]+['final_evaluation.json']:
        data=get(name)
        if data is not None:
            check(name,lambda data=data:audit_evaluation(data,cfg,condition))
            evaluations[name]=data
    if 'final_evaluation.json' in evaluations and 'stage4_evaluation.json' in evaluations:
        check('final_evaluation_matches_fixed_stage4',lambda:require(evaluations['final_evaluation.json']==evaluations['stage4_evaluation.json'],'Final evaluation differs from fixed stage4') or True)
    metrics=folder/'metrics.jsonl'
    if metrics.exists():
        def audit_log():
            lines=metrics.read_text().splitlines();rows=[]
            for i,line in enumerate(lines):
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if i==len(lines)-1 and not require_complete:return dict(rows=len(rows),partial_last_upload=True)
                    raise
                finite_tree(row);rows.append(row)
                for key in ('loss','value_loss','entropy','grad_norm','replay_error','free_policy_fraction'):
                    require(isinstance(row[key],(float,int)),f'Non-numeric training metric:{key}')
                require(row['replay_error']<3e-4,'Logged noisy replay error')
                require(row['learning_frames']==max(0,row['update']-200)*FRAMES_PER_UPDATE,'Logged frame accounting')
                require(row['total_exposure_frames']==row['update']*FRAMES_PER_UPDATE,'Logged total exposure accounting')
                require(not any(row['schedule']['coverage_violations']),'Logged event coverage violation')
            require([r['update'] for r in rows]==list(range(20,20*(len(rows)+1),20)),'Training log missing/duplicated update')
            if require_complete:require(len(rows)==107 and rows[-1]['update']==2140,'Incomplete final training log')
            return dict(rows=len(rows),last_update=rows[-1]['update'] if rows else None)
        check('metrics.jsonl',audit_log)
        artifacts['metrics.jsonl']=dict(bytes=metrics.stat().st_size,sha256=sha(metrics))
    else:pending.append('metrics.jsonl')
    training=get('training.json');summary=get('summary.json')
    progress=get('progress.json')
    if progress is not None:
        def audit_progress():
            require(set(progress)=={'stages'},'Unexpected progress fields')
            require(len(progress['stages'])<=4,'Too many fixed stages')
            for s,item in enumerate(progress['stages'],1):
                require(item['stage']==s and item['learning_frames']==s*STAGE_FRAMES,'Progress stage order/budget differs')
                name=f'stage{s}_evaluation.json'
                if name in evaluations:require(item['evaluation']==evaluations[name],'Progress evaluation differs')
            if training is not None:require(progress['stages']==training['stages'],'Final progress/training stages differ')
            return dict(stages=len(progress['stages']))
        check('progress.json',audit_progress)
    if training is not None:
        def audit_training():
            finite_tree(training)
            require(training['completed'] is True and training['learning_frames']==TOTAL_FRAMES,'Training is not fixed budget complete')
            require(training['warmup_exposure_frames']==WARMUP_FRAMES and training['warmup_policy_unchanged'] is True,'Warmup exposure/hash mismatch')
            require(training['initial_policy_hash']==policy_digest(original['policy']),'Training did not begin at original parent')
            require(training['noise_replayed_updates']==2156 and training['noise_replay_max_error']<3e-4,'Incomplete noise replay')
            require(len(training['stages'])==4,'Missing fixed evaluation stage')
            for s,item in enumerate(training['stages'],1):
                require(item['stage']==s and item['max_training_delay']==DELAYS[s-1] and item['learning_frames']==s*STAGE_FRAMES,'Stage budget/order differs')
                name=f'stage{s}_evaluation.json'
                if name in evaluations:require(item['evaluation']==evaluations[name],'Embedded stage evaluation differs')
            if 'final.pt' in policy_hashes:require(training['final_policy_hash']==policy_hashes['final.pt'],'Final trained policy hash mismatch')
            return dict(learning_frames=TOTAL_FRAMES,warmup_exposure_frames=WARMUP_FRAMES,total_exposure_frames=TOTAL_FRAMES+WARMUP_FRAMES)
        check('training.json',audit_training)
    if summary is not None:
        def audit_summary():
            require(summary['completed'] is True and summary['condition']==condition,'Wrong completed summary')
            if training is not None:require(summary['training']==training,'Summary training differs')
            if 'baseline.json' in evaluations:require(summary['baseline']==evaluations['baseline.json'],'Summary baseline differs')
            if 'final.pt' in artifacts:require(summary['final_checkpoint_sha256']==artifacts['final.pt']['sha256'],'Final checkpoint file hash differs')
            return True
        check('summary.json',audit_summary)
    if (folder/'incomplete.json').exists():
        failures.append(dict(artifact='incomplete.json',error=json.loads((folder/'incomplete.json').read_text())))
    complete=not pending and not failures
    return dict(version='blind_reorientation_independent_artifact_audit_v1',
        audited_at=dt.datetime.now(dt.timezone.utc).isoformat(),folder=str(folder),condition=condition,
        status='failed' if failures else 'complete' if complete else 'pending',
        complete=complete,required_complete=require_complete,passed=(not failures and (complete or not require_complete)),
        protocol_sha256=EXPECTED_PROTOCOL_HASH,auditor_sha256=sha(__file__),checks=checks,failures=failures,pending=pending,
        artifacts=artifacts,interpretation='Data integrity audit only; it does not select favorable stages or infer a trained-seed population effect')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--condition',choices=CONDITIONS,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--require-complete',action='store_true')
    args=ap.parse_args();torch.set_num_threads(2)
    try:result=run(args.folder,args.condition,args.require_complete)
    except Exception as error:
        result=dict(status='failed',complete=False,passed=False,error=f'{type(error).__name__}: {error}',auditor_sha256=sha(__file__))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temp=args.out.with_suffix(args.out.suffix+'.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(args.out)
    print(json.dumps({k:result.get(k) for k in ('status','complete','passed','pending','failures','error')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
