"""Independent read-only reconciliation of fixed40M primary/causal artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
BASE=ROOT/'results/lstm_reorientation_mechanism/final'
CUES=('native','ordinary','plus60','minus60')
METRICS=('raw','relative_rms')
WORKERS={'gc100':'ring-20261105-reorient-grid-01','pgc33':'ring-20261105-reorient-pgc-02'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def path_upper_bound(motor,v_max,noise_v,dt,tau):
    """All-action path-length bound; body rotations preserve the velocity norm."""
    noise=np.moveaxis(np.asarray(motor,dtype=np.float64)[:,:2],1,2)*noise_v
    targets=np.array([[0,0],[v_max,0],[-v_max,0],[0,v_max],[0,-v_max]])
    max_target=np.linalg.norm(noise[:,:,None,:]+targets[None,None,:,:],axis=-1).max(2)
    retained=math.exp(-dt/tau);speed=np.zeros(noise.shape[1]);distance=np.zeros_like(speed)
    for row in max_target:
        speed=retained*speed+(1-retained)*row;distance+=dt*speed
    return distance


def audit_primary(folder,condition,checkpoint_hash,context):
    result=read(folder/f'context{context}.json');path=folder/f'context{context}_states.npz'
    assert result['checkpoint_sha256']==checkpoint_hash
    assert result['source_sha256']==sha(ROOT/'scripts/lstm_reorientation_mechanism.py')
    physics=result['physics'];assert physics['phases']==768 and physics['physical_clearance']['eligible']
    assert physics['rotated_wall_hits']==physics['rotated_goal_hits']==0
    assert physics['saved_goal_unchanged'] and physics['all_other_cue_inputs_unchanged']
    assert physics['current_cue_change_slots']==(0 if condition=='gc100' else 169)
    assert tuple(result['branches'])==CUES
    with np.load(path) as a:
        assert str(a['checkpoint_sha256'].item())==checkpoint_hash
        assert tuple(a['times'])==(0,1,5,15,30,45,60)
        assert a['ordinary_input'].shape==(1,1287) and np.array_equal(a['ordinary_input'],a['native_input'])
        for cue,branch in result['branches'].items():
            assert a[cue+'_states'].shape==(7,768,512) and np.isfinite(a[cue+'_states']).all()
            assert branch['passes_both_metric_candidate_screen']==all(all(g.values()) for g in branch['candidate_gates'].values())
            if condition=='gc100':
                assert np.array_equal(a[cue+'_states'],a['native_states'])
                assert np.array_equal(a[cue+'_glimpse_inputs'],a['native_glimpse_inputs'])
    return dict(context=context,archive_sha256=sha(path),physics_eligible=True,
        primary_candidates={cue:b['passes_both_metric_candidate_screen'] for cue,b in result['branches'].items()},
        grid_arrays_exact_shams=condition=='gc100')


def audit_causal(folder,primary_folder,context,checkpoint_hash,condition):
    result=read(folder/'result.json');manifest=read(folder/'protocol.json');inputs=read(folder/'inputs.json')
    assert inputs['checkpoint_sha256']==checkpoint_hash
    assert inputs['state_archive_sha256']==sha(primary_folder/f'context{context}_states.npz')
    for source,digest in manifest['source_sha256'].items():assert sha(ROOT/source)==digest
    audit=result['audit'];assert audit['phases']==192 and audit['dense_reference_phases']==768
    assert audit['other_cue_inputs_byte_identical'] and audit['post_glimpse_reset_erases_all_state_edit_differences_exactly']
    assert audit['predictive_current_slots']==(0 if condition=='gc100' else 169)
    if condition=='gc100':assert audit['grid_sham'] and audit['grid_behavior_sham_exact']
    with np.load(primary_folder/f'context{context}_states.npz') as a:goal=a['goal_center'];endpoint=a['endpoint']
    configs=read(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json')
    groups=0;windows=0;hits=0;max_progress_error=0.;max_length_error=0.
    with np.load(folder/'navigation_paths.npz') as a:
        assert a['theta'].shape==(192,) and a['motor_draws'].shape==(30,3,192)
        assert array_hash(a['motor_draws'])==audit['motor_draw_sha256']
        assert array_hash(a['action_uniforms'])==audit['action_uniform_sha256']
        from lstm_reorientation_causal_navigation import physical_free_start
        env=physical_free_start(configs,a['theta'],endpoint,goal)
        assert not env.vf.any() and not env.vs.any()
        bound=path_upper_bound(a['motor_draws'],env.v_max,env.noise_v,env.dt,env.tau)
        required=max(float(np.linalg.norm(endpoint-goal))-env.goal_radius,0.)
        opportunity=dict(initial_goal_distance_m=float(np.linalg.norm(endpoint-goal)),
            goal_radius_m=env.goal_radius,minimum_required_translation_m=required,
            maximum_path_bound_over_phases_m=float(bound.max()),
            phases_with_goal_impossible_under_any_action_sequence=int((bound+1e-5<required).sum()),
            phases=192,rounding_cushion_m=1e-5,
            method='Native motor low-pass speed norm bounded recursively by maximum target speed among all six actions plus the fixed translational-noise draws; body rotation preserves norm and walls can only shorten displacement. Shared zero translational initial velocity. This is a descriptive opportunity bound, never a context exclusion.')
        assert len(result['branches'])==18
        for branch,record in result['branches'].items():
            assert 'none' in record['navigation']
            for edit,stats in record['navigation'].items():
                prefix=f'{branch}__{edit}__';actions=a[prefix+'actions'];path=a[prefix+'path']
                first=a[prefix+'first_hit_step'];probs=a[prefix+'first_action_probabilities']
                assert actions.shape==(30,192) and path.shape==(31,192,2) and first.shape==(192,)
                assert probs.shape==(192,6) and np.allclose(probs.sum(1),1,atol=1e-12)
                assert np.isfinite(path).all() and np.isfinite(probs).all()
                assert ((first==-1)|((first>=1)&(first<=30))).all()
                assert stats['windows']==192 and stats['goals']==int((first>0).sum())
                assert stats['zero_hit_windows']==int((first<0).sum())
                assert abs(stats['hit_fraction']-(first>0).mean())<3e-8
                assert stats['processed_inputs_after_first_free_action']==29
                for index,hit in enumerate(first):
                    stop=int(hit) if hit>0 else 30
                    assert ((actions[:stop,index]>=0)&(actions[:stop,index]<6)).all()
                    assert (actions[stop:,index]==-1).all()
                    assert np.array_equal(path[stop:,index],np.repeat(path[stop:stop+1,index],31-stop,axis=0))
                    if hit>0:assert np.linalg.norm(path[hit,index]-goal)<configs['goal_radius']+1e-6
                progress=np.linalg.norm(path[0]-goal,axis=1)-np.linalg.norm(path[-1]-goal,axis=1)
                length=np.linalg.norm(np.diff(path,axis=0),axis=2).sum(0)
                pe=float(np.abs(progress-a[prefix+'total_progress_m']).max());le=float(np.abs(length-a[prefix+'path_length_m']).max())
                assert pe<2e-6 and le<2e-6
                max_progress_error=max(max_progress_error,pe);max_length_error=max(max_length_error,le)
                groups+=1;windows+=192;hits+=stats['goals']
                if '__native_reset' in branch:
                    for key in ('actions','path','first_hit_step','first_action_probabilities'):
                        assert np.array_equal(a[prefix+key],a[f'{branch}__none__{key}'])
        if condition=='gc100':
            for metric in METRICS:
                for anchor in ('','_radial15'):
                    for edit in result['branches'][f'{metric}__ordinary{anchor}']['navigation']:
                        for key in ('actions','path','first_hit_step','first_action_probabilities'):
                            reference=a[f'{metric}__ordinary{anchor}__{edit}__{key}']
                            for cue in CUES:assert np.array_equal(reference,a[f'{metric}__{cue}{anchor}__{edit}__{key}'])
    return dict(context=context,perturbation_groups=groups,windows=windows,hit_windows=hits,
        complete192_phase_denominators=True,exact_first_hit_stop=True,shared_tapes_verified=True,
        max_path_recomputed_progress_error= max_progress_error,max_path_recomputed_length_error=max_length_error,
        opportunity=opportunity,result_sha256=sha(folder/'result.json'),array_sha256=sha(folder/'navigation_paths.npz'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    reports={};pending=[]
    for condition,worker in WORKERS.items():
        folder=BASE/condition;checkpoint=ROOT/'results/ring_cloud_artifacts'/worker/'final.pt'
        if not (folder/'summary.json').exists():pending.append(condition+'/primary');continue
        checksum=sha(checkpoint);manifest=read(folder/'implementation_protocol.json')
        for source,digest in manifest['source_sha256'].items():assert sha(ROOT/source)==digest
        collection=read(folder/'collection.json');assert collection['contexts']==2 and collection['checkpoint_sha256']==checksum
        report=dict(checkpoint_sha256=checksum,primary=[],causal=[])
        for context in (0,1):
            report['primary'].append(audit_primary(folder,condition,checksum,context))
            c=folder/'causal'/f'context{context}'
            if not (c/'result.json').exists() or not (c/'navigation_paths.npz').exists():pending.append(f'{condition}/causal/context{context}')
            else:report['causal'].append(audit_causal(c,folder,context,checksum,condition))
        reports[condition]=report
    if args.require_complete:assert not pending
    output=dict(passed=True,complete=not pending,pending=pending,source_sha256=sha(__file__),models=reports,
        scope='Independent saved-artifact reconciliation; does not replace native-physics/source tests or rerun neural outcomes. All defined interventions/192phases and zero-hit windows retained.')
    (BASE/'independent_artifact_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(passed=True,complete=not pending,pending=pending)))


if __name__=='__main__':main()
