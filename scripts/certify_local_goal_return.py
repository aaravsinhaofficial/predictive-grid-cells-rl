"""Tape-specific common-information success bound for local return trials.

No policy search, neural replay, training or task modification. With identical
available inputs/body motors and antipodal headings, common body-frame noise
and action randomness give reflected paths before hits. Visiting both reflected
goal disks costs at least3(D-R). A shorter all-action path budget and full-path
wall clearance certify at most one success per equally weighted heading pair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'results/rl_blind_reorientation_review/local_goal_preflight_v2'
ROUNDING_CUSHION_M=1e-4
HEADING_TOLERANCE_RAD=1e-5


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_rows(a):
    a=np.asarray(a)
    return all(a[0].tobytes()==row.tobytes() for row in a[1:])


def all_action_path_bound(motor,v_max,noise_v,dt,tau,initial_body_velocity):
    """Triangle-inequality envelope of the exact native first-order motors.

    Motor tape is[T,3,N], with the first two rows being body-frame noise.
    The maximum over six actions includes zero translation for both turns.
    Rotation preserves speed norm; clipping/damping cannot increase the bound.
    Wall clearance is checked separately to preserve reflected-path symmetry.
    """
    tape=np.asarray(motor,dtype=np.float64)
    assert tape.ndim==3 and tape.shape[1]==3 and np.isfinite(tape).all()
    velocity=np.asarray(initial_body_velocity,dtype=np.float64)
    assert velocity.shape==(tape.shape[2],2)
    targets=np.array([[0.,0.],[v_max,0.],[-v_max,0.],[0.,v_max],[0.,-v_max]])
    noise=np.moveaxis(tape[:,:2],1,2)*noise_v
    maximum=np.linalg.norm(noise[:,:,None,:]+targets[None,None,:,:],axis=-1).max(2)
    alpha=math.exp(-dt/tau);speed=np.linalg.norm(velocity,axis=1);length=np.zeros_like(speed)
    for row in maximum:
        speed=alpha*speed+(1-alpha)*row
        length+=dt*speed
    return length


def certify(arrays,cfg,goal):
    """Certify every antipodal pair, or retain explicit reasons it fails.

    This is conditional on the frozen preflight's observation rule: black RGB,
    held ordinary position snapshot and saved goal throughout FREE; only actual
    previous action updates, and reward stays zero until first-hit termination.
    The null controller starts with the same state/randomness distribution in
    both cases. A retained heading-dependent LSTM state is outside that null.
    """
    a={key:np.asarray(value) for key,value in arrays.items()}
    pos=a['terminal_pos'];hd=a['terminal_hd'];motors=a['terminal_motors']
    x=a['first_free_input'];tape=a['motor'];uniform=a['uniform']
    n=len(hd);assert n>=2 and n%2==0
    assert pos.shape==(n,2) and motors.shape==(n,3) and x.shape==(n,1287)
    assert tape.shape==(30,3,n) and uniform.shape==(30,n)
    assert a['goal_snapshot'].shape==(n,512) and a['erased_state'].shape==(n,512)
    assert all(np.isfinite(value).all() for value in (pos,hd,motors,x,tape,uniform,a['goal_snapshot'],a['erased_state']))
    assert ((uniform>=0)&(uniform<1)).all()
    goal=np.asarray(goal,dtype=np.float64);assert goal.shape==(2,) and np.isfinite(goal).all()
    # build_env uses these unchanged ArenaEnv defaults; the producer supplies
    # cfg for the remaining native parameters and saves both source hashes.
    dt=cfg['action_repeat']/60.;tau=.25;noise_v=.03;wall_limit=cfg['arena_half']-.08
    bounds=all_action_path_bound(tape,cfg['v_max'],noise_v,dt,tau,motors[:,:2])
    goal_matches=np.array_equal(x[:,768:1280],a['goal_snapshot'])
    pa_left=np.zeros((n,6),dtype=x.dtype);pa_left[:,0]=1
    common_protocol=dict(goal_input_matches_saved_snapshot=goal_matches,
        common_last_left_action=np.array_equal(x[:,1280:1286],pa_left),
        previous_reward_zero=bool((x[:,-1]==0).all()))
    pairs=[]
    for i in range(n//2):
        j=i+n//2;ids=[i,j]
        angular_error=abs(float(np.angle(np.exp(1j*(float(hd[j])-float(hd[i])-math.pi)))))
        distance=float(np.linalg.norm(pos[i]-goal));gap=distance-cfg['goal_radius']
        wall_distance=float(np.min(wall_limit-np.abs(pos[ids])))
        bound=float(bounds[ids].max())
        conditions=dict(common_protocol,
            terminal_input_byte_identical=common_rows(x[ids]),
            saved_goal_byte_identical=common_rows(a['goal_snapshot'][ids]),
            erased_recurrent_state_byte_identical=common_rows(a['erased_state'][ids]),
            terminal_position_byte_identical=common_rows(pos[ids]),
            body_motor_state_byte_identical=common_rows(motors[ids]),
            headings_antipodal_within_tolerance=angular_error<=HEADING_TOLERANCE_RAD,
            body_motor_tapes_byte_identical=tape[:,:,i].tobytes()==tape[:,:,j].tobytes(),
            action_uniforms_byte_identical=uniform[:,i].tobytes()==uniform[:,j].tobytes(),
            starts_outside_goal=gap>ROUNDING_CUSHION_M,
            every_action_path_avoids_walls=bound+ROUNDING_CUSHION_M<wall_distance,
            path_shorter_than_both_goal_visits=bound+3*ROUNDING_CUSHION_M<3*gap)
        conditions={key:bool(value) for key,value in conditions.items()}
        passed=all(conditions.values())
        pairs.append(dict(headings=ids,conditions=conditions,certified=passed,
            common_policy_mean_success_upper_bound=.5 if passed else 1.,
            heading_antipodal_error_rad=angular_error,goal_center_distance_m=distance,
            goal_disk_gap_m=gap,three_gap_required_path_m=3*gap,
            all_action_path_upper_bound_m=bound,nearest_wall_distance_m=wall_distance,
            three_gap_margin_m=3*gap-bound,wall_clearance_margin_m=wall_distance-bound,
            failures=[key for key,value in conditions.items() if not value]))
    return dict(certified_all_pairs=all(row['certified'] for row in pairs),pairs=pairs,
        heading_trials=n,equally_weighted_pairs=n//2,
        common_information_success_upper_bound=float(np.mean([row['common_policy_mean_success_upper_bound'] for row in pairs])),
        native_physics=dict(dt=dt,tau=tau,v_max=cfg['v_max'],noise_v=noise_v,wall_limit=wall_limit,goal_radius=cfg['goal_radius']),
        rounding_cushion_m=ROUNDING_CUSHION_M,heading_tolerance_rad=HEADING_TOLERANCE_RAD,
        guarantee='Conditional on these fixed coupled motor tapes, equally weighted opposite-heading starts, the documented held-input/no-prehit-feedback rule, and no heading-informed initial controller/private-random state. Covers arbitrary common stochastic action sequences with clocks and previous-action memory, not only constant actions. Mathematical bound uses native equations with the stated numerical tolerance; it is not a universal Gaussian-noise ceiling.',
        stopping='Stop success at first actual hit, before respawn. If both cases succeed, the surviving case supplies a single continued reflected path through both disks; post-hit reset/reward in the stopped case is irrelevant.',
        proof='With common body motors/actions/noise and a pi heading offset, free displacements are opposite until a hit. Before any wall the goal targets in one common displacement frame are disks centered at plus/minus(G-P). From zero, visiting one disk costs at least D-R; passing to the other costs at least2(D-R). Therefore both successes require path length at least3(D-R). The all-action motor-norm recurrence bounds any policy path.',
        unconditional_noise_scope='Gaussian noise is unbounded. For a newly drawn tape, a uniform statement can use success<=0.5+0.5*Pr(the path/wall certificate event fails), with that failure probability separately bounded; this script does not estimate it.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=OUT)
    p.add_argument('--out',type=Path);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    producer_summary_path=args.inputs/'summary.json'
    producer_summary=json.loads(producer_summary_path.read_text())
    assert producer_summary['source_sha256']==sha(ROOT/'scripts/preflight_local_goal_return_v2.py')
    assert producer_summary['v1_helper_sha256']==sha(ROOT/'scripts/preflight_local_goal_return.py')
    assert producer_summary['contexts']==8 and producer_summary['headings']==16
    rows=[];pending=[];sources={}
    for condition in ('gc100','pgc33'):
        config=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json'
        cfg=json.loads(config.read_text());sources[str(config.relative_to(ROOT))]=sha(config)
        for context in range(8):
            folder=args.inputs/condition/f'context{context}'
            if not (folder/'result.json').exists():pending.append(f'{condition}/context{context}');continue
            producer=json.loads((folder/'result.json').read_text())
            assert (producer['condition'],producer['context'])==(condition,context)
            summary_rows=[row for row in producer_summary['rows'] if (row['condition'],row['context'])==(condition,context)]
            assert len(summary_rows)==1 and summary_rows[0]==producer
            if producer['audit']['failures']:
                rows.append(dict(condition=condition,context=context,physical_preparation_failed=True,
                    certified_all_pairs=False,failures=producer['audit']['failures']))
                continue
            with np.load(folder/'prepared.npz') as arrays:a={key:arrays[key] for key in arrays.files}
            with np.load(folder/'setup.npz') as setup:goal=setup['goal_center'].copy()
            record=certify(a,cfg,goal)
            record.update(condition=condition,context=context,physical_preparation_failed=False,
                input_sha256={name:sha(folder/name) for name in ('prepared.npz','setup.npz','result.json')})
            rows.append(record)
    complete=not pending and len(rows)==16
    if args.require_complete:assert complete,pending
    report=dict(complete=complete,pending=pending,all_contexts_certified=complete and all(row['certified_all_pairs'] for row in rows),
        rows=rows,source_sha256=sha(__file__),source_dependencies_sha256={
            name:sha(ROOT/name) for name in ('code/env.py','code/common.py','scripts/preflight_local_goal_return.py','scripts/preflight_local_goal_return_v2.py')},
        config_sha256=sources,producer_summary_sha256=sha(producer_summary_path),
        scope='Design preflight only. All16condition/context rows retained; shared physical contexts are duplicated across model input checks and are not independent trials. No training or new neural data; failure never selects a replacement context.')
    output=args.out or args.inputs/'common_information_certificate.json'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('complete','all_contexts_certified','pending')}))


if __name__=='__main__':main()
