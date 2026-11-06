"""Saved-baseline physical reachability bound for RIGHT-turn/RIGHT-strafe only.

No environment, policy, optimizer or random generator is executed. The bound is
for arbitrary switching within the stated two-action set, not softmax support.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch

PROTOCOL=ROOT/'docs/research/feedback_restricted_action_bound_protocol.json'
BEHAVIOR=ROOT/'docs/research/local_return_feedback_behavior_protocol.json'
BASELINE=ROOT/'results/local_return_feedback_design/portable/cloud_audit'
N=30
POSITION_MARGIN=1e-4
ANGLE_MARGIN=1e-4


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def bound(position,heading,motors,goal,tape,cfg):
    """One saved physical start/tape. Undefined premises retain possibility=True."""
    p=np.asarray(position,dtype=float);g=np.asarray(goal,dtype=float)
    m=np.asarray(motors,dtype=float);noise=np.asarray(tape,dtype=float)
    assert p.shape==g.shape==(2,) and m.shape==(3,) and noise.ndim==2 and noise.shape[1]==3
    assert all(np.isfinite(x).all() for x in (p,g,m,noise)) and math.isfinite(heading)
    n=len(noise);dt=cfg['action_repeat']/60.;v=cfg['v_max'];omega=cfg['rot_max'];a=math.exp(-dt/.25)
    assert v>0 and omega>0 and n>0
    result=dict(possible=True,reason=None)
    if not np.array_equal(m[:2],np.zeros(2)) or m[2]<0:
        result['reason']='Initial zero translation and LEFT-or-zero rotation premise failed';return result
    # Native additive motor noise is independent of actions in body coordinates.
    noise_velocity=np.zeros(3);velocity=[]
    for draw in noise:
        noise_velocity=a*noise_velocity+(1-a)*draw*np.array([.03,.03,.15]);velocity.append(noise_velocity.copy())
    velocity=np.array(velocity)
    E=dt*np.linalg.norm(velocity[:,:2],axis=1).sum()+POSITION_MARGIN
    phi=float(np.abs(dt*np.cumsum(velocity[:,2])).max())+ANGLE_MARGIN
    displacement=g-p;D=float(np.linalg.norm(displacement));R=cfg['goal_radius']
    wall=cfg['arena_half']-.08-float(np.abs(p).max())
    path_bound=n*dt*v+E
    result.update(distance_m=D,translation_noise_radius_m=E,angular_noise_margin_rad=phi,
                  all_action_path_bound_m=path_bound,wall_clearance_m=wall)
    if path_bound>=wall:
        result['reason']='Whole possible path is not certified wall-free';return result
    expanded=R+E
    if D<=expanded:
        result['reason']='Noise-expanded goal contains start';return result
    q0=-float(m[2]);z0=q0/omega
    powers=a**np.arange(1,n+1)
    Z=dt*np.cumsum(1+(z0-1)*powers)
    Lmin=D-expanded
    qmin=q0*dt*float(powers.sum())
    qmax=max(0.,omega*(max(0.,float(Z.max()))-Lmin/v))
    width=qmax-qmin+2*phi
    result.update(minimum_control_path_m=Lmin,minimum_clockwise_angle_rad=qmin,
                  maximum_clockwise_angle_rad=qmax,control_sector_width_rad=width)
    if width>=math.pi:
        result['reason']='Control direction sector is not narrower than pi';return result
    center=heading-math.pi/2-(qmin+qmax)/2
    bearing=math.atan2(displacement[1],displacement[0])
    separation=abs(math.atan2(math.sin(bearing-center),math.cos(bearing-center)))
    half=width/2+math.asin(expanded/D)+ANGLE_MARGIN
    result.update(possible=separation<=half,reason='Certified native-equation angle sector',
                  sector_center_rad=center,goal_bearing_rad=bearing,
                  expanded_half_angle_rad=half,angle_exclusion_margin_rad=separation-half)
    return result


def analyze():
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['source_sha256']==sha(__file__) and protocol['status']=='frozen_before_bound_counts'
    assert protocol['actions']==[1,5] and protocol['free_updates']==N
    inputs={str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    def tracked(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None:assert digest==expected,path
        inputs[str(path.relative_to(ROOT))]=digest;return path
    design=json.loads(tracked(BEHAVIOR).read_text());record=design['bank']
    plan=torch.load(tracked(ROOT/record['path'],record['sha256']),map_location='cpu',weights_only=True)
    assert plan['headings']==16 and plan['n_contexts']==16
    rows=[];physical={};actual_counts={}
    for condition in ('gc100','pgc33'):
        audit_path=BASELINE/f'baseline_{condition}_retained_actor.json'
        audit=json.loads(tracked(audit_path).read_text())
        assert audit['passed'] and audit['checks']['evaluations']['baseline']['complete']
        assert audit['protocol_sha256']==protocol['training_protocol_sha256']
        folder=ROOT/'results/ring_cloud_artifacts'/f"ring-20261105-feedback-{'grid' if condition=='gc100' else 'pgc'}-retain-02"/'experiment'
        cfg=json.loads(tracked(folder/'training_config.json',audit['artifacts']['training_config.json']['sha256']).read_text())
        assert {k:cfg[k] for k in protocol['physics']}==protocol['physics']
        actual_counts[condition]=audit['checks']['evaluations']['baseline']['totals']['action_counts']
        for context in range(16):
            prefix=f'baseline_evaluation/context{context}/'
            result=json.loads(tracked(folder/prefix/'result.json',audit['artifacts'][prefix+'result.json']['sha256']).read_text())
            if not result['valid']:
                rows.extend(dict(condition=condition,context=context,heading_index=h,possible=True,
                    reason='Invalid archived context; retained as unresolved') for h in range(16))
                continue
            raw_path=tracked(folder/prefix/result['raw']['file'],result['raw']['sha256'])
            with gzip.open(raw_path,'rb') as f:raw=torch.load(f,map_location='cpu',weights_only=True)
            assert len(raw['first_free'])==16
            start=15+int(plan['duration_a'][context*16])+int(plan['duration_b'][context*16])+4+int(plan['delay'][context*16])
            for h in range(16):
                i=context*16+h;first=raw['first_free'][h]
                data=(first['position'].numpy(),float(first['heading']),first['motors'].numpy(),plan['goal'][i].numpy(),plan['motor'][start:start+N,:,i].numpy())
                if condition=='gc100':physical[(context,h)]=data
                elif (context,h) in physical:
                    assert all(np.array_equal(a,b) for a,b in zip(data,physical[(context,h)])), 'Populations have different saved physical histories'
                value=bound(*data,cfg)
                rows.append(dict(condition=condition,context=context,heading_index=h,first_free_global_step=start,**value))
            del raw
    summaries=[]
    for condition in ('gc100','pgc33'):
        selected=[r for r in rows if r['condition']==condition]
        summaries.append(dict(condition=condition,trials=256,possible_trials=sum(r['possible'] for r in selected),
            unknown_trials=sum(r['reason']!='Certified native-equation angle sector' for r in selected),
            bound_percent=100*sum(r['possible'] for r in selected)/256,
            actual_baseline_action_counts=actual_counts[condition],
            contexts=[dict(context=c,possible=sum(r['possible'] for r in selected if r['context']==c),headings=16) for c in range(16)]))
    return dict(version='feedback_restricted_actions_saved_bound_v1',complete=True,source_sha256=sha(__file__),
        protocol_sha256=sha(PROTOCOL),input_sha256=inputs,rows=rows,summaries=summaries,
        shared_physical_bank_exact_for_available_contexts=True,no_policy_environment_optimizer_or_rng_execution=True,
        scope=protocol['scope'],floating_point_scope=protocol['floating_point_scope'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/local_return_feedback_design/portable/restricted_action_bound')
    args=parser.parse_args();assert not args.out.exists(),'Preserve previous descriptive analysis'
    result=analyze();args.out.mkdir(parents=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Can turning right and strafing right reach every goal?','',
        'This is a secondary calculation from the fixed **pretraining baseline**. It bounds any controller restricted to these two actions, with the saved native motor-noise tapes. It runs no policy or environment.','',
        '| Population | Potentially reachable headings | Analytical upper bound | Unresolved headings |',
        '|---|---:|---:|---:|']
    for row in result['summaries']:
        lines.append(f"| {row['condition']} | {row['possible_trials']}/256 | {row['bound_percent']:.1f}% | {row['unknown_trials']} |")
    lines += ['', 'All 16 contexts and 16 headings remain. A heading that passes the necessary angle condition is only potentially reachable. Missing premises would retain that heading as possible, never exclude it.', '',
        'The retained Grid baseline sampled only actions 1 and 5; PGC also sampled nine backward actions. Both full softmax policies can assign nonzero probability to other actions. This bound concerns the restricted action repertoire, not the full networks, and does not measure memory or a ring.', '',
        'The native-equation calculation uses fixed 0.1 mm spatial and 0.0001 rad angular margins. It is not a formal proof of every hardware floating-point instruction. [Full tables](summary.json) retain every angular margin and physical premise. [Derivation](../../../../docs/research/feedback_restricted_action_bound.md).','']
    (args.out/'README.md').write_text('\n'.join(lines));print(json.dumps(result['summaries']))


if __name__=='__main__':main()
