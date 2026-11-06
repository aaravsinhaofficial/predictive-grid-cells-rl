"""Separate v2 local physical preflight with opposite-heading noise coupling.

Preserves v1 source/results. No training, final task freeze or model selection.
The rigorous path-length certificate is supplied by a separate audit.
"""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_env,build_policy
from lstm_endogenous_curve_recovery import sha,save
from lstm_reorientation_mechanism import physical_clearance
from lstm_reorientation_causal_navigation import navigation_draws
import preflight_local_goal_return as original
import rl_blind_reorientation as task

SEED=281801
OUT=ROOT/'results/rl_blind_reorientation_review/local_goal_preflight_v2'


def common_draws(context):
    motor,uniform=navigation_draws(1,SEED+2000+context)
    return motor.expand(-1,-1,original.NHEADINGS).clone(),uniform.expand(-1,original.NHEADINGS).clone()


class UniformPolicy:
    def pi(self,h):return torch.zeros(len(h),6,dtype=h.dtype)
    def lstm(self,x,state):return state


def make_contexts(cfg):
    rng=np.random.default_rng(SEED);contexts=[];attempts=[]
    for attempt in range(100):
        endpoint=rng.uniform(-.30,.30,size=2).astype(np.float32)
        radius=float(rng.uniform(.58,.62));bearing=float(rng.uniform(-math.pi,math.pi))
        goal=endpoint+radius*np.array([math.cos(bearing),math.sin(bearing)],dtype=np.float32)
        heading_offset=float(rng.uniform(-math.pi,math.pi));pickup_angle=float(rng.uniform(-math.pi,math.pi))
        env=build_env(dict(cfg,episode_len=10000,img_noise=0.),1,SEED+attempt,'cpu',stagger=False)
        env.reset_all();env.noise_v=env.noise_r=0.;env.goal[:]=torch.tensor(goal)
        direction=torch.tensor([math.cos(pickup_angle),math.sin(pickup_angle)])
        env.pos[0]=torch.tensor(goal)-.02*direction;env.hd.fill_(pickup_angle);env.mdir[0]=direction
        env.vf.zero_();env.vs.zero_();env.vr.zero_()
        _,reward,done,info=env.step(torch.tensor([2]))
        assert bool(info['hit'][0]) and float(reward[0])==10. and not bool(done[0])
        durations=rng.integers(5,11,size=2);turns=rng.integers(0,2,size=2);delay=int(rng.integers(30,61))
        phases=[task.VISIBLE]*15+[task.TURN_A]*int(durations[0])+[task.TURN_B]*int(durations[1])+[task.PROBE]*4
        actions=[0]*15+[int(turns[0])]*int(durations[0])+[int(turns[1])]*int(durations[1])+[2]*4
        c=dict(attempt=attempt,endpoint=endpoint,goal_center=goal,
            initial_land_pos=env.land_pos[0].numpy().copy(),initial_stripe_phase=env.stripe_phase[0].numpy().copy(),
            initial_vf=0.,initial_vs=0.,initial_vr=0.,phases=np.array(phases),actions=np.array(actions),
            pickup_position=info['hit_pos'][0].numpy().copy(),pickup_direction=info['hit_dir'][0].numpy().copy(),
            pickup_velocity=info['hit_vel'][0].numpy().copy(),pickup_reward=float(reward[0]),
            radius_m=radius,goal_bearing_rad=bearing,heading_offset_rad=heading_offset,delay=delay)
        clearance=physical_clearance(cfg,c)
        attempts.append(dict(attempt=attempt,accepted=clearance['eligible'],clearance=clearance,
            endpoint=endpoint.tolist(),goal_center=goal.tolist(),radius_m=radius,delay=delay))
        if clearance['eligible']:contexts.append(c)
        if len(contexts)==original.NCONTEXTS:break
    assert len(contexts)==original.NCONTEXTS
    return contexts,attempts


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    inherited=json.loads((original.OUT/'summary.json').read_text())
    assert inherited['source_sha256']==sha(Path(original.__file__)),'Preserve v1 implementation'
    cfg=json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    contexts,attempts=make_contexts(cfg);save(args.out/'initialization_attempts.json',attempts);rows=[]
    for condition in ('gc100','pgc33'):
        stem=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M';cfg=json.loads(stem.with_suffix('.json').read_text())
        checkpoint=stem.with_suffix('.pt');digest=sha(checkpoint);data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy'])
        for i,c in enumerate(contexts):
            target=args.out/condition/f'context{i}';target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/'setup.npz',**c)
            audit,prepared=original.prepare(net,cfg,c)
            row=dict(condition=condition,context=i,checkpoint_sha256=digest,audit=audit)
            if prepared is not None:
                terminal,states,dark,arrays=prepared;motor,uniform=common_draws(i)
                assert torch.equal(motor,motor[:,:,:1].expand_as(motor)) and torch.equal(uniform,uniform[:,:1].expand_as(uniform))
                np.savez_compressed(target/'prepared.npz',**arrays,motor=motor.numpy(),uniform=uniform.numpy())
                result={};raw={}
                for name in ('native','erase_after_glimpse'):
                    result[name],raw[name]=original.navigate(terminal,motor,uniform,name,net,states[name],dark)
                if condition=='gc100':
                    for noise in (False,True):
                        for mode in ('planner',*(f'constant{a}' for a in range(6)),'uniform'):
                            key=mode+('_native_noise' if noise else '_noiseless')
                            if mode=='uniform':
                                z=(torch.zeros(original.NHEADINGS,256),torch.zeros(original.NHEADINGS,256))
                                result[key],raw[key]=original.navigate(terminal,motor,uniform,mode,UniformPolicy(),z,dark,noise)
                            else:result[key],raw[key]=original.navigate(terminal,motor,uniform,mode,noise=noise)
                    progress=(terminal.pos-terminal.goal).norm(dim=1)[:,None]-original.predicted_next_distance(terminal)
                    row['instantaneous_information_bound']=dict(best_common_action_progress_m=float(progress.mean(0).max()),
                        heading_aware_progress_m=float(progress.max(1).values.mean()),qualification='One decision only; fullinterval theorem is separately audited')
                row['results']=result
                np.savez_compressed(target/'navigation.npz',**{f'{name}__{key}':value for name,a in raw.items() for key,value in a.items()})
            rows.append(row);save(target/'result.json',row)
            print(json.dumps(dict(condition=condition,context=i,failed=bool(audit['failures']),
                hits={name:r['hits'] for name,r in row.get('results',{}).items()})),flush=True)
        assert sha(checkpoint)==digest
    save(args.out/'summary.json',dict(version='bounded_local_goal_return_physical_preflight_v2',seed=SEED,
        contexts=original.NCONTEXTS,headings=original.NHEADINGS,rows=rows,
        source_sha256=sha(__file__),v1_helper_sha256=sha(Path(original.__file__)),
        geometry='Endpoints independently uniform central[-.30,.30]^2; goal distanceuniform[.58,.62]m; goalbearing andheadingoffset independent. Everyphysicalrejection andforcedfailure retained.',
        noise='Preparatorymotor/image noise disabled for exactaliases;FREE30×3 BODY-frame noise tape and30actionuniforms repeatedacrossall16headings withincontext; originalnative noisestandarddeviations and zero-noise planner/baseline controls. Counterfactual pathsstopfirsthit.',
        initialization='Same actualnative pickup-code procedure asv1: controlledsetupinsidegoaldisk(center−.02direction),actualforwardstep+10,hitposition/direction/velocity→CodeSource.goal→NativeGoal.advance;onesnapshotfixedpercontext. Setuprewardoutside scoredtrial, nolearning. Explicitnearbytrialpose,zeromotors,h/czero,pa−1/pr0. Firstfreeactionusesalreadyprocessedinput;noadditionalbrake.',
        scope='Designphysicalpreflight only,no training/finalsettingfreeze. Sixconstants anduniformstochasticbaseline are notthegeneral memorylessceiling; separate3d/wallclearance certificate handles that. Plannerfeasiblegreedyexpected-next-distance withtruebodyheading,nofuture-noise knowledge. Original1Bactorsareuntrainedtransfercontrols,notnewlocaltaskresults.'))


if __name__=='__main__':main()
