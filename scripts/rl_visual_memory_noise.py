"""Reward-only blackout training with prespecified recurrent state noise.

Paired with the frozen noiseless50.048M extension, starting from the same25.024M
parents. Noise follows the LSTM update and precedes actor/critic evaluation. It
is sampled once per rollout and reused exactly for differentiable replay.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import inspect
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy
import rl_visual_memory_extend as extension
from rl_goal_memory import atomic_json,state_digest

pilot=extension.pilot
SIGMA=.02
NOISE_SEED=126201
EVALUATION_NOISE_SEED=128201


class StateNoise:
    def __init__(self,device,sigma=SIGMA,seed=NOISE_SEED):
        self.sigma=float(sigma);self.device=device
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.max_replay_error=0.;self.replays=0
        self.base_collect=pilot.collect_rollout
        self.base_replay=extension.replay
        self.base_capture=extension.capture_runtime
        self.base_restore=extension.restore_runtime

    def draw(self,state,mask,groups=1):
        n=state[0].shape[0];assert n%groups==0
        base=torch.randn(2,n//groups,state[0].shape[1],generator=self.generator,
                         device=state[0].device,dtype=state[0].dtype)
        noise=base.repeat(1,groups,1)*self.sigma*mask[None,:,None]
        return noise[0],noise[1]

    def collect(self,*args,**kwargs):
        bound=inspect.signature(self.base_collect).bind(*args,**kwargs).arguments
        net,clock,length=bound['net'],bound['clock'],bound['length']
        recorded=[]
        def hook(module,inputs,state):
            eps=self.draw(state,clock.mask())
            recorded.append(tuple(e.detach().clone() for e in eps))
            return tuple(s+e for s,e in zip(state,eps))
        handle=net.lstm.register_forward_hook(hook)
        try:result=self.base_collect(*args,**kwargs)
        finally:handle.remove()
        assert len(recorded)==length+1,'Exactly one independent bootstrap draw is required'
        result[0]['recurrent_noise']=recorded[:length]
        # The final draw belongs only to the bootstrap value, never saved state.
        result[0]['bootstrap_recurrent_noise']=recorded[-1]
        return result

    def replay(self,net,roll,state0):
        index=0;noise=roll['recurrent_noise']
        def hook(module,inputs,state):
            nonlocal index
            eps=noise[index];index+=1
            return tuple(s+e for s,e in zip(state,eps))
        handle=net.lstm.register_forward_hook(hook)
        try:logits,values=self.base_replay(net,roll,state0)
        finally:handle.remove()
        assert index==len(noise)
        discrepancy=float((logits.detach()-torch.stack(roll['actor_logits'])).abs().max())
        assert discrepancy<3e-4,f'Noisy rollout/replay mismatch:{discrepancy}'
        self.max_replay_error=max(self.max_replay_error,discrepancy);self.replays+=1
        return logits,values

    def capture(self,runtime,device):
        return dict(self.base_capture(runtime,device),recurrent_noise_rng=self.generator.get_state().cpu(),
                    recurrent_noise_sigma=self.sigma,recurrent_noise_replays=self.replays,
                    recurrent_noise_max_replay_error=self.max_replay_error)

    def restore(self,runtime,snapshot,device):
        self.base_restore(runtime,snapshot,device)
        assert snapshot['recurrent_noise_sigma']==self.sigma
        self.generator.set_state(snapshot['recurrent_noise_rng'].cpu())
        self.replays=snapshot['recurrent_noise_replays']
        self.max_replay_error=snapshot['recurrent_noise_max_replay_error']

    @contextmanager
    def installed(self):
        with patch.object(pilot,'collect_rollout',self.collect),patch.object(extension,'replay',self.replay),\
             patch.object(extension,'capture_runtime',self.capture),patch.object(extension,'restore_runtime',self.restore):
            yield self


def evaluate_noisy(net,cfg,*,device,deadline=float('inf')):
    """Same per-slot disturbances across all six matched behavioral branches."""
    holder={};base_clock=pilot.VisualClock
    class EvaluationClock(base_clock):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);holder['clock']=self
    noise=StateNoise(device,seed=EVALUATION_NOISE_SEED)
    def hook(module,inputs,state):
        eps=noise.draw(state,holder['clock'].mask(),groups=len(pilot.VARIANTS))
        return tuple(s+e for s,e in zip(state,eps))
    with patch.object(pilot,'VisualClock',EvaluationClock):
        handle=net.lstm.register_forward_hook(hook)
        try:
            result=pilot.evaluate(net,cfg,device=device,seed=98201,episodes=64,n_envs=32,
                                  max_gap=60,deadline=deadline)
        finally:handle.remove()
    result['recurrent_noise']=dict(sigma_h=SIGMA,sigma_c=SIGMA,seed=EVALUATION_NOISE_SEED,
        injection='After LSTM update, before actor/critic, on every scheduled black-input update; also during corresponding seen-control times',
        matched_noise_across_branches=True)
    return result


def protocol():
    base=extension.protocol()
    base.update(version='visual_memory_state_noise_v3',
        literature='https://arxiv.org/html/2408.00109v2 section6: state-noise robustness motivates attraction; their task and noise scale differ',
        noise=dict(sigma_h=SIGMA,sigma_c=SIGMA,seed=NOISE_SEED,distribution='Independent zero-mean Gaussian per coordinate and update',
                   timing='After the LSTM computes(h,c), before actor and critic; active only when VisualClock.mask is true',
                   replay='Store each sampled h/c noise vector and reuse it exactly through BPTT; independent bootstrap draw; noise has no gradients',
                   sampling='Dedicated RNG; draws do not consume environment, clock or action generators; full RNG saved for exact resume',
                   units='Same absolute0.02 in both h and c, not equal relative disturbance because their amplitudes differ. No outcome-based tuning.'),
        comparator='Frozen noiseless v2 continuation from exactly the same two v1 final policies and RMSProp states, with the same task, learning frames and fresh-episode seeds',
        extra_evaluation='Matched noisy6-branch navigation before training and at the fixed final endpoint; clean evaluation before and at every fixed block as in v2',
        hypotheses='Whether generic reward-trained noise robustness produces selective recovery and a persistent circular state, and whether PGC removal/restoration changes it. No ring, angular target, angle loss, decoder or controller is added.',
        source_sha256={**base['source_sha256'],**{name:pilot.sha(ROOT/name) for name in
             ['scripts/rl_visual_memory_noise.py','tests/test_rl_visual_memory_noise.py']}})
    return base


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition',choices=tuple(extension.NAMES));ap.add_argument('--out',type=Path)
    ap.add_argument('--device',default='cuda');ap.add_argument('--max-seconds',type=int,default=6600)
    ap.add_argument('--protocol',type=Path,default=ROOT/'docs/research/rl_visual_memory_noise_protocol.json')
    ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
    design=protocol()
    if args.freeze_only:
        if args.protocol.exists():raise FileExistsError('Preserve frozen noise protocol')
        design['frozen_at']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(args.protocol,design)
        print(pilot.sha(args.protocol));return
    if args.condition is None or args.out is None:ap.error('--condition and --out required')
    frozen=json.loads(args.protocol.read_text());frozen.pop('frozen_at')
    assert frozen==design,'Frozen noise sources/assets changed'
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'protocol.json').exists():raise FileExistsError('Noise experiment output already exists')
    atomic_json(args.out/'protocol.json',design)
    folder=extension.parent_folder(args.condition)
    parent=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    assert parent['completed'] and parent['extra_frames']==extension.PARENT_FRAMES
    assert {float(s['step']) for s in parent['optimizer']['state'].values()}=={2155.}
    cfg=parent['config'];assert cfg['cond']==args.condition
    net=build_policy(cfg,args.device);net.load_state_dict(parent['policy'],strict=True)
    assert extension.tree_equal(net.state_dict(),parent['policy'])
    deadline=time.monotonic()+args.max_seconds
    try:
        incoming_hash=state_digest(net)
        baseline=pilot.evaluate(net,cfg,device=args.device,seed=98201,episodes=64,n_envs=32,max_gap=60,deadline=deadline)
        expected=json.loads((folder/'final_evaluation.json').read_text())
        assert all(baseline['variants'][v]['rows']==expected['variants'][v]['rows'] for v in pilot.VARIANTS)
        atomic_json(args.out/'baseline.json',baseline)
        atomic_json(args.out/'baseline_noisy.json',evaluate_noisy(net,cfg,device=args.device,deadline=deadline))
        assert state_digest(net)==incoming_hash
        noise=StateNoise(args.device)
        with noise.installed():result=extension.train(net,cfg,parent,args,design,deadline)
        atomic_json(args.out/'training.json',result)
        final_hash=state_digest(net)
        atomic_json(args.out/'final_noisy_evaluation.json',evaluate_noisy(net,cfg,device=args.device,deadline=deadline))
        assert state_digest(net)==final_hash
        atomic_json(args.out/'summary.json',dict(completed=True,condition=args.condition,training=result,
            recurrent_noise=dict(sigma_h=SIGMA,sigma_c=SIGMA,replayed_updates=noise.replays,
                                 max_rollout_replay_error=noise.max_replay_error),
            baseline_matches_parent_behavior_exactly=True,final_sha256=pilot.sha(args.out/'final.pt')))
    except Exception as error:
        atomic_json(args.out/'incomplete.json',dict(error_type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':main()
