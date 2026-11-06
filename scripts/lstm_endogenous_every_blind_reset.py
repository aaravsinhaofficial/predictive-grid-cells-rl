"""Test ongoing recurrent computation by resetting before EVERY blind update.

The frozen evaluator is untouched. Only its clock's returned reset indicator
changes; internal clock states, duration draws, pixels and code inputs do not.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import rl_visual_memory as parent
from code.common import build_policy
from lstm_endogenous_curve_recovery import sha,save


class EveryBlindUpdateClock(parent.VisualClock):
    def advance(self,hit,done,max_gap):
        super().advance(hit,done,max_gap)
        return self.mask()


def protocol():
    return dict(experiment='Every scheduled blind-update recurrent reset; separate from frozen onset-reset comparison.',
        models=['original gc100 seed1','original pgc33 seed1','final25.024M visual-memory gc100 seed1','final25.024M visual-memory pgc33 seed1'],
        evaluation_seed=98201,episodes=64,n_envs=32,max_gap=60,visible_updates=15,
        intervention='Subclass VisualClock.advance: call the original method, then return self.mask() instead of the entering-blind indicator. The frozen evaluator resets h and c before each next scheduled blind input in *_reset branches.',
        controls='All internal clock transitions and RNG calls unchanged. Seen-reset follows the same scheduled periods while retaining normal images. Before first discovery no additional resets occur. Goal snapshots, current code, previous action/reward, weights, and rewards unchanged.',
        validation='All four nonreset episode-row arrays must exactly match the frozen parent evaluator on the same checkpoint/backend. A supplied same-backend parent JSON may be reused; otherwise rerun unmodified parent locally. Mismatch is retained and raises an error.',
        inference='Every-step reset tests continuing recurrent computation; onset reset tests carrying earlier state into the blackout. Neither alone identifies a ring.',
        source_sha256=sha(__file__),parents={p:sha(ROOT/p)for p in ['scripts/rl_visual_memory.py','scripts/rl_goal_memory.py','scripts/lstm_code_swap.py','code/nets.py','code/env.py']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--device',default='mps');p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/every_blind_update_reset')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    design=protocol();path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==design
    else:save(path,design)
    if args.freeze_only:return
    if args.checkpoint is None:p.error('--checkpoint required')
    started=time.monotonic();data=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=data.get('config') or json.loads((args.config or args.checkpoint.with_suffix('.json')).read_text())
    original=data.get('frames')==1_000_000_000
    continued=data.get('completed') and data.get('extra_frames')==25_024_000
    assert original or continued,'Only the prespecified original and fixed25.024M final checkpoints are allowed.'
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
    if args.reference:
        reference=json.loads(args.reference.read_text());reference_source=dict(path=str(args.reference),sha256=sha(args.reference),role='Supplied parent evaluator result; exact nonreset row identity required.')
    else:
        reference=parent.evaluate(net,cfg,device=args.device,seed=98201,episodes=64,n_envs=32,max_gap=60)
        save(args.out/'parent_onset_reference.json',reference);reference_source=dict(path='parent_onset_reference.json',role='Unmodified frozen parent evaluator rerun on the same backend.')
    assert reference['seed']==98201 and reference['episodes_per_variant']==64 and reference['max_blind_gap']==60
    with patch.object(parent,'VisualClock',EveryBlindUpdateClock):
        result=parent.evaluate(net,cfg,device=args.device,seed=98201,episodes=64,n_envs=32,max_gap=60)
    checks={v:result['variants'][v]['rows']==reference['variants'][v]['rows']for v in parent.VARIANTS if not v.endswith('_reset')}
    score=lambda r,v:r['variants'][v]['mean_score']
    result.update(condition=cfg['cond'],cohort='original' if original else 'continued25M',checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        source_sha256=sha(__file__),device=args.device,torch_version=torch.__version__,reference_source=reference_source,
        nonreset_episode_rows_exactly_match_parent=checks,all_nonreset_rows_identical=all(checks.values()),
        onset_parent_reset_costs=reference['descriptive_contrasts'],
        every_step_minus_onset_reset_cost=dict(blind=(score(result,'blind')-score(result,'blind_reset'))-(score(reference,'blind')-score(reference,'blind_reset')),
            seen=(score(result,'seen')-score(result,'seen_reset'))-(score(reference,'seen')-score(reference,'seen_reset'))),seconds=time.monotonic()-started)
    save(args.out/'result.json',result)
    print(json.dumps(dict(condition=cfg['cond'],cohort=result['cohort'],all_nonreset_rows_identical=all(checks.values()),
        contrasts=result['descriptive_contrasts'],state_reset_counts=result['state_reset_counts'],seconds=result['seconds'])),flush=True)
    assert all(checks.values()),'Nonreset rows changed; result retained but cannot be interpreted as the isolated reset intervention.'


if __name__=='__main__':main()
