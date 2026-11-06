"""Frozen held-out original-seed travel-memory replication, continuous native RGB.

Seeds1/2 are method development. Seeds3–8 use unchanged travel preparation and
candidate gates, with fresh policy-generated contexts selected by geometry only.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from code.common import build_policy
from lstm_heading_memory_probe import sha,save
import lstm_continued_heading_memory as continued
import lstm_travel_memory_probe as travel

NATIVE_ELIGIBILITY=continued.eligible_window


class VisibleWindowClock:
    """Bookkeeping windows after observed goals; RGB is always available."""
    def __init__(self,n,device,seed,groups=1,visible_steps=15):
        self.known=torch.zeros(n*groups,dtype=torch.bool,device=device)
        self.blind=torch.zeros_like(self.known)
        self.remaining=torch.full((n*groups,),visible_steps,dtype=torch.long,device=device)
        self.visible_steps=visible_steps
    def mask(self):
        return self.blind  # always false; no observation intervention
    def advance(self,hit,done,max_gap):
        active=self.known & ~hit & ~done
        self.remaining-=active.long()
        complete=active & (self.remaining==0)
        restart=complete | hit | done
        self.remaining=torch.where(restart,torch.full_like(self.remaining,self.visible_steps),self.remaining)
        self.known=(self.known | hit) & ~done
        return complete


def run(condition,seed,args):
    stem=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    data=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert data['frames']==1_000_000_000
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(data['policy'])
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    collection_args=SimpleNamespace(device=args.device,out=folder)
    def eligible(window,slot,radius):
        ok,reason=NATIVE_ELIGIBILITY(window,slot,radius)
        if not ok:return ok,reason
        geometry=travel.path_clearance(window[0]['pos'][slot],window[0]['goal_center'][slot],cfg)
        return (True,'accepted') if geometry['eligible'] else (False,'all_angle_forward_path_clearance')
    with patch.object(continued,'VisualClock',VisibleWindowClock),patch.object(continued,'eligible_window',eligible):
        contexts=continued.collect_contexts(net,cfg,stem.with_suffix('.pt'),collection_args)
    # Correct the shared collector's descriptive schedule string explicitly.
    metadata=json.loads((folder/'collection.json').read_text())
    metadata.update(schedule='Continuous native RGB at every update.15-update bookkeeping windows after actual goal discovery; actual hits restart windows and episode resets clear them.',
                    heldout_seed=seed,window_clock_source_sha256=sha(__file__))
    save(folder/'collection.json',metadata)
    results=[]
    for context,window in enumerate(contexts):
        result=travel.analyze(dict(checkpoint=str(stem.with_suffix('.pt')),config=str(stem.with_suffix('.json'))),window,context,args)
        result['heldout_source_sha256']=sha(__file__)
        result['context_collection']='Fresh original-policy rollouts with continuous RGB, first two physically eligible windows from distinct slots'
        save(folder/f'context{context}.json',result);results.append(result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='mps');p.add_argument('--out',type=Path,default=ROOT/'results/lstm_travel_memory/heldout_original')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();args.cohort='heldout_original'
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='heldout_original_travel_v1',seeds=list(range(3,9)),conditions=['gc100','pgc33'],
        method_development_seeds=[1,2],contexts_per_model=2,
        collection='Original1B policy with continuous native RGB/current/goal/action/reward streams. Deterministic consecutive15-update windows after goal discovery; actual hits restart and episode endings clear the bookkeeping clock. First2 physically eligible complete windows from distinct slots, chronological order, seed98201,n_envs32,maximum6000decisions.',
        physical_eligibility='Same unchanged sufficient all-angle forward-path clearance>.01m from walls and actual goal disk; selection never uses neural geometry, decoding, performance or cue response.',
        analysis='Exact unchanged coherent-travel preparation, explicit stop, current-PGC release, full512raw/relative geometry,2s+2s fixed-curve/radial candidate gates,192readout angles with768rendered phases. No active predictive current cue in autonomous screen. All outcomes retained.',
        reporting='Trained seed is the replication unit; contexts and phases are nested measurements. Report12models/24contexts, both populations and every frozen gate. Preserve seeds1/2 as development cohort.',
        execution_order='After the final25.024M pair primary assays. This protocol is frozen before inspecting heldout heading/travel outcomes.',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_travel_memory_heldout.py','scripts/lstm_travel_memory_probe.py','scripts/lstm_continued_heading_memory.py','scripts/lstm_heading_curve_flow.py','scripts/lstm_heading_memory_probe.py','scripts/lstm_endogenous_curve_recovery.py','code/env.py','code/nets.py','code/codes.py']},
        checkpoints={f'{c}_s{s}':sha(ROOT/f'models/{c}/{c}_lstm_s{s}_1000M.pt') for c in ('gc100','pgc33') for s in range(3,9)})
    frozen=args.out/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol,'Frozen heldout protocol changed'
    else:save(frozen,protocol)
    if args.freeze_only:return
    results=[]
    for condition in ('gc100','pgc33'):
        for seed in range(3,9):results.extend(run(condition,seed,args))
    save(args.out/'summary.json',dict(models=12,contexts=len(results),
        candidates=sum(r['ordinary_release']['passes_both_metric_candidate_screen'] for r in results),
        results=[dict(condition=r['condition'],seed=r['seed'],context=r['context'],
                      candidate=r['ordinary_release']['passes_both_metric_candidate_screen']) for r in results]))


if __name__=='__main__':main()
