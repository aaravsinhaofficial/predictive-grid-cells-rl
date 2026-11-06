"""Release original PGC heading families from their fixed current-direction cue.

Reuse the frozen curve-flow implementation with an explicit initial-state/input
provider. Native saved states are byte-identical; only current PGC slots change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import lstm_heading_curve_flow as flow
from lstm_heading_memory_probe import sha,save

NATIVE=ROOT/'results/lstm_heading_memory/curve_flow'


def ordinary_current_input(pop,x,position,direction):
    ordinary=pop.activity(torch.as_tensor([position]),torch.as_tensor([direction]),torch.zeros(1,2),shift_scale=0.).double()
    result=x.clone();indices=torch.as_tensor(pop.pgc_idx,dtype=torch.long)+256
    result[:,indices]=ordinary[:,pop.is_pgc]
    changed=torch.zeros(x.shape[1],dtype=torch.bool);changed[indices]=True
    assert torch.equal(result[:,~changed],x[:,~changed])
    return result,dict(allowed_changed_slots=int(changed.sum()),
                      numerically_changed_slots=int((result!=x).sum()),
                      saved_goal_max_change=float((result[:,768:1280]-x[:,768:1280]).abs().max()),
                      other_input_max_change=float((result[:,~changed]-x[:,~changed]).abs().max()))


def run(seed,args):
    folder=NATIVE/f'pgc33_s{seed}';native=json.loads((folder/'context0.json').read_text())
    archive=np.load(folder/'context0_states.npz')
    initial=torch.tensor(archive['states'][0],dtype=torch.float64)
    x=torch.tensor(archive['dark_input'],dtype=torch.float64)
    checks={}
    def provider(net,pop,cfg,unused_context,context,parameters):
        assert context==0 and parameters.angles*2==len(initial)
        modified,audit=ordinary_current_input(pop,x,native['physics_checks']['position'],native['physics_checks']['fixed_last_movement_direction'])
        checks.update(audit)
        return initial.clone(),modified,None,None,dict(native['physics_checks'],current_direction_release=audit)
    parameters=SimpleNamespace(phases=native['phases'],remesh=native['remesh_points'],embed_device='cpu',out=args.out)
    # This provider replaces only the elicitation call; all geometry, recurrent
    # evolution, radial kicks and candidate criteria are the frozen base method.
    with patch.object(flow,'elicit',provider):
        record=flow.model_context('pgc33',seed,parameters)
    record.update(input_condition='current_pgc_as_ordinary_grid',
                  release_source_sha256=sha(__file__),paired_native_record_sha256=sha(folder/'context0.json'),
                  paired_native_states_sha256=sha(folder/'context0_states.npz'),
                  initial_full_state_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                  input_change_audit=checks,
                  release_interpretation='The common fixed current PGC direction may pin a phase. This matched release changes only current predictive slots; native goal snapshot and all other inputs stay fixed. Saved native initial states are injected into the frozen flow implementation exactly.')
    assert record['checkpoint_sha256']==native['checkpoint_sha256']
    save(args.out/f'pgc33_s{seed}/context0.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_heading_memory/current_release')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(experiment='Original PGC heading-family release from fixed current predictive-direction cue',
        seeds=[1,2],context=0,baseline=str(NATIVE),
        change='Only169 CURRENT PGC slots become ordinary-grid values at the identical position. Saved native goal code, black-image embedding, previous turn action, reward and recurrent weights unchanged.',
        initial_states='Exactly the saved768 native-prepared full(h,c) states from the paired original flow assay; no new preparation or initial-state fitting.',
        analysis='Identical frozen100s full-state flow, fixed-curve invariance and radial-vs-tangent criteria; both raw and relativeRMS metrics. The earlier native runs are paired controls. Retain release results regardless of outcome.',
        base_protocol_sha256=sha(NATIVE/'protocol.json'),
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_heading_current_release.py','scripts/lstm_heading_curve_flow.py','scripts/lstm_heading_memory_probe.py','scripts/lstm_endogenous_curve_recovery.py']},
        paired_native_sha256={f's{s}_{name}':sha(NATIVE/f'pgc33_s{s}/{name}') for s in (1,2) for name in ('context0.json','context0_states.npz')})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen release protocol changed'
    else:save(path,protocol)
    if args.freeze_only:return
    results=[run(s,args) for s in (1,2)]
    save(args.out/'summary.json',dict(candidates=sum(r['passes_both_metric_candidate_screen'] for r in results),
        results=[{k:r[k] for k in ('seed','candidate_gates','passes_both_metric_candidate_screen','input_change_audit')} for r in results]))


if __name__=='__main__':main()
