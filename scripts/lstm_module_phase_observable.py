"""Prospective fixed-output factor screen for every native module-phase family.

The prior full-state outcomes are retained. Both all-hidden and all-centered-
actor-score future observations reuse the previous assay's geometry and gates.
"""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import lstm_module_phase_ring as primary
import lstm_heading_curve_flow as flow
import lstm_endogenous_observable_flow as old
from lstm_endogenous_curve_recovery import advance,describe

PARENT=ROOT/'results/lstm_module_phase_ring'


def necessary_gates(initial,evolved):
    geometry=flow.curve_geometry(initial)
    evolved_geometry=flow.curve_geometry(evolved)
    radius=geometry['radius']
    _,initial_phase,_,_=flow.nearest_closed_segments(initial,initial)
    records={}
    for name,centered in [('noncentered',False),('centered',True)]:
        reference=initial-initial.mean(0) if centered else initial
        current=evolved-evolved.mean(0) if centered else evolved
        distance,phase,_,_=flow.nearest_closed_segments(current,reference)
        transport=flow.phase_transport(phase,initial_phase)
        departure=describe(distance/max(radius,1e-30))
        gates={'observable_radius_RMS_at_least1e_minus4':radius/np.sqrt(initial.shape[1])>=1e-4,
            'max_edge_below10percent_radius':geometry['max_edge_over_radius']<=.1,
            'nonlocal_separation_above5percent':geometry['nonlocal_distance_over_radius']['p10']>=.05,
            'evolved_radius_retains5percent':evolved_geometry['radius']/max(radius,1e-30)>=.05,
            'p90_invariance_residual_below5percent':departure['p90']<=.05,
            'phase_coverage80percent':transport['circular_coverage']>=.8,
            'phase_degree_one':abs(transport['winding_degree']-1)<.1,
            'phase_order_preserved':transport['backward_edge_fraction']<=.05}
        gates={k:bool(v) for k,v in gates.items()}
        records[name]={'gates':gates,'passes_necessary_gates':all(gates.values()),
                       'departure_over_radius':departure,'phase_transport':transport}
    return {'geometry':geometry,'evolved_geometry':evolved_geometry,
            'common_mean_shift_over_radius':float(np.linalg.norm(evolved.mean(0)-initial.mean(0))/max(radius,1e-30)),
            'centering':records}


@torch.inference_mode()
def observe_unperturbed(transition,curve,scale,net):
    # Exactly the baseline starts and fixed future maps used by observable_probe.
    points=flow.uniform_arc_samples(curve/scale,384)*scale
    state=torch.tensor(points,dtype=torch.float64)
    needed=set(old.OBSERVATIONS)|{old.EVOLUTION+t for t in old.OBSERVATIONS}
    hidden={};scores={}
    for step in range(old.EVOLUTION+max(old.OBSERVATIONS)+1):
        if step in needed:
            hidden[step]=state[:,:256].numpy().copy()
            logits=net.pi(state[:,:256]);logits-=logits.mean(1,keepdim=True)
            scores[step]=logits.numpy().copy()
        if step<old.EVOLUTION+max(old.OBSERVATIONS):state=transition(state)
    results={}
    for label,observed in [('all_hidden',hidden),('all_centered_actor_scores',scores)]:
        initial=np.concatenate([observed[t] for t in old.OBSERVATIONS],axis=1)
        evolved=np.concatenate([observed[old.EVOLUTION+t] for t in old.OBSERVATIONS],axis=1)
        results[label]=necessary_gates(initial,evolved)
    return results


@torch.inference_mode()
def run_model(condition,seed,args):
    begun=time.monotonic();last=begun
    _,checkpoint,net,_=primary.base.load_model(condition,seed,'cpu');net.double()
    archive_path=PARENT/f'{condition}_s{seed}/states.npz'
    archive=np.load(archive_path,allow_pickle=False)
    source=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'
    native=np.load(source,allow_pickle=False)
    rows=[]
    for regime,mode in enumerate(primary.REGIMES):
        x=torch.tensor(native['history_x'][14:15],dtype=torch.float64)
        if mode=='all_goal_withheld':x[:,768:1280]=0.
        drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
        transition=lambda z:advance(net.lstm,drive,z)
        actor=lambda z:net.pi(z[:,:256]).softmax(1)
        def scores(z):
            logits=net.pi(z[:,:256])
            return logits-logits.mean(1,keepdim=True)
        maps={'all_hidden':lambda z:z[:,:256],'all_centered_actor_scores':scores}
        for family in range(8):
            module,axis=divmod(family,2)
            scales={'raw_full_state_kicks':np.ones(512),
                    'h_c_RMS_full_state_kicks':np.repeat(archive['scales'][family],256)}
            for anchor in primary.ANCHORS:
                curve=archive[f'states_{anchor}'][regime,family]
                for metric,scale in scales.items():
                    measurements=observe_unperturbed(transition,curve,scale,net)
                    for label,record in measurements.items():
                        promote=any(c['passes_necessary_gates'] for c in record['centering'].values())
                        row={'condition':condition,'seed':seed,'module':module,'axis':axis,
                            'regime':mode,'anchor_updates':anchor,'metric':metric,'observable':label,
                            'necessary_screen':record,'full_recovery_tested':promote,
                            'centered_candidate':False,'noncentered_candidate':False}
                        if promote:
                            probe=old.observable_probe(transition,curve,scale,maps[label],actor,count=384)
                            # Baseline necessary gates must exactly agree with unchanged full helper.
                            for centering in ('centered','noncentered'):
                                expected={k:probe['candidate_gates'][centering][k] for k in record['centering'][centering]['gates']}
                                assert expected==record['centering'][centering]['gates']
                            row['full_probe']=probe
                            row['centered_candidate']=probe['centered_conditional_factor_candidate']
                            row['noncentered_candidate']=probe['noncentered_observable_candidate']
                        rows.append(row)
                    if time.monotonic()-last>25:
                        print(f'{condition} seed{seed}: {len(rows)}/192 fixed-observable cases',flush=True)
                        last=time.monotonic()
    result={'condition':condition,'seed':seed,'seconds':time.monotonic()-begun,
            'checkpoint_sha256':primary.sha(checkpoint),'archive_sha256':primary.sha(archive_path),'rows':rows}
    primary.save(args.out/f'{condition}_s{seed}.json',result)
    print(f'{condition} seed{seed}: {sum(r["centered_candidate"] for r in rows)} centered / {sum(r["noncentered_candidate"] for r in rows)} noncentered candidates; {sum(r["full_recovery_tested"] for r in rows)} recovery probes; {result["seconds"]:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/lstm_module_phase_observable')
    parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    assert json.loads((PARENT/'summary.json').read_text())['candidate_count']==0
    design={'role':'Prospective separate fixed-output follow-up after all primary full-state module-phase screens failed; original outcomes retained',
        'cohort':'All32 original1B module/axis families, both common-input regimes,5/30/100s anchors, both full-state kick metrics',
        'maps':'All256h or all6 trained actor scores with per-state action-common score subtracted; fixed observations0/5/10/15updates, evolve30updates',
        'cases':768,'centering':'Both noncentered and one common phase-family mean removed; no independent perturbed-family centering',
        'staging':'Compute exact384uniform full-state-arc baseline future maps and all8 necessary gates for everycase. Any passing centered/noncentered branch triggers unchanged observable_probe including ±radial/tangent full-state kicks, nontriviality guards and gain gates. Untested recovery never passes.',
        'thresholds':'Every existing observable_probe threshold unchanged, including RMS amplitude1e-4, residual.05radius,coverage.8,radialgain.5,radial/tangent.5',
        'controls':'Shared-drift ring passes centered necessary gates; shrinkingplane and vanishing observable fail; existing full observable positive/negative tests reused',
        'parent_protocol_sha256':primary.sha(PARENT/'protocol.json'),'parent_summary_sha256':primary.sha(PARENT/'summary.json'),
        'source_sha256':{p:primary.sha(ROOT/p) for p in ['scripts/lstm_module_phase_observable.py','scripts/lstm_module_phase_ring.py',
            'scripts/lstm_endogenous_observable_flow.py','scripts/lstm_heading_curve_flow.py','scripts/lstm_actor_logit_observable.py']},
        'archive_sha256':{f'{c}_s{s}':primary.sha(PARENT/f'{c}_s{s}/states.npz') for c in ('gc100','pgc33') for s in (1,2)}}
    manifest=args.out/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==design,'Frozen source changed; use fresh directory'
    else:primary.save(manifest,design)
    if args.freeze_only:return
    assert not (args.out/'summary.json').exists(),'Do not overwrite completed outcomes'
    results=[run_model(c,s,args) for c in ('gc100','pgc33') for s in (1,2)]
    rows=[row for result in results for row in result['rows']]
    primary.save(args.out/'summary.json',{'cases':len(rows),'centered_candidates':sum(r['centered_candidate'] for r in rows),
        'noncentered_candidates':sum(r['noncentered_candidate'] for r in rows),'recovery_probes':sum(r['full_recovery_tested'] for r in rows),'results':results})


if __name__=='__main__':main()
