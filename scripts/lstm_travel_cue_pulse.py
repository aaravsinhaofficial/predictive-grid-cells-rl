"""A brief current-PGC cue after a controlled travel-memory phase substitution.

Consumes independently collected coherent travel families; supplies no ring loss,
decoder, or controller to the policy. Ring geometry is tested in a separate assay.
All conditions preserve the goal code and every nonpredictive input slot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from lstm_endogenous_curve_recovery import advance,describe
from lstm_heading_curve_flow import nearest_closed_segments,curve_geometry
from lstm_heading_memory_probe import sha,save,wrap

HORIZONS=(0,1,5,15,30,45)
BRANCHES=('ordinary','correct_pulse','wrong_plus60_pulse','wrong_minus60_pulse','correct_sustained')
EDITS=('none','plus30','minus30','matched_random')


def semantic_nearest(points,curve,theta):
    """Full-coordinate closest segment, interpolated physical travel angle."""
    distance,_,closest,segment=nearest_closed_segments(points,curve)
    nxt=(segment+1)%len(curve)
    edge=curve[nxt]-curve[segment]
    fraction=np.sum((closest-curve[segment])*edge,1)/np.maximum(np.sum(edge*edge,1),1e-30)
    phase=wrap(theta[segment]+fraction*wrap(theta[nxt]-theta[segment]))
    return distance,phase


def select_input(name,step,ordinary,native):
    """step1 is the first update following the state intervention."""
    if name=='ordinary' or (name.endswith('_pulse') and step!=1):
        return ordinary.expand_as(native)
    if name=='wrong_plus60_pulse':return torch.roll(native,-len(native)//6,0)
    if name=='wrong_minus60_pulse':return torch.roll(native,len(native)//6,0)
    if name in ('correct_pulse','correct_sustained'):return native
    raise ValueError(name)


def initial_edits(state,scale):
    n=len(state);assert n%12==0
    plus=np.roll(state,-n//12,0);minus=np.roll(state,n//12,0)
    tangent=(np.roll(state,-1,0)-np.roll(state,1,0))/scale
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-30)
    random=np.random.default_rng(762021).standard_normal(state.shape)
    random-=np.sum(random*tangent,1,keepdims=True)*tangent
    random/=np.maximum(np.linalg.norm(random,axis=1,keepdims=True),1e-30)
    size=np.linalg.norm((plus-state)/scale,axis=1,keepdims=True)
    return {'none':state.copy(),'plus30':plus,'minus30':minus,
            'matched_random':state+random*size*scale}


def run_dynamics(step_function,start,ordinary,native,scale,horizons=HORIZONS):
    edits=initial_edits(start,scale)
    states={name:torch.tensor(np.concatenate([edits[e] for e in EDITS]),dtype=torch.float64)
            for name in BRANCHES}
    records={0:{name:value.numpy().reshape(len(EDITS),len(start),-1).copy() for name,value in states.items()}}
    with torch.no_grad():
        for t in range(1,max(horizons)+1):
            for name in BRANCHES:
                inp=select_input(name,t,ordinary,native).repeat(len(EDITS),1)
                states[name]=step_function(states[name],inp)
            if t in horizons:
                records[t]={name:value.numpy().reshape(len(EDITS),len(start),-1).copy() for name,value in states.items()}
    return records


def analyze(records,theta,scale,net):
    report={}
    for t,branches in records.items():
        row={}
        for name,edits in branches.items():
            # Ordinary has its own unperturbed reference. All PGC branches use
            # the correct single-pulse reference, retaining wrong-cue bias.
            # A sustained cue additionally has a matched sustained reference.
            reference_name='ordinary' if name=='ordinary' else 'correct_pulse'
            reference=branches[reference_name][0]
            baseline=edits[0]
            probs=net.pi(torch.tensor(edits[:,:,:256],dtype=torch.float64)).softmax(-1).detach().numpy()
            p0=probs[0]
            rows={}
            for j,edit in enumerate(EDITS):
                metrics={}
                for metric,s in [('raw',np.ones(512)),('relative_rms',scale)]:
                    geometry=curve_geometry(reference/s)
                    residual,phase=semantic_nearest(edits[j]/s,reference/s,theta)
                    _,own_phase=semantic_nearest(edits[j]/s,baseline/s,theta)
                    ordered=np.sort(np.mod(phase,2*np.pi))
                    coverage=1-float(np.diff(np.r_[ordered,ordered[0]+2*np.pi]).max())/(2*np.pi)
                    metrics[metric]=dict(
                        phase_mae_deg=float(np.rad2deg(np.abs(wrap(phase-theta))).mean()),
                        signed_phase_bias_deg=float(np.rad2deg(np.angle(np.exp(1j*(phase-theta)).mean()))),
                        phase_error_concentration=float(abs(np.exp(1j*(phase-theta)).mean())),
                        decoded_phase_coverage=coverage,
                        residual_over_reference_radius=describe(residual/max(geometry['radius'],1e-30)),
                        matched_branch_phase_mae_deg=float(np.rad2deg(np.abs(wrap(own_phase-theta))).mean()),
                        paired_state_error=describe(np.linalg.norm((edits[j]-baseline)/s,axis=1)),
                        reference_geometry=geometry)
                rows[edit]=dict(metrics=metrics,
                    paired_actor_total_variation=describe(.5*np.abs(probs[j]-p0).sum(-1)),
                    actor_family_probability_rms=float(np.sqrt(np.mean((probs[j]-probs[j].mean(0))**2))))
            reference_prob=net.pi(torch.tensor(reference[:,:256],dtype=torch.float64)).softmax(-1).detach().numpy()
            row[name]=dict(reference=reference_name,edits=rows,
                reference_actor_family_probability_rms=float(np.sqrt(np.mean((reference_prob-reference_prob.mean(0))**2))))
        report[str(t)]=row
    # Pilot summaries retain both signs; no inferential p-values across phases.
    contrasts={}
    for t,row in report.items():
        for metric in ('raw','relative_rms'):
            avg=lambda branch:float(np.mean([row[branch]['edits'][e]['metrics'][metric]['phase_mae_deg'] for e in ('plus30','minus30')]))
            contrasts.setdefault(t,{})[metric]={
                'ordinary_phase_error_deg':avg('ordinary'),
                'single_correct_phase_error_deg':avg('correct_pulse'),
                'single_correct_improvement_deg':avg('ordinary')-avg('correct_pulse'),
                'single_wrong_plus_phase_error_deg':avg('wrong_plus60_pulse'),
                'single_wrong_minus_phase_error_deg':avg('wrong_minus60_pulse'),
                'sustained_correct_phase_error_deg':avg('correct_sustained')}
    return dict(trajectory=report,contrasts=contrasts,
        interpretation='Direction readout is an analysis-only full-state closest-curve measurement. It does not drive the actor. A cue effect must be interpreted with independent curve geometry, retained phase coverage, radial recovery, observable activity and navigation results. Persistent cue influence alone can also occur in a non-attracting memory system.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    source_files=['scripts/lstm_travel_cue_pulse.py','scripts/lstm_endogenous_curve_recovery.py',
                  'scripts/lstm_heading_curve_flow.py','scripts/lstm_heading_memory_probe.py','code/nets.py','code/codes.py','code/common.py']
    protocol=dict(version='travel_cue_pulse_v1',frozen_before_travel_family_outcomes=True,
        phase_count=192,anchor_updates=15,recovery_updates=list(HORIZONS),last_time_seconds=4,
        phase_edits_degrees=[-30,30],wrong_cue_degrees=[-60,60],
        controls='Unperturbed, norm-matched random non-tangent state perturbation, ordinary current throughout, correct one-update PGC pulse, wrong ±60-degree one-update pulse, sustained correct current PGC.',
        intervention='Whole full-state phase substitutions from the naturally elicited travel-history family at1s; do not fit/rotate a PCA or decoder plane. Only169currentPGC input slots can differ between cue branches. Native savedgoal and everyotherinput unchanged.',
        evaluation='Same common-input recurrent dynamics through4s. One pulse lasts1/15s. Full-state nearest-segment angle, off-curve residual, paired fullstate error and original actor probability changes; raw and h/cRMS-scaled metrics. No phase is an independent trained-agent replicate.',
        reference='Ordinary branch uses unperturbed ordinary family. Allcuebranches use unperturbed correct-single-pulse family; also report own-branch paired phase/error to distinguish generic cue effects. Geometric validity and observable actor expression required to interpret phase.',
        relation_to_ring='A separate frozen travel-family analysis tests attraction and persistence. This pulse assay alone is not a ring test.',
        source_sha256={f:sha(ROOT/f) for f in source_files})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen pulse implementation changed; preserve prior protocol and use a versioned directory'
    else:save(path,protocol)
    if args.freeze_only:return
    if args.states is None or args.checkpoint is None:p.error('--states and --checkpoint required')
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=json.loads(args.config.read_text()) if args.config else data['config']
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(data['policy']);net.double()
    archive=np.load(args.states);times=archive['times'];index=int(np.flatnonzero(times==15)[0])
    all_states=archive['states'];n=len(archive['theta']);assert n>=192 and n%192==0
    subset=np.arange(0,n,n//192)
    start=all_states[index,subset].astype(np.float64);theta=archive['theta'][subset].astype(np.float64)
    assert np.max(np.abs(wrap(np.roll(theta,-1)-theta)-2*np.pi/len(theta)))<1e-4,'Need increasing uniform physical travel angles'
    ordinary=torch.tensor(archive['ordinary_input'],dtype=torch.float64)
    native=torch.tensor(archive['native_inputs'][subset],dtype=torch.float64)
    scale=archive['scale'];pop=build_population(cfg,'cpu')
    allowed=torch.zeros(1287,dtype=torch.bool);allowed[256+torch.as_tensor(pop.pgc_idx,dtype=torch.long)]=True
    assert torch.equal(native[:,~allowed],ordinary.expand_as(native)[:,~allowed]),'Cue changes nonpredictive/goal/other inputs'
    def step(state,inp):
        drive=torch.nn.functional.linear(inp,net.lstm.weight_ih,net.lstm.bias_ih)
        return advance(net.lstm,drive,state)
    records=run_dynamics(step,start,ordinary,native,scale)
    result=analyze(records,theta,scale,net)
    audit=dict(condition=cfg['cond'],checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        state_archive=str(args.states),state_archive_sha256=sha(args.states),allowed_changed_slots=int(allowed.sum()),
        nonpredictive_goal_and_other_input_max_change=0.,original_family_phases=n,analysis_phases=len(theta),
        grid_identity=bool(torch.equal(native,ordinary.expand_as(native))))
    if audit['grid_identity']:
        for t,branches in records.items():
            assert all(np.array_equal(branches['ordinary'],v) for v in branches.values()),'Grid sham branches differ'
    save(args.out/'result.json',dict(audit=audit,**result))
    np.savez_compressed(args.out/'pulse_states.npz',times=np.array(HORIZONS),theta=theta,scale=scale,
        **{name:np.stack([records[t][name] for t in HORIZONS]) for name in BRANCHES})
    print(json.dumps(dict(audit=audit,final_contrasts=result['contrasts']['45'])),flush=True)


if __name__=='__main__':main()
