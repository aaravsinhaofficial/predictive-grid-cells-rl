"""Saved-pose renderer/predicate diagnosis; no environment or model execution."""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/native_turn_counterbalanced_readout_design'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)


def raycast(pos,hd,context,cfg):
    """Literal native deterministic raycast algebra on archived numeric poses."""
    n=len(hd);W=84;H=cfg['arena_half'];fov=math.radians(90.)
    cols=(torch.arange(W)+.5)/W;col_ang=fov/2-fov*cols;v=(torch.arange(W)+.5)/W*2-1
    ang=hd[:,None]+col_ang[None];dx,dy=torch.cos(ang),torch.sin(ang);x,y=pos[:,:1],pos[:,1:]
    big=torch.full_like(dx,1e9)
    tx=torch.where(dx>1e-9,(H-x)/dx,torch.where(dx< -1e-9,(-H-x)/dx,big))
    ty=torch.where(dy>1e-9,(H-y)/dy,torch.where(dy< -1e-9,(-H-y)/dy,big))
    hit_x=tx<ty;t=torch.where(hit_x,tx,ty);zero=torch.zeros_like(dx,dtype=torch.long)
    wall=torch.where(hit_x,torch.where(dx>0,zero,zero+2),torch.where(dy>0,zero+1,zero+3))
    along=torch.where(hit_x,y+t*dy,x+t*dx);depth=(t*torch.cos(col_ang)[None]).clamp_min(.05);wall_h=(.45/depth).clamp(0.,1.)
    colors=torch.tensor([[.85,.25,.25],[.25,.75,.30],[.30,.40,.90],[.90,.80,.25]])
    base=colors[wall];phase=torch.gather(context['stripe_phase'].expand(n,-1),1,wall)
    stripe_value=(((along+H)/(2*H))*6+phase)%1.;stripe=stripe_value<.15
    col=torch.where(stripe[...,None],base*.45,base)
    land_distance=(along-torch.gather(context['land_pos'].expand(n,-1),1,wall)).abs();land=land_distance<.12
    col=torch.where(land[...,None],torch.ones_like(col),col);shade=(1/(1+.5*depth)).clamp(.25,1.);col=col*shade[...,None]
    vv=v[None,:,None];is_wall=vv.abs()<wall_h[:,None,:];floor_shade=.25+.45*vv.clamp(0.,1.)
    bg=torch.where((vv<0)[...,None],torch.tensor([.12,.12,.16]),torch.tensor([.5,.5,.5])*floor_shade[...,None])
    rgb=torch.where(is_wall[...,None],col[:,None],bg).clamp(0.,1.).permute(0,3,1,2).contiguous()
    return rgb,dict(hit_x=hit_x,wall=wall,stripe=stripe,land=land,is_wall=is_wall,
        stripe_value=stripe_value,land_distance=land_distance,wall_h=wall_h)


def main():
    torch.set_num_threads(1)
    paths=dict(family=ROOT/'results/native_turn_geometry/inputs/gc100/context11_prefix1_probe_glimpse.pt',
        raw=ROOT/'results/native_turn_teacher_preflight/gc100/context11_probe_glimpse.pt',
        bank=ROOT/'results/native_turn_teacher_preflight/banks/gc100.pt',
        readout=ROOT/'results/native_turn_counterbalanced_readout/summary.json')
    family,raw,bank=(load(paths[k])for k in ('family','raw','bank'));context=bank['contexts'][11];cfg=bank['config']
    assert cfg['vision']=='raycast';time=2;case=15;phase=120
    a,ap=raycast(family['physical']['pos'][time],family['physical']['hd'][time],context,cfg)
    b,bp=raycast(raw['physical']['pos'][time],raw['physical']['hd'][time],context,cfg)
    assert tensor_sha(b)==raw['full_native_rgb_sha256_by_observation'][time]
    changed={k:(ap[k][phase]!=bp[k][case]).nonzero().tolist()for k in ('hit_x','wall','stripe','land','is_wall')}
    rgbdiff=(a[phase]-b[case]).abs();where=(rgbdiff>1e-5).nonzero();details=[]
    for key in ('stripe','land','is_wall'):
        for index in changed[key]:
            column=index[-1]
            details.append(dict(predicate=key,index=index,old=bool(bp[key][case][tuple(index)]),new=bool(ap[key][phase][tuple(index)]),
                old_stripe_value=float(bp['stripe_value'][case,column]),new_stripe_value=float(ap['stripe_value'][phase,column]),
                old_land_distance=float(bp['land_distance'][case,column]),new_land_distance=float(ap['land_distance'][phase,column]),
                old_wall_h=float(bp['wall_h'][case,column]),new_wall_h=float(ap['wall_h'][phase,column])))
    downstream=[]
    cases=np.arange(1,16,2);indices=np.arange(64,128,8)
    result=json.loads(paths['readout'].read_text())
    protocol=ROOT/'docs/research/native_turn_counterbalanced_readout_protocol.json';paths['protocol']=protocol
    tolerance=json.loads(protocol.read_text())['numerical_provenance_tolerances']
    for endpoint in ('original','clean','noise'):
        folder=ROOT/'results/native_turn_original_memory/gc100'if endpoint=='original'else ROOT/f'results/native_turn_distillation/gc100/{endpoint}/final_paired'
        paired_path=folder/'context11_probe_glimpse.pt';paired=load(paired_path);paths['paired_'+endpoint]=paired_path
        for branch in ('native','ordinary','pre_cue_reset_native','pre_cue_reset_ordinary'):
            path=ROOT/f'results/native_turn_counterbalanced_readout/gc100/{endpoint}/context11_prefix1_probe_glimpse_{branch}.npz';paths[endpoint+'_'+branch]=path
            with np.load(path)as z:newq=z['probabilities'][indices].copy();newlog=z['logits'][indices].copy();newphysics=z['action_progress_m'][indices].astype(float)
            row=next(r for r in result['rows']if(r['condition'],r['endpoint'],r['context'],r['kind'],r['prefix'],r['branch'])==('gc100',endpoint,11,'probe_glimpse',1,branch))
            name=row['exact_sham_of']or f'context11_prefix1_probe_glimpse_{branch}'
            sp=ROOT/f'results/native_turn_geometry/gc100/{endpoint}/{name}/left/baseline_states.npz';paths[endpoint+'_'+branch+'_state']=sp
            assert sha(sp)==row['source_state_sha256']
            with np.load(sp)as z:newstate=z['t8'][indices].copy()
            oldstate=paired['variants'][branch]['first_choice_state_after_input'][cases].numpy().astype(float)
            oldq=paired['variants'][branch]['first_choice_probabilities'][cases].numpy().astype(float);oldq/=oldq.sum(1,keepdims=True)
            oldlog=paired['variants'][branch]['first_choice_logits'][cases].numpy().astype(float)
            oldphysics=raw['first_update_physics']['progress_m'][cases].numpy().astype(float)
            hdiff=newstate[:,:256]-oldstate[:,:256];cdiff=newstate[:,256:]-oldstate[:,256:]
            logdiff=(newlog-newlog.mean(1,keepdims=True))-(oldlog-oldlog.mean(1,keepdims=True))
            pdiff=newq-oldq;progressdiff=1000*((newq*newphysics).sum(1)-(oldq*oldphysics).sum(1))
            downstream.append(dict(endpoint=endpoint,branch=branch,embedding_equivalence_rejected=True,cases=cases.tolist(),geometry_indices=indices.tolist(),
                downstream_frozen_tolerance_checks=dict(hidden=bool(np.all(np.abs(hdiff)<=tolerance['hidden'])),
                    cell=bool(np.all(np.abs(cdiff)<=tolerance['cell_atol']+tolerance['cell_rtol']*np.abs(oldstate[:,256:]))),
                    centered_logits=bool(np.all(np.abs(logdiff)<=tolerance['centered_logits'])),
                    probability=bool(np.all(np.abs(pdiff)<=tolerance['probability']))),
                hidden_max_abs=float(np.max(np.abs(hdiff))),hidden_RMS=float(np.sqrt(np.mean(hdiff**2))),
                cell_max_abs=float(np.max(np.abs(cdiff))),cell_RMS=float(np.sqrt(np.mean(cdiff**2))),
                centered_logit_max_abs=float(np.max(np.abs(logdiff))),probability_max_abs=float(np.max(np.abs(pdiff))),
                expected_progress_difference_mm=progressdiff.tolist(),expected_progress_mean_difference_mm=float(progressdiff.mean()),
                expected_progress_max_abs_difference_mm=float(np.max(np.abs(progressdiff))),
                discrepant_phase120=dict(hidden_max_abs=float(np.max(np.abs(hdiff[-1]))),cell_max_abs=float(np.max(np.abs(cdiff[-1]))),
                    centered_logit_max_abs=float(np.max(np.abs(logdiff[-1]))),probability_max_abs=float(np.max(np.abs(pdiff[-1]))),expected_progress_difference_mm=float(progressdiff[-1]))))
    record=dict(complete=True,diagnosis_only=True,scientific_run_complete=False,readout_status_unchanged=True,
        condition='gc100',context=11,kind='probe_glimpse',prefix_action=1,observation_index=time,label=family['audit']['labels'][time],paired_case=case,geometry_phase=phase,
        old_rgb_batch_sha256=tensor_sha(b),archived_old_rgb_batch_sha256=raw['full_native_rgb_sha256_by_observation'][time],
        exact_old_rgb_reconstruction=True,pose_difference={k:(family['physical'][k][time,phase]-raw['physical'][k][time,case]).tolist()for k in family['physical']},
        embedding_max_abs=float((family['native_x'][time,phase,:256]-raw['student_native_x'][time,case,:256]).abs().max()),
        changed_predicates=changed,predicate_values=details,rgb_max_abs=float(rgbdiff.max()),rgb_components_above_1e_minus5=int(len(where)),
        changed_pixel_rows=sorted(set(where[:,1].tolist())),changed_pixel_columns=sorted(set(where[:,2].tolist())),
        downstream_saved_array_differences=downstream,
        interpretation='A tiny phase/initial-position arithmetic difference crosses a hard native rendering predicate during visible encoding. The physical timing/history actions agree, but the encoded observations are not equivalent. Existing tolerance failures remain binding; no tensors, cases or statuses are changed.',
        no_CNN_recurrent_policy_or_environment_execution=True,renderer_algebra_only=True,
        source_sha256=sha(__file__),renderer_source_sha256=sha(ROOT/'code/env.py'),input_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths.values()})
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'mismatch_diagnosis.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez_compressed(OUT/'mismatch_rgb_arrays.npz',old=b[case].numpy(),new=a[phase].numpy(),absolute_difference=rgbdiff.numpy())
    print(json.dumps({k:record[k]for k in ('changed_predicates','predicate_values','rgb_max_abs','changed_pixel_rows','changed_pixel_columns')},indent=2))


if __name__=='__main__':main()
