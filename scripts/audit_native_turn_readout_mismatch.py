"""Independent saved-input/scalar-rendering audit; no environment or model run."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from decompose_native_turn_prefix import prefix_labels

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)

def categorical_rays(pos,heading,context,half):
    """Only ray intersections and categorical predicates, never a new image."""
    n=len(heading);columns=(torch.arange(84)+.5)/84
    ray_offset=math.pi/4-(math.pi/2)*columns
    dx=torch.cos(heading[:,None]+ray_offset);dy=torch.sin(heading[:,None]+ray_offset)
    x,y=pos[:,:1],pos[:,1:]
    large=torch.full_like(dx,1e9)
    tx=torch.where(dx>1e-9,(half-x)/dx,torch.where(dx< -1e-9,(-half-x)/dx,large))
    ty=torch.where(dy>1e-9,(half-y)/dy,torch.where(dy< -1e-9,(-half-y)/dy,large))
    horizontal=tx<ty;distance=torch.where(horizontal,tx,ty)
    wall=torch.where(horizontal,torch.where(dx>0,0,2),torch.where(dy>0,1,3))
    along=torch.where(horizontal,y+distance*dy,x+distance*dx)
    phases=context['stripe_phase'].expand(n,-1).gather(1,wall)
    stripe_value=(((along+half)/(2*half))*6+phases)%1
    depth=(distance*torch.cos(ray_offset)).clamp_min(.05);height=(.45/depth).clamp(0,1)
    row_position=(torch.arange(84)+.5)/84*2-1
    return dict(wall=wall,stripe=stripe_value<.15,stripe_value=stripe_value,
        landmark=(along-context['land_pos'].expand(n,-1).gather(1,wall)).abs()<.12,
        wall_pixel=row_position[None,:,None].abs()<height[:,None,:])

def main():
    torch.set_num_threads(1)
    out=ROOT/'results/native_turn_counterbalanced_readout_design'
    paths=dict(family=ROOT/'results/native_turn_geometry/inputs/gc100/context11_prefix1_probe_glimpse.pt',
        paired=ROOT/'results/native_turn_teacher_preflight/gc100/context11_probe_glimpse.pt',
        bank=ROOT/'results/native_turn_teacher_preflight/banks/gc100.pt',
        diagnostic=out/'mismatch_diagnosis.json')
    family,paired,bank=(load(paths[k])for k in ('family','paired','bank'));context=bank['contexts'][11]
    labels,_=prefix_labels(paired);cases=np.flatnonzero(labels==1)
    angle=torch.atan2(torch.sin(family['target_choice_heading'][:,None]-paired['target_heading'][cases]),
        torch.cos(family['target_choice_heading'][:,None]-paired['target_heading'][cases])).abs()
    ids=angle.argmin(0);assert ids.tolist()==list(range(64,128,8))
    a=family['native_x'][:,ids];b=paired['student_native_x'][:-8,cases]
    error=(a-b).abs();bad=(error[:,:,:256].amax(2)>1e-5).nonzero()
    assert bad.tolist()==[[2,7]]
    ti,ii=bad[0].tolist();phase=int(ids[ii]);case=int(cases[ii]);assert phase==120 and case==15
    assert torch.equal(a[:,:,768:],b[:,:,768:])
    ray_a=categorical_rays(family['physical']['pos'][ti,phase][None],family['physical']['hd'][ti,phase][None],context,bank['config']['arena_half'])
    ray_b=categorical_rays(paired['physical']['pos'][ti,case][None],paired['physical']['hd'][ti,case][None],context,bank['config']['arena_half'])
    changed={k:(ray_a[k]!=ray_b[k]).nonzero().tolist()for k in ('wall','stripe','landmark','wall_pixel')}
    assert changed==dict(wall=[],stripe=[[0,33]],landmark=[],wall_pixel=[])
    old=float(ray_b['stripe_value'][0,33]);new=float(ray_a['stripe_value'][0,33]);assert new<.15<=old
    diagnosis=json.loads(paths['diagnostic'].read_text())
    assert old==diagnosis['predicate_values'][0]['old_stripe_value']and new==diagnosis['predicate_values'][0]['new_stripe_value']
    bearing=torch.atan2(context['goal_center'][1]-context['endpoint'][1],context['goal_center'][0]-context['endpoint'][0])
    circle=bearing+torch.arange(128,dtype=torch.float32)*(2*math.pi/128)
    first=bearing+torch.arange(8,dtype=torch.float32)*(math.pi/8)
    pairs=torch.stack((first,first+math.pi),1).reshape(-1)
    assert torch.equal(circle,family['target_choice_heading'])and torch.equal(pairs,paired['target_heading'])
    dense_checks=[]
    for condition in ('gc100','pgc33'):
        p=ROOT/f'results/native_turn_teacher_preflight/banks/{condition}.pt';paths['bank_'+condition]=p
        for c in load(p)['contexts']:
            v=c['goal_center']-c['endpoint'];origin=torch.atan2(v[1],v[0])
            coarse=origin+torch.arange(128,dtype=torch.float32)*(2*math.pi/128)
            dense=origin+torch.arange(1024,dtype=torch.float32)*(2*math.pi/1024)
            assert torch.equal(coarse,dense[::8]);dense_checks.append(dict(condition=condition,context=c['context_id'],shared_targets_exact=True))
    record=dict(complete=True,readout_status_unchanged=True,scientific_readout_complete=False,
        observation=family['audit']['labels'][ti],paired_case=case,geometry_phase=phase,
        embedding_max_abs=float(error[:,:,:256].max()),current_code_max_abs=float(error[:,:,256:768].max()),goal_action_reward_exact=True,
        target_angle_difference_rad=float(circle[phase]-pairs[case]),
        pose_difference={k:(family['physical'][k][ti,phase]-paired['physical'][k][ti,case]).tolist()for k in ('pos','hd','mdir','vf','vs','vr','t')},
        changed_ray_predicates=changed,paired_stripe_value=old,geometry_stripe_value=new,stripe_threshold=.15,
        dense_shared_target_checks=dense_checks,
        conclusion='Float32 addition order in the paired versus full-circle target-angle formulas creates a tiny pose difference that crosses one hard stripe threshold. This explains a genuinely different visible input; the prescribed compatibility failure remains binding.',
        dense_scope='The original128 and dense1024 constructions use the same power-of-two phase formula; all32 contexts have byte-identical shared targets. This does not replace their separately required full physical/input/state guards.',
        no_CNN_RNN_environment_execution=True,no_images_regenerated=True,no_threshold_or_case_changes=True,
        source_sha256=sha(__file__),input_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths.values()})
    path=out/'mismatch_independent_validation.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k]for k in ('embedding_max_abs','target_angle_difference_rad','changed_ray_predicates','paired_stripe_value','geometry_stripe_value')}))

if __name__=='__main__':main()
