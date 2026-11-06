"""Prospective two-phase module manifold screen in frozen native navigation LSTMs.

Periodic synthetic inputs do not imply an attracting torus. Geometry, full
transition invariance, two surviving tangents, and true transverse recovery
are measured independently. Any candidate requires further finite-amplitude
and strongest-normal validation before a mechanistic attractor claim.
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
import lstm_module_phase_ring as one
from lstm_endogenous_curve_recovery import advance,describe

ANCHORS=(75,450)
HORIZONS=(15,30)
OFFSETS=(0,5,10,15)


def torus_code(pop,pickup,arrival,module,phases):
    phases=np.atleast_2d(phases).astype(float)
    angles=pop.orientation[module]+np.array([np.pi/6,np.pi/2])
    basis=pop.spacings[module]*np.c_[np.cos(angles),np.sin(angles)]
    selected=pop.module==module
    k,phi,shifts=(pop._np[name] for name in ('K','phi','shifts'))
    base=np.asarray(pickup)[None]+shifts[:,None]*np.asarray(arrival)[None]
    x=np.broadcast_to(base,(len(phases),pop.n_cells,2)).copy()
    x[:,selected]+=(phases@basis)[:,None]
    projection=np.einsum('bnd,nid->bni',x-phi,k)
    rates=np.clip((np.cos(projection).sum(-1)+1.5)/4.5,0.,1.)
    speed=np.einsum('nid,jd->nij',k,basis)*selected[:,None,None]
    derivative=-np.einsum('bni,nij->bnj',np.sin(projection),speed)/4.5
    derivative*=pop.sharpness*rates[:,:,None]**(pop.sharpness-1)
    return rates**pop.sharpness,derivative,basis


def normal_component(vectors,tangents):
    """Project off BOTH tangent directions, allowing arbitrary output dimension."""
    gram=np.einsum('ndi,ndj->nij',tangents,tangents)
    cross=np.einsum('ndi,ndk->nik',tangents,vectors)
    coefficients=np.linalg.pinv(gram,rcond=1e-12)@cross
    residual=vectors-np.einsum('ndi,nik->ndk',tangents,coefficients)
    return residual,coefficients


def differential_metrics(initial,final,phase_initial,phase_final,normal_initial,normal_final,amplitude):
    first,_=normal_component(normal_initial,phase_initial)
    last,phase_shift=normal_component(normal_final,phase_final)
    n0=np.linalg.norm(first,axis=1);n1=np.linalg.norm(last,axis=1)
    gain=n1/np.maximum(n0,1e-30)
    gram=np.einsum('ndi,ndj->nij',phase_initial,phase_initial)
    values,basis=np.linalg.eigh(gram)
    inverse=basis@np.array([np.diag(1/np.sqrt(np.maximum(v,1e-30))) for v in values])@np.swapaxes(basis,1,2)
    whitened=np.einsum('ndi,nij->ndj',phase_final,inverse)
    singular=np.linalg.svd(whitened,compute_uv=False)
    phase_gain=singular[:,-1]
    tangent_singular=np.linalg.svd(phase_initial,compute_uv=False)
    ratio=tangent_singular[:,-1]/np.maximum(tangent_singular[:,0],1e-30)
    radius=np.sqrt(np.mean(np.sum((initial-initial.mean(0))**2,axis=1)))
    nontrivial=amplitude*n0>=1e-4*max(radius,1e-30)
    floor=1e-4*max(radius,1e-30)
    total_initial=np.linalg.norm(normal_initial,axis=1)
    total_final=np.linalg.norm(normal_final,axis=1)
    emerging=(amplitude*total_initial<floor)&(amplitude*total_final>=floor)
    per_direction=[]
    for k in range(n0.shape[1]):
        informative=nontrivial[:,k]
        if informative.mean()>=.8:
            g=float(np.median(gain[informative,k]))
            ratio_gain=float(np.median(gain[informative,k]/np.maximum(phase_gain[informative],1e-30)))
            status='initial_transverse_response_measurable'
            passes=g<=.5 and ratio_gain<=.5
        elif np.all(amplitude*n0[:,k]<floor) and np.all(amplitude*n1[:,k]<floor) and not emerging[:,k].any():
            g=None;ratio_gain=None
            status='no_transverse_response_in_either_window'
            passes=True
        else:
            g=None;ratio_gain=None;status='initial_transverse_visibility_insufficient';passes=False
        per_direction.append({'direction':k,'status':status,'initial_transverse_visible_fraction':float(informative.mean()),
            'normal_gain_median_informative':g,'normal_phase_ratio_median_informative':ratio_gain,
            'emerging_total_observable_fraction':float(emerging[:,k].mean()),'passes':bool(passes and not emerging[:,k].any())})
    return {'normal_gain':describe(gain),'minimum_phase_gain':describe(phase_gain),
        'normal_over_minimum_phase_gain':describe(gain/np.maximum(phase_gain[:,None],1e-30)),
        'initial_phase_rank_ratio':describe(ratio),'phase_rank2_fraction':float(np.mean(ratio>=.05)),
        'normal_observable_nontrivial_fraction':float(nontrivial.mean()),
        'normal_induced_phase_coefficients_RMS':float(np.sqrt(np.mean(phase_shift**2))),
        'initial_normal_norms':n0.tolist(),'final_normal_norms':n1.tolist(),
        'initial_total_observable_norms':total_initial.tolist(),'final_total_observable_norms':total_final.tolist(),
        'emerging_observable_fraction':float(emerging.mean()),'per_direction':per_direction,
        'phase_minimum_gains':phase_gain.tolist(),
        'gates':{'phase_rank2_in90percent':bool(np.mean(ratio>=.05)>=.9),
                 'both_phase_directions_retain_half_median':bool(np.median(phase_gain)>=.5),
                 'normal_observable_nontrivial_in80percent':bool(nontrivial.mean()>=.8),
                 'no_initially_silent_later_visible_kicks':bool(not emerging.any()),
                 'all_normal_directions_qualify_and_recover':bool(all(r['passes'] for r in per_direction)),
                 'normal_gain_median_at_most_half':bool(np.median(gain)<=.5),
                 'normal_phase_ratio_median_at_most_half':bool(np.median(gain/np.maximum(phase_gain[:,None],1e-30))<=.5)}}


def mesh_indices(n):
    tri=[];coordinates=[]
    for i in range(n):
        for j in range(n):
            tri.append([i*n+j,((i+1)%n)*n+j,i*n+(j+1)%n])
            coordinates.append([[i/n,j/n],[(i+1)/n,j/n],[i/n,(j+1)/n]])
            tri.append([((i+1)%n)*n+(j+1)%n,i*n+(j+1)%n,((i+1)%n)*n+j])
            coordinates.append([[(i+1)/n,(j+1)/n],[i/n,(j+1)/n],[(i+1)/n,j/n]])
    return np.array(tri),np.array(coordinates)


def nearest_torus_mesh(points,vertices,n):
    """Closest point on every periodic full-dimensional triangle; no decoder."""
    triangles,coordinates=mesh_indices(n)
    center=vertices.mean(0);p=points-center;v=vertices-center
    a,b,c=(v[triangles[:,j]] for j in range(3));e=b-a;f=c-a
    ee=(e*e).sum(1);ff=(f*f).sum(1);ef=(e*f).sum(1)
    u=p@e.T-(a*e).sum(1);w=p@f.T-(a*f).sum(1)
    base=(p*p).sum(1)[:,None]+(a*a).sum(1)-2*p@a.T
    determinant=ee*ff-ef**2
    alpha=(ff*u-ef*w)/np.maximum(determinant,1e-30)
    beta=(ee*w-ef*u)/np.maximum(determinant,1e-30)
    valid=(determinant>1e-12*np.maximum(ee*ff,1e-30))[None]&(alpha>=0)&(beta>=0)&(alpha+beta<=1)
    distances=[np.where(valid,np.maximum(base-alpha*u-beta*w,0.),np.inf)]
    alphas=[alpha];betas=[beta]
    t=np.clip(u/np.maximum(ee,1e-30),0,1)
    distances.append(np.maximum(base-2*t*u+t*t*ee,0));alphas.append(t);betas.append(np.zeros_like(t))
    t=np.clip(w/np.maximum(ff,1e-30),0,1)
    distances.append(np.maximum(base-2*t*w+t*t*ff,0));alphas.append(np.zeros_like(t));betas.append(t)
    edge=ee+ff-2*ef;dot=w-u-ef+ee
    t=np.clip(dot/np.maximum(edge,1e-30),0,1)
    distances.append(np.maximum(base-2*u+ee-2*t*dot+t*t*edge,0));alphas.append(1-t);betas.append(t)
    stacked=np.stack(distances,axis=-1);which=np.argmin(stacked,axis=-1)
    d=np.take_along_axis(stacked,which[...,None],axis=-1)[...,0]
    aa=np.take_along_axis(np.stack(alphas,axis=-1),which[...,None],axis=-1)[...,0]
    bb=np.take_along_axis(np.stack(betas,axis=-1),which[...,None],axis=-1)[...,0]
    best=d.argmin(1);row=np.arange(len(points));aa=aa[row,best];bb=bb[row,best]
    coord=coordinates[best,0]+aa[:,None]*(coordinates[best,1]-coordinates[best,0])+bb[:,None]*(coordinates[best,2]-coordinates[best,0])
    return np.sqrt(d[row,best]),np.mod(coord,1.)


def geometry_metrics(initial,final,n):
    radius=np.sqrt(np.mean(np.sum((initial-initial.mean(0))**2,axis=1)))
    grid=initial.reshape(n,n,-1)
    maxedge=max(np.linalg.norm(np.roll(grid,-1,axis=a)-grid,axis=-1).max() for a in (0,1))/max(radius,1e-30)
    uv=np.stack(np.meshgrid(np.arange(n)/n,np.arange(n)/n,indexing='ij'),axis=-1).reshape(-1,2)
    phase_delta=np.abs(uv[:,None]-uv[None]);phase_delta=np.minimum(phase_delta,1-phase_delta)
    mask=np.linalg.norm(phase_delta,axis=-1)>=.25
    z=initial-initial.mean(0)
    distances=np.sqrt(np.maximum((z*z).sum(1)[:,None]+(z*z).sum(1)[None]-2*z@z.T,0))
    distances[~mask]=np.inf
    nonlocal_p10=float(np.quantile(distances.min(1)/max(radius,1e-30),.1))
    results={}
    for label,center in [('noncentered',False),('centered',True)]:
        a=initial-initial.mean(0) if center else initial
        b=final-final.mean(0) if center else final
        departure,phase=nearest_torus_mesh(b,a,n)
        phase=phase.reshape(n,n,2)
        du=np.mod(np.roll(phase,-1,axis=0)-phase+.5,1)-.5
        dv=np.mod(np.roll(phase,-1,axis=1)-phase+.5,1)-.5
        wind_u=du.sum(0);wind_v=dv.sum(1)
        winding_fraction=float(np.mean(np.r_[np.all(np.abs(wind_u-[1,0])<.1,axis=1),np.all(np.abs(wind_v-[0,1])<.1,axis=1)]))
        determinant=du[:,:,0]*dv[:,:,1]-du[:,:,1]*dv[:,:,0]
        orientation_fraction=float(np.mean(determinant>0))
        bins=np.floor(phase.reshape(-1,2)*6).astype(int)%6
        coverage=len(np.unique(bins,axis=0))/36
        gates={'observable_RMS_at_least1e_minus4':bool(radius/np.sqrt(initial.shape[1])>=1e-4),
            'mesh_max_edge_at_most40percent_radius':bool(maxedge<=.4),
            'nonlocal_separation_p10_at_least5percent_radius':bool(nonlocal_p10>=.05),
            'invariance_p90_at_most5percent_radius':bool(np.quantile(departure,.9)/max(radius,1e-30)<=.05),
            'both_fundamental_windings_in90percent_slices':bool(winding_fraction>=.9),
            'phase_orientation_positive_in95percent':bool(orientation_fraction>=.95),
            'phase_grid_coverage80percent':bool(coverage>=.8)}
        results[label]={'gates':gates,'departure_over_radius':describe(departure/max(radius,1e-30)),
            'winding_identity_fraction':winding_fraction,'orientation_positive_fraction':orientation_fraction,'phase_grid_coverage':coverage}
    return {'radius':float(radius),'max_edge_over_radius':float(maxedge),'nonlocal_p10_over_radius':nonlocal_p10,'centering':results}


def frozen_protocol(args):
    return {'question':'Does each native grid module support two independent recurrent phase directions with preferential recovery transverse to BOTH?',
        'cohort':'All4modules, Grid/PGC seeds1/2, context0,24x24 phases; same synthetic pickup and native predictive offsets as one-dimensional screen',
        'grid_side':args.side,'regimes':list(one.REGIMES),'anchors_updates':list(ANCHORS),'probe_horizons_updates':list(HORIZONS),
        'elicitation':'15 actual recorded updates; vary only one goalcode module across both primitive periods; all future inputs common',
        'tangents':'Exact input-phase derivatives, exact original-LSTM JVPs for BOTH columns throughout elicitation and common-input evolution',
        'normals':'3 fixed-seed Gaussian vectors,centroidradial,h-only,c-only;each projectedoffBOTHfullstatephase tangents;raw andfixedinitialh/cRMSmetrics',
        'observables':'Full state, all256 futurehidden outputs, or all6 centeredtrainedaction scores at offsets0/5/10/15; outputobservablesfixed, nofit',
        'geometry':'Nearest full-dimensional periodic triangulated surface; localrank2; globalnonlocalseparation; preserved2fundamentalwindings/orientation; commonmean-centered andnoncentered',
        'nomination':'Allgeometry/differentialgatespass;initialtransverseobservablekick≥1e-4radius for80%;bothphasegainsmedian≥.5;normalgainandnormal/minphase≤.5 foreachinformative normal direction, notonlypooled. InitialTOTALsilent-laterTOTALvisiblekickrejects. Initial/finaltransversesilentdirection explicitlyreported andneverrescuespooledvisibilityfailure.',
        'candidate_validation':'Anycandidate remainsprovisional until finite±2%kicks and strongest-normal search at8fixedphase locations; refine24to48phase geometry beforemechanisticclaim',
        'controls':'StableCliffordtorus, torus+sharedclock in fixedobservable, shrinkingtorus, collapsedrank1; sourceexactperiod andcompleteelicitationFD checks',
        'source_sha256':{p:one.sha(ROOT/p) for p in ['scripts/lstm_module_phase_torus.py','scripts/lstm_module_phase_ring.py','scripts/lstm_endogenous_curve_recovery.py','code/codes.py','code/nets.py','tests/test_lstm_module_phase_torus.py','tests/test_lstm_module_phase_torus_review.py']},
        'parent_protocol_sha256':one.sha(ROOT/'results/lstm_module_phase_ring/protocol.json')}


@torch.inference_mode()
def elicit(net,inputs,initial,codes,code_derivatives=None):
    state=initial[None].repeat(len(codes),1)
    directions=torch.zeros(len(codes),512,2,dtype=torch.float64)
    if code_derivatives is not None:
        derivatives=[]
        for axis in range(2):
            dx=torch.zeros(len(codes),1287,dtype=torch.float64)
            dx[:,768:1280]=torch.tensor(code_derivatives[:,:,axis])
            derivatives.append(torch.nn.functional.linear(dx,net.lstm.weight_ih))
    for native in inputs:
        x=native.repeat(len(codes),1);x[:,768:1280]=torch.tensor(codes)
        drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
        if code_derivatives is None:state=advance(net.lstm,drive,state)
        else:
            next_state,first=advance(net.lstm,drive,state,directions[:,:,:1],derivatives[0])
            _,second=advance(net.lstm,drive,state,directions[:,:,1:],derivatives[1])
            state=next_state;directions=torch.cat((first,second),axis=2)
    return state,directions


def normal_directions(state,tangents,scale,seed):
    x=state/scale
    t=tangents/scale[None,:,None]
    rng=np.random.default_rng(seed)
    directions=rng.normal(size=(len(state),state.shape[1],6))
    directions[:,:,3]=x-x.mean(0)
    directions[:,256:,4]=0.;directions[:,:256,5]=0.
    directions[:,:,:4],_=normal_component(directions[:,:,:4],t)
    directions[:,:256,4:5],_=normal_component(directions[:,:256,4:5],t[:,:256])
    directions[:,256:,5:6],_=normal_component(directions[:,256:,5:6],t[:,256:])
    lengths=np.linalg.norm(directions,axis=1)
    directions/=np.maximum(lengths[:,None,:],1e-30)
    radius=np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    return directions*scale[None,:,None],.02*radius


def slice_cross_phase_fraction(state,tangents,scale,n):
    z=(state/scale).reshape(n,n,-1)
    t=(tangents/scale[None,:,None]).reshape(n,n,-1,2)
    results=[]
    for axis in range(2):
        points=z[:,0] if axis==0 else z[0,:]
        tangent=t[:,0] if axis==0 else t[0,:]
        primary=tangent[:,:,axis:axis+1]
        other=tangent[:,:,1-axis:2-axis]
        radial=(points-points.mean(0))[:,:,None]
        radial,_=normal_component(radial,primary)
        other,_=normal_component(other,primary)
        numerator=np.sum(radial*other,axis=1)**2
        denominator=np.sum(radial**2,axis=1)*np.sum(other**2,axis=1)
        results.append(describe(np.clip(numerator/np.maximum(denominator,1e-30),0,1)))
    return results


@torch.inference_mode()
def probe(net,drive,state,tangents,scales,n,seed):
    directions=[tangents];amplitudes=[]
    for metric,scale in enumerate(scales):
        normal,amplitude=normal_directions(state,tangents,scale,seed)
        directions.append(normal);amplitudes.append(amplitude)
    z=torch.tensor(state,dtype=torch.float64)
    derivatives=torch.tensor(np.concatenate(directions,axis=2),dtype=torch.float64)
    observed={};needed={t+h for h in (0,*HORIZONS) for t in OFFSETS}
    for step in range(max(HORIZONS)+max(OFFSETS)+1):
        if step in needed:
            logits=net.pi(z[:,:256]);logits-=logits.mean(1,keepdim=True)
            dl=torch.einsum('od,bdk->bok',net.pi.weight,derivatives[:,:256])
            dl-=dl.mean(1,keepdim=True)
            observed[step]={'h':z[:,:256].numpy().copy(),'dh':derivatives[:,:256].numpy().copy(),
                            'scores':logits.numpy().copy(),'dscores':dl.numpy().copy()}
            if step in (0,*HORIZONS):
                observed[step]['state']=z.numpy().copy();observed[step]['dstate']=derivatives.numpy().copy()
        if step<max(HORIZONS)+max(OFFSETS):z,derivatives=advance(net.lstm,drive,z,derivatives)
    rows=[];geometry_cache={}
    for metric,scale in enumerate(scales):
        cross_phase=slice_cross_phase_fraction(state,tangents,scale,n)
        for observable in ('full_state','all_hidden_future','all_actor_scores_future'):
            def get(offset):
                if observable=='full_state':
                    return observed[offset]['state']/scale,observed[offset]['dstate']/scale[None,:,None]
                key='h' if observable=='all_hidden_future' else 'scores'
                return (np.concatenate([observed[offset+t][key] for t in OFFSETS],axis=1),
                        np.concatenate([observed[offset+t]['d'+key] for t in OFFSETS],axis=1))
            initial,di=get(0)
            for horizon in HORIZONS:
                final,df=get(horizon)
                cachekey=(observable,horizon,metric if observable=='full_state' else 0)
                if cachekey not in geometry_cache:geometry_cache[cachekey]=geometry_metrics(initial,final,n)
                geometric=geometry_cache[cachekey]
                selected=slice(2+6*metric,8+6*metric)
                recovery=differential_metrics(initial,final,di[:,:,:2],df[:,:,:2],di[:,:,selected],df[:,:,selected],amplitudes[metric])
                candidates={centering:all(g['gates'].values()) and all(recovery['gates'].values())
                    for centering,g in geometric['centering'].items()}
                rows.append({'metric':'raw' if metric==0 else 'relative_rms','observable':observable,
                    'horizon_updates':horizon,'geometry':geometric,'recovery':recovery,
                    'old_slice_radial_second_phase_fraction':cross_phase,'candidates':candidates})
    return rows


@torch.inference_mode()
def run_model(condition,seed,args):
    began=time.monotonic();last=began
    cfg,checkpoint,net,pop=one.base.load_model(condition,seed,'cpu');net.double()
    source=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'
    history=np.load(source,allow_pickle=False)
    inputs=torch.tensor(history['history_x'][:15],dtype=torch.float64)
    initial=torch.tensor(np.r_[history['history_h_before'][0],history['history_c_before'][0]],dtype=torch.float64)
    pickup=history['history_pos'][14].astype(float)+[.46,0.];arrival=np.array([1.,0.])
    phases=np.stack(np.meshgrid(np.arange(args.side)/args.side,np.arange(args.side)/args.side,indexing='ij'),axis=-1).reshape(-1,2)
    count=len(phases)
    evaluated=[torus_code(pop,pickup,arrival,module,phases) for module in range(4)]
    codes=np.concatenate([v[0] for v in evaluated]);code_derivatives=np.concatenate([v[1] for v in evaluated])
    state,tangents=elicit(net,inputs,initial,codes,code_derivatives)
    fd_records=[]
    indices=np.linspace(0,count-1,8).astype(int)
    for module in range(4):
        for axis in range(2):
            delta=np.eye(2)[axis]*1e-5
            plus=torus_code(pop,pickup,arrival,module,phases[indices]+delta)[0]
            minus=torus_code(pop,pickup,arrival,module,phases[indices]-delta)[0]
            zp,_=elicit(net,inputs,initial,plus);zm,_=elicit(net,inputs,initial,minus)
            fd=(zp-zm)/(2e-5);exact=tangents[module*count+indices,:,axis]
            error=float((fd-exact).norm()/exact.norm().clamp_min(1e-30))
            assert error<1e-5,error
            fd_records.append({'module':module,'axis':axis,'relative_error':error})
    families=state.reshape(4,count,512)
    scale_values=np.stack([families[:,:,:256].square().mean((1,2)).sqrt().numpy(),
                           families[:,:,256:].square().mean((1,2)).sqrt().numpy()],axis=1).clip(1e-8)
    common_drives=[]
    for mode in one.REGIMES:
        x=inputs[-1:].clone()
        if mode=='all_goal_withheld':x[:,768:1280]=0.
        common_drives.append(torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih))
    drive=torch.cat([d.expand(4*count,-1) for d in common_drives])
    state=state.repeat(2,1);tangents=tangents.repeat(2,1,1)
    saved={};saved_tangents={}
    for step in range(max(ANCHORS)+1):
        if step in (0,*ANCHORS):
            saved[step]=state.reshape(2,4,count,512).numpy().copy()
            saved_tangents[step]=tangents.reshape(2,4,count,512,2).numpy().copy()
        if step<max(ANCHORS):state,tangents=advance(net.lstm,drive,state,tangents)
        if time.monotonic()-last>25:
            print(f'{condition} seed{seed}: two-phase common evolution {step}/{max(ANCHORS)}',flush=True);last=time.monotonic()
    output=args.out/f'{condition}_s{seed}';output.mkdir()
    np.savez_compressed(output/'states_and_tangents.npz',**{f'states_{t}':v for t,v in saved.items()},
                        **{f'tangents_{t}':v for t,v in saved_tangents.items()},phase=phases,scales=scale_values)
    records=[]
    for regime,mode in enumerate(one.REGIMES):
        for module in range(4):
            scales=[np.ones(512),np.repeat(scale_values[module],256)]
            for anchor in ANCHORS:
                rows=probe(net,common_drives[regime],saved[anchor][regime,module],saved_tangents[anchor][regime,module],
                           scales,args.side,74000+seed*1000+regime*100+module*10+anchor)
                for row in rows:row.update(condition=condition,seed=seed,module=module,regime=mode,anchor_updates=anchor)
                records.extend(rows)
                if time.monotonic()-last>25:
                    print(f'{condition} seed{seed}: {len(records)}/192 torus observable/metric/horizon cases',flush=True);last=time.monotonic()
    result={'condition':condition,'seed':seed,'seconds':time.monotonic()-began,'elicitation_FD_checks':fd_records,
            'checkpoint_sha256':one.sha(checkpoint),'history_sha256':one.sha(source),'records':records}
    one.save(output/'result.json',result)
    print(f'{condition} seed{seed}: {sum(r["candidates"]["centered"] for r in records)} centered / {sum(r["candidates"]["noncentered"] for r in records)} provisional torus nominations; {result["seconds"]:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/lstm_module_phase_torus')
    parser.add_argument('--side',type=int,default=24);parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    protocol=frozen_protocol(args);manifest=args.out/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==protocol,'Frozen source changed; choose fresh output directory'
    else:one.save(manifest,protocol)
    if args.freeze_only:return
    assert not (args.out/'summary.json').exists(),'Do not overwrite completed outcomes'
    results=[run_model(c,s,args) for c in ('gc100','pgc33') for s in (1,2)]
    rows=[r for m in results for r in m['records']]
    one.save(args.out/'summary.json',{'cases':len(rows),'centered_nominations':sum(r['candidates']['centered'] for r in rows),
        'noncentered_nominations':sum(r['candidates']['noncentered'] for r in rows),'results':results})


if __name__=='__main__':main()
