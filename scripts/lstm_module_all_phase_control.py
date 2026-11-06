"""Local known-phase nuisance control; this is not a global eight-torus fit.

The same perturbations are analyzed after removing the selected module's two
phase tangents or all eight analytically defined module-phase tangents. Large
coefficients and initially unobservable responses remain explicit.
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
import lstm_module_phase_torus as torus
from lstm_endogenous_curve_recovery import advance,describe

PARENT=ROOT/'results/lstm_module_phase_torus'


def all_phase_codes(pop,pickup,arrival,phases):
    phases=np.asarray(phases).reshape(-1,4,2)
    code=np.zeros((len(phases),512));derivatives=np.zeros((len(phases),512,8))
    for module in range(4):
        values,gradient,_=torus.torus_code(pop,pickup,arrival,module,phases[:,module])
        mask=pop.module==module
        code[:,mask]=values[:,mask]
        derivatives[:,:,2*module:2*module+2]=gradient
    return code,derivatives


@torch.inference_mode()
def elicit(net,inputs,initial,codes,gradients):
    state=initial[None].repeat(len(codes),1)
    directions=torch.zeros(len(codes),512,8,dtype=torch.float64)
    differential=[]
    for k in range(8):
        dx=torch.zeros(len(codes),1287,dtype=torch.float64)
        dx[:,768:1280]=torch.tensor(gradients[:,:,k])
        differential.append(torch.nn.functional.linear(dx,net.lstm.weight_ih))
    for native in inputs:
        x=native.repeat(len(codes),1);x[:,768:1280]=torch.tensor(codes)
        drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
        pieces=[]
        for k in range(8):
            next_state,piece=advance(net.lstm,drive,state,directions[:,:,k:k+1],differential[k])
            pieces.append(piece)
        state=next_state;directions=torch.cat(pieces,axis=2)
    return state,directions


def compare_known_phases(initial,final,phase0,phase1,normal0,normal1,module,amplitude,reference_radius=None):
    chosen=slice(2*module,2*module+2)
    other=[k for k in range(8) if k not in range(2*module,2*module+2)]
    radius=np.sqrt(np.mean(np.sum((initial-initial.mean(0))**2,axis=1))) if reference_radius is None else reference_radius
    floor=1e-4*max(radius,1e-30)
    output={}
    for name,values,tangent in [('initial',normal0,phase0),('final',normal1,phase1)]:
        residual_two,selected_coefficients=torus.normal_component(values,tangent[:,:,chosen])
        extra,selected_compensation=torus.normal_component(tangent[:,:,other],tangent[:,:,chosen])
        residual_eight,coefficients=torus.normal_component(residual_two,extra)
        n2=np.linalg.norm(residual_two,axis=1);n8=np.linalg.norm(residual_eight,axis=1)
        explained=np.clip(1-n8**2/np.maximum(n2**2,1e-30),0.,1.)
        physical_coefficients=np.zeros((len(values),8,values.shape[-1]))
        physical_coefficients[:,other]=coefficients
        physical_coefficients[:,chosen]=-selected_compensation@coefficients
        total_coefficients=physical_coefficients.copy()
        total_coefficients[:,chosen]+=selected_coefficients
        displacement=amplitude*np.max(np.abs(physical_coefficients),axis=1)
        total_displacement=amplitude*np.max(np.abs(total_coefficients),axis=1)
        singular=np.linalg.svd(tangent,compute_uv=False)
        rank=np.sum(singular>np.maximum(singular[:,:1]*1e-8,1e-12),axis=1)
        selected_singular=np.linalg.svd(tangent[:,:,chosen],compute_uv=False)
        extra_singular=np.linalg.svd(extra,compute_uv=False)
        selected_rank=np.sum(selected_singular>selected_singular[:,:1]*1e-6,axis=1)
        extra_rank=np.sum(extra_singular>extra_singular[:,:1]*1e-6,axis=1)
        output[name]={'normal_two':n2,'normal_eight':n8,'explained':explained,'phase_displacement':displacement,
                      'phase_coefficients':physical_coefficients,'total_phase_coefficients':total_coefficients,
                      'total_phase_displacement':total_displacement,'rank':rank,
                      'selected_projection_rank':selected_rank,'extra_projection_rank':extra_rank,
                      'total_response':np.linalg.norm(values,axis=1)}
    a,b=output['initial'],output['final']
    valid2=amplitude*a['normal_two']>=floor;valid8=amplitude*a['normal_eight']>=floor
    local=(a['phase_displacement']<=.1)&(b['phase_displacement']<=.1)
    total_local=(a['total_phase_displacement']<=.1)&(b['total_phase_displacement']<=.1)
    ratio2=b['normal_two']/np.maximum(a['normal_two'],1e-30)
    ratio8=b['normal_eight']/np.maximum(a['normal_eight'],1e-30)
    new_total=(amplitude*a['total_response']<floor)&(amplitude*b['total_response']>=floor)
    new_normal8=(amplitude*a['normal_eight']<floor)&(amplitude*b['normal_eight']>=floor)
    per=[]
    for k in range(normal0.shape[-1]):
        local_valid=valid2[:,k]&local[:,k]
        per.append({'direction':k,'two_phase_initial_visible_fraction':float(valid2[:,k].mean()),
            'eight_phase_initial_visible_fraction':float(valid8[:,k].mean()),
            'two_phase_normal_gain_median':float(np.median(ratio2[valid2[:,k],k])) if valid2[:,k].any() else None,
            'eight_phase_normal_gain_median':float(np.median(ratio8[valid8[:,k],k])) if valid8[:,k].any() else None,
            'additional_phase_explained_fraction_median':float(np.median(b['explained'][valid2[:,k],k])) if valid2[:,k].any() else None,
            'coefficient_locality_pass_fraction':float(local[:,k].mean()),
            'total_fit_locality_pass_fraction':float(total_local[:,k].mean()),
            'locally_explained_fraction_median':float(np.median(b['explained'][local_valid,k])) if local_valid.any() else None,
            'max_required_orthogonalized_eight_phase_displacement':float(max(a['phase_displacement'][:,k].max(),b['phase_displacement'][:,k].max())),
            'max_required_total_fit_phase_displacement':float(max(a['total_phase_displacement'][:,k].max(),b['total_phase_displacement'][:,k].max())),
            'initially_silent_later_total_visible_fraction':float(new_total[:,k].mean()),
            'initially_eight_normal_silent_later_visible_fraction':float(new_normal8[:,k].mean())})
    return {'observable_radius_RMS':float(radius/np.sqrt(initial.shape[1])),
        'initial_known_phase_rank':a['rank'].tolist(),'final_known_phase_rank':b['rank'].tolist(),
        'initial_effective_selected_projection_rank':a['selected_projection_rank'].tolist(),
        'initial_effective_extra_projection_rank':a['extra_projection_rank'].tolist(),
        'final_effective_selected_projection_rank':b['selected_projection_rank'].tolist(),
        'final_effective_extra_projection_rank':b['extra_projection_rank'].tolist(),
        'normal_gain_two_phase':ratio2.tolist(),'normal_gain_eight_phase':ratio8.tolist(),
        'extra_phase_explained_fraction_initial':a['explained'].tolist(),
        'extra_phase_explained_fraction_final':b['explained'].tolist(),
        'extra_phase_displacement_initial':a['phase_displacement'].tolist(),
        'extra_phase_displacement_final':b['phase_displacement'].tolist(),
        'orthogonalized_eight_phase_coefficients_initial':(amplitude*a['phase_coefficients']).tolist(),
        'orthogonalized_eight_phase_coefficients_final':(amplitude*b['phase_coefficients']).tolist(),
        'total_fit_eight_phase_coefficients_initial':(amplitude*a['total_phase_coefficients']).tolist(),
        'total_fit_eight_phase_coefficients_final':(amplitude*b['total_phase_coefficients']).tolist(),
        'per_direction':per,'interpretation':'Local projection onto explicit known input-phase derivatives only; no global T8 or attractor nomination'}


def cell_centroid_normal(full_surface,selected,tangents,scale):
    radial=np.zeros_like(selected)
    radial[:,256:]=(selected[:,256:]-full_surface[:,256:].mean(0))/scale[256:]
    projected,_=torus.normal_component(radial[:,256:,None],tangents[:,256:]/scale[None,256:,None])
    radial[:,256:]=projected[:,:,0]
    radial/=np.maximum(np.linalg.norm(radial,axis=1,keepdims=True),1e-30)
    return radial*scale


@torch.inference_mode()
def probe(net,drive,state,phase_derivatives,full_surface,full_two_derivatives,indices,scales,module,seed,parent_radii):
    all_normals=[];amplitudes=[]
    for scale in scales:
        existing,amplitude=torus.normal_directions(full_surface,full_two_derivatives,scale,seed)
        special=cell_centroid_normal(full_surface,state,phase_derivatives[:,:,2*module:2*module+2],scale)
        all_normals.append(np.concatenate((existing[indices],special[:,:,None]),axis=2));amplitudes.append(amplitude)
    z=torch.tensor(state,dtype=torch.float64)
    derivatives=torch.tensor(np.concatenate((phase_derivatives,*all_normals),axis=2),dtype=torch.float64)
    needed={t+h for h in (0,*torus.HORIZONS) for t in torus.OFFSETS};observed={}
    for step in range(max(torus.HORIZONS)+max(torus.OFFSETS)+1):
        if step in needed:
            logits=net.pi(z[:,:256]);logits-=logits.mean(1,keepdim=True)
            dl=torch.einsum('od,bdk->bok',net.pi.weight,derivatives[:,:256]);dl-=dl.mean(1,keepdim=True)
            observed[step]={'h':z[:,:256].numpy().copy(),'dh':derivatives[:,:256].numpy().copy(),
                            'scores':logits.numpy().copy(),'dscores':dl.numpy().copy()}
            if step in (0,*torus.HORIZONS):
                observed[step]['state']=z.numpy().copy();observed[step]['dstate']=derivatives.numpy().copy()
        if step<max(torus.HORIZONS)+max(torus.OFFSETS):z,derivatives=advance(net.lstm,drive,z,derivatives)
    records=[]
    for metric,scale in enumerate(scales):
        for observable in ('full_state','all_hidden_future','all_actor_scores_future'):
            def values(offset):
                if observable=='full_state':return observed[offset]['state']/scale,observed[offset]['dstate']/scale[None,:,None]
                key='h' if observable=='all_hidden_future' else 'scores'
                return (np.concatenate([observed[offset+t][key] for t in torus.OFFSETS],axis=1),
                        np.concatenate([observed[offset+t]['d'+key] for t in torus.OFFSETS],axis=1))
            initial,di=values(0)
            for horizon in torus.HORIZONS:
                final,df=values(horizon);normal=slice(8+7*metric,15+7*metric)
                label='raw' if metric==0 else 'relative_rms'
                result=compare_known_phases(initial,final,di[:,:,:8],df[:,:,:8],di[:,:,normal],df[:,:,normal],module,amplitudes[metric],parent_radii[(label,observable,horizon)])
                records.append({'metric':'raw' if metric==0 else 'relative_rms','observable':observable,
                    'horizon_updates':horizon,'comparison':result})
    return records


@torch.inference_mode()
def run_model(condition,seed,args):
    began=time.monotonic()
    _,checkpoint,net,pop=torus.one.base.load_model(condition,seed,'cpu');net.double()
    native_path=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'
    native=np.load(native_path,allow_pickle=False)
    parent_path=PARENT/f'{condition}_s{seed}/states_and_tangents.npz';parent=np.load(parent_path,allow_pickle=False)
    parent_results=json.loads((parent_path.parent/'result.json').read_text())['records']
    inputs=torch.tensor(native['history_x'][:15],dtype=torch.float64)
    initial=torch.tensor(np.r_[native['history_h_before'][0],native['history_c_before'][0]],dtype=torch.float64)
    pickup=native['history_pos'][14].astype(float)+[.46,0.];arrival=np.array([1.,0.])
    n=int(round(np.sqrt(len(parent['phase']))));assert n==24
    j=np.arange(8);indices=(3*j)*n+3*((3*j)%8)
    phase_values=np.zeros((4,8,4,2))
    for module in range(4):phase_values[module,:,module]=parent['phase'][indices]
    code,gradient=all_phase_codes(pop,pickup,arrival,phase_values.reshape(-1,4,2))
    state,tangents=elicit(net,inputs,initial,code,gradient)
    finite=[]
    for axis in range(8):
        parameters=phase_values[:,axis].reshape(4,8)
        delta=np.eye(8)[axis]*1e-5
        plus=all_phase_codes(pop,pickup,arrival,parameters+delta)[0]
        minus=all_phase_codes(pop,pickup,arrival,parameters-delta)[0]
        zp,_=torus.elicit(net,inputs,initial,plus);zm,_=torus.elicit(net,inputs,initial,minus)
        fd=(zp-zm)/(2e-5);exact=tangents[np.arange(4)*8+axis,:,axis]
        error=float((fd-exact).norm()/exact.norm().clamp_min(1e-30));assert error<1e-5
        finite.append(error)
    drives=[]
    for mode in torus.one.REGIMES:
        x=inputs[-1:].clone()
        if mode=='all_goal_withheld':x[:,768:1280]=0.
        drives.append(torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih))
    drive=torch.cat([d.expand(32,-1) for d in drives]);state=state.repeat(2,1);tangents=tangents.repeat(2,1,1)
    saved={};saved_tangents={};agreement=[]
    for step in range(max(torus.ANCHORS)+1):
        if step in (0,*torus.ANCHORS):
            saved[step]=state.reshape(2,4,8,512).numpy().copy()
            saved_tangents[step]=tangents.reshape(2,4,8,512,8).numpy().copy()
            zerror=float(np.max(np.abs(saved[step]-parent[f'states_{step}'][:,:,indices])))
            derror=max(float(np.max(np.abs(saved_tangents[step][:,m,:,:,2*m:2*m+2]-parent[f'tangents_{step}'][:,m,indices]))) for m in range(4))
            assert zerror<1e-7 and derror<1e-6,(zerror,derror)
            agreement.append({'step':step,'state_max_abs_error':zerror,'selected_phase_JVP_max_abs_error':derror})
        if step<max(torus.ANCHORS):state,tangents=advance(net.lstm,drive,state,tangents)
    output=args.out/f'{condition}_s{seed}';output.mkdir()
    np.savez_compressed(output/'states_and_phase_derivatives.npz',**{f'states_{t}':v for t,v in saved.items()},
                        **{f'derivatives_{t}':v for t,v in saved_tangents.items()},selected_grid_indices=indices)
    records=[]
    for regime,mode in enumerate(torus.one.REGIMES):
        for module in range(4):
            scales=[np.ones(512),np.repeat(parent['scales'][module],256)]
            for anchor in torus.ANCHORS:
                radii={(r['metric'],r['observable'],r['horizon_updates']):r['geometry']['radius'] for r in parent_results
                       if r['module']==module and r['regime']==mode and r['anchor_updates']==anchor}
                rows=probe(net,drives[regime],saved[anchor][regime,module],saved_tangents[anchor][regime,module],
                    parent[f'states_{anchor}'][regime,module],parent[f'tangents_{anchor}'][regime,module],
                    indices,scales,module,74000+seed*1000+regime*100+module*10+anchor,radii)
                for row in rows:row.update(condition=condition,seed=seed,module=module,regime=mode,anchor_updates=anchor)
                records.extend(rows)
    result={'condition':condition,'seed':seed,'seconds':time.monotonic()-began,
        'checkpoint_sha256':torus.one.sha(checkpoint),'history_sha256':torus.one.sha(native_path),
        'parent_archive_sha256':torus.one.sha(parent_path),'elicitation_FD_relative_errors':finite,
        'parent_replay_agreement':agreement,'records':records}
    torus.one.save(output/'result.json',result)
    print(f'{condition} seed{seed}: {len(records)} local nuisance comparisons complete in{result["seconds"]:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/lstm_module_all_phase_control')
    parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    parent_summary=json.loads((PARENT/'summary.json').read_text());assert parent_summary['cases']==768
    protocol={'role':'Final local known-phase nuisance control for thisbatch; notglobalT8geometry orattractorclaim',
        'cohort':'All4models4modules2commonfutures5/30sanchors;8fixedLatin-gridpoints(modulephasej/8,3jmod8/8)',
        'cases':768,'horizons_updates':[15,30],'observable_offsets':list(torus.OFFSETS),
        'phase_directions':'All8 explicitmodulephaseinputderivatives; nonselectedmodules atzerooffsets; exactoriginalLSTMJVPs',
        'perturbations':'First6physicaldirections exactlymatchselectedpointsinparent2Dexperiment; addc-onlycentroidradial,hidenticallyunchanged,projectedwithincblockagainstselected2tangents;keepgenericc-only',
        'comparison':'Samephysicalresponses projectedoffselected2 versusselected2plusorthogonalizedadditional6phasetangents;state,hfuture,centeredactorfuture;raw/RMS',
        'coefficient_guard':'OrthogonalizedadditionalphaseexplanationqualifiedlocalonlywhenALL8physicalcoefficients(includingselectedphasecompensation)≤.1periodforactual2%kickatBOTHwindows. Fullresponsephasefitcoefficientsalso reportedseparately; nohiddenlargecompensations.',
        'projection_rank':'Effective selected/extra ranks use singular cutoff1e-6, matching Gram pseudoinverse1e-12. Separatelyreported full8 descriptive rank usesrelative1e-8andabsolute1e-12.',
        'guards':'Retainphaseobservable/numericalrank,initialtransversenontriviality,initialsilentlaterobservable,and8-phase-normalreappearance;nogainfromnearzerodenominatorcountsasrestoration',
        'source_sha256':{p:torus.one.sha(ROOT/p) for p in ['scripts/lstm_module_all_phase_control.py','tests/test_lstm_module_all_phase_control.py','scripts/lstm_module_phase_torus.py','scripts/lstm_endogenous_curve_recovery.py']},
        'parent_protocol_sha256':torus.one.sha(PARENT/'protocol.json'),'parent_summary_sha256':torus.one.sha(PARENT/'summary.json'),
        'parent_archives':{f'{c}_s{s}':torus.one.sha(PARENT/f'{c}_s{s}/states_and_tangents.npz') for c in ('gc100','pgc33') for s in (1,2)}}
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen source changed; usefreshdirectory'
    else:torus.one.save(path,protocol)
    if args.freeze_only:return
    assert not (args.out/'summary.json').exists(),'Do notoverwritecompletedoutcomes'
    results=[run_model(c,s,args) for c in ('gc100','pgc33') for s in (1,2)]
    torus.one.save(args.out/'summary.json',{'cases':sum(len(m['records']) for m in results),'results':results,
        'interpretation':'Local known-phase nuisance decomposition only; no global eight-dimensional attractor nomination'})


if __name__=='__main__':main()
