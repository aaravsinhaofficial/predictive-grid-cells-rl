"""Direction subspace of learned PGC encoder drive, with causal interventions.

Only fitting/scoring use direction labels. Every intervention receives the two
observed input vectors (PGC and same-position ordinary code) and a fixed fitted
projector. It never receives true heading or a target output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from ring_heading_model import load_checkpoint
from evaluate_ring_heading_transfer import alter_measurements


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


class DriveInterventions:
    """Frozen drive intervention; no physical-pose or heading arguments."""
    def __init__(self,model,basis,sham_bases):
        self.model=model
        self.basis=torch.as_tensor(basis,dtype=next(model.parameters()).dtype,device=next(model.parameters()).device)
        self.shams=[torch.as_tensor(q,dtype=self.basis.dtype,device=self.basis.device) for q in sham_bases]

    def transform_drives(self,native_drive,ordinary_drive):
        delta=native_drive-ordinary_drive
        component=(delta@self.basis)@self.basis.T
        residual=delta-component
        drives={'native':native_drive,'ordinary':ordinary_drive,
                'remove_direction':native_drive-component,
                'restore_direction':ordinary_drive+component,
                'residual_only':ordinary_drive+residual}
        dose={'delta_norm':delta.norm(dim=-1),'direction_norm':component.norm(dim=-1),
              'residual_norm':residual.norm(dim=-1)}
        for j,q in enumerate(self.shams):
            candidate=(delta@q)@q.T
            norm=candidate.norm(dim=-1,keepdim=True)
            fallback=q[:,0].expand_as(candidate)
            direction=torch.where(norm>1e-12,candidate/norm.clamp_min(1e-12),fallback)
            matched=direction*component.norm(dim=-1,keepdim=True)
            drives[f'sham{j}_restore']=ordinary_drive+matched
            drives[f'sham{j}_remove']=native_drive-matched
            dose[f'sham{j}_dose_error']=(matched.norm(dim=-1)-component.norm(dim=-1)).abs()
            dose[f'sham{j}_relative_dose_error']=dose[f'sham{j}_dose_error']/component.norm(dim=-1).clamp_min(1e-9)
            dose[f'sham{j}_fallback']=((norm[...,0]<=1e-12)&(component.norm(dim=-1)>0)).float()
        return drives,dose

    def paired_drives(self,native_input,ordinary_input):
        return self.transform_drives(self.model.input(native_input),self.model.input(ordinary_input))

    def step_from_drive(self,drive,state):
        return (1-self.model.config.leak)*state+self.model.config.leak*(drive+self.model.recurrent(state)).tanh()


@torch.no_grad()
def static_dataset(model,task,rng,positions,angles,offset=0.,chunk=4096):
    p=rng.uniform(-1.25,1.25,(positions,2));p[:,0]+=offset
    phase=2*np.pi*np.arange(angles)/angles+rng.uniform(-np.pi,np.pi,(positions,1))
    pos=np.repeat(p,angles,axis=0).astype(np.float32)
    phase=phase.reshape(-1).astype(np.float32)
    direction=np.column_stack((np.cos(phase),np.sin(phase))).astype(np.float32)
    deltas=[]
    for start in range(0,len(pos),chunk):
        pt=torch.from_numpy(pos[start:start+chunk]).to(task.device)
        dr=torch.from_numpy(direction[start:start+chunk]).to(task.device)
        zero=pt[:,0]*0;one=zero+1
        native=task.inputs(pt,dr,zero,zero,one,zero,'predictive')
        ordinary=task.inputs(pt,dr,zero,zero,one,zero,'grid')
        assert torch.equal(native[:,task.config.n_code:],ordinary[:,task.config.n_code:])
        deltas.append((model.input(native)-model.input(ordinary)).cpu().numpy())
    return {'position':pos,'phase':phase,'direction':direction,'delta':np.concatenate(deltas),
            'positions':positions,'angles':angles,'offset':offset}


def design(phase,harmonics):
    return np.column_stack([np.ones(len(phase))]+[f(k*phase) for k in range(1,harmonics+1) for f in (np.cos,np.sin)])


def fit_harmonics(data,harmonics):
    x=design(data['phase'].astype(float),harmonics);y=data['delta'].astype(float)
    coefficients=np.linalg.lstsq(x,y,rcond=None)[0]
    u,s,vh=np.linalg.svd(coefficients[1:],full_matrices=False)
    rank=min(2*harmonics,y.shape[1]);basis=vh[:rank].T
    assert np.allclose(basis.T@basis,np.eye(rank),atol=1e-12)
    # A separate direction readout tests linear accessibility, not fit quality.
    decoders={}
    for name,features in [('full',y),('projected',y@basis)]:
        z=np.column_stack((np.ones(len(features)),features))
        penalty=np.eye(z.shape[1])*1e-3;penalty[0,0]=0
        decoders[name]=np.linalg.solve(z.T@z+penalty,z.T@data['direction'])
    return {'harmonics':harmonics,'coefficients':coefficients,'basis':basis,
            'coefficient_singular_values':s,'decoders':decoders}


def fit_metrics(data,fit):
    y=data['delta'].astype(float);pred=design(data['phase'].astype(float),fit['harmonics'])@fit['coefficients']
    centered=y-y.mean(0);within=y.reshape(data['positions'],data['angles'],-1)
    within=(within-within.mean(1,keepdims=True)).reshape(y.shape)
    harmonic=pred-fit['coefficients'][0]
    projected=y@fit['basis'];center_projected=centered@fit['basis']
    result=dict(harmonic_regression_r2=float(1-np.sum((y-pred)**2)/np.sum(centered**2)),
        within_position_harmonic_r2=float(1-np.sum((within-harmonic)**2)/np.sum(within**2)),
        projected_raw_energy_fraction=float(np.sum(projected**2)/np.sum(y**2)),
        projected_variance_fraction=float(np.sum(center_projected**2)/np.sum(centered**2)),
        samples=len(y),independent_positions=data['positions'],position_offset_x_m=data['offset'])
    for name,features in [('full',y),('projected',projected)]:
        output=np.column_stack((np.ones(len(features)),features))@fit['decoders'][name]
        angle=np.arctan2(output[:,1],output[:,0])
        result[name+'_direction_mae_deg']=float(np.rad2deg(np.abs(wrap(angle-data['phase']))).mean())
        result[name+'_decoded_amplitude_median']=float(np.median(np.linalg.norm(output,axis=1)))
    return result


def make_shams(basis,seed,count=3):
    rng=np.random.default_rng(seed);result=[]
    for j in range(count):
        q=rng.normal(size=basis.shape);q-=basis@(basis.T@q);q=np.linalg.qr(q)[0]
        assert np.max(np.abs(basis.T@q))<1e-12
        result.append(q)
    return result


def state_radius(x):return float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))))


@torch.no_grad()
def anchoring(model,task,intervention,args):
    n=args.n_phase
    assert n%36==0
    xy=np.array([[-.8,-.8],[-.8,.8],[.8,-.8],[.8,.8],[0.,0.],[10.,10.]],np.float32)
    pos=torch.from_numpy(np.repeat(xy,n,axis=0)).to(task.device)
    phase=torch.arange(n,device=task.device)*2*math.pi/n;phase=phase.repeat(len(xy))
    direction=torch.stack((phase.cos(),phase.sin()),-1);zero=phase*0;one=zero+1
    h=torch.zeros(len(pos),model.hidden_size,device=task.device)
    cue=task.inputs(pos,direction,zero,zero,one,one,'predictive')
    for _ in range(model.config.initial_heading_steps):h=model.step(cue,h)
    blank=torch.zeros(len(pos),model.input_size,device=task.device)
    for _ in range(40):h=model.step(blank,h)
    curve=h.reshape(len(xy),n,-1)
    initial=torch.cat((h,curve.roll(-n//36,1).reshape_as(h),curve.roll(n//36,1).reshape_as(h)))
    xp=task.inputs(pos,direction,zero,zero,one,zero,'predictive')
    xg=task.inputs(pos,direction,zero,zero,one,zero,'grid')
    drives,dose=intervention.paired_drives(xp,xg)
    drives['blank']=model.input(blank)
    variants=list(drives);drive=torch.stack([drives[v].repeat(3,1) for v in variants]).reshape(-1,model.hidden_size)
    state=initial.repeat(len(variants),1);b=len(pos)
    initial_out=model.readout(initial).reshape(3,len(xy),n,2)
    initial_angles=torch.atan2(initial_out[...,1],initial_out[...,0]).cpu().numpy()
    initial_phase_error=wrap(initial_angles[1:]-initial_angles[0])
    initial_distance=(initial[b:].reshape(2,b,-1)-initial[:b]).norm(dim=-1).cpu().numpy().reshape(2,len(xy),n)
    rows=[];archive={'initial_curve':h.cpu().numpy(),'physical_heading':phase.cpu().numpy(),'positions':pos.cpu().numpy()}
    for t in range(args.anchor_steps+1):
        if t in (0,1,5,10,20,40,60,100,200,args.anchor_steps):
            allstate=state.reshape(len(variants),3,len(xy),n,-1)
            output=model.readout(allstate);angles=torch.atan2(output[...,1],output[...,0]).cpu().numpy()
            points=allstate.cpu().numpy()
            for vi,v in enumerate(variants):
                for context in range(len(xy)):
                    aa=angles[vi,:,context];hh=points[vi,:,context]
                    phase_shift=wrap(aa[1:]-aa[0]);distance=np.linalg.norm(hh[1:]-hh[0],axis=-1)
                    initial_abs=np.abs(initial_phase_error[:,context])
                    valid=initial_abs>1e-5
                    local_gain=np.abs(phase_shift[valid])/initial_abs[valid]
                    ordered=np.sort(np.mod(aa[0],2*np.pi));gap=np.diff(np.r_[ordered,ordered[0]+2*np.pi]).max()
                    rows.append(dict(variant=v,context=context,position=xy[context].tolist(),step=t,seconds=t*model.config.dt,
                        baseline_mae_deg=float(np.rad2deg(np.abs(wrap(aa[0]-phase[:n].cpu().numpy()))).mean()),
                        phase_error_retained=float(np.mean(np.abs(phase_shift))/np.mean(np.abs(initial_phase_error[:,context]))),
                        phase_local_gain_median=float(np.median(local_gain)),
                        phase_local_gain_q10=float(np.quantile(local_gain,.1)),
                        phase_local_gain_q90=float(np.quantile(local_gain,.9)),
                        phase_local_gain_valid_fraction=float(valid.mean()),
                        full_state_error_retained=float(distance.mean()/initial_distance[:,context].mean()),
                        circle_coverage=float(1-gap/(2*np.pi)),
                        state_radius_retained=state_radius(hh[0])/state_radius(curve[context].cpu().numpy())))
        if t<args.anchor_steps:state=intervention.step_from_drive(drive,state)
    final=state.reshape(len(variants),3,len(xy),n,-1).cpu().numpy()
    for i,v in enumerate(variants):archive[v+'_final_state']=final[i]
    np.savez_compressed(args.out/'anchor_states.npz',**archive)
    return dict(rows=rows,positions=xy.tolist(),phase_perturbation_degrees=10.,
                full_state_metrics=True,dose={k:float(x.mean()) for k,x in dose.items()})


@torch.no_grad()
def forced_trajectories(model,task,intervention,args):
    generator=torch.Generator(device=task.device).manual_seed(args.evaluation_seed)
    base=task.realization(args.length,args.trajectories,generator)
    results={};archives={};checks={}
    for scenario,offset in [('same_region',0.),('new_places',10.)]:
        data=alter_measurements(base,offset,1.)
        xp,_,_=task.batch(args.length,args.trajectories,generator,'predictive',data)
        xg,_,_=task.batch(args.length,args.trajectories,generator,'grid',data)
        drives,dose=intervention.paired_drives(xp,xg)
        for key in drives:drives[key][:task.config.initial_heading_steps]=drives['native'][:task.config.initial_heading_steps]
        variants=list(drives);b=args.trajectories;h=torch.zeros(len(variants)*b,model.hidden_size,device=task.device)
        states=[]
        for t in range(args.length):
            drive=torch.cat([drives[v][t] for v in variants]);h=intervention.step_from_drive(drive,h)
            states.append(h.reshape(len(variants),b,-1))
        states=torch.stack(states);output=model.readout(states)
        angle=torch.atan2(output[...,1],output[...,0]);error=torch.atan2(torch.sin(angle-data['heading'][:,None]),torch.cos(angle-data['heading'][:,None])).abs()*180/math.pi
        ordinary_reference=xg.clone();ordinary_reference[:task.config.initial_heading_steps]=xp[:task.config.initial_heading_steps]
        _,native_reference=model(xp,return_states=True);_,plain_reference=model(ordinary_reference,return_states=True)
        validation=dict(native_full_state_max_error=float((states[:,variants.index('native')]-native_reference).abs().max()),
                        ordinary_full_state_max_error=float((states[:,variants.index('ordinary')]-plain_reference).abs().max()),
                        removal_residual_identity_max_error=float((states[:,variants.index('remove_direction')]-states[:,variants.index('residual_only')]).abs().max()),
                        direction_dose_on_blank_max=float(dose['direction_norm'][data['visible']==0].max()))
        assert validation['native_full_state_max_error']<1e-5 and validation['ordinary_full_state_max_error']<1e-5,validation
        assert validation['removal_residual_identity_max_error']<1e-5,validation
        assert validation['direction_dose_on_blank_max']==0,validation
        checks[scenario]=validation;results[scenario]={}
        eligible=torch.arange(args.length,device=task.device)[:,None]>=8
        for vi,v in enumerate(variants):
            err=error[:,vi]
            masks={'all':eligible.expand_as(err),'visible':eligible&(data['visible']>0),'blank':eligible&(data['visible']==0)}
            results[scenario][v]={key+'_mae_deg':float(err[mask].mean()) for key,mask in masks.items()}
            results[scenario][v]['final_mae_deg']=float(err[-1].mean())
            results[scenario][v]['state_rms_from_native']=float((states[8:,vi]-states[8:,variants.index('native')]).square().mean().sqrt())
            archives[scenario+'_'+v+'_episode_mae']=err[8:].mean(0).cpu().numpy()
            archives[scenario+'_'+v+'_error']=err.cpu().numpy()
        results[scenario]['dose']={k:float(x.mean()) for k,x in dose.items()}
        print(scenario,{v:round(results[scenario][v]['all_mae_deg'],3) for v in variants},flush=True)
    np.savez_compressed(args.out/'forced_trajectory_errors.npz',**archives)
    return dict(scenarios=results,validation=checks,length=args.length,trajectories=args.trajectories,
                common_native_initialization_steps=task.config.initial_heading_steps)


def plot_results(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=['native','ordinary','remove_direction','restore_direction']
    colors=['#D87532','#9299A1','#7761A8','#3379B5']
    labels=['Native','Ordinary','Remove direction','Restore direction']
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for v,c,label in zip(variants,colors,labels):
        rows=[r for r in result['anchoring']['rows'] if r['variant']==v and r['context']<5]
        times=sorted(set(r['step'] for r in rows))
        for ax,metric in zip(axes[:2],['phase_local_gain_median','circle_coverage']):
            ax.plot(np.array(times)*result['config']['dt'],[np.mean([r[metric] for r in rows if r['step']==t]) for t in times],color=c,label=label)
    axes[0].set(title='Local phase correction',xlabel='Seconds',ylabel='Median phase error / initial error')
    axes[1].set(title='Preserving distinct headings',xlabel='Seconds',ylabel='Circle coverage')
    values=[result['forced']['scenarios']['new_places'][v]['all_mae_deg'] for v in variants]
    axes[2].bar(np.arange(4),values,color=colors);axes[2].set(title='Held-out places',ylabel='Heading error (degrees)',xticks=np.arange(4),xticklabels=['Native','Ordinary','Remove','Restore'])
    axes[2].tick_params(axis='x',labelrotation=30)
    axes[0].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Frozen learned encoder and RNN · Rank-2 direction-drive projection · One-checkpoint exploratory analysis',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'direction_subspace.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/ring_heading/encoder_pilot/predictive_s1006.pt')
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_heading/direction_subspace/predictive_s1006')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--fit-positions',type=int,default=1024);p.add_argument('--test-positions',type=int,default=256)
    p.add_argument('--angles',type=int,default=32);p.add_argument('--fit-seed',type=int,default=517913)
    p.add_argument('--evaluation-seed',type=int,default=931731)
    p.add_argument('--n-phase',type=int,default=72);p.add_argument('--anchor-steps',type=int,default=200)
    p.add_argument('--length',type=int,default=256);p.add_argument('--trajectories',type=int,default=256)
    args=p.parse_args();torch.set_num_threads(args.threads);args.out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();model,task,ckpt=load_checkpoint(args.checkpoint,args.device);model.requires_grad_(False)
    assert ckpt['condition']=='predictive'
    protocol=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),source_sha256=sha(__file__),
        fit_positions=args.fit_positions,test_positions=args.test_positions,angles=args.angles,fit_seed=args.fit_seed,
        regression='Delta encoder drive on intercept,cos(theta),sin(theta), globally across independent positions. P is orthogonal projector onto the two direction coefficient vectors; intercept excluded.',
        labels_used_only_for_fit_and_score=True,inference_intervention_arguments=['native_drive','ordinary_drive'],
        shams='Three fixed rank2 subspaces orthogonal to learned direction subspace, each scaled to exact per-input projected-component norm; both deletion and restoration tested.',
        primary_subspace='First harmonic,rank2',secondary_subspace='Harmonics1-3,rank6 fit-only diagnostic; no outcome-based subspace selection.',
        original_weights_frozen=True,evaluation_seed=args.evaluation_seed,checkpoint_training_steps=ckpt['steps_completed'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.out/'analysis_source.py').write_text(Path(__file__).read_text())
    rng=np.random.default_rng(args.fit_seed)
    train=static_dataset(model,task,rng,args.fit_positions,args.angles)
    tests={'train':train,'same_region_test':static_dataset(model,task,rng,args.test_positions,args.angles),
           'new_places_test':static_dataset(model,task,rng,args.test_positions,args.angles,10.)}
    primary=fit_harmonics(train,1);secondary=fit_harmonics(train,3)
    fits={label:{name:fit_metrics(data,fit) for name,data in tests.items()} for label,fit in [('first_harmonic',primary),('harmonics_1_to_3',secondary)]}
    shams=make_shams(primary['basis'],args.fit_seed+1)
    np.savez_compressed(args.out/'fitted_subspaces.npz',basis=primary['basis'],coefficients=primary['coefficients'],
        secondary_basis=secondary['basis'],secondary_coefficients=secondary['coefficients'],sham_bases=np.stack(shams),
        train_positions=train['position'],train_directions=train['direction'],train_delta=train['delta'],
        checkpoint_sha256=protocol['checkpoint_sha256'],model_seed=ckpt['seed'],fit_seed=args.fit_seed)
    print('Fit heldout:',json.dumps(fits['first_harmonic']['new_places_test']),flush=True)
    intervention=DriveInterventions(model,primary['basis'],shams)
    with torch.no_grad():
        # Independent algebra/orthogonality/dose checks on held-out drive differences.
        delta=torch.from_numpy(tests['new_places_test']['delta']).to(task.device)
        drives,dose=intervention.transform_drives(delta,torch.zeros_like(delta))
        identity=float((drives['remove_direction']-drives['residual_only']).abs().max())
        reconstruction=float((drives['restore_direction']+drives['residual_only']-delta).abs().max())
        assert identity==0 and reconstruction<1e-6,(identity,reconstruction)
        assert max(float(dose[f'sham{j}_relative_dose_error'].max()) for j in range(3))<1e-6
    result=dict(protocol=protocol,config=ckpt['config'],condition=ckpt['condition'],seed=ckpt['seed'],fit=fits,
        projection_validation=dict(removal_residual_identity_error=identity,component_sum_reconstruction_error=reconstruction,
            projector_idempotence_max_error=float(np.abs((primary['basis']@primary['basis'].T)@(primary['basis']@primary['basis'].T)-primary['basis']@primary['basis'].T).max()),
            sham_dose_max_errors=[float(dose[f'sham{j}_dose_error'].max()) for j in range(3)],
            sham_relative_dose_max_errors=[float(dose[f'sham{j}_relative_dose_error'].max()) for j in range(3)]))
    result['anchoring']=anchoring(model,task,intervention,args)
    result['forced']=forced_trajectories(model,task,intervention,args)
    assert sha(args.checkpoint)==protocol['checkpoint_sha256']
    result['elapsed_seconds']=time.monotonic()-started
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_results(result,args.out)
    print('DONE',result['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__':main()
