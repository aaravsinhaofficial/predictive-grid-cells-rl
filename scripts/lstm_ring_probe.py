"""Exploratory circular-memory screen of ORIGINAL frozen trained LSTMs.

Native states are continued under one common input, zero input, or their own
frozen input. Circular decoding is cross-validated by episode. Perturbations
are decoder-defined phase-changing and decoder-silent directions, NOT fitted
manifold tangents/normals. No fitted circle is treated as attractor evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from code.common import build_policy


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_diagnostic_controls(steps=150):
    """Known ring versus contracting point: phase retention alone is insufficient."""
    theta=np.linspace(-np.pi,np.pi,96,endpoint=False)
    initial=np.column_stack((np.cos(theta),np.sin(theta),np.zeros_like(theta)))
    decoder=dict(mean=np.zeros(3),scale=np.ones(3),
                 beta=np.array([[1.,0.],[0.,1.],[0.,0.]]),intercept=np.zeros(2))
    result={}
    for name in ('known_ring','contracting_point'):
        base=initial.copy()
        phase=initial.copy()
        phase[:,:2]=np.column_stack((np.cos(theta+.02),np.sin(theta+.02)))
        silent=initial.copy();silent[:,2]=.02
        for _ in range(steps):
            for a in (base,phase,silent):
                if name=='known_ring':
                    radius=np.linalg.norm(a[:,:2],axis=1,keepdims=True)
                    a[:,:2]*=1+.3*(1-radius)/radius
                    a[:,2]*=.7
                else:
                    a*=.7
        r=summarize_state(base,initial,decoder,theta,theta)
        phase_distance=float(np.mean(np.linalg.norm(phase-base,axis=1))/(2*np.sin(.01)))
        silent_distance=float(np.mean(np.linalg.norm(silent-base,axis=1))/.02)
        r.update(phase_state_distance_retention=phase_distance,
                 silent_state_distance_retention=silent_distance)
        result[name]=r
    assert result['known_ring']['phase_state_distance_retention']>.999
    assert result['known_ring']['silent_state_distance_retention']<1e-20
    assert result['contracting_point']['between_state_variance_ratio']<1e-20
    return result


def fit_decoder(state, angle, groups, rng):
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.zeros_like(y)
    null_pred = np.zeros_like(y)
    # A shuffled-label baseline uses the same folds and features.
    shuffled = y[rng.permutation(len(y))]
    for tr, te in cv.split(state, groups=groups):
        sc = StandardScaler().fit(state[tr])
        xtr, xte = sc.transform(state[tr]), sc.transform(state[te])
        pred[te] = Ridge(alpha=10.).fit(xtr, y[tr]).predict(xte)
        null_pred[te] = Ridge(alpha=10.).fit(xtr, shuffled[tr]).predict(xte)
    sc = StandardScaler().fit(state)
    ridge = Ridge(alpha=10.).fit(sc.transform(state), y)
    # Avoid enormous interventions in nearly constant coordinates.
    scale = np.maximum(sc.scale_, .01)
    beta = ridge.coef_.T / sc.scale_[:, None] * scale[:, None]
    intercept = ridge.intercept_
    return dict(mean=sc.mean_, scale=scale, beta=beta, intercept=intercept), dict(
        cv_mae_deg=float(np.rad2deg(np.abs(wrap(np.arctan2(pred[:,1], pred[:,0])-angle))).mean()),
        cv_cosine_accuracy=float(np.cos(np.arctan2(pred[:,1],pred[:,0])-angle).mean()),
        shuffled_label_cv_mae_deg=float(np.rad2deg(np.abs(wrap(np.arctan2(null_pred[:,1],null_pred[:,0])-angle))).mean()),
        target_resultant=float(abs(np.exp(1j*angle).mean())),
        folds=min(5,len(np.unique(groups))), alpha=10.)


def decode(state, decoder):
    out = ((state-decoder['mean'])/decoder['scale']) @ decoder['beta'] + decoder['intercept']
    return np.arctan2(out[:,1], out[:,0]), np.linalg.norm(out,axis=1)


def perturbations(state, decoder, rng, amplitude):
    z = (state-decoder['mean'])/decoder['scale']
    value = z @ decoder['beta'] + decoder['intercept']
    beta = decoder['beta']
    tangent = value[:,0,None]*beta[:,1] - value[:,1,None]*beta[:,0]
    tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
    basis = np.linalg.qr(beta)[0]
    silent = rng.normal(size=state.shape)
    silent -= (silent @ basis) @ basis.T
    silent /= np.linalg.norm(silent,axis=1,keepdims=True)
    eps = amplitude * np.sqrt(state.shape[1])
    return {'phase_plus':state+eps*tangent*decoder['scale'],
            'phase_minus':state-eps*tangent*decoder['scale'],
            'silent_plus':state+eps*silent*decoder['scale'],
            'silent_minus':state-eps*silent*decoder['scale']}


def summarize_state(state, initial, decoder, target, initial_phase):
    phase, radius = decode(state,decoder)
    z=(state-decoder['mean'])/decoder['scale']
    zi=(initial-decoder['mean'])/decoder['scale']
    v0=np.mean(np.var(zi,axis=0))
    return dict(target_mae_deg=float(np.rad2deg(np.abs(wrap(phase-target))).mean()),
        phase_change_from_initial_deg=float(np.rad2deg(np.abs(wrap(phase-initial_phase))).mean()),
        phase_cosine_retention=float(np.cos(phase-initial_phase).mean()),
        decoded_phase_resultant=float(abs(np.exp(1j*phase).mean())),
        between_state_variance_ratio=float(np.mean(np.var(z,axis=0))/max(v0,1e-12)),
        decoded_radius_median=float(np.median(radius)))


def simulate(net, inputs, states, steps, horizons):
    n=len(states['base'])
    names=list(states)
    full=np.concatenate([states[k] for k in names]).astype(np.float32)
    h,c=torch.from_numpy(full).chunk(2,dim=1)
    x=torch.from_numpy(np.tile(inputs,(len(names),1)).astype(np.float32))
    w=net.lstm
    with torch.inference_mode():
        drive=torch.nn.functional.linear(x,w.weight_ih,w.bias_ih)
        # Validate hand-expanded implementation against actual LSTMCell.
        a,b=w(x,(h,c))
        i,f,g,o=(drive+torch.nn.functional.linear(h,w.weight_hh,w.bias_hh)).chunk(4,1)
        cc=f.sigmoid()*c+i.sigmoid()*g.tanh()
        hh=o.sigmoid()*cc.tanh()
        error=max(float((a-hh).abs().max()),float((b-cc).abs().max()))
        assert error < 3e-6,error
        trajectories={0:{k:states[k].copy() for k in names}}
        for t in range(1,steps+1):
            i,f,g,o=(drive+torch.nn.functional.linear(h,w.weight_hh,w.bias_hh)).chunk(4,1)
            c=f.sigmoid()*c+i.sigmoid()*g.tanh()
            h=o.sigmoid()*c.tanh()
            if t in horizons:
                a=torch.cat((h,c),1).numpy().copy()
                trajectories[t]={k:a[j*n:(j+1)*n] for j,k in enumerate(names)}
    return trajectories,error


def model_run(condition,seed,args):
    start=time.monotonic()
    root=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    path=ROOT/f'results/lstm_navigation_phases_1B/{condition}_s{seed}_action_states.npz'
    cfg=json.loads(root.with_suffix('.json').read_text())
    checkpoint=torch.load(root.with_suffix('.pt'),map_location='cpu',weights_only=True)
    net=build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(checkpoint['policy'])
    with np.load(path) as a:
        data=dict(a)
    state=np.concatenate((data['h_before'],data['c_before']),axis=1).astype(np.float64)
    with torch.inference_mode():
        x=net._concat(torch.from_numpy(data['embedding']),torch.from_numpy(data['current_code']),
            torch.from_numpy(data['goal_code']),torch.from_numpy(data['previous_action']),
            torch.from_numpy(data['previous_reward']),None).numpy()
    groups=data['slot']*100+data['episode']
    bearing=np.arctan2((data['goal_center']-data['pos'])[:,1],(data['goal_center']-data['pos'])[:,0])
    targets={'facing':data['hd'],'travel_direction':np.arctan2(data['dir'][:,1],data['dir'][:,0]),
             'goal_bearing_egocentric':wrap(bearing-data['hd'])}
    rng=np.random.default_rng(9762+seed)
    decoders,decoding={},{}
    states={'base':state}
    for variable,angle in targets.items():
        decoder,quality=fit_decoder(state,angle,groups,rng)
        decoders[variable]=decoder
        decoding[variable]=quality
        for name,p in perturbations(state,decoder,rng,args.amplitude).items():
            states[f'{variable}/{name}']=p
    regimes={'common_mean_input':np.broadcast_to(x.mean(0),x.shape).copy(),
             'zero_input':np.zeros_like(x),'own_frozen_input':x}
    horizons=sorted({0,1,5,15,30,60,args.steps}&set(range(args.steps+1)))
    results={}
    maxerror=0.
    for regime,inp in regimes.items():
        print(f'{condition} s{seed}: {regime}, n={len(state)}, horizon={args.steps}',flush=True)
        traj,error=simulate(net,inp,states,args.steps,horizons)
        maxerror=max(maxerror,error)
        rows={}
        for variable,decoder in decoders.items():
            initial_phase,_=decode(state,decoder)
            rows[variable]={}
            for t in horizons:
                base=traj[t]['base']
                bp,_=decode(base,decoder)
                row=summarize_state(base,state,decoder,targets[variable],initial_phase)
                for axis in ('phase','silent'):
                    norms=[];phase_shift=[];norms0=[];phase0=[]
                    for sign in ('plus','minus'):
                        key=f'{variable}/{axis}_{sign}'
                        other=traj[t][key]
                        phase,_=decode(other,decoder)
                        first_phase,_=decode(states[key],decoder)
                        norms.append(np.linalg.norm((other-base)/decoder['scale'],axis=1))
                        norms0.append(np.linalg.norm((states[key]-state)/decoder['scale'],axis=1))
                        phase_shift.append(np.abs(wrap(phase-bp)))
                        phase0.append(np.abs(wrap(first_phase-initial_phase)))
                    row[f'{axis}_state_distance_retention']=float(np.mean(norms)/np.mean(norms0))
                    row[f'{axis}_phase_shift_deg']=float(np.rad2deg(np.mean(phase_shift)))
                    if axis=='phase':
                        row['phase_perturbation_angle_retention']=float(np.mean(phase_shift)/max(np.mean(phase0),1e-12))
                rows[variable][str(t)]=row
        results[regime]=rows
    record=dict(condition=condition,seed=seed,n=len(state),episodes=len(np.unique(groups)),
        state_dimensions=state.shape[1],checkpoint_sha256=digest(root.with_suffix('.pt')),
        snapshot_sha256=digest(path),decoding=decoding,continuation=results,
        amplitude_rms_standardized_state=args.amplitude,step_horizons=horizons,
        lstm_reconstruction_error=maxerror,seconds=time.monotonic()-start)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'{condition}_s{seed}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(f'{condition} s{seed}: done in {record["seconds"]:.1f}s',flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=[1])
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--amplitude',type=float,default=.02)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_attractor/original_pilot')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    runs=[model_run(c,s,args) for s in args.seeds for c in ('gc100','pgc33')]
    summary=dict(protocol='Original-agent exploratory circular-memory screen; no attractor claim',
        common_input='Per-model native input mean repeated identically for all snapshot states; approximate OOD cue withdrawal.',
        zero_input='All 1287 input values zero; stronger OOD cue withdrawal.',
        driven_reference='Each snapshot repeats its own complete native input.',
        perturbations='Decoder-defined phase-changing and decoder-silent directions in diagonally standardized full h/c coordinates; not geometric manifold tangents/normals.',
        decoder='Cosine/sine ridge alpha=10; all states fit for continuation; held-out-episode decoding reported separately. One label-shuffle reference, not a permutation p-value.',
        step_duration_s=1/15,amplitude=args.amplitude,seeds=args.seeds,
        diagnostic_controls=synthetic_diagnostic_controls(args.steps),runs=runs)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    variable='travel_direction'
    for condition,color,label in [('gc100','#3379B5','Grid'),('pgc33','#D87532','PGC + grid')]:
        group=[r for r in runs if r['condition']==condition]
        for ax,key in zip(axes,['phase_change_from_initial_deg','phase_state_distance_retention','silent_state_distance_retention']):
            values=[]
            for r in group:
                rows=r['continuation']['common_mean_input'][variable]
                t=np.array([int(k) for k in rows])/15
                values.append([v[key] for v in rows.values()])
            values=np.array(values)
            ax.plot(t,values.mean(0),color=color,label=label,lw=2.5)
            if len(values)>1:
                for v in values:
                    ax.plot(t,v,color=color,lw=.6,alpha=.18)
    for ax,title,ylabel in zip(axes,['Direction drift','Phase-changing perturbation','Readout-silent perturbation'],['Mean phase change (degrees)','State distance / initial','State distance / initial']):
        ax.set(title=title,xlabel='Seconds under common input',ylabel=ylabel)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.supxlabel('Original trained agents · Exploratory cue-withdrawal screen · Perturbations defined by decoder',fontsize=9)
    fig.savefig(args.out/'original_ring_screen.png',dpi=170)
    fig.savefig(args.out/'original_ring_screen.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
