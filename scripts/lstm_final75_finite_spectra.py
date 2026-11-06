"""Prespecified2s-to4s propagator spectra in the four fixed75M models.

Full512-state spectra and empirical tangent/normal propagation; no fitted ring,
new candidate threshold, phase selection, or alteration of earlier probes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits
from lstm_endogenous_observable_flow import load_net
from lstm_endogenous_curve_recovery import advance,sha,save
from lstm_endogenous_equilibrium_elimination import stable_full_jacobian
from lstm_endogenous_continued_observable_v2 import validate_checkpoint,VERSIONS

OUT=ROOT/'results/lstm_final75_finite_spectra'
REFERENCE,HORIZON,PHASES=30,30,16
METRICS=('raw_full_state','h_c_RMS_full_state')


def periodic_tangent(curve,theta):
    """Three-point derivative in actual angular coordinates, with periodic wrap."""
    dm=np.mod(theta-np.roll(theta,1),2*np.pi)
    dp=np.mod(np.roll(theta,-1)-theta,2*np.pi)
    assert np.all(dm>0) and np.all(dp>0)
    assert np.all(dm<np.pi) and np.all(dp<np.pi)
    return (-dp/(dm*(dm+dp)))[:,None]*np.roll(curve,1,axis=0)+\
           ((dp-dm)/(dm*dp))[:,None]*curve+\
           (dm/(dp*(dm+dp)))[:,None]*np.roll(curve,-1,axis=0)


def compose(jacobians):
    """Column-vector perturbations: final Jacobian multiplies on the left."""
    product=torch.eye(jacobians[0].shape[0],dtype=jacobians[0].dtype)
    for jac in jacobians:product=jac@product
    return product


def describe(a):
    a=np.asarray(a,dtype=float)
    return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()))


def metric_spectrum(product,tangent0,tangent1,radial0,scale,actor):
    """The fixed coordinate transform is q=z/scale at BOTH interval endpoints."""
    m=product*scale[None,:]/scale[:,None]
    a=tangent0/scale;b=tangent1/scale
    an,bn=np.linalg.norm(a),np.linalg.norm(b)
    assert an>1e-20 and bn>1e-20,'An empirical tangent disappeared; retain failure rather than invent a direction'
    u0,u1=a/an,b/bn
    eigenvalues,eigenvectors=scipy.linalg.eig(m,check_finite=False)
    order=np.argsort(-np.abs(eigenvalues),kind='stable')
    eigenvalues=eigenvalues[order];eigenvectors=eigenvectors[:,order]
    singular=scipy.linalg.svdvals(m,check_finite=False)
    # P_normal,out M P_normal,in, evaluated by rank-one subtraction.
    normal=m-np.outer(m@u0,u0)
    normal=normal-np.outer(u1,u1@normal)
    _,normal_singular,normal_vh=scipy.linalg.svd(normal,full_matrices=False,check_finite=False)
    propagated=m@u0;gain=np.linalg.norm(propagated)
    actual=m@a
    radial=radial0/scale;radial=radial-u0*np.dot(radial,u0)
    radial_norm=np.linalg.norm(radial)
    if radial_norm>1e-20:
        radial/=radial_norm;rd=m@radial
        radial_normal_gain=float(np.linalg.norm(rd-u1*np.dot(rd,u1)))
        radial_total_gain=float(np.linalg.norm(rd))
    else:radial_normal_gain=None;radial_total_gain=None
    physical_vectors=eigenvectors*scale[:,None]
    physical_norm=np.linalg.norm(physical_vectors,axis=0)
    hidden_size=len(scale)//2
    hidden_fraction=np.linalg.norm(physical_vectors[:hidden_size],axis=0)/physical_norm
    actor_gain=np.linalg.norm(actor@physical_vectors[:hidden_size],axis=0)/physical_norm
    alignment=np.abs(u0.conj()@eigenvectors)/np.linalg.norm(eigenvectors,axis=0)
    values=dict(eigenvalues=eigenvalues,singular_values=singular,normal_singular_values=normal_singular,
        eigenvector_tangent_alignment=alignment,eigenvector_hidden_fraction=hidden_fraction,
        eigenvector_actor_logit_gain=actor_gain,top16_eigenvectors_metric=eigenvectors[:,:16],
        worst_normal_initial_metric=normal_vh[0],initial_tangent_metric=u0,final_tangent_metric=u1)
    summary=dict(spectral_radius=float(np.abs(eigenvalues).max()),max_singular_gain=float(singular[0]),
        maximum_transverse_gain=float(normal_singular[0]),tangent_total_gain=float(gain),
        tangent_final_alignment=float(abs(np.dot(propagated,u1))/max(gain,1e-300)),
        tangent_signed_final_component=float(np.dot(propagated,u1)),
        empirical_tangent_norm_retention=float(bn/an),
        propagated_empirical_tangent_relative_error=float(np.linalg.norm(actual-b)/bn),
        radial_normal_gain=radial_normal_gain,radial_total_gain=radial_total_gain,
        eigenvalue_magnitude_counts={str(t):int(np.sum(np.abs(eigenvalues)>=t)) for t in (.1,.5,.9,.99,1.,1.01)},
        singular_gain_counts={str(t):int(np.sum(singular>=t)) for t in (.1,.5,.9,.99,1.,1.01)},
        maximum_eigenvector_tangent_alignment=float(alignment.max()),
        top16_eigenvector_hidden_fraction=hidden_fraction[:16].tolist(),
        top16_eigenvector_actor_logit_gain=actor_gain[:16].tolist(),
        worst_normal_initial_tangent_dot=float(abs(normal_vh[0]@u0)))
    return summary,values


def validate_product(cell,drive,state,product,steps):
    z=state.detach().clone().requires_grad_(True)
    direction=torch.sin(torch.arange(len(state),dtype=torch.float64)+.37)
    direction/=direction.norm()
    def trajectory(v):
        for _ in range(steps):v=advance(cell,drive,v[None])[0]
        return v
    _,auto=torch.autograd.functional.jvp(trajectory,z,direction)
    analytic=product@direction
    error=float((analytic-auto).abs().max())
    relative=float((analytic-auto).norm()/auto.norm().clamp_min(1e-30))
    assert error<1e-7 and relative<1e-7,(error,relative)
    eps=1e-5
    finite=(trajectory(state+eps*direction)-trajectory(state-eps*direction))/(2*eps)
    return dict(autograd_jvp_max_absolute_error=error,autograd_jvp_relative_error=relative,
        centered_difference_epsilon=eps,centered_difference_relative_error=float((analytic-finite).norm()/finite.norm().clamp_min(1e-30)))


def protocol():
    files=['scripts/lstm_final75_finite_spectra.py','tests/test_lstm_final75_finite_spectra.py',
        'scripts/lstm_endogenous_equilibrium_elimination.py','scripts/lstm_endogenous_curve_recovery.py',
        'scripts/lstm_endogenous_continued_observable_v2.py','scripts/lstm_endogenous_observable_flow.py']
    return dict(version='final75_finite_time_spectra_v1',arms=VERSIONS,conditions=['gc100','pgc33'],contexts=[0,1],
        checkpoint='Each fixed completed75.072M endpoint, exact50.048M extension/25.024Mparent provenance; no stage selection',
        family='Same native travel-family archives as the primary collector; ordinary-current common-input release only',
        selection=dict(reference_updates=REFERENCE,reference_seconds=2.,horizon_updates=HORIZON,end_seconds=4.,
                       phases=PHASES,rule='Indices0,48,...,720 of all768 actual phases; independent of activity and outcome'),
        state='All256h+256c, original neuron coordinates; no fitted plane or quotient',
        jacobian='Validated stable_full_jacobian, float64, stable sigmoid(-forget_logit) complement; product J29...J0 along actual evolving state under one locked ordinary input',
        scales='Raw units and the existing h/c block RMS computed once from time0 full family, held fixed at both endpoints',
        tangent='Periodic three-point finite difference in actual angular coordinates, using all768 phases at2s and4s',
        spectra='All512 propagator eigenvalues, all512 singular values, all512 normal-propagator singular values; every eigenvector tangent/hidden/actor alignment; top16full eigenvectors retained',
        normals='Project initial and final tangent directions out in each metric; report exact largest singular gain of Pnormal,out M Pnormal,in and radial-in-family normal gain',
        validation='Known noncommuting matrix product/order, known slow+contracting system, analytic LSTM/autograd check, each archive firstphase product JVP and centered finite difference; compare end state with saved4s curve and native LSTMCell',
        interpretation='Descriptive finite-time propagation, not a fixed-point eigensystem; no new ring candidate gate or outcome-based selection. Singular gains depend on metric; eigenvectors are not compared across unrelated neuron labels.',
        source_sha256={p:sha(ROOT/p) for p in files},
        parent_observable_protocol_sha256=sha(ROOT/'results/lstm_endogenous_ring/observable_history/final75/protocol.json'))


def run(checkpoint,archive,arm,out):
    net,data,cfg=load_net(checkpoint);parent_hash=validate_checkpoint(data,cfg,arm)
    a=np.load(archive);assert str(a['checkpoint_sha256'].item())==sha(checkpoint)
    assert 'states' in a and 'native_states' not in a,'This prespecified audit uses travel release only'
    times=list(a['times']);curve0=a['states'][times.index(REFERENCE)].astype(np.float64)
    curve1=a['states'][times.index(REFERENCE+HORIZON)].astype(np.float64)
    initial=a['states'][times.index(0)].astype(np.float64);theta=a['theta'].astype(np.float64)
    assert curve0.shape==curve1.shape==(768,512) and theta.shape==(768,)
    tang0,tang1=periodic_tangent(curve0,theta),periodic_tangent(curve1,theta)
    scales=np.repeat(np.sqrt(np.mean(initial.reshape(-1,2,256)**2,axis=(0,2))),256)
    scales=np.maximum(scales,1e-8)
    assert np.allclose(scales,a['scale'],atol=1e-12,rtol=1e-12),'Metric differs from saved original family scale'
    x=torch.tensor(a['ordinary_input'],dtype=torch.float64);assert x.shape==(1,1287)
    drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
    indices=np.arange(0,768,768//PHASES);rows=[];arrays={};products=[];ends=[]
    actor=net.pi.weight.detach().numpy()
    for k,index in enumerate(indices):
        z=torch.tensor(curve0[index],dtype=torch.float64);start=z.clone();product=torch.eye(512,dtype=torch.float64)
        native_h,native_c=start[:256][None].clone(),start[256:][None].clone()
        with torch.no_grad():
            for _ in range(HORIZON):
                jac,_=stable_full_jacobian(net.lstm,drive,z)
                product=jac@product;z=advance(net.lstm,drive,z[None])[0]
                native_h,native_c=net.lstm(x,(native_h,native_c))
        validation=validate_product(net.lstm,drive,start,product,HORIZON) if k==0 else None
        native=torch.cat((native_h,native_c),1)[0]
        end_error=z.numpy()-curve1[index]
        row=dict(phase_index=int(index),theta=float(theta[index]),
            archive_end_state_max_absolute_error=float(np.max(np.abs(end_error))),
            archive_end_state_RMS_error=float(np.sqrt(np.mean(end_error**2))),
            native_LSTMCell_end_state_max_absolute_error=float((z-native).abs().max()),
            product_validation=validation,metrics={})
        assert row['archive_end_state_max_absolute_error']<1e-7,'Replayed state differs from saved ordinary-input family'
        assert row['native_LSTMCell_end_state_max_absolute_error']<1e-7,'Manual transition differs from native LSTMCell'
        for name,scale in zip(METRICS,(np.ones(512),scales)):
            summary,values=metric_spectrum(product.numpy(),tang0[index],tang1[index],
                                           curve0[index]-curve0.mean(0),scale,actor)
            row['metrics'][name]=summary
            for key,value in values.items():arrays.setdefault(f'{name}__{key}',[]).append(value)
        rows.append(row);products.append(product.numpy());ends.append(z.numpy())
    folder=out/arm/cfg['cond'];folder.mkdir(parents=True,exist_ok=True)
    stem=folder/archive.name.replace('_states.npz','')
    np.savez_compressed(stem.with_suffix('.npz'),phase_indices=indices,theta=theta[indices],
        raw_products=np.stack(products),propagated_states=np.stack(ends),scale=scales,
        initial_states=curve0[indices],saved_final_states=curve1[indices],
        **{key:np.stack(values) for key,values in arrays.items()})
    record=dict(arm=arm,condition=cfg['cond'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        parent25M_sha256=parent_hash,archive=str(archive),archive_sha256=sha(archive),
        source_sha256=sha(__file__),protocol_sha256=sha(out/'protocol.json'),
        full_spectra_npz=str(stem.with_suffix('.npz')),full_spectra_sha256=sha(stem.with_suffix('.npz')),
        reference_updates=REFERENCE,horizon_updates=HORIZON,phases=rows,
        summary={name:{key:describe([row['metrics'][name][key] for row in rows]) for key in
            ('spectral_radius','max_singular_gain','maximum_transverse_gain','tangent_total_gain','tangent_final_alignment',
             'empirical_tangent_norm_retention','propagated_empirical_tangent_relative_error')} for name in METRICS})
    save(stem.with_suffix('.json'),record)
    print(json.dumps(dict(arm=arm,condition=cfg['cond'],archive=str(archive),summary=record['summary'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--archives',type=Path,nargs='+');p.add_argument('--arm',choices=tuple(VERSIONS))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'protocol.json';design=protocol()
    if args.freeze_only:
        if path.exists():raise FileExistsError('Preserve frozen spectral protocol')
        save(path,design);print(sha(path));return
    assert path.exists() and json.loads(path.read_text())==design,'Spectral protocol missing or changed'
    if args.checkpoint is None or not args.archives or args.arm is None:p.error('--checkpoint, --archives, --arm required')
    with threadpool_limits(limits=2):
        for archive in args.archives:run(args.checkpoint,archive,args.arm,args.out)


if __name__=='__main__':main()
