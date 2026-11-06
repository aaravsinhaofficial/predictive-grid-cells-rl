"""Read-only decomposition of archived40M direction families through the actor.

No environment, recurrent flow, optimizer, action intervention or fitted decoder.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from lstm_endogenous_curve_recovery import sha,save
from lstm_endogenous_reorientation_observable import validate_checkpoint,validate_archive,WORKERS

OUT=ROOT/'results/lstm_reorientation_actor_readout'
INPUT=ROOT/'results/lstm_reorientation_mechanism/final'
TIMES=(0,15,30,60)
CUES=('native','ordinary','plus60','minus60')
ACTIONS=('turn_left','turn_right','forward','backward','strafe_left','strafe_right')
SOURCES=('scripts/lstm_reorientation_actor_readout.py','tests/test_lstm_reorientation_actor_readout.py',
    'scripts/lstm_endogenous_reorientation_observable.py','code/nets.py')


def rms_vector(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x,dtype=np.float64)**2,axis=-1))))


def ratio(a,b):
    return float(a/b) if b>1e-30 else None


def softmax(x):
    x=np.asarray(x,dtype=np.float64);e=np.exp(x-x.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def decompose(hidden,weight,bias,theta):
    """Exact linear actor row/null decomposition with no estimated neural map."""
    h,w,b,theta=[np.asarray(x,dtype=np.float64) for x in (hidden,weight,bias,theta)]
    assert h.ndim==2 and w.shape==(6,h.shape[1]) and b.shape==(6,) and theta.shape==(len(h),)
    assert all(np.isfinite(x).all() for x in (h,w,b,theta))
    q=np.eye(6)-np.ones((6,6))/6
    a=q@w;qb=q@b;mu=h.mean(0);dh=h-mu
    logits=h@w.T+b;centered=logits@q
    mean=a@mu+qb;signal=dh@a.T
    reconstructed=mean+signal
    error=float(np.max(np.abs(centered-reconstructed)))
    assert np.allclose(centered,reconstructed,atol=1e-10,rtol=1e-11)
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    tol=max(a.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    rank=int(np.count_nonzero(s>tol));basis=vt[:rank]
    projected=(dh@basis.T)@basis;null=dh-projected
    assert np.allclose(projected@a.T,signal,atol=1e-10,rtol=1e-10)
    assert np.allclose(null@a.T,0,atol=1e-10)
    hidden_energy=float(np.mean(np.sum(dh**2,axis=1)))
    projected_energy=float(np.mean(np.sum(projected**2,axis=1)))
    signal_rms=rms_vector(signal);hidden_rms=rms_vector(dh)
    p=softmax(logits);p0=softmax(mean);dp=p-p.mean(0)
    assert np.allclose(p,softmax(centered),atol=1e-14)
    jac=np.diag(p0)-np.outer(p0,p0)
    local=signal@jac.T
    harmonic=np.column_stack((np.cos(theta),np.sin(theta)))
    harmonic-=harmonic.mean(0)
    coeff=np.linalg.lstsq(harmonic,signal,rcond=None)[0]
    first_harmonic=harmonic@coeff
    preferred=int(np.argmax(mean));runner=int(np.argsort(mean)[-2]);translation=int(2+np.argmax(mean[2:]))
    margins=[]
    for other in range(6):
        if other==preferred:continue
        baseline=float(mean[preferred]-mean[other]);variation=signal[:,preferred]-signal[:,other]
        margins.append(dict(other_action=ACTIONS[other],mean_margin=baseline,
            directional_margin_rms=float(np.sqrt(np.mean(variation**2))),
            variation_over_mean_margin=ratio(float(np.sqrt(np.mean(variation**2))),abs(baseline)) if abs(baseline)>1e-12 else None,
            fraction_other_outweighs_preferred=float(np.mean(centered[:,other]>centered[:,preferred]))))
    return dict(phases=len(h),hidden_dimensions=h.shape[1],actor_rank=rank,
        hidden_family_rms=hidden_rms,actor_rowspace_hidden_variance_fraction=ratio(projected_energy,hidden_energy),
        actor_nullspace_hidden_variance_fraction=ratio(float(np.mean(np.sum(null**2,axis=1))),hidden_energy),
        isotropic_rowspace_fraction=rank/h.shape[1],actor_singular_values=s.tolist(),
        centered_logit_family_rms=signal_rms,hidden_to_centered_logit_gain=ratio(signal_rms,hidden_rms),
        isotropic_gain_reference=float(np.linalg.norm(a)/np.sqrt(h.shape[1])),
        hidden_to_logit_gain_over_isotropic=ratio(signal_rms,hidden_rms*np.linalg.norm(a)/np.sqrt(h.shape[1])),
        common_logit_shift_family_rms=float(np.sqrt(np.mean((dh@w.mean(0))**2))),
        centered_logit_mean=mean.tolist(),mean_hidden_logit_component=(a@mu).tolist(),centered_actor_bias=qb.tolist(),
        reconstruction_max_abs_error=error,
        preferred_mean_logit_action=ACTIONS[preferred],runner_up_mean_logit_action=ACTIONS[runner],
        preferred_mean_logit_margin=float(mean[preferred]-mean[runner]),
        right_turn_minus_best_translation_mean_logit=float(mean[1]-mean[translation]),
        pairwise_margins=margins,
        probabilities=dict(at_mean_logits=p0.tolist(),mean_across_phases=p.mean(0).tolist(),
            family_rms=rms_vector(dp),best_translation_mean=float(p[:,2:].max(1).mean()),
            total_translation_mean=float(p[:,2:].sum(1).mean()),
            right_turn_mean=float(p[:,1].mean()),preferred_action_fraction=float(np.mean(p.argmax(1)==preferred)),
            fraction_phases_changing_mean_preferred_action=float(np.mean(p.argmax(1)!=preferred)),
            entropy_mean=float(np.mean(-np.sum(p*np.log(p.clip(1e-300)),axis=1)))),
        softmax=dict(local_sensitivity_singular_values=np.linalg.svd(jac,compute_uv=False).tolist(),
            local_observed_signal_gain=ratio(rms_vector(local),signal_rms),
            actual_observed_signal_gain=ratio(rms_vector(dp),signal_rms),
            local_prediction_error_rms=rms_vector((p-p0)-local)),
        direction_first_harmonic=dict(centered_logit_rms=rms_vector(first_harmonic),
            fraction_centered_logit_variance=ratio(rms_vector(first_harmonic)**2,signal_rms**2),
            coefficients=coeff.tolist(),interpretation='Fixed Fourier cosine/sine projection of actual travel phase; descriptive angular structure, not decoder/generalization evidence or proof of actor use'))


def protocol():
    return dict(version='fixed40M_actor_linear_readout_decomposition_v1',
        cohort='All two populations×two physically selected fixed40Mcontexts×fourcuebranches×times0,15,30,60=64rows;768angles each. One originalseedpair; nested measurements, no stage/context/outcome selection.',
        inputs='Saved hidden state h only (first256coordinates), exact frozen learned pi.weight and pi.bias, saved actualtraveltheta. Cell state has no direct actor connection. No environment or recurrent simulation.',
        linear='For Q=I−11T/6 and A=QW, decompose h=mean(h)+delta_h and Qlogits=Amean(h)+Qb+A delta_h. Exact SVD rowspace/nullspace projection of A; tolerance max(shape)*machineepsilon*largest singular value. Remove only action-common logit directions that leave softmax identical.',
        quantities='Hidden variance fraction in actor rowspace, centeredlogit RMS variation and gain per hidden RMS, rank/256 isotropic orientation reference, firstangularharmonic share, mean preferred-action margins and directional pairwise-margin RMS, best/summedtranslation probabilities, actual and linearized softmax sensitivity to observed logit variation.',
        interpretation='Rowspace energy measures hidden variation in actor-sensitive directions, not guaranteed choice effects; centered-logit and exactprobability variation are separate. Small probability variation despite appreciable logit variation and large stable margins indicates output saturation. Both may coexist. Harmonic share is a descriptive fixedFourier projection, not decoder/generalization evidence. Marginratios are null for absolute meanmargin<=1e-12. Linearizedsoftmax sensitivity is separate from exactprobabilities; report fraction of phases changing meanpreferredaction. No actor normalization intervention.',
        timing='Targeted follow-up motivated by already observed near-deterministic40MPGCturning, frozen before computing these new decomposition values; not a pre-training hypothesis test.',
        read_only='No training, rollout, noise resampling, actor intervention or recurrent state update. Checkpoint and archive hashes checked before/after. Grid cue identities exact.',
        tests='Exact logit reconstruction, nullspace elimination, softmax/shared-logit-shift invariance, and synthetic separation of nullspace versus saturation.',
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        primary_manifest_sha256=sha(ROOT/'results/lstm_reorientation_mechanism/implementation_protocol.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    design=protocol();manifest=args.out/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==design
    elif args.freeze_only:save(manifest,design)
    else:raise RuntimeError('Freeze reviewed protocol before computing outcomes')
    if args.freeze_only:return
    torch.set_num_threads(2);rows=[];sources={}
    for condition,worker in WORKERS.items():
        path=ROOT/'results/ring_cloud_artifacts'/worker/'final.pt';before=sha(path)
        data=torch.load(path,map_location='cpu',weights_only=True);validate_checkpoint(data,condition)
        w=data['policy']['pi.weight'].double().numpy();b=data['policy']['pi.bias'].double().numpy()
        sources[str(path.relative_to(ROOT))]=before
        for context in range(2):
            archive_path=INPUT/condition/f'context{context}_states.npz';digest=sha(archive_path)
            with np.load(archive_path) as loaded:a={k:loaded[k] for k in loaded.files}
            validate_archive(a,before,condition);sources[str(archive_path.relative_to(ROOT))]=digest
            records={}
            for cue in CUES:
                records[cue]={}
                for t in TIMES:
                    h=a[cue+'_states'][list(a['times']).index(t),:,:256]
                    record=decompose(h,w,b,a['theta']);records[cue][t]=record
                    rows.append(dict(condition=condition,context=context,cue=cue,time_updates=t,
                        time_seconds=t/15,checkpoint_sha256=before,archive_sha256=digest,**record))
            if condition=='gc100':assert all(records[cue]==records['native'] for cue in CUES)
            assert sha(archive_path)==digest
        assert sha(path)==before
    assert len(rows)==64
    save(args.out/'summary.json',dict(protocol_sha256=sha(manifest),source_sha256=sources,rows=rows,
        complete=True,all64rows_retained=True,grid_cue_shams_exact=True,no_environment_or_state_updates=True))


if __name__=='__main__':main()
