"""Original navigation LSTMs: visible heading families, then common-input turns.

No ring, decoder, or controller is trained into the agent. A phase decoder is
analysis-only. Full (h,c) perturbations are projected against the evolving
family tangent, not against the initial curve or a PCA plane.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from code.common import build_env, build_policy, build_population
from lstm_endogenous_curve_recovery import advance, decomposition, describe

TIMES = (0, 1, 5, 15, 30, 75, 150)
SETTLES = (0, 15, 75)
PROBE_TIMES = (0, 1, 5, 15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def periodic_tangent(state, delta):
    return (state.roll(-1, 0) - state.roll(1, 0)) / (2 * delta)


def fit_readout(state, theta):
    """Fit on even phases; odd phases are held out at every time point."""
    x = np.asarray(state, dtype=np.float64)
    train = np.arange(len(x)) % 2 == 0
    center = x[train].mean(0)
    xx = x[train] - center
    y = np.column_stack((np.cos(theta[train]), np.sin(theta[train])))
    gram = xx @ xx.T
    alpha = max(float(np.trace(gram) / len(gram)) * .01, 1e-12)
    weight = xx.T @ np.linalg.solve(gram + alpha * np.eye(len(gram)), y)
    return center, weight, ~train


def readout_metrics(state, theta, decoder, movement=0.):
    center, weight, test = decoder
    y = (np.asarray(state) - center) @ weight
    phase = np.arctan2(y[:, 1], y[:, 0])
    amplitude = np.linalg.norm(y, axis=1)
    # Subtract the known common physical turn before comparing initial phases.
    error = wrap(phase - movement - theta)
    ordered = np.sort(np.mod(phase[test], 2 * np.pi))
    max_gap = float(np.diff(np.r_[ordered, ordered[0] + 2*np.pi]).max())
    valid = amplitude[test] >= .1
    return dict(heldout_mae_deg=float(np.rad2deg(np.abs(error[test])).mean()),
                heldout_cosine_accuracy=float(np.cos(error[test]).mean()),
                heldout_amplitude_median=float(np.median(amplitude[test])),
                heldout_amplitude_at_least_point1_fraction=float(valid.mean()),
                decoded_phase_coverage_fraction=1-max_gap/(2*np.pi),
                heldout_points=int(test.sum()))


def candidate_gates(trajectory, recovery):
    row = trajectory['75']; probe = recovery['75']['horizons']['15']
    phase = row['fixed_readout_movement_corrected']
    return dict(radius_after_5s_at_least_20_percent=row['scaled_radius_retention'] >= .2,
                fixed_heading_mae_after_5s_at_most_30deg=phase['heldout_mae_deg'] <= 30,
                fixed_heading_coverage_after_5s_at_least_half=phase['decoded_phase_coverage_fraction'] >= .5,
                fixed_readout_amplitude_valid_in_80_percent=phase['heldout_amplitude_at_least_point1_fraction'] >= .8,
                one_second_tangent_gain_at_least_half=probe['latent_tangent_total_gain']['median'] >= .5,
                one_second_normal_residual_below_half=probe['normal_transverse_gain']['median'] < .5,
                tangent_over_transverse_greater_than_two=probe['tangent_over_transverse'] > 2)


def make_env(cfg, archive, n, context):
    env = build_env(dict(cfg, episode_len=10000, img_noise=0.), n,
                    716021+context, 'cpu', stagger=False)
    env.reset_all()
    env.noise_v = env.noise_r = 0.
    env.pos[:] = torch.as_tensor(archive['history_pos'][0])
    env.spawn[:] = env.pos
    env.goal[:] = torch.as_tensor(archive['history_goal_center'][0])
    env.mdir[:] = torch.as_tensor(archive['history_direction'][0])
    env.land_pos[:] = torch.as_tensor(archive['history_landmark_pos'])
    env.stripe_phase[:] = torch.as_tensor(archive['history_stripe_phase'])
    env.vf.zero_(); env.vs.zero_(); env.vr.zero_()
    env.hd[:] = torch.arange(n)*(2*np.pi/n) + float(archive['history_hd'][0])
    env.hd[:] = torch.remainder(env.hd+np.pi, 2*np.pi)-np.pi
    assert torch.all((env.pos-env.goal).norm(dim=1) > env.goal_radius+.01)
    return env


@torch.inference_mode()
def elicit(net, pop, cfg, archive, context, args):
    """Native renderer/CNN/maps and native pure-turn physics; no noise."""
    n = args.angles * 2  # doubled sampling audits the initial curve derivative
    env = make_env(cfg, archive, n, context)
    original_pos, original_mdir = env.pos.clone(), env.mdir.clone()
    action_id = context % 2
    actions = torch.full((n,), action_id, dtype=torch.long)
    current = pop.activity(env.pos, env.mdir, torch.zeros_like(env.pos))
    goal = torch.as_tensor(archive['history_x'][0, 768:1280]).expand(n, -1)
    initial = np.r_[archive['history_h_before'][0], archive['history_c_before'][0]]
    z = torch.as_tensor(initial, device=args.embed_device).expand(n, -1)
    state = tuple(v.contiguous() for v in z.chunk(2, 1))
    prior = archive['history_x'][0, 1280:1286]
    pa0 = int(prior.argmax()) if prior.sum() else -1
    code_d, goal_d = current.to(args.embed_device), goal.to(args.embed_device)
    pr = torch.zeros(n, device=args.embed_device)
    heading_history = []; maxpos = maxmdir = maxreward = 0.
    for step in range(args.prepare):
        heading_history.append(env.hd.numpy().copy())
        embedding = net.embed(env.render().to(args.embed_device))
        pa = torch.full((n,), pa0 if step == 0 else action_id,
                        dtype=torch.long, device=args.embed_device)
        x = net._concat(embedding, code_d, goal_d, pa, pr, None)
        state = net.lstm(x, state)
        _, reward, done, info = env.step(actions)
        assert not bool(done.any() or info['hit'].any())
        maxpos = max(maxpos, float((env.pos-original_pos).abs().max()))
        maxmdir = max(maxmdir, float((env.mdir-original_mdir).abs().max()))
        maxreward = max(maxreward, float(reward.abs().max()))
    full = torch.cat(state, 1).cpu().double()
    theta0 = heading_history[-1][::2].astype(np.float64)
    # Darkness means actual black RGB through the trained encoder, including
    # its learned bias and normalization. It is not a zero vision embedding.
    dark_embedding = net.embed(torch.zeros(1, 3, 84, 84, device=args.embed_device))
    dark_x = net._concat(dark_embedding, code_d[:1], goal_d[:1],
                        torch.tensor([action_id], device=args.embed_device),
                        torch.zeros(1, device=args.embed_device), None).cpu().double()
    blank_heading = {0: theta0}
    env.vision = 'none'
    for step in range(1, max(TIMES)+1):
        blank_heading[step] = env.hd.numpy()[::2].copy().astype(np.float64)
        _, reward, done, info = env.step(actions)
        assert not bool(done.any() or info['hit'].any())
        maxpos = max(maxpos, float((env.pos-original_pos).abs().max()))
        maxmdir = max(maxmdir, float((env.mdir-original_mdir).abs().max()))
        maxreward = max(maxreward, float(reward.abs().max()))
    assert maxpos == maxmdir == maxreward == 0.
    final_code = pop.activity(env.pos, env.mdir, torch.zeros_like(env.pos))
    code_drift = float((final_code-current).abs().max())
    assert code_drift == 0.
    # Average circularly near the wrap discontinuity, then verify a shared turn.
    motion = {str(k): float(np.angle(np.exp(1j*(v-theta0)).mean())) for k,v in blank_heading.items()}
    offset_error = max(float(np.abs(wrap(v-theta0-motion[str(k)])).max()) for k,v in blank_heading.items())
    assert offset_error < 1e-4, offset_error
    checks = dict(position_max_change_m=maxpos, last_movement_direction_max_change=maxmdir,
                  spatial_code_max_change=code_drift, max_reward=maxreward,
                  common_physical_turn_offset_max_error_rad=offset_error,
                  dark_input_family_max_difference=0., image_noise=0., motor_noise=0.,
                  action_id=action_id, fixed_last_movement_direction=original_mdir[0].tolist(),
                  position=original_pos[0].tolist(), goal_center=env.goal[0].tolist())
    return full, dark_x, theta0, motion, checks


def recovery(cell, drive, z, tangent, scale, seed):
    """Exact finite-time JVP, with normals measured against the evolving curve."""
    t = tangent/scale
    unit = t/t.norm(dim=1, keepdim=True).clamp_min(1e-30)
    gen = torch.Generator().manual_seed(seed)
    normals = []
    for _ in range(3):
        q = torch.randn(z.shape, generator=gen, dtype=z.dtype)
        q -= (q*unit).sum(1, keepdim=True)*unit
        normals.append(q/q.norm(dim=1, keepdim=True).clamp_min(1e-30))
    directions = torch.stack([tangent, unit*scale, *[q*scale for q in normals]], -1)
    initial_directions = directions.clone()
    state = z.clone(); rows = {}
    for k in range(max(PROBE_TIMES)+1):
        if k in PROBE_TIMES:
            reference = directions[:,:,0]/scale
            latent = directions[:,:,1]/scale
            phase, residual = decomposition(latent, reference)
            total, normal, induced = [], [], []
            for column in range(2, 5):
                p = directions[:,:,column]/scale
                dphase, perpendicular = decomposition(p, reference)
                total.append(p.norm(dim=1).numpy()); normal.append(perpendicular.numpy())
                induced.append(dphase.abs().numpy())
            row = dict(latent_tangent_total_gain=describe(latent.norm(dim=1).numpy()),
                       latent_tangent_transverse_gain=describe(residual.numpy()),
                       curve_tangent_norm_retention=describe((reference.norm(dim=1)/t.norm(dim=1).clamp_min(1e-30)).numpy()),
                       latent_tangent_phase_coordinate_gain=describe((phase*t.norm(dim=1)).numpy()),
                       normal_total_gain=describe(np.array(total)),
                       normal_transverse_gain=describe(np.array(normal)),
                       normal_induced_phase_radians_per_unit_state=describe(np.array(induced)))
            row['tangent_over_transverse'] = row['latent_tangent_total_gain']['median']/max(row['normal_transverse_gain']['median'],1e-30)
            rows[str(k)] = row
        if k < max(PROBE_TIMES):
            state, directions = advance(cell, drive, state, directions)
    # Full-state finite differences independently check all four kick axes.
    eps = 1e-4; delta = initial_directions[0,:,1:].T
    plus, minus = z[:1]+eps*delta, z[:1]-eps*delta
    for _ in range(max(PROBE_TIMES)):
        plus = advance(cell,drive,plus); minus = advance(cell,drive,minus)
    fd = (plus-minus)/(2*eps); expected = directions[0,:,1:].T
    err = float((fd-expected).norm()/expected.norm().clamp_min(1e-30))
    assert err < .005, err
    return dict(horizons=rows, finite_difference_relative_error=err)


def family_metrics(z, tangent, initial, scale, theta, movement, decoder, cell, drive):
    centered = z-z.mean(0); c0=initial-initial.mean(0)
    raw_radius=float(centered.square().mean().sqrt())
    scaled_radius=float((centered/scale).square().mean().sqrt())
    singular=torch.linalg.svdvals(centered/scale); variance=singular.square()
    fractions=variance/variance.sum().clamp_min(1e-40)
    neighbors=(z-z.roll(1,0)).norm(dim=1)
    displacement=(advance(cell,drive,z)-z)/scale
    phase_speed, normal_speed=decomposition(displacement,tangent/scale)
    refit=fit_readout((z/scale).numpy(),theta)
    return dict(raw_family_radius=raw_radius, scaled_family_radius=scaled_radius,
                raw_radius_retention=raw_radius/float(c0.square().mean().sqrt()),
                scaled_radius_retention=scaled_radius/float((c0/scale).square().mean().sqrt()),
                covariance_participation_ratio=float(1/fractions.square().sum()),
                first_five_pc_fractions=fractions[:5].tolist(),
                neighbor_max_over_median=float(neighbors.max()/neighbors.median().clamp_min(1e-30)),
                phase_tangent_norm=describe((tangent/scale).norm(dim=1).numpy()),
                step_total_drift_relative_to_family_radius=describe((displacement.square().mean(1).sqrt()/max(scaled_radius,1e-30)).numpy()),
                step_normal_drift_relative_to_family_radius=describe((normal_speed/np.sqrt(z.shape[1])/max(scaled_radius,1e-30)).numpy()),
                step_tangent_phase_drift_rad=describe(phase_speed.numpy()),
                fixed_readout_movement_corrected=readout_metrics((z/scale).numpy(),theta,decoder,movement),
                fixed_readout_initial_phase_memory=readout_metrics((z/scale).numpy(),theta,decoder),
                time_refit_information_only=readout_metrics((z/scale).numpy(),theta,refit))


def run_model_context(model, context, args):
    started=time.monotonic(); condition=model['condition']; seed=model['seed']
    checkpoint=Path(model['checkpoint']); cfg=json.loads(Path(model['config']).read_text())
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=build_policy(cfg,args.embed_device).eval().requires_grad_(False)
    net.load_state_dict(data['policy'],strict=True)
    assert cfg['policy']=='lstm' and cfg['nh']==256 and cfg['extra_input']=='none'
    pop=build_population(cfg,'cpu')
    source=args.context_root/f'{condition}_s{seed}/context{context}_states.npz'
    archive=np.load(source)
    full, dark_x, theta, movement, checks=elicit(net,pop,cfg,archive,context,args)
    net.cpu().double(); cell=net.lstm
    state=full[::2].clone(); initial=state.clone()
    tangent=periodic_tangent(full,np.pi/args.angles)[::2]
    coarse=periodic_tangent(state,2*np.pi/args.angles)
    scales=torch.stack([state[:,:256].square().mean().sqrt(),state[:,256:].square().mean().sqrt()]).clamp_min(1e-8)
    scale=scales.repeat_interleave(256)
    cosine=torch.nn.functional.cosine_similarity(tangent/scale,coarse/scale,dim=1)
    derivative_audit=dict(fine_samples=args.angles*2, analysis_samples=args.angles,
                         coarse_vs_fine_cosine=describe(cosine.numpy()),
                         coarse_over_fine_norm=describe(((coarse/scale).norm(dim=1)/(tangent/scale).norm(dim=1).clamp_min(1e-30)).numpy()),
                         method='Initial tangent is a central difference of actual rendered heading histories; its later propagation through the LSTM is exact. Hard renderer pixel boundaries preclude a globally smooth autograd heading derivative.')
    drive=torch.nn.functional.linear(dark_x,cell.weight_ih,cell.bias_ih)
    hh,cc=cell(dark_x.expand(args.angles,-1),(state[:,:256],state[:,256:]))
    reconstruction=float((advance(cell,drive,state)-torch.cat((hh,cc),1)).abs().max())
    assert reconstruction < 1e-12
    decoder=fit_readout((initial/scale).numpy(),theta)
    trajectory={}; probes={}; snapshots={}; tangent_snapshots={}; coarse_snapshots={}
    with torch.no_grad():
        for step in range(max(TIMES)+1):
            if step in TIMES:
                trajectory[str(step)]=family_metrics(state,tangent,initial,scale,theta,movement[str(step)],decoder,cell,drive)
                snapshots[step]=state.numpy().copy(); tangent_snapshots[step]=tangent.numpy().copy()
            if step in SETTLES:
                probes[str(step)]=recovery(cell,drive,state,tangent,scale,78091+100*seed+context)
                # Same random normals after changing only the tangent estimate.
                coarse_snapshots[str(step)]=recovery(cell,drive,state,coarse,scale,78091+100*seed+context)
            if step<max(TIMES):
                state,directions=advance(cell,drive,state,torch.stack((tangent,coarse),-1))
                tangent,coarse=directions[:,:,0],directions[:,:,1]
    gates=candidate_gates(trajectory,probes)
    record=dict(condition=condition,seed=seed,context=context,checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),checkpoint_frames=int(data.get('frames',-1)),
                context_archive_sha256=sha(source),source_sha256=sha(__file__),
                prepare_updates=args.prepare,angles=args.angles,state_dimensions=512,
                step_seconds=cfg['action_repeat']/60.,embedding_device=args.embed_device,
                recurrent_analysis_dtype='float64',full_state_block_rms_scale=scales.tolist(),
                physics_checks=checks,tangent_resolution_audit=derivative_audit,
                native_lstm_step_max_error=reconstruction,physical_turn_offset_rad=movement,
                trajectory=trajectory,local_recovery=probes,
                coarse_tangent_sensitivity=coarse_snapshots,candidate_gates=gates,
                passes_candidate_screen=all(gates.values()),seconds=time.monotonic()-started,
                interpretation='Exploratory heading-memory candidate screen in the original navigation LSTM. Time-refit decoding measures information only. A unit phase-coordinate gain on an advected family is algebraic and is not evidence of stable physical memory. No ring is imposed or declared from PCA.')
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    save(folder/f'context{context}.json',record)
    np.savez_compressed(folder/f'context{context}_states.npz',
                        theta=theta,scale=scale.numpy(),dark_input=dark_x.numpy(),
                        times=np.array(TIMES),states=np.stack([snapshots[t] for t in TIMES]),
                        tangents=np.stack([tangent_snapshots[t] for t in TIMES]),
                        initial_fine_states=full.numpy(),readout_center=decoder[0],readout_weight=decoder[1])
    row=trajectory['75']
    print(json.dumps(dict(condition=condition,seed=seed,context=context,seconds=record['seconds'],
                          radius_5s=row['scaled_radius_retention'],
                          fixed_heading_mae_5s=row['fixed_readout_movement_corrected']['heldout_mae_deg'],
                          refit_mae_5s=row['time_refit_information_only']['heldout_mae_deg'],
                          candidate=record['passes_candidate_screen'])),flush=True)
    return record


def plot_summary(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
    colors={'gc100':'#2878b8','pgc33':'#e66b2f'}
    for condition,label in [('gc100','Grid'),('pgc33','PGC + grid')]:
        rows=[r for r in records if r['condition']==condition]
        for j,r in enumerate(rows):
            time=np.array(TIMES)*r['step_seconds']; color=colors[condition]
            axes[0].plot(time,[r['trajectory'][str(t)]['scaled_radius_retention'] for t in TIMES],color=color,alpha=.6,label=label if j==0 else None)
            axes[1].plot(time,[r['trajectory'][str(t)]['fixed_readout_movement_corrected']['heldout_mae_deg'] for t in TIMES],color=color,alpha=.6)
            for s in SETTLES:
                p=r['local_recovery'][str(s)]['horizons']['15']
                axes[2].scatter(p['latent_tangent_total_gain']['median'],p['normal_transverse_gain']['median'],color=color,alpha=.65,s=22)
    axes[0].set(xlabel='Seconds in darkness',ylabel='Remaining family size',title='Different headings remain distinct?')
    axes[0].axhline(.2,color='.6',ls=':',lw=1);axes[0].legend(frameon=False)
    axes[1].set(xlabel='Seconds in darkness',ylabel='Heading error (degrees)',title='A fixed readout tracks the turn?')
    axes[1].axhline(90,color='.6',ls=':',lw=1)
    axes[2].set(xlabel='Along-curve gain',ylabel='Across-curve residual gain',title='One-second perturbation recovery')
    axes[2].axhline(.5,color='.6',ls=':',lw=1);axes[2].axvline(.5,color='.6',ls=':',lw=1)
    axes[2].set(xscale='log',yscale='log')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'heading_memory.png',dpi=180,bbox_inches='tight');fig.savefig(out/'heading_memory.pdf',bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2]);p.add_argument('--conditions',nargs='+',default=['gc100','pgc33'])
    p.add_argument('--contexts',type=int,nargs='+',default=[0,1]);p.add_argument('--angles',type=int,default=96)
    p.add_argument('--prepare',type=int,default=30);p.add_argument('--embed-device',default='mps')
    p.add_argument('--context-root',type=Path,default=ROOT/'results/original_goal_bearing_manifold/pilot')
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_heading_memory/original')
    p.add_argument('--model-manifest',type=Path,help='JSON list: condition, seed, checkpoint, config; use a separate output directory for continued agents')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    assert args.angles>=16 and args.angles%4==0
    assert args.prepare>=15 and args.out.resolve()!=ROOT
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    if args.model_manifest:
        models=json.loads(args.model_manifest.read_text())
        assert args.out.resolve()!=(ROOT/'results/lstm_heading_memory/original').resolve(),'Continued agents need a separate output directory'
    else:
        models=[dict(condition=c,seed=s,checkpoint=str(ROOT/f'models/{c}/{c}_lstm_s{s}_1000M.pt'),
                     config=str(ROOT/f'models/{c}/{c}_lstm_s{s}_1000M.json')) for c in args.conditions for s in args.seeds]
    protocol=dict(experiment='Original navigation LSTM heading memory during stationary turns in darkness',
        models=[dict(m,checkpoint_sha256=sha(m['checkpoint']),config_sha256=sha(m['config'])) for m in models],
        contexts=args.contexts,analysis_angles=args.angles,rendered_angles=2*args.angles,visible_updates=args.prepare,
        sample_updates=TIMES,local_recovery_start_updates=SETTLES,local_recovery_horizons=PROBE_TIMES,
        task='Actual native goal snapshot, position, room and initial LSTM state from each archive; heading alone spans a circle. Translation and rotation noise, image noise and initial velocity are zero. Both angular acceleration and turn rate use unchanged environment physics.',
        blackout='All-zero RGB through original CNN; native current and saved goal code stay constant. The same pure-turn action and zero reward enter every family member. Action 0 in even contexts, action 1 in odd contexts.',
        candidate_criteria='At5s family radius>=20%; frozen heldout phase readout tracks true heading with MAE<=30deg, phase coverage>=50%, amplitude>=.1 in>=80% of heldout angles. At5s the following1s tangent gain>=.5, normal residual<.5, ratio>2. These gates nominate a candidate for further invariance/causal tests; they do not define a proven ring.',
        statistics='Exploratory two-seed/two-context screen. Every context retained. No pilot seed-level significance claim.',
        geometry='Full(h,c), raw and h/c-block-RMS scaling; initial central-difference tangent checked against doubled rendering resolution, exact recurrent propagation thereafter. Normals projected against the evolving tangent.',
        readout='Even phases fit one ridge readout at blackout onset; odd phases remain held out. Fixed movement-corrected and initial-phase errors reported separately. Time-refit is information-only.',
        source_sha256={name:sha(ROOT/name) for name in ['scripts/lstm_heading_memory_probe.py','scripts/lstm_endogenous_curve_recovery.py','code/env.py','code/nets.py','code/codes.py']},
        context_sha256={f'{m["condition"]}_s{m["seed"]}/context{c}':sha(args.context_root/f'{m["condition"]}_s{m["seed"]}/context{c}_states.npz') for m in models for c in args.contexts})
    frozen=args.out/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==json.loads(json.dumps(protocol)),'Frozen protocol or source changed; choose a new output directory'
    else:save(frozen,protocol)
    if args.freeze_only:return
    records=[run_model_context(m,c,args) for m in models for c in args.contexts]
    save(args.out/'summary.json',dict(contexts=len(records),passes=sum(r['passes_candidate_screen'] for r in records),
                                    results=[{k:r[k] for k in ('condition','seed','context','candidate_gates','passes_candidate_screen')} for r in records]))
    plot_summary(records,args.out)


if __name__=='__main__':main()
