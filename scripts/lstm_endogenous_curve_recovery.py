"""Corrected local recovery screen in ORIGINAL navigation LSTMs.

Goal-message families relax under one common natural physical input. A nominal
normal kick may end on a shifted phase: measure its residual transverse to the
ADVECTED full-state curve separately from distance to the original trajectory.
No circle, recurrent weights, or memory decoder is fitted into the agent.
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
import lstm_goal_bearing_manifold as base
from refine_lstm_goal_bearing_manifold import elicited

SETTLE = (0, 15, 75, 150)
HORIZONS = (0, 1, 15, 75)
LONG = (0, 15, 75, 150, 450, 1500)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def advance(cell, drive, state, directions=None, drive_theta=None):
    """Exact LSTM step and differential; direction0 is the family derivative.

    All other columns perturb only recurrent state. drive_theta is added only
    to column0, so cue-family motion is kept separate from a latent perturbation
    with the sensory cue held fixed.
    """
    size = cell.hidden_size
    h, c = state[:, :size], state[:, size:]
    gates = drive + torch.nn.functional.linear(h, cell.weight_hh, cell.bias_hh)
    ii, ff, gg, oo = gates.chunk(4, 1)
    i, f, g, o = ii.sigmoid(), ff.sigmoid(), gg.tanh(), oo.sigmoid()
    new_c = f*c + i*g
    tc = new_c.tanh()
    new_h = o*tc
    out = torch.cat((new_h, new_c), 1)
    if directions is None:
        return out
    d = torch.einsum('od,bdk->bok', cell.weight_hh, directions[:, :size])
    if drive_theta is not None:
        d[:, :, 0] += drive_theta
    di, df, dg, do = d.chunk(4, 1)
    di = di*(i*(1-i))[:, :, None]
    df = df*(f*(1-f))[:, :, None]
    dg = dg*(1-g.square())[:, :, None]
    do = do*(o*(1-o))[:, :, None]
    dc = f[:, :, None]*directions[:, size:] + c[:, :, None]*df + g[:, :, None]*di + i[:, :, None]*dg
    dh = tc[:, :, None]*do + (o*(1-tc.square()))[:, :, None]*dc
    return out, torch.cat((dh, dc), 1)


def decomposition(direction, tangent):
    """Full-state local phase displacement and orthogonal residual."""
    theta = (direction*tangent).sum(-1)/tangent.square().sum(-1).clamp_min(1e-40)
    normal = direction-tangent*theta[:, None]
    return theta, normal.norm(dim=-1)


def describe(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)))


def family_metrics(state, tangent, scale, initial_radius, drive, cell):
    x = state/scale
    radius = float((x-x.mean(0)).square().mean().sqrt())
    displacement = (advance(cell, drive, state)-state)/scale
    speed = displacement.square().mean(1).sqrt()
    phase_speed, normal_speed = decomposition(displacement, tangent/scale)
    separation = torch.pdist(x).div(np.sqrt(x.shape[1]))
    singular = torch.linalg.svdvals(x-x.mean(0))
    variance = singular.square()
    fractions = variance/variance.sum().clamp_min(1e-40)
    neighbors = (x-x.roll(1,0)).norm(dim=1)/np.sqrt(x.shape[1])
    return dict(relative_rms_family_radius=radius, radius_retention=radius/initial_radius,
                step_speed_relative_rms=describe(speed.numpy()),
                pairwise_distance_relative_rms=describe(separation.numpy()),
                angular_tangent_norm=describe((tangent/scale).norm(dim=1).numpy()),
                covariance_participation_ratio=float(1/fractions.square().sum()),
                first_five_pc_fractions=fractions[:5].tolist(),
                ordered_neighbor_max_over_median=float(neighbors.max()/neighbors.median().clamp_min(1e-40)),
                local_phase_speed_radians_per_step=describe(phase_speed.numpy()),
                local_normal_drift_relative_to_family_radius=describe((normal_speed/np.sqrt(x.shape[1])/max(radius,1e-30)).numpy()))


def probe(cell, start, tangent, scale, drive, drive_theta, rng_seed):
    n, dims = start.shape
    t = tangent/scale
    t_unit = t/t.norm(dim=1, keepdim=True).clamp_min(1e-30)
    gen = torch.Generator().manual_seed(rng_seed)
    normals = []
    for _ in range(3):
        q = torch.randn(n, dims, generator=gen, dtype=start.dtype)
        q -= (q*t_unit).sum(1, keepdim=True)*t_unit
        normals.append(q/q.norm(dim=1, keepdim=True))
    directions = torch.stack((tangent, t_unit*scale, *[q*scale for q in normals]), -1)
    initial_directions = directions.clone()
    z = start.clone()
    rows = {}
    for step in range(max(HORIZONS)+1):
        if step in HORIZONS:
            ref = directions[:, :, 0]/scale
            row = dict(tangent_curve_norm_retention=describe((ref.norm(dim=1)/t.norm(dim=1).clamp_min(1e-30)).numpy()))
            normal_total, normal_transverse, normal_phase = [], [], []
            for k in range(1, 5):
                perturb = directions[:, :, k]/scale
                phase, transverse = decomposition(perturb, ref)
                if k == 1:
                    initial_phase = 1/t.norm(dim=1).clamp_min(1e-30)
                    row['latent_tangent_total_gain'] = describe(perturb.norm(dim=1).numpy())
                    row['latent_tangent_phase_gain'] = describe((phase/initial_phase).numpy())
                    row['latent_tangent_transverse_gain'] = describe(transverse.numpy())
                else:
                    normal_total.append(perturb.norm(dim=1).numpy())
                    normal_transverse.append(transverse.numpy())
                    normal_phase.append(phase.abs().numpy())
            row['normal_total_gain_old_metric'] = describe(np.array(normal_total))
            row['normal_transverse_gain_corrected'] = describe(np.array(normal_transverse))
            row['normal_induced_phase_per_unit_state'] = describe(np.array(normal_phase))
            row['tangent_over_normal_transverse'] = row['latent_tangent_total_gain']['median']/max(row['normal_transverse_gain_corrected']['median'], 1e-30)
            rows[str(step)] = row
        if step < max(HORIZONS):
            z, directions = advance(cell, drive, z, directions, drive_theta)
    # Independent finite-amplitude check of the recurrent differential, one
    # phase and all four directions, at the same fixed cue used above.
    index = 0
    epsilon = 1e-4
    delta = initial_directions[index, :, 1:].T
    plus = start[index:index+1] + epsilon*delta
    minus = start[index:index+1] - epsilon*delta
    for _ in range(max(HORIZONS)):
        plus = advance(cell, drive[index:index+1], plus)
        minus = advance(cell, drive[index:index+1], minus)
    fd = (plus-minus)/(2*epsilon)
    analytic = directions[index, :, 1:].T
    fd_error = float((fd-analytic).norm()/analytic.norm().clamp_min(1e-30))
    assert fd_error < .005, fd_error
    return dict(horizons=rows, finite_difference_relative_error=fd_error)


def run_context(condition, seed, context, args):
    started = time.monotonic()
    cfg, checkpoint, net, pop = base.load_model(condition, seed, 'cpu')
    net.double()
    source = ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context{context}_states.npz'
    a = np.load(source)
    inputs = torch.tensor(a['history_x'][:15], dtype=torch.float64)
    initial = torch.tensor(np.r_[a['history_h_before'][0], a['history_c_before'][0]], dtype=torch.float64)
    observer = torch.tensor(a['history_pos'][14], dtype=torch.float64)
    theta = torch.arange(args.angles, dtype=torch.float64)*(2*np.pi/args.angles)
    fun = lambda angle: elicited(net, pop, inputs, initial, observer, angle, 0., False)
    (state, goal), (tangent, goal_theta) = torch.autograd.functional.jvp(fun, theta, torch.ones_like(theta))
    state, tangent, goal, goal_theta = [v.detach() for v in (state,tangent,goal,goal_theta)]
    # Validate hand-expanded actual LSTM step, including recurrent derivative.
    cell = net.lstm
    x = inputs[-1:].repeat(args.angles, 1)
    x[:,768:1280] = 0
    drive = torch.nn.functional.linear(x, cell.weight_ih, cell.bias_ih)
    h, c = cell(x, (state[:,:256],state[:,256:]))
    reconstruction = float((advance(cell,drive,state)-torch.cat((h,c),1)).abs().max())
    assert reconstruction < 1e-12
    scales = torch.stack((state[:,:256].square().mean().sqrt(),state[:,256:].square().mean().sqrt()))
    scale = scales.repeat_interleave(256)
    initial_radius = float(((state-state.mean(0))/scale).square().mean().sqrt())
    snapshots = {}
    trajectory = {}
    with torch.no_grad():
        for step in range(max(LONG)+1):
            if step in LONG:
                trajectory[str(step)] = family_metrics(state,tangent,scale,initial_radius,drive,cell)
            if step in SETTLE:
                snapshots[step] = (state.clone(), tangent.clone())
            if step < max(LONG):
                state, dt = advance(cell,drive,state,tangent[:,:,None]); tangent=dt[:,:,0]
        goal_conditions = {'goal_withheld':(torch.zeros_like(goal),torch.zeros_like(goal_theta)), 'native_goal':(goal,goal_theta)}
        if condition == 'pgc33':
            fun_gc = lambda angle: elicited(net,pop,inputs,initial,observer,angle,0.,True)[1]
            # requires autograd even inside the surrounding analysis block
            with torch.enable_grad():
                gc, dgc = torch.autograd.functional.jvp(fun_gc,theta,torch.ones_like(theta))
            goal_conditions['ordinary_goal'] = (gc.detach(), dgc.detach())
        probes = {}
        for settle, (z,tan) in snapshots.items():
            probes[str(settle)] = {}
            for name,(cue,dcue) in goal_conditions.items():
                xx=x.clone();xx[:,768:1280]=cue
                dd=torch.nn.functional.linear(xx,cell.weight_ih,cell.bias_ih)
                dx=torch.nn.functional.linear(dcue.double(),cell.weight_ih[:,768:1280],None)
                probes[str(settle)][name]=probe(cell,z,tan,scale,dd,dx,190731+context+seed*100)
    screen = probes['75']['goal_withheld']['horizons']['15']
    gates = dict(family_radius_after_5s_at_least_20_percent=trajectory['75']['radius_retention']>=.2,
        one_second_latent_tangent_median_at_least_half=screen['latent_tangent_total_gain']['median']>=.5,
        one_second_transverse_median_below_half=screen['normal_transverse_gain_corrected']['median']<.5,
        tangent_over_transverse_median_greater_than_two=screen['tangent_over_normal_transverse']>2)
    result=dict(condition=condition,seed=seed,context=context,angles=args.angles,
        checkpoint_sha256=sha(checkpoint),native_context_archive_sha256=sha(source),
        source_sha256=sha(__file__),torch_version=str(torch.__version__),device='cpu_float64',
        step_seconds=1/15,block_rms_scale=scales.tolist(),native_step_error=reconstruction,
        common_input='Actual endpoint visual embedding, current code, previous action and zero reward; saved goal slots zero. Repeated input is a dynamical intervention, not a closed-loop trajectory.',
        trajectory=trajectory,local_recovery=probes,candidate_gates=gates,passes_candidate_screen=all(gates.values()),seconds=time.monotonic()-started,
        interpretation='The time-advected curve is not assumed invariant. Family radius, tangent norm and absolute drift remain necessary gates before any endogenous-ring interpretation. Corrected transverse residual removes induced phase displacement.')
    folder=args.out/f'{condition}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    save(folder/f'context{context}.json',result)
    print(json.dumps({'condition':condition,'seed':seed,'context':context,'seconds':result['seconds'],
        'radius_100s':trajectory['1500']['radius_retention'],
        '5s_settle_1s_blank':probes['75']['goal_withheld']['horizons']['15']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2]);p.add_argument('--conditions',nargs='+',default=['gc100','pgc33'])
    p.add_argument('--contexts',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--angles',type=int,default=96)
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/corrected_recovery')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(experiment='Original-checkpoint exploratory corrected-transverse-recovery screen',seeds=args.seeds,conditions=args.conditions,
        contexts=args.contexts,angles=args.angles,settle_updates=SETTLE,probe_updates=HORIZONS,long_updates=LONG,
        no_imposed_ring=True,initialization='Valid native goal-message family, all other actual history inputs shared; offset0 for the first mechanistic screen.',
        primary_screen='After5s common-input relaxation, compare1s latent-tangent total gain with residual transverse gain; retain full family size and all earlier/later times. No seed-level significance claim from pilot pairs.',
        candidate_followup='Advance to slow-point/spectral/invariance analysis if after5s relaxation family radius retains>=20% and over the next1s median latent tangent gain>=.5, corrected normal gain<.5, tangent/normal ratio>2. Retain every metric/context regardless; these are candidate-screen thresholds, not ring-proof criteria. A near-neutral exact continuum is not required; slow rings may contain fixed points and saddles.',
        sources={str(Path(__file__).relative_to(ROOT)):sha(__file__), 'scripts/lstm_goal_bearing_manifold.py':sha(ROOT/'scripts/lstm_goal_bearing_manifold.py'),
                 'scripts/refine_lstm_goal_bearing_manifold.py':sha(ROOT/'scripts/refine_lstm_goal_bearing_manifold.py'),'code/nets.py':sha(ROOT/'code/nets.py')})
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol)),'Protocol/source changed'
    else:save(path,protocol)
    if args.freeze_only:return
    for condition in args.conditions:
        for seed in args.seeds:
            for context in args.contexts:run_context(condition,seed,context,args)


if __name__=='__main__':main()
