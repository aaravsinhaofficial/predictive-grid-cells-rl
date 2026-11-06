"""Bounded analytic root search for ORIGINAL native LSTM equilibrium branches.

Unlike the earlier Adam/LBFGS residual minimization, this solves the exact
256-dimensional consistency equation with its analytic Jacobian, diverse starts,
and a preassigned subset deflated against already known roots. No angle labels,
circle objective, network training, or change to the eight fixed inputs is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.linalg import eig
from scipy.optimize import root
from scipy.special import expit
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite_float(value):
    return float(value) if np.isfinite(value) else None


class EliminatedLSTM:
    def __init__(self, weight_hh, drive, actor=None, actor_bias=None):
        self.w = np.asarray(weight_hh, dtype=np.float64)
        self.drive = np.asarray(drive, dtype=np.float64).reshape(-1)
        self.n = self.w.shape[1]
        self.wi, self.wf, self.wg, self.wo = np.split(self.w, 4)
        self.actor = actor
        self.actor_bias = actor_bias

    def gates(self, h):
        ai, af, ag, ao = np.split(self.drive + self.w @ h, 4)
        i, f, g, o = expit(ai), expit(af), np.tanh(ag), expit(ao)
        return i, f, g, o, expit(-af), i * expit(-ai), o * expit(-ao), 4 * expit(2 * ag) * expit(-2 * ag)

    def eliminated(self, h):
        i, f, g, o, d, di, do, dg = self.gates(h)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            c = i * g / d
            tc = np.tanh(c)
            sech = 4 * expit(2 * c) * expit(-2 * c)
            # Multiply sech before division to avoid inf*0 for saturated c.
            first = np.divide(o * sech, d, out=np.zeros_like(d), where=d > 0)
            dc_numerator = (g * di)[:, None] * self.wi + (i * dg)[:, None] * self.wg
            jac = first[:, None] * dc_numerator + (o * sech * c * f)[:, None] * self.wf
            jac += (tc * do)[:, None] * self.wo
            jac -= np.eye(self.n)
        return o * tc - h, jac, c, d

    def advance(self, z):
        z = np.asarray(z, dtype=np.float64)
        one = z.ndim == 1
        z = np.atleast_2d(z)
        gates = z[:, :self.n] @ self.w.T + self.drive
        ai, af, ag, ao = np.split(gates, 4, axis=1)
        cn = expit(af) * z[:, self.n:] + expit(ai) * np.tanh(ag)
        hn = expit(ao) * np.tanh(cn)
        answer = np.concatenate((hn, cn), axis=1)
        return answer[0] if one else answer

    def stable_residual(self, z):
        h, c = np.split(z, 2)
        i, f, g, o, d, *_ = self.gates(h)
        dc = i * g - d * c
        # c + dc avoids the cancellation in (f*c+ig)-c.
        return np.concatenate((o * np.tanh(c + dc) - h, dc))

    def full_jacobian(self, z):
        h, c = np.split(z, 2)
        i, f, g, o, d, di, do, dg = self.gates(h)
        cn = f * c + i * g
        tc = np.tanh(cn)
        sech = 4 * expit(2 * cn) * expit(-2 * cn)
        ch = ((c * d) * f)[:, None] * self.wf + (g * di)[:, None] * self.wi + (i * dg)[:, None] * self.wg
        hh = (o * sech)[:, None] * ch + (tc * do)[:, None] * self.wo
        return np.block([[hh, np.diag(o * sech * f)], [ch, np.diag(f)]])


def qualify(model, h, tolerance=1e-8):
    residual, _, c, d = model.eliminated(h)
    z = np.concatenate((h, c))
    stable = model.stable_residual(z)
    hr = float(np.sqrt(np.mean(residual ** 2)))
    sr = float(np.sqrt(np.mean(stable ** 2)))
    finite = bool(np.isfinite(z).all() and np.isfinite(stable).all())
    bounded = bool(np.max(np.abs(h)) <= 1 + 1e-9)
    return dict(qualified=finite and bounded and hr <= tolerance and sr <= tolerance,
                finite=finite, hidden_bounds_valid=bounded, hidden_residual_rms=finite_float(hr),
                stable_full_residual_rms=finite_float(sr), max_abs_h=finite_float(np.max(np.abs(h))),
                max_abs_c=finite_float(np.max(np.abs(c))), min_one_minus_f=finite_float(d.min())), z


def unique_hidden(rows, tolerance=1e-5):
    selected = []
    for i, row in enumerate(rows):
        if not selected or min(np.sqrt(np.mean((row - rows[j]) ** 2)) for j in selected) > tolerance:
            selected.append(i)
    return selected


def deflated_residual(model, h, known):
    residual, jac, _, _ = model.eliminated(h)
    if len(known) == 0:
        return residual, jac
    delta = h[None, :] - known
    distance2 = np.mean(delta ** 2, axis=1) + 1e-12
    log_multiplier = np.log1p(1 / distance2).sum()
    multiplier = np.exp(min(log_multiplier, 200))
    gradient_log = np.sum(-2 * delta / (model.n * distance2 * (1 + distance2))[:, None], axis=0)
    if log_multiplier >= 200:
        gradient_log *= 0
    return multiplier * residual, multiplier * (jac + np.outer(residual, gradient_log))


class EvaluationLimit(Exception):
    pass


def solve_start(model, initial, known, deflate, max_evaluations):
    calls = 0
    jac_calls = 0
    last = initial.copy()
    best = initial.copy()
    best_rms = np.inf
    history = []
    cache_h = None
    cache = None

    def evaluate(h):
        nonlocal cache_h, cache
        if cache_h is None or not np.array_equal(h, cache_h):
            cache_h = h.copy()
            cache = deflated_residual(model, h, known) if deflate else model.eliminated(h)[:2]
        return cache

    def fun(h):
        nonlocal calls, last, best, best_rms
        if calls >= max_evaluations:
            raise EvaluationLimit
        calls += 1
        value = evaluate(h)[0]
        raw = model.eliminated(h)[0] if deflate else value
        rms = float(np.sqrt(np.mean(raw ** 2)))
        history.append([calls, finite_float(rms), finite_float(np.max(np.abs(h)))])
        if np.isfinite(value).all():
            last = h.copy()
            if rms < best_rms:
                best, best_rms = h.copy(), rms
        else:
            raise FloatingPointError('Nonfinite residual')
        return value

    def jac(h):
        nonlocal jac_calls
        jac_calls += 1
        value = evaluate(h)[1]
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite derivative')
        return value

    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = root(fun, initial, jac=jac, method='hybr', options=dict(xtol=1e-10, maxfev=max_evaluations, factor=.25))
        candidate = result.x
        status, success, message = int(result.status), bool(result.success), str(result.message)
    except (EvaluationLimit, FloatingPointError) as error:
        candidate = best
        status, success, message = -1, False, type(error).__name__
    check, z = qualify(model, candidate)
    best_check, best_z = qualify(model, best)
    # Retain both endpoint and best residual; qualification never uses scipy status.
    chosen = best if best_check['qualified'] and not check['qualified'] else candidate
    chosen_check, chosen_z = qualify(model, chosen)
    record = dict(deflated=bool(deflate), nfev=calls, njev=jac_calls, solver_status=status,
                  solver_success=success, solver_message=message, elapsed_seconds=time.perf_counter()-start,
                  endpoint=check, best=best_check, selected=chosen_check, evaluation_history=history)
    return record, chosen_z, candidate, best


def initializers(native, known, seed):
    rng = np.random.default_rng(seed)
    n = native.shape[1]
    rows = [h.copy() for h in native[np.linspace(0, len(native)-1, 16, dtype=int)]]
    labels = ['native'] * 16
    rows.append(np.zeros(n)); labels.append('zero')
    for _ in range(23):
        rows.append(rng.uniform(-1, 1, n)); labels.append('uniform')
    centers = known if len(known) else native[:1]
    for k in range(24):
        scale = (.03, .1, .3)[k % 3]
        rows.append(np.clip(centers[k % len(centers)] + scale * rng.normal(size=n), -.999999999, .999999999))
        labels.append(f'known_root_noise_{scale}')
    return np.stack(rows), labels


def load_case(condition, seed, regime):
    checkpoint = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['frames'] == 1_000_000_000
    weights = {k: v.detach().double().numpy() for k, v in saved['policy'].items()}
    source = ROOT / f'results/lstm_endogenous_ring/fixed_points/{condition}_s{seed}/context0_{regime}.npz'
    prior = ROOT / f'results/lstm_endogenous_ring/equilibrium_elimination/{condition}_s{seed}/context0_{regime}.npz'
    with np.load(source, allow_pickle=False) as a:
        native, x = a['initial'].copy(), a['input'][0].copy()
    with np.load(prior, allow_pickle=False) as a:
        previous = a['solution'].copy()
    drive = weights['lstm.weight_ih'] @ x + weights['lstm.bias_ih'] + weights['lstm.bias_hh']
    model = EliminatedLSTM(weights['lstm.weight_hh'], drive, weights['pi.weight'], weights['pi.bias'])
    qualified = [z[:model.n] for z in previous if qualify(model, z[:model.n])[0]['qualified']]
    qualified = np.asarray(qualified).reshape(-1, model.n)
    known = qualified[unique_hidden(qualified)]
    return model, native, known, dict(checkpoint_sha256=sha(checkpoint), initializer_sha256=sha(source),
                                     prior_solution_sha256=sha(prior), checkpoint=str(checkpoint.relative_to(ROOT)),
                                     initializer=str(source.relative_to(ROOT)), prior_solution=str(prior.relative_to(ROOT)))


def validate_root(model, z, native_snapshots, known):
    check, _ = qualify(model, z[:model.n])
    jac = model.full_jacobian(z)
    values, vectors = eig(jac, check_finite=False)
    order = np.argsort(np.abs(values))[::-1]
    values, vectors = values[order], vectors[:, order]
    sizes = np.linalg.norm(vectors, axis=0)
    hfrac = np.linalg.norm(vectors[:model.n], axis=0) / sizes
    gain = np.linalg.norm(model.actor @ vectors[:model.n], axis=0) / sizes
    unstable = np.abs(values) > 1 + 1e-6
    # Direct derivative check in four fixed directions, scaled to h/c units.
    rng = np.random.default_rng(601)
    directions = rng.normal(size=(4, 2*model.n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    eps = 1e-5
    # Stable residual's derivative is J-I and avoids subtracting large c values.
    errors = []
    for v in directions:
        observed = (model.stable_residual(z + eps*v) - model.stable_residual(z - eps*v))/(2*eps)
        predicted = (jac - np.eye(2*model.n)) @ v
        errors.append(float(np.linalg.norm(observed-predicted)/max(np.linalg.norm(predicted), 1e-12)))
    reach = {}
    logits = model.actor @ z[:model.n] + model.actor_bias
    for t, native in native_snapshots.items():
        hd = np.sqrt(np.mean((native[:, :model.n] - z[:model.n])**2, axis=1))
        cd = np.sqrt(np.mean((native[:, model.n:] - z[model.n:])**2, axis=1))
        ld = np.sqrt(np.mean((native[:, :model.n] @ model.actor.T + model.actor_bias - logits)**2, axis=1))
        reach[str(t)] = dict(hidden_rms_min=float(hd.min()), hidden_rms_median=float(np.median(hd)),
                            cell_rms_min=float(cd.min()), cell_rms_median=float(np.median(cd)),
                            actor_logit_rms_min=float(ld.min()), actor_logit_rms_median=float(np.median(ld)),
                            native_max_abs_c=float(np.abs(native[:, model.n:]).max()))
    d = model.gates(z[:model.n])[4]
    check.update(spectral_radius=float(np.abs(values).max()), unstable_count=int(unstable.sum()),
                 near_unit_count=int((np.abs(np.abs(values)-1) <= .01).sum()),
                 eigenvalues=np.column_stack((values.real, values.imag)).tolist(), hidden_fraction=hfrac.tolist(),
                 actor_logit_gain=gain.tolist(), unstable_hidden_fraction=hfrac[unstable].tolist(),
                 unstable_actor_logit_gain=gain[unstable].tolist(), jacobian_directional_relative_errors=errors,
                 actor_logits=logits.tolist(), native_forward_reachability=reach,
                 nearest_prior_hidden_rms=float(cdist(z[None, :model.n], known).min()/np.sqrt(model.n)) if len(known) else None,
                 max_cell_relaxation_seconds=finite_float(1/d.min()/15),
                 cell_scale_over_initial_native=finite_float(np.max(np.abs(z[model.n:]))/max(np.max(np.abs(native_snapshots[0][:, model.n:])),1e-12)))
    return check, values, vectors


def run_case(condition, seed, regime, args):
    folder = args.out / f'{condition}_s{seed}' / f'context0_{regime}'
    if (folder/'summary.json').exists():
        print(json.dumps(dict(case=str(folder.relative_to(args.out)), skipped_complete=True)), flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    model, native, known, provenance = load_case(condition, seed, regime)
    rng_seed = 2026110600 + 100*(condition=='pgc33') + 10*seed + (regime=='actual_goal_held')
    starts, labels = initializers(native[:, :model.n], known, rng_seed)
    np.savez_compressed(folder/'inputs.npz', initial=starts, native=native, known=known, drive=model.drive)
    outcomes, solutions, endpoints, bests = [], [], [], []
    began = time.perf_counter()
    for index, h in enumerate(starts):
        # Deflation schedule is locked before solving; no selection by result.
        deflate = index >= 17 and index % 2 == 1
        result, z, endpoint, best = solve_start(model, h, known, deflate, args.max_evaluations)
        result.update(index=index, initializer_kind=labels[index])
        outcomes.append(result); solutions.append(z); endpoints.append(endpoint); bests.append(best)
        if (index+1) % 8 == 0:
            save(folder/'outcomes.json', outcomes)
            print(json.dumps(dict(condition=condition, seed=seed, regime=regime, starts=index+1,
                                  qualified=sum(r['selected']['qualified'] for r in outcomes), seconds=time.perf_counter()-began)), flush=True)
    solutions = np.stack(solutions)
    np.savez_compressed(folder/'all_starts.npz', initial=starts, selected_solution=solutions,
                        endpoints=np.stack(endpoints), best_hidden=np.stack(bests))
    save(folder/'outcomes.json', outcomes)
    accepted = [z for result,z in zip(outcomes,solutions) if result['selected']['qualified']]
    previous_z = [qualify(model,h)[1] for h in known]
    combined = np.asarray(previous_z + accepted).reshape(-1, 2*model.n)
    representatives = unique_hidden(combined[:, :model.n])
    roots = combined[representatives]
    snapshots = {0:native.copy()}
    evolving = native.copy()
    for t in range(1,1501):
        evolving = model.advance(evolving)
        if t in (60,150,1500): snapshots[t] = evolving.copy()
    records, eigenvalues, eigenvectors = [], [], []
    for index,z in enumerate(roots):
        record, values, vectors = validate_root(model,z,snapshots,known)
        record['root_index'] = index
        record['is_new_hidden_branch'] = record['nearest_prior_hidden_rms'] is None or record['nearest_prior_hidden_rms'] > 1e-5
        record['start_indices'] = [i for i,(r,s) in enumerate(zip(outcomes,solutions)) if r['selected']['qualified'] and np.sqrt(np.mean((s[:model.n]-z[:model.n])**2)) <= 1e-5]
        records.append(record); eigenvalues.append(values); eigenvectors.append(vectors)
    np.savez_compressed(folder/'roots.npz', roots=roots, eigenvalues=np.asarray(eigenvalues), eigenvectors=np.asarray(eigenvectors),
                        **{f'native_t{t}':z for t,z in snapshots.items()})
    summary = dict(condition=condition, seed=seed, context=0, regime=regime, source_sha256=sha(__file__), **provenance,
                   rng_seed=rng_seed, starts=64, max_evaluations=args.max_evaluations, prior_unique_root_count=len(known),
                   qualified_start_count=sum(r['selected']['qualified'] for r in outcomes), unique_root_count=len(roots),
                   new_root_count=sum(r['is_new_hidden_branch'] for r in records),
                   unstable_root_count=sum(r['unstable_count']>0 for r in records), roots=records,
                   elapsed_seconds=time.perf_counter()-began,
                   interpretation='Bounded root search only. Mathematical equilibria, spectral instability, and native accessibility are reported separately. No ring criterion or label-fitting objective enters the search.')
    save(folder/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('condition','seed','regime','unique_root_count','new_root_count','unstable_root_count','elapsed_seconds')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/ring_nonlinear_design/native_equilibria')
    parser.add_argument('--max-evaluations',type=int,default=200)
    parser.add_argument('--condition',choices=['gc100','pgc33'])
    parser.add_argument('--seed',type=int,choices=[1,2])
    parser.add_argument('--regime',choices=['goal_withheld','actual_goal_held'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    protocol=dict(source_sha256=sha(__file__),conditions=['gc100','pgc33'],seeds=[1,2],context=0,
                  regimes=['goal_withheld','actual_goal_held'],starts_per_case=64,max_function_evaluations=args.max_evaluations,
                  method='Analytic exact-elimination Jacobian; scipy hybr Newton with factor0.25 and xtol1e-10. Fixed alternating deflation schedule after native/zero starts; prior roots only.',
                  initializers='16 evenly spaced native h;1zero;23uniform[-1,1];24known-root Gaussian perturbations at componentSD0.03/0.1/0.3, clipped insidehbound.',
                  qualification='Hidden and stable original512D residualRMS<=1e-8; finitefullstate;|h|<=1+1e-9. ClusterhRMS1e-5. Revalidatepreservedpriorroots.',
                  stability='Correctfloat64fullstateJacobian;|multiplier|>1+1e-6unstable;hiddenfractionandactorlogitgain retained. Fourfixedstable-residualdirectionchecks perroot.',
                  reachability='Nativefullrecurrence0,60,150,1500updates;h,c,andactor distances separately. No native-c instantaneous equilibration.')
    args.out.mkdir(parents=True,exist_ok=True)
    p=args.out/'protocol.json'
    if p.exists():
        assert json.loads(p.read_text())==protocol, 'Existing protocol differs; use a new output directory.'
    else: save(p,protocol)
    with threadpool_limits(limits=1):
        for condition in [args.condition] if args.condition else protocol['conditions']:
            for seed in [args.seed] if args.seed else protocol['seeds']:
                for regime in [args.regime] if args.regime else protocol['regimes']:
                    run_case(condition,seed,regime,args)


if __name__=='__main__':
    main()
