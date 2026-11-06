"""Separate cue-dependent Jacobians from changes in the recurrent operating state.

The recurrent weights stay fixed. Compare native, ordinary and absent inputs at
the same cue-free states and at the native cue-anchored states. Phase gains use
the full-state empirical tangent and exact derivative of the trained readout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_heading_memory import distribution, eigenspace_alignment
from evaluate_ring_dynamics import jacobian
from ring_dynamics import tangents, wrap
from ring_heading_model import load_checkpoint


def phase_gradient(model, state):
    y = model.readout(state)
    # readout is affine, so its bias affects the phase denominator but not dy/dh.
    numerator = y[:, :1]*model.readout.weight[1] - y[:, 1:2]*model.readout.weight[0]
    return numerator/y.square().sum(-1, keepdim=True).clamp_min(1e-20)


def one_state_spectrum(model, state, x, tangent):
    matrices = jacobian(model, state, x).detach().cpu().numpy()
    next_state = model.step(x, state)
    before_amplitude = model.readout(state).norm(dim=-1).detach().cpu().numpy()
    after_amplitude = model.readout(next_state).norm(dim=-1).detach().cpu().numpy()
    grad_before = phase_gradient(model, state).detach().cpu().numpy()
    grad_after = phase_gradient(model, next_state).detach().cpu().numpy()
    denominator = np.einsum("bi,bi->b", grad_before, tangent)
    numerator = np.einsum("bi,bij,bj->b", grad_after, matrices, tangent)
    valid = (np.abs(denominator) > 1e-5) & (before_amplitude > 1e-4) & (after_amplitude > 1e-4)
    gains = np.full(len(state), np.nan)
    gains[valid] = numerator[valid]/denominator[valid]
    rows, eigenvalues = [], []
    for j, tan, gain, amp0, amp1 in zip(matrices, tangent, gains, before_amplitude, after_amplitude):
        eigen, vectors = np.linalg.eig(j)
        slow = int(np.argmin(np.abs(eigen-1)))
        dominant = int(np.argmax(np.abs(eigen)))
        normal = np.linalg.qr(tan[:, None], mode="complete")[0][:, 1:]
        reduced = normal.T@j@normal
        rows.append(dict(phase_gain=float(gain), spectral_radius=float(np.abs(eigen).max()),
                         readout_amplitude_before=float(amp0), readout_amplitude_after=float(amp1),
                         slow_eigenvalue_real=float(eigen[slow].real), slow_eigenvalue_imag=float(eigen[slow].imag),
                         slow_mode_tangent_alignment=eigenspace_alignment(vectors[:, slow], tan),
                         dominant_mode_tangent_alignment=eigenspace_alignment(vectors[:, dominant], tan),
                         tangent_euclidean_gain=float(np.linalg.norm(j@tan)),
                         normal_largest_singular_value=float(np.linalg.svd(reduced, compute_uv=False)[0]),
                         largest_singular_value=float(np.linalg.svd(j, compute_uv=False)[0])))
        eigenvalues.append(eigen)
    return {"per_phase": rows, "distributions": {k: distribution([r[k] for r in rows]) for k in rows[0]},
            "phase_gain_resolvable_fraction": float(valid.mean())}, np.asarray(eigenvalues)


@torch.no_grad()
def evaluate(checkpoint, output, n_phase=72):
    model, task, ckpt = load_checkpoint(checkpoint, "cpu")
    model = model.double()
    model.requires_grad_(False)
    phase = torch.arange(n_phase)*2*np.pi/n_phase
    direction = torch.stack((phase.cos(), phase.sin()), -1)
    zeros, ones = torch.zeros(n_phase), torch.ones(n_phase)
    archive, rows, checks = {}, [], []
    for context, xy in enumerate(((-.8,-.8),(-.8,.8),(.8,-.8),(.8,.8),(0.,0.),(10.,10.))):
        position = torch.tensor(xy).expand(n_phase, 2)
        def inputs(condition, initial=False):
            return task.inputs(position, direction, zeros, zeros, ones, ones if initial else zeros, condition).double()
        cue = inputs(ckpt["condition"], initial=True)
        state = torch.zeros(n_phase, model.hidden_size, dtype=torch.float64)
        for _ in range(4):
            state = model.step(cue, state)
        blank = torch.zeros_like(cue)
        for _ in range(40):
            state = model.step(blank, state)
        current = {"blank": blank, "native": inputs(ckpt["condition"]), "ordinary": inputs("grid")}
        references = {"cue_free": state.clone()}
        for _ in range(200):
            state = model.step(current["native"], state)
        references["native_anchored"] = state
        for reference, state in references.items():
            curve = state.numpy()
            tangent = tangents(curve)
            y = model.readout(state)
            angles = torch.atan2(y[:,1], y[:,0]).numpy()
            sorted_angles = np.sort(angles%(2*np.pi))
            coverage = float(1-np.diff(np.r_[sorted_angles,sorted_angles[0]+2*np.pi]).max()/(2*np.pi))
            for variant, x in current.items():
                report, eigenvalues = one_state_spectrum(model, state, x, tangent)
                report.update(context=context, position=list(xy), reference_state=reference, input=variant,
                              baseline_heading_mae_deg=float(np.abs(wrap(angles-phase.numpy())).mean()*180/np.pi),
                              baseline_circle_coverage=coverage)
                rows.append(report)
                archive[f"context{context}_{reference}_{variant}_eigenvalues"] = eigenvalues
                # Autograd and central finite differences independently validate
                # the analytic Jacobian and circular-readout directional gain.
                if context == 0:
                    h, obs = state[0].clone(), x[0]
                    with torch.enable_grad():
                        auto = torch.autograd.functional.jacobian(lambda s: model.step(obs, s), h)
                    analytic = jacobian(model, state[:1], x[:1])[0]
                    jac_error = float((auto-analytic).abs().max())
                    tan = torch.tensor(tangent[0])
                    eps = 1e-5
                    def decoded(s):
                        out = model.readout(s)
                        return float(torch.atan2(out[1], out[0]))
                    before = wrap(decoded(h+eps*tan)-decoded(h-eps*tan))
                    after = wrap(decoded(model.step(obs,h+eps*tan))-decoded(model.step(obs,h-eps*tan)))
                    finite = after/before
                    analytic_gain = report["per_phase"][0]["phase_gain"]
                    gain_error = abs(float(finite)-analytic_gain)
                    assert jac_error < 1e-10 and gain_error < 1e-5
                    checks.append(dict(reference=reference,input=variant,jacobian_max_error=jac_error,
                                       phase_gain_finite_difference_error=gain_error))
            archive[f"context{context}_{reference}_states"] = curve
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=ckpt["seed"],
                  condition=ckpt["condition"], config=ckpt["config"], rows=rows, numerical_checks=checks,
                  protocol=dict(n_phase=n_phase, initial_cue_steps=4, cue_free_steps=40, anchor_steps=200,
                      same_state_input_counterfactuals=True, recurrent_weights_changed=False,
                      primary_reference="cue_free", secondary_reference="native_anchored",
                      phase_gain="grad phase at next state * J(input,state) * empirical tangent / grad phase at current state * tangent",
                      note="Instantaneous eigenmodes at stated operating points; finite-time recovery is assessed separately."))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(output.with_suffix(".npz"), **archive)
    summary = {}
    for reference in references:
        for variant in current:
            selected = [r for r in rows if r["context"] < 5 and r["reference_state"] == reference and r["input"] == variant]
            summary[reference+"_"+variant] = {key:float(np.mean([r["distributions"][key]["median"] for r in selected]))
                                            for key in ("phase_gain","spectral_radius","slow_mode_tangent_alignment")}
    print(json.dumps({"seed":ckpt["seed"],"summary":summary}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate(args.checkpoint, args.out)
