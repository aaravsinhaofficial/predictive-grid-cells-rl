"""Known-equation tests for independent drift fitting and unfitted interventions."""
import json

import numpy as np

from evaluate_heading_phase_reduction import basis, fit_reduction, predict_velocity


def main():
    rng = np.random.default_rng(213)
    phi, cue = rng.uniform(-np.pi, np.pi, (2, 800))
    f0_coefficients = np.array([.03, .01, .02, -.02, .01, .015, -.025])
    k = 2.75
    f0 = basis(phi) @ f0_coefficients
    row = {"phi": phi, "cue": cue, "blank": f0, "native": f0 + k * np.sin(cue - phi)}
    fit = fit_reduction([row])
    assert np.allclose(fit["f0_coefficients"], f0_coefficients)
    assert abs(fit["k_per_second"] - k) < 1e-10
    # New phases, so coefficient recovery and out-of-sample predictions differ.
    phi, cue = rng.uniform(-np.pi, np.pi, (2, 800))
    f0 = basis(phi) @ f0_coefficients
    expected = {"native": f0 + k * np.sin(cue - phi), "reverse": f0 - k * np.sin(cue - phi),
                "blank": f0, "ordinary": f0, "self_generated": f0}
    for variant, actual in expected.items():
        predicted = predict_velocity(phi, cue, variant, "sine", fit, "predictive")
        assert np.allclose(predicted, actual), variant
    grid_native = predict_velocity(phi, cue, "native", "sine", fit, "grid")
    grid_reverse = predict_velocity(phi, cue, "reverse", "sine", fit, "grid")
    assert np.array_equal(grid_native, grid_reverse)
    assert np.array_equal(predict_velocity(phi, cue, "native", "constant", fit, "grid"), np.zeros_like(phi))
    print(json.dumps({"passed": True, "checks": ["known_coefficient_recovery", "independent_blank_drift",
        "heldout_native_and_unfitted_reversal", "no_external_anchor_for_self_generated", "grid_reversal_noop"]}))


if __name__ == "__main__":
    main()
