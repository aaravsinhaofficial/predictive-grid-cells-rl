# A simple calculation inside the supplied predictive code

**Comparing ordinary and predictive grid phases reveals travel direction.** The comparison cancels the current position.

For one grid wave vector `k`, the ordinary population represents phase `k·x`. Predictive cells with the same grid spacing/orientation represent `k·(x + d u)`, where `d = 20 cm` and `u` is the unit travel direction. Subtracting the phases leaves

`predictive phase − ordinary phase = d k·u`.

Several nonparallel grid waves provide the two components of `u`. The current location `x` disappears from this calculation. In the original spatial-shift rule this gives **direction**, not speed or a guaranteed future trajectory.

## Implemented decoder

The decoder first learns linear population readouts of the sine and cosine of each grid phase from a separate calibration set. It then multiplies the predictive complex phase by the conjugate of the ordinary phase and solves the known wave-vector geometry. The comparison and geometric inversion are specified operations; they are not claimed to have been discovered inside the original LSTMs. Inference receives only the 512 firing rates.

The three coarser grid modules are used because a 20 cm displacement cannot wrap their phase beyond ±π. This avoids choosing an unwrapping rule using the answer. Cell identities, phases, orientations, module spacings, and predictive fraction match the original population generator.

Across all eight original population seeds, noiseless direction error is approximately **0.000006 degrees** on new samples and **0.00007 degrees** after moving the test region 10 m. These tiny errors reflect the exact analytic population model and floating-point precision; they are not expected biological accuracy. With independent rate noise of standard deviation 0.03 on rates ranging from 0 to 1, mean direction error is **0.56 degrees**.

The causal checks have the expected geometry:

- Replacing predictive cells with ordinary cells removes the displacement vector: its decoded length falls to approximately zero.
- Reversing the spatial preview reverses the decoded direction.
- Halving the preview halves the decoded displacement vector.

This explains a concrete capability supplied to an RNN: it can obtain a direction reference by comparing current and predictive spatial phases. The learned-heading experiments test whether that reference corrects drift in recurrent memory and improves navigation. The broader idea that grid activity can convey movement direction has prior precedent; the contribution sought here is the explicit phase comparison and its tested causal connection to learned memory dynamics. See [Zutshi, Leutgeb & Leutgeb, 2017](https://pubmed.ncbi.nlm.nih.gov/29333481/) and the [original predictive-grid observations](https://pubmed.ncbi.nlm.nih.gov/39146428/).

Code: [decoder](../../scripts/decode_predictive_phase_difference.py). Results: [per-seed reports](../../results/ring_heading/phase_decoder/).
