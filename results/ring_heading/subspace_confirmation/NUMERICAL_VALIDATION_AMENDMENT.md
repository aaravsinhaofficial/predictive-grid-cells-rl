# Numerical validation amendment: the residual-only alias

After fifteen confirmation models had completed, the last model, seed 2012,
failed one validation assertion. The primary interventions and outcomes were
not changed, and this seed remains in every analysis.

The two expressions `bN − P(bN−bO)` and `bO + (bN−bO) − P(bN−bO)` describe the
same counterfactual drive. The second is an algebraic validation alias called
`residual_only`; it is absent from every primary contrast and navigation run.
Their different float32 operation order creates a maximum drive difference of
4.77×10⁻⁷ in seed 2012. After 256 recurrent updates the maximum hidden-coordinate
difference reaches 1.79×10⁻⁵, slightly exceeding the original 10⁻⁵ assertion.
The maximum decoded-heading discrepancy across both scenarios is 0.001995°.

An independent check recomputed both expressions and the recurrent map in
float64, using the same frozen float32 encoder drives converted to float64.
The maximum hidden-coordinate difference fell below 6.2×10⁻¹⁵ and the maximum
heading difference below 3×10⁻¹²°. This identifies numerical operation order,
rather than a changed scientific intervention. [Audit data](numerical_alias_audit_s2012.json).

The amended validation tolerance for this one unused alias is 10⁻⁴. It allows
roughly three times 256 float32 machine-epsilon units, with an additional
independent float64 identity check and a decoded-discrepancy check below 0.01°.
This scale is a practical finite-sequence tolerance, not a theorem bounding
every possible recurrent system's roundoff amplification. Untouched native and
ordinary trajectories still match their independent references exactly.

The [new wrapper](../../../scripts/run_heading_alias_validation_amendment.py)
changes only this assertion in a process-local copy of the function. Every
drive value, trajectory, summary statistic, random draw, other assertion,
navigation computation, and statistical contrast remains unchanged. The
original evaluator and frozen manifest retain their original bytes and hashes.
The original failing log is retained as
`predictive_s2012/analysis_original_tolerance_failure.log`. The completed
seed-2012 subspace and tonic reports record the original function hash, effective
function hash, wrapper hash, and independent audit hash. Other fifteen models'
completed reports are preserved. The new tolerance is a common numerical
criterion; only seed 2012 required rerunning under it.

This validation-only amendment was recorded after identifying the numerical
failure and before computing the missing model's final scientific summaries.
It does not alter the original four heading plus four navigation comparisons or
the separate three-plus-three tonic-diagnostic comparisons.
