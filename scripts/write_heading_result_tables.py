"""Write numerical tables only from complete, fixed-cohort summaries."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/ring_heading"


def read(relative):
    return json.loads((BASE/relative).read_text())


def number(value):
    if value and abs(value)<.001:
        return f"{value:.2g}"
    return f"{value:.3f}"


def estimate(values,scale=1):
    x=np.asarray(values,dtype=float)*scale
    if len(x)!=16 or not np.isfinite(x).all():
        raise ValueError("Expected 16 finite seed values")
    mean=float(x.mean()); half=float(stats.t.ppf(.975,15)*stats.sem(x))
    return f"{number(mean)} [{number(mean-half)}, {number(mean+half)}]"


def effect_table(family,p_key,labels=None,scale=1):
    lines=["| Contrast | Mean difference [95% paired interval] | Holm p |",
           "| --- | ---: | ---: |"]
    for key,row in family.items():
        label=labels.get(key,key.replace("_"," ")) if labels else key.replace("_"," ")
        lo,hi=[v*scale for v in row["ci95"]]
        lines.append(f"| {label} | {number(row['mean_difference']*scale)} [{number(lo)}, {number(hi)}] | {number(row[p_key])} |")
    return "\n".join(lines)


def main():
    primary=read("confirmation_summary.json")
    subspace=read("subspace_confirmation/summary.json")
    tonic=read("tonic_direction_confirmation/summary.json")
    cue=read("cue_spectra_confirmation/summary.json")
    raw=read("raw_sensitivity_evaluation/summary.json")
    geometry=read("geometry_confirmation/summary.json")
    if (primary["seeds"]!=list(range(2001,2017)) or not subspace["complete"] or
        not tonic["complete"] or cue["n"]!=16):
        raise ValueError("All 16 seeds in every required cohort must be complete")
    if raw["n_seeds"]!=8 or not geometry["complete"] or geometry["models_loaded"]!=32:
        raise ValueError("All separate architecture and geometry cohorts must be complete")
    lines=["# Complete learned-heading results", "",
        "All 16 planned training seeds are included. Intervals are nominal 95% paired-t or group-mean intervals across seeds, not simultaneous intervals. Exact two-sided sign-flip tests are corrected within the declared families. Individual trajectories and phases are not independent training replicates.",
        "", "## Four locked primary endpoints", "",
        "| Network | Native heading error (degrees) | Error at new places (degrees) |",
        "| --- | ---: | ---: |"]
    for condition,label in (("grid","Grid"),("predictive","PGC + grid"),("grid_heading","Grid + explicit direction")):
        rows=primary["per_seed"]
        lines.append(f"| {label} | {estimate([r[condition]['same_region_native_mae_deg'] for r in rows])} | {estimate([r[condition]['new_places_native_mae_deg'] for r in rows])} |")
    labels={"tracking_pgc_minus_grid_deg":"PGC − Grid: native heading error (degrees)",
            "removal_minus_native_deg":"Ordinary replacement − native PGC: error (degrees)",
            "phase_gain_native_minus_ordinary":"Native − ordinary: retained local phase gain at 5 s",
            "transfer_pgc_minus_grid_deg":"PGC − Grid: error at new places (degrees)"}
    lines += ["",effect_table(primary["primary_endpoints"],"p_holm_four_primary",labels),"",
        f"The accurate-anchoring criterion is met by {primary['accurate_anchoring_seeds']}/16 seeds. This criterion combines decoded coverage with lower true-heading error than ordinary replacement and is never used to exclude a model. Ordinary replacement can contract disturbances by collapsing toward a wrong state; phase gain alone is therefore not evidence of useful anchoring or a globally smooth ring.",
        "", "## Direction-plane intervention at new places", "",
        "| Frozen-PGC intervention | Heading error (degrees) | Goal reached (%) | Sustained goal holding (%) |",
        "| --- | ---: | ---: | ---: |"]
    for variant,label in (("native","Native"),("ordinary","Ordinary replacement"),
                          ("remove_direction","Remove direction plane"),("restore_direction","Restore direction plane")):
        lines.append(f"| {label} | {estimate([r['heading_'+variant] for r in subspace['rows']])} | {estimate([r['navigation_'+variant] for r in subspace['rows']],100)} | {estimate([r['holding_'+variant] for r in subspace['rows']],100)} |")
    random_heading=[np.mean([r[f'heading_sham{j}_restore'] for j in range(3)]) for r in subspace['rows']]
    random_navigation=[np.mean([r[f'navigation_sham{j}_restore'] for j in range(3)]) for r in subspace['rows']]
    random_holding=[np.mean([r[f'holding_sham{j}_restore'] for j in range(3)]) for r in subspace['rows']]
    lines.append(f"| Restore random planes (within-model mean) | {estimate(random_heading)} | {estimate(random_navigation,100)} | {estimate(random_holding,100)} |")
    lines += ["", "Heading improvements are measured in degrees; positive contrasts mean improvement or a removal cost, as specified in the frozen protocol. Navigation contrasts are percentage points.","",
              effect_table(subspace["heading_tests"],"holm_exact_p"),"",
              effect_table(subspace["navigation_tests"],"holm_exact_p",scale=100),"",
              "## Additional fixed-mean diagnostic", "",
              "This follow-up was specified after the pilot and after six models had their original subspace analyses computed. Its three contrasts per endpoint form separate exploratory correction families. It does not replace the original four-plus-four subspace families.","",
              "| Intervention | Heading error (degrees) | Goal reached (%) |",
              "| --- | ---: | ---: |"]
    for variant,label in (("mean_only_restore","Restore only fixed mean"),("mean_preserving_removal","Remove variation, preserve fixed mean")):
        lines.append(f"| {label} | {estimate([r['heading_'+variant] for r in tonic['rows']])} | {estimate([r['navigation_'+variant] for r in tonic['rows']],100)} |")
    lines += ["",effect_table(tonic["heading_exploratory_tests"],"holm_exact_p"),"",
              effect_table(tonic["navigation_exploratory_tests"],"holm_exact_p",scale=100),"",
              "## Same-state Jacobian measurements", "",
              "The state and recurrent weights are fixed while the input changes. Phase gain is a derivative in the decoded-angle coordinate, not an eigenvalue. These median local measurements do not certify global hidden-state topology; the branch audit is reported separately.","",
              "| Input at cue-free states | One-step phase gain | Spectral radius |",
              "| --- | ---: | ---: |"]
    for variant,label in (("blank","Input off"),("ordinary","Ordinary code"),("native","Predictive code")):
        key="cue_free_"+variant
        lines.append(f"| {label} | {estimate([r[key]['phase_gain'] for r in cue['per_seed']])} | {estimate([r[key]['spectral_radius'] for r in cue['per_seed']])} |")
    lines += ["",effect_table(cue["contrasts"],"p_holm_three_secondary"),"",
        "## Separate raw-input architecture cohort", "",
        "Eight fresh seed triplets, 24 models. Removing the nonlinear encoder also reduces parameter count; this is architecture/capacity sensitivity at a fixed training budget. The three tests have their own correction family. [Tracking figure and group means](raw_sensitivity_evaluation/README.md); [memory and anchoring diagnostics](raw_memory_evaluation/README.md).", "",
        effect_table(raw["contrasts"],"p_holm_three_secondary",labels), "",
        "## Independent geometry confirmation", "",
        "Sixteen fresh Grid/PGC pairs, 32 models. The single prespecified endpoint is log10 of estimated peak-to-median angular sensitivity of the full initialization-to-memory map. This is not a test of global ring topology or transverse stability.", ""]
    g=geometry["primary_inference"]
    lines += ["| Contrast | Mean difference [95% paired interval] | Exact two-sided p |",
        "| --- | ---: | ---: |",
        f"| PGC − Grid log10 sharpness | {g['mean']:.3f} [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}] | {g['p_two_sided_exact']:.6f} |", "",
        f"PGC has lower estimated sharpness in {g['PGC_lower_pairs_descriptive']}/16 pairs. The primary test does not confirm a reliable population difference. All models are retained. [All-seed figure and controls](geometry_confirmation/README.md).", "",
        "## Source data", "",
        "[Primary summary](confirmation_summary.json) · [Subspace summary](subspace_confirmation/summary.json) · [Fixed-mean summary](tonic_direction_confirmation/summary.json) · [Jacobian summary](cue_spectra_confirmation/summary.json) · [Geometry audits](geometry_audit/) · [Independent geometry summary](geometry_confirmation/summary.json) · [Raw-input summary](raw_sensitivity_evaluation/summary.json) · [Full-state memory](confirmation_evaluation/memory_summary.json) · [Reduced phase model and memory pulses](mechanistic_confirmation/summary.json)", ""]
    (BASE/"RESULTS.md").write_text("\n".join(lines))
    print("Wrote results/ring_heading/RESULTS.md from six complete cohort summaries")


if __name__=="__main__":main()
