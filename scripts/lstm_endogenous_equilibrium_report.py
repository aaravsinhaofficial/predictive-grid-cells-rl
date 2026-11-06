"""Summarize all locked original-equilibrium and saturation-audit outcomes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_endogenous_curve_recovery import sha,save


def main():
    root=ROOT/'results/lstm_endogenous_ring'
    files={'full':sorted((root/'fixed_points').glob('*/*.json')),'quotient':sorted((root/'quotient_points').glob('*/*.json')),'eliminated':sorted((root/'equilibrium_elimination').glob('*/*.json'))}
    assert len(files['full'])==16 and len(files['quotient'])==8 and len(files['eliminated'])==16
    datasets={k:[json.loads(p.read_text())for p in pp]for k,pp in files.items()}
    for name,folder,script in [('full','fixed_points','lstm_endogenous_fixed_points.py'),('quotient','quotient_points','lstm_endogenous_quotient_points.py'),('eliminated','equilibrium_elimination','lstm_endogenous_equilibrium_elimination.py')]:
        protocol=json.loads((root/folder/'protocol.json').read_text())
        source=sha(ROOT/'scripts'/script)
        assert (protocol.get('source_sha256') or protocol['sources']['scripts/'+script])==source
        for d in datasets[name]:
            assert d['source_sha256']==source
            assert d['checkpoint_sha256']==sha(ROOT/f"models/{d['condition']}/{d['condition']}_lstm_s{d['seed']}_1000M.pt")
    rows=[];max_naive=0.;all_qualified_c=[]
    for p,d in zip(files['eliminated'],datasets['eliminated']):
        a=np.load(p.with_suffix('.npz'));ok=(np.array(d['hidden_residual_RMS'])<=1e-8)&(np.array(d['stable_full_residual_RMS'])<=1e-8)
        slow=(np.array(d['hidden_residual_RMS'])<=1e-5)&(np.array(d['stable_full_residual_RMS'])<=1e-5)
        labels=np.array(d['hidden_cluster_labels']);qualified_spectra=[s for s in d['spectra']if s['qualified_fixed']]
        cmax=np.max(np.abs(a['solution'][:,256:]),axis=1)
        nearest_size=float(cmax[ok].min()) if ok.any() else None
        if ok.any():
            max_naive=max(max_naive,float(np.max(np.array(d['naive_float64_full_residual_RMS'])[ok])));all_qualified_c.append(nearest_size)
        rows.append(dict(condition=d['condition'],seed=d['seed'],context=d['context'],regime=d['regime'],fixed_initializers=int(ok.sum()),slow_initializers=int(slow.sum()),
            qualified_hidden_clusters=len(np.unique(labels[ok])),slow_hidden_clusters=len(np.unique(labels[slow])),
            minimum_qualified_max_abs_c=nearest_size,native_max_abs_c_after100s=d['native_forward_reachability']['1500']['native_max_abs_c'],
            nearest_solution_hidden_rms_after4s=d['native_forward_reachability']['60']['nearest_solution_hidden_RMS_median'],
            nearest_solution_hidden_rms_after100s=d['native_forward_reachability']['1500']['nearest_solution_hidden_RMS_median'],
            qualified_spectral_slow_counts=[s['near_neutral_count']for s in qualified_spectra],qualified_spectral_unstable_counts=[s['unstable_count']for s in qualified_spectra],
            forget_sigmoid_rounds_to_one_count=d['forget_sigmoid_rounds_to_one_count']))
    quotients=[dict(condition=d['condition'],seed=d['seed'],context=d['context'],removed_c=d['candidate_count'],
        max_heldout_hidden_error=max(v['hidden_max']for v in d['validation'].values()),max_heldout_actor_error=max(v['actor_max']for v in d['validation'].values()),
        passed=d['passes_heldout_validation'],fixed_initializers=d['fixed_count'],slow_initializers=d['slow_count'])for d in datasets['quotient']]
    summary=dict(locked_original_input_cases=16,original_initializers=1024,full_search_fixed_initializers=sum(d['fixed_count']for d in datasets['full']),
        full_search_slow_initializers=sum(d['slow_count']for d in datasets['full']),full_search_case_median_residual_range=[float(min(np.median(d['residual_raw_rms'])for d in datasets['full'])),float(max(np.median(d['residual_raw_rms'])for d in datasets['full']))],
        saturation_quotients=quotients,eliminated_cases=rows,eliminated_fixed_initializers=sum(r['fixed_initializers']for r in rows),
        eliminated_slow_initializers=sum(r['slow_initializers']for r in rows),cases_with_qualified_equilibrium=sum(r['qualified_hidden_clusters']>0 for r in rows),
        qualified_c_size_range=[min(all_qualified_c),max(all_qualified_c)],max_qualified_naive_float64_recurrence_RMS=max_naive,
        all_source_and_checkpoint_hashes_verified=True,scope='Two original seed pairs; descriptive dynamical audit, no population inference. A failed search is not an absence proof; qualified fixed points are distinguished from unqualified optimizer outputs.')
    save(root/'fixed_points_and_saturation_summary.json',summary)
    colors={'gc100':'#367fb5','pgc33':'#db7933'};names={'gc100':'Grid','pgc33':'PGC + grid'}
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.8),layout='constrained')
    for condition in ('gc100','pgc33'):
        group=[r for r in rows if r['condition']==condition];index=0 if condition=='gc100' else 1
        jitter=np.linspace(-.14,.14,len(group))
        axes[0].scatter(index+jitter,[r['qualified_hidden_clusters']for r in group],color=colors[condition],s=38,alpha=.75)
        for j,r in enumerate(group):
            if r['minimum_qualified_max_abs_c'] is None:continue
            axes[1].plot(np.array([0,1])+jitter[j],[r['native_max_abs_c_after100s'],r['minimum_qualified_max_abs_c']],'-o',color=colors[condition],alpha=.55,ms=3)
        qgroup=[q for q in quotients if q['condition']==condition]
        axes[2].scatter(index+np.linspace(-.12,.12,len(qgroup)),[q['removed_c']for q in qgroup],color=colors[condition],s=40,label=names[condition])
    axes[0].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],yticks=[0,1],ylim=(-.2,1.3),ylabel='Distinct qualified h equilibria found',title='One equilibrium in 13 of 16 inputs')
    axes[0].text(.5,1.17,'64 initial phases per input',ha='center',fontsize=9)
    axes[1].set(xticks=[0,1],xticklabels=['Native flow\nafter 100 s','Qualified float64\nequilibria'],yscale='log',ylabel='Largest cell-memory magnitude |c|',title='Equilibria require much larger memory')
    axes[2].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],ylim=(0,27),ylabel='c coordinates removed, out of 256',title='Some memory amplitudes are spectators')
    axes[2].text(.5,25,'All h retained; 4 s output test passed',ha='center',fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for suffix in ('png','pdf','svg'):fig.savefig(root/f'fixed_points_and_saturation.{suffix}',dpi=180)
    text='''# Fixed points, hidden activity, and saturated memory

We tested a concrete alternative: perhaps the full LSTM memory keeps drifting while the useful neuronal activity forms a stable ring. We retained every hidden activity coordinate, `h`, and examined the internal cell-memory coordinates, `c`, separately.

![Original-agent float64 fixed points and causal saturation audit](fixed_points_and_saturation.png)

The equilibrium results use float64 analysis. At these extreme memory amplitudes, the original float32 recurrence exhibits cell-value quantization; the separate precision audit below reports that effect.

**The final equilibrium solver found one distinct qualified activity equilibrium in each of 13 of the 16 fixed-input cases.** Across 1,024 initial states, 641 converged to those equilibria. There were no qualified fixed points in the other three cases; one of those had slow residuals. Every qualified representative was locally stable. This landscape suggests convergence toward individual activity patterns under the tested constant inputs. The retained native-trajectory and recovery tests remain essential for finding a finite-time ring between equilibria.

The original full-state optimizer had found zero fixed points. Its failure was largely a numerical conditioning problem: the equilibrium memory values are much larger than the native values, and the residual changes very slowly with those memory coordinates. We corrected that directly using the exact LSTM equilibrium equation, `c = i*g/(1-f)`, and then solving the remaining activity equation. No angle label, circular loss, learned decoder, or new recurrent weight entered this calculation. All failed earlier searches remain available.

Even the smallest qualified equilibrium's largest memory amplitude was about **5,800**, while the most distant case required about **116 million**. The actual full LSTM, started from the native goal-conditioned states and held under the same input for 100 seconds, reached maximum memory amplitudes of about **900–1,700**. Its activity moved closer to the equilibria, while the memory coordinates remained far away. These remote equilibrium states describe asymptotic dynamics; the relevant navigation mechanism must operate along the native trajectories.

**Some memory amplitudes really can be omitted without changing the output.** In eight separate native-history tests, we found 8–24 such `c` coordinates. The rule used only persistent saturation across eight preset discovery phases. We then removed those amplitudes, retained every `h` and every other `c`, and tested the resulting dynamics on 56 held-out phases. Across six different 4 second input futures—including recorded native input, predictive-cue removal, ordinary-code substitution, and a switch to another recorded context—the maximum hidden-activity error was **1.13×10⁻⁷**, and the maximum action-probability error was **2.68×10⁻⁹**. This validates a finite-domain spectator factor. It does not turn that factor into a ring.

Simply assuming that all initially saturated coordinates were irrelevant failed in all 16 initial tests: some later became unsaturated and affected the output. A stronger forced-clamp manipulation also showed why instantaneous saturation is insufficient. The validated reduction instead replaces only persistently saturated `tanh(c)` values by their fixed sign. This distinction was checked with explicit mathematical tests. The reduced-state fixed-point searches still did not converge before the analytic conditioning fix.

The full-state spectra of the qualified equilibria have **13–41 near-unit eigenvalues**, largely associated with memory coordinates. Their existence is therefore not, by itself, evidence of a one-dimensional angular attractor. The preceding recovery audit also found that raw-coordinate short-term contraction gives way to longer transient growth, and the goal families contract. Together these results point to slow, partly saturated memory dynamics plus input-dependent activity patterns in the tested original agents. A causal ring explanation needs an actual activity family with preferential radial recovery and preserved useful phase; these assays keep those requirements visible instead of inferring a ring from a circular input or a near-unit eigenvalue.

Numerical checks: no fitted forget sigmoid rounded exactly to 1 in the **float64 analysis**. We computed `1-f` as `sigmoid(-forget_logit)` to avoid cancellation, recorded both stable and ordinary float64 recurrence residuals, and evaluated the full-state Jacobian with a stable forget-gate derivative. The largest ordinary float64 recurrence RMS among the qualified solutions was 3.48×10⁻⁹.

We also evaluated every qualified solution in the **original float32 LSTM**. Here 474 cell-gate instances rounded the forget gate to 1. Each corresponding nonzero write was lost because the cell amplitude was so large that float32 could not resolve it; the largest spacing between representable cell values was 8. Elsewhere, a cell changed by as much as 1 in one update. The full-state float32 residual RMS reached 0.0442, while the activity residual RMS remained below 2.63×10⁻⁷. Thus the enormous finite mathematical equilibria also encounter numerical quantization in the native implementation. This reinforces the need to study mechanisms at the actual task-scale states. [Native float32 audit](native_float32_precision_audit.json).

Source and checkpoint hashes pass. These are descriptive results from two original seed pairs, not a population significance claim.

[Machine-readable summary](fixed_points_and_saturation_summary.json) · [Full-state protocol](fixed_points/protocol.json) · [Saturation intervention protocol](saturation_audit/protocol.json) · [Held-out quotient protocol](quotient_points/protocol.json) · [Final exact-equilibrium protocol](equilibrium_elimination/protocol.json) · [Recovery and spectra audit](README.md).

Reproduce:

```bash
python scripts/lstm_endogenous_fixed_points.py
python scripts/lstm_endogenous_saturation_audit.py
python scripts/lstm_endogenous_quotient_points.py
python scripts/lstm_endogenous_equilibrium_elimination.py
python scripts/lstm_endogenous_precision_audit.py
python scripts/lstm_endogenous_equilibrium_report.py
python -m pytest -q tests/test_lstm_endogenous_*.py
```
'''
    (root/'fixed_points_and_saturation.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items()if k not in ('eliminated_cases','saturation_quotients')},indent=2))


if __name__=='__main__':main()
