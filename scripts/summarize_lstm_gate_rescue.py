"""Summarize closed-loop rescue of selected LSTM components across eight seeds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from summarize_lstm_code_swap import estimate, paired, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_gate_rescue_1B'
SEEDS = range(1, 9)
VARIANTS = ('intact', 'both_grid', 'restore_input', 'restore_forget',
            'restore_candidate', 'restore_output', 'restore_candidate_output', 'restore_input_forget')
LABELS = ('Intact predictive code', 'Grid code substitution', 'Restore input gate',
          'Restore forget gate', 'Restore candidate update', 'Restore output gate',
          'Restore candidate + output', 'Restore input + forget')


def main():
    runs = {s:json.loads((OUT/f'pgc33_s{s}.json').read_text())for s in SEEDS}
    protocol = json.loads((OUT/'protocol.json').read_text())
    pair_control = {s:json.loads((OUT/'pair_control'/f'pgc33_s{s}.json').read_text()) for s in SEEDS}
    for s in SEEDS:
        control=pair_control[s]
        assert (control['episodes'],control['eval_seed'],control['n_envs'],control['checkpoint_sha256']) == (
            runs[s]['episodes'],runs[s]['eval_seed'],runs[s]['n_envs'],runs[s]['checkpoint_sha256'])
        assert control['variants']['both_grid']['rows'] == runs[s]['variants']['both_grid']['rows'], 'Supplemental baseline did not reproduce exactly'
        runs[s]['variants']['restore_input_forget']=control['variants']['restore_input_forget']
    for r in runs.values():
        assert (r['episodes'],r['eval_seed']) == (protocol['episodes_per_seed_variant'],protocol['eval_seed'])
        assert all(v['episodes']==r['episodes'] for v in r['variants'].values())
    metrics = ('mean_score', 'mean_path_eff', 'post_discovery_goal_rate', 'mean_first_goal_step', 'frac_found_goal')
    estimates, comparisons, fractions = {}, {}, {}
    for metric in metrics:
        values={v:[runs[s]['variants'][v][metric]for s in SEEDS]for v in VARIANTS}
        estimates[metric]={v:estimate(a)for v,a in values.items()}
        family={v+'_minus_grid':paired(values['both_grid'], values[v])for v in VARIANTS if v!='both_grid'}
        family['candidate_minus_output']=paired(values['restore_output'],values['restore_candidate'])
        family['candidate_output_minus_intact']=paired(values['intact'],values['restore_candidate_output'])
        family['candidate_output_minus_input_forget']=paired(values['restore_input_forget'],values['restore_candidate_output'])
        holm(family)
        comparisons[metric]=family
        gap=np.mean(values['intact'])-np.mean(values['both_grid'])
        fractions[metric]={v:float((np.mean(values[v])-np.mean(values['both_grid']))/gap)for v in VARIANTS[2:]}
    result=dict(protocol=protocol, supplemental_pair_control='pair_control/protocol.json',
                supplemental_baseline_episode_records_match_exactly=True,
                estimates=estimates,comparisons=comparisons,
                fraction_of_mean_intact_grid_gap_recovered=fractions,
                gate_reconstruction_max_error=max(r['gate_reconstruction_max_error']for r in runs.values()),
                statistics='Eight training-seed means are replicates. Paired t intervals and tests; exact two-sided sign flips. Holm correction within each ten-comparison outcome family. Recovery fraction is a ratio of group-mean differences, descriptive, not a per-seed average or proof of mediation. The input+forget pair control was added after inspecting early rescue results; it reproduces the common baseline episode records exactly.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    figure(result)
    for metric in ('mean_score','mean_path_eff','post_discovery_goal_rate'):
        print(metric)
        for name,d in estimates[metric].items():
            print(f'  {name}: {d["mean"]:.4f} +/- {d["sem"]:.4f}')
    print('Score recovery fractions:',fractions['mean_score'])
    for name,r in comparisons['mean_score'].items():
        print(name, r['difference'], r['ci95'], 'p', r['paired_t_p'], 'Holm',r['holm_t_p'])


def figure(result):
    fig, axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    colors=['#d66b37','#87919b','#6aa0a8','#9f93af','#4d7fae','#b28d57','#4a9b72','#9a6c8d']
    for ax,metric,title,xlabel in zip(axes,['mean_score','mean_path_eff'],
        ['A  |  Navigation score','B  |  Return-path efficiency'],['Score / episode','Path efficiency']):
        for i,(v,color)in enumerate(zip(VARIANTS,colors)):
            r=result['estimates'][metric][v]
            jitter=np.linspace(-.12,.12,8)
            ax.scatter(r['values'],i+jitter,color=color,alpha=.5,s=23)
            ax.errorbar(r['mean'],i,xerr=r['sem'],fmt='o',color=color,capsize=4,ms=7)
        ax.axvline(result['estimates'][metric]['both_grid']['mean'],color='#87919b',ls=':',lw=1)
        ax.axvline(result['estimates'][metric]['intact']['mean'],color='#d66b37',ls=':',lw=1)
        ax.set(yticks=np.arange(8),yticklabels=LABELS if metric=='mean_score'else ['']*8,
               xlabel=xlabel,title=title,ylim=(7.6,-.6))
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.suptitle('Does restoring a PGC-driven LSTM component rescue behavior?',fontsize=15)
    fig.supxlabel('Frozen PGC-trained models; eight seeds × 128 fresh episodes per condition. Dots: seeds; error bars: SEM.\nComponents not restored receive grid-code drive; all conditions keep their own evolving recurrent state and behavior.',fontsize=9)
    fig.savefig(OUT/'gate_rescue.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'gate_rescue.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
