"""Compare repeated present inputs with chronological recent inputs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_lstm_code_swap import estimate, paired, holm

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('gc100', 'pgc33')
VARIANTS = ('full', 'k1', 'repeat15', 'k15', 'repeat60', 'k60')


def correct(records):
    holm(records)
    running = 0.
    for j, key in enumerate(sorted(records, key=lambda name: records[name]['exact_signflip_p'])):
        running = max(running, min(1., (len(records)-j)*records[key]['exact_signflip_p']))
        records[key]['holm_exact_p'] = running


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/lstm_history_settling_1B')
    parser.add_argument('--history', type=Path, default=ROOT/'results/lstm_history_horizon_1B')
    args = parser.parse_args()
    runs = {s:json.loads((args.out/f's{s}.json').read_text()) for s in range(1,9)}
    history = {s:json.loads((args.history/f's{s}.json').read_text()) for s in range(1,9)}
    protocol = json.loads((args.out/'protocol.json').read_text())
    for seed in runs:
        run, prior = runs[seed], history[seed]
        assert (run['episodes'],run['n_envs'],run['eval_seed']) == (prior['episodes'],prior['n_envs'],prior['eval_seed'])
        assert run['repetitions'] == [15,60]
        assert all(run['checks'].values()), (seed,run['checks'])
        assert all(prior['checks'].values()), (seed,prior['checks'])
        for condition in CONDITIONS:
            for variant in ('full','k1'):
                assert run['variants'][f'{condition}/{variant}']['rows'] == prior['variants'][f'{condition}/{variant}']['rows']
    metrics = ('mean_score','post_discovery_goal_rate','mean_first_goal_step','frac_found_goal','frac_two_goals')
    estimates = {}
    for metric in metrics:
        estimates[metric] = {}
        for condition in CONDITIONS:
            for variant in VARIANTS:
                source = history if variant in ('k15','k60') else runs
                values = [source[s]['variants'][f'{condition}/{variant}'][metric] for s in range(1,9)]
                entry = estimate(values)
                entry['ci95'] = paired(np.zeros(8),values)['ci95']
                estimates[metric][f'{condition}/{variant}'] = entry
    scores = estimates['mean_score']
    gaps = {variant:paired(scores[f'gc100/{variant}']['values'],scores[f'pgc33/{variant}']['values']) for variant in VARIANTS}
    primary = {}
    for depth in (15,60):
        repeated = gaps[f'repeat{depth}']['differences']
        primary[f'repeat{depth}_minus_k1_gap'] = paired(gaps['k1']['differences'],repeated)
        primary[f'repeat{depth}_minus_chronological_k{depth}_gap'] = paired(gaps[f'k{depth}']['differences'],repeated)
    correct(primary)
    model_contrasts = {f'{condition}/repeat{depth}_minus_{reference}':paired(
        scores[f'{condition}/{reference}']['values'],scores[f'{condition}/repeat{depth}']['values'])
        for condition in CONDITIONS for depth in (15,60) for reference in ('k1',f'k{depth}','full')}
    result = dict(protocol=protocol,estimates=estimates,population_score_gaps=gaps,
        primary_gap_comparisons=primary,descriptive_model_score_comparisons=model_contrasts,
        checks=dict(all_full_and_k1_episode_rows_exact=True,all_first_discovery_times_equal=True,
                    all_checkpoints_unchanged=True),
        statistics='Eight training-seed pairs. Means and t95 intervals use training seeds as independent units. Four exploratory-follow-up score-gap contrasts receive Holm correction separately for paired-t and exact two-sided sign flips. Other intervals/contrasts are descriptive, not tests of equivalence.',
        models=[dict(seed=seed, source=f's{seed}.json', checks=run['checks'],
                     checkpoint_sha256=run['checkpoint_sha256']) for seed, run in runs.items()])
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    labels = ['Current\nonce','Current\n×15','Recent\n15','Current\n×60','Recent\n60','Full\nhistory']
    order = ('k1','repeat15','k15','repeat60','k60','full')
    colors = ['#929AA1','#b66c37','#3b718d','#b66c37','#3b718d','#656477']
    fig,ax = plt.subplots(figsize=(7.7,3.8),layout='constrained')
    for i,(variant,color) in enumerate(zip(order,colors)):
        record = gaps[variant]
        ax.scatter(i+np.linspace(-.12,.12,8),record['differences'],s=19,color=color,alpha=.4,zorder=2)
        lo,hi = record['ci95'];mean = record['difference']
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,ms=6,capsize=4,lw=2,zorder=3)
    ax.axhline(0,color='#8b9299',lw=1)
    ax.set(xticks=range(len(order)),xticklabels=labels,ylabel='PGC − Grid score',
           title='Does repeating the present replace recent history?')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',length=0,pad=9)
    fig.supxlabel('Frozen models · same discovery path · 8 seed pairs · mean and 95% interval',fontsize=9)
    for suffix in ('png','svg','pdf'):
        destination = args.out/f'settling_vs_history.{suffix}'
        fig.savefig(destination,dpi=180,bbox_inches='tight',metadata={'Date':None} if suffix == 'svg' else None)
        if suffix == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(fig)
    pretty = {'full':'Full history','k1':'Current input once','repeat15':'Repeat current input 15 times',
              'k15':'Actual last 15 inputs','repeat60':'Repeat current input 60 times','k60':'Actual last 60 inputs'}
    lines = ['# Repeating the present versus using recent history','',
        '![Settling control](settling_vs_history.png)','',
        'For the repeated-input controllers, we cleared both LSTM states before every decision. We then showed the **same current input** 15 or 60 times before taking one action. The environment moved only once. We compared this with replaying the actual last 15 or 60 inputs. All network weights stayed fixed.','',
        f'**Repeating the present recovered much of the lost absolute score, but did not demonstrate recovery of the PGC advantage.** PGC score rose from {scores["pgc33/k1"]["mean"]:.1f} with one input to {scores["pgc33/repeat15"]["mean"]:.1f} with 15 repetitions; full-history PGC score was {scores["pgc33/full"]["mean"]:.1f}. The population gap under repetition is uncertain.','',
        '| Input supplied before each action | Grid score | PGC score | PGC − Grid, 95% interval |',
        '| --- | ---: | ---: | ---: |']
    for variant in order:
        gap = gaps[variant]
        lines.append(f'| {pretty[variant]} | {scores[f"gc100/{variant}"]["mean"]:.2f} | '
                     f'{scores[f"pgc33/{variant}"]["mean"]:.2f} | {gap["difference"]:+.2f} '
                     f'[{gap["ci95"][0]:+.2f}, {gap["ci95"][1]:+.2f}] |')
    chronological15 = model_contrasts['pgc33/repeat15_minus_k15']
    exceptional = runs[7]['variants']
    historical_exceptional = history[7]['variants']
    lines += ['', f'Actual recent 15-input history gave PGC score {scores["pgc33/k15"]["mean"]:.1f}, versus {scores["pgc33/repeat15"]["mean"]:.1f} when repeating the present. The paired difference (repeat minus actual history) was {chronological15["difference"]:+.2f}, with descriptive 95% interval [{chronological15["ci95"][0]:+.2f}, {chronological15["ci95"][1]:+.2f}]; this does not establish equivalence.','',
        f'**Recovery was not universal.** Seed7 PGC scored {exceptional["pgc33/repeat60"]["mean_score"]:.1f} after 60 repeated inputs, compared with {historical_exceptional["pgc33/k60"]["mean_score"]:.1f} using 60 real inputs and {exceptional["pgc33/full"]["mean_score"]:.1f} with full history. All eight seeds remain in every primary comparison.','',
        '**What remained:** the current image, native live spatial code, externally saved goal code, previous action and previous reward. Repeating the input also repeats those last two values; it is an artificial computation, not a reconstructed physical journey. Each decision starts its repeated updates at zero.','',
        '**What this can show:** if repeating the present recovers the PGC advantage, chronological recent inputs are not required for that advantage under this constructed controller. The native trained policy may still use them. If repetition fails, missing useful history and an unfamiliar operating regime remain possible explanations. A nonsignificant difference does not establish equivalence.','',
        'Every branch followed the same intact Grid controller until first discovery. PGC full-history states therefore experienced Grid-controlled exploration. From then on, each controller chose its own actions. Full-history and current-once results reproduced every corresponding episode from the earlier experiment exactly. The saved goal remained available; this is not removal of all memory.','',
        'This was an **exploratory follow-up**, motivated by seeing seed1 lose performance with one input and recover with 15 chronological inputs. The four comparisons below were specified before any settling scores were observed. They compare the change in the PGC − Grid gap, not a percentage of performance explained.','',
        '| Gap comparison | Difference, 95% interval | Holm t p | Holm sign-flip p |',
        '| --- | ---: | ---: | ---: |']
    for depth in (15,60):
        for key,label in ((f'repeat{depth}_minus_k1_gap',f'Repeat {depth} − current once'),
                          (f'repeat{depth}_minus_chronological_k{depth}_gap',f'Repeat {depth} − actual last {depth}')):
            record = primary[key]
            lines.append(f'| {label} | {record["difference"]:+.2f} [{record["ci95"][0]:+.2f}, '
                         f'{record["ci95"][1]:+.2f}] | {record["holm_t_p"]:.4g} | {record["holm_exact_p"]:.4g} |')
    lines += ['', 'Eight training seeds are the independent replicates; 128 matched episodes per seed and controller. Intervals are paired t intervals across seeds. Frozen-checkpoint repeated-input calculations were checked against explicit native LSTMCell updates on recorded real inputs.','',
              '[Numerical results](summary.json) · [Protocol](protocol.json) · [Chronological-history experiment](../lstm_history_horizon_1B/README.md)']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    for variant in order:
        print(variant, 'grid',round(scores[f'gc100/{variant}']['mean'],3),'pgc',
              round(scores[f'pgc33/{variant}']['mean'],3),'gap',gaps[variant]['difference'],gaps[variant]['ci95'])
    for key,record in primary.items():
        print(key,record['difference'],record['ci95'],record['holm_t_p'],record['holm_exact_p'])


if __name__ == '__main__':main()
