"""Audit and summarize the frozen original-RL natural goal-state pulse cohort."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm_goal_state_pulse import write_json

CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}
COLORS = {'gc100': '#2675b6', 'pgc33': '#e66a2b'}


def stats(values, signed_contrast=True):
    x = np.asarray(values, float); n = len(x)
    mean = float(x.mean()); sem = float(x.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.
    interval = t.ppf(.975, n-1)*sem if n > 1 else 0.
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=n)))
    p = float((np.abs((signs*x).mean(1)) >= abs(mean)-1e-12).mean()) if signed_contrast else None
    return {'n_seeds': n, 'mean': mean, 'ci95': [mean-interval, mean+interval],
            'exact_two_sided_sign_flip_p': p, 'seed_values': x.tolist()}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/lstm_goal_message_state_pulse')
    args = parser.parse_args()
    manifest_path = ROOT/'docs/research/original_rl_goal_state_pulse_sources.json'
    manifest = json.loads(manifest_path.read_text())
    for p, digest in manifest['sources'].items(): assert sha(ROOT/p) == digest, p
    rows, traces, missing = [], {}, []
    for seed in range(1, 9):
        for condition in CONDITIONS:
            run = args.root/f'pilot_{condition}_s{seed}'
            if not (run/'summary.json').exists(): missing.append(f'{condition}_s{seed}'); continue
            summary = json.loads((run/'summary.json').read_text())
            protocol = summary['protocol']; arguments = protocol['arguments']
            assert arguments['seed'] == seed and arguments['condition'] == condition
            for key,value in {'episodes':64,'eval_seed':872601,'prepare':15,'horizon':300,'angle_degrees':15.}.items():
                assert arguments[key] == value, (run,key,arguments[key])
            for p, digest in protocol['source_hashes'].items(): assert manifest['sources'][p] == digest, (run,p)
            assert sha(ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt') == protocol['checkpoint_sha256']
            assert summary['checkpoint_unchanged'] and summary['validity']['native_replay_max_error'] < 4e-4
            trajectory = np.load(run/'closed_loop.npz'); replay = np.load(run/'same_observation_replay.npz')
            windows = np.load(run/'native_windows.npz')
            n = summary['eligible_trials']; assert trajectory['first_hit_step'].shape == (3,4,n)
            if condition == 'gc100':
                for key in trajectory.files:
                    value = trajectory[key]; axis = 0 if key == 'first_hit_step' else 1
                    assert np.array_equal(np.take(value,0,axis=axis),np.take(value,1,axis=axis)), (run,key)
                    assert np.array_equal(np.take(value,0,axis=axis),np.take(value,2,axis=axis)), (run,key)
                for key in replay.files:
                    assert np.array_equal(replay[key][:,0],replay[key][:,1]), (run,key)
                    assert np.array_equal(replay[key][:,0],replay[key][:,2]), (run,key)
            # h is held exactly in the c-only pulse; the first actor distribution must match.
            assert np.array_equal(trajectory['action_probability'][0,:,0],trajectory['action_probability'][0,:,2])
            row = {'seed': seed, 'condition': condition, 'n_trials': n,
                   'native_replay_max_error': summary['validity']['native_replay_max_error']}
            for i, input_name in enumerate(('native','goal_ordinary','current_ordinary')):
                for j, pulse in enumerate(('full','c_only','h_only'), 1):
                    prefix = input_name+'_'+pulse+'_'
                    tv = replay['action_tv'][:,i,j-1]
                    row[prefix+'action_tv_1s'] = float(tv[15].mean())
                    row[prefix+'action_tv_mean_first_second'] = float(tv[1:16].mean())
                    row[prefix+'action_tv_mean_second_second'] = float(tv[16:31].mean())
                    row[prefix+'state_gain_1s'] = float(np.mean(replay['state_error_norm'][15,i,j-1]/np.maximum(replay['state_error_norm'][0,i,j-1],1e-12)))
                    first = trajectory['first_hit_step'][i,j]; baseline = trajectory['first_hit_step'][i,0]
                    row[prefix+'goal_time_change_s'] = float(np.mean(np.minimum(first+1,arguments['horizon'])-np.minimum(baseline+1,arguments['horizon']))/15)
                    row[prefix+'goal_hit_change'] = float(((first<arguments['horizon']).astype(float)-(baseline<arguments['horizon'])).mean())
                    path = trajectory['position'][:16,i,j]-trajectory['position'][:16,i,0]
                    row[prefix+'path_change_first_second_m'] = float(np.linalg.norm(path,axis=-1).mean())
                    original_name = ('false_goal_state','false_goal_c_only','false_goal_h_only')[j-1]
                    row[prefix+'initial_signed_action_shift'] = summary['results'][input_name][original_name]['initial_movement_probability_shift_toward_false_goal']
                    row[prefix+'signed_cross_goal_change_at1s_m'] = summary['results'][input_name][original_name]['first_second_signed_cross_goal_change_m']
            row['goal_ordinary_minus_native_c_only_residual_tv_1s'] = row['goal_ordinary_c_only_action_tv_1s']-row['native_c_only_action_tv_1s']
            row['native_c_minus_h_residual_tv_1s'] = row['native_c_only_action_tv_1s']-row['native_h_only_action_tv_1s']
            rows.append(row); traces[(condition,seed)] = replay['action_tv'][:61].mean(-1)
            trajectory.close(); replay.close(); windows.close()
    result = {'status': 'final' if not missing else 'partial', 'completed_models': len(rows), 'missing': missing,
              'manifest_sha256': sha(manifest_path), 'all_frozen_sources_verified': True,
              'grid_input_substitution_bitwise_noop_verified': True, 'c_only_initial_actor_unchanged_verified': True,
              'rows': rows, 'population': {}}
    for condition in CONDITIONS:
        group = [r for r in rows if r['condition']==condition]
        if group:
            result['population'][condition] = {key: stats([r[key] for r in group], signed_contrast=not any(
                pattern in key for pattern in ('_action_tv_','_state_gain_','_path_change_first_second_m'))) for key in group[0]
                                              if key not in ('seed','condition','n_trials','native_replay_max_error')}
    if not missing:
        lookup = {(r['condition'],r['seed']):r for r in rows}
        comparison_keys = ('native_full_initial_signed_action_shift','native_c_only_action_tv_1s',
                           'native_c_minus_h_residual_tv_1s','native_full_goal_time_change_s',
                           'native_full_goal_hit_change','native_full_path_change_first_second_m')
        result['paired_pgc_minus_grid'] = {key: stats([lookup['pgc33',s][key]-lookup['gc100',s][key] for s in range(1,9)])
                                          for key in comparison_keys}
        result['statistics_note'] = ('Exploratory unadjusted signed contrasts; intervals and exact p-values do not imply a confirmatory family of discoveries. '
                                    'No sign-flip p-value is assigned to raw nonnegative TV distances, state gains or path-displacement norms.')
    write_json(args.root/'cohort_summary.json', result)
    if missing:
        print(json.dumps({'status':'partial','completed':len(rows),'missing':missing})); return
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.dpi':150, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(1,3,figsize=(12.1,3.8),layout='constrained')
    for k, condition in enumerate(CONDITIONS):
        pop = result['population'][condition]; x = pop['native_full_initial_signed_action_shift']
        ax[0].bar(k,x['mean'],color=COLORS[condition],width=.6,alpha=.85)
        ax[0].errorbar(k,x['mean'],yerr=[[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]],fmt='none',color='black',capsize=4)
        ax[0].scatter(np.full(8,k)+np.linspace(-.12,.12,8),x['seed_values'],color='black',s=13,zorder=4)
        tv = np.stack([traces[(condition,seed)][:,0] for seed in range(1,9)])
        for index,style in ((1,'-'),(2,'--')):
            mean = tv[:,:,index].mean(0); ci=t.ppf(.975,7)*tv[:,:,index].std(0,ddof=1)/np.sqrt(8)
            ax[1].plot(np.arange(61)/15,mean,color=COLORS[condition],ls=style,
                       label=LABELS[condition]+(' c' if index==1 else ' h'))
            if index==1: ax[1].fill_between(np.arange(61)/15,np.maximum(mean-ci,0),mean+ci,color=COLORS[condition],alpha=.13)
        for j,input_name in enumerate(('native','goal_ordinary')):
            y=pop[input_name+'_full_goal_time_change_s']; xpos=k*2+j
            ax[2].errorbar(xpos,y['mean'],yerr=[[y['mean']-y['ci95'][0]],[y['ci95'][1]-y['mean']]],fmt='o',
                           color=COLORS[condition],mfc=COLORS[condition] if j==0 else 'white',capsize=4)
            ax[2].scatter(np.full(8,xpos)+np.linspace(-.12,.12,8),y['seed_values'],color=COLORS[condition],s=12,alpha=.45)
    ax[0].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],ylabel='Action shift toward the false goal',title='A goal message changes the state')
    ax[0].axhline(0,color='.6',lw=.8)
    ax[1].set(xlabel='Time after restoring real inputs (s)',ylabel='Action-distribution change',title='The cell state carries the lasting effect',xlim=(0,1.05))
    ax[1].legend(frameon=False,fontsize=8)
    ax[2].set(xticks=[0,1,2,3],xticklabels=['Grid\nnative','Grid\nordinary','PGC\nnative','PGC\nordinary'],
              ylabel='Change in return time (s)',title='Effect on actual return trips')
    ax[2].axhline(0,color='.6',lw=.8)
    for a in ax: a.grid(axis='y',alpha=.15)
    for ext in ('png','pdf','svg'): fig.savefig(args.root/('original_rl_goal_state_pulse.'+ext))
    plt.close(fig)
    lines=['# Goal-conditioned state in the original RL networks','',
           '![Original-policy state interventions](original_rl_goal_state_pulse.png)','',
           'These are the original square-arena reward-trained LSTMs: eight Grid models and eight PGC+grid models. The actor and every policy weight stayed unchanged. No heading-supervised substitute agent was used.','',
           'We replayed one second of each agent’s actual recent experience while changing only its saved goal snapshot to point 15° to one side. We then inserted the resulting state at the real physical location, restored the real goal message and let the original actor continue.','',
           '| Measure, averaged across eight trained seeds | Grid | PGC + grid |','|---|---:|---:|']
    metrics=[('Initial movement-probability shift toward fictitious goal','native_full_initial_signed_action_shift',1),
             ('Action effect after1s: c-only','native_c_only_action_tv_1s',1),
             ('Action effect after1s: h-only','native_h_only_action_tv_1s',1),
             ('First-second mean path change, mm','native_full_path_change_first_second_m',1000),
             ('Return-time change, seconds','native_full_goal_time_change_s',1)]
    for label,key,scale in metrics:
        values=[result['population'][c][key]['mean']*scale for c in CONDITIONS]
        lines.append(f'| {label} | {values[0]:.4g} | {values[1]:.4g} |')
    lines += ['', 'The altered goal message changes the original actor. The signed first-action result above checks whether its translational component moves toward the fictitious goal; it does not include turning actions. For this particular goal-conditioned state substitution, the persisting action effect resides chiefly in c. The immediate h contribution fades quickly. Both trained populations show this organization.', '',
              'A lasting influence is not a ring attractor. The native versus ordinary-goal comparisons in the full summary test whether PGC input improves recovery; the existence of a persisting c effect by itself does not answer that question. The table reports actual path and return-time effects. Neither the total state norm nor the hybrid-component comparison is an equal-amplitude general comparison of h and c importance.', '',
              'Dots represent trained seeds; intervals are descriptive95% intervals across seeds. The pilot preceded the fixed exploratory replication. Exact sign-flip statistics and every seed/trial count are in [cohort_summary.json](cohort_summary.json). No model was excluded. The external goal snapshot remains supplied throughout the task, so this is goal-conditioned recurrent processing, not a demonstration that the goal must be stored without a cue.', '',
              'Validation: the unchanged recent input history reconstructs the original state; c-only edits leave the first action distribution unchanged; native/ordinary substitutions in Grid-only models give bitwise-identical complete neural and behavioral arrays. Frozen checkpoint/source hashes are verified for every model.']
    (args.root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'final','models':len(rows),'figure':str(args.root/'original_rl_goal_state_pulse.png')}))


if __name__ == '__main__': main()
