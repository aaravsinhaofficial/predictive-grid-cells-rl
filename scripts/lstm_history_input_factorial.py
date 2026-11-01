"""Separate current spatial input from the state produced by earlier inputs.

Default: native snapshots, full versus cleared pre-input h/c crossed with current
code substitutions. --shared-replay: four complete prior code histories crossed
with four current inputs on identical physical experience, without extra resets.
CPU only; frozen models; no states cross model boundaries. The external goal
message and previous action/reward remain available. Action sensitivity is not
navigation performance or a measure of useful memory.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from code.common import build_env, build_policy, build_population
from lstm_action_computation import held_go_capture, choose_rows
from lstm_code_swap import verify_codes
from summarize_lstm_code_swap import estimate, paired, holm

SOURCE = ROOT / 'results/lstm_navigation_phases_1B'
OUT = ROOT / 'results/lstm_history_input_factorial_1B'
INPUTS = ('native', 'current_ordinary', 'goal_ordinary', 'both_ordinary')
REPLAY_SOURCE = ROOT / 'results/lstm_variable_1B'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_average(values, episodes, mask=None):
    if mask is None:
        mask = np.ones(len(episodes), dtype=bool)
    return float(np.mean([values[mask & (episodes == ep)].mean()
                         for ep in np.unique(episodes) if np.any(mask & (episodes == ep))]))


def tv(p, q):
    return .5 * np.abs(p - q).sum(-1)


def correct(family):
    holm(family)
    running = 0.
    for j, key in enumerate(sorted(family, key=lambda k: family[k]['exact_signflip_p'])):
        running = max(running, min(1., (len(family)-j) * family[key]['exact_signflip_p']))
        family[key]['holm_exact_p'] = running


@torch.inference_mode()
def run_native(condition, seed, source, out):
    path = source / f'{condition}_s{seed}_action_states.npz'
    data = dict(np.load(path))
    stem = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg = json.loads(stem.with_suffix('.json').read_text())
    ckpt = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    net = build_policy(cfg, 'cpu').eval().requires_grad_(False)
    net.load_state_dict(ckpt['policy'], strict=True)
    pop = build_population(cfg, 'cpu')
    e = torch.from_numpy(data['embedding'])
    current, goal = (torch.from_numpy(data[k]) for k in ('current_code', 'goal_code'))
    ordinary_current = pop.activity(*(torch.from_numpy(data[k]) for k in ('pos', 'dir', 'vel')), shift_scale=0)
    ordinary_goal = pop.activity(*(torch.from_numpy(data[k]) for k in ('memory_pos', 'memory_dir', 'memory_vel')), shift_scale=0)
    if condition == 'pgc33':
        # Preserve all unchanged channels bit-for-bit, including MPS rounding.
        ordinary_current = torch.where(pop.is_pgc[None], ordinary_current, current)
        ordinary_goal = torch.where(pop.is_pgc[None], ordinary_goal, goal)
        assert torch.equal(ordinary_current[:, ~pop.is_pgc], current[:, ~pop.is_pgc])
        assert torch.equal(ordinary_goal[:, ~pop.is_pgc], goal[:, ~pop.is_pgc])
    pa, pr = (torch.from_numpy(data[k]) for k in ('previous_action', 'previous_reward'))
    full = tuple(torch.from_numpy(data[k]) for k in ('h_before', 'c_before'))
    histories = {'full': full, 'zero': net.zero_state(len(e), 'cpu')}
    variants = INPUTS if condition == 'pgc33' else ('native',)
    arrays = {}; hidden = {}; logits = {}; outputs = {}
    episodes = data['episode'] * 64 + data['slot']
    for history, state in histories.items():
        for name in variants:
            cur = ordinary_current if name in ('current_ordinary', 'both_ordinary') else current
            gol = ordinary_goal if name in ('goal_ordinary', 'both_ordinary') else goal
            lo, value, newstate = net.step_from_embedding(e, cur, gol, pa, pr, state)
            key = history + '/' + name
            logits[key] = lo.numpy()
            arrays[key] = lo.softmax(1).numpy()
            hidden[key] = tuple(a.numpy() for a in newstate)
    # Independent reproduction of the earlier same-state native-input probe.
    old = ROOT / f'results/lstm_action_computation_1B/on_policy/{condition}_s{seed}_actions.npz'
    previous = dict(np.load(old))
    validation_error = float(np.max(np.abs(arrays['full/native'] - previous['intact_probability'])))
    assert validation_error < 1e-6, validation_error
    go = int(arrays['full/native'][:, 2:].mean(0).argmax() + 2)
    turn = int(arrays['full/native'][:, :2].mean(0).argmax())
    hit = held_go_capture(data, go, cfg)
    near = np.linalg.norm(data['pos'] - data['goal_center'], axis=1) < .6
    hitmask, missmask = near & hit, near & ~hit
    for key, p in arrays.items():
        history = key.split('/')[0]
        entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(1)
        go_hit = episode_average(p[:, go], episodes, hitmask)
        turn_miss = episode_average(p[:, turn], episodes, missmask)
        outputs[key] = dict(
            action_tv_from_full_native=episode_average(tv(p, arrays['full/native']), episodes),
            action_tv_from_same_history_native=episode_average(tv(p, arrays[history + '/native']), episodes),
            h_rms_from_full_native=episode_average(np.sqrt(((hidden[key][0]-hidden['full/native'][0])**2).mean(1)), episodes),
            c_rms_from_full_native=episode_average(np.sqrt(((hidden[key][1]-hidden['full/native'][1])**2).mean(1)), episodes),
            entropy_nats=episode_average(entropy, episodes),
            go_probability_capture=go_hit, turn_probability_miss=turn_miss,
            balanced_capture_agreement=.5*(go_hit+turn_miss),
            chosen_action_changed=episode_average((p.argmax(1) != arrays['full/native'].argmax(1)).astype(float), episodes))
    interactions = {}
    for name in variants[1:]:
        # Explicit probability-vector difference-in-differences, before taking a norm.
        direct_full = arrays['full/'+name] - arrays['full/native']
        direct_zero = arrays['zero/'+name] - arrays['zero/native']
        interaction = .5*np.abs(direct_full-direct_zero).sum(1)
        interactions[name] = dict(probability_interaction_half_l1=episode_average(interaction, episodes),
            native_input_history_tv=outputs['zero/native']['action_tv_from_full_native'],
            ordinary_input_history_tv=episode_average(tv(arrays['full/'+name], arrays['zero/'+name]), episodes))
    record = dict(condition=condition, seed=seed, samples=len(e), episodes=len(np.unique(episodes)),
                  variants=outputs, interactions=interactions,
                  dominant_go=go, dominant_turn=turn,
                  validation=dict(native_probability_vs_existing_probe_max_error=validation_error),
                  checkpoint_sha256=digest(stem.with_suffix('.pt')), snapshot_sha256=digest(path))
    np.savez_compressed(out/f'{condition}_s{seed}_factorial.npz', **{k.replace('/', '_'):v for k,v in arrays.items()},
                        episodes=episodes, near=near, held_go_capture=hit)
    (out/f'{condition}_s{seed}.json').write_text(json.dumps(record, indent=2)+'\n')
    print(condition, seed, 'history TV', round(outputs['zero/native']['action_tv_from_full_native'], 4), flush=True)


def summarize(out):
    runs = {(c,s):json.loads((out/f'{c}_s{s}.json').read_text())
            for c in ('gc100','pgc33') for s in range(1,9)}
    for (c,s),record in runs.items():
        for values in record['interactions'].values():
            if 'probability_interaction_tv' in values:
                values['probability_interaction_half_l1']=values.pop('probability_interaction_tv')
        (out/f'{c}_s{s}.json').write_text(json.dumps(record,indent=2)+'\n')
    estimates = {}
    for c in ('gc100','pgc33'):
        for key, values in runs[c,1]['variants'].items():
            estimates[c+'/'+key] = {m:estimate([runs[c,s]['variants'][key][m] for s in range(1,9)]) for m in values}
    # Four planned sensitivity comparisons, not navigation or memory-performance tests.
    family = {'pgc_minus_grid_reset_tv': paired(
        estimates['gc100/zero/native']['action_tv_from_full_native']['values'],
        estimates['pgc33/zero/native']['action_tv_from_full_native']['values'])}
    for name in INPUTS[1:]:
        family[name+'_full_minus_zero_input_tv'] = paired(
            estimates['pgc33/zero/'+name]['action_tv_from_same_history_native']['values'],
            estimates['pgc33/full/'+name]['action_tv_from_same_history_native']['values'])
    correct(family)
    interactions={name:{m:estimate([runs['pgc33',s]['interactions'][name][m] for s in range(1,9)])
                       for m in runs['pgc33',1]['interactions'][name]} for name in INPUTS[1:]}
    summary=dict(protocol=json.loads((out/'protocol.json').read_text()), estimates=estimates,
                 comparisons=family, interactions=interactions, models=list(runs.values()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig, ax=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained')
    for i,c in enumerate(('gc100','pgc33')):
        v=100*np.array(estimates[c+'/zero/native']['action_tv_from_full_native']['values'])
        ax[0].scatter(np.full(8,i),v,color=('#3379B5','#D87532')[i],alpha=.65,s=18)
        ax[0].plot([i-.15,i+.15],[v.mean(),v.mean()],color='black',lw=3)
    ax[0].set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],ylabel='Action probability moved (%)',title='Clearing carried state')
    for j,h in enumerate(('full','zero')):
        means=[100*estimates['pgc33/'+h+'/'+name]['action_tv_from_same_history_native']['mean'] for name in INPUTS[1:]]
        errors=[100*estimates['pgc33/'+h+'/'+name]['action_tv_from_same_history_native']['sem'] for name in INPUTS[1:]]
        ax[1].errorbar(np.arange(3)+(j-.5)*.1,means,yerr=errors,marker='o',capsize=3,label=('Carried state','Cleared state')[j])
    ax[1].set(xticks=range(3),xticklabels=['Live code','Goal code','Both'],ylabel='Action probability moved (%)',title='Removing prediction now')
    ax[1].legend(frameon=False,fontsize=8)
    for a in ax:a.spines[['top','right']].set_visible(False);a.set_ylim(bottom=0)
    fig.supxlabel('Matched observations; saved goal remains available; frozen models; 8 seeds',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'history_input_factorial.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Current input and carried recurrent state','',
           '![Matched-state sensitivity](history_input_factorial.png)','',
           'At recorded native-policy states, we changed two things independently: the h/c state carried into the RNN, and the predictive spatial code presented on the current step. The visual embedding, previous action/reward, and external saved goal message remained available. All weights stayed frozen.','',
           '| Intervention | Grid | PGC + grid |','| --- | ---: | ---: |',
           '| Clear h/c: action probability moved | '+ ' | '.join(f"{100*estimates[c+'/zero/native']['action_tv_from_full_native']['mean']:.1f}%" for c in ('gc100','pgc33'))+' |','',
           '| Remove prediction now | Carried state | Cleared state |','| --- | ---: | ---: |']
    for name,label in zip(INPUTS[1:],('Live code','Goal code','Both')):
        lines.append('| '+label+' | '+' | '.join(f"{100*estimates['pgc33/'+h+'/'+name]['action_tv_from_same_history_native']['mean']:.2f}%" for h in ('full','zero'))+' |')
    lines += ['', '**Interpretation:** these tests separate the direct effect of the current code from sensitivity to the preceding recurrent state. Any difference between the two columns means that the same predictive input is processed differently depending on the state already in the RNN. It does not identify which past information is useful.','',
              '**Limit:** clearing h/c is an unfamiliar operating condition for these frozen networks. A large reset effect is not evidence of better memory, and a small input effect after reset can result from saturated or poorly calibrated gates. These are action-distribution changes, not navigation scores. Native snapshots also differ between populations because each policy visited its own states.','',
              'TV means the fraction of action probability redistributed (0% unchanged, 100% disjoint distributions). Episode averages are weighted equally within each seed; the independent comparison units are eight paired seeds. Four planned sensitivity contrasts receive Holm correction separately for paired-t and exact sign-flip tests. The auxiliary turn/go diagnostic is descriptive.','',
              'Reproduce with `python scripts/lstm_history_input_factorial.py`. Requires the local 1B checkpoints and native `*_action_states.npz` archives. Raw arrays and weights remain local.','',
              '[Numerical results](summary.json) · [Protocol](protocol.json) · [Past-input history versus present input](shared_replay/README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


@torch.inference_mode()
def recover_replay_images(cfg, replay):
    """Reproduce the complete original CPU replay, including its RGB hash."""
    n, steps = int(replay['episodes']), int(replay['episode_len'])
    env = build_env(cfg, n, int(replay['replay_seed']), 'cpu', stagger=False, paired=True)
    obs = env.reset_all()
    rgb = torch.empty(steps*n, 3, 84, 84)
    image_hash = hashlib.sha256()
    for t in range(steps):
        sl = slice(t*n, (t+1)*n)
        for name in ('pos','hd','dir','vel'):
            assert np.array_equal(obs[name].numpy(), replay[name][sl]), (name,t)
        rgb[sl] = obs['rgb']
        image_hash.update(np.ascontiguousarray(obs['rgb'].numpy()))
        obs, _, _, info = env.step(torch.from_numpy(replay['action'][sl]))
        assert np.array_equal(info['hit'].numpy(), replay['hit'][sl])
    expected = json.loads((REPLAY_SOURCE/'summary.json').read_text())['protocol']['rgb_sha256']
    assert image_hash.hexdigest() == expected
    return rgb, image_hash.hexdigest()


@torch.inference_mode()
def run_shadow(seed, replay, rgb, out):
    start = time.monotonic()
    condition = 'pgc33'
    stem = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg = json.loads(stem.with_suffix('.json').read_text())
    checkpoint = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert checkpoint['frames'] == 1_000_000_000
    net = build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(checkpoint['policy'], strict=True)
    pop = build_population(cfg, 'cpu')
    n, steps = int(replay['episodes']), int(replay['episode_len'])
    embedding = torch.cat([net.embed(rgb[i:i+256]) for i in range(0,len(rgb),256)])
    current = pop.activity(*(torch.from_numpy(replay[k]) for k in ('pos','dir','vel')))
    current_plain = pop.activity(*(torch.from_numpy(replay[k]) for k in ('pos','dir','vel')), shift_scale=0)
    goal = pop.activity(*(torch.from_numpy(replay['memory_'+k]) for k in ('pos','dir','vel')))
    goal_plain = pop.activity(*(torch.from_numpy(replay['memory_'+k]) for k in ('pos','dir','vel')), shift_scale=0)
    known = torch.from_numpy(replay['goal_known'])[:,None]
    goal *= known; goal_plain *= known
    assert torch.equal(current[:,~pop.is_pgc], current_plain[:,~pop.is_pgc])
    assert torch.equal(goal[:,~pop.is_pgc], goal_plain[:,~pop.is_pgc])
    pa = torch.from_numpy(replay['prev_action'])
    pr = torch.from_numpy(replay['prev_reward'])
    # Exactly the same known-goal replay samples used by the prior action probe.
    rows = choose_rows(replay)
    next_index = replay['sample_index'][rows] + n
    lookup = {int(index):j for j,index in enumerate(next_index)}
    before = [torch.empty(4,len(rows),256), torch.empty(4,len(rows),256)]
    state = net.zero_state(4*n, 'cpu')
    native_saved = {'h':[], 'c':[], 'logits':[]}
    selected = replay['selected'].reshape(steps,n)
    for t in range(steps):
        sl=slice(t*n,(t+1)*n)
        for slot in range(n):
            index=t*n+slot
            if index in lookup:
                j=lookup[index]
                for target, source in zip(before,state):target[:,j]=source.reshape(4,n,256)[:,slot]
        code = torch.cat((current[sl], current_plain[sl], current[sl], current_plain[sl]))
        gol = torch.cat((goal[sl],goal[sl],goal_plain[sl],goal_plain[sl]))
        lo, _, state = net.step_from_embedding(embedding[sl].repeat(4,1),code,gol,pa[sl].repeat(4),pr[sl].repeat(4),state)
        if selected[t].any():
            for key, arr in zip(native_saved, (*state, lo)):
                native_saved[key].append(arr[:n][selected[t]].numpy().copy())
    old = dict(np.load(REPLAY_SOURCE/f'pgc33_s{seed}_activity.npz'))
    validation = {k:float(np.max(np.abs(np.concatenate(v)-old[k]))) for k,v in native_saved.items()}
    assert max(validation.values()) < 2e-4, validation
    arrays = {}; logits = {}; new_states = {}; outputs = {}
    for h, history in enumerate(INPUTS):
        for present in INPUTS:
            key=history+'/'+present
            code=current_plain if present in ('current_ordinary','both_ordinary') else current
            gol=goal_plain if present in ('goal_ordinary','both_ordinary') else goal
            lo, value, newstate=net.step_from_embedding(embedding[next_index],code[next_index],gol[next_index],
                pa[next_index],pr[next_index],(before[0][h],before[1][h]))
            logits[key]=lo.numpy(); arrays[key]=lo.softmax(1).numpy()
            new_states[key]=tuple(z.numpy() for z in newstate)
    episodes=replay['episode'][next_index]
    base=arrays['native/native']
    for key,p in arrays.items():
        history,present=key.split('/')
        outputs[key]=dict(action_tv_from_native=episode_average(tv(p,base),episodes),
            action_tv_from_same_history_native_input=episode_average(tv(p,arrays[history+'/native']),episodes),
            action_tv_from_same_input_native_history=episode_average(tv(p,arrays['native/'+present]),episodes),
            h_rms_from_native=episode_average(np.sqrt(((new_states[key][0]-new_states['native/native'][0])**2).mean(1)),episodes),
            c_rms_from_native=episode_average(np.sqrt(((new_states[key][1]-new_states['native/native'][1])**2).mean(1)),episodes),
            chosen_action_changed=episode_average((p.argmax(1)!=base.argmax(1)).astype(float),episodes))
    effects={}
    for name in INPUTS[1:]:
        interaction=arrays[name+'/'+name]-arrays[name+'/native']-arrays['native/'+name]+base
        effects[name]=dict(immediate_tv=outputs['native/'+name]['action_tv_from_native'],
            prior_history_tv=outputs[name+'/native']['action_tv_from_native'],
            sustained_tv=outputs[name+'/'+name]['action_tv_from_native'],
            interaction_half_l1=episode_average(.5*np.abs(interaction).sum(1),episodes))
    record=dict(seed=seed,condition='pgc33',samples=len(rows),episodes=len(np.unique(episodes)),
        variants=outputs,effects=effects,validation=validation,
        checkpoint_sha256=digest(stem.with_suffix('.pt')),activity_sha256=digest(REPLAY_SOURCE/f'pgc33_s{seed}_activity.npz'),
        elapsed_seconds=time.monotonic()-start)
    np.savez_compressed(out/f'pgc33_s{seed}_shadow.npz',**{k.replace('/','__'):p for k,p in arrays.items()},
        before_h=before[0].numpy(),before_c=before[1].numpy(),episodes=episodes,next_index=next_index)
    (out/f'pgc33_s{seed}.json').write_text(json.dumps(record,indent=2)+'\n')
    print('shadow',seed,'seconds',round(record['elapsed_seconds']),effects,flush=True)


def summarize_shadow(out):
    runs=[json.loads((out/f'pgc33_s{s}.json').read_text()) for s in range(1,9)]
    for record in runs:
        for values in record['effects'].values():
            if 'interaction_tv' in values:
                values['interaction_half_l1']=values.pop('interaction_tv')
        (out/f'pgc33_s{record["seed"]}.json').write_text(json.dumps(record,indent=2)+'\n')
    protocol=json.loads((out/'protocol.json').read_text())
    protocol['independent_grid_equivalence_checks']={str(s):verify_codes(s) for s in range(1,9)}
    protocol['interaction_metric']='Half-L1 norm of p(past changed,current changed)-p(past changed,current native)-p(past native,current changed)+p(both native). This signed four-condition contrast is not a conventional TV distance and need not be <=1.'
    replay=dict(np.load(REPLAY_SOURCE/'replay.npz'))
    selected_rows=choose_rows(replay)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    effects={name:{metric:estimate([r['effects'][name][metric] for r in runs])
                   for metric in runs[0]['effects'][name]} for name in INPUTS[1:]}
    comparisons={name+'_history_minus_immediate':paired(effects[name]['immediate_tv']['values'],
                 effects[name]['prior_history_tv']['values']) for name in INPUTS[1:]}
    correct(comparisons)
    # Exploratory covariance check on the same pre-input states, with equal
    # episode weights. This is not the original 3393-state PCA dataset.
    geometry_runs=[]
    geometry_names=INPUTS+('grid_native',)
    for seed in range(1,9):
        data=dict(np.load(out/f'pgc33_s{seed}_shadow.npz'))
        assert np.array_equal(data['next_index'],replay['sample_index'][selected_rows]+int(replay['episodes']))
        original=dict(np.load(REPLAY_SOURCE/f'pgc33_s{seed}_activity.npz'))
        timing_error={key:float(np.max(np.abs(data['before_'+key][0]-original[key][selected_rows]))) for key in ('h','c')}
        assert max(timing_error.values())<2e-4,timing_error
        runs[seed-1]['validation']['native_pre_input_h_vs_previous_saved_h']=timing_error['h']
        runs[seed-1]['validation']['native_pre_input_c_vs_previous_saved_c']=timing_error['c']
        episodes=data['episodes']; ids=np.unique(episodes)
        geometry={}
        weights=np.array([1/(len(ids)*np.sum(episodes==ep)) for ep in episodes])
        assert np.isclose(weights.sum(),1)
        grid_path=REPLAY_SOURCE/f'gc100_s{seed}_activity.npz'
        grid_original=dict(np.load(grid_path))
        states=[(name,data['before_h'][j]) for j,name in enumerate(INPUTS)]
        states.append(('grid_native',grid_original['h'][selected_rows]))
        for name,values in states:
            x=values.astype(float)
            means=np.stack([x[episodes==ep].mean(0) for ep in ids])
            grand=means.mean(0)
            centered=x-grand
            covariance=(centered*weights[:,None]).T@centered
            eigenvalues=np.maximum(np.linalg.eigvalsh(covariance),0)
            total=float(eigenvalues.sum())
            between=float(((means-grand)**2).sum(1).mean())
            assert 0<=between<=total+1e-10
            geometry[name]=dict(episode_fraction=between/total,pc10_fraction=float(eigenvalues[-10:].sum()/total),
                between_variance_per_neuron=between/x.shape[1],total_variance_per_neuron=total/x.shape[1])
        geometry_runs.append(dict(seed=seed,variants=geometry,grid_activity_sha256=digest(grid_path)))
    geometry_estimates={name:{m:estimate([r['variants'][name][m] for r in geometry_runs])
                             for m in geometry_runs[0]['variants'][name]} for name in geometry_names}
    geometry_comparisons={name+'_minus_native':paired(geometry_estimates['native']['episode_fraction']['values'],
        geometry_estimates[name]['episode_fraction']['values']) for name in INPUTS[1:]}
    correct(geometry_comparisons)
    population_comparisons={name+'_minus_native_grid':paired(geometry_estimates['grid_native']['episode_fraction']['values'],
        geometry_estimates[name]['episode_fraction']['values']) for name in ('native','both_ordinary')}
    correct(population_comparisons)
    summary=dict(protocol=protocol,effects=effects,
                 comparisons=comparisons,models=runs,
                 exploratory_state_geometry=dict(estimates=geometry_estimates,comparisons=geometry_comparisons,
                     population_comparisons=population_comparisons,models=geometry_runs,
                     definition='Equal episode weights; h before current input. Episode fraction is between-episode variance divided by total variance. Top10PC uses weighted centered covariance.',
                     status='Exploratory follow-up proposed after launching the shadow experiment, before viewing these covariance values. Three history-minus-native contrasts for episode fraction form a separate Holm family. After seeing those results, the existing grid activity reference was added at exactly the same rows; its two population comparisons form another exploratory Holm family.',
                     population_caveat='Shared external inputs do not mean identical RNN input vectors: trained visual encoders differ. Persistent activity differences can arise from the learned encoder and recurrent network; this does not isolate recurrent weights alone.'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7.8,3.6),layout='constrained')
    colors=['#3379B5','#D87532','#8563A8']
    for j,(metric,label) in enumerate((('immediate_tv','Change only this input'),
               ('prior_history_tv','Change only earlier inputs'),('sustained_tv','Change both'))):
        means=[100*effects[n][metric]['mean'] for n in INPUTS[1:]]
        sem=[100*effects[n][metric]['sem'] for n in INPUTS[1:]]
        ax.errorbar(np.arange(3)+(j-1)*.16,means,yerr=sem,marker='o',ls='none',color=colors[j],capsize=3,label=label)
    ax.set(xticks=range(3),xticklabels=['Live code','Goal code','Both'],ylabel='Action probability moved (%)',
           title='Predictive cues act now and through carried state',ylim=(0,None))
    ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Same physical replay and current observation · 8 frozen PGC models · mean ± SEM',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'past_present_factorial.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),layout='constrained')
    names=('grid_native','both_ordinary','native')
    labels=('Grid-trained\ngrid input','PGC-trained\ngrid input','PGC-trained\nmixed input')
    colors=('#3379B5','#D87532','#D87532')
    for ax,metric,scale,title,ylabel in zip(axes,
        ('episode_fraction','between_variance_per_neuron'),(100,1),
        ('Share of activity variance','Size of episode shifts'),
        ('Episode-associated variance (%)','Between-episode variance / neuron')):
        values=np.array([geometry_estimates[name][metric]['values'] for name in names])*scale
        for seed in range(8):ax.plot(range(3),values[:,seed],color='.7',lw=.8,alpha=.65)
        for j,color in enumerate(colors):
            ax.errorbar(j,values[j].mean(),yerr=values[j].std(ddof=1)/np.sqrt(8),fmt='o',color=color,capsize=4,ms=7,lw=2)
        ax.set(xticks=range(3),xticklabels=labels,ylabel=ylabel,title=title,ylim=(0,None))
        ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Same physical replay · 8 paired seeds · lines: seeds · points: mean ± SEM',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'matched_code_covariance.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Predictive inputs now versus in the past','',
           '![Past and current predictive-input effects](past_present_factorial.png)','',
           'We replayed the same images, movements, pickups, previous actions and rewards through each frozen PGC RNN. Four copies received native codes, ordinary live codes, ordinary goal codes, or both ordinary codes. They all started normally at zero and kept their own resulting h/c states. At selected moments, we independently changed the current input. No state was cleared or moved between trained networks.','',
           '| Remove prediction from | Only this input | Only previous inputs | Previous + current |',
           '| --- | ---: | ---: | ---: |']
    for name,label in zip(INPUTS[1:],('Live code','Goal code','Both')):
        lines.append('| '+label+' | '+' | '.join(f'{100*effects[name][m]["mean"]:.2f}%' for m in ('immediate_tv','prior_history_tv','sustained_tv'))+' |')
    lines+=['','The numbers are the fraction of action probability redistributed. **Only previous inputs** means that the current predictive input has already been restored: any remaining difference must reach the current action through the RNN state produced by its earlier inputs. **Only this input** holds that earlier state fixed. The columns are not additive; recurrent state and current input interact. None is a fraction of navigation performance explained.','',
            'This cleanly separates two computational routes. It does not establish whether the history-dependent part contains useful goal or movement information, whether it improves navigation, or whether it causes the larger episode-average activity shifts. The policies do not control this scripted replay; all branches are forced to receive identical physical experience. Changing the past codes may still move a frozen network away from its trained operating regime.','',
            'The external goal snapshot identity and its refresh times are identical in every branch. When prediction is removed, only the 169 predictive channels lose their 20 cm look-ahead shift; all 343 ordinary channels stay identical. Goal messages remain available on every step after the first pickup.','',
            'An exploratory check asks whether the histories also change episode-associated covariance, measured immediately before the current input. These are '+str(runs[0]['samples'])+' sampled states, with equal episode weights; they are not the original pooled PCA numbers.','',
            '![Activity covariance with matched ordinary spatial codes](matched_code_covariance.png)','',
            '| Previous input history | Episode-associated variance | First 10 PCs |','| --- | ---: | ---: |']
    for name,label in zip(('grid_native',)+INPUTS,('Grid-trained: native','PGC-trained: native','PGC-trained: ordinary live','PGC-trained: ordinary goal','PGC-trained: both ordinary')):
        lines.append('| '+label+' | '+' | '.join(f'{100*geometry_estimates[name][m]["mean"]:.1f}%' for m in ('episode_fraction','pc10_fraction'))+' |')
    validation_max=max(max(r['validation'].values()) for r in runs)
    validation_text='exactly' if validation_max==0 else f'within {validation_max:.2g}'
    absolute_between=[geometry_estimates[name]['between_variance_per_neuron']['mean'] for name in ('native','goal_ordinary')]
    absolute_total=[geometry_estimates[name]['total_variance_per_neuron']['mean'] for name in ('native','goal_ordinary')]
    lines+=['','A smaller fraction does not mean the episode offsets were removed. With ordinary goal history, absolute between-episode variance changes only from '+f'{absolute_between[0]:.5f} to {absolute_between[1]:.5f}'+', while total variance increases from '+f'{absolute_total[0]:.5f} to {absolute_total[1]:.5f}'+'. These are variances per hidden unit, averaged across seeds.','',
            'The grid reference uses the same saved replay rows. When both histories use ordinary codes, their external inputs match, but their trained visual encoders and recurrent networks differ. Persistent activity differences therefore cannot be assigned to recurrent weights alone. This covariance intervention still does not show that episode offsets are useful for behavior; the goal cue is supplied externally throughout each episode.','',
            'The full replay RGB hash exactly matches the earlier recording. Native reconstructed h/c and logits match saved activity '+validation_text+f'. The {runs[0]["samples"]} selected known-goal states cover all {runs[0]["episodes"]} replay episodes; each episode receives equal weight within a seed. Three planned history-versus-immediate sensitivity contrasts receive Holm correction.','',
            'Reproduce with `python scripts/lstm_history_input_factorial.py --shared-replay`. Requires the local 1B checkpoints and original replay/activity archives. Raw arrays and weights remain local.','',
            '[Numerical results](summary.json) · [Protocol](protocol.json) · [Native-state reset factorial](../README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def shared_replay(args):
    out=args.out/'shared_replay';out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        summarize_shadow(out);return
    replay=dict(np.load(REPLAY_SOURCE/'replay.npz'))
    cfg=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    protocol=dict(models='Eight frozen PGC+grid 1B LSTMs; grid codes have no predictive shift to remove.',
        replay='Original 16-episode scripted replay; policies do not control actions.',
        history_variants=list(INPUTS),current_variants=list(INPUTS),
        intervention='Full 4x4 history/current factorial. Each prior history starts at zero and processes the entire preceding physical replay with native or ordinary current/goal codes. Current-step inputs then independently vary. No h/c resets or cross-network states.',
        retained='Identical physical trajectory, RGB, previous actions/rewards, goal snapshot identities and refresh times, ordinary code slots, and frozen weights.',
        selection='Every eighth selected known-goal replay sample, taking its actual next input; identical to the existing shared-replay action probe.',
        statistics='Eight seed means; equal episode weights within seed. Three planned prior-history TV minus immediate TV contrasts, with Holm correction separately for t and exact sign flips.',
        caveats=['Scripted off-policy action sensitivity, not navigation performance.','Changed past inputs can affect the operating regime; history sensitivity does not by itself prove useful memory.','The external goal message is present every step after discovery.'],
        replay_sha256=digest(REPLAY_SOURCE/'replay.npz'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rgb,rgb_hash=recover_replay_images(cfg,replay)
    protocol['exact_replay_rgb_sha256']=rgb_hash
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Exact full RGB replay reconstructed.',flush=True)
    for seed in args.seeds:run_shadow(seed,replay,rgb,out)
    if all((out/f'pgc33_s{s}.json').exists() for s in range(1,9)):summarize_shadow(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=SOURCE)
    ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--seeds',type=int,nargs='+',default=list(range(1,9)))
    ap.add_argument('--conditions',nargs='+',default=['gc100','pgc33']);ap.add_argument('--summarize-only',action='store_true')
    ap.add_argument('--shared-replay',action='store_true',help='Prior predictive-input histories crossed with present input on identical physical replay.')
    args=ap.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    if args.shared_replay:
        shared_replay(args);return
    protocol=dict(source=str(args.source.relative_to(ROOT)), models='Frozen 1B checkpoints; 8 paired seeds',
        selection='Existing native-policy snapshots, known-goal states at decisions 600/900/1200 and slots 0..31.',
        interventions='Within each model: retain native pre-input h/c or clear both to zero; independently retain or deshift the current live code, goal code, or both. Only 169 predictive slots change.',
        retained='Exact current visual embedding, external goal snapshot identity, previous action/reward, physical state, all network weights. No h/c cross-network swaps.',
        history_semantics='Zero means no carried h/c before this current input, not no prior information: external goal memory and explicit previous action/reward remain.',
        statistics='Four planned sensitivity contrasts; seed means use equal sampled-episode weighting. Holm correction across four contrasts, separately for t and exact sign-flip tests.',
        limitations=['Cleared states are out of training distribution.','Native states differ across populations.','Action changes are not navigation performance or an optimal-action test.'])
    if not args.summarize_only:
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for c in args.conditions:
            for s in args.seeds:run_native(c,s,args.source,args.out)
    if all((args.out/f'{c}_s{s}.json').exists() for c in ('gc100','pgc33') for s in range(1,9)):
        summarize(args.out)


if __name__=='__main__':main()
