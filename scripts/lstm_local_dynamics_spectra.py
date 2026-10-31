"""Local spectra of frozen LSTM state-update Jacobians, with fixed inputs.

Uses saved post-update states and the actual next input in the shared replay.
These are local derivatives, not covariance eigenvalues or global stability.
CPU only; no training, policy rollouts, or source-recording modifications.
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
from scipy import linalg
import torch
import torch.nn.functional as F

from code.common import CodeSource, build_env, build_policy, build_population
from summarize_lstm_code_swap import estimate, paired, holm

SOURCE = ROOT / 'results/lstm_variable_1B'
OUT = ROOT / 'results/lstm_local_dynamics_spectra_1B'
COLORS = {'gc100': '#3379B5', 'pgc33': '#D87532'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_states(replay):
    saved = replay['sample_index']
    rows = []
    for episode in range(int(replay['episodes'])):
        candidates = np.flatnonzero((replay['episode'][saved] == episode)
            & replay['goal_known'][saved]
            & (replay['step'][saved] < int(replay['episode_len']) - 1))
        for target in (700, 1000):
            row = int(candidates[np.argmin(np.abs(replay['step'][saved[candidates]] - target))])
            assert row not in rows
            rows.append(row)
    return np.array(rows, dtype=int)


@torch.inference_mode()
def recover_images(cfg, replay, selected_rows):
    """Stream the original actions; keep only the requested next-step images."""
    n = int(replay['episodes'])
    next_index = replay['sample_index'][selected_rows] + n
    wanted = {int(index): j for j, index in enumerate(next_index)}
    env = build_env(cfg, n, int(replay['replay_seed']), 'cpu', stagger=False, paired=True)
    obs = env.reset_all()
    rgb = torch.empty(len(next_index), 3, 84, 84)
    image_hash = hashlib.sha256()
    action = torch.from_numpy(replay['action'].reshape(-1, n))
    for t in range(int(replay['episode_len'])):
        sl = slice(t*n, (t+1)*n)
        for name in ('pos', 'dir', 'vel', 'hd'):
            assert np.array_equal(obs[name].numpy(), replay[name][sl]), (name, t)
        pixels = obs['rgb'].numpy()
        image_hash.update(np.ascontiguousarray(pixels))
        for e in range(n):
            index = t*n + e
            if index in wanted:
                rgb[wanted[index]] = obs['rgb'][e]
        obs, reward, done, info = env.step(action[t])
        assert np.array_equal(info['hit'].numpy(), replay['hit'][sl])
    expected = json.loads((SOURCE / 'summary.json').read_text())['protocol']['rgb_sha256']
    assert image_hash.hexdigest() == expected, 'Regenerated RGB differs from original replay'
    return rgb, next_index, image_hash.hexdigest()


def local_jacobian(net, x, h, c):
    """Jacobian in state order (h,c), evaluated in float64."""
    ih = net.lstm.weight_ih.detach().double()
    hh = net.lstm.weight_hh.detach().double()
    pre = F.linear(x, ih, net.lstm.bias_ih.detach().double()) + F.linear(h, hh, net.lstm.bias_hh.detach().double())
    i, f, g, o = pre.chunk(4)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    wi, wf, wg, wo = hh.chunk(4)
    new_c = f*c + i*g
    a = (c*f*(1-f))[:, None]*wf + (g*i*(1-i))[:, None]*wi + (i*(1-g*g))[:, None]*wg
    tangent = new_c.tanh()
    exposure = o*(1-tangent*tangent)
    dhdh = (tangent*o*(1-o))[:, None]*wo + exposure[:, None]*a
    dhdc = torch.diag(exposure*f)
    j = torch.cat((torch.cat((dhdh, dhdc), 1), torch.cat((a, torch.diag(f)), 1)), 0)
    return j


def autograd_check(net, x, h, c, analytic):
    ih, hh = net.lstm.weight_ih.detach().double(), net.lstm.weight_hh.detach().double()
    bi, bh = net.lstm.bias_ih.detach().double(), net.lstm.bias_hh.detach().double()
    def update(state):
        old_h, old_c = state.chunk(2)
        i, f, g, o = (F.linear(x, ih, bi) + F.linear(old_h, hh, bh)).chunk(4)
        new_c = f.sigmoid()*old_c + i.sigmoid()*g.tanh()
        return torch.cat((o.sigmoid()*new_c.tanh(), new_c))
    auto = torch.autograd.functional.jacobian(update, torch.cat((h,c)), vectorize=True)
    error = float((auto-analytic).abs().max())
    assert error < 1e-10, error
    return error


def run_model(condition, seed, replay, rows, rgb, next_index, out):
    start = time.monotonic()
    stem = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M'
    cfg = json.loads(stem.with_suffix('.json').read_text())
    ckpt = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    net = build_policy(cfg, 'cpu').eval().requires_grad_(False)
    net.load_state_dict(ckpt['policy'], strict=True)
    pop = build_population(cfg, 'cpu')
    source = CodeSource(cfg, pop, 'cpu')
    path = SOURCE / f'{condition}_s{seed}_activity.npz'
    with np.load(path) as data:
        h = torch.from_numpy(data['h'][rows]).double()
        c = torch.from_numpy(data['c'][rows]).double()
    with torch.inference_mode():
        embedding = net.embed(rgb)
        pos, direction, vel = [torch.from_numpy(replay[k][next_index]) for k in ('pos','dir','vel')]
        current = source(pos, direction, vel)
        memory_args = [torch.from_numpy(replay[f'memory_{k}'][next_index]) for k in ('pos','dir','vel')]
        goal = source.goal(*memory_args)
        known = torch.from_numpy(replay['goal_known'][next_index])[:, None]
        goal *= known
        pa = torch.from_numpy(replay['prev_action'][next_index])
        pr = torch.from_numpy(replay['prev_reward'][next_index])
        inputs = {'intact': net._concat(embedding,current,goal,pa,pr,None).double()}
        if condition == 'pgc33':
            plain_goal = pop.activity(*memory_args, shift_scale=0) * known
            inputs['goal_grid_same_state'] = net._concat(embedding,current,plain_goal,pa,pr,None).double()
    results, archive = {}, {}
    max_autograd_error = 0.
    for variant, x in inputs.items():
        points, eigenvalues, singular_values = [], [], []
        for j in range(len(rows)):
            matrix = local_jacobian(net, x[j], h[j], c[j])
            if seed == 1 and j == 0:
                max_autograd_error = max(max_autograd_error,autograd_check(net,x[j],h[j],c[j],matrix))
            array = matrix.numpy()
            ev = linalg.eigvals(array, check_finite=False)
            sv = linalg.svdvals(array, check_finite=False)
            absolute = np.abs(ev)
            points.append(dict(episode=int(replay['episode'][next_index[j]]),
                input_step=int(replay['step'][next_index[j]]), spectral_radius=float(absolute.max()),
                largest_singular_value=float(sv[0]), fraction_eigenvalues_above_one=float(np.mean(absolute>1)),
                fraction_eigenvalues_above_1_01=float(np.mean(absolute>1.01)),
                singular_to_radius_ratio=float(sv[0]/absolute.max()),
                fraction_complex=float(np.mean(np.abs(ev.imag)>1e-8))))
            eigenvalues.append(ev)
            singular_values.append(sv)
        archive[f'{variant}_eigenvalues'] = np.array(eigenvalues)
        archive[f'{variant}_singular_values'] = np.array(singular_values)
        metrics = [k for k in points[0] if k not in ('episode','input_step')]
        results[variant] = dict(points=points,
            means={k:float(np.mean([p[k] for p in points])) for k in metrics},
            mean_ordered_eigenvalue_modulus=np.sort(np.abs(eigenvalues),axis=1)[:,::-1].mean(0).tolist(),
            mean_ordered_singular_values=np.mean(singular_values,axis=0).tolist())
    np.savez_compressed(out/f'{condition}_s{seed}_spectra.npz', **archive)
    result = dict(condition=condition,seed=seed,points=len(rows),checkpoint_frames=ckpt['frames'],
        checkpoint_sha256=digest(stem.with_suffix('.pt')), activity_sha256=digest(path),
        variants=results,autograd_max_error=max_autograd_error,elapsed_seconds=time.monotonic()-start)
    (out/f'{condition}_s{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{condition} s{seed}: rho={results["intact"]["means"]["spectral_radius"]:.4f}, '
          f'opnorm={results["intact"]["means"]["largest_singular_value"]:.3f}; {time.monotonic()-start:.1f}s',flush=True)


def summarize(out, protocol):
    runs={(c,s):json.loads((out/f'{c}_s{s}.json').read_text()) for c in ('gc100','pgc33') for s in range(1,9)}
    original={(r['condition'],r['seed']):r for r in json.loads((SOURCE/'summary.json').read_text())['models']}
    for (condition,seed),r in runs.items():
        assert r['checkpoint_sha256'] == original[condition,seed]['checkpoint_sha256']
        assert r['activity_sha256'] == digest(SOURCE/f'{condition}_s{seed}_activity.npz')
        assert r['checkpoint_sha256'] == digest(ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt')
    metrics = list(runs['gc100',1]['variants']['intact']['means'])
    estimates = {m:{c:estimate([runs[c,s]['variants']['intact']['means'][m] for s in range(1,9)])
                   for c in ('gc100','pgc33')} for m in metrics}
    primary={m:paired(estimates[m]['gc100']['values'],estimates[m]['pgc33']['values'])
             for m in ('spectral_radius','largest_singular_value')}
    holm(primary)
    ordered=sorted(primary,key=lambda k:primary[k]['exact_signflip_p'])
    maximum=0.
    for j,key in enumerate(ordered):
        maximum=max(maximum,min(1.,(len(primary)-j)*primary[key]['exact_signflip_p']))
        primary[key]['holm_signflip_p']=maximum
    same_state={m:paired([runs['pgc33',s]['variants']['intact']['means'][m] for s in range(1,9)],
                        [runs['pgc33',s]['variants']['goal_grid_same_state']['means'][m] for s in range(1,9)])
                for m in ('spectral_radius','largest_singular_value')}
    result=dict(protocol=protocol,estimates=estimates,primary_comparisons=primary,
        exploratory_same_state_goal_deshift=same_state,
        autograd_max_error=max(r['autograd_max_error'] for r in runs.values()),models=list(runs.values()),
        validation=dict(original_checkpoint_hashes_match=True,source_activity_hashes_unchanged=True),
        interpretation='These spectra describe infinitesimal one-step state perturbations while external input is fixed. They do not establish long-run stability, attractors, persistent memory, or a causal explanation of navigation performance. Native models have different weights and saved states; the same-state PGC goal swap changes only next-step goal input.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for c,color in COLORS.items():
        spectra=np.array([runs[c,s]['variants']['intact']['mean_ordered_eigenvalue_modulus'] for s in range(1,9)])
        x=np.arange(1,513)
        axes[0].plot(x,spectra.mean(0),color=color,label='Grid' if c=='gc100' else 'PGC + grid')
        axes[0].fill_between(x,spectra.mean(0)-spectra.std(0,ddof=1)/np.sqrt(8),
                            spectra.mean(0)+spectra.std(0,ddof=1)/np.sqrt(8),color=color,alpha=.15)
    axes[0].axhline(1,color='#999999',ls='--',lw=1)
    axes[0].set(xlabel='Eigenvalue rank',ylabel='Eigenvalue magnitude',title='Local update spectrum')
    axes[0].legend(frameon=False)
    for ax,metric,title in zip(axes[1:],('spectral_radius','largest_singular_value'),('Largest eigenvalue magnitude','Largest one-step amplification')):
        a,b=[estimates[metric][c]['values'] for c in ('gc100','pgc33')]
        for aa,bb in zip(a,b): ax.plot([0,1],[aa,bb],color='#c7c7c7',lw=1)
        for j,c in enumerate(('gc100','pgc33')):
            values=estimates[metric][c]['values']; r=estimates[metric][c]
            ax.scatter(np.full(8,j),values,color=COLORS[c],s=20)
            ax.errorbar(j+.09,r['mean'],yerr=r['sem'],color=COLORS[c],fmt='s',capsize=3)
        ax.set(xticks=[0,1],xticklabels=['Grid','PGC + grid'],title=title,xlim=(-.25,1.3))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('32 matched states per model; 8 paired seeds. Fixed-input local derivatives, not global stability.',fontsize=9)
    fig.savefig(out/'local_update_spectra.png',dpi=180)
    fig.savefig(out/'local_update_spectra.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,m,title in zip(axes,('spectral_radius','largest_singular_value'),('Largest eigenvalue magnitude','Largest one-step amplification')):
        r=same_state[m]
        for a,b in zip(r['a'],r['b']): ax.plot([0,1],[a,b],color='#c9c9c9',lw=1)
        for i,(key,color) in enumerate([('a',COLORS['pgc33']),('b','#939CA4')]):
            v=np.array(r[key]);ax.scatter(np.full(8,i),v,color=color,s=23)
            ax.errorbar(i+.08,v.mean(),yerr=v.std(ddof=1)/np.sqrt(8),color=color,fmt='s',capsize=3)
        ax.set(xticks=[0,1],xticklabels=['Predictive goal','Ordinary goal'],title=title,xlim=(-.25,1.3))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('PGC-trained RNNs: same saved state and next input, except goal code. Dots: training-seed means.',fontsize=9)
    fig.savefig(out/'goal_input_sensitivity.png',dpi=180)
    fig.savefig(out/'goal_input_sensitivity.pdf')
    plt.close(fig)
    lines = ['# Local RNN-update spectra', '',
        '**This measures how a small change in the current RNN state affects the next state. It is different from PCA, which measures how activity is distributed.**', '',
        '![Local update eigenvalues and amplification](local_update_spectra.png)', '',
        '| Quantity, mean across eight training seeds | Grid | PGC + grid |',
        '|---|---:|---:|']
    for label,m,precision in [('Largest eigenvalue magnitude','spectral_radius',6),
                              ('Largest one-step amplification','largest_singular_value',3)]:
        lines.append(f'| {label} | {estimates[m]["gc100"]["mean"]:.{precision}f} | {estimates[m]["pgc33"]["mean"]:.{precision}f} |')
    lines += ['', 'The largest eigenvalue magnitudes are very close to one. The largest singular values are much larger: some combinations of hidden/cell-state changes can be amplified over one step even when the eigenvalue magnitudes are near one. These are worst-case local perturbations, not typical noise or measured action changes.', '',
        '| PGC minus grid | Difference | 95% paired interval | Holm t p | Holm exact p |',
        '|---|---:|---|---:|---:|']
    for label,m in [('Largest eigenvalue magnitude','spectral_radius'),('Largest one-step amplification','largest_singular_value')]:
        r=primary[m]
        lines.append(f'| {label} | {r["difference"]:+.6f} | [{r["ci95"][0]:+.6f}, {r["ci95"][1]:+.6f}] | {r["holm_t_p"]:.4g} | {r["holm_signflip_p"]:.4g} |')
    lines += ['', 'The PGC spectral radius is slightly larger in all eight pairs, but by only about 0.00045. The exact sign-flip and paired-t tests disagree at the 5% threshold; the table reports both. The native-model amplification difference is uncertain. Neither comparison establishes a different global dynamical regime.', '',
        '## Same-state predictive-goal removal', '',
        '![Local sensitivity after changing only the next goal input](goal_input_sensitivity.png)', '',
        'For PGC models, we also removed the predictive shift from the next goal input while holding the saved state and every other next input fixed. This is an instantaneous input intervention; the preceding history stays intact.', '',
        '| Quantity | Change after removing predictive goal shift | 95% paired interval |',
        '|---|---:|---|']
    for label,m in [('Largest eigenvalue magnitude','spectral_radius'),('Largest one-step amplification','largest_singular_value')]:
        r=same_state[m]
        lines.append(f'| {label} | {r["difference"]:+.6f} | [{r["ci95"][0]:+.6f}, {r["ci95"][1]:+.6f}] |')
    lines += ['', '**Removing the predictive goal shift increases worst-case one-step amplification in all eight seeds: 5.30 → 7.40.** Its exploratory paired-t p is 0.000168 and exact sign-flip p is 0.00781; the spectral-radius change is uncertain. These comparisons describe how the input changes local sensitivity. They do not show that the sensitivity change causes the navigation deficit.', '',
        '## What was measured', '',
        'We used 32 matched states per model: two saved moving states in each of 16 scripted episodes, selected near decisions 700 and 1000 without inspecting the neural result. Each saved state is the output after input t; its Jacobian uses the actual next input at t+1. Both h and c are included, giving a 512×512 derivative.', '',
        'The original replay was reconstructed exactly, including its full image hash, while retaining only the 32 required next-step images. Analytic derivatives matched automatic differentiation to '+f'{result["autograd_max_error"]:.2g}'+'. No model was retrained and no source recording was changed.', '',
        '**Limits:** fixed-input, one-step derivatives cannot establish long-run stability, attractors, or memory duration. Real trajectories have changing inputs and successive Jacobians. Singular-value amplification depends on the relative scaling of h and c. Raw eigenvectors are not equated between separately trained networks.', '',
        '[Numerical results](summary.json) · [Protocol](protocol.json) · [Covariance spectra](../lstm_eigenspectra_1B/README.md)', '',
        '```sh', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_local_dynamics_spectra.py', '```', '',
        'Reproduction requires the local 1B checkpoints and saved replay/activity arrays. Per-model JSON files retain sampled-state measurements; local NumPy archives retain the complete complex eigenvalues and singular spectra.', '']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'estimates':estimates,'primary':primary,'same_state':same_state},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'replay.npz') as file: replay={k:file[k] for k in file.files}
    selected=select_states(replay)
    protocol=dict(seeds=args.seeds,conditions=['gc100','pgc33'],points_per_model=len(selected),
        selection='For each of 16 episodes, choose saved moving known-goal states nearest decision 700 and 1000; use their actual next input. Selection is independent of model activity or Jacobians.',
        selected_saved_rows=selected.tolist(),selected_replay_indices=replay['sample_index'][selected].tolist(),
        state='Saved h_t,c_t after processing input t; differentiate next update at t+1 with next external input held fixed. State order (h,c), 512 dimensions.',
        arithmetic='Float64 analytic LSTM Jacobian; original weights/states/inputs are float32 values promoted to double. Autograd checks at first selected point of both seed-1 native models and the PGC goal-deshift control.',
        primary=['Paired PGC-minus-grid mean local spectral radius','Paired PGC-minus-grid mean largest singular value'],
        statistics='Eight training-seed means, not individual sampled states, are replicates. Two-sided paired t and exact sign-flip; Holm within two primary comparisons. Same-state goal-deshift comparisons are exploratory and unadjusted.',
        goal_swap='PGC only: same saved h,c, same next vision/current code/action/reward, but predictive slots in next goal code evaluated with zero shift. This is instantaneous, not a new history.',
        limitations=['The replay is scripted; policies do not select its actions.',
                    'A local eigenvalue magnitude above one is not proof of an unstable trajectory; time-varying inputs and Jacobian products matter.',
                    'Largest singular values depend on the chosen Euclidean scaling of concatenated h and c and concern worst-case perturbations, not typical noise.',
                    'These are update-Jacobian spectra, not hidden-activity covariance spectra or raw recurrent-weight spectra.',
                    'Eigenvectors of nonsymmetric Jacobians are not compared across independently trained neuron coordinates.'],
        replay_sha256=digest(SOURCE/'replay.npz'))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cfg=json.loads((ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text())
    rgb,next_index,rgb_hash=recover_images(cfg,replay,selected)
    protocol['reconstructed_full_rgb_sha256']=rgb_hash
    protocol['exact_replay_reconstruction']=True
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(f'Reconstructed original replay exactly; retained {len(rgb)} next-input images.',flush=True)
    for seed in args.seeds:
        for condition in ('gc100','pgc33'):
            path=args.out/f'{condition}_s{seed}.json'
            if path.exists():
                print(f'Using completed {condition} s{seed}',flush=True)
            else: run_model(condition,seed,replay,selected,rgb,next_index,args.out)
    if all((args.out/f'{c}_s{s}.json').exists() for c in ('gc100','pgc33') for s in range(1,9)):
        summarize(args.out,protocol)


if __name__=='__main__': main()
