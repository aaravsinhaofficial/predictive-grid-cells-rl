"""Covariance spectra, eigenvector subspaces, and action-readout alignment.

Only reads saved hidden activity and frozen checkpoint actor weights. No model
training, policy rollouts, or intervention is performed by this script.
"""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/lstm_variable_1B'
OUT = ROOT / 'results/lstm_eigenspectra_1B'
CONDITIONS = ('gc100', 'pgc33')
K = 10
COLORS = {'gc100': '#3379B5', 'pgc33': '#D87532'}
LABELS = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}
PRIMARY = ('total_participation_ratio', 'within_participation_ratio',
           'top10_episode_subspace_overlap', 'top10_actor_variance_share',
           'episode_actor_variance_share')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covariances(h, episodes):
    x = h - h.mean(0)
    between = np.zeros_like(x)
    for episode in np.unique(episodes):
        mask = episodes == episode
        between[mask] = x[mask].mean(0)
    within = x - between
    n = len(h)
    total, bcov, wcov = x.T @ x / n, between.T @ between / n, within.T @ within / n
    error = float(np.linalg.norm(total - bcov - wcov) / np.linalg.norm(total))
    assert error < 1e-12
    return dict(total=total, between=bcov, within=wcov), error


def eigen(covariance):
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    assert values.min() > -1e-10 * max(values.max(), 1.)
    values = values.clip(0)
    assert np.allclose(vectors @ np.diag(values) @ vectors.T, covariance, atol=1e-10)
    return values, vectors


def spectral_summary(values):
    total = values.sum()
    proportions = values / total
    positive = proportions > 0
    return dict(eigenvalues=values.tolist(), proportions=proportions.tolist(),
        cumulative=proportions.cumsum().tolist(), total_variance=float(total),
        numerical_rank=int(np.sum(values > values[0] * 1e-10)),
        participation_ratio=float(1 / np.sum(proportions ** 2)),
        effective_rank=float(np.exp(-np.sum(proportions[positive] * np.log(proportions[positive])))),
        dimensions_90pct=int(np.searchsorted(proportions.cumsum(), .9) + 1),
        top10_share=float(proportions[:K].sum()),
        pc10_pc11_relative_gap=float((values[K-1] - values[K]) / values[K-1]))


def subspace(a, b):
    cosines = np.linalg.svd(a.T @ b, compute_uv=False).clip(0, 1)
    # For unequal dimensions, this is the fraction of a's squared basis
    # length captured by b; the symmetric canonical mean is stored separately.
    return dict(overlap_fraction_of_first=float(np.sum(cosines ** 2) / a.shape[1]),
                mean_cosine_squared=float(np.mean(cosines ** 2)),
                principal_angles_degrees=np.degrees(np.arccos(cosines)).tolist(),
                first_dimensions=a.shape[1], second_dimensions=b.shape[1])


def split_stability(h, episodes, steps, episode_splits):
    early = np.zeros(len(h), bool)
    for episode in np.unique(episodes):
        mask = episodes == episode
        ordered = np.flatnonzero(mask)[np.argsort(steps[mask])]
        early[ordered[:len(ordered)//2]] = True
    outputs = {}
    for name, masks in [('early_late', [(early, ~early)]),
                        ('episode_halves', [(np.isin(episodes, a), ~np.isin(episodes, a)) for a in episode_splits])]:
        records = []
        for a, b in masks:
            first, _ = covariances(h[a], episodes[a])
            second, _ = covariances(h[b], episodes[b])
            entry = dict(samples=[int(a.sum()), int(b.sum())])
            for kind in ('total', 'within', 'between'):
                va, ua = eigen(first[kind])
                vb, ub = eigen(second[kind])
                rank = min(K, int(np.sum(va > va[0]*1e-10)), int(np.sum(vb > vb[0]*1e-10)))
                entry[kind] = subspace(ua[:, :rank], ub[:, :rank])
            records.append(entry)
        outputs[name] = dict(splits=records, mean_overlap={kind:float(np.mean([
            r[kind]['overlap_fraction_of_first'] for r in records])) for kind in ('total','within','between')})
    return outputs


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    sem = stats.sem(d)
    ci = stats.t.interval(.95, len(d)-1, loc=d.mean(), scale=sem) if sem else (d.mean(), d.mean())
    signs = np.asarray(list(itertools.product((-1,1), repeat=len(d))))
    return dict(grid=a.tolist(), pgc=b.tolist(), grid_mean=float(a.mean()), pgc_mean=float(b.mean()),
                difference=float(d.mean()), differences=d.tolist(), ci95=list(map(float,ci)),
                t_p=float(stats.ttest_rel(b,a).pvalue) if sem else (1. if d.mean()==0 else 0.),
                exact_p=float(np.mean(np.abs((signs*d).mean(1)) >= abs(d.mean())-1e-12)),
                pgc_greater_seeds=int((d>0).sum()))


def mean_sem(rows, condition, getter):
    array = np.asarray([getter(r) for r in rows if r['condition'] == condition])
    return array.mean(0), stats.sem(array, axis=0)


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def plots(rows):
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for j,kind in enumerate(('total','between','within')):
        for condition in CONDITIONS:
            mean, se = mean_sem(rows, condition, lambda r: np.asarray(r['spectra'][kind]['eigenvalues']))
            length = 15 if kind=='between' else 256
            x = np.arange(1,length+1)
            axes[0,j].plot(x,mean[:length],color=COLORS[condition],label=LABELS[condition])
            axes[0,j].fill_between(x,np.maximum(mean[:length]-se[:length],1e-10),mean[:length]+se[:length],color=COLORS[condition],alpha=.15)
            mean, se = mean_sem(rows, condition, lambda r: np.asarray(r['spectra'][kind]['cumulative'])*100)
            axes[1,j].plot(x,mean[:length],color=COLORS[condition])
            axes[1,j].fill_between(x,mean[:length]-se[:length],mean[:length]+se[:length],color=COLORS[condition],alpha=.15)
        axes[0,j].set(yscale='log',title={'total':'All activity','between':'Episode averages','within':'Within episodes'}[kind],xlabel='Eigenvalue rank')
        axes[1,j].set(xlabel='Number of eigenvectors',ylim=(0,101))
    axes[0,0].set_ylabel('Eigenvalue (log scale)')
    axes[1,0].set_ylabel('Cumulative variance (%)')
    axes[0,0].legend(frameon=False)
    fig.supxlabel('Mean ± SEM over 8 seeds. Episode-average covariance has at most 15 nonzero eigenvalues.',fontsize=10)
    save(fig,'eigenvalue_spectra')

    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for condition in CONDITIONS:
        mean,se=mean_sem(rows,condition,lambda r:np.asarray(r['eigenvectors']['between_fraction_per_total_pc'][:30])*100)
        x=np.arange(1,31)
        axes[0].plot(x,mean,color=COLORS[condition],label=LABELS[condition])
        axes[0].fill_between(x,mean-se,mean+se,color=COLORS[condition],alpha=.15)
        mean,se=mean_sem(rows,condition,lambda r:np.asarray(r['eigenvectors']['actor_variance_cumulative'])*100)
        x=np.arange(1,257)
        axes[1].plot(x,mean,color=COLORS[condition])
        axes[1].fill_between(x,mean-se,mean+se,color=COLORS[condition],alpha=.15)
        mean,se=mean_sem(rows,condition,lambda r:np.asarray(r['eigenvectors']['actor_weight_cumulative'])*100)
        axes[2].plot(x,mean,color=COLORS[condition])
        axes[2].fill_between(x,mean-se,mean+se,color=COLORS[condition],alpha=.15)
    axes[0].set(title='What each PC contains',xlabel='Total-activity PC rank',ylabel='Episode share of PC variance (%)',ylim=(0,100))
    axes[1].set(title='Activity reaching the action readout',xlabel='Number of total-activity PCs',ylabel='Action-logit variance (%)',ylim=(0,101))
    axes[2].set(title='Readout-weight alignment',xlabel='Number of total-activity PCs',ylabel='Actor-weight energy (%)',ylim=(0,101))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    fig.supxlabel('Mean ± SEM over 8 seeds. Readout alignment measures linear influence, not behavioral necessity.',fontsize=10)
    save(fig,'eigenvector_readout')

    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    metrics=[('Total','total_participation_ratio'),('Within','within_participation_ratio')]
    for i,condition in enumerate(CONDITIONS):
        xs=np.arange(2)+(i-.5)*.28
        values=[mean_sem(rows,condition,lambda r:r['metrics'][key]) for _,key in metrics]
        axes[0].bar(xs,[v[0] for v in values],.26,yerr=[v[1] for v in values],capsize=3,color=COLORS[condition],label=LABELS[condition])
        for ax, fields, labels in [(axes[1],['top10_episode_subspace_overlap','top10_within_subspace_overlap'],['Episode span','Within PCs']),
                                  (axes[2],['early_late_total_overlap','episode_halves_total_overlap'],['Early / late','Episode halves'])]:
            vals=[mean_sem(rows,condition,lambda r:r['metrics'][key]*100) for key in fields]
            ax.bar(xs,[v[0] for v in vals],.26,yerr=[v[1] for v in vals],capsize=3,color=COLORS[condition])
            ax.set(xticks=np.arange(2),xticklabels=labels,ylim=(0,100),ylabel='Subspace overlap (%)')
    axes[0].set(xticks=np.arange(2),xticklabels=[v[0] for v in metrics],ylabel='Participation ratio',title='How spread out is the spectrum?')
    axes[1].set_title('Leading 10 PCs align with…')
    axes[2].set_title('Leading-PC reproducibility')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    fig.supxlabel('Mean ± SEM over 8 seeds. Overlap uses principal angles; it is unchanged by eigenvector signs.',fontsize=10)
    save(fig,'subspace_summary')


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261031)
    episode_splits=[]
    while len(episode_splits)<12:
        half=set(rng.choice(16,8,replace=False).tolist())
        if 0 not in half: half=set(range(16))-half
        half=sorted(half)
        if half not in episode_splits: episode_splits.append(half)
    protocol=dict(seeds=list(range(1,9)), samples=3393,episodes=16,units=256,leading_dimensions=K,
        analysis='Covariance eigenvalues/eigenvectors of saved hidden h, not recurrent-weight or dynamical-system eigenvalues. Divisor n; no neuron standardization.',
        decomposition='Exact sample-weighted total covariance = covariance of episode means + covariance of within-episode deviations.',
        rank_measures='Participation ratio=(sum lambda)^2/sum(lambda^2); effective rank=exp(entropy(lambda/sum lambda)).',
        actor='Subtract the mean across the six rows of the linear actor weight matrix, removing the softmax-invariant common logit. PC j contributes lambda_j * ||A u_j||^2 to total centered-logit variance.',
        eigenvectors='Compare subspaces only within the same network. Principal-angle overlap handles arbitrary eigenvector signs and rotations within degenerate spaces; 10-PC cutoff remains sensitive to small boundary gaps.',
        stability='Early/late halves of selected steps within each episode; and 12 fixed balanced episode splits. Each half refits covariance independently. This is sampling reproducibility, not stability of recurrent dynamics.',
        episode_half_splits=episode_splits,primary_comparisons=list(PRIMARY),
        statistics='Five exploratory primary paired contrasts over eight training seeds. Two-sided paired t and exhaustive sign flips, Holm corrected separately within this family. Other spectra, overlaps, and stability metrics are descriptive; pointwise SEM bands are not simultaneous confidence bands.',
        limitations=['Episode means confound goal, room, and history.', 'Shared scripted replay is off-policy and almost entirely post-discovery.', 'Readout-aligned covariance is not proof of causal behavioral necessity.', 'No between-network neuron or eigenvector coordinate alignment is assumed.', 'Covariance spectra do not identify attractors or dynamical stability.'])
    # Write choices before any model results are analyzed.
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    hashes={}
    replay_path=SOURCE/'replay.npz';hashes[str(replay_path.relative_to(ROOT))]=digest(replay_path)
    with np.load(replay_path) as r:
        indices=r['sample_index'];episodes=r['episode'][indices];steps=r['step'][indices]
    rows=[];vectors_saved={}
    for seed in range(1,9):
        for condition in CONDITIONS:
            path=SOURCE/f'{condition}_s{seed}_activity.npz'
            checkpoint=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt'
            for source in (path,checkpoint): hashes[str(source.relative_to(ROOT))]=digest(source)
            with np.load(path) as f: h=f['h'].astype(float);logits=f['logits'].astype(float)
            assert h.shape==(3393,256)
            covs,decomp_error=covariances(h,episodes)
            decomposed={kind:eigen(cov) for kind,cov in covs.items()}
            spectra={kind:spectral_summary(vals) for kind,(vals,_) in decomposed.items()}
            eigenvalues,u=decomposed['total'];bvalues,bu=decomposed['between'];_,wu=decomposed['within']
            brank=spectra['between']['numerical_rank'];assert brank<=15
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True);assert ckpt['frames']==1_000_000_000
            actor=ckpt['policy']['pi.weight'].numpy().astype(float);actor-=actor.mean(0)
            weights_per_pc=np.sum((actor@u)**2,axis=0)
            power=eigenvalues*weights_per_pc
            actor_total=float(np.trace(actor@covs['total']@actor.T))
            actor_between=float(np.trace(actor@covs['between']@actor.T))
            actor_within=float(np.trace(actor@covs['within']@actor.T))
            recorded=logits-logits.mean(1,keepdims=True);recorded-=recorded.mean(0)
            recorded_power=float(np.mean(np.sum(recorded**2,axis=1)))
            assert np.isclose(power.sum(),actor_total,rtol=1e-10)
            assert np.isclose(actor_between+actor_within,actor_total,rtol=1e-10)
            assert np.isclose(recorded_power,actor_total,rtol=2e-6)
            between_diag=np.diag(u.T@covs['between']@u)
            fractions=between_diag/eigenvalues
            assert fractions.min()>-1e-8 and fractions.max()<1+1e-8
            episode_overlap=subspace(u[:,:K],bu[:,:brank]);within_overlap=subspace(u[:,:K],wu[:,:K])
            # An actor row-space basis ignores action-weight magnitudes.
            _,sv,vt=np.linalg.svd(actor,full_matrices=False);arank=int(np.sum(sv>sv[0]*1e-10))
            actor_overlap=subspace(u[:,:K],vt[:arank].T)
            stability=split_stability(h,episodes,steps,episode_splits)
            metrics=dict(total_participation_ratio=spectra['total']['participation_ratio'],
                within_participation_ratio=spectra['within']['participation_ratio'],
                between_participation_ratio=spectra['between']['participation_ratio'],
                total_effective_rank=spectra['total']['effective_rank'],within_effective_rank=spectra['within']['effective_rank'],
                top10_episode_subspace_overlap=episode_overlap['overlap_fraction_of_first'],
                top10_within_subspace_overlap=within_overlap['overlap_fraction_of_first'],
                top10_actor_subspace_overlap=actor_overlap['overlap_fraction_of_first'],
                top10_actor_variance_share=float(power[:K].sum()/actor_total),
                episode_actor_variance_share=actor_between/actor_total,
                pc1_episode_fraction=float(fractions[0]),
                pc1_actor_variance_share=float(power[0]/actor_total),
                top10_actor_weight_share=float(weights_per_pc[:K].sum()/weights_per_pc.sum()),
                total_actor_variance=actor_total,between_actor_variance=actor_between,within_actor_variance=actor_within,
                top10_mean_unit_participation=float(np.mean(1/np.sum(u[:,:K]**4,axis=0))),
                early_late_total_overlap=stability['early_late']['mean_overlap']['total'],
                episode_halves_total_overlap=stability['episode_halves']['mean_overlap']['total'])
            row=dict(condition=condition,seed=seed,spectra=spectra,metrics=metrics,
                eigenvectors=dict(total10_episode_span=episode_overlap,total10_within10=within_overlap,
                    total10_actor_span=actor_overlap,actor_rank=arank,
                    between_fraction_per_total_pc=fractions.clip(0,1).tolist(),
                    actor_variance_per_pc=power.tolist(),actor_variance_cumulative=(power.cumsum()/actor_total).tolist(),
                    actor_weight_fraction_per_pc=(weights_per_pc/weights_per_pc.sum()).tolist(),
                    actor_weight_cumulative=(weights_per_pc.cumsum()/weights_per_pc.sum()).tolist(),
                    unit_participation_per_total_pc=(1/np.sum(u**4,axis=0)).tolist()),
                stability=stability,checks=dict(covariance_decomposition_relative_error=decomp_error,
                    saved_logits_vs_actor_variance_relative_error=abs(recorded_power-actor_total)/actor_total))
            rows.append(row)
            for kind,(_,vectors) in decomposed.items(): vectors_saved[f'{condition}_s{seed}_{kind}']=vectors
            print(f'{condition} s{seed}: PR={metrics["total_participation_ratio"]:.1f}, within PR={metrics["within_participation_ratio"]:.1f}; '
                  f'top10 actor share={100*metrics["top10_actor_variance_share"]:.1f}%, episode actor share={100*metrics["episode_actor_variance_share"]:.1f}%',flush=True)
    comparisons={key:paired(*[[r['metrics'][key] for r in rows if r['condition']==c] for c in CONDITIONS]) for key in rows[0]['metrics']}
    for field in ('t_p','exact_p'):
        running=0.
        for i,key in enumerate(sorted(PRIMARY,key=lambda key:comparisons[key][field])):
            running=max(running,min(1.,(len(PRIMARY)-i)*comparisons[key][field]))
            comparisons[key][field+'_holm5']=running
    result=dict(protocol=protocol,models=rows,comparisons=comparisons,source_sha256=hashes)
    assert all(digest(ROOT/path)==value for path,value in hashes.items())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'eigenvectors.npz',**vectors_saved)
    plots(rows)
    for key in PRIMARY:
        r=comparisons[key]
        print(key,f'grid={r["grid_mean"]:.6f}, pgc={r["pgc_mean"]:.6f}, delta={r["difference"]:.6f}, '
              f'CI={r["ci95"]}, t Holm={r["t_p_holm5"]:.6g}, exact Holm={r["exact_p_holm5"]:.6g}',flush=True)


if __name__=='__main__':
    main()
