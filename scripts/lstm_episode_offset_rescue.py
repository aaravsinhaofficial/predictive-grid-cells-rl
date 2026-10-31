"""Past-only episode-offset rescue in frozen PGC-trained LSTMs.

Two natural shadow RNNs receive the same actual experience, differing only in
the saved goal code. All variants follow the intact policy for 600 decisions.
Their predictive-input-induced mean hidden difference is estimated from steps
450--599 and frozen before intervention. Constant-offset patches normally
change only the actor readout; explicitly named feedback variants also modify
the corresponding shadow's hidden state. Optional subtraction variants test
necessity of the estimated episode component. No policy weights are fitted.
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

import numpy as np
import torch

from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena, summarize_episodes, verify_codes

VARIANTS = ('intact', 'goal_grid', 'global_mean', 'episode_mean',
            'wrong_episode_mean', 'between_episode_mean', 'between_dynamic',
            'random_dynamic', 'episode_mean_feedback', 'wrong_mean_feedback')
ABLATION_VARIANTS = ('remove_episode_mean', 'remove_wrong_episode_mean',
                    'remove_between_episode_mean', 'remove_episode_mean_feedback',
                    'remove_wrong_episode_mean_feedback')
ALL_VARIANTS = VARIANTS + ABLATION_VARIANTS
CALIBRATION_START = 450
HANDOFF = 600
CALIBRATION_STEPS = HANDOFF - CALIBRATION_START
BASIS_RANK = 10
PROTOCOL_VERSION = 1
PRIMARY_CONTRASTS = (
    ('episode_mean_minus_global_mean', 'global_mean', 'episode_mean'),
    ('episode_mean_minus_wrong_episode_mean', 'wrong_episode_mean', 'episode_mean'),
    ('between_episode_mean_minus_global_mean', 'global_mean', 'between_episode_mean'),
)
ABLATION_CONTRASTS = (
    ('remove_episode_minus_intact', 'intact', 'remove_episode_mean'),
    ('remove_episode_minus_remove_wrong', 'remove_wrong_episode_mean', 'remove_episode_mean'),
    ('remove_between_minus_intact', 'intact', 'remove_between_episode_mean'),
    ('remove_episode_feedback_minus_intact', 'intact', 'remove_episode_mean_feedback'),
    ('remove_episode_feedback_minus_wrong_feedback', 'remove_wrong_episode_mean_feedback', 'remove_episode_mean_feedback'),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_bases(seed, device):
    """Between-episode PCs are fitted only to the earlier independent replay."""
    source = ROOT / 'results/lstm_variable_1B'
    replay_path = source / 'replay.npz'
    activity_path = source / f'pgc33_s{seed}_activity.npz'
    with np.load(replay_path) as data:
        episodes = data['episode'][data['sample_index']]
    with np.load(activity_path) as data:
        h = data['h'].astype(np.float64)
    assert h.shape == (len(episodes), 256)
    labels, counts = np.unique(episodes, return_counts=True)
    assert len(labels) == 16
    means = np.stack([h[episodes == label].mean(0) for label in labels])
    center = np.average(means, axis=0, weights=counts)
    weighted = (means - center) * np.sqrt(counts / counts.sum())[:, None]
    _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    source_rank = int(np.linalg.matrix_rank(weighted))
    assert source_rank == 15
    between = vt[:BASIS_RANK].T
    random_seed = 980000 + seed
    random = np.random.default_rng(random_seed).normal(size=(256, BASIS_RANK))
    random -= between @ (between.T @ random)
    random, _ = np.linalg.qr(random)
    errors = dict(
        between_orthogonality=float(np.max(np.abs(between.T @ between - np.eye(BASIS_RANK)))),
        random_orthogonality=float(np.max(np.abs(random.T @ random - np.eye(BASIS_RANK)))),
        random_between_overlap=float(np.max(np.abs(between.T @ random))))
    assert max(errors.values()) < 1e-10
    metadata = dict(rank=BASIS_RANK, between_source_rank=source_rank,
        source_samples=len(h), source_episodes=len(labels),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (replay_path, activity_path)},
        between_eigenvalues=(singular * singular).tolist(), random_seed=random_seed,
        construction='Weighted episode-mean covariance from independent scripted replay; random orthonormal basis lies in the complement of its first 10 PCs.',
        checks=errors)
    return (torch.tensor(between, dtype=torch.float32, device=device),
            torch.tensor(random, dtype=torch.float32, device=device), metadata)


def variance_from_sums(sum_h, sum_h2, eligible):
    """Equal-length calibration windows, with each eligible episode weighted equally."""
    sums = np.asarray(sum_h, dtype=np.float64)[eligible]
    squares = np.asarray(sum_h2, dtype=np.float64)[eligible]
    if len(sums) == 0:
        return dict(total_variance=None, between_episode_variance=None,
                    between_episode_fraction=None)
    means = sums / CALIBRATION_STEPS
    overall = means.mean(0)
    total = float(np.sum(squares.mean(0) / CALIBRATION_STEPS - overall**2))
    between = float(np.mean(np.sum((means - overall)**2, axis=1)))
    assert total >= -1e-4 and between <= total + 1e-4
    total = max(total, 0.)
    return dict(total_variance=total, between_episode_variance=between,
                between_episode_fraction=between / total if total > 0 else 0.)


def freeze_offsets(sum_mixed, sum_grid, sum_mixed2, sum_grid2, sum_delta2,
                   eligible, between_basis, actor_weight, batch_index):
    """Freeze offsets at handoff using only the preceding calibration window."""
    mean_mixed = sum_mixed / CALIBRATION_STEPS
    mean_grid = sum_grid / CALIBRATION_STEPS
    delta = mean_mixed - mean_grid
    ids = eligible.nonzero().squeeze(1)
    count = len(ids)
    grand = delta[ids].mean(0) if count else torch.zeros_like(delta[0])
    episode_delta = delta - grand
    wrong_episode = torch.zeros_like(episode_delta)
    donor = torch.full((len(delta),), -1, dtype=torch.long, device=delta.device)
    wrong_zero_norm_count = 0
    norm_error = 0.
    if count > 1:
        donor_ids = ids.roll(-1)
        donor[ids] = donor_ids
        source = episode_delta[donor_ids]
        source_norm = source.norm(dim=1)
        own_norm = episode_delta[ids].norm(dim=1)
        valid = source_norm > 1e-10
        # A zero donor cannot be scaled to a positive norm; keep zero and log it.
        scales = torch.where(valid, own_norm / source_norm.clamp_min(1e-10), 0.)
        wrong_episode[ids] = source * scales[:, None]
        wrong_zero_norm_count = int(((~valid) & (own_norm > 1e-8)).sum())
        norm_error = float((wrong_episode[ids].norm(dim=1) - own_norm).abs().max())
        assert torch.all(donor[ids] != ids)
    # With one eligible episode its centered episode offset is exactly zero.
    wrong = grand[None] + wrong_episode
    between_mean = grand[None] + (episode_delta @ between_basis) @ between_basis.T
    arrays = {name: tensor.detach().cpu().numpy().copy() for name, tensor in dict(
        eligible=eligible, donor_slot=donor, mixed_mean=mean_mixed,
        grid_mean=mean_grid, delta_e=delta, grand_delta=grand,
        episode_delta=episode_delta, wrong_delta=wrong,
        between_mean_delta=between_mean, sum_mixed=sum_mixed,
        sum_grid=sum_grid, sum_mixed2=sum_mixed2,
        sum_grid2=sum_grid2, sum_delta2=sum_delta2).items()}
    mask = arrays['eligible']
    basis = between_basis.cpu().numpy().astype(float)
    ep = arrays['episode_delta'][mask].astype(float)
    d = arrays['delta_e'][mask].astype(float)
    centered_actor = actor_weight - actor_weight.mean(0, keepdim=True)
    aw = centered_actor.detach().cpu().numpy().astype(float)
    energy = float(np.sum(ep * ep))
    projected_energy = float(np.sum((ep @ basis)**2))
    dynamic_energy = float(arrays['sum_delta2'][mask].sum() / CALIBRATION_STEPS)
    actor_grand = arrays['grand_delta'].astype(float) @ aw.T
    actor_ep = ep @ aw.T
    record = dict(episode=batch_index, eligible_count=count, total_slots=len(delta),
        mixed=variance_from_sums(arrays['sum_mixed'], arrays['sum_mixed2'], mask),
        grid=variance_from_sums(arrays['sum_grid'], arrays['sum_grid2'], mask),
        mean_delta_l2=float(np.linalg.norm(d, axis=1).mean()) if count else None,
        grand_delta_l2=float(np.linalg.norm(arrays['grand_delta'])),
        mean_episode_delta_l2=float(np.linalg.norm(ep, axis=1).mean()) if count else None,
        between_projection_energy_fraction=projected_energy / energy if energy > 0 else 0.,
        mean_delta_energy_fraction=float(np.sum(d*d) / dynamic_energy) if dynamic_energy > 0 else 0.,
        episode_delta_energy_fraction=energy / dynamic_energy if dynamic_energy > 0 else 0.,
        grand_centered_logit_energy=float(np.sum(actor_grand**2)),
        mean_episode_centered_logit_energy=float(np.mean(np.sum(actor_ep**2, axis=1))) if count else None,
        wrong_derangement_available=count > 1,
        wrong_zero_norm_count=wrong_zero_norm_count,
        wrong_norm_max_error=norm_error,
        calibration_start=CALIBRATION_START, calibration_stop_exclusive=HANDOFF)
    return dict(delta=delta, grand=grand, wrong=wrong, between_mean=between_mean), record, arrays


def summarize_rows(rows, episode_len):
    result = summarize_episodes(rows, episode_len)
    eligible = [r for r in rows if r['eligible']]
    result.update(eligible_episodes=len(eligible),
        post_handoff_score=float(np.mean([r['post_score'] for r in eligible])) if eligible else None,
        eligible_mean_score=float(np.mean([r['ret'] for r in eligible])) if eligible else None,
        mean_prefix_score=float(np.mean([r['prefix_score'] for r in rows])),
        eligible_mean_prefix_score=float(np.mean([r['prefix_score'] for r in eligible])) if eligible else None,
        mean_prefix_score_denominator='All completed episodes; eligible_mean_prefix_score uses eligible episodes only.')
    return result


def future_offset_validation(sum_delta, sum_delta2, past_delta, eligible,
                             actor_weight, future_steps):
    """Diagnostic only: called after the episode, never used by the policy."""
    arrays = {name: tensor.detach().cpu().numpy().copy() for name, tensor in dict(
        future_delta_sum=sum_delta, future_delta2_sum=sum_delta2,
        future_eligible=eligible, past_delta_for_validation=past_delta).items()}
    mask = arrays['future_eligible']
    future = arrays['future_delta_sum'].astype(float) / future_steps
    past = arrays['past_delta_for_validation'].astype(float)
    count = int(mask.sum())
    future_grand = future[mask].mean(0) if count else np.zeros(future.shape[1])
    past_grand = past[mask].mean(0) if count else np.zeros(past.shape[1])
    future_centered, past_centered = future - future_grand, past - past_grand
    centered_actor = actor_weight - actor_weight.mean(0, keepdim=True)
    aw = centered_actor.detach().cpu().numpy().astype(float)

    def compare(prediction, target):
        target_energy = float(np.sum(target**2))
        prediction_energy = float(np.sum(prediction**2))
        denominator = np.sqrt(target_energy * prediction_energy)
        per_episode_denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
        valid = per_episode_denominator > 1e-12
        return dict(
            cosine=float(np.sum(prediction*target) / denominator) if denominator > 1e-12 else None,
            mean_episode_cosine=float(np.mean(np.sum(prediction[valid]*target[valid], axis=1) /
                                             per_episode_denominator[valid])) if valid.any() else None,
            prediction_r2=1. - float(np.sum((prediction-target)**2)) / target_energy if target_energy > 1e-12 else None,
            squared_error=float(np.sum((prediction-target)**2)),
            target_energy=target_energy, prediction_energy=prediction_energy,
            nonzero_episode_pairs=int(valid.sum()))

    dynamic_energy = float(arrays['future_delta2_sum'][mask].astype(float).sum() / future_steps)
    constant_energy = float(np.sum(future[mask]**2))
    episode_energy = float(np.sum(future_centered[mask]**2))
    record = dict(eligible_count=count, future_start=HANDOFF,
        future_stop_exclusive=HANDOFF + future_steps, future_steps=future_steps,
        centered_hidden=compare(past_centered[mask], future_centered[mask]),
        centered_actor_logits=compare(past_centered[mask] @ aw.T, future_centered[mask] @ aw.T),
        future_mean_energy_fraction=constant_energy / dynamic_energy if dynamic_energy > 0 else 0.,
        future_episode_mean_energy_fraction=episode_energy / dynamic_energy if dynamic_energy > 0 else 0.,
        future_dynamic_mean_squared_l2=dynamic_energy / count if count else None,
        future_constant_mean_squared_l2=constant_energy / count if count else None,
        future_grand_delta_l2=float(np.linalg.norm(future_grand)),
        past_grand_delta_l2=float(np.linalg.norm(past_grand)),
        scope='Post-episode diagnostic of the intact branch only. Past and future episode offsets are separately centered over the same prefix-eligible episodes. R2 is 1-SSE/target-energy without fitting or rescaling and may be negative. Actor logits remove the action-common weight direction. No future measurement enters actions, frozen offsets, or eligibility.')
    arrays.update(future_delta_mean=future.copy(), future_grand_delta=future_grand.copy(),
                  future_episode_delta=future_centered.copy())
    return record, arrays


def equal_groups(value, n, groups):
    return (value.reshape((groups, n) + value.shape[1:]) == value[:n][None]).all()


@torch.inference_mode()
def run(seed, args):
    started = time.perf_counter()
    cfg_path = ROOT / f'models/pgc33/pgc33_lstm_s{seed}_1000M.json'
    cfg = json.loads(cfg_path.read_text())
    expected = dict(cond='pgc33', policy='lstm', nh=256, n_cells=512,
        code_mode='full', code_noise=0., pgc_shift_mode='space', pgc_shift=.2,
        pgc_shift_jitter=0., pgc_frac=.33, episode_len=1350,
        goal_code_shift='intact', extra_input='none')
    assert all(cfg.get(k) == value for k, value in expected.items()), 'Unsupported model configuration'
    assert not cfg.get('silence_pgc', False)
    assert cfg.get('cell_type', 'grid') == 'grid'
    checkpoint = cfg_path.with_suffix('.pt')
    checkpoint_hash = sha256(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device == 'mps':
        torch.mps.manual_seed(args.eval_seed)
    net = build_policy(cfg, args.device).eval().requires_grad_(False)
    net.load_state_dict(ckpt['policy'], strict=True)
    pop = build_population(cfg, args.device)
    assert int(pop.is_pgc.sum()) == 169
    between_basis, random_basis, basis_metadata = independent_bases(seed, args.device)
    variants = tuple(getattr(args, 'variants', VARIANTS))
    assert variants[0] == 'intact' and len(variants) == len(set(variants))
    ablation_mode = any(name in ABLATION_VARIANTS for name in variants)
    n, groups, nh = args.n_envs, len(variants), cfg['nh']
    total = n * groups
    env = MatchedArena(cfg, n, groups, args.eval_seed, args.device)
    obs = env.reset_all()
    mixed_state = net.zero_state(total, args.device)
    grid_state = net.zero_state(total, args.device)
    mixed_goal = torch.zeros(total, cfg['n_cells'], device=args.device)
    grid_goal = torch.zeros_like(mixed_goal)
    pa = torch.full((total,), -1, dtype=torch.long, device=args.device)
    pr = torch.zeros(total, device=args.device)
    action_rng = torch.Generator(device=args.device).manual_seed(args.eval_seed + 1)
    eligible = torch.zeros(n, dtype=torch.bool, device=args.device)
    accumulators = {name: torch.zeros(n, nh, device=args.device) for name in
        ('mixed', 'grid', 'mixed2', 'grid2', 'delta2')}
    future_delta_sum = torch.zeros(n, nh, device=args.device)
    future_delta2_sum = torch.zeros_like(future_delta_sum)
    offsets = None
    prefix_score = torch.zeros(total, device=args.device)
    post_score = torch.zeros_like(prefix_score)
    prefix_equal = torch.tensor(True, device=args.device)
    prefix_checks = 0
    full_rescue_logit_max_error = 0.
    random_zero_norm_failures = torch.zeros((), device=args.device)
    random_norm_max_error = torch.zeros((), device=args.device)
    random_scale_max = torch.zeros((), device=args.device)
    # Columns: samples, sum patch L2, outside units, outside samples, max |h|.
    patch_stats = torch.zeros(groups, 5, device=args.device)
    rows = {name: [] for name in variants}
    calibration = []
    raw = {}
    step = 0
    last_progress = started
    while len(rows['intact']) < args.episodes:
        phase = step % cfg['episode_len']
        batch_index = step // cfg['episode_len']
        if phase < HANDOFF:
            for key in ('pos', 'hd', 'dir', 'vel', 'rgb'):
                prefix_equal &= equal_groups(obs[key], n, groups)
            prefix_equal &= equal_groups(mixed_goal, n, groups)
            prefix_equal &= equal_groups(grid_goal, n, groups)
            prefix_equal &= equal_groups(pa, n, groups) & equal_groups(pr, n, groups)
            prefix_checks += 1
        if phase == CALIBRATION_START:
            eligible = (env.first_goal_t[:n] >= 0) & (env.first_goal_t[:n] < CALIBRATION_START)
        current = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        embedding = net.embed(obs['rgb'])
        _, _, mixed_state = net.step_from_embedding(embedding, current, mixed_goal, pa, pr, mixed_state)
        _, _, grid_state = net.step_from_embedding(embedding, current, grid_goal, pa, pr, grid_state)
        hm, hg = mixed_state[0], grid_state[0]
        if phase < HANDOFF:
            for state in (mixed_state, grid_state):
                prefix_equal &= equal_groups(state[0], n, groups) & equal_groups(state[1], n, groups)
        if CALIBRATION_START <= phase < HANDOFF:
            accumulators['mixed'] += hm[:n]
            accumulators['grid'] += hg[:n]
            accumulators['mixed2'] += hm[:n].square()
            accumulators['grid2'] += hg[:n].square()
            accumulators['delta2'] += (hm[:n] - hg[:n]).square()
        if phase == HANDOFF:
            assert bool(prefix_equal), 'Variants diverged during the forced shared prefix'
            offsets, record, arrays = freeze_offsets(
                *(accumulators[k] for k in ('mixed', 'grid', 'mixed2', 'grid2', 'delta2')),
                eligible, between_basis, net.pi.weight, batch_index)
            calibration.append(record)
            for key, value in arrays.items():
                raw[f'batch{batch_index}_{key}'] = value
        actor_h = hm.clone()
        if phase >= HANDOFF:
            # These future values are diagnostic only, on the intact branch.
            # They are read for validation after the complete episode ends.
            future_delta = hm[:n] - hg[:n]
            future_delta_sum += future_delta
            future_delta2_sum += future_delta.square()
            grid_h_next = hg.clone()
            mixed_h_next = hm.clone()
            for j, name in enumerate(variants):
                sl = slice(j*n, (j+1)*n)
                diff = hm[sl] - hg[sl]
                if name == 'intact':
                    patched = hm[sl]
                elif name == 'goal_grid':
                    patched = hg[sl]
                elif name == 'global_mean':
                    patched = hg[sl] + offsets['grand']
                elif name in ('episode_mean', 'episode_mean_feedback'):
                    patched = hg[sl] + offsets['delta']
                elif name in ('wrong_episode_mean', 'wrong_mean_feedback'):
                    patched = hg[sl] + offsets['wrong']
                elif name == 'between_episode_mean':
                    patched = hg[sl] + offsets['between_mean']
                elif name == 'between_dynamic':
                    patched = hg[sl] + (diff @ between_basis) @ between_basis.T
                elif name == 'random_dynamic':
                    target = (diff @ between_basis) @ between_basis.T
                    random = (diff @ random_basis) @ random_basis.T
                    target_norm, random_norm = target.norm(dim=1), random.norm(dim=1)
                    valid = random_norm > 1e-10
                    scale = torch.where(valid, target_norm / random_norm.clamp_min(1e-10), 0.)
                    patch = random * scale[:, None]
                    random_zero_norm_failures += ((~valid) & (target_norm > 1e-8) & eligible).sum()
                    random_norm_max_error = torch.maximum(random_norm_max_error,
                        torch.where(eligible, (patch.norm(dim=1) - target_norm).abs(), 0.).max())
                    random_scale_max = torch.maximum(random_scale_max, torch.where(eligible, scale, 0.).max())
                    patched = hg[sl] + patch
                elif name in ('remove_episode_mean', 'remove_episode_mean_feedback'):
                    patched = hm[sl] - (offsets['delta'] - offsets['grand'])
                elif name in ('remove_wrong_episode_mean', 'remove_wrong_episode_mean_feedback'):
                    patched = hm[sl] - (offsets['wrong'] - offsets['grand'])
                elif name == 'remove_between_episode_mean':
                    patched = hm[sl] - (offsets['between_mean'] - offsets['grand'])
                else:
                    raise AssertionError(name)
                actor_h[sl] = torch.where(eligible[:, None], patched, hm[sl])
                if name in ('episode_mean_feedback', 'wrong_mean_feedback'):
                    grid_h_next[sl] = torch.where(eligible[:, None], patched, hg[sl])
                if name in ('remove_episode_mean_feedback', 'remove_wrong_episode_mean_feedback'):
                    mixed_h_next[sl] = torch.where(eligible[:, None], patched, hm[sl])
                mixed_reference = name in ABLATION_VARIANTS or (ablation_mode and name == 'intact')
                reference_h = hm[sl] if mixed_reference else hg[sl]
                patch_norm = (patched - reference_h).norm(dim=1)
                outside = patched.abs() > 1.
                patch_stats[j, 0] += eligible.sum()
                patch_stats[j, 1] += torch.where(eligible, patch_norm, 0.).sum()
                patch_stats[j, 2] += (outside & eligible[:, None]).sum()
                patch_stats[j, 3] += (outside.any(1) & eligible).sum()
                patch_stats[j, 4] = torch.maximum(patch_stats[j, 4],
                    torch.where(eligible[:, None], patched.abs(), 0.).max())
            # Only explicitly named feedback variants alter recurrence; c is retained.
            grid_state = (grid_h_next, grid_state[1])
            mixed_state = (mixed_h_next, mixed_state[1])
            if phase in (HANDOFF, 900, cfg['episode_len'] - 1):
                full = hg + (hm - hg)
                error = float((net.pi(full) - net.pi(hm)).abs().max())
                full_rescue_logit_max_error = max(full_rescue_logit_max_error, error)
                assert error < 2e-5, f'Full hidden rescue algebra failed: {error}'
        logits = net.pi(actor_h)
        uniforms = torch.rand(n, generator=action_rng, device=args.device).repeat(groups)
        action = (uniforms[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        if phase < HANDOFF:
            action = action[:n].repeat(groups)
            prefix_equal &= equal_groups(action, n, groups)
        obs, reward, done, info = env.step(action)
        if phase < HANDOFF:
            prefix_score += reward
            prefix_equal &= equal_groups(reward, n, groups) & equal_groups(info['hit'], n, groups)
            prefix_equal &= equal_groups(info['hit_pos'], n, groups) & equal_groups(info['hit_dir'], n, groups)
        else:
            post_score += reward
        hit = info['hit']
        if bool(hit.any()):
            mixed = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'])
            plain = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0)
            mixed_goal = torch.where(hit[:, None], mixed, mixed_goal)
            grid_goal = torch.where(hit[:, None], plain, grid_goal)
        pa = action
        pr = reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            assert bool(done.all()), 'Expected synchronized complete episodes'
            assert bool(prefix_equal), 'Shared prefix failed equality check'
            future_record, future_arrays = future_offset_validation(
                future_delta_sum, future_delta2_sum, offsets['delta'], eligible,
                net.pi.weight, cfg['episode_len'] - HANDOFF)
            assert calibration[-1]['episode'] == batch_index
            calibration[-1]['future_validation'] = future_record
            for key, value in future_arrays.items():
                raw[f'batch{batch_index}_{key}'] = value
            es = {key: value.cpu().tolist() for key, value in info['ep_stats'].items()}
            eligibility = eligible.cpu().tolist()
            prefix_values, post_values = prefix_score.cpu().tolist(), post_score.cpu().tolist()
            for i, global_slot in enumerate(es['env']):
                base_slot = global_slot % n
                row = dict(slot=base_slot, episode=batch_index, eligible=eligibility[base_slot],
                    prefix_score=prefix_values[global_slot], post_score=post_values[global_slot],
                    **{key: es[key][i] for key in ('ret', 'n_goals', 'first_goal_t', 'path_eff', 'path_eff_n')})
                assert row['ret'] == row['prefix_score'] + row['post_score']
                rows[variants[global_slot // n]].append(row)
            mixed_state = net.zero_state(total, args.device)
            grid_state = net.zero_state(total, args.device)
            mixed_goal.zero_()
            grid_goal.zero_()
            pa = torch.full_like(action, -1)
            pr.zero_()
            eligible.zero_()
            offsets = None
            prefix_score.zero_()
            post_score.zero_()
            for value in accumulators.values():
                value.zero_()
            future_delta_sum.zero_()
            future_delta2_sum.zero_()
            prefix_equal.fill_(True)
        step += 1
        if time.perf_counter() - last_progress >= 25:
            print(f'pgc33 s{seed} offset rescue: {step}/{cfg["episode_len"]*(args.episodes//n)}; '
                  f'{time.perf_counter()-started:.0f}s', flush=True)
            last_progress = time.perf_counter()
    first = [r['first_goal_t'] for r in rows['intact']]
    prefix = [r['prefix_score'] for r in rows['intact']]
    for name in variants:
        assert [r['first_goal_t'] for r in rows[name]] == first
        assert [r['prefix_score'] for r in rows[name]] == prefix
        for original, altered in zip(rows['intact'], rows[name]):
            if not original['eligible']:
                assert all(original[k] == altered[k] for k in original), (name, original, altered)
    baseline_match = None
    baseline_reason = 'Earlier reference does not use these exact run settings.'
    old_path = ROOT / f'results/lstm_code_swap_1B/pgc33_s{seed}.json'
    if old_path.exists():
        old = json.loads(old_path.read_text())
        assert old['checkpoint_sha256'] == checkpoint_hash
        if (old['episodes'], old['n_envs'], old['eval_seed']) == (args.episodes, n, args.eval_seed):
            keys = ('slot', 'episode', 'ret', 'n_goals', 'first_goal_t', 'path_eff', 'path_eff_n')
            current_rows = [{key: row[key] for key in keys} for row in rows['intact']]
            baseline_match = current_rows == old['variants']['intact']['rows']
            assert baseline_match, 'Intact episode records differ from prior baseline'
            baseline_reason = 'Every intact episode record matches the earlier code-swap baseline exactly.'
    assert sha256(checkpoint) == checkpoint_hash
    diagnostics = {}
    for name, values in zip(variants, patch_stats.cpu().numpy().astype(float)):
        count, norm_sum, outside_units, outside_samples, peak = values
        diagnostics[name] = dict(samples=int(count), mean_patch_l2=norm_sum/count if count else 0.,
            patch_reference='mixed_hidden' if name in ABLATION_VARIANTS or (ablation_mode and name == 'intact') else 'ordinary_goal_hidden',
            units_outside_unit_range_fraction=outside_units/(count*nh) if count else 0.,
            samples_outside_unit_range_fraction=outside_samples/count if count else 0.,
            maximum_absolute_actor_hidden=float(peak))
    diagnostics['random_control'] = dict(zero_norm_failures=int(random_zero_norm_failures),
        norm_match_max_absolute_error=float(random_norm_max_error), maximum_rescaling=float(random_scale_max))
    raw['between_basis'] = between_basis.cpu().numpy()
    raw['random_complement_basis'] = random_basis.cpu().numpy()
    archive = args.out / f'pgc33_s{seed}_calibration.npz'
    np.savez_compressed(archive, **raw)
    result = dict(protocol_version=PROTOCOL_VERSION, condition='pgc33', seed=seed,
        checkpoint_frames=ckpt['frames'], checkpoint_sha256=checkpoint_hash,
        episodes=args.episodes, n_envs=n, eval_seed=args.eval_seed, device=args.device,
        variants={name: summarize_rows(r, cfg['episode_len']) for name, r in rows.items()},
        calibration=dict(batches=calibration, raw_archive=archive.name, archive_sha256=sha256(archive)),
        patch_diagnostics=diagnostics, basis=basis_metadata,
        checks=dict(prefix_exact=True, prefix_decision_steps_checked=prefix_checks,
            first_goal_records_identical=True, prefix_reward_records_identical=True,
            ineligible_episode_records_identical=True,
            full_rescue_logit_max_error=full_rescue_logit_max_error,
            intact_baseline_match=baseline_match, intact_baseline_note=baseline_reason,
            frozen_checkpoint_unchanged=True), elapsed_seconds=time.perf_counter()-started)
    (args.out / f'pgc33_s{seed}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    scores = ', '.join(f'{name}={value["post_handoff_score"]:.2f}'
        if value['post_handoff_score'] is not None else f'{name}=no eligible episodes'
        for name, value in result['variants'].items())
    print(f'DONE pgc33 s{seed} eligible post-handoff score: {scores}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--n-envs', type=int, default=64)
    parser.add_argument('--eval-seed', type=int, default=737373)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 9)))
    parser.add_argument('--variants', nargs='+', choices=ALL_VARIANTS, default=list(VARIANTS),
                        help='Intact is required and is always placed first; default retains the original ten rescue variants.')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_episode_offset_rescue_1B')
    args = parser.parse_args()
    assert args.episodes > 0 and args.n_envs > 0 and args.episodes % args.n_envs == 0
    assert len(args.seeds) == len(set(args.seeds))
    assert 'intact' in args.variants and len(args.variants) == len(set(args.variants))
    args.variants = ['intact'] + [name for name in args.variants if name != 'intact']
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(protocol_version=PROTOCOL_VERSION, seeds=args.seeds,
        episodes_per_seed_variant=args.episodes, n_envs_per_variant=args.n_envs,
        eval_seed=args.eval_seed, device=args.device, episode_len=1350,
        variants=args.variants, calibration_start=CALIBRATION_START,
        calibration_stop_exclusive=HANDOFF, handoff_step=HANDOFF,
        eligibility='The goal was discovered before decision 450 in the common intact prefix.',
        primary_outcome='Mean reward sum over decisions 600--1349, among eligible episodes only.',
        primary_contrasts=[dict(name=name, reference=a, intervention=b) for name, a, b in PRIMARY_CONTRASTS
                           if a in args.variants and b in args.variants],
        exploratory_ablation_contrasts=[dict(name=name, reference=a, intervention=b) for name, a, b in ABLATION_CONTRASTS
                                       if a in args.variants and b in args.variants],
        statistics='Eight training-seed means are the replicates; paired t and exhaustive sign-flip tests. The three original primary rescue contrasts and five supplemental exploratory ablation contrasts are separate Holm families; only available contrasts are listed. Other comparisons are exploratory.',
        current_code='Original mixed PGC code in every variant.',
        shadows='Two RNN states per branch, receiving its identical actual images/actions/rewards/current code. One has mixed goal snapshots, the other ordinary-grid goal snapshots.',
        common_prefix='All actions are copied from the intact branch for decisions 0--599. Sensor states, shadow states, rewards and hit records are checked exactly.',
        frozen_offset='Mean mixed-minus-grid hidden difference over decisions 450--599. Grand mean over eligible slots is separated from each episode deviation.',
        wrong_offset='Grand mean plus cyclic-next-eligible episode deviation, rescaled to the own episode deviation L2 norm; zero-donor failures are reported.',
        between_basis='First 10 eigenvectors of weighted episode-mean covariance from the prior independent 16-episode replay.',
        random_control='Fixed seeded rank-10 orthonormal basis in the complement of the between-episode basis. Its dynamic patch is rescaled each step to the corresponding between-basis patch L2 norm on that same branch experience.',
        feedback='episode_mean_feedback and wrong_mean_feedback overwrite grid-shadow h after each step. remove_episode_mean_feedback and remove_wrong_episode_mean_feedback instead overwrite mixed-shadow h. Each c state remains untouched; all other patches affect the actor only.',
        ablation_followup=dict(active=any(name in ABLATION_VARIANTS for name in args.variants),
            timing='Exploratory necessity follow-up planned after the first rescue seed had been inspected.',
            intervention='Subtract the frozen correct or wrong CENTERED episode offset from mixed hidden activity; the between variant subtracts its first-10-between-PC projection. Mixed current and goal inputs remain supplied. Feedback repeats this subtraction in h before the next recurrent update; it does not literally clamp the actual future episode mean.'),
        future_validation='After each episode, compare the frozen past offset with the mean intact-branch mixed-minus-grid hidden difference over decisions 600--1349. Record centered hidden and centered-action-logit cosine/R2 and future mean/dynamic energy fractions. These future accumulators never affect actions, offsets, or eligibility.',
        code_identity_checks={seed: verify_codes(seed) for seed in args.seeds},
        caveats=[
            'A constant actor-only hidden patch is a fixed action-logit bias. A null rescue does not rule out recurrent or time-varying uses of episode information.',
            'Between-subspace dynamic restoration includes moment-to-moment signals, not just average offsets.',
            'All rescues reintroduce predictive information; none is a pure ordinary-grid success.',
            'Feedback patches can leave the natural hidden-state range; violations are measured, not clipped.',
            'Supplemental ablations subtract an estimate of predictive-input-induced episode differences, not every source of episode-associated activity. They are exploratory necessity tests added after initial rescue results.',
            'The episode-specific offset comes from a finite past calibration window and remains frozen as goal snapshots update.',
            'The global and wrong-episode controls use contemporaneous calibration from other batch episodes; they are diagnostic interventions, not a proposed deployable agent.',
            'Eligibility is fixed before intervention from the common prefix. No future hidden activity enters calibration.'])
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for seed in args.seeds:
        path = args.out / f'pgc33_s{seed}.json'
        if path.exists():
            old = json.loads(path.read_text())
            assert (old['protocol_version'], old['episodes'], old['n_envs'], old['eval_seed']) == (
                PROTOCOL_VERSION, args.episodes, args.n_envs, args.eval_seed)
            assert set(old['variants']) == set(args.variants)
            print(f'Using completed pgc33 s{seed}', flush=True)
            continue
        run(seed, args)


if __name__ == '__main__':
    main()
