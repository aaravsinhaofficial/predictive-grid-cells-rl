"""Independent saved-JSON baseline accounting; never runs either model."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/native_turn_distillation'
PROTOCOL = ROOT/'docs/research/native_turn_distillation_protocol.json'
EXPECTED = 'ac37ba336f79299ed285c4bb0ef126296e5f8d46aee795e431d58544aab06c42'
VARIANTS = ('seen_native', 'seen_reset_every_choice', 'blind_native', 'blind_ordinary',
            'blind_reset_every_choice', 'blind_ordinary_reset_every_choice')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def close(a, b):
    assert (a is None) == (b is None)
    if a is not None: assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def policy_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_model(condition):
    folder = BASE/condition/'original_baseline'
    summary = json.loads((folder/'summary.json').read_text()); live = json.loads((folder/'live.json').read_text())
    execution_path = BASE/f'{condition}_baseline_execution.json'; execution = json.loads(execution_path.read_text())
    assert summary['complete'] and summary['condition'] == condition and summary['protocol_sha256'] == EXPECTED
    assert all(summary[k] for k in ('source_unchanged', 'policy_unchanged', 'shared_across_arms', 'no_optimization'))
    assert summary['live_sha256'] == sha(folder/'live.json')
    assert summary['paired_original_summary_sha256'] == sha(ROOT/summary['paired_original_summary_path'])
    checkpoint = torch.load(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt', map_location='cpu', weights_only=True)
    assert policy_hash(checkpoint['policy']) == summary['policy_hash'] and checkpoint['frames'] == 1_000_000_000
    assert execution['complete'] and execution['condition'] == condition and execution['kind'] == 'baseline'
    assert len(execution['stages']) == 1 and execution['stages'][0]['exit_code'] == 0 and execution['no_cloud']
    assert execution['stages'][0]['stage'] == 'baseline' and 0 < summary['seconds'] < 600
    assert execution['stages'][0]['monotonic_seconds'] >= summary['seconds']
    committed = subprocess.check_output(['git', 'show', execution['source_commit']+':docs/research/native_turn_distillation_protocol.json'], cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest() == EXPECTED
    assert live['complete'] and live['episodes_per_variant'] == 16 and live['seed'] == 984901
    assert tuple(live['variants']) == VARIANTS and live['evaluation_state_noise'] == 0
    assert live['grid_native_ordinary_sham_exact'] == (condition == 'gc100')
    discovery = None; variants = {}
    for i, variant in enumerate(VARIANTS):
        result = live['variants'][variant]; rows = result['rows']
        assert len(rows) == 16 and {r['slot'] for r in rows} == set(range(16))
        ordered = sorted(rows, key=lambda r: r['slot']); first = [r['first_goal_step'] for r in ordered]
        if discovery is None: discovery = first
        assert first == discovery
        found = [r for r in rows if r['first_goal_step'] >= 0]
        efficiencies = [r['path_eff'] for r in rows if r['path_eff_n'] > 0]
        post_time = sum(1350-r['first_goal_step'] for r in found)
        assert result['episodes'] == 16
        expected = dict(mean_score=statistics.mean(r['score'] for r in rows),
            mean_goals=statistics.mean(r['n_goals'] for r in rows), fraction_discovered=len(found)/16,
            fraction_at_least_two_goals=sum(r['n_goals'] >= 2 for r in rows)/16,
            mean_returns_per_episode=statistics.mean(max(0, r['n_goals']-1) for r in rows),
            post_discovery_returns_per_1000_steps=1000*sum(max(0, r['n_goals']-1) for r in found)/post_time if post_time else None,
            conditional_mean_path_efficiency=statistics.mean(efficiencies) if efficiencies else None)
        for k, v in expected.items(): close(result[k], v)
        assert all(r['score'] == 10*r['n_goals'] and r['path_eff_n'] == max(0, r['n_goals']-1) for r in rows)
        exposure = live['exposure'][variant]; phase = live['event_clock']['phase_update_counts'][i]
        assert exposure == dict(choice=phase[6], other_control=phase[0]+phase[1], forced=sum(phase[2:6]))
        assert sum(phase) == sum(exposure.values()) == 16*1350
        for kind, count in exposure.items():
            close(live['mean_teacher_kl'][variant][kind], live['teacher_kl_sums'][variant][kind]/count if count else None)
            assert math.isfinite(live['teacher_kl_sums'][variant][kind])
        assert live['choice_erasure_counts'][variant] == (exposure['choice'] if 'reset' in variant else 0)
        rewards = live['reward_by_phase'][variant]
        close(sum(rewards), sum(r['score'] for r in rows))
        clock = live['event_clock']; audit = live['input_audit']
        assert clock['coverage_violations'][i] == 0 and clock['interrupted_done'][i] == 0
        assert clock['admitted'][i] == clock['completed'][i]+clock['interrupted_reward'][i]+clock['interrupted_done'][i]
        assert 3*clock['completed'][i] <= exposure['choice'] <= 3*clock['admitted'][i]
        close(audit['scheduled_blind_updates'][i], sum(phase[2:]))
        close(audit['forced_goal_hits'][i], sum(rewards[2:6])/10)
        close(audit['free_goal_hits'][i], rewards[6]/10)
        assert clock['completed'][i] <= audit['pulses'][i] <= clock['admitted'][i]
        variants[variant] = dict(mean_score=result['mean_score'], exposure=exposure,
            choice_erasure_count=live['choice_erasure_counts'][variant], reward_total=sum(rewards),
            teacher_kl_sum=live['teacher_kl_sums'][variant], cue_pulses=audit['pulses'][i])
    if condition == 'gc100':
        for a, b in (('blind_native', 'blind_ordinary'), ('blind_reset_every_choice', 'blind_ordinary_reset_every_choice')):
            for key in ('variants', 'exposure', 'teacher_kl_sums', 'mean_teacher_kl', 'choice_erasure_counts', 'reward_by_phase'):
                assert live[key][a] == live[key][b]
            ia, ib = VARIANTS.index(a), VARIANTS.index(b)
            for category in ('event_clock', 'input_audit'):
                assert all(values[ia] == values[ib] for values in live[category].values())
    return dict(complete=True, summary_sha256=sha(folder/'summary.json'), live_sha256=sha(folder/'live.json'),
        execution_sha256=sha(execution_path), source_commit=execution['source_commit'],
        episodes=96, variants=variants, first_discovery_steps_common_to_all_branches=discovery,
        exact_grid_cue_shams=(condition == 'gc100'), no_model_execution=True)


def main():
    assert sha(PROTOCOL) == EXPECTED; design = json.loads(PROTOCOL.read_text())
    for category in ('source_sha256', 'asset_sha256'):
        for name, expected in design[category].items(): assert sha(ROOT/name) == expected, name
    result = dict(complete=True, protocol_sha256=EXPECTED, audit_source_sha256=sha(__file__),
        models={c: audit_model(c) for c in ('gc100', 'pgc33')},
        source_files_verified=len(design['source_sha256']), assets_verified=len(design['asset_sha256']),
        procedure='Read saved episode rows, phase counts, reward and KL sums; recompute all episode summaries and reconcile counts/rewards/erasures, exact first-discovery matching and Grid shams. Verify checkpoint policy, protocol/source/assets, committed protocol and successful execution hashes.',
        input_scope='Seen controls restore RGB only. Current code is held ordinary during blind-event intervals with the declared native/ordinary endpoint cue; native goal code remains supplied. These are modified-task scores, not unmodified native navigation scores.',
        limitation='No per-update live input/logit tape was archived. This verifies saved accounting and the frozen input construction, not an independent rerun of live policies.')
    output = BASE/'original_baseline_independent_validation.json'; output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, path=str(output), sha256=sha(output), episodes=192)))


if __name__ == '__main__': main()
