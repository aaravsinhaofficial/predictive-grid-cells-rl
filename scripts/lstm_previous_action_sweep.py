"""Prospective action-input sensitivity of original navigation LSTM dynamics.

Reuse all four original pilot travel families. Only the six previous-action
input entries change. This is a clamped-input neural experiment, not a claim
that an immobile agent repeatedly performs translation actions.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from lstm_endogenous_observable_flow import load_net, observable_probe
from lstm_endogenous_curve_recovery import advance, sha, save
from lstm_continued_heading_memory import analyze_branch, TIMES

OUT = ROOT/'results/lstm_previous_action_sweep'
INPUT = ROOT/'results/lstm_travel_memory/original'
ACTIONS = ('left', 'right', 'forward', 'backward', 'strafe_left', 'strafe_right')
MODELS = tuple((c, s) for c in ('gc100', 'pgc33') for s in (1, 2))
SOURCES = ('scripts/lstm_previous_action_sweep.py',
    'scripts/lstm_continued_heading_memory.py', 'scripts/lstm_endogenous_observable_flow.py',
    'scripts/lstm_heading_curve_flow.py', 'scripts/lstm_endogenous_curve_recovery.py',
    'scripts/lstm_heading_memory_probe.py', 'scripts/plot_lstm_heading_memory.py',
    'code/nets.py', 'code/common.py')


def action_input(original, action):
    assert original.shape == (1, 1287) and 0 <= action < 6
    result = original.clone()
    result[:, 1280:1286] = 0
    result[:, 1280+action] = 1
    assert torch.equal(result[:, :1280], original[:, :1280])
    assert torch.equal(result[:, 1286:], original[:, 1286:])
    return result


def assets(condition, seed):
    name = f'{condition}_s{seed}'
    return (ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt',
            INPUT/name/'context0_states.npz')


def protocol():
    return dict(version='original_lstm_previous_action_sensitivity_v1',
        motivation='A split attractor design for rapidly writing a navigational goal, Nature2026, doi10.1038/s41586-026-11144-9, reports behavior-dependent gating in a fly goal circuit. This motivates testing input dependence, not asserting biological homology.',
        cohort='Original1B models, both populations, pilotseeds1and2, existing context0 travel family for each. All24 model/action cases; no selection on new results.',
        intervention='Replace only previous-action onehot entries1280:1286 with each of allsix actual action IDs. Start every condition from the exact same archived768x512 fullstate family at release time0. Black embedding, ordinary currentcode, native savedgoal and previousreward unchanged.',
        scope='Clamped-input neural dynamics. Translating action messages with held position/vision are explicitly counterfactual; no claim of physical trajectories, spontaneous actor selection or improved navigation. Follow any candidate with native behavior and cue perturbation.',
        geometry='Unchanged analyze_branch, full512state, rawandfixedh/cRMS, time30reference plus30recovery,384uniformarcprobes and all existing gates. Initial travelphase readouts only; no fictitious heading integration under translation.',
        observable='Unchanged observable_probe at15updates: all256hidden outputs at0/5/10/15futureupdates, evolve30, latest60. Raw and fixedh/cRMS fullstate kicks; report centeredandnoncentered gates. All48action/metricchecks retained.',
        identity='The original previousLEFT input must exactly equal action0; replay all archived0/1/5/15/30/45/60states within1e-9abs/1e-11rel before interpreting any new condition.',
        inference='Two originalpilotseedpairs; six conditions are within-model interventions. Candidate search precedes heldout confirmation; no population-level significance or threshold tuning.',
        source_sha256={p:sha(ROOT/p) for p in SOURCES},
        assets=[dict(condition=c,seed=s,checkpoint=str(assets(c,s)[0].relative_to(ROOT)),
            checkpoint_sha256=sha(assets(c,s)[0]),
            config=str(assets(c,s)[0].with_suffix('.json').relative_to(ROOT)),
            config_sha256=sha(assets(c,s)[0].with_suffix('.json')),
            archive=str(assets(c,s)[1].relative_to(ROOT)),
            archive_sha256=sha(assets(c,s)[1])) for c,s in MODELS])


def validate_archive(a, checkpoint):
    assert str(a['checkpoint_sha256']) == sha(checkpoint)
    assert tuple(a['times']) == TIMES and a['states'].shape == (7,768,512)
    assert a['theta'].shape == (768,) and a['scale'].shape == (512,)
    assert np.isfinite(a['states']).all() and np.isfinite(a['scale']).all()
    assert (a['scale'] > 0).all()
    x = torch.tensor(a['ordinary_input'], dtype=torch.float64)
    assert torch.equal(action_input(x, 0), x), 'Original release was not previousLEFT'


def run(condition, seed, out):
    checkpoint, archive = assets(condition, seed)
    net, data, cfg = load_net(checkpoint)
    assert data['frames'] == 1_000_000_000
    assert cfg['cond'] == condition and cfg['seed'] == seed
    with np.load(archive) as loaded:
        a = {k:loaded[k] for k in loaded.files}
    validate_archive(a, checkpoint)
    start = torch.tensor(a['states'][0], dtype=torch.float64)
    original = torch.tensor(a['ordinary_input'], dtype=torch.float64)
    folder = out/f'{condition}_s{seed}'; folder.mkdir(parents=True, exist_ok=True)
    for action, name in enumerate(ACTIONS):
        x = action_input(original, action)
        result, states = analyze_branch(net, start, x, a['theta'],
                            {str(t):0. for t in range(61)}, a['scale'])
        for row in result['trajectory'].values():
            row['fixed_travel_phase'] = row.pop('fixed_initial_phase')
            row.pop('fixed_current_heading')
        replay_error = None
        if action == 0:
            replay_error = max(float(np.max(np.abs(states[t]-a['states'][i])))
                               for i,t in enumerate(TIMES))
            for i,t in enumerate(TIMES):
                assert np.allclose(states[t],a['states'][i],atol=1e-9,rtol=1e-11)
        drive = torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
        transition = lambda z: advance(net.lstm,drive,z)
        observable = {metric:observable_probe(transition,states[15],scale,
            lambda z:z[:,:256],lambda z:net.pi(z[:,:256]).softmax(1))
            for metric,scale in [('raw',np.ones(512)),('relative_rms',a['scale'])]}
        record = dict(condition=condition,seed=seed,action=action,action_name=name,
            checkpoint_sha256=sha(checkpoint),archive_sha256=sha(archive),
            source_sha256=sha(__file__),only_previous_action_changed=True,
            left_identity_replay_max_error=replay_error,
            full_state=result,observable=observable)
        save(folder/f'{name}.json',record)
        print(json.dumps(dict(condition=condition,seed=seed,action=name,
            full_state_candidate=result['passes_both_metric_candidate_screen'],
            centered_candidates={m:v['centered_conditional_factor_candidate']
                                 for m,v in observable.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=('gc100','pgc33'))
    p.add_argument('--seed',type=int,choices=(1,2));p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    design=protocol();frozen=args.out/'protocol.json'
    if frozen.exists(): assert json.loads(frozen.read_text())==design
    else:
        assert args.freeze_only,'Freeze and review before new neural outcomes'
        save(frozen,design)
    if args.freeze_only:return
    assert args.condition and args.seed
    with torch.no_grad(),threadpool_limits(limits=2):run(args.condition,args.seed,args.out)


if __name__=='__main__':main()
