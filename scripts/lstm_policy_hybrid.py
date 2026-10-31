"""Causal return-navigation comparison using intact trained policy controllers.

All branches use the grid policy until the first pickup. Afterward select
grid/PGC action probabilities everywhere, near the goal, or far from it.
Both RNNs always follow the branch's actual experience with native inputs.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from lstm_code_swap import MatchedArena,summarize_episodes,verify_codes

VARIANTS=('grid','pgc_after_discovery','pgc_near','pgc_far')


@torch.inference_mode()
def run(seed,a):
    started=time.perf_counter()
    torch.manual_seed(a.eval_seed)
    if a.device=='mps':torch.mps.manual_seed(a.eval_seed)
    nets,configs,hashes={},{},{}
    for c in ('gc100','pgc33'):
        path=ROOT/f'models/{c}/{c}_lstm_s{seed}_1000M.json'
        configs[c]=json.loads(path.read_text())
        ck=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
        assert ck['frames']==1_000_000_000
        hashes[c]=hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()
        nets[c]=build_policy(configs[c],a.device).eval().requires_grad_(False)
        nets[c].load_state_dict(ck['policy'])
    cfg=configs['gc100']
    assert all(configs['pgc33'][k]==cfg[k] for k in ('arena_half','episode_len','goal_radius','n_cells','nh','vision',
        'action_repeat','v_max','rot_max','reward_clip','img_noise','randomize_textures'))
    pop=build_population(configs['pgc33'],a.device)
    n,groups=a.n_envs,len(VARIANTS)
    total=n*groups
    env=MatchedArena(cfg,n,groups,a.eval_seed,a.device)
    obs=env.reset_all()
    states={c:net.zero_state(total,a.device) for c,net in nets.items()}
    goals={c:torch.zeros(total,512,device=a.device) for c in nets}
    pa=torch.full((total,),-1,dtype=torch.long,device=a.device)
    pr=torch.zeros(total,device=a.device)
    gen=torch.Generator(device=a.device).manual_seed(a.eval_seed+1)
    rows={v:[] for v in VARIANTS}
    counts=torch.zeros(groups,3,dtype=torch.long,device=a.device)
    step,last=0,started
    while len(rows['grid'])<a.episodes:
        known=env.n_goals>0
        near=(obs['pos']-env.goal).norm(dim=1)<=a.near_radius
        use_pgc=torch.cat((torch.zeros_like(known[:n]),known[n:2*n],
                           known[2*n:3*n]&near[2*n:3*n],known[3*n:]&~near[3*n:]))
        codes={'gc100':pop.activity(obs['pos'],obs['dir'],obs['vel'],shift_scale=0),
               'pgc33':pop.activity(obs['pos'],obs['dir'],obs['vel'])}
        logits={}
        for c,net in nets.items():
            logits[c],_,states[c]=net.step(obs['rgb'],codes[c],goals[c],pa,pr,states[c])
        policy=torch.where(use_pgc[:,None],logits['pgc33'],logits['gc100'])
        uniform=torch.rand(n,generator=gen,device=a.device).repeat(groups)
        action=(uniform[:,None]>policy.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        counts+=torch.stack((known,known&near,use_pgc),1).reshape(groups,n,3).sum(1)
        obs,reward,done,info=env.step(action)
        hit=info['hit']
        if bool(hit.any()):
            for c in nets:
                v=pop.activity(info['hit_pos'],info['hit_dir'],info['hit_vel'],shift_scale=0 if c=='gc100' else 1)
                goals[c]=torch.where(hit[:,None],v,goals[c])
        pa,pr=action,reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            keep=(~done).float()
            for c in nets:
                states[c]=tuple(s*keep[:,None] for s in states[c])
                goals[c]*=keep[:,None]
            pa=torch.where(done,torch.full_like(pa,-1),pa)
            pr*=keep
            es={k:v.cpu().tolist() for k,v in info['ep_stats'].items()}
            for i,slot in enumerate(es['env']):
                rows[VARIANTS[slot//n]].append(dict(slot=slot%n,episode=step//cfg['episode_len'],
                    **{k:es[k][i] for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')}))
        step+=1
        if time.perf_counter()-last>25:
            print(f'hybrid s{seed}: {step}/{cfg["episode_len"]*(a.episodes//n)}; {time.perf_counter()-started:.0f}s',flush=True)
            last=time.perf_counter()
    first=[r['first_goal_t'] for r in rows['grid']]
    assert all([r['first_goal_t'] for r in row]==first for row in rows.values())
    exact=None
    if (a.episodes,a.n_envs,a.eval_seed)==(128,64,737373):
        old=json.loads((ROOT/f'results/lstm_code_swap_1B/gc100_s{seed}.json').read_text())
        assert rows['grid']==old['variants']['intact']['rows'],(seed,'grid baseline mismatch')
        exact=True
    for c in nets:
        assert hashlib.sha256((ROOT/f'models/{c}/{c}_lstm_s{seed}_1000M.pt').read_bytes()).hexdigest()==hashes[c]
    result=dict(seed=seed,episodes=a.episodes,n_envs=n,eval_seed=a.eval_seed,near_radius=a.near_radius,
        checkpoint_sha256=hashes,device=a.device,
        checks=dict(first_goal_times_identical=True,grid_baseline_exact=exact,checkpoint_hashes_unchanged=True),
        exposure={v:dict(zip(('known_steps','near_steps','pgc_controller_steps'),row)) for v,row in zip(VARIANTS,counts.cpu().tolist())},
        variants={v:summarize_episodes(row,cfg['episode_len']) for v,row in rows.items()},
        elapsed_seconds=time.perf_counter()-started)
    (a.out/f's{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'DONE hybrid s{seed}: '+', '.join(f'{v}={d["mean_score"]:.2f}' for v,d in result['variants'].items()),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='mps')
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--n-envs',type=int,default=64)
    p.add_argument('--eval-seed',type=int,default=737373)
    p.add_argument('--near-radius',type=float,default=.6)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_policy_hybrid_1B')
    a=p.parse_args()
    assert a.episodes%a.n_envs==0
    torch.set_num_threads(2)
    a.out.mkdir(exist_ok=True,parents=True)
    protocol=dict(seeds=a.seeds,episodes=a.episodes,n_envs=a.n_envs,eval_seed=a.eval_seed,
        near_radius=a.near_radius,variants=VARIANTS,
        primary_score_comparisons=[['grid','pgc_after_discovery'],['grid','pgc_near'],['pgc_near','pgc_far']],
        test_family='Three primary paired seed contrasts, Holm t and exhaustive sign flips separately. Remaining comparisons exploratory.',
        design='Every branch follows intact grid until first goal. Then select intact grid/PGC actor everywhere, only within0.6m of true goal, or only outside0.6m. Both frozen RNNs receive their native current/goal codes, the branch actual observations/actions/rewards, and maintain separate natural h/c. No neural state is transferred across models.',
        caveats=['Uses privileged goal distance to choose controller; diagnostic not deployable policy.',
                 'Switching controller changes future trajectories and both shadow-state histories.',
                 'PGC-after-discovery is not the original native PGC policy before discovery.',
                 'Localizes usefulness of learned controllers; does not identify individual neurons or fully explain training.'],
        code_identity={s:verify_codes(s) for s in a.seeds})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for s in a.seeds:
        dest=a.out/f's{s}.json'
        if dest.exists():
            old=json.loads(dest.read_text())
            assert (old['episodes'],old['n_envs'],old['eval_seed'],old['near_radius'])==(a.episodes,a.n_envs,a.eval_seed,a.near_radius)
            print(f'Using completed hybrid s{s}',flush=True)
        else:run(s,a)


if __name__=='__main__':main()
