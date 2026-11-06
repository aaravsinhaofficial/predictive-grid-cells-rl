"""DESIGNED goal-location ring connected to the original frozen RL actor.

The original Grid LSTM chooses all six actions in GoalCircleArena. A2D designed
ring replaces its saved-goal input. Actual analytic map rates alone initialize
ring phase; hidden goal coordinates appear only in evaluation diagnostics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy,build_population
from rl_goal_memory import GoalCircleArena

VARIANTS=('native_saved_code','ring_pickup','ring_ahead','ring_ahead_plus45','ring_ahead_minus45')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def ring_step(state,radial_multiplier=.9):
    radius=state.square().sum(-1,keepdim=True).sqrt()
    if bool((radius<=0).any()):raise ValueError('Ring state must have nonzero radius')
    return state*(radial_multiplier+(1-radial_multiplier)/radius)


def phase_rotate(state,angle):
    c,s=torch.cos(angle),torch.sin(angle)
    return torch.stack((c*state[:,0]-s*state[:,1],s*state[:,0]+c*state[:,1]),1)


class MapInverse:
    """Known analytic map inverse; no true positions or goal coordinates accepted.

    Coarse dictionary search initializes4 candidates. Batched Gauss–Newton then
    fits the observed cell rates. All rates used here are ordinary base maps;
    PGC goal slots therefore decode their shifted sampling position directly.
    """
    def __init__(self,pop,indices,limit=1.17,resolution=81):
        self.indices=torch.as_tensor(indices,dtype=torch.long)
        self.K=pop.K[self.indices].detach().cpu().double()
        self.phi=pop.phi[self.indices].detach().cpu().double()
        self.sharpness=float(pop.sharpness);self.limit=float(limit)
        axis=torch.linspace(-limit,limit,resolution,dtype=torch.float64)
        self.dictionary_points=torch.cartesian_prod(axis,axis)
        self.dictionary_rates=self.rates_jacobian(self.dictionary_points,False)
        self.dictionary_norm=self.dictionary_rates.square().sum(1)
        self.calls=0;self.samples=0;self.max_code_rmse=0.;self.seconds=0.

    def rates_jacobian(self,points,jacobian=True):
        projection=torch.einsum('bnd,nid->bni',points[:,None]-self.phi[None],self.K)
        base=((projection.cos().sum(-1)+1.5)/4.5).clamp(0,1)
        rates=base.pow(self.sharpness)
        if not jacobian:return rates
        db=-torch.einsum('bni,nid->bnd',projection.sin(),self.K)/4.5
        derivative=self.sharpness*base.pow(self.sharpness-1)[...,None]*db
        return rates,derivative

    @torch.no_grad()
    def decode(self,observed):
        started=time.perf_counter();rates=observed.detach().cpu().double()
        if rates.shape[1]!=len(self.indices):raise ValueError('Decoder must receive only its specified rate channels')
        distance=rates.square().sum(1,keepdim=True)+self.dictionary_norm[None]-2*rates@self.dictionary_rates.T
        starts=distance.topk(4,largest=False).indices
        points=self.dictionary_points[starts].reshape(-1,2).clone()
        target=rates.repeat_interleave(4,0)
        eye=torch.eye(2,dtype=torch.float64)
        for _ in range(9):
            pred,jac=self.rates_jacobian(points)
            residual=pred-target
            gradient=torch.einsum('bnd,bn->bd',jac,residual)
            curvature=torch.einsum('bni,bnj->bij',jac,jac)+1e-8*eye
            delta=torch.linalg.solve(curvature,gradient.unsqueeze(-1)).squeeze(-1)
            delta*=torch.clamp(.06/delta.norm(dim=1,keepdim=True).clamp_min(1e-20),max=1)
            points=(points-delta).clamp(-self.limit,self.limit)
        error=(self.rates_jacobian(points,False)-target).square().mean(1).reshape(-1,4)
        best=error.argmin(1);selected=points.reshape(-1,4,2)[torch.arange(len(rates)),best]
        rmse=error[torch.arange(len(rates)),best].sqrt()
        self.calls+=1;self.samples+=len(rates);self.max_code_rmse=max(self.max_code_rmse,float(rmse.max()));self.seconds+=time.perf_counter()-started
        if float(rmse.max())>2e-5:raise AssertionError(f'Observed map rates not reconstructed: {float(rmse.max())}')
        return selected.float(),rmse

    def diagnostics(self):
        return dict(calls=self.calls,samples=self.samples,max_code_rmse=self.max_code_rmse,seconds=self.seconds,
                    channels=len(self.indices),dictionary_points=len(self.dictionary_points),initial_candidates=4,gauss_newton_steps=9)


def summarize(rows,length):
    found=[r for r in rows if r['first_goal_step']>=0]
    eligible=[r for r in found if r['first_goal_step']<=length-300]
    return dict(eligible_first_returns=len(eligible),
        eligible_first_return_within20s_fraction=float(np.mean([r['first_return_steps']>=0 and r['first_return_steps']<=300 for r in eligible]))if eligible else None,
        eligible_first_return_capped20s_steps=float(np.mean([min(r['first_return_steps'],300)if r['first_return_steps']>=0 else 300 for r in eligible]))if eligible else None,
        excluded_late_discoveries=len(found)-len(eligible),episodes=len(rows),mean_score=float(np.mean([r['score']for r in rows])),
        fraction_discovered=len(found)/len(rows),fraction_at_least_two_goals=float(np.mean([r['n_goals']>=2 for r in rows])),
        first_return_within20s_fraction=float(np.mean([r['first_return_steps']>=0 and r['first_return_steps']<=300 for r in found]))if found else None,
        first_return_capped20s_steps=float(np.mean([min(r['first_return_steps'],300)if r['first_return_steps']>=0 else 300 for r in found]))if found else None,
        mean_returns_per_episode=float(np.mean([max(0,r['n_goals']-1)for r in rows])),rows=rows)


def validate_inverse(pop,ordinary,predictive):
    # Independent test positions, never used to initialize a decoder or its dictionary.
    gen=torch.Generator().manual_seed(719243)
    positions=(torch.rand(128,2,generator=gen)*2-1)*1.15
    rates=pop.activity(positions,shift_scale=0)
    errors={}
    for name,decoder in [('ordinary343',ordinary),('predictive169',predictive)]:
        recovered,_=decoder.decode(rates[:,decoder.indices]);error=(recovered-positions).norm(dim=1)
        assert float(error.max())<2e-5,float(error.max())
        errors[name]=dict(max_position_error_m=float(error.max()),mean_position_error_m=float(error.mean()),samples=128)
    return errors


def validate_ring():
    theta=torch.arange(360,dtype=torch.float64)*math.pi/180
    original=torch.stack((theta.cos(),theta.sin()),1)
    x=original*1.1
    for _ in range(30):x=ring_step(x)
    radius=x.norm(dim=1)
    assert torch.allclose(radius,torch.full_like(radius,1+.1*.9**30),atol=1e-13)
    assert torch.allclose(x/x.norm(dim=1,keepdim=True),original,atol=1e-13)
    # Identical cue-off map by construction: initial cue source is not an argument.
    return dict(radial_multiplier=.9,phase_multiplier=1.,steps=30,
                maximum_phase_error_deg=float(torch.rad2deg(torch.acos(torch.sum((x/x.norm(dim=1,keepdim=True))*original,1).clamp(-1,1))).max()),
                radius_error_gain=float((radius-1).mean()/.1),expected_radius_error_gain=.9**30,
                cue_source_does_not_enter_autonomous_map=True)


@torch.inference_mode()
def evaluate(args):
    started=time.perf_counter();cfgpath=ROOT/f'models/gc100/gc100_lstm_s{args.seed}_1000M.json';checkpoint=cfgpath.with_suffix('.pt')
    cfg=json.loads(cfgpath.read_text());weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert weights['frames']==1_000_000_000
    original_hash=sha(checkpoint)
    torch.manual_seed(args.eval_seed)
    if args.device=='mps':torch.mps.manual_seed(args.eval_seed)
    net=build_policy(cfg,args.device).eval().requires_grad_(False);net.load_state_dict(weights['policy'])
    live_pop=build_population(cfg,args.device)
    mixed_cfg=dict(cfg,cond='pgc33',pgc_frac=.33)
    mixed_pop=build_population(mixed_cfg,args.device);cpu_pop=build_population(mixed_cfg,'cpu')
    ordinary_idx=(~cpu_pop.is_pgc).nonzero().flatten();predictive_idx=cpu_pop.is_pgc.nonzero().flatten()
    ordinary=MapInverse(cpu_pop,ordinary_idx);predictive=MapInverse(cpu_pop,predictive_idx)
    inverse_check=validate_inverse(cpu_pop,ordinary,predictive);ring_check=validate_ring()
    n=args.n_envs;groups=len(VARIANTS);total=n*groups
    env=GoalCircleArena(cfg,n,args.eval_seed,args.device,radius=.65,paired=True,groups=groups)
    obs=env.reset_all()
    for key in ('pos','hd','dir','rgb'):
        assert all(torch.equal(obs[key][:n],obs[key][j*n:(j+1)*n])for j in range(1,groups))
    state=net.zero_state(total,args.device)
    ring=torch.zeros(total,2,device=args.device);ring[:,0]=1
    known=torch.zeros(total,dtype=torch.bool,device=args.device)
    native_goal=torch.zeros(n,512,device=args.device)
    pa=torch.full((total,),-1,dtype=torch.long,device=args.device);pr=torch.zeros(total,device=args.device)
    generator=torch.Generator(device=args.device).manual_seed(args.eval_seed+1)
    first_hit=torch.full((total,),-1,dtype=torch.long,device=args.device)
    first_return=torch.full_like(first_hit,-1)
    rows={v:[]for v in VARIANTS};decoding_rows=[];first_geometry=[];first_commands=[];last=started
    # Track first-second movement after the shared first pickup, before any branch
    # has returned. These are original policy trajectories, never scripted actions.
    pulse_origin=torch.zeros(total,2,device=args.device);pulse_recorded=torch.zeros(total,dtype=torch.bool,device=args.device)
    pulse_positions=[]
    for step in range(cfg['episode_len']*(args.episodes//n)):
        ring=ring_step(ring)
        remembered=.65*ring/ring.norm(dim=1,keepdim=True)
        goal=live_pop.activity(remembered,shift_scale=0)*known[:,None]
        goal[:n]=native_goal
        current=live_pop.activity(obs['pos'],obs['dir'],obs['vel'])
        logits,_,state=net.step(obs['rgb'],current,goal,pa,pr,state,None)
        just_received=(first_hit>=0)&(step-first_hit==1)
        if bool(just_received.any()):
            probabilities=logits.softmax(1)
            for i in just_received.nonzero().flatten().cpu().tolist():
                first_commands.append(dict(variant=VARIANTS[i//n],slot=i%n,episode=step//cfg['episode_len'],
                    position=obs['pos'][i].cpu().tolist(),heading=float(obs['hd'][i]),
                    ring_target=remembered[i].cpu().tolist(),probabilities=probabilities[i].cpu().tolist()))
        uniform=torch.rand(n,generator=generator,device=args.device).repeat(groups)
        action=(uniform[:,None]>logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        before_goal=env.goal.clone();before_known=known.clone();before_count=env.n_goals.clone()
        obs,reward,done,info=env.step(action);hit=info['hit']
        initial=hit&~before_known
        second=hit&(before_count==1)
        first_return=torch.where(second,step-first_hit,first_return)
        first_hit=torch.where(initial,torch.full_like(first_hit,step),first_hit)
        if bool(hit.any()):
            ids=hit.nonzero().flatten()
            # Exact original map generator: physical coordinates/direction are used
            # ONLY to generate the same sensory rates the original system supplies.
            rate=mixed_pop.activity(info['hit_pos'][ids],info['hit_dir'][ids],info['hit_vel'][ids])
            ring_ids=ids[ids>=n];ring_rate=rate[ids>=n]
            if len(ring_ids):
                # All candidates receive the identical noisy/noiseless rate interface;
                # decoder inputs contain no pose, direction, reward-center or angle.
                decoded_pickup,pickup_rmse=ordinary.decode(ring_rate[:,ordinary_idx.to(args.device)])
                decoded_ahead,ahead_rmse=predictive.decode(ring_rate[:,predictive_idx.to(args.device)])
                group=(ring_ids//n).cpu()
                point=torch.where((group==1)[:,None],decoded_pickup,decoded_ahead).to(args.device)
                written=point/point.norm(dim=1,keepdim=True)
                pulse_angle=torch.where((ring_ids//n)==3,torch.full((len(ring_ids),),math.pi/4,device=args.device),
                    torch.where((ring_ids//n)==4,torch.full((len(ring_ids),),-math.pi/4,device=args.device),torch.zeros(len(ring_ids),device=args.device)))
                pulse_angle*=initial[ring_ids]
                written=phase_rotate(written,pulse_angle)
                ring[ring_ids]=written
                # Ground truth is diagnostic only and is never passed back to a decoder.
                true_pickup=info['hit_pos'][ring_ids].cpu();true_ahead=true_pickup+.20*info['hit_dir'][ring_ids].cpu()
                decoding_rows.append(dict(samples=len(ring_ids),pickup_max_error_m=float((decoded_pickup-true_pickup).norm(dim=1).max()),
                    ahead_max_error_m=float((decoded_ahead-true_ahead).norm(dim=1).max()),
                    pickup_code_rmse_max=float(pickup_rmse.max()),ahead_code_rmse_max=float(ahead_rmse.max())))
            baseline_ids=ids[ids<n]
            if len(baseline_ids):native_goal[baseline_ids]=live_pop.activity(info['hit_pos'][baseline_ids],info['hit_dir'][baseline_ids],info['hit_vel'][baseline_ids])
            known|=hit
        if bool(initial.any()):
            # A common first discovery/respawn is the start of every pulse comparison.
            pulse_origin[initial]=obs['pos'][initial]
            for slot in initial[:n].nonzero().flatten().cpu().tolist():
                first_geometry.append(dict(slot=slot,episode=step//cfg['episode_len'],step=step,goal=before_goal[slot].cpu().tolist(),
                    pickup=info['hit_pos'][slot].cpu().tolist(),arrival=info['hit_dir'][slot].cpu().tolist()))
        record=(first_hit>=0)&(step-first_hit==15)&~pulse_recorded&(env.n_goals==1)
        if bool(record.any()):
            for i in record.nonzero().flatten().cpu().tolist():
                pulse_positions.append(dict(variant=VARIANTS[i//n],slot=i%n,episode=step//cfg['episode_len'],
                    origin=pulse_origin[i].cpu().tolist(),position=obs['pos'][i].cpu().tolist(),true_goal=env.goal[i].cpu().tolist()))
            pulse_recorded|=record
        pa,pr=action,reward.clamp(-cfg['reward_clip'],cfg['reward_clip'])
        if bool(done.any()):
            stats=info['ep_stats']
            for j,index in enumerate(stats['env'].cpu().tolist()):
                group,slot=divmod(index,n)
                rows[VARIANTS[group]].append(dict(slot=slot,episode=step//cfg['episode_len'],score=float(stats['ret'][j]),
                    n_goals=int(stats['n_goals'][j]),first_goal_step=int(stats['first_goal_t'][j]),first_return_steps=int(first_return[index])))
            keep=(~done).float();state=tuple(z*keep[:,None]for z in state)
            known&=~done;ring[done]=torch.tensor([1.,0.],device=args.device);native_goal*=keep[:n,None]
            pa=torch.where(done,torch.full_like(pa,-1),pa);pr*=keep
            first_hit=torch.where(done,torch.full_like(first_hit,-1),first_hit);first_return=torch.where(done,torch.full_like(first_return,-1),first_return);pulse_recorded&=~done
        if time.perf_counter()-last>25:
            print(f'Original goal-ring {step+1}/{cfg["episode_len"]*(args.episodes//n)} decisions;{time.perf_counter()-started:.1f}s',flush=True);last=time.perf_counter()
    first=[(r['slot'],r['episode'],r['first_goal_step'])for r in rows[VARIANTS[0]]]
    assert all([(r['slot'],r['episode'],r['first_goal_step'])for r in rr]==first for rr in rows.values())
    assert sha(checkpoint)==original_hash
    result=dict(stage='timing_smoke'if args.smoke else'heldout_original_actor_replication',
        original_actor=f'gc100_lstm_s{args.seed}_1000M',checkpoint_sha256=original_hash,original_weights_unchanged=True,
        episodes_per_variant=args.episodes,eval_seed=args.eval_seed,device=args.device,seconds=time.perf_counter()-started,
        variants={v:summarize(r,cfg['episode_len'])for v,r in rows.items()},
        validation=dict(all_first_discoveries_identical=True,map_inverse_independent_positions=inverse_check,ring=ring_check,
            decoder_live_max_pickup_error_m=max(r['pickup_max_error_m']for r in decoding_rows),
            decoder_live_max_ahead_error_m=max(r['ahead_max_error_m']for r in decoding_rows),
            decoder_ordinary=ordinary.diagnostics(),decoder_predictive=predictive.diagnostics()),
        first_hit_geometry=first_geometry,first_second_positions=pulse_positions,first_postpickup_commands=first_commands,
        interpretation='The ring is designed, not discovered inside the LSTM. Identical cue-free ring dynamics make intrinsic stability identical across cue sources. Any difference measures cue accuracy/behavior through the original actor.')
    save_json(args.out/'results.json',result)
    print(json.dumps({v:r['mean_score']for v,r in result['variants'].items()}),flush=True)
    return result


def protocol(args):
    sources=['scripts/original_rl_goal_ring_cohort.py','scripts/rl_goal_memory.py','code/env.py','code/common.py','code/codes.py','code/nets.py']
    return dict(stage='Designed goal-location ring attached to the original frozen RL actor; exploratory single-seed demonstration',
        original_actor=f'Grid LSTMseed{args.seed} at1Bframes',variants=VARIANTS,eval_seed=args.eval_seed,episodes_per_variant=args.episodes,
        environment='Original square ArenaEnv with hidden goal centers on radius0.65m; six original actions, original sparse reward and respawn',
        inference='Observed343ordinary goal-map rates or169predictive goal-map rates → exactknown-map inverse →2D goal-location ring → full512ordinary spatial goal code → frozen original actor',
        ring=dict(radial_multiplier=.9,phase_multiplier=1.,dimension=2,radius_m=.65,update='Every actual reward writes a unit state from the decoded observed point; otherwise autonomous recurrence only',pulse='±45degrees once immediately after first reward in the two pulse branches'),
        evidence_boundary='No true goal coordinate or direction/angle label enters decoder/ring/actor. Physical pose/direction are used only by the original map generator and evaluation. Designed ring and identical cue-off dynamics are explicit.',
        primary_descriptive_comparisons=['ring_ahead vs ring_pickup navigation and firstreturn','ring_ahead vs±45phasepulses firstreturn and actual displacement'],
        statistics='One preselected original trainingseed; episode-level descriptions only, no population-significance claim',
        source_sha256={p:sha(ROOT/p)for p in sources})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/original_rl_goal_ring/confirmation')
    ap.add_argument('--device',default='mps');ap.add_argument('--seeds',type=int,nargs='+',default=[2,3,4,5,6,7,8])
    ap.add_argument('--episodes',type=int,default=64);ap.add_argument('--n-envs',type=int,default=32)
    ap.add_argument('--eval-seed',type=int,default=63001);ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    if args.seeds!=[2,3,4,5,6,7,8] or args.episodes!=64 or args.n_envs!=32 or args.eval_seed!=63001:
        raise ValueError('This replication uses the fixed complete seed2–8 protocol')
    args.smoke=False;args.seed=2;args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    plan=protocol(args)
    plan.update(stage='Heldout original Gridactor replication; seed1 remains the descriptive feasibility pilot',
        original_actor_seeds=args.seeds,statistical_unit='Original trained Grid policy; n7 independent trainingseeds',
        primary=dict(name='pickup_minus_ahead_first_return_seconds',
            definition='Within eachseed, mean firstreturntime capped20s for ringpickup minus ringahead; include only episodes whose shared firstdiscovery occurs bystep1050, ensuring300updates opportunity; every eligiblefailure contributes20s',
            eligibility='A shared preintervention firstdiscovery≤1050; no postintervention outcome exclusions',
            tests='Two-sided exact paired signflip across all128seed sign combinations; pairedt and95%t interval reported alongside; one primarycontrast'),
        secondary=dict(contrasts=['plus45_minus_ahead_first_return_seconds','minus45_minus_ahead_first_return_seconds'],
            definition='Same eligibility/capping as primary; two-sided exact pairedsignflip and pairedt; two contrasts corrected with Holm as a separate prespecified family'),
        descriptive='All64episodes perbranch retained for reward; all-discovered censored20s summaries and initialpostpickup command changes remain descriptive; seed1 never enters confirmatorytests',
        amendment_from_v1='Before any seed2–8 outcomes: primary requires shared firstdiscovery≤1050. v1pilot all-discovered endpoint remains unchanged; late censoredtrips are not assumed tohave failed after the observedepisode in replicationprimary.',
        original_v1_source_sha256=sha(ROOT/'scripts/original_rl_goal_ring.py'),
        original_checkpoint_sha256={str(s):sha(ROOT/f'models/gc100/gc100_lstm_s{s}_1000M.pt')for s in args.seeds})
    plan['statistics']='One primarycontrast and a separate two-comparisonsecondaryHolmfamily; allsevenheldouttrainingseeds included regardless result'
    p=args.out/'protocol.json'
    if not p.exists():save_json(p,plan)
    else:
        prior=json.loads(p.read_text());assert prior==plan,'Frozen replication protocol or source changed'
    if args.freeze_only:return
    root_out=args.out
    for seed in args.seeds:
        args.seed=seed;args.out=root_out/f'seed{seed}';args.out.mkdir(exist_ok=True)
        if (args.out/'results.json').exists():
            print(f'Seed{seed} already complete; retained without rerunning',flush=True);continue
        evaluate(args)

if __name__=='__main__':main()
