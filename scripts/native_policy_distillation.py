"""Small recurrent-only policy-distillation continuation in the original native arena."""
from __future__ import annotations
import argparse,copy,datetime as dt,hashlib,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from code.common import build_policy,build_population
from lstm_code_swap import MatchedArena
from native_movement_cycles import verify_step,ENV_KEYS
from rl_visual_memory import VisualClock,NativeGoal
from rl_goal_memory import atomic_json,atomic_torch,state_digest,reset_state,episode_summary
PROTOCOL=ROOT/'docs/research/native_policy_distillation_protocol.json'
BASE=ROOT/'results/native_policy_distillation'
TRAIN_SEED=971201;EVAL_SEED=971901;REPLAY_SEED=972201
N=16;UNROLL=90;BLOCK_UPDATES=60;GAPS=(4,15,30);BETAS=(.5,.25,0.)
VARIANTS=('seen','blind','blind_current_grid','blind_every_reset','blind_current_grid_every_reset','seen_scheduled_every_reset')
SOURCES=('scripts/native_policy_distillation.py','tests/test_native_policy_distillation.py','tests/test_native_policy_distillation_review.py','scripts/native_movement_cycles.py','scripts/rl_visual_memory.py','scripts/rl_goal_memory.py','scripts/lstm_code_swap.py','code/__init__.py','code/common.py','code/codes.py','code/env.py','code/nets.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Fixed30-minute pilot cap reached; partial evidence retained')
def frozen_digest(net):
    digest=hashlib.sha256()
    for name,value in sorted(net.state_dict().items()):
        if not name.startswith('lstm.'):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def kl(teacher_logits,student_logits):
    target=teacher_logits.detach().log_softmax(-1);prediction=student_logits.log_softmax(-1)
    raw=(target.exp()*(target-prediction)).sum(-1)
    # Exact equality is the KL minimum; suppress floating-point residual gradients there.
    return torch.where((target==prediction).all(-1),student_logits.sum(-1)*0.,raw)
def sample(logits,uniform):return(uniform[:,None]>logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(5)
def trainable_student(teacher):
    student=copy.deepcopy(teacher).eval().requires_grad_(False)
    student.lstm.requires_grad_(True)
    assert {name for name,p in student.named_parameters()if p.requires_grad}=={'lstm.weight_ih','lstm.weight_hh','lstm.bias_ih','lstm.bias_hh'}
    return student


def load_models(condition):
    path=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json';cfg=json.loads(path.read_text())
    checkpoint=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
    assert checkpoint['frames']==1_000_000_000 and cfg['episode_len']==1350 and cfg['nh']==256 and cfg['extra_input']=='none'
    assert cfg['img_noise']==.02 and cfg['code_noise']==0 and cfg['goal_code_shift']=='intact'
    teacher=build_policy(cfg,'cpu').eval().requires_grad_(False);teacher.load_state_dict(checkpoint['policy'])
    return teacher,trainable_student(teacher),cfg


def chunk_forward(net,x,state,n):
    states=[];logits=[]
    for start in range(0,len(x),n):
        part=slice(start,start+n);z=net.lstm(x[part],(state[0][part],state[1][part]))
        states.append(z);logits.append(net.pi(z[0]))
    return torch.cat(logits),tuple(torch.cat([z[k]for z in states])for k in (0,1))


@torch.no_grad()
def feature_inputs(net,pop,obs,goal,pa,pr,n):
    embeddings=torch.cat([net.embed(obs['rgb'][j:j+n])for j in range(0,len(pa),n)])
    native=torch.cat([pop.activity(obs['pos'][j:j+n],obs['dir'][j:j+n],obs['vel'][j:j+n])for j in range(0,len(pa),n)])
    ordinary=torch.cat([pop.activity(obs['pos'][j:j+n],obs['dir'][j:j+n],obs['vel'][j:j+n],shift_scale=0.)for j in range(0,len(pa),n)])
    assert torch.equal(native[:,~pop.is_pgc],ordinary[:,~pop.is_pgc])
    return net._concat(embeddings,native,goal,pa,pr,None),ordinary


def student_inputs(native,ordinary,black_embedding,blind,swap):
    x=native.clone();x[:,:256]=torch.where(blind[:,None],black_embedding,x[:,:256]);x[:,256:768]=torch.where(swap[:,None],ordinary,x[:,256:768])
    assert torch.equal(x[:,768:],native[:,768:])
    return x


@torch.no_grad()
def advance_goal(pop,goal,info,done,n):
    hit=info['hit'];code=torch.zeros_like(goal.saved)
    for j in range(0,len(hit),n):
        part=slice(j,j+n);mask=hit[part]
        if bool(mask.any()):
            values=code[part];values[mask]=pop.activity(info['hit_pos'][part][mask],info['hit_dir'][part][mask],info['hit_vel'][part][mask])
    goal.advance(hit,done,code)


def replay_logits(student,inputs,state0,done):
    state=tuple(z.clone()for z in state0);values=[]
    for x,d in zip(inputs,done):
        state=student.lstm(x,state);values.append(student.pi(state[0]));state=reset_state(state,d)
    return torch.stack(values),state


def train_unroll(student,teacher,pop,cfg,env,runtime,opt,gap,beta,length=UNROLL):
    obs=runtime['obs'];state=tuple(z.detach()for z in runtime['student_state']);teacher_state=runtime['teacher_state']
    state0=tuple(z.clone()for z in state);goal=runtime['goal'];clock=runtime['clock'];pa=runtime['pa'];pr=runtime['pr'];swap_interval=runtime['swap_interval']
    inputs=[];targets=[];logits_saved=[];done_saved=[];blacks=[];swaps=[];choices=[];losses=[];episodes=[]
    for _ in range(length):
        check_time(runtime['deadline']);scheduled=clock.mask()
        native,ordinary=feature_inputs(teacher,pop,obs,goal.saved,pa,pr,N)
        x=student_inputs(native,ordinary,runtime['black_embedding'],scheduled,scheduled&swap_interval)
        with torch.no_grad():teacher_logits,teacher_state=chunk_forward(teacher,native,teacher_state,N)
        student_logits,state=chunk_forward(student,x,state,N);losses.append(kl(teacher_logits,student_logits))
        with torch.no_grad():
            choose=torch.rand(N,generator=runtime['mixture_rng'])<beta
            action=sample(torch.where(choose[:,None],teacher_logits,student_logits.detach()),torch.rand(N,generator=runtime['action_rng']))
        inputs.append(x);targets.append(teacher_logits.detach());logits_saved.append(student_logits.detach());blacks.append(scheduled.clone());swaps.append((scheduled&swap_interval).clone());choices.append(choose)
        obs,raw,done,info=env.step(action);done_saved.append(done.clone());advance_goal(pop,goal,info,done,N)
        entering=clock.advance(info['hit'],done,gap)
        draws=torch.rand(N,generator=runtime['substitution_rng'])<.5
        swap_interval=torch.where(entering,draws,swap_interval)&clock.mask()
        state=reset_state(state,done);teacher_state=reset_state(teacher_state,done)
        pa=torch.where(done,torch.full_like(action,-1),action);pr=raw.clamp(-cfg['reward_clip'],cfg['reward_clip'])*(~done)
        if info['ep_stats'] is not None:episodes.extend(info['ep_stats']['ret'].tolist())
    all_losses=torch.stack(losses);loss=all_losses.mean();assert bool(torch.isfinite(loss))
    xstored=torch.stack(inputs);dstored=torch.stack(done_saved);lstored=torch.stack(logits_saved)
    with torch.no_grad():replayed,replayed_state=replay_logits(student,xstored,state0,dstored)
    replay_error=float((replayed-lstored).abs().max());assert replay_error==0
    assert all(torch.equal(a,b.detach())for a,b in zip(replayed_state,state))
    before=[p.detach().clone()for p in student.lstm.parameters()];opt.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in student.lstm.parameters())
    gradient=float(torch.nn.utils.clip_grad_norm_(student.lstm.parameters(),1.));assert torch.isfinite(torch.tensor(gradient))
    opt.step();change=sum(float((p.detach()-b).square().sum())for p,b in zip(student.lstm.parameters(),before))**.5
    assert all(bool(torch.isfinite(p).all())for p in student.lstm.parameters())
    runtime.update(obs=obs,student_state=tuple(z.detach()for z in state),teacher_state=tuple(z.detach()for z in teacher_state),pa=pa,pr=pr,swap_interval=swap_interval)
    blind=torch.stack(blacks);swapped=torch.stack(swaps)
    metrics=dict(loss=float(loss.detach()),blind_kl=float(all_losses.detach()[blind].mean())if bool(blind.any())else None,
        visible_kl=float(all_losses.detach()[~blind].mean())if bool((~blind).any())else None,
        blind_updates=int(blind.sum()),ordinary_current_updates=int(swapped.sum()),teacher_actions=int(torch.stack(choices).sum()),
        grad_norm_before_clip=gradient,parameter_update_l2=change,replay_max_error=replay_error,completed_episode_scores=episodes)
    probe=dict(student_inputs=xstored,teacher_logits=torch.stack(targets),student_logits=lstored,state0=state0,done=dstored,blind=blind,ordinary_current=swapped)
    return metrics,probe


def masks_for_variants(scheduled,n):
    blind=torch.tensor([name.startswith('blind')for name in VARIANTS]).repeat_interleave(n)&scheduled
    swap=torch.tensor(['current_grid'in name for name in VARIANTS]).repeat_interleave(n)&scheduled
    reset=torch.tensor(['reset'in name for name in VARIANTS]).repeat_interleave(n)&scheduled
    return blind,swap,reset


@torch.inference_mode()
def evaluate_behavior(student,teacher,cfg,deadline):
    n=N;groups=len(VARIANTS);env=MatchedArena(cfg,n,groups,EVAL_SEED,'cpu');obs=env.reset_all();pop=build_population(cfg,'cpu')
    clock=VisualClock(n,'cpu',EVAL_SEED+2,groups=groups);goal=NativeGoal(n*groups,512,'cpu');pa=torch.full((n*groups,),-1,dtype=torch.long);pr=torch.zeros(n*groups)
    state=student.zero_state(n*groups,'cpu');ts=teacher.zero_state(n*groups,'cpu');rng=torch.Generator().manual_seed(EVAL_SEED+1)
    black=student.embed(torch.zeros(n,3,84,84)).repeat(groups,1);rows={name:[]for name in VARIANTS};kl_sum=torch.zeros(groups,dtype=torch.float64);blind_kl=torch.zeros_like(kl_sum);blind_count=torch.zeros_like(kl_sum)
    for step in range(cfg['episode_len']):
        if step%100==0:check_time(deadline)
        scheduled=clock.mask();blind,swap,reset=masks_for_variants(scheduled,n);state=reset_state(state,reset)
        native,ordinary=feature_inputs(teacher,pop,obs,goal.saved,pa,pr,n);x=student_inputs(native,ordinary,black,blind,swap)
        logits,state=chunk_forward(student,x,state,n);tlogits,ts=chunk_forward(teacher,native,ts,n);loss=kl(tlogits,logits).view(groups,n).double()
        kl_sum+=loss.sum(1);blind_kl+=(loss*scheduled.view(groups,n)).sum(1);blind_count+=scheduled.view(groups,n).sum(1)
        action=torch.cat([sample(logits[g*n:(g+1)*n],u)for g,u in enumerate([torch.rand(n,generator=rng)]*groups)])
        obs,raw,done,info=env.step(action);advance_goal(pop,goal,info,done,n);clock.advance(info['hit'],done,30)
        state=reset_state(state,done);ts=reset_state(ts,done);pa=torch.where(done,torch.full_like(action,-1),action);pr=raw.clamp(-cfg['reward_clip'],cfg['reward_clip'])*(~done)
        stats=info['ep_stats']
        if stats is not None:
            for j,index in enumerate(stats['env'].tolist()):
                group,slot=divmod(index,n);rows[VARIANTS[group]].append(dict(slot=slot,score=float(stats['ret'][j]),n_goals=int(stats['n_goals'][j]),first_goal_step=int(stats['first_goal_t'][j]),path_eff=float(stats['path_eff'][j]),path_eff_n=int(stats['path_eff_n'][j])))
    assert all(len(rr)==N for rr in rows.values());assert all([r['first_goal_step']for r in rr]==[r['first_goal_step']for r in rows['seen']]for rr in rows.values())
    if cfg['cond']=='gc100':assert rows['blind']==rows['blind_current_grid']and rows['blind_every_reset']==rows['blind_current_grid_every_reset']
    metrics={name:episode_summary(rr,cfg['episode_len'])for name,rr in rows.items()};scores={name:r['mean_score']for name,r in metrics.items()}
    return dict(complete=True,episodes_per_variant=N,seed=EVAL_SEED,max_gap=30,variants=metrics,
        mean_teacher_kl=dict(zip(VARIANTS,(kl_sum/(N*cfg['episode_len'])).tolist())),scheduled_gap_teacher_kl=dict(zip(VARIANTS,(blind_kl/blind_count.clamp_min(1)).tolist())),
        scheduled_gap_counts=dict(zip(VARIANTS,blind_count.tolist())),grid_shams_exact=cfg['cond']=='gc100',
        contrasts=dict(blind_reset_cost=scores['blind']-scores['blind_every_reset'],ordinary_blind_reset_cost=scores['blind_current_grid']-scores['blind_current_grid_every_reset'],
            seen_scheduled_reset_cost=scores['seen']-scores['seen_scheduled_every_reset'],additional_blind_reset_cost=(scores['blind']-scores['blind_every_reset'])-(scores['seen']-scores['seen_scheduled_every_reset']),
            additional_ordinary_blind_reset_cost=(scores['blind_current_grid']-scores['blind_current_grid_every_reset'])-(scores['seen']-scores['seen_scheduled_every_reset']),current_prediction_benefit=scores['blind']-scores['blind_current_grid']))


@torch.inference_mode()
def evaluate_replay(student,condition,deadline):
    folder=ROOT/f'results/native_movement_cycles/{condition}_s1';data=torch.load(folder/'native_trace.pt',map_location='cpu',weights_only=True)
    bank=torch.load(BASE/'masks'/f'{condition}.pt',map_location='cpu',weights_only=True);assert bank['trace_sha256']==sha(folder/'native_trace.pt')
    groups=len(VARIANTS);n=32;state=student.zero_state(groups*n,'cpu');black=student.embed(torch.zeros(n,3,84,84)).repeat(groups,1)
    sums=torch.zeros(groups,dtype=torch.float64);gap_sums=sums.clone();gap_counts=sums.clone();tv_sums=sums.clone()
    for step in range(len(data['x'])):
        if step%100==0:check_time(deadline)
        scheduled=bank['blind'][step].repeat(groups);blind,swap,reset=masks_for_variants(scheduled,n);state=reset_state(state,reset)
        native=data['x'][step].repeat(groups,1);ordinary=data['current_ordinary'][step].repeat(groups,1)
        x=student_inputs(native,ordinary,black,blind,swap);logits,state=chunk_forward(student,x,state,n)
        target=data['logits'][step].repeat(groups,1);loss=kl(target,logits).view(groups,n).double();sums+=loss.sum(1)
        gap_sums+=(loss*scheduled.view(groups,n)).sum(1);gap_counts+=scheduled.view(groups,n).sum(1)
        tv_sums+=(logits.softmax(1)-target.softmax(1)).abs().sum(1).view(groups,n).double().sum(1)/2
        state=reset_state(state,data['done'][step].repeat(groups))
    return dict(complete=True,episodes=32,mask_sha256=sha(BASE/'masks'/f'{condition}.pt'),trace_sha256=bank['trace_sha256'],
        mean_kl=dict(zip(VARIANTS,(sums/(n*len(data['x']))).tolist())),scheduled_gap_kl=dict(zip(VARIANTS,(gap_sums/gap_counts.clamp_min(1)).tolist())),
        mean_action_tv=dict(zip(VARIANTS,(tv_sums/(n*len(data['x']))).tolist())),scheduled_gap_counts=dict(zip(VARIANTS,gap_counts.tolist())),evaluation_only=True)


def snapshot(student,teacher,cfg,opt,env,runtime,updates,design):
    return dict(policy=student.state_dict(),teacher_policy_hash=state_digest(teacher),optimizer=opt.state_dict(),config=cfg,updates=updates,
        original_frames=1_000_000_000,extra_frames=updates*N*UNROLL*cfg['action_repeat'],protocol_sha256=sha(PROTOCOL),
        cpu_rng=torch.get_rng_state(),environment_rng=env.gen.get_state(),environment_tensors={k:v.clone()for k,v in vars(env).items()if isinstance(v,torch.Tensor)},
        environment_scalars={k:v for k,v in vars(env).items()if isinstance(v,(int,float,bool,str))},
        student_state=runtime['student_state'],teacher_state=runtime['teacher_state'],obs=runtime['obs'],goal=runtime['goal'].saved,pa=runtime['pa'],pr=runtime['pr'],swap_interval=runtime['swap_interval'],
        clock=dict(known=runtime['clock'].known,blind=runtime['clock'].blind,remaining=runtime['clock'].remaining,rng=runtime['clock'].generator.get_state()),
        action_rng=runtime['action_rng'].get_state(),mixture_rng=runtime['mixture_rng'].get_state(),substitution_rng=runtime['substitution_rng'].get_state(),source_sha256=design['source_sha256'])


def run(condition,out,max_seconds):
    started=time.monotonic();deadline=started+max_seconds;out=Path(out);assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    design=json.loads(PROTOCOL.read_text())
    for key in ('source_sha256','asset_sha256'):
        for name,digest in design[key].items():assert sha(ROOT/name)==digest,name
    torch.manual_seed(TRAIN_SEED);teacher,student,cfg=load_models(condition);teacher_hash=state_digest(teacher);fixed_hash=frozen_digest(student)
    pop=build_population(cfg,'cpu');verification=verify_step(teacher,pop,cfg);assert state_digest(student)==teacher_hash
    atomic_json(out/'protocol.json',dict(protocol_sha256=sha(PROTOCOL),condition=condition,torch_version=str(torch.__version__),threads=torch.get_num_threads(),teacher_policy_hash=teacher_hash,frozen_tensor_hash=fixed_hash,verification=verification))
    atomic_torch(out/'initial.pt',dict(policy=student.state_dict(),config=cfg,original_frames=1_000_000_000,updates=0,protocol_sha256=sha(PROTOCOL)))
    updates=0
    try:
        baseline=evaluate_behavior(student,teacher,cfg,deadline);assert baseline['mean_teacher_kl']['seen']==0
        atomic_json(out/'baseline_behavior.json',baseline);atomic_json(out/'baseline_replay.json',evaluate_replay(student,condition,deadline))
        print(json.dumps(dict(condition=condition,phase='baseline_complete',seconds=time.monotonic()-started,seen_score=baseline['variants']['seen']['mean_score'])),flush=True)
        env=MatchedArena(cfg,N,1,TRAIN_SEED,'cpu');obs=env.reset_all();opt=torch.optim.Adam(student.lstm.parameters(),lr=1e-4)
        runtime=dict(obs=obs,student_state=student.zero_state(N,'cpu'),teacher_state=teacher.zero_state(N,'cpu'),goal=NativeGoal(N,512,'cpu'),clock=VisualClock(N,'cpu',TRAIN_SEED+2),
            pa=torch.full((N,),-1,dtype=torch.long),pr=torch.zeros(N),swap_interval=torch.zeros(N,dtype=torch.bool),deadline=deadline,
            black_embedding=teacher.embed(torch.zeros(N,3,84,84)).detach(),action_rng=torch.Generator().manual_seed(TRAIN_SEED+1),mixture_rng=torch.Generator().manual_seed(TRAIN_SEED+3),substitution_rng=torch.Generator().manual_seed(TRAIN_SEED+4))
        saved_probe=False;any_update=False
        with(out/'metrics.jsonl').open('w')as log:
            for update in range(BLOCK_UPDATES*3):
                stage=update//BLOCK_UPDATES;metrics,probe=train_unroll(student,teacher,pop,cfg,env,runtime,opt,GAPS[stage],BETAS[stage]);updates=update+1
                row=dict(update=updates,stage=stage+1,gap=GAPS[stage],teacher_beta=BETAS[stage],physics_frames=updates*N*UNROLL*4,seconds=time.monotonic()-started,**metrics)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();any_update|=metrics['parameter_update_l2']>0
                if not saved_probe and metrics['blind_updates']>0:
                    atomic_torch(out/'first_blind_unroll.pt',dict(update=updates,**probe));saved_probe=True
                if updates%10==0:print(json.dumps(row),flush=True)
                if updates%BLOCK_UPDATES==0:
                    assert frozen_digest(student)==fixed_hash and state_digest(teacher)==teacher_hash
                    atomic_torch(out/f'stage{stage+1}.pt',snapshot(student,teacher,cfg,opt,env,runtime,updates,design))
        assert any_update and saved_probe and updates==180
        assert frozen_digest(student)==fixed_hash and state_digest(teacher)==teacher_hash
        atomic_torch(out/'final.pt',snapshot(student,teacher,cfg,opt,env,runtime,updates,design))
        atomic_json(out/'final_behavior.json',evaluate_behavior(student,teacher,cfg,deadline));atomic_json(out/'final_replay.json',evaluate_replay(student,condition,deadline))
        for key in ('source_sha256','asset_sha256'):
            for name,digest in design[key].items():assert sha(ROOT/name)==digest,name
        result=dict(complete=True,condition=condition,updates=updates,learning_physics_frames=updates*N*UNROLL*4,policy_decisions=updates*N*UNROLL,
            teacher_policy_hash=teacher_hash,student_final_policy_hash=state_digest(student),frozen_tensor_hash=fixed_hash,frozen_tensors_unchanged=True,teacher_unchanged=True,
            seconds=time.monotonic()-started,max_seconds=max_seconds,protocol_sha256=sha(PROTOCOL),no_cloud=True,cost_usd=0,optimization='recurrent-only forward-KL policy distillation',
            artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in out.iterdir()if p.is_file()})
        atomic_json(out/'summary.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        atomic_json(out/'incomplete.json',dict(complete=False,updates=updates,seconds=time.monotonic()-started,error_type=type(error).__name__,error=str(error),no_automatic_extension=True));raise


def freeze():
    assert not PROTOCOL.exists();(BASE/'masks').mkdir(parents=True,exist_ok=True);assets={}
    for condition in ('gc100','pgc33'):
        for suffix in ('.pt','.json'):
            p=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M{suffix}';assets[str(p.relative_to(ROOT))]=sha(p)
        path=ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt';data=torch.load(path,map_location='cpu',weights_only=True)
        summary=path.with_name('summary.json');assert sha(path)==json.loads(summary.read_text())['trace_sha256'];assets[str(path.relative_to(ROOT))]=sha(path);assets[str(summary.relative_to(ROOT))]=sha(summary)
        clock=VisualClock(32,'cpu',REPLAY_SEED);masks=[]
        for hit,done in zip(data['hit'],data['done']):masks.append(clock.mask().clone());clock.advance(hit,done,30)
        bank=BASE/'masks'/f'{condition}.pt';assert not bank.exists();atomic_torch(bank,dict(blind=torch.stack(masks),seed=REPLAY_SEED,trace_sha256=sha(path),evaluation_only=True));assets[str(bank.relative_to(ROOT))]=sha(bank)
    design=dict(version='native_recurrent_policy_distillation_v1',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),conditions=['gc100','pgc33'],model_seed=1,
        train_seed=TRAIN_SEED,eval_seed=EVAL_SEED,replay_seed=REPLAY_SEED,n_envs=N,unroll=UNROLL,updates_per_stage=60,total_updates=180,physics_frames_per_model=180*N*UNROLL*4,
        gaps=list(GAPS),gap_minimum='ceil(cap/2)',visible_updates=15,teacher_action_mixture=list(BETAS),current_grid_probability_per_blackout=.5,
        loss='Mean forward KL teacher||student attemperature1 over all actual visited decisions; originalteacher follows sameactual nativeobservations/actions/rewards.',
        optimizer='Fresh Adam lr1e-4, gradnormclip1; only existing fourLSTMtensors trainable.',state_noise=0,unchanged='Original maps,sixactions,physics,rewards,1350step episodes,CNN,actor,critic,populationcode,nativegoalcache,image/motornoise.',
        variants=list(VARIANTS),evaluation_episodes_per_variant=N,holdout='Baseline/final realstudent rollouts; all32archived originalcensus trajectories replayed with maskssealedbeforetraining; noholdoutoptimization.',
        checkpoints='Initial, all3fixed60update endpoints, final; noadaptiveselection orautomaticextension.',
        post_training_geometry='BOTH fixedfinal models receive the existing direction-memory geometry screen irrespective of behavioralsuccess; compareoriginalinitial andfixedfinal only. Allstage checkpointsretained, never selectedforgeometry. Behavioralerasure andfixed-input replay determine candidateusefulness. Geometryrunsareseparate from theacquisitionwallcap, notadditionaltraining.',
        behavior_clock_scope='Variants share durationdraws andschedulingrule; branchspecifichits renewvisibility, soactualblackout times can diverge. Scheduled exposurecounts retained. Fixedcensusreplay usesexactlytime-matched masks acrossvariants.',process_cap_seconds=1800,no_cloud=True,
        interpretation='New recurrent-policy distillation continuation, not evidence for a preexisting frozen-model ring. Native current/goal information remains; memory requireserasure contrasts againstmatchedvisible scheduled erasure. Different teachers andeffectivePGCdropout precludeclean acrosspopulationcausal attribution.',
        source_sha256={name:sha(ROOT/name)for name in SOURCES},asset_sha256=assets)
    atomic_json(PROTOCOL,design);print(sha(PROTOCOL))


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=('gc100','pgc33'));p.add_argument('--out',type=Path);p.add_argument('--max-seconds',type=int,default=1800);p.add_argument('--freeze',action='store_true');args=p.parse_args();torch.set_num_threads(2)
    if args.freeze:freeze();return
    assert args.condition and args.out and 0<args.max_seconds<=1800;run(args.condition,args.out,args.max_seconds)
if __name__=='__main__':main()
