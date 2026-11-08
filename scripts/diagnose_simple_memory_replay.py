"""Bounded reconstruction of one failed replay check, never a continuation run.

The frozen trainer is imported unchanged. Only its 188 already-recorded updates
are reconstructed. At the failed boundary we collect once, then compare neural
replays without backward/optimizer steps. CPU APIs exist solely for private tests;
real checkpoint execution requires the exact original CUDA T4 software stack.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from code import train_memory as train
import numpy as np
import torch

RESUME_SHA='ebda7dfbabf5a268711d167321548af1e1aca0dc1c622f86bd3a073651a6a83c'
START_UPDATE,LAST_SUCCESS,FAILED_UPDATE=27000,27188,27189
PROTOCOL=ROOT/'docs/research/simple_memory_training_protocol.json'
SOURCE_NAMES=('scripts/diagnose_simple_memory_replay.py','tests/test_diagnose_simple_memory_replay.py',*train.SOURCES)
INPUT_NAMES=('resume.pt','config.json','provenance.json','metrics_000.jsonl','attempt_000.json')
RTOL,ATOL=1e-4,1e-5
TECHNICAL=('frames','updates','rollout_steps','loss','policy_loss','value_loss','entropy','grad_norm',
           'replay_logit_max_error','replay_value_max_error','max_gap')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def require(value,message):
    if not value:raise RuntimeError(message)
def check_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Diagnostic deadline; no continuation authorized')
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):train.atomic_json(Path(path),obj)


def tree_hash(obj):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):v=v.detach().cpu().numpy()
        if isinstance(v,np.ndarray):
            h.update(b'array');h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode())
            for item in v:add(item)
        else:h.update((type(v).__name__+':'+repr(v)).encode())
    add(obj);return h.hexdigest()


def array_schema(obj):
    result={}
    def visit(v,path):
        if isinstance(v,torch.Tensor):result[path]=dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tree_hash(v))
        elif isinstance(v,dict):
            for k,x in v.items():visit(x,path+'/'+str(k))
        elif isinstance(v,(tuple,list)):
            for i,x in enumerate(v):visit(x,path+'/'+str(i))
    visit(obj,'');return result


def save_raw(path,obj):
    train.atomic_torch(path,obj)
    return dict(path=Path(path).name,bytes=Path(path).stat().st_size,sha256=sha(path),tensor_schema=array_schema(obj))


def make_manifest(run):
    run=Path(run)
    require(sha(run/'resume.pt')==RESUME_SHA,'Wrong diagnostic resume input')
    return dict(schema=1,status='draft',name='Single failed simple-memory replay diagnostic',
        input_sha256={name:sha(run/name) for name in INPUT_NAMES},
        source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},training_protocol_sha256=sha(PROTOCOL),
        start_update=START_UPDATE,last_successful_update=LAST_SUCCESS,failed_update=FAILED_UPDATE,
        maximum_seconds=900,original_rtol=RTOL,original_atol=ATOL,
        scope='Reconstruct only already-logged updates27001..27188, then collect failed27189 once. No failing optimizer step, continuation, tolerance change or behavioral result.',
        paths=['A_original_chunk_grad','B_same_batch_no_grad','C_same_batch_grad',
               'B32_encoder_grad_only','B32_recurrent_grad_only','D_actor_embedding_no_grad','D_actor_embedding_grad'])


def authenticate_inputs(run,manifest_path):
    run=Path(run);manifest=read(manifest_path)
    require(manifest['status']=='frozen' and manifest['schema']==1,'Diagnostic manifest must be frozen')
    require(set(manifest['source_sha256'])==set(SOURCE_NAMES),'Diagnostic dependency closure changed')
    for name,digest in manifest['source_sha256'].items():require(sha(ROOT/name)==digest,'Changed source: '+name)
    require(set(manifest['input_sha256'])==set(INPUT_NAMES),'Diagnostic input closure changed')
    for name,digest in manifest['input_sha256'].items():require(sha(run/name)==digest,'Changed input: '+name)
    require(sha(run/'resume.pt')==RESUME_SHA and manifest['training_protocol_sha256']==sha(PROTOCOL),'Resume/protocol identity changed')
    require((manifest['start_update'],manifest['last_successful_update'],manifest['failed_update'])==
            (START_UPDATE,LAST_SUCCESS,FAILED_UPDATE),'Diagnostic boundary changed')
    require(manifest['maximum_seconds']==900 and manifest['original_rtol']==RTOL and manifest['original_atol']==ATOL,'Diagnostic tolerance/budget changed')
    require(manifest['paths']==[row[0] for row in PATHS],'Diagnostic comparison inventory changed')
    saved=torch.load(run/'resume.pt',map_location='cpu',weights_only=False)
    cfg=read(run/'config.json');proof=read(run/'provenance.json');protocol=read(PROTOCOL)
    scientific={k:v for k,v in cfg.items() if k!='population'}
    require(saved['schema']==1 and saved['config']==scientific and saved['provenance']==proof,'Snapshot config/provenance mismatch')
    require(saved['frames']==saved['runtime']['counters']['frames']==345_600_000 and
            saved['updates']==saved['runtime']['counters']['updates']==START_UPDATE,'Wrong saved boundary')
    require(saved['runtime']['device_type']=='cuda','Wrong checkpoint backend')
    require((cfg['cond'],cfg['seed'],cfg['task_max_gap'])==('pgc33',1,60),'Wrong logical cohort endpoint')
    require(proof['source_sha256']==protocol['source_sha256'] and proof['protocol_sha256']==sha(PROTOCOL),'Original source/protocol changed')
    for k,v in protocol['training_config'].items():require(cfg[k]==v,'Changed training setting: '+k)
    require(proof['config_sha256']==hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest(),'Config digest mismatch')
    h=hashlib.sha256()
    for k,v in sorted(saved['policy'].items()):h.update(k.encode());h.update(v.numpy().tobytes())
    require(h.hexdigest()==saved['policy_sha256'],'Policy digest mismatch')
    require(all(float(v['step'])==START_UPDATE for v in saved['optimizer']['state'].values()),'Optimizer boundary changed')
    rows=[json.loads(line) for line in (run/'metrics_000.jsonl').read_text().splitlines() if line.strip()]
    require(rows[-1]['updates']==LAST_SUCCESS and rows[-1]['frames']==348_006_400,'Original final accepted row changed')
    selected=[r for r in rows if START_UPDATE<r['updates']<=LAST_SUCCESS]
    require([r['updates'] for r in selected]==list(range(START_UPDATE+1,LAST_SUCCESS+1)),'Missing/duplicate reconstruction rows')
    return saved,scientific,proof,selected,manifest


def authenticate_cuda(proof):
    require(torch.cuda.is_available(),'CUDA required; real checkpoint must never train locally on another backend')
    actual=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                cuda_device_name=torch.cuda.get_device_name(0),cuda_capability=list(torch.cuda.get_device_capability(0)))
    require(actual['cuda_device_name']=='Tesla T4' and all(actual[k]==proof[k] for k in actual),'Original T4/Torch/CUDA/cuDNN stack required')
    require(proof['deterministic_algorithms'] and not proof['cudnn_tf32'] and not proof['matmul_tf32'] and not proof['cudnn_benchmark'],'Original FP32 determinism settings changed')
    return actual


class Trace:
    """Hooks only copy/detach actual inputs/outputs; no RNG or output mutation."""
    def __init__(self,policy,deadline=float('inf')):
        self.policy=policy;self.deadline=deadline;self.enc=[];self.rnn=[];self.logits=[];self.values=[]
    def __enter__(self):
        def encoder(module,args,out):
            check_time(self.deadline)
            self.enc.append(dict(rgb_sha256=tree_hash(args[0]),batch=len(args[0]),embedding=out.detach().cpu().clone()))
        def pre(module,args):
            check_time(self.deadline)
            x,state=args
            self.rnn.append(dict(x=x.detach().cpu().clone(),h_before=state[0].detach().cpu().clone(),c_before=state[1].detach().cpu().clone()))
        def post(module,args,out):
            self.rnn[-1].update(h_after=out[0].detach().cpu().clone(),c_after=out[1].detach().cpu().clone())
        self.hooks=[self.policy.encoder.register_forward_hook(encoder),self.policy.lstm.register_forward_pre_hook(pre),
          self.policy.lstm.register_forward_hook(post),self.policy.pi.register_forward_hook(lambda m,a,o:self.logits.append(o.detach().cpu().clone())),
          self.policy.v.register_forward_hook(lambda m,a,o:self.values.append(o.detach().cpu().clone().squeeze(-1)))]
        return self
    def __exit__(self,*args):
        for hook in self.hooks:hook.remove()
    def arrays(self):
        return dict(**{k:torch.stack([r[k] for r in self.rnn]) for k in ('x','h_before','c_before','h_after','c_after')},
                    logits=torch.stack(self.logits),values=torch.stack(self.values),
                    encoder_calls=[dict(batch=r['batch'],rgb_sha256=r['rgb_sha256']) for r in self.enc],
                    embeddings=torch.cat([r['embedding'] for r in self.enc]) if self.enc else None)


def concat_cpu(embedding,code,goal,pa,pr):
    onehot=torch.zeros((len(pa),6),dtype=embedding.dtype)
    onehot.scatter_(1,pa.clamp_min(0).long().unsqueeze(1),1.)
    onehot*= (pa>=0).float().unsqueeze(1)
    return torch.cat((embedding,code,goal,onehot,pr.unsqueeze(1)),dim=-1)


def validate_actor_trace(roll,state0,actor,bootstrap_inputs=None):
    length=len(roll['rgb']);n=len(roll['pa'][0])
    require(len(actor['encoder_calls'])==len(actor['logits'])==length+1,'Actor/bootstrap call count mismatch')
    require(all(row['batch']==n for row in actor['encoder_calls']),'Actor batch shape changed')
    embeddings=actor['embeddings'].view(length+1,n,-1)
    incoming=tuple(x.detach().cpu().clone() for x in state0)
    for t in range(length+1):
        fields=bootstrap_inputs if t==length else {k:roll[k][t] for k in ('rgb','code','goal','pa','pr')}
        if fields is None:break
        fields=train.cpu_tree(fields)
        require(actor['encoder_calls'][t]['rgb_sha256']==tree_hash(fields['rgb']),f'Actual encoder RGB changed at{t}')
        expected=concat_cpu(embeddings[t],fields['code'],fields['goal'],fields['pa'],fields['pr'])
        require(torch.equal(expected,actor['x'][t]),f'Actual recurrent input changed at{t}')
        require(torch.equal(incoming[0],actor['h_before'][t]) and torch.equal(incoming[1],actor['c_before'][t]),f'Incoming state/reset mismatch at{t}')
        if t<length:
            require(torch.equal(actor['logits'][t],roll['actor_logits'][t].detach().cpu()) and
                    torch.equal(actor['values'][t],roll['actor_values'][t].detach().cpu()),f'Actor output capture mismatch at{t}')
            keep=1-roll['d'][t].detach().cpu()
            incoming=(actor['h_after'][t]*keep[:,None],actor['c_after'][t]*keep[:,None])
    return dict(qualified=True,rollout_calls=length,bootstrap_calls=1,batch=n,rgb_hashes_exact=True,
                concatenated_inputs_exact=True,incoming_and_done_resets_exact=True,actor_heads_exact=True)


def capture_rollout(policy,source,runtime,cfg,length,deadline=float('inf')):
    with Trace(policy,deadline) as trace:
        roll,state0,boot,bootstrap_inputs,episodes=train.collect_rollout(policy,source,runtime,cfg,length)
    actor=trace.arrays()
    validation=validate_actor_trace(roll,state0,actor,bootstrap_inputs)
    # Episodes/rewards are preserved in the raw technical artifact but no
    # behavioral score/success summaries are computed or selected.
    return roll,state0,boot,bootstrap_inputs,episodes,actor,validation


def original_assert(actual,expected):
    try:
        torch.testing.assert_close(actual.detach(),expected.detach(),rtol=RTOL,atol=ATOL)
        return dict(passed=True)
    except AssertionError as exc:return dict(passed=False,error=str(exc))


def compare_array(actual,expected):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    require(a.shape==b.shape,'Compared shape mismatch')
    finite=np.isfinite(a)&np.isfinite(b);diff=np.abs(a-b);allowed=ATOL+RTOL*np.abs(b)
    changed=(a!=b)|~finite;bad=(diff>allowed)|~finite
    first=lambda mask: [int(x) for x in np.argwhere(mask)[0]] if mask.any() else None
    max_index=list(np.unravel_index(np.nanargmax(diff),diff.shape)) if np.isfinite(diff).any() else None
    return dict(shape=list(a.shape),bitwise_equal=bool(np.array_equal(a,b)),nonfinite_count=int((~finite).sum()),
                changed_elements=int(changed.sum()),first_changed=first(changed),original_tolerance_violations=int(bad.sum()),
                first_tolerance_violation=first(bad),maximum_absolute_error=float(np.max(diff[finite])) if finite.any() else None,
                maximum_error_index=[int(x) for x in max_index] if max_index is not None else None,
                tolerance=dict(rtol=RTOL,atol=ATOL,reference='captured online actor'))


def replay_path(policy,roll,state0,actor,*,kind,encoder_grad=True,recurrent_grad=True,deadline=float('inf')):
    """One forward-only graph at a time; no backward or optimizer invocation."""
    length,n=len(roll['rgb']),len(roll['pa'][0]);device=state0[0].device
    with Trace(policy,deadline) as trace:
        if kind=='original':
            with torch.enable_grad():logits,values,final=train.replay(policy,roll,state0)
        else:
            state=state0;lls=[];vvs=[]
            for t in range(length):
                check_time(deadline)
                if kind=='captured':e=actor['embeddings'].view(length+1,n,-1)[t].to(device)
                else:
                    with torch.set_grad_enabled(encoder_grad):e=policy.embed(roll['rgb'][t])
                with torch.set_grad_enabled(recurrent_grad):
                    ll,vv,state=policy.step_from_embedding(e,roll['code'][t],roll['goal'][t],roll['pa'][t],roll['pr'][t],state)
                    lls.append(ll);vvs.append(vv)
                    state=tuple(v*(1-roll['d'][t])[:,None] for v in state)
            logits,values,final=torch.stack(lls),torch.stack(vvs),state
    observed=trace.arrays()
    if kind=='captured':observed['embeddings']=actor['embeddings'][:length*n].clone()
    observed['embeddings']=observed['embeddings'].view(length,n,-1)
    expected_embeddings=actor['embeddings'].view(length+1,n,-1)[:length]
    if kind=='original':
        concatenated=torch.cat(roll['rgb'],0)
        cursor=0
        for call in observed['encoder_calls']:
            require(call['rgb_sha256']==tree_hash(concatenated[cursor:cursor+call['batch']]),'Original chunk RGB mismatch')
            cursor+=call['batch']
        require(cursor==length*n,'Original encoder coverage mismatch')
    elif kind!='captured':
        require(len(observed['encoder_calls'])==length,'B32 encoder call count')
        for t,call in enumerate(observed['encoder_calls']):
            require(call['batch']==n and call['rgb_sha256']==tree_hash(roll['rgb'][t]),'B32 actual RGB mismatch')
    incoming=tuple(x.detach().cpu() for x in state0)
    for t in range(length):
        expected=concat_cpu(observed['embeddings'][t],*(roll[k][t].detach().cpu() for k in ('code','goal','pa','pr')))
        require(torch.equal(observed['x'][t],expected),'Replay nonvisual input/concat mismatch')
        require(torch.equal(incoming[0],observed['h_before'][t]) and torch.equal(incoming[1],observed['c_before'][t]),'Replay reset rule mismatch')
        incoming=tuple(observed[k][t]*(1-roll['d'][t].detach().cpu())[:,None] for k in ('h_after','c_after'))
    report=dict(kind=kind,encoder_grad=encoder_grad,recurrent_grad=recurrent_grad,
                logits_requires_grad=logits.requires_grad,encoder_batch_sizes=[r['batch'] for r in observed['encoder_calls']],
                inputs_and_reset_rule_verified=True,
                original_logit_assert=original_assert(logits,torch.stack(roll['actor_logits'])),
                original_value_assert=original_assert(values,torch.stack(roll['actor_values'])))
    report['differences']={key:compare_array(observed[key].numpy(),expected_embeddings.numpy() if key=='embeddings' else actor[key][:length].numpy())
        for key in ('embeddings','x','h_before','c_before','h_after','c_after','logits','values')}
    # Returned arrays contain no autograd references. Graph locals die here.
    return report,observed


PATHS=(('A_original_chunk_grad','original',True,True),('B_same_batch_no_grad','same',False,False),
       ('C_same_batch_grad','same',True,True),('B32_encoder_grad_only','same',True,False),
       ('B32_recurrent_grad_only','same',False,True),('D_actor_embedding_no_grad','captured',False,False),
       ('D_actor_embedding_grad','captured',False,True))


def execute(run,out,manifest_path,max_seconds=900):
    run,out=Path(run),Path(out)
    require(0<max_seconds<=900,'Diagnostic cap must be <=900s')
    require(not out.exists() or not any(out.iterdir()),'Diagnostic output must be new; preserve prior evidence')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();deadline=started+max_seconds
    progress=dict(complete=False,started_utc=utc(),max_seconds=max_seconds,paths={},
                  producer_sha256=sha(__file__),requested_manifest_sha256=sha(manifest_path) if Path(manifest_path).is_file() else None,input_authentication_complete=False,
                  scope='Numerical diagnostic only; no failing update or subsequent training authorized.')
    write(out/'summary.json',progress)
    try:
        saved,cfg,proof,expected,manifest=authenticate_inputs(run,manifest_path)
        progress.update(manifest_sha256=sha(manifest_path),source_sha256=manifest['source_sha256'],input_sha256=manifest['input_sha256'],
                        original_resume_frames=saved['frames'],original_resume_updates=saved['updates'],input_authentication_complete=True)
        write(out/'summary.json',progress)
        progress['software']=authenticate_cuda(proof)
        cfg=dict(cfg,device='cuda:0',threads=2,protocol=str(PROTOCOL),out=str(out),resume=True,max_seconds=max_seconds)
        policy,opt,source,runtime=train.fresh_training(cfg)
        actual_proof=train.provenance(cfg);require(actual_proof==proof,'Frozen restore provenance mismatch')
        train.restore(saved,policy,opt,runtime,cfg,actual_proof)
        require(tree_hash(train.snapshot(policy,opt,runtime,cfg,proof))==tree_hash(saved),'Exact restored full state differs')
        progress['restore_exact']=True;write(out/'summary.json',progress)
        with (out/'reconstruction.jsonl').open('x') as log:
            for target in expected:
                check_time(deadline)
                # NO hooks, wrapping or substitutions in reconstructed updates.
                actual,_=train.train_update(policy,opt,source,runtime,cfg)
                comparisons={k:actual[k]==target[k] for k in TECHNICAL}
                row=dict(update=target['updates'],technical={k:actual[k] for k in TECHNICAL},
                         matches_original=comparisons,exact=all(comparisons.values()))
                log.write(json.dumps(row)+'\n');log.flush()
                progress['last_reconstructed_update']=actual['updates']
                require(row['exact'],f'Earlier reconstruction differs at update{target["updates"]}; stop attribution')
                if actual['updates']%20==0:write(out/'summary.json',progress)
        require(runtime['counters']['updates']==LAST_SUCCESS,'Reconstruction endpoint mismatch')
        boundary=train.snapshot(policy,opt,runtime,cfg,proof)
        progress['boundary']=save_raw(out/'reconstructed_boundary.pt',dict(schema='diagnostic_boundary_v1',
            original_training_snapshot=boundary,diagnostic_manifest_sha256=sha(manifest_path),parent_resume_sha256=RESUME_SHA,
            scope='Reconstructed original accepted boundary; not a final endpoint or independently trained model.'))
        before_policy=train.policy_hash(policy);before_optimizer=tree_hash(opt.state_dict())
        before_gradients=tree_hash([p.grad for p in policy.parameters()])
        check_time(deadline)
        roll,state0,boot,bi,episodes,actor,validation=capture_rollout(policy,source,runtime,cfg,cfg['rollout'],deadline)
        progress['actor_capture_validation']=validation
        progress['failing_rollout']=save_raw(out/'failing_rollout.pt',train.cpu_tree(dict(roll=roll,state0=state0,
            bootstrap=boot,bootstrap_inputs=bi,episodes=episodes,actor=actor)))
        runtime_after=tree_hash(train.capture_runtime(runtime));write(out/'summary.json',progress)
        for name,kind,encoder_grad,recurrent_grad in PATHS:
            check_time(deadline)
            report,arrays=replay_path(policy,roll,state0,actor,kind=kind,encoder_grad=encoder_grad,
                                      recurrent_grad=recurrent_grad,deadline=deadline)
            require(tree_hash(train.capture_runtime(runtime))==runtime_after,'Diagnostic replay changed runtime/RNG')
            require(train.policy_hash(policy)==before_policy and tree_hash(opt.state_dict())==before_optimizer,'Diagnostic replay updated parameters/optimizer')
            require(tree_hash([p.grad for p in policy.parameters()])==before_gradients,'Diagnostic replay changed accumulated gradients')
            report['arrays']=save_raw(out/(name+'.pt'),arrays)
            progress['paths'][name]=report;write(out/'summary.json',progress)
            del arrays;gc.collect()
        progress.update(complete=True,failed_update_optimizer_executed=False,accepted_optimizer_updates_reconstructed=188,
            boundary_frames=boundary['frames'],boundary_updates=boundary['updates'],
            captured_rollout_frames=runtime['counters']['frames'],
            original_logit_check_failed_on_reconstruction=not progress['paths']['A_original_chunk_grad']['original_logit_assert']['passed'],
            parameters_and_optimizer_unchanged_after_boundary=True,runtime_and_rng_unchanged_during_replays=True,
            accumulated_gradients_unchanged=True,
            interpretation='The failed-check Boolean is not a claim of identical failure signature. Compare retained counts, locations and errors with the original failure. If A passes, attribution remains unresolved. No automatic repair, tolerance relaxation or continuation follows these results.')
    except BaseException as exc:
        progress['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        progress.update(finished_utc=utc(),seconds=time.monotonic()-started)
        if progress.get('input_authentication_complete'):
            try:
                sources_match=all(sha(ROOT/name)==digest for name,digest in progress['source_sha256'].items())
                inputs_match=all(sha(run/name)==digest for name,digest in progress['input_sha256'].items())
                progress['source_and_input_closure_unchanged']=sources_match and inputs_match
            except Exception as exc:
                progress['source_and_input_closure_unchanged']=False
                progress['closure_check_error']=f'{type(exc).__name__}: {exc}'
            if not progress['source_and_input_closure_unchanged']:
                progress['complete']=False
                progress['closure_error']='Authenticated source/input changed during diagnostic'
        progress['retained_artifacts']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='summary.json'}
        write(out/'summary.json',progress)
    return progress


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--manifest',type=Path);p.add_argument('--write-draft',type=Path)
    p.add_argument('--max-seconds',type=float,default=900)
    args=p.parse_args(argv)
    if args.write_draft:
        require(not args.write_draft.exists(),'Preserve existing draft');args.write_draft.parent.mkdir(parents=True,exist_ok=True)
        write(args.write_draft,make_manifest(args.run));return 0
    require(args.manifest is not None and args.out is not None,'Execution requires --manifest and --out')
    require(0<args.max_seconds<=900,'Diagnostic cap must be <=900s')
    def timeout(signum,frame):raise TimeoutError('Diagnostic time/signal limit; partial evidence retained')
    signal.signal(signal.SIGTERM,timeout);signal.signal(signal.SIGINT,timeout);signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    try:result=execute(args.run,args.out,args.manifest,args.max_seconds)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(complete=result['complete'],seconds=result['seconds'],error=result.get('error'))),flush=True)
    return 0 if result['complete'] else 1


if __name__=='__main__':sys.exit(main())
