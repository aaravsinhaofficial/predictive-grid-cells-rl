"""Random native-shape unit validation only: no pretrained policy or study bank."""
from pathlib import Path
import copy,datetime as dt,hashlib,io,json,sys,time
ROOT=Path('/Users/aaravsinha/predictive-grid-cells-rl');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import goal_cue_memory as task
import goal_cue_memory_run as run
from code.common import build_policy
from rl_visual_memory_extend import tree_equal

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={name:sha(ROOT/name)for name in ['scripts/goal_cue_memory.py','scripts/goal_cue_memory_run.py','code/nets.py','code/common.py','code/env.py','code/codes.py']}
torch.set_num_threads(2);started=time.monotonic();rows=[]
for condition in ('gc100','pgc33'):
 cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text());cfg.update(n_envs=2,rollout=160,img_noise=0.)
 def create():
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(8911);net=build_policy(cfg,'cpu')
  run.initializer.configure_trainable(net)
  return net,run.initializer.optimizer(net,cfg),task.TrialStreams(8912),task.make_source(cfg,'cpu')
 net,opt,streams,source=create();counters=run.fresh_counters();counters['evaluation_records']['baseline']={'synthetic_only':True}
 initial={'synthetic_random_native_architecture':True};provenance={'synthetic_only':True}
 def update(n,o,s,c,src):
  plan=s.draw_training(2,cfg,delay_cap=0);metrics,trialrows=run.update(n,cfg,o,src,plan)
  run.advance_counters(c,metrics,cfg);return plan,metrics,trialrows
 first=update(net,opt,streams,counters,source)
 snapshot=run.snapshot(net,opt,streams,cfg,initial,counters,provenance)
 buffer=io.BytesIO();torch.save(snapshot,buffer);buffer.seek(0);snapshot=torch.load(buffer,map_location='cpu',weights_only=True)
 expected=update(net,opt,streams,counters,source)
 expected_final=run.snapshot(net,opt,streams,cfg,initial,counters,provenance)
 other,other_opt,other_streams,other_source=create()
 other_counters=run.restore(other,other_opt,other_streams,snapshot,cfg,provenance)
 actual=update(other,other_opt,other_streams,other_counters,other_source)
 actual_final=run.snapshot(other,other_opt,other_streams,cfg,initial,other_counters,provenance)
 assert tree_equal(expected,actual)and tree_equal(expected_final,actual_final)
 assert first[1]['replay_error']==first[1]['value_replay_error']==expected[1]['replay_error']==expected[1]['value_replay_error']==0
 assert all(torch.isfinite(p.grad).all() for p in other.parameters() if p.grad is not None)
 assert len(other_opt.state)==len(list(other.parameters()))
 rows.append(dict(condition=condition,network_parameters=sum(p.numel()for p in net.parameters()),hidden_units=cfg['nh'],
   lstm_input_size=net.lstm.input_size,complete_synthetic_trial_updates=2,trials_per_update=2,
   exact_policy_optimizer_rng_counters_plans_metrics_and_rows_after_resume=True,logit_replay_error=0.,value_replay_error=0.,
   every_parameter_has_optimizer_state=True,all_gradients_finite=True))
 print(json.dumps(rows[-1]),flush=True)
for name,digest in pins.items():assert sha(ROOT/name)==digest
out=ROOT/'results/goal_cue_memory_design/native_shape_resume_validation.json';assert not out.exists()
source=out.parent/'validation_sources/native_shape_resume_smoke.py';source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(Path(__file__).read_bytes())
record=dict(version='goal_cue_memory_random_native_shape_resume_v1',passed=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),
 scope='Randomly initialized original CNN/LayerNorm/LSTM256 architecture; private synthetic two-trial plans. No pretrained policy tensors, actual study bank or study outcome was loaded or generated. Optimizer steps are unit checks only.',
 private_unit_model_seed=8911,private_unit_plan_seed=8912,torch_version=str(torch.__version__),device='cpu',seconds=time.monotonic()-started,
 source_sha256=pins,validation_source=dict(path=str(source.relative_to(ROOT)),sha256=sha(source)),models=rows)
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(path=str(out.relative_to(ROOT)),sha256=sha(out),passed=True,seconds=record['seconds'])),flush=True)
