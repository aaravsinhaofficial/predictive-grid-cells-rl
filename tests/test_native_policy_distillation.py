import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
import torch
from native_policy_distillation import (ROOT,N,kl,student_inputs,masks_for_variants,trainable_student,
    train_unroll,feature_inputs,frozen_digest,state_digest,replay_logits)
from code.common import build_policy,build_population
from lstm_code_swap import MatchedArena
from rl_visual_memory import VisualClock,NativeGoal


def test_teacher_kl_identity_and_target_detach():
    teacher=torch.tensor([[2.,-1.,.4,0.,-2.,1.]],requires_grad=True)
    student=teacher.detach().clone().requires_grad_(True)
    identity=kl(teacher,student);assert float(identity.detach())==0
    identity.sum().backward();assert bool((student.grad==0).all())and teacher.grad is None
    student.grad=None
    altered=student+torch.tensor([[0.,0.,1.,0.,0.,0.]])
    loss=kl(teacher,altered).mean();loss.backward()
    assert teacher.grad is None and float(student.grad.abs().max())>0 and float(loss.detach())>0


def test_input_masks_preserve_native_goal_history_and_grid_sham():
    gen=torch.Generator().manual_seed(9911);x=torch.randn(6,1287,generator=gen);ordinary=x[:,256:768].clone();black=torch.zeros(6,256)
    blind=torch.tensor([0,1,1,1,1,0],dtype=torch.bool);swap=torch.tensor([0,0,1,0,1,0],dtype=torch.bool)
    actual=student_inputs(x,ordinary,black,blind,swap)
    assert torch.equal(actual[:,768:],x[:,768:])
    assert torch.equal(actual[~blind],x[~blind])and bool((actual[blind,:256]==0).all())
    assert torch.equal(actual,student_inputs(x,ordinary,black,blind,torch.zeros_like(swap)))
    b,s,r=masks_for_variants(torch.ones(6,dtype=torch.bool),1)
    assert b.tolist()==[False,True,True,True,True,False]
    assert s.tolist()==[False,False,True,False,True,False]
    assert r.tolist()==[False,False,False,True,True,True]


@pytest.mark.parametrize('condition',['gc100','pgc33'])
def test_native_shape_shared_history_real_gradient_frozen_weights_and_replay(condition):
    torch.set_num_threads(2);torch.manual_seed(9931)
    cfg=json.loads((ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    # Fresh random network only; no original trained weights or held-out trajectories.
    teacher=build_policy(cfg,'cpu').eval().requires_grad_(False);student=trainable_student(teacher);pop=build_population(cfg,'cpu')
    class RecordedArena(MatchedArena):
        def step(self,a):
            result=super().step(a);self.records.append((a.clone(),result[1].clone(),result[2].clone()));return result
    env=RecordedArena(cfg,N,1,993101,'cpu');env.records=[];obs=env.reset_all();goal=NativeGoal(N,512,'cpu');pa=torch.full((N,),-1,dtype=torch.long);pr=torch.zeros(N);zero=student.zero_state(N,'cpu')
    native,_=feature_inputs(teacher,pop,obs,goal.saved,pa,pr,N)
    direct=teacher.step(obs['rgb'],pop.activity(obs['pos'],obs['dir'],obs['vel']),goal.saved,pa,pr,zero)
    exact=teacher.lstm(native,zero)
    assert torch.equal(direct[0],teacher.pi(exact[0]))and all(torch.equal(a,b)for a,b in zip(direct[2],exact))
    assert float(kl(direct[0],student.pi(student.lstm(native,zero)[0])).detach().max())==0
    clock=VisualClock(N,'cpu',993103);clock.known.fill_(True);clock.blind.fill_(True);clock.remaining.fill_(30)
    runtime=dict(obs=obs,student_state=zero,teacher_state=teacher.zero_state(N,'cpu'),goal=goal,clock=clock,pa=pa,pr=pr,swap_interval=torch.ones(N,dtype=torch.bool),deadline=time.monotonic()+60,
        black_embedding=teacher.embed(torch.zeros(N,3,84,84)).detach(),action_rng=torch.Generator().manual_seed(993102),mixture_rng=torch.Generator().manual_seed(993104),substitution_rng=torch.Generator().manual_seed(993105))
    fixed=frozen_digest(student);teacher_hash=state_digest(teacher);opt=torch.optim.Adam(student.lstm.parameters(),lr=1e-4)
    metrics,probe=train_unroll(student,teacher,pop,cfg,env,runtime,opt,30,.5,length=3)
    assert metrics['loss']>0 and metrics['grad_norm_before_clip']>0 and metrics['parameter_update_l2']>0
    assert metrics['replay_max_error']==0 and frozen_digest(student)==fixed and state_digest(teacher)==teacher_hash
    assert all(p.grad is None for name,p in student.named_parameters()if not name.startswith('lstm.'))
    for t in (1,2):
        action,reward,done=env.records[t-1];expected=torch.nn.functional.one_hot(action,6).float()*(~done)[:,None]
        assert torch.equal(probe['student_inputs'][t,:,1280:1286],expected)
        assert torch.equal(probe['student_inputs'][t,:,1286],reward.clamp(-1,1)*(~done))


def test_bptt_reaches_earlier_inputs_and_terminal_mask_cuts_it():
    class Small(torch.nn.Module):
        def __init__(self):super().__init__();self.lstm=torch.nn.LSTMCell(3,4);self.pi=torch.nn.Linear(4,2)
    torch.manual_seed(993201);net=Small();inputs=torch.randn(3,2,3,requires_grad=True);zero=(torch.zeros(2,4),torch.zeros(2,4));done=torch.zeros(3,2,dtype=torch.bool)
    logits,_=replay_logits(net,inputs,zero,done);gradient=torch.autograd.grad(logits[-1].square().sum(),inputs)[0];assert float(gradient[0].abs().max())>0
    done[0]=True;logits,_=replay_logits(net,inputs,zero,done);gradient=torch.autograd.grad(logits[-1].square().sum(),inputs)[0];assert bool((gradient[0]==0).all())
