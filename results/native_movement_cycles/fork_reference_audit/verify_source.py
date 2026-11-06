"""Independent reconstruction audit of saved native environment/noise states.

This uses recorded actions, not new policy sampling or interventions. It checks
that the saved clone plus original 32-slot random draw shapes reproduces the
next transition, image, and original LSTM state.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from native_movement_cycles import TraceArena,ENV_KEYS,PHYSICAL_KEYS
from code.common import build_policy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    records=[]
    for condition in ('gc100','pgc33'):
        for seed in (1,2):
            folder=ROOT/f'results/native_movement_cycles/{condition}_s{seed}'
            trace_path=folder/'native_trace.pt'
            data=torch.load(trace_path,map_location='cpu',weights_only=True)
            cfg=data['config']
            protocol=json.loads((folder/'protocol.json').read_text())
            net=build_policy(cfg,'cpu').eval().requires_grad_(False)
            ckpt=ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt'
            net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['policy'])
            record=dict(condition=condition,seed=seed,trace_sha256=sha(trace_path),checkpoint_sha256=sha(ckpt),windows=[])
            with torch.inference_mode():
                for anchor in (100,500):
                    env=TraceArena(cfg,32,1,protocol['eval_seed'],'cpu')
                    for key in ENV_KEYS:setattr(env,key,data['pre'][key][anchor].clone())
                    for key,value in data['world'].items():setattr(env,key,value.clone())
                    env.gen.set_state(data['environment_rng'][anchor].clone())
                    max_physical=0.;max_embedding=0.;max_neural=0.;max_logits=0.;rng_exact=True;hit_exact=True
                    for t in range(anchor,anchor+3):
                        incoming=data['state_before'][t]
                        state=net.lstm(data['x'][t],(incoming[:,:256],incoming[:,256:]))
                        max_neural=max(max_neural,float((torch.cat(state,1)-data['state_after'][t]).abs().max()))
                        max_logits=max(max_logits,float((net.pi(state[0])-data['logits'][t]).abs().max()))
                        env.trace_first=None
                        obs,reward,done,info=env.step(data['action'][t])
                        for key in PHYSICAL_KEYS:
                            lhs=env.trace_first[key];rhs=data['post'][key][t]
                            max_physical=max(max_physical,float((lhs.double()-rhs.double()).abs().max()))
                        hit_exact=hit_exact and torch.equal(info['hit'],data['hit'][t]) and torch.equal(done,data['done'][t])
                        rng_exact=rng_exact and torch.equal(env.gen.get_state(),data['environment_rng'][t+1])
                        max_embedding=max(max_embedding,float((net.embed(obs['rgb'])-data['x'][t+1,:,:256]).abs().max()))
                    record['windows'].append(dict(anchor=anchor,steps=3,physical_max_error=max_physical,
                        embedding_max_error=max_embedding,neural_max_error=max_neural,logit_max_error=max_logits,
                        rng_exact=rng_exact,hit_done_exact=hit_exact))
                    assert max_physical==0 and max_embedding==0 and max_neural==0 and max_logits==0
                    assert rng_exact and hit_exact
            records.append(record)
            print(json.dumps(record),flush=True)
    result=dict(source_sha256=sha(__file__),cases=records,all_exact=True,
                interpretation='Independent saved native32-slot clone/noise reconstruction only; not a test of the new causal fork implementation.')
    Path(__file__).with_name('summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
