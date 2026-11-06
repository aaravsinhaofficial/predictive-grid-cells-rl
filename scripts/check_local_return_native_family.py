"""Native-architecture implementation smoke, never a trained-model ring assay."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import argparse
import datetime as dt
import json
import numpy as np
import torch
import local_goal_return_training as task
from local_goal_return_training_run import load_baseline_policy
from lstm_local_return_family import collect_family
from lstm_endogenous_curve_recovery import advance,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert not args.out.exists();torch.set_num_threads(2);rows=[]
    for condition in ('gc100','pgc33'):
        net,cfg,initialization=load_baseline_policy(condition,'cpu');net.eval().requires_grad_(False)
        checkpoint=ROOT/f'results/local_goal_return_baseline/{condition}/initialized_policy.pt';digest=sha(checkpoint)
        plan=task.TrialStreams(781927).draw(1,task.STAGES[3],cfg,headings=96);plan['delay'].fill_(60)
        family=collect_family(net,cfg,plan,chunk_size=48)
        net.double();dark=family['dark_input'].double().reshape(1,-1)
        drive=torch.nn.functional.linear(dark,net.lstm.weight_ih,net.lstm.bias_ih)
        errors={}
        for cue in ('native','ordinary','plus60','minus60'):
            z=family['starts'][cue].double()
            for _ in range(60):z=advance(net.lstm,drive,z)
            error=float((z-family['first_free_states'][cue].double()).abs().max())
            assert error<3e-4,(condition,cue,error);errors[cue]=error
        assert sha(checkpoint)==digest
        rows.append(dict(condition=condition,checkpoint_sha256=digest,full_initialized_policy_hash=initialization['full_initialized_policy_hash'],
            phases=96,state_shape=list(family['pre_cue_state'].shape),dark_shape=list(dark.shape),
            actual_native_architecture=True,all_cue_inputs_and_first_free_timing_valid=True,
            native_float32_vs_double_replay_max_abs_error=errors,
            clean_state_noise_sigma=0.,policy_checkpoint_unchanged=True))
    record=dict(passed=True,rows=rows,private_physical_seed=781927,no_sealed_bank=True,
        source_sha256={str(Path(p).relative_to(ROOT)):sha(p) for p in (__file__,ROOT/'scripts/lstm_local_return_family.py',ROOT/'scripts/local_goal_return_training.py')},
        torch_version=str(torch.__version__),checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        optimizer_updates=0,trained_endpoint_runs=0,geometry_or_decoder_tests=0,
        scope='Shapes, exact permitted-input changes and numerical recurrent timing only, using the stored untrained local-task initializers and one private physical context per population; no ring outcomes.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
