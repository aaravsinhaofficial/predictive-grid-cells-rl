"""Evaluate every qualified equilibrium in the original float32 recurrence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import lstm_goal_bearing_manifold as base
from lstm_endogenous_curve_recovery import sha,save


def main():
    torch.set_num_threads(2);out=ROOT/'results/lstm_endogenous_ring'
    rows=[]
    with torch.no_grad(),threadpool_limits(limits=2):
        for path in sorted((out/'equilibrium_elimination').glob('*/*.json')):
            d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
            ok=(np.array(d['hidden_residual_RMS'])<=1e-8)&(np.array(d['stable_full_residual_RMS'])<=1e-8)
            if not ok.any():continue
            _,checkpoint,net,_=base.load_model(d['condition'],d['seed'],'cpu')
            assert d['checkpoint_sha256']==sha(checkpoint)
            z=torch.tensor(a['solution'][ok],dtype=torch.float32);x=torch.tensor(a['input'],dtype=torch.float32).expand(len(z),-1)
            h,c=z[:,:256],z[:,256:]
            hn,cn=net.lstm(x,(h,c));native=torch.cat((hn,cn),1)
            gates=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)+torch.nn.functional.linear(h,net.lstm.weight_hh,net.lstm.bias_hh)
            gi,gf,gg,go=gates.chunk(4,1)
            i,f,g=gi.sigmoid(),gf.sigmoid(),gg.tanh()
            # Float32 gate values, with subsequent c arithmetic evaluated in
            # float64, isolate write/forget drift hidden by a large float32 ULP.
            rounded_gate_drift=f.double()*c.double()+i.double()*g.double()-c.double()
            native_drift=cn.double()-c.double()
            rounded=(f==1);ulp=torch.nextafter(c,torch.full_like(c,float('inf')))-c
            rows.append(dict(condition=d['condition'],seed=d['seed'],context=d['context'],regime=d['regime'],qualified_initializers=len(z),
                checkpoint_sha256=sha(checkpoint),equilibrium_result_sha256=sha(path),
                native_float32_full_residual_RMS=(native.double()-z.double()).square().mean(1).sqrt().tolist(),
                native_float32_hidden_residual_RMS=(hn.double()-h.double()).square().mean(1).sqrt().tolist(),
                native_float32_c_max_step_change=float(native_drift.abs().max()),
                float32_forget_equal_one_count=int(rounded.sum()),float32_forget_equal_one_per_state=rounded.sum(1).tolist(),
                max_write_drift_with_rounded_gates_and_float64_c_arithmetic=float(rounded_gate_drift.abs().max()),
                rounded_forget_nonzero_write_count=int((rounded&(i*g!=0)).sum()),
                rounded_forget_nonzero_write_but_actual_c_unchanged_count=int((rounded&(i*g!=0)&(native_drift==0)).sum()),
                max_c_ulp_float32=float(ulp.abs().max()),minimum_stable_float32_sigmoid_negative_f=float((-gf).sigmoid().min())))
    report=dict(source_sha256=sha(__file__),role='Precision sensitivity of already fitted equilibria; no new equilibrium optimization or geometry selection.',cases=rows,
        native_precision='Original unmodified torch float32 LSTMCell, with fitted solution/input cast tofloat32.',
        interpretation='A finite float64 equilibrium may become a rounded-forget accumulator in float32. At sufficiently large c, the write can be smaller than the floating-point spacing and the stored c stops changing. Report actual recurrence and higher-precision arithmetic with the rounded gates separately; neither is an assumption about native accessibility.')
    save(out/'native_float32_precision_audit.json',report)
    print(json.dumps(dict(cases=len(rows),qualified_initializers=sum(r['qualified_initializers']for r in rows),
        forget_equal_one_count=sum(r['float32_forget_equal_one_count']for r in rows),
        nonzero_writes_lost_to_c_quantization=sum(r['rounded_forget_nonzero_write_but_actual_c_unchanged_count']for r in rows),
        max_native_hidden_RMS=max(max(r['native_float32_hidden_residual_RMS'])for r in rows),
        max_native_full_RMS=max(max(r['native_float32_full_residual_RMS'])for r in rows),
        max_c_step=max(r['native_float32_c_max_step_change']for r in rows),max_c_ulp=max(r['max_c_ulp_float32']for r in rows)),indent=2))


if __name__=='__main__':main()
