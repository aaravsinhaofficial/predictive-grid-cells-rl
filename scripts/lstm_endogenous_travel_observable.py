"""Metadata adapter: unchanged observable control on released travel families."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from lstm_endogenous_observable_flow import observable_probe,load_net
from lstm_endogenous_curve_recovery import advance,sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=ROOT/'results/lstm_travel_memory/original')
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/observable_history/travel_original')
    p.add_argument('--continued',action='store_true');p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    sources=sorted(args.source_root.glob('*_s*/context*_states.npz'))
    protocol=dict(role='Metadata adapter only; frozen observable-history measures and thresholds unchanged.',
        source_root=str(args.source_root),archives=[str(s)for s in sources],continued=args.continued,
        mapping='states=ordinary-current-released family; ordinary_input is shared across all phases. Phase-specific native_inputs and native_cue_states are excluded from the autonomous analysis.',
        source_sha256=sha(__file__),observable_source_sha256=sha(ROOT/'scripts/lstm_endogenous_observable_flow.py'),
        observable_protocol_sha256=sha(ROOT/'results/lstm_endogenous_ring/observable_history/protocol.json'))
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    if args.freeze_only:return
    rows=[]
    with torch.no_grad(),threadpool_limits(limits=2):
        for path in sources:
            a=np.load(path);checkpoint=Path(str(a['checkpoint_path'].item()))
            assert sha(checkpoint)==str(a['checkpoint_sha256'].item())
            net,data,cfg=load_net(checkpoint)
            if args.continued:assert data.get('completed') and data['extra_frames']==25_024_000
            index=list(a['times']).index(15);curve=a['states'][index].astype(np.float64);initial=a['states'][0].astype(np.float64)
            scale=np.repeat(np.sqrt(np.mean(initial.reshape(-1,2,256)**2,axis=(0,2))),256);scale=np.maximum(scale,1e-8)
            x=torch.tensor(a['ordinary_input'],dtype=torch.float64);assert x.shape==(1,1287)
            drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
            transition=lambda z:advance(net.lstm,drive,z)
            branch={name:observable_probe(transition,curve,s,lambda z:z[:,:256],lambda z:net.pi(z[:,:256]).softmax(1))for name,s in [('raw_full_state_kicks',np.ones(512)),('h_c_RMS_full_state_kicks',scale)]}
            record=dict(condition=cfg['cond'],archive=str(path),archive_sha256=sha(path),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                source_sha256=sha(__file__),observable_source_sha256=protocol['observable_source_sha256'],
                target='Last actual travel direction; body heading continues turning.',input_condition='Shared ordinary-current release, saved goal unchanged.',
                reference_updates=15,latest_update=60,branches={'ordinary_current_released':branch})
            target=args.out/path.parent.name/path.name.replace('_states.npz','.json');target.parent.mkdir(parents=True,exist_ok=True);save(target,record)
            for metric,r in branch.items():rows.append(dict(model=path.parent.name,context=path.stem,metric=metric,candidate=r['centered_conditional_factor_candidate'],
                radial_over_tangent=r['radial_over_tangent_observable_gain']['median'],radial_gain=r['radial_observable_normal_gain']['median'],
                centered_invariance_p90=r['invariance']['centered']['residual_to_reference_curve_over_radius']['p90'],
                failed_gates=[k for k,v in r['candidate_gates']['centered'].items()if not v]))
            print(json.dumps(dict(archive=str(path),candidates={name:r['centered_conditional_factor_candidate']for name,r in branch.items()})),flush=True)
    save(args.out/'summary.json',dict(cases=len(sources),rows=rows))


if __name__=='__main__':main()
