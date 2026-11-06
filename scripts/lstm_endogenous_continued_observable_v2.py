"""Final75M metadata adapter; original frozen observable probe is unchanged."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from lstm_endogenous_observable_flow import load_net,observable_probe
from lstm_endogenous_curve_recovery import advance,sha,save

VERSIONS={'v2':'visual_memory_extension_v2','v3':'visual_memory_state_noise_v3'}
PARENT_NAMES={'gc100':'ring-20261105-vision-grid-01','pgc33':'ring-20261105-vision-pgc-01'}


def validate_checkpoint(data,cfg,arm):
    assert data['completed'] is True
    assert data['original_frames']==1_000_000_000
    assert data['extra_frames']==75_072_000 and data['extension_frames']==50_048_000
    assert data['provenance']['version']==VERSIONS[arm]
    parent=ROOT/'results/ring_cloud_artifacts'/PARENT_NAMES[cfg['cond']]/'final.pt'
    assert data['provenance']['parent_sha256'][cfg['cond']]['final.pt']==sha(parent)
    if arm=='v3':
        assert data['runtime_snapshot']['recurrent_noise_sigma']==.02
        assert 'recurrent_noise_rng' in data['runtime_snapshot']
    return sha(parent)


def run(checkpoint,archive,out,arm):
    net,data,cfg=load_net(checkpoint);parent_hash=validate_checkpoint(data,cfg,arm)
    a=np.load(archive)
    assert str(a['checkpoint_sha256'].item())==sha(checkpoint),'Saved family belongs to a different checkpoint'
    index=list(a['times']).index(15)
    if 'native_states' in a:
        family='heading';variants=[('native_current','native_states','native_input'),('ordinary_current','ordinary_states','ordinary_input')]
    else:
        family='travel';variants=[('ordinary_current_released','states','ordinary_input')]
    record=dict(arm=arm,family=family,condition=cfg['cond'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),parent25M_sha256=parent_hash,
        archive=str(archive),archive_sha256=sha(archive),source_sha256=sha(__file__),
        observable_source_sha256=sha(ROOT/'scripts/lstm_endogenous_observable_flow.py'),
        extra_frames=data['extra_frames'],extension_frames=data['extension_frames'],branches={})
    with torch.no_grad():
        for name,state_key,input_key in variants:
            curve=a[state_key][index].astype(np.float64);initial=a[state_key][0].astype(np.float64)
            scale=np.repeat(np.sqrt(np.mean(initial.reshape(-1,2,256)**2,axis=(0,2))),256);scale=np.maximum(scale,1e-8)
            x=torch.tensor(a[input_key],dtype=torch.float64);assert x.shape==(1,1287),'Autonomous test requires shared input across phases'
            drive=torch.nn.functional.linear(x,net.lstm.weight_ih,net.lstm.bias_ih)
            transition=lambda z:advance(net.lstm,drive,z)
            record['branches'][name]={metric:observable_probe(transition,curve,s,lambda z:z[:,:256],lambda z:net.pi(z[:,:256]).softmax(1))for metric,s in [('raw_full_state_kicks',np.ones(512)),('h_c_RMS_full_state_kicks',scale)]}
    target=out/arm/family/cfg['cond']/archive.name.replace('_states.npz','.json');target.parent.mkdir(parents=True,exist_ok=True);save(target,record)
    print(json.dumps(dict(archive=str(archive),arm=arm,condition=cfg['cond'],family=family,
        candidates={name:{metric:r['centered_conditional_factor_candidate']for metric,r in branch.items()}for name,branch in record['branches'].items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path);p.add_argument('--archives',type=Path,nargs='+')
    p.add_argument('--arm',choices=tuple(VERSIONS));p.add_argument('--out',type=Path,default=ROOT/'results/lstm_endogenous_ring/observable_history/final75')
    p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(role='Metadata adaptation only for prespecified fixed75.072M final models; frozen observable calculations and gates unchanged.',
        arms=VERSIONS,required_frames=dict(original=1_000_000_000,total_extra=75_072_000,extension=50_048_000),
        validation='Completed final only; arm provenance; exact25Mparent checkpoint hash; archive checkpoint hash; v3 sigma=.02 and stored private noise RNG.',
        families='Heading: both shared native and ordinary inputs. Travel: only ordinary-current release; phase-specific sustained native input is excluded.',
        numerical_method='Same observable_probe, reference15 + evolution30 + observation15 updates, every h at4fixed times, same full-state radial/tangent kicks, centering and guards.',
        source_sha256=sha(__file__),observable_source_sha256=sha(ROOT/'scripts/lstm_endogenous_observable_flow.py'),
        observable_protocol_sha256=sha(ROOT/'results/lstm_endogenous_ring/observable_history/protocol.json'))
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    if args.freeze_only:return
    if args.checkpoint is None or not args.archives or args.arm is None:p.error('--checkpoint, --archives, --arm required')
    with threadpool_limits(limits=2):
        for archive in args.archives:run(args.checkpoint,archive,args.out,args.arm)


if __name__=='__main__':main()
