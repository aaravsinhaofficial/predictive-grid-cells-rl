"""Retained robustness follow-up: bound hidden-state pulses, then equalize by shrinking."""
from __future__ import annotations
import argparse,copy,datetime as dt,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from code.common import build_policy,build_population
from native_movement_cycles import sha,verify_step
from native_movement_pulses import (MODES,PULSES,run_case,summarize_case,
    verify_selected_native_fork,MANIFEST as ORIGINAL_MANIFEST)
from rl_goal_memory import atomic_json,atomic_torch,state_digest
PROTOCOL=ROOT/'docs/research/native_movement_bounded_pulse_protocol.json'
MANIFEST=ROOT/'docs/research/native_movement_bounded_pulse_source_manifest.json'


def cosine_matrix(pulses,scale):
    directions=pulses[1:].double()/scale.double()
    norms=directions.norm(dim=1)
    assert bool((norms>0).all())
    return ((directions@directions.T)/(norms[:,None]*norms[None,:])).tolist()


def bounded_case(original,incoming):
    """No phase fitting: clip h only, then shrink all six full h/c displacements."""
    case=copy.deepcopy(original);pulses=original['pulses'].clone();scale=original['scale']
    clipped=pulses.clone()
    clipped[:,:256]=(incoming[None,:256]+pulses[:,:256]).clamp(-1,1)-incoming[None,:256]
    assert torch.equal(clipped[:,256:],pulses[:,256:])
    norms=(clipped.double()/scale.double()).square().mean(1).sqrt()
    target=norms[1:].min();assert target>0
    factors=torch.ones(7,dtype=torch.float64);factors[1:]=target/norms[1:]
    assert bool((factors<=1).all())
    bounded=(clipped.double()*factors[:,None]).to(pulses.dtype)
    proposed=incoming[None,:]+bounded
    assert bool((proposed[:,:256].abs()<=1).all()),'rounding exceeded legal h bounds'
    actual=proposed-incoming[None,:]
    actual_norm=(actual.double()/scale.double()).square().mean(1).sqrt()
    intended_norm=(bounded.double()/scale.double()).square().mean(1).sqrt()
    # Recorded float32 state addition need not preserve ideal delta at its last bit.
    assert float((actual_norm[1:]-target).abs().max()/target)<2e-5
    assert float((intended_norm[1:]-target).abs().max()/target)<2e-7
    original_scaled=pulses[1:].double()/scale.double();bounded_scaled=actual[1:].double()/scale.double()
    cosine=(original_scaled*bounded_scaled).sum(1)/(original_scaled.norm(dim=1)*bounded_scaled.norm(dim=1))
    case['pulses']=bounded
    case['metadata']=dict(case['metadata'],
        original_metadata=original['metadata'],
        bound_robustness=dict(method='Clip proposed h to[-1,1], leave c displacement unchanged, then shrink all6 full displacements to minimum clipped scaled norm; no rotations or newrandomdraws.',
            minimum_clipped_scaled_norm=float(target),shrink_factors=factors.tolist(),
            original_requested_norms=(pulses.double()/scale.double()).square().mean(1).sqrt().tolist(),
            clipped_norms=norms.tolist(),final_intended_norms=intended_norm.tolist(),final_realized_norms=actual_norm.tolist(),
            realized_norm_max_relative_error=float((actual_norm[1:]-target).abs().max()/target),
            original_cosine_matrix=cosine_matrix(pulses,scale),bounded_realized_cosine_matrix=cosine_matrix(actual,scale),
            cosine_with_original=cosine.tolist(),h_bound_exact=True,c_displacement_unchanged_during_clipping=True,
            nominal_direction_labels_retained=True,exact_orthogonality_not_assumed=True),
        initial_scaled_rms=actual_norm.tolist(),h_abs_max=proposed[:,:256].abs().max(1).values.tolist(),
        h_bound_overshoot_count=(proposed[:,:256].abs()>1).sum(1).tolist())
    return case


@torch.inference_mode()
def run(condition,seed,out):
    started=time.monotonic();out=Path(out);assert not out.exists()or not any(out.iterdir())
    manifest=json.loads(MANIFEST.read_text())
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    native=ROOT/f'results/native_movement_cycles/{condition}_s{seed}';native_summary=json.loads((native/'summary.json').read_text())
    assert sha(native/'native_trace.pt')==native_summary['trace_sha256']
    data=torch.load(native/'native_trace.pt',map_location='cpu',weights_only=True)
    original=ROOT/f'results/native_movement_pulses/{condition}_s{seed}';summary=json.loads((original/'summary.json').read_text());assert summary['complete']
    cfg=data['config'];net=build_policy(cfg,'cpu').eval();checkpoint=torch.load(ROOT/f'models/{condition}/{condition}_lstm_s{seed}_1000M.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint['policy']);net.requires_grad_(False);digest=state_digest(net);pop=build_population(cfg,'cpu')
    verification=verify_step(net,pop,cfg);cases=[]
    for name,meta in sorted(summary['raw_assets'].items()):
        path=original/name;assert sha(path)==meta['sha256']and path.stat().st_size==meta['bytes']
        raw=torch.load(path,map_location='cpu',weights_only=True);case0=raw['case']
        cases.append(bounded_case(case0,data['state_before'][case0['anchor'],case0['slot']]))
    assert len(cases)==summary['eligible'];out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'selection.json',dict(episodes=32,selected=[{k:v for k,v in c.items()if k not in ('scale','pulses')}for c in cases],missing=summary['missing'],identical_original_case_slots=True,original_summary_sha256=sha(original/'summary.json')))
    prechecks=[verify_selected_native_fork(net,pop,data,case)for case in cases]
    atomic_json(out/'native_only_prechecks.json',dict(complete=True,before_any_counterfactual=True,cases=prechecks))
    rows=[];checks=[]
    for i,case in enumerate(cases):
        raw=run_case(net,pop,data,case);atomic_torch(out/f"slot{case['slot']:02d}.pt",dict(case=case,**raw))
        rows.extend(summarize_case(raw,case));checks.append(raw['native_fork_checks'])
        print(json.dumps(dict(condition=condition,seed=seed,case=i+1,total=len(cases),slot=case['slot'],native_fork_checks=raw['native_fork_checks'],seconds=time.monotonic()-started)),flush=True)
    assert state_digest(net)==digest
    for name,expected in manifest['sha256'].items():assert sha(ROOT/name)==expected,name
    result=dict(complete=True,condition=condition,seed=seed,episodes=32,eligible=len(cases),missing=summary['missing'],rows=rows,
        fork_checks=checks,verification=verification,no_training=True,no_cloud=True,policy_unchanged=True,
        original_summary_sha256=sha(original/'summary.json'),protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(MANIFEST),seconds=time.monotonic()-started,
        raw_assets={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in out.glob('slot*.pt')})
    atomic_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items()if k not in ('rows','missing','raw_assets')}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=('gc100','pgc33'));parser.add_argument('--seed',type=int,choices=(1,2));parser.add_argument('--out',type=Path);parser.add_argument('--freeze',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    if args.freeze:
        assert not MANIFEST.exists();names=set(json.loads(ORIGINAL_MANIFEST.read_text())['sha256'])
        names|={str(ORIGINAL_MANIFEST.relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT)),'scripts/native_movement_bounded_pulses.py','tests/test_native_movement_bounded_pulses.py'}
        for condition in ('gc100','pgc33'):
            for seed in (1,2):
                folder=ROOT/f'results/native_movement_pulses/{condition}_s{seed}'
                names|={str((folder/name).relative_to(ROOT))for name in ('summary.json','selection.json')}
                names|={str(p.relative_to(ROOT))for p in folder.glob('slot*.pt')}
        atomic_json(MANIFEST,dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sha256={n:sha(ROOT/n)for n in sorted(names)}));print(sha(MANIFEST));return
    assert args.condition and args.seed and args.out;run(args.condition,args.seed,args.out)
if __name__=='__main__':main()
