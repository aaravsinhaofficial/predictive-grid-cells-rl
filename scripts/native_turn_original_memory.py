"""Read-only memory/cue interventions on every sealed teacher-preflight history.

No optimizer, environment stepping, learned decoder, or scientific pass/fail gate.
First-step utility uses the six native physical action outcomes already archived.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import native_turn_teacher_preflight as parent
from code.common import build_policy,build_population
from rl_goal_memory import atomic_json,atomic_torch,state_digest

BASE=ROOT/'results/native_turn_original_memory'
PROTOCOL=ROOT/'docs/research/native_turn_original_memory_protocol.json'
VARIANTS=('native','ordinary','cue_plus60','cue_minus60',
    'pre_cue_reset_native','pre_cue_reset_ordinary','choice_reset_native','choice_reset_ordinary')
SOURCES=('scripts/native_turn_original_memory.py','tests/test_native_turn_original_memory.py')


@torch.no_grad()
def replay_own_prefixes(net,bank,data,require_original_cache):
    """Replay each checkpoint's OWN h/c from zero, with native32-slot resets.

    Saved original h/c is only a provenance/comparison target, never an input.
    This function therefore also supports a later changed-weight evaluation.
    """
    by_step={};last=max(c['step']for c in bank['contexts'])
    for c in bank['contexts']:
        slot,t=c['slot'],c['step']
        assert torch.equal(c['preceding_native_x'],data['x'][:t,slot])
        assert torch.equal(c['preceding_done'],data['done'][:t,slot])
        assert torch.equal(c['native_x'],data['x'][t,slot])
        assert torch.equal(c['initial_hc'],data['state_before'][t,slot])
        by_step.setdefault(t,[]).append(c)
    state=net.zero_state(32,'cpu');states={};rows=[]
    for t in range(last+1):
        for c in by_step.get(t,[]):
            actual=torch.cat(state,1)[c['slot']].clone()
            error=float((actual-c['initial_hc']).abs().max())
            if require_original_cache:assert error==0.,(c['context_id'],error)
            states[c['context_id']]=actual
            rows.append(dict(context=c['context_id'],slot=c['slot'],step=t,
                original_cache_max_error=error,own_incoming_state_sha256=parent.tensor_sha(actual)))
        if t<last:
            state=net.lstm(data['x'][t],state)
            state=tuple(v*(~data['done'][t])[:,None]for v in state)
    return states,dict(complete=True,replayed_updates=last,original_batch_shape=32,
        cached_original_state_used_as_input=False,require_original_cache=require_original_cache,contexts=rows)


def rotate(v,angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*v[:,0]-s*v[:,1],s*v[:,0]+c*v[:,1]),1)


@torch.no_grad()
def variant_history(raw,audit,pop,variant):
    assert variant in VARIANTS
    ordinary=variant.endswith('ordinary')
    x=raw['student_ordinary_x'if ordinary else'student_native_x'].clone()
    cue=audit['cue_index'];reset=None
    if variant in ('cue_plus60','cue_minus60')and cue is not None:
        angle=math.pi/3*(1 if variant=='cue_plus60'else-1)
        p=raw['physical'];indices=torch.as_tensor(pop.pgc_idx,dtype=torch.long)
        code=pop.activity(p['pos'][cue],rotate(p['mdir'][cue],angle),rotate(p['obs_vel'][cue],angle))
        x[cue,:,256+indices]=code[:,indices]
    if variant.startswith('pre_cue_reset'):
        reset=cue if cue is not None else raw['labels'].index('suffix_0')
    elif variant.startswith('choice_reset'):reset=len(x)-1
    # The only possible input change is the one measured predictive-current cue.
    outside=torch.ones(1287,dtype=torch.bool)
    outside[256+torch.as_tensor(pop.pgc_idx,dtype=torch.long)]=False
    assert torch.equal(x[:,:,outside],raw['student_native_x'][:,:,outside])
    if cue is None:assert torch.equal(x,raw['student_native_x'])
    else:
        keep=torch.ones(len(x),dtype=torch.bool);keep[cue]=False
        assert torch.equal(x[keep],raw['student_native_x'][keep])
    assert torch.equal(x[-1],raw['student_native_x'][-1])
    return x,reset,dict(cue_index=cue,reset_before_input_index=reset,
        reset_timing=('Before measured probe cue input'if cue is not None else
            'Before suffix_0 observation, after its common forced action; no intervening recurrent update')
            if variant.startswith('pre_cue_reset')else('Immediately before first choice input'if reset is not None else None),
        cue_operation_noop=cue is None or len(pop.pgc_idx)==0,
        altered_input_sha256=parent.tensor_sha(x))


@torch.no_grad()
def replay_variant(net,x,initial_hc,reset):
    n=x.shape[1]
    state=tuple(v.clone()for v in initial_hc.expand(n,-1).chunk(2,1))
    saved={}
    for t,item in enumerate(x):
        before=torch.cat(state,1).clone()
        if t==reset:
            state=tuple(torch.zeros_like(v)for v in state)
            saved['reset_state_before']=before
            saved['reset_state_after']=torch.cat(state,1).clone()
        if t==len(x)-1:
            saved['first_choice_state_before_intervention']=before
            saved['first_choice_state_before_input']=torch.cat(state,1).clone()
        state=net.lstm(item,state)
    logits=net.pi(state[0])
    saved.update(first_choice_state_after_input=torch.cat(state,1).clone(),
        first_choice_logits=logits.clone(),first_choice_probabilities=logits.softmax(1),
        first_choice_input=x[-1].clone())
    assert all(torch.isfinite(v).all()for v in saved.values())
    return saved


def normalized(p):
    p=p.double();assert torch.isfinite(p).all()and(p>=0).all()
    return p/p.sum(1,keepdim=True)


def forward_kl(target,predicted):
    p,q=normalized(target),normalized(predicted);tiny=torch.finfo(torch.float64).tiny
    return (p*(p.clamp_min(tiny).log()-q.clamp_min(tiny).log())).sum(1)


def context_metrics(saved,target,physics,context,kind,variant,reference_pairs):
    p=normalized(saved['first_choice_probabilities']);q=physics['progress_m'].double()
    kl=forward_kl(target,p);progress=(p*q).sum(1)
    alignment=(p[:,:2]*physics['body_goal_angle_error_reduction_rad'][:,:2].double()).sum(1)
    rows=[];pairs=[]
    for i in range(len(p)):
        rows.append(dict(context=context,kind=kind,variant=variant,case=i,pair=i//2,member=i%2,
            teacher_forward_kl_nats=float(kl[i]),expected_progress_m=float(progress[i]),
            turn_weighted_body_goal_alignment_rad=float(alignment[i]),argmax_action=int(p[i].argmax())))
    for i,ref in enumerate(reference_pairs):
        a,b=2*i,2*i+2;ceiling=float(q[a:b].mean(0).max())
        assert abs(ceiling-ref['best_common_action_ceiling_m'])<1e-12
        mean=float(progress[a:b].mean())
        pairs.append(dict(context=context,kind=kind,variant=variant,pair=i,
            teacher_forward_kl_nats=float(kl[a:b].mean()),expected_progress_m=mean,
            best_common_action_ceiling_m=ceiling,progress_above_common_ceiling_m=mean-ceiling,
            student_pair_total_variation=float(.5*(p[a]-p[a+1]).abs().sum()),
            teacher_pair_jensen_shannon_nats=ref['teacher_jensen_shannon'],
            teacher_pair_total_variation=ref['teacher_total_variation']))
    return rows,pairs


@torch.no_grad()
def evaluate_context(net,pop,raw,reference,context,kind,initial_hc):
    saved={};rows=[];pairs=[];operations={}
    for variant in VARIANTS:
        x,reset,operation=variant_history(raw,reference['audit'],pop,variant)
        result=replay_variant(net,x,initial_hc,reset)
        if operation['cue_index']is not None:result['actual_cue_input']=x[operation['cue_index']].clone()
        rr,pp=context_metrics(result,raw['teacher_first_choice_probabilities'],raw['first_update_physics'],
            context,kind,variant,reference['pairs'])
        rows.extend(rr);pairs.extend(pp);saved[variant]=result;operations[variant]=operation
    def same(names):
        for name in names[1:]:
            for key in ('first_choice_state_before_input','first_choice_state_after_input','first_choice_logits'):
                assert torch.equal(saved[names[0]][key],saved[name][key]),(context,kind,names,key)
    same(('choice_reset_native','choice_reset_ordinary'))
    if kind=='turn_only'or len(pop.pgc_idx)==0:
        same(('native','ordinary','cue_plus60','cue_minus60'))
        same(('pre_cue_reset_native','pre_cue_reset_ordinary'))
    zero=saved['choice_reset_native']['first_choice_logits']
    assert torch.equal(zero,zero[:1].expand_as(zero))
    for key in ('first_choice_state_before_input','first_choice_state_after_input','first_choice_logits'):
        common=saved['pre_cue_reset_ordinary'][key]
        assert torch.equal(common,common[:1].expand_as(common)),(context,kind,key)
    assert all(v['progress_above_common_ceiling_m']<=1e-8 for v in pairs if v['variant'].startswith('choice_reset'))
    return dict(own_prefix_state=initial_hc,variants=saved,
        teacher_first_choice_probabilities=raw['teacher_first_choice_probabilities'],
        first_update_physics=raw['first_update_physics']),dict(complete=True,context=context,kind=kind,
        operations=operations,rows=rows,pairs=pairs,zero_state_same_input_exact=True,
        pre_cue_reset_ordinary_all_phases_exact=True,
        cue_shams_exact=(kind=='turn_only'or len(pop.pgc_idx)==0),choice_reset_sham_exact=True)


def grouped_comparisons(inputs,targets,refs,predictions):
    information=parent.identical_input_information(inputs,targets,refs)
    lookup={(r['context'],r['kind'],r['case']):i for i,r in enumerate(refs)}
    for group in information['records']:
        indices=[lookup[(r['context'],r['kind'],r['case'])]for r in group['cases']]
        group['variants']={}
        for variant in VARIANTS:
            p=predictions[variant][indices]
            value=float(forward_kl(targets[indices],p).mean())
            group['variants'][variant]=dict(mean_teacher_forward_kl_nats=value,
                memoryless_floor_minus_observed_kl_nats=group['generalized_js_nats']-value)
            if variant.startswith('choice_reset'):
                assert torch.equal(p,p[:1].expand_as(p))
                assert value>=group['generalized_js_nats']-1e-10
    return information


def validate_frozen(design):
    assert design['no_optimization']and design['no_cloud']and design['variants']==list(VARIANTS)
    for key in ('source_sha256','asset_sha256'):
        for name,digest in design[key].items():assert parent.sha(ROOT/name)==digest,name


def freeze():
    assert not PROTOCOL.exists()
    old=json.loads(parent.PROTOCOL.read_text());assets=dict(old['asset_sha256'])
    assets[str(parent.PROTOCOL.relative_to(ROOT))]=parent.sha(parent.PROTOCOL)
    for condition in parent.CONDITIONS:
        bank=old['banks'][condition];assets[bank['path']]=bank['sha256']
        folder=parent.BASE/condition;summary=json.loads((folder/'summary.json').read_text())
        assert summary['complete']and summary['protocol_sha256']==parent.sha(parent.PROTOCOL)
        assert summary['pair_counts']=={k:128 for k in parent.KINDS}
        assets[str((folder/'summary.json').relative_to(ROOT))]=parent.sha(folder/'summary.json')
        for name,record in summary['artifacts'].items():
            path=folder/name;assert parent.sha(path)==record['sha256']and path.stat().st_size==record['bytes']
            assets[str(path.relative_to(ROOT))]=record['sha256']
    sources={**old['source_sha256'],**{name:parent.sha(ROOT/name)for name in SOURCES}}
    design=dict(version='native_turn_original_memory_v1',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parent_protocol_sha256=parent.sha(parent.PROTOCOL),source_sha256=sources,asset_sha256=assets,
        conditions=list(parent.CONDITIONS),variants=list(VARIANTS),kinds=list(parent.KINDS),
        contexts=16,pairs_per_kind_model_variant=128,cases_per_kind_model_variant=256,
        initial_models='Exact original1B seed1 checkpoints, all tensors fixed; no optimization',
        own_prefix_replay='Full native32-slot input tape from zero with original terminal masks; cached teacher h/c never supplied as input',
        cue='Only measured probe endpoint169 predictive-current slots; goal code unchanged. Turn-only cue changes are exact no-ops.',
        resets='Pre-cue before probe cue input; turn-only uses suffix_0 input. Choice reset immediately before identical first-choice input.',
        references='Per exact-input-group generalized JS is the minimum mean teacher forward KL of a common memoryless distribution. Paired best-common-action native first-step progress is a separate physical reference.',
        interpretation='Positive paired progress above its common-action ceiling demonstrates useful history for this controlled first choice. KL below the group floor is a stronger teacher-imitation criterion, not required for useful progress. Neither implies ring geometry.',
        qualification=None,no_optimization=True,no_cloud=True)
    validate_frozen(design);atomic_json(PROTOCOL,design);print(parent.sha(PROTOCOL))


@torch.no_grad()
def run(condition,out):
    design=json.loads(PROTOCOL.read_text());validate_frozen(design)
    old=json.loads(parent.PROTOCOL.read_text());bank_record=old['banks'][condition]
    bank=torch.load(ROOT/bank_record['path'],map_location='cpu',weights_only=True)
    assert parent.tensor_inventory(bank)==bank_record['tensors']
    cfg=bank['config'];checkpoint=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
    original=torch.load(checkpoint,map_location='cpu',weights_only=True);assert original['frames']==1_000_000_000
    rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):net=build_policy(cfg,'cpu').eval().requires_grad_(False)
    net.load_state_dict(original['policy'],strict=True);digest=state_digest(net)
    assert digest==bank['census_collection_provenance']['policy_hash']
    pop=build_population(cfg,'cpu');out=Path(out);assert not out.exists()or not any(out.iterdir())
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic();utc=dt.datetime.now(dt.timezone.utc).isoformat()
    data=torch.load(ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt',map_location='cpu',weights_only=True)
    states,prefix_report=replay_own_prefixes(net,bank,data,True);del data
    atomic_json(out/'prefix_replay.json',prefix_report)
    rows=[];pairs=[];inputs=[];targets=[];refs=[];predictions={v:[]for v in VARIANTS}
    for context in bank['contexts']:
        for kind in parent.KINDS:
            name=f"context{context['context_id']:02d}_{kind}";folder=parent.BASE/condition
            raw=torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=True)
            reference=json.loads((folder/f'{name}.json').read_text())
            assert reference['complete']and raw['context_reference']=={k:context[k]for k in ('context_id','slot','step')}
            assert torch.equal(raw['initial_hc'],states[context['context_id']].expand_as(raw['initial_hc']))
            try:
                saved,result=evaluate_context(net,pop,raw,reference,context['context_id'],kind,states[context['context_id']])
                result['source_raw_sha256']=parent.sha(folder/f'{name}.pt')
                result['source_reference_sha256']=parent.sha(folder/f'{name}.json')
                atomic_torch(out/f'{name}.pt',saved);atomic_json(out/f'{name}.json',result)
                rows.extend(result['rows']);pairs.extend(result['pairs'])
                inputs.append(raw['student_native_x'][-1]);targets.append(raw['teacher_first_choice_probabilities'])
                refs.extend(dict(context=context['context_id'],kind=kind,case=i,pair=i//2,member=i%2)for i in range(16))
                for v in VARIANTS:predictions[v].append(saved['variants'][v]['first_choice_probabilities'])
            except Exception as error:
                atomic_json(out/'incomplete.json',dict(complete=False,context=context['context_id'],kind=kind,error_type=type(error).__name__,error=str(error)));raise
    inputs=torch.cat(inputs);targets=torch.cat(targets);predictions={k:torch.cat(v)for k,v in predictions.items()}
    groups={'all_modes':grouped_comparisons(inputs,targets,refs,predictions)}
    for kind in parent.KINDS:
        ix=[i for i,r in enumerate(refs)if r['kind']==kind]
        groups[kind]=grouped_comparisons(inputs[ix],targets[ix],[refs[i]for i in ix],{v:p[ix]for v,p in predictions.items()})
    summaries={}
    for kind in parent.KINDS:
        summaries[kind]={}
        for v in VARIANTS:
            rr=[r for r in rows if r['kind']==kind and r['variant']==v]
            pp=[p for p in pairs if p['kind']==kind and p['variant']==v]
            assert len(rr)==256 and len(pp)==128
            summaries[kind][v]=dict(cases=len(rr),pairs=len(pp),
                **{key:sum(r[key]for r in rr)/len(rr)for key in ('teacher_forward_kl_nats','expected_progress_m','turn_weighted_body_goal_alignment_rad')},
                best_common_action_ceiling_m=sum(p['best_common_action_ceiling_m']for p in pp)/len(pp),
                progress_above_common_ceiling_m=sum(p['progress_above_common_ceiling_m']for p in pp)/len(pp),
                memoryless_teacher_kl_floor_nats=groups[kind]['case_weighted_minimum_forward_kl_nats'])
    assert state_digest(net)==digest and torch.equal(rng,torch.get_rng_state())
    validate_frozen(design)
    artifacts={p.name:dict(sha256=parent.sha(p),bytes=p.stat().st_size)for p in sorted(out.iterdir())if p.is_file()}
    atomic_json(out/'summary.json',dict(complete=True,condition=condition,protocol_sha256=parent.sha(PROTOCOL),
        checkpoint_sha256=parent.sha(checkpoint),policy_hash=digest,source_unchanged=True,policy_unchanged=True,
        global_cpu_rng_preserved=True,no_optimization=True,no_cloud=True,qualification=None,
        start_utc=utc,end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),monotonic_seconds=time.monotonic()-start,
        rows=rows,pairs=pairs,summary=summaries,exact_input_groups=groups,artifacts=artifacts))


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--condition',choices=parent.CONDITIONS);p.add_argument('--out',type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    if args.freeze:freeze();return
    assert args.condition and args.out;run(args.condition,args.out)
if __name__=='__main__':main()
