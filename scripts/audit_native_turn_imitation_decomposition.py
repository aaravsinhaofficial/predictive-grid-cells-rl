"""Independent saved-array check of the descriptive imitation decomposition."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_turn_imitation_decomposition'
FIELDS=('observed_kl','teacher_floor','mean_policy_mismatch','history_gain',
        'mean_policy_counterfactual_kl','conditional_case_mismatch')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def dkl(p,q):
    q=np.broadcast_to(q,p.shape);terms=np.zeros_like(p);active=p>0
    assert np.all(q[active]>0)
    terms[active]=p[active]*np.log(p[active]/q[active])
    return terms.sum(-1)


def calculate(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    p/=p.sum(1,keepdims=True);q/=q.sum(1,keepdims=True)
    assert (q>0).all()
    pm=p.mean(0);qm=q.mean(0)
    actual=float(dkl(p,q).mean());floor=float(dkl(p,pm).mean())
    mismatch=float(dkl(pm[None],qm)[0]);counter=float(dkl(p,qm).mean())
    active=pm>0
    conditional=float(np.sum(pm[active]*dkl((p[:,active]/(len(p)*pm[active])).T,
                                           (q[:,active]/(len(q)*qm[active])).T)))
    return dict(observed_kl=actual,teacher_floor=floor,mean_policy_mismatch=mismatch,
        history_gain=counter-actual,mean_policy_counterfactual_kl=counter,
        conditional_case_mismatch=conditional),pm,qm


def main():
    torch.set_num_threads(1)
    result=read(BASE/'summary.json');assert result['complete']and result['no_new_rollout']and result['no_optimization']
    assert result['producer_sha256']==sha(ROOT/'scripts/decompose_native_turn_imitation.py')
    assert result['test_sha256']==sha(ROOT/'tests/test_decompose_native_turn_imitation.py')
    for name,entry in result['input_sha256'].items():
        path=ROOT/name;assert path.stat().st_size==entry['bytes']and sha(path)==entry['sha256'],name
    for name,entry in read(BASE/'artifacts.json').items():
        path=BASE/name;assert path.stat().st_size==entry['bytes']and sha(path)==entry['sha256'],name
    audit=read(ROOT/'results/native_turn_distillation/report/independent_validation.json')
    assert audit['complete']and audit['paired_cases']==24576
    assert len(result['aggregates'])==144 and len(result['groups'])==2304 and len(result['improvements'])==96
    cache={};values=defaultdict(list);cases=defaultdict(set);max_error=0.;zero_controls=0
    for group in result['groups']:
        c,e,scope,v=(group[k]for k in ('condition','endpoint','scope','variant'))
        p=[];q=[];xx=[];refs=[]
        for case in group['cases_ref']:
            context,kind,i=case['context'],case['kind'],case['case'];key=(c,e,context,kind)
            assert c in('gc100','pgc33')and e in('original','clean','noise')and kind in('turn_only','probe_glimpse')
            assert context in range(16)and i in range(16)and(scope=='all_modes'or scope==kind)
            if key not in cache:
                folder=(ROOT/'results/native_turn_original_memory'/c if e=='original'else ROOT/'results/native_turn_distillation'/c/e/'final_paired')
                assert sha(folder/'summary.json')==audit['models'][c][e]['paired']['summary_sha256']
                raw=torch.load(folder/f'context{context:02d}_{kind}.pt',map_location='cpu',weights_only=True)
                cache[key]=raw
            raw=cache[key]
            p.append(raw['teacher_first_choice_probabilities'][i].numpy())
            q.append(raw['variants'][v]['first_choice_probabilities'][i].numpy())
            xx.append(raw['variants'][v]['first_choice_input'][i].numpy())
            refs.append((context,kind,i))
        assert len(refs)==len(set(refs))==group['cases']
        assert all(np.array_equal(x,xx[0])for x in xx)
        assert hashlib.sha256(np.ascontiguousarray(xx[0]).tobytes()).hexdigest()==group['input_sha256']
        computed,pm,qm=calculate(p,q)
        for field in FIELDS:
            error=abs(computed[field]-group[field]);max_error=max(max_error,error);assert error<5e-12,(c,e,scope,v,field,error)
        assert np.allclose(pm,group['teacher_mean'],atol=1e-14,rtol=1e-14)
        assert np.allclose(qm,group['student_mean'],atol=1e-14,rtol=1e-14)
        assert computed['history_gain']<=computed['teacher_floor']+5e-12
        assert computed['observed_kl']>=computed['mean_policy_mismatch']-5e-12
        if v.startswith('choice_reset'):
            assert np.array_equal(q,np.repeat(np.asarray(q[:1]),len(q),axis=0))
            assert abs(computed['history_gain'])<5e-12;zero_controls+=1
        tag=(c,e,scope,v)
        assert not cases[tag].intersection(refs)
        cases[tag].update(refs);values[tag].append((len(refs),computed))
    assert len(cache)==192 and zero_controls==576
    means={}
    for row in result['aggregates']:
        tag=tuple(row[k]for k in ('condition','endpoint','scope','variant'));scope=tag[2]
        expected={(c,k,i)for c in range(16)for k in ('turn_only','probe_glimpse')for i in range(16)if scope=='all_modes'or k==scope}
        assert cases[tag]==expected and row['cases']==len(expected)and row['groups']==len(values[tag])
        mean={field:sum(n*x[field]for n,x in values[tag])/len(expected)for field in FIELDS}
        for field in FIELDS:
            error=abs(mean[field]-row[field]);max_error=max(max_error,error);assert error<5e-12
        means[tag]=mean
    primary=[]
    for row in result['improvements']:
        c,e,s,v=(row[k]for k in ('condition','endpoint','scope','variant'))
        before,after=means[c,'original',s,v],means[c,e,s,v]
        change=dict(kl_improvement=before['observed_kl']-after['observed_kl'],
            mean_policy_improvement=before['mean_policy_mismatch']-after['mean_policy_mismatch'],
            history_gain_change=after['history_gain']-before['history_gain'])
        for field,expected in change.items():assert abs(row[field]-expected)<5e-12
        assert abs(change['kl_improvement']-change['mean_policy_improvement']-change['history_gain_change'])<5e-12
        for field,key in(('mean_policy_fraction','mean_policy_improvement'),('history_fraction','history_gain_change')):
            if abs(change['kl_improvement'])<=1e-10:assert row[field]is None
            else:assert abs(row[field]-change[key]/change['kl_improvement'])<1e-10
        if s=='probe_glimpse'and v=='native':primary.append(row)
    out=BASE/'independent_validation.json'
    record=dict(complete=True,auditor_source_sha256=sha(__file__),producer_sha256=result['producer_sha256'],
        summary_sha256=sha(BASE/'summary.json'),training_independent_audit_sha256=sha(ROOT/'results/native_turn_distillation/report/independent_validation.json'),
        groups=2304,aggregates=144,improvements=96,choice_reset_zero_gain_groups=576,
        unique_raw_context_mode_files=192,maximum_absolute_error=max_error,primary_changes=primary,
        no_model_or_environment_execution=True,no_new_rollout=True,
        procedure='Independently rebuild every exact-input group from raw saved p/q and input bytes; recompute probability-mixture and posterior-KL identities, case-weighted aggregates, all signed improvements and reset-zero controls. Verify all input and report hashes.',
        report_artifact_sha256={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in sorted(BASE.iterdir())if p.is_file()and p.name not in('independent_validation.json','artifacts.json')})
    out.write_text(json.dumps(record,indent=2)+'\n')
    (BASE/'artifacts.json').write_text(json.dumps({p.name:dict(sha256=sha(p),bytes=p.stat().st_size)for p in sorted(BASE.iterdir())if p.is_file()and p.name!='artifacts.json'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items()if k not in('report_artifact_sha256','primary_changes')},indent=2))


if __name__=='__main__':main()
