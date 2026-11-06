"""Read-only independent reconciliation of completed feedback-acquisition artifacts."""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='docs/research/local_return_feedback_portable_protocol.json'
PROTOCOL_SHA='4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
ROUTING='docs/research/local_return_feedback_portable_launch.json'
AUDIT='results/local_return_feedback_design/portable/cloud_audit'
BASELINE_SHA='73e7d0b3ccded5b155b6728608a4e3a9924a3523fb0ed1a19501a88e290da698'
ARMS={(c,m)for c in ('gc100','pgc33')for m in ('retained_actor','reset_actor')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(same(v,b[k])for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
    return a==b


def run(require_complete=False):
    inputs={}
    def checked(relative,digest=None,size=None):
        relative=Path(relative);assert not relative.is_absolute()and'..'not in relative.parts
        path=ROOT/relative;actual=sha(path)
        if digest is not None:assert actual==digest,relative
        if size is not None:assert path.stat().st_size==size,relative
        inputs[str(relative)]=actual;return path
    protocol=read(checked(PROTOCOL,PROTOCOL_SHA));routing=read(checked(ROUTING))
    assert routing['protocol_sha256']==PROTOCOL_SHA
    assert len(routing['workers'])==4 and {(w['condition'],w['actor_mode'])for w in routing['workers']}==ARMS
    old=read(checked(AUDIT+'/all_baselines_independent_review.json',BASELINE_SHA))
    baseline={(r['condition'],r['actor_mode']):r for r in old['rows']}
    init_name='docs/research/local_return_feedback_initialization.json'
    init=read(checked(init_name,protocol['input_assets'][init_name]))
    init={(r['condition'],r['actor_mode']):r for r in init['policies']}
    env=checked('code/env.py',protocol['source_sha256']['code/env.py'])
    labels=next(ast.literal_eval(n.value)for n in ast.parse(env.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ACTIONS'for t in n.targets))
    assert labels[1]=='turn_right'and labels[5]=='strafe_right'
    models=[];pending=[];final_streams={}
    for w in routing['workers']:
        key=(w['condition'],w['actor_mode']);a_path=ROOT/w['audit_path']
        audit=read(a_path)if a_path.exists()else{}
        if not audit.get('complete'):
            pending.append(dict(condition=key[0],actor_mode=key[1],status=audit.get('status','missing')));continue
        checked(w['audit_path']);assert audit['passed']and audit['status']=='complete'and not audit['failures']and not audit['pending']
        assert (audit['condition'],audit['actor_mode'])==key and audit['protocol_sha256']==PROTOCOL_SHA
        checked('scripts/audit_local_return_feedback.py',audit['auditor_sha256'])
        folder=Path(w['local_artifact_root'])/'experiment'
        assert Path(audit['folder']).parts[-len(folder.parts):]==folder.parts
        for name,proof in audit['artifacts'].items():checked(folder/name,proof['sha256'],proof['bytes'])
        history=baseline[key]['baseline_audit']
        before_audit=read(checked(history['path'],history['sha256'],history.get('bytes')))
        for name,proof in before_audit['artifacts'].items():
            if name.startswith('baseline')or name=='incoming.pt':checked(folder/name,proof['sha256'],proof['bytes'])
        initial=init[key];saved=torch.load(checked(initial['path'],initial['sha256'],initial['bytes']),map_location='cpu',weights_only=True)
        incoming=torch.load(ROOT/folder/'incoming.pt',map_location='cpu',weights_only=True)
        before=torch.load(ROOT/folder/'final_before_evaluation.pt',map_location='cpu',weights_only=True)
        final=torch.load(ROOT/folder/'final.pt',map_location='cpu',weights_only=True)
        assert same(incoming['policy'],saved['policy'])and incoming['counters']['updates']==0
        assert final['completed']and(final['counters']['updates'],final['counters']['trials'])==(1024,32768)
        assert final['provenance']['protocol_sha256']==PROTOCOL_SHA
        assert(final['provenance']['condition'],final['provenance']['actor_mode'])==key
        assert same(final['initialization'],incoming['initialization'])and same(final['config'],incoming['config'])
        for name in ('policy','optimizer','streams','global_rng','boundary_state'):assert same(before[name],final[name]),name
        final_streams[w['name']]=final['streams']
        metric=[json.loads(line)for line in(ROOT/folder/'metrics.jsonl').read_text().splitlines()]
        assert len(metric)==1024 and [r['update']for r in metric]==list(range(1,1025))
        assert sum(r['completed_trials']for r in metric)==32768
        for r in metric:
            assert all(math.isfinite(v)for v in r.values()if isinstance(v,(int,float)))
            assert r['replay_error']==r['value_replay_error']==r['setup_learning_reward']==r['forced_hits']==0
            assert r['terminal_bootstrap_zero']and r['fresh_feedback_on_every_active_input']and r['midtrial_resets']==r['state_noise_sigma']==0
        endpoint=read(ROOT/folder/'final_evaluation/summary.json');rows=[]
        for c in range(16):
            record=read(ROOT/folder/f'final_evaluation/context{c}/result.json')
            assert record['valid']and record['context']==c and len(record['rows'])==16
            assert sorted((r['context'],r['heading_index'])for r in record['rows'])==[(c,h)for h in range(16)]
            rows.extend(record['rows'])
        assert endpoint['rows']==rows
        successes=sum(r['hit']for r in rows);actions=[sum(r['action_counts'][i]for r in rows)for i in range(6)]
        total=endpoint['totals'];assert total==audit['checks']['evaluations']['final']['totals']
        assert total['trials']==total['valid_trials']==256 and total['invalid_trials']==0
        assert total['successes']==successes and total['action_counts']==actions
        assert total['free_updates']==sum(r['free_steps']for r in rows)==sum(actions)
        for target,field in [('mean_path_m','free_path_m'),('mean_progress_m','progress_m'),('mean_free_entropy','mean_free_entropy'),('mean_initial_free_entropy','first_free_entropy')]:
            assert math.isclose(total[target],sum(r[field]for r in rows)/256,rel_tol=0,abs_tol=1e-12),target
        assert endpoint['criterion_pass']==(successes>=192)==audit['checks']['evaluations']['final']['criterion_pass']
        assert endpoint['success_threshold']==192
        summary=read(ROOT/folder/'summary.json');assert summary['completed']and summary['counters']==final['counters']
        assert summary['final_checkpoint_sha256']==sha(ROOT/folder/'final.pt')
        for name in ('baseline','final'):
            r=final['counters']['evaluation_records'][name]
            assert r['policy_optimizer_private_global_rng_unchanged']
            assert r['summary_sha256']==sha(ROOT/folder/f'{name}_evaluation/summary.json')
        models.append(dict(condition=key[0],actor_mode=key[1],worker=w['name'],audit_path=w['audit_path'],audit_sha256=sha(a_path),
            final_checkpoint_sha256=summary['final_checkpoint_sha256'],updates=1024,trials=32768,evaluation_valid_trials=256,
            baseline_successes=baseline[key]['baseline_totals']['successes'],final_successes=successes,criterion_pass=successes>=192,
            final_action_counts={label:count for label,count in zip(labels,actions)},mean_progress_m=total['mean_progress_m'],mean_path_m=total['mean_path_m'],
            mean_free_entropy=total['mean_free_entropy'],baseline_artifacts_preserved=True,exact_final_evaluation_state_preservation=True,
            changed_trained_parameter_names=sorted(k for k in final['policy']if not same(final['policy'][k],incoming['policy'][k]))))
    streams=dict(complete=False,reason='Wait for all4 fixed finals and owner terminal comparison')
    stream_path=ROOT/AUDIT/'paired_final_training_validation.json'
    if len(models)==4 and stream_path.exists():
        report=read(checked(stream_path.relative_to(ROOT)));assert report['complete']and report['passed']and report['protocol_sha256']==PROTOCOL_SHA
        assert report['routing_sha256']==sha(ROOT/ROUTING)
        checked('scripts/compare_feedback_training_streams.py',report['producer_sha256'])
        assert report['checks']=={k:True for k in ('plan','motor','action','state')}
        assert len(report['inputs'])==4 and {r['name']for r in report['inputs']}==set(final_streams)
        for r in report['inputs']:
            checked(r['checkpoint'],r['checkpoint_sha256']);checked(r['audit'],r['audit_sha256'])
        raw=report['raw'];states=torch.load(checked(raw['path'],raw['sha256'],raw['bytes']),map_location='cpu',weights_only=True)
        assert set(states)==set(final_streams)
        first=next(iter(states.values()))
        for name,state in states.items():
            assert same(state,final_streams[name])and same(state,first)
            assert(state['seed'],state['batches'],state['contexts'])==(494801,1024,32768)
        streams=dict(complete=True,all_four_named_private_streams_identical=True,exact_saved_checkpoint_comparison=True,
            scope='Matching generator draw consumption and saved states; not a replay of every intervening Gaussian value.')
    complete=len(models)==4 and streams['complete']
    if require_complete:assert complete,'All4 complete finals and final private-stream comparison required'
    return dict(version='feedback_fixed_final_independent_review_v1',utc=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,complete=complete,
        source_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,input_sha256=inputs,models=models,pending=pending,private_streams=streams,
        action_labels=list(labels),scope='Independent saved-checkpoint, authenticated raw-artifact and JSON-row arithmetic review; no raw trajectory decompression, CNN/LSTM replay, policy/environment execution or training. Existing strict audits supply the raw physical/input/readout validation.',
        interpretation='Retained/reset names describe initial actor composition. All weights were trainable; final outcomes compare complete learned policies, not isolated final actor swaps.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/AUDIT/'completed_finals_independent_review.json');a=p.parse_args()
    result=run(require_complete=not a.partial)
    if not a.partial:
        assert not a.out.exists(),'Preserve the completed independent review'
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ('complete','models','pending','private_streams')}))


if __name__=='__main__':main()
