"""Saved-hash consistency of wrong cues and equivalent genuine donor states.

This is a deterministic implementation check, not a new condition, gate or
independent scientific replication. No tensor archive is decompressed and no
model, environment, fitted representation or optimizer is executed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/goal_cue_memory/original_diagnostic'
PROTOCOL=ROOT/'docs/research/goal_cue_memory_original_diagnostic_protocol.json'
PROTOCOL_SHA='2d9386fad85f9774d80ced37d7d6220d6bb75c682879684511a0da4d4bb04125'
WHOLE_FIELDS=('action','position','observation_position','observation_heading','observation_direction',
    'observation_velocity','motors_before_action','reward','raw_reward','active','terminal','physics_active',
    'policy_mask','pa','pr','rgb','code')
FIRST_FIELDS=('state_before_input','state','input','goal_input','position','heading','motors','probabilities','logits','value')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def checked_result(folder,audit,context,name):
    relative=f'evaluation/context{context}/{name}/result.json';path=folder/relative
    proof=audit['artifacts'][relative]
    assert sha(path)==proof['sha256']and path.stat().st_size==proof['bytes']
    result=read(path);assert result['context']==context and result['variant']==name
    if result['valid']:
        raw=result['raw'];raw_path=path.parent/raw['file']
        assert raw['file']=='raw.pt.gz'and raw['compression']=='gzip_level1_lossless'
        prior=audit['artifacts'][str(raw_path.relative_to(folder))]
        assert raw['sha256']==prior['sha256']==sha(raw_path)and raw['bytes']==prior['bytes']==raw_path.stat().st_size
    return result


def check():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    rows=[];audit_hashes={}
    for condition in ('gc100','pgc33'):
        audit_path=BASE/'audits'/f'{condition}.json';audit=read(audit_path)
        assert audit['passed']and audit['complete']and not audit['failures']and not audit['pending']
        assert audit['phase']=='original_diagnostic'and audit['condition']==condition and audit['protocol_sha256']==PROTOCOL_SHA
        assert audit['auditor_sha256']==sha(ROOT/'scripts/audit_goal_cue_memory_original_diagnostic.py')
        audit_hashes[condition]=sha(audit_path);folder=BASE/condition
        for context in range(16):
            for sign in ('plus90','minus90'):
                wrong=checked_result(folder,audit,context,f'wrong_{sign}')
                donor=checked_result(folder,audit,context,f'donor_{sign}')
                item=dict(condition=condition,context=context,sign=sign,eligible=wrong['valid']and donor['valid'])
                if item['eligible']:
                    a,b=wrong['raw']['tensor_schema'],donor['raw']['tensor_schema']
                    keys=['/'+field for field in WHOLE_FIELDS]
                    keys += [f'/evidence/first_free/{index}/{field}'for index in range(16)for field in FIRST_FIELDS]
                    different=[key for key in keys if a[key]!=b[key]]
                    item.update(exact_all_checked_tensors=not different,checked_tensor_count=len(keys),different_tensor_keys=different,
                        wrong_raw_sha256=wrong['raw']['sha256'],donor_raw_sha256=donor['raw']['sha256'])
                else:item.update(exact_all_checked_tensors=None,reason='A prescribed branch failed collection; pair retained without substitution.')
                rows.append(item)
    assert len(rows)==64
    return dict(version='goal_cue_wrong_cue_donor_equivalence_v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        complete=True,protocol_sha256=PROTOCOL_SHA,checker_sha256=sha(__file__),audit_sha256=audit_hashes,
        prescribed_pairs=64,eligible_pairs=sum(r['eligible']for r in rows),
        exactly_equal_pairs=sum(r.get('exact_all_checked_tensors')is True for r in rows),
        mismatched_pairs=sum(r.get('exact_all_checked_tensors')is False for r in rows),rows=rows,
        scope='Exact authenticated tensor-schema hashes for first-FREE incoming/post states, inputs and heads, plus complete saved action/physical/current-input histories. Cue-prefix states and goal inputs intentionally differ and are not equated. Whole logit/value tensors include those different prefixes and are not directly compared.',
        interpretation='Implementation consistency only. Same stationary non-goal input makes the wrong goal cue generate the corresponding genuine donor incoming state; both then receive the same recipient FREE inputs and tapes.',
        no_new_condition_or_gate=True,no_independent_scientific_replication=True,no_new_rollout_or_tensor_decompression=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=BASE/'audits/donor_equivalence.json');args=parser.parse_args()
    assert not args.out.exists(),'Preserve previous results; use an explicit fresh --out'
    result=check();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key]for key in ('complete','prescribed_pairs','eligible_pairs','exactly_equal_pairs','mismatched_pairs')}))
