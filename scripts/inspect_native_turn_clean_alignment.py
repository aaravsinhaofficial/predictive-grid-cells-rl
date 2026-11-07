"""Saved-output auxiliary: pure turns and action separation in the clean pair."""
from pathlib import Path
import datetime as dt
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import native_turn_original_memory as fixed
import native_turn_distillation as training
from rl_goal_memory import atomic_json


def describe(p,angles,context):
    p=p.double();p=p/p.sum(1,keepdim=True)
    weighted=(p[:,:2]*angles[:,:2].double()).sum(1)
    mass=p[:,:2].sum(1);tv=.5*(p[::2]-p[1::2]).abs().sum(1)
    return dict(context=context,cases=16,pairs=8,
        turn_weighted_body_goal_alignment_deg=(weighted*180/math.pi).tolist(),
        turn_probability=mass.tolist(),pair_policy_total_variation=tv.tolist())


def aggregate(contexts):
    assert len(contexts)==16 and {c['context']for c in contexts}==set(range(16))
    angles=[v for c in contexts for v in c['turn_weighted_body_goal_alignment_deg']]
    mass=[v for c in contexts for v in c['turn_probability']]
    tv=torch.tensor([v for c in contexts for v in c['pair_policy_total_variation']],dtype=torch.float64)
    assert len(angles)==len(mass)==256 and len(tv)==128
    return dict(cases=256,pairs=128,contexts=contexts,mean_turn_weighted_alignment_deg=sum(angles)/256,
        mean_turn_probability=sum(mass)/256,
        pooled_alignment_given_turn_deg=sum(angles)/sum(mass)if sum(mass)>0 else None,
        pair_tv_mean=float(tv.mean()),pair_tv_median=float(tv.median()),
        pair_tv_p90=float(torch.quantile(tv,.9)),pair_tv_max=float(tv.max()))


def main():
    torch.set_num_threads(2);source_hashes={};models={}
    frozen=json.loads(fixed.PROTOCOL.read_text())
    for condition in ('gc100','pgc33'):
        folder=training.BASE/condition/'clean';summary=json.loads((folder/'summary.json').read_text())
        execution=json.loads((training.BASE/f'{condition}_clean_execution.json').read_text())
        assert summary['complete']and summary['updates']==180 and execution['complete']and execution['exit_code']==0
        originals=json.loads((fixed.BASE/condition/'summary.json').read_text())
        final_pair=json.loads((folder/'final_paired/summary.json').read_text())
        assert summary['artifacts']['final_paired/summary.json']['sha256']==fixed.parent.sha(folder/'final_paired/summary.json')
        source_hashes[str((folder/'summary.json').relative_to(ROOT))]=fixed.parent.sha(folder/'summary.json')
        models[condition]={}
        for kind in fixed.parent.KINDS:
            collected={'teacher':{'full_native':[]},'original':{v:[]for v in fixed.VARIANTS},'clean':{v:[]for v in fixed.VARIANTS}}
            for context in range(16):
                name=f'context{context:02d}_{kind}.pt';parent_path=fixed.parent.BASE/condition/name
                target=torch.load(parent_path,map_location='cpu',weights_only=True)
                source_hashes[str(parent_path.relative_to(ROOT))]=fixed.parent.sha(parent_path)
                assert source_hashes[str(parent_path.relative_to(ROOT))]==frozen['asset_sha256'][str(parent_path.relative_to(ROOT))]
                angles=target['first_update_physics']['body_goal_angle_error_reduction_rad']
                collected['teacher']['full_native'].append(describe(target['teacher_first_choice_probabilities'],angles,context))
                for endpoint,path in (('original',fixed.BASE/condition/name),('clean',folder/'final_paired'/name)):
                    raw=torch.load(path,map_location='cpu',weights_only=True)
                    source_hashes[str(path.relative_to(ROOT))]=fixed.parent.sha(path)
                    expected=originals if endpoint=='original'else final_pair
                    assert source_hashes[str(path.relative_to(ROOT))]==expected['artifacts'][name]['sha256']
                    assert torch.equal(raw['first_update_physics']['body_goal_angle_error_reduction_rad'],angles)
                    result=json.loads(path.with_suffix('.json').read_text())
                    for v in fixed.VARIANTS:
                        values=describe(raw['variants'][v]['first_choice_probabilities'],angles,context)
                        for row in result['rows']:
                            if row['variant']==v:
                                assert abs(values['turn_weighted_body_goal_alignment_deg'][row['case']]*math.pi/180-row['turn_weighted_body_goal_alignment_rad'])<1e-12
                        collected[endpoint][v].append(values)
            models[condition][kind]={e:{v:aggregate(rows)for v,rows in variants.items()}for e,variants in collected.items()}
    previous=training.ROOT/'results/native_turn_distillation_design/clean_endpoint_inspection.json'
    result=dict(complete=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),producer_sha256=fixed.parent.sha(Path(__file__)),
        previous_clean_inspection_sha256=fixed.parent.sha(previous),source_sha256=source_hashes,models=models,
        scope='All16 contexts/128pairs in both modes, both original and fixed clean endpoints, all8 interventions. Saved data only; no policy, environment, training or geometry execution.',
        interpretation='Unconditional turn-probability-weighted reduction of absolute body-to-goal bearing error, in degrees per native update. Descriptive facing alignment is not an oracle navigation criterion because strafing can be useful without turning toward the goal. Pair policy TV uses all six actions and is not restricted to translational progress.')
    out=training.ROOT/'results/native_turn_distillation_design/clean_alignment_auxiliary.json';atomic_json(out,result)
    lines=['# Clean endpoint: turning and action-separation check','',
        'This saved-data check includes all16 contexts and128 opposite-heading pairs per mode. It supplements the first-step movement measure; useful pure turns can have zero translation.','',
        '| Model | History | Policy | Turn-weighted facing improvement (degrees/update) | Turn probability | Pair action TV | Largest pair TV |',
        '|---|---|---|---:|---:|---:|---:|']
    for c,kinds in models.items():
        for kind,endpoints in kinds.items():
            for e,v in (('teacher','full_native'),('original','native'),('clean','native'),('clean','ordinary'),('clean','pre_cue_reset_native')):
                a=endpoints[e][v]
                lines.append(f"| {c} | {kind} | {e}/{v} | {a['mean_turn_weighted_alignment_deg']:.7f} | {a['mean_turn_probability']:.4f} | {a['pair_tv_mean']:.6f} | {a['pair_tv_max']:.6f} |")
    lines+=['','Facing the goal is descriptive, not an oracle utility measure: the native agent can strafe. The JSON retains every context, individual case alignment/probability, and pair TV for all8 interventions. The earlier clean inspection is preserved unchanged and linked by hash.']
    g=models['gc100']['probe_glimpse'];p=models['pgc33']['probe_glimpse']
    lines+=['',f"**There is no broadly expressed hidden turn-choice effect in the clean endpoints.** Mean paired action TV is {100*g['clean']['native']['pair_tv_mean']:.2f}% for Grid and {100*p['clean']['native']['pair_tv_mean']:.2f}% for PGC+grid, compared with {100*g['teacher']['full_native']['pair_tv_mean']:.2f}% and {100*p['teacher']['full_native']['pair_tv_mean']:.2f}% for their teachers. The largest individual pair differences ({100*g['clean']['native']['pair_tv_max']:.2f}% and {100*p['clean']['native']['pair_tv_max']:.2f}%) are retained, so this is an aggregate statement rather than an assertion that every pair is identical.",
        '',f"The teachers' mean turn-weighted facing changes are {g['teacher']['full_native']['mean_turn_weighted_alignment_deg']:.3f}° and {p['teacher']['full_native']['mean_turn_weighted_alignment_deg']:.3f}° per update despite their positive goalward movement. This illustrates why facing alignment must remain descriptive, rather than a success criterion."]
    (out.with_suffix('.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps({c:{k:{e:{v:{name:a[name]for name in('mean_turn_weighted_alignment_deg','mean_turn_probability','pair_tv_mean','pair_tv_max')}for v,a in vv.items()if v in('native','ordinary','pre_cue_reset_native','full_native')}for e,vv in ee.items()}for k,ee in kk.items()}for c,kk in models.items()}))
if __name__=='__main__':main()
