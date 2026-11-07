"""Count failure mechanisms in already-audited resolution records only."""
from pathlib import Path
import collections
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_turn_geometry_resolution'
SPACING='max_edge_below_10percent_radius'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counts(items):return dict(sorted(collections.Counter(items).items()))


def inspect(rows):
    gates={n:counts(g for r in rows for g,v in r['densities'][n]['gates'].items()if not v)for n in ('256','512','1024')}
    resolution=counts('convergence_and_spacing'if not r['convergence']['passes']and any(not r['densities'][n]['gates'][SPACING]for n in ('512','1024'))else
        'convergence_only'if not r['convergence']['passes']else'spacing_only'if any(not r['densities'][n]['gates'][SPACING]for n in ('512','1024'))else'resolved_sampling'for r in rows)
    probes=[]
    for r in rows:
        if not r['normal_test_run']:continue
        for density,rec in r['recovery'].items():
            last=rec['bounded_followups'][-1]if rec['bounded_followups']else None
            result=last['result']if last else rec['nominal'];directions=result['directions'][:-1];md=result['perturbation_design']
            construction=md.get('physical_construction_conclusive',result['hidden_bound_excess_count']==0)
            invisible=[d['name']for d in directions if d['normal_informative_fraction']<.8]
            tangent=result['tangent_nontrivial_fraction']>=.8
            assert result['physical_conclusive']==bool(construction and tangent and not invisible)
            reasons=[]
            if not construction:reasons.append('invalid_bounded_direction_or_zero_size')
            if not tangent:reasons.append('initial_tangent_invisible')
            if invisible:reasons.append('initial_normal_invisible')
            failed=[d for d in directions if not d['passes']]
            probes.append(dict(context=r['context'],prefix=r['prefix'],kind=r['kind'],branch=r['branch'],regime=r['regime'],
                window_updates=r['window_updates'],observable=r['observable'],centered=r['centered'],metric=r['metric'],density=density,
                exact_alias=bool(r['source_exact_sham_of']),chosen_method=last['method']if last else'nominal',
                physical_conclusive=result['physical_conclusive'],physical_status=result['physical_status'],inconclusive_reasons=reasons,
                final_tangent_nontrivial_fraction=result['final_tangent_nontrivial_fraction'],
                initially_invisible_normals=invisible,
                failed_conclusive_directions=[dict(name=d['name'],informative_pair_fraction=d['informative_pair_fraction'],normal_gain=d['normal_gain'],normal_over_tangent_gain=d['normal_over_tangent_gain'],factor_leakage_count=d['factor_leakage_count'])for d in failed]if result['physical_conclusive']else[]))
    return dict(metric_rows=len(rows),resolution_status=resolution,failed_geometry_gates_by_density=gates,
        resolved_geometry_metric_rows=sum(r['resolved_geometry']for r in rows),normal_test_metric_rows=sum(r['normal_test_run']for r in rows),
        physical_probe_rows=len(probes),physical_status=counts(r['physical_status']for r in probes),chosen_methods=counts(r['chosen_method']for r in probes),
        inconclusive_reason_counts=counts(v for r in probes for v in r['inconclusive_reasons']),
        invisible_direction_counts=counts(v for r in probes for v in r['initially_invisible_normals']),
        conclusive_failed_direction_counts=counts(v['name']for r in probes for v in r['failed_conclusive_directions']),probes=probes)


def main():
    report=BASE/'report_original';summary=read(report/'summary.json');validation=read(report/'validation.json')
    assert summary['complete']and validation['complete']and validation['summary_sha256']==sha(report/'summary.json')
    models={};sources={}
    for condition in ('gc100','pgc33'):
        path=BASE/condition/'original/summary.json';data=read(path);assert data['complete'];sources[str(path.relative_to(ROOT))]=sha(path)
        model=next(r for r in summary['models']if r['condition']==condition);assert len(data['rows'])==model['selected_rows']
        models[condition]=dict(first_unmet_metric_pair_status=model['statuses'],selected_pairs=model['selected_metric_pairs'],exact_alias_pairs=model['exact_alias_pairs'],
            all_metric_rows=inspect(data['rows']),nonalias_metric_rows=inspect([r for r in data['rows']if not r['source_exact_sham_of']]))
    out=BASE/'original_failure_details.json';assert not out.exists()
    out.write_text(json.dumps(dict(complete=True,source_sha256=sha(__file__),reviewed_report_summary_sha256=sha(report/'summary.json'),
        reviewed_report_validation_sha256=sha(report/'validation.json'),input_summary_sha256=sources,models=models,
        unit='Metric rows and metric-row×density probes, not independent trained models or necessarily unique numerical arrays. Aliases retained in all totals and excluded in a separate section.',
        interpretation='First-unmet pair status is separate from per-metric diagnostics. Inconclusive visibility/construction is not failed recovery. Gate failures can overlap.',
        no_new_model_environment_or_recurrent_execution=True),indent=2)+'\n')
    for c,r in models.items():print(c,json.dumps({k:v for k,v in r.items()if k not in ('all_metric_rows','nonalias_metric_rows')}),json.dumps({k:v for k,v in r['nonalias_metric_rows'].items()if k!='probes'}))


if __name__=='__main__':main()
