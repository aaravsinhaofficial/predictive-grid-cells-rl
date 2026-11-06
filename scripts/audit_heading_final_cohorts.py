"""Independent complete-cohort arithmetic/provenance audit, no new tests.

Refuses to calculate partial-cohort inference. Reads saved arrays/reports only;
does not run training or simulation or modify frozen sources.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    geo = read('results/ring_heading/geometry_confirmation/summary.json')
    raw_path = ROOT / 'results/ring_heading/raw_sensitivity_evaluation/summary.json'
    raw_memory = read('results/ring_heading/raw_memory_evaluation/memory_summary.json')
    raw_anchor = read('results/ring_heading/raw_memory_evaluation/anchor_summary.json')
    if not (geo['complete'] and raw_path.exists() and raw_memory['complete'] and raw_anchor['complete']):
        print(json.dumps({'ready':False,'geometry_loaded':geo['models_loaded'],
            'raw_tracking_complete':raw_path.exists(),'raw_memory_loaded':raw_memory['models_loaded'],
            'raw_anchor_loaded':raw_anchor['models_loaded']}));return
    raw = json.loads(raw_path.read_text())
    locked = read('docs/research/heading_geometry_confirmation_sources.json')['source_hashes']
    for path,digest in locked.items():assert sha(ROOT/path)==digest,path
    records=[]
    for seed in range(5001,5017):
        for condition in ('grid','predictive'):
            prefix=ROOT/f'results/ring_heading/geometry_confirmation/{condition}_s{seed}'
            report=json.loads(prefix.with_suffix('.json').read_text())
            assert report['source_hashes']==locked and report['stage']=='independent_geometry_confirmation'
            assert report['seed']==seed and report['condition']==condition and report['steps_completed']==6000
            candidates=list((ROOT/'results/ring_cloud_artifacts').glob(f'ring-20261105-geometry-*/{condition}_s{seed}.pt'))
            assert len(candidates)==1 and sha(candidates[0])==report['checkpoint_sha256']
            archive=np.load(prefix.with_suffix('.npz'))
            for key,summary in report['maps'].items():
                uniform=np.linalg.norm(archive[key+'__derivative'],axis=1)
                refined=np.linalg.norm(archive[key+'__center_derivative'],axis=1)
                peak=max(uniform.max(),refined.max());median=np.median(uniform)
                expected=np.log10(max(peak,1e-12)/max(median,1e-12))
                assert np.isclose(expected,summary['log10_peak_over_median'],atol=1e-12,rtol=0)
                assert np.isclose(peak,summary['absolute_peak_sensitivity'],atol=1e-10,rtol=1e-12)
                assert np.isclose(median,summary['absolute_median_sensitivity'],atol=1e-12,rtol=1e-12)
            assert report['primary']==report['maps']['native_blank40']['log10_peak_over_median']
            records.append({'seed':seed,'condition':condition,'primary':report['primary'],
                'checkpoint_sha256':report['checkpoint_sha256']})
    lookup={(r['seed'],r['condition']):r['primary'] for r in records}
    delta=np.array([lookup[s,'predictive']-lookup[s,'grid'] for s in range(5001,5017)])
    signs=np.array(list(itertools.product((-1.,1.),repeat=16)))
    expected_p=np.mean(np.abs((signs*delta).mean(axis=1))>=abs(delta.mean())-1e-12)
    half=t.ppf(.975,15)*delta.std(ddof=1)/np.sqrt(16)
    inference=geo['primary_inference']
    assert expected_p==inference['p_two_sided_exact']
    np.testing.assert_allclose(inference['ci95'],[delta.mean()-half,delta.mean()+half],atol=1e-12)
    assert np.isclose(inference['geometric_mean_sharpness_ratio'],10**delta.mean())
    assert [r['seed'] for r in raw['per_seed']]==list(range(3001,3009))
    raw_records=[]
    training_locked=read('docs/research/heading_confirmation_sources.json')['sources']
    training_locked={k:v for k,v in training_locked.items() if k!='scripts/run_ring_batch.py'}
    for seed in range(3001,3009):
        for condition in ('grid','predictive','grid_heading'):
            r=read(f'results/ring_heading/raw_sensitivity_evaluation/{condition}_s{seed}.json')
            assert r['training_seed']==seed and r['condition']==condition
            assert r['cohort']=='raw_architecture_sensitivity' and r['steps_completed']==6000
            assert r['config']['encoder_size']==0 and r['config']['hidden_size']==64
            assert r['provenance']['verified_training_sources']==training_locked
            candidates=list((ROOT/'results/ring_cloud_artifacts').glob(f'ring-20261105-raw-*/{condition}_s{seed}.pt'))
            assert len(candidates)==1 and sha(candidates[0])==r['checkpoint_sha256']
            if condition!='grid_heading':
                for assay in ('memory','anchor'):
                    other=read(f'results/ring_heading/raw_memory_evaluation/{assay}_{condition}_s{seed}.json')
                    assert other['checkpoint_sha256']==r['checkpoint_sha256']
            raw_records.append({'seed':seed,'condition':condition,'checkpoint_sha256':r['checkpoint_sha256']})
    exact=[]
    signs8=np.array(list(itertools.product((-1.,1.),repeat=8)))
    expected_effects={
        'tracking_pgc_minus_grid_deg':[r['predictive']['same_region_native_mae_deg']-r['grid']['same_region_native_mae_deg'] for r in raw['per_seed']],
        'removal_minus_native_deg':[r['predictive']['ordinary_minus_native_mae_deg'] for r in raw['per_seed']],
        'transfer_pgc_minus_grid_deg':[r['predictive']['new_places_native_mae_deg']-r['grid']['new_places_native_mae_deg'] for r in raw['per_seed']]}
    for label,test in raw['contrasts'].items():
        value=np.array(test['paired_values']);assert len(value)==8
        np.testing.assert_array_equal(value,expected_effects[label])
        p=np.mean(np.abs((signs8*value).mean(axis=1))>=abs(value.mean())-1e-12)
        assert p==test['p_exact_sign_flip'];exact.append((p,label))
        half=t.ppf(.975,7)*value.std(ddof=1)/np.sqrt(8)
        np.testing.assert_allclose(test['ci95'],[value.mean()-half,value.mean()+half],atol=1e-12)
    previous=0
    for i,(p,label) in enumerate(sorted(exact)):
        adjusted=min(1,max(previous,p*(len(exact)-i)));previous=adjusted
        assert adjusted==raw['contrasts'][label]['p_holm_three_secondary']
    result={'ready':True,'audit_source_sha256':sha(__file__),
        'geometry':{'n_models':len(records),'all_frozen_sources_match':True,
                    'all_primary_values_recomputed_from_saved_derivatives':True,
                    'exact_test_and_interval_verified':True,'models':records},
        'raw':{'n_models':len(raw_records),'all_checkpoints_and_memory_anchor_identities_match':True,
               'three_exact_tests_and_intervals_verified':True,'models':raw_records}}
    out=ROOT/'docs/research/ring_final_cohort_audit.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'ready':True,'geometry_models':len(records),'raw_models':len(raw_records),
        'geometry_inference':inference,'raw_contrasts':raw['contrasts'],'output':str(out)},indent=2))


if __name__=='__main__':main()
