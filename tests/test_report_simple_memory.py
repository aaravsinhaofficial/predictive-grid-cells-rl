"""Private synthetic producer summaries only; never load trained model outputs."""
import copy
import json

import numpy as np
import pytest

from scripts import report_simple_memory as report


def evaluation(score):
    scores=dict(blind=score,seen=score+5,blind_reset_onset=score-8,seen_reset_onset=score+3,
        blind_reset_each=score-12,seen_reset_each=score+1,blind_current_grid=score-3,seen_current_grid=score+4)
    return dict(variants={name:dict(mean_score=value,rows=[{'private_fixture':True}]*128) for name,value in scores.items()},
        reset_contrasts=dict(onset=dict(blind_reset_cost=8,seen_reset_cost=2,additional_reset_cost_in_blackout=6),
                            each=dict(blind_reset_cost=12,seen_reset_cost=4,additional_reset_cost_in_blackout=8)),
        current_predictive_input_benefit=dict(blind=3,seen=1))


def cohort():
    rows=[]
    for condition,gap,seed in report.COHORT:
        p=int(condition=='pgc33');b=int(gap==60)
        score=100+seed+p*(2+seed*.5)+b*(3+.2*seed)+p*b*5
        rows.append(dict(condition=condition,task_max_gap=gap,seed=seed,stages={'evaluation':evaluation(score)}))
    return rows


def dynamics(index,*,passes=True,recovery=True,qualified=True):
    metric=dict(spectral_radius=.9,largest_singular_value=1.2,top_eigenvector_tangent_overlap=.3,
        top_singular_vector_tangent_overlap=.2,singular_values=[1.2,.8,.4,.1],ordered_eigenvalue_modulus=[.9,.7,.4,.1])
    regime=dict(conditional_ring_candidate=passes,tested_normal_recovery_confirmed=recovery,
        gates={name:dict(radius_retains5percent=passes,radial_starts_within_hidden_bounds=True,
                        radial_directions_nondegenerate=True) for name in ('raw','relative_rms')},
        local_spectra={t:{name:copy.deepcopy(metric) for name in ('raw','relative_rms')} for t in ('0','30')})
    return dict(index=index,episode_id=index+4,global_step=100+index,qualified=qualified,
        regimes={name:copy.deepcopy(regime) for name in ('constant','forward_backward')},
        native_occupancy={'qualified':False,'broad_native_support':False})


def intervention(index,phase,behavior,*,qualified=True):
    return dict(index=index,episode_id=index+4,global_step=100+index,qualified=qualified,
        family={'qualified':True},
        cue_repair=dict(rows={'15':{'correct':{'none':{'correct_reference':dict(defined=True,radius=1.,coverage=.95)}}}},
            contrasts={'15':{alt+'_minus_correct':dict(donor_mean=phase,donor_minus_no_edit=phase/2)
                             for alt in ('ordinary','wrong')}}),
        behavior=dict(validation_qualified=qualified,tapes=4,horizon=150,
            contrasts={'correct_minus_'+alt:dict(donor_mean=behavior,donor_minus_no_edit=behavior/2) for alt in ('ordinary','wrong')},
            records=[dict(qualified=True,decisions=150,blind_steps=60,first_hit_censored=False)]*100))


def test_four_seed_exact_resolution_and_small_sample_descriptive_interval():
    same=report.paired_estimate({s:2. for s in report.SEEDS})
    assert same['n_pairs']==4 and same['exact_signflip_p']==.125
    assert same['minimum_two_sided_exact_p']==.125 and same['descriptive_ci95']==[2.,2.]
    variable=report.paired_estimate({1:1,2:2,3:3,4:4})
    assert variable['mean']==2.5
    assert variable['descriptive_ci95'][0]<variable['mean']<variable['descriptive_ci95'][1]
    one=report.paired_estimate({1:4})
    assert one['descriptive_ci95'] is None and one['exact_signflip_p']==1
    empty=report.paired_estimate({})
    assert empty['mean'] is None and empty['n_pairs']==0


def test_factorial_effects_are_within_seed_not_pooled_episode_differences():
    source=cohort();first=report.aggregate(source)
    interaction=first['paired_seed_effects']['score_blind']['population_by_training_interaction']
    np.testing.assert_allclose(interaction['seed_effects'],[5]*4,atol=1e-12)
    assert interaction['n_pairs']==4 and interaction['exact_signflip_p']==.125
    # Multiplying repeated-episode/tape rows creates no additional trained seed.
    for row in source:
        row['stages']['evaluation']['variants']['blind']['rows']*=100
    second=report.aggregate(source)
    assert first['paired_seed_effects']==second['paired_seed_effects']
    assert first['models'][0]['metrics']['onset_additional_reset_cost_in_blackout']==6


def test_missing_pair_is_not_zero_imputed_or_given_four_seed_p_value():
    source=[r for r in cohort() if not(r['condition']=='pgc33' and r['task_max_gap']==60 and r['seed']==4)]
    summary=report.aggregate(source)
    effect=summary['paired_seed_effects']['score_blind']['pgc_minus_grid_blackout_training']
    assert effect['n_pairs']==3 and effect['missing_seeds']==[4] and effect['seed_effects'][-1] is None
    assert effect['minimum_two_sided_exact_p']==.25
    assert len(summary['models'])==16 and summary['stage_available_models']['evaluation']==15
    missing=[r for r in summary['models'] if r['model']=='pgc33_gap60_s4'][0]
    assert missing['context_counts']['selected_contexts'] is None
    assert missing['metrics']['constant_ring_count'] is None
    assert summary['status']=='partial'


def test_repair_and_paired_behavior_require_confirmed_recovery_but_broad_behavior_does_not():
    ds=[dynamics(0),dynamics(1,recovery=False),dynamics(2,passes=False,recovery=False),dynamics(3,qualified=False)]
    ins=[intervention(0,2,3),intervention(1,100,7),intervention(2,1000,21),intervention(3,9999,9999,qualified=False)]
    model=report.extract_model('pgc33',60,1,{'dynamics':{'contexts':ds},'interventions':{'contexts':ins}})
    assert model['metrics']['phase_repair_ordinary_donor_mean']==2
    assert model['metrics']['paired_geometry_behavior_ordinary_donor_mean']==3
    assert model['metrics']['native_behavior_ordinary_donor_mean']==pytest.approx(31/3)
    assert model['context_counts']['paired_geometry_behavior']==1
    assert model['context_counts']['technically_valid_native_behavior']==3
    assert model['context_counts']['constant']['passed']==2
    assert model['context_counts']['constant']['failed']==1
    assert model['context_counts']['constant']['unqualified']==1
    assert not any(c['passive_broad_support'] for c in model['contexts'])


def test_contexts_get_equal_weight_and_identity_mismatch_cannot_qualify_repair():
    ds=[dynamics(0),dynamics(1)]
    ins=[intervention(0,2,3),intervention(1,10,13)]
    ins[0]['behavior']['records']*=50
    model=report.extract_model('pgc33',60,1,{'dynamics':{'contexts':ds},'interventions':{'contexts':ins}})
    assert model['metrics']['native_behavior_ordinary_donor_mean']==8
    assert model['metrics']['phase_repair_ordinary_donor_mean']==6
    ins[1]['episode_id']=999
    model=report.extract_model('pgc33',60,1,{'dynamics':{'contexts':ds},'interventions':{'contexts':ins}})
    assert model['metrics']['phase_repair_ordinary_donor_mean']==2
    assert model['metrics']['paired_geometry_behavior_ordinary_donor_mean']==3
    assert model['metrics']['native_behavior_ordinary_donor_mean']==8


def test_raw_cue_sensitivity_uses_same_contexts_and_both_donor_signs():
    d=dynamics(0);i=intervention(0,2,3)
    for cue,value in [('ordinary',20),('correct',5),('wrong_plus',25),('wrong_minus',35)]:
        i['cue_repair']['rows']['15'].setdefault(cue,{})
        for edit in ('none','plus30','minus30'):
            i['cue_repair']['rows']['15'][cue].setdefault(edit,{})
            i['cue_repair']['rows']['15'][cue][edit]['raw_correct_reference']={'phase_mae_deg':value if edit!='none' else value/2}
    model=report.extract_model('pgc33',60,1,{'dynamics':{'contexts':[d]},'interventions':{'contexts':[i]}})
    assert model['metrics']['phase_raw_sensitivity_ordinary_donor_mean']==15
    assert model['metrics']['phase_raw_sensitivity_wrong_donor_mean']==25
    assert model['metrics']['phase_raw_sensitivity_ordinary_donor_minus_no_edit']==7.5


def test_nonphysical_or_collapsed_reference_stays_unqualified():
    d=dynamics(0);d['regimes']['constant']['conditional_ring_candidate']=False
    d['regimes']['constant']['gates']['raw']['radial_starts_within_hidden_bounds']=False
    assert report.ring_status(d)=='unqualified'
    assert report.ring_status(None)=='missing'
    i=intervention(0,3,4);i['cue_repair']['rows']['15']['correct']['none']['correct_reference']['radius']=1e-8
    card=report.context_card(dynamics(0),i,0)
    assert not card['phase_repair_qualified'] and card['technically_valid_native_behavior']


def test_zero_context_is_complete_scientific_data_but_not_an_imputed_ring_count():
    model=report.extract_model('gc100',0,1,{'dynamics':{'contexts':[]},'interventions':{'contexts':[]}})
    assert model['availability']['dynamics']=='available'
    assert model['context_counts']['selected_contexts']==0 and model['context_counts']['not_selected']==4
    assert model['metrics']['constant_ring_count'] is None
    assert model['metrics']['phase_repair_ordinary_donor_mean'] is None


def test_duplicate_or_unexpected_models_are_rejected():
    with pytest.raises(ValueError):report.aggregate(cohort()+[cohort()[0]])
    with pytest.raises(ValueError):report.aggregate([dict(condition='pgc33',task_max_gap=60,seed=9,stages={})])


def pair_fixture(tmp_path):
    pair=tmp_path/'ring-20261105-simplemem-analysis-native-s1';pair.mkdir()
    runs=[]
    for condition in ('gc100','pgc33'):
        folder=pair/condition;folder.mkdir()
        stages={}
        for stage,producer in [('evaluation','behavior'),('activity','activity'),('dynamics','dynamics'),('interventions','interventions')]:
            relative=report.DEFAULTS[stage];path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'private_fixture':True,'contexts':[]}))
            stages[producer]={'validation':{'summary_path':relative}}
        runs.append(dict(run=condition,endpoint=dict(condition=condition,training_task_max_gap=0,training_seed=1),stages=stages))
    (pair/'pipeline.json').write_text(json.dumps({'runs':runs}))
    return pair


def test_pair_discovery_requires_pipeline_verifier_before_using_summaries(tmp_path):
    pair=pair_fixture(tmp_path);calls=[]
    def verified(path):calls.append(path);return dict(complete=True,private_fixture=True)
    inputs,proof,plots,_=report.load_pairs([pair],verify=verified)
    assert calls==[pair] and len(inputs)==2
    assert all(r['availability']['evaluation']=='available' for r in inputs)
    assert proof['pair_packages'][0]['status']=='verified'
    assert ('gc100_gap0_s1','dynamics') in plots
    def rejected(path):raise ValueError('private hash mismatch')
    invalid,proof,_,_=report.load_pairs([pair],verify=rejected)
    assert all(r['availability']['evaluation']=='invalid' for r in invalid)
    assert all(r['stages']=={} for r in invalid)
    assert proof['pair_packages'][0]['status']=='invalid'


def test_pair_location_identity_and_missing_packages_are_explicit(tmp_path):
    pair=pair_fixture(tmp_path)
    data=json.loads((pair/'pipeline.json').read_text());data['runs'][1]['endpoint']['training_seed']=2
    (pair/'pipeline.json').write_text(json.dumps(data))
    rows,_,_,_=report.load_pairs([pair],verify=lambda p:{'complete':True})
    assert all(r['availability']['evaluation']=='invalid' for r in rows)
    missing=tmp_path/'ring-20261105-simplemem-analysis-blackout-s4'
    rows,_,_,_=report.load_pairs([missing],verify=lambda p:pytest.fail('Missing pair must not verify'))
    assert len(rows)==2 and all(r['availability']['activity']=='missing' for r in rows)
    with pytest.raises(report.ArtifactError):report.load_pairs([tmp_path/'unprescribed-pair'],verify=lambda p:{})


def test_private_figures_and_readme_use_actual_summary_values_without_claiming_significance(tmp_path):
    source=cohort();summary=report.aggregate(source)
    # No generated fixtures leave pytest's private temporary directory.
    gallery=report.make_figures(summary,source,{},tmp_path,context_figures=False)
    assert gallery==[]
    assert {p.stem for p in (tmp_path/'figures').glob('*.png')}=={
        'behavior','activity','angular_information','mechanism','local_spectra'}
    assert all(p.stat().st_size>1000 for p in (tmp_path/'figures').glob('*.png'))
    text=report.readme(summary,gallery)
    assert '0.125' in text and 'descriptive' in text
    assert '16/16 navigation evaluations' in text and '0/16 dynamics analyses' in text
    assert 'statistically significant' not in text and 'proved' not in text
    json.dumps(summary,allow_nan=False)


def test_actual_private_producer_schemas_and_every_context_gallery(tmp_path):
    from scripts import analyze_simple_memory_activity as activity
    from scripts import analyze_simple_memory_dynamics as dynamics_producer
    from tests.test_analyze_simple_memory_activity import synthetic
    from tests.test_analyze_simple_memory_dynamics import circle,ring_transition
    x,target,episode,visible,known=synthetic(n_episodes=10,rows=20)
    angle=np.arctan2(target[:,0,1],target[:,0,0])
    arrays=dict(h=x.astype(np.float32),c=(x*.7).astype(np.float32),episode_id=episode,
        known=known,blind=~visible,phase_age=np.tile(np.arange(1,11),20),body_heading=angle,
        travel_dir=target[:,1],pos=np.zeros((len(x),2)),goal_pos=np.column_stack((np.cos(angle*.5),np.sin(angle*.5))))
    act,_=activity.analyze_arrays(arrays,np.arange(24).reshape(6,4))
    curve,basis=circle();regimes={};numeric={}
    for regime in ('constant','forward_backward'):
        row,_,saved=dynamics_producer.analyze_regime(ring_transition(basis),curve,np.ones(4),regime,arc_probes=24,anchors=4)
        row.update(conditional_ring_candidate=False,tested_normal_recovery_confirmed=False)
        regimes[regime]=row
        numeric.update({regime+'_'+k:v for k,v in saved.items()})
    numeric['constant_spectra_t30_eigenvalues']=np.linalg.eigvals(np.array([[.8,-.1],[.1,.8]]))
    artifact=tmp_path/'private_dynamics.npz';np.savez_compressed(artifact,**numeric)
    context=dict(index=0,qualified=True,episode_id=4,global_step=100,regimes=regimes,
                 arrays={'path':artifact.name,'sha256':report.sha(artifact)})
    inputs=[dict(condition='gc100',task_max_gap=0,seed=1,
        stages={'activity':act,'dynamics':{'contexts':[context]},'evaluation':evaluation(10)})]
    summary=report.aggregate(inputs)
    model=summary['models'][0]
    assert model['pca_counts']['known_blind']['rows']==100
    assert len(model['curves']['pca_all_total'])==4
    assert len(model['readouts']['h']['body_heading']['known_visible']['null_mae'])==20
    gallery=report.make_figures(summary,inputs,{('gc100_gap0_s1','dynamics'):tmp_path},tmp_path/'report')
    assert len(gallery)==2 and {g['regime']for g in gallery}=={'constant','forward_backward'}
    assert all((tmp_path/'report'/g['path']).is_file()for g in gallery)
