import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('module_phase',Path(__file__).resolve().parents[1]/'scripts/lstm_module_phase_ring.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_both_primitive_periods_preserve_native_and_ordinary_codes_and_tangents():
    population=module.GridPopulation(pgc_frac=.33,seed=7)
    controls=module.input_controls(population,np.array([.27,-.13]),np.array([1.,0.]))
    assert len(controls['families'])==8
    for row in controls['families']:
        assert row['module_size']==128
        assert row['native_predictive_offsets']['endpoint_code_max_error']<1e-12
        assert row['ordinary_same_maps']['endpoint_code_max_error']<1e-12


def test_only_selected_interleaved_module_changes_and_predictive_offsets_remain():
    population=module.GridPopulation(pgc_frac=.33,seed=5)
    pickup=np.array([.2,.4]);arrival=np.array([1.,0.])
    native,_,_=module.module_phase_code(population,pickup,arrival,2,1,[0.,.371])
    ordinary,_,_=module.module_phase_code(population,pickup,arrival,2,1,[0.,.371],0.)
    selected=population.module==2
    assert np.array_equal(native[0,~selected],native[1,~selected])
    assert np.linalg.norm(native[0,selected]-native[1,selected])>1
    assert np.array_equal(native[:,population.gc_idx],ordinary[:,population.gc_idx])
    assert np.linalg.norm(native[:,population.pgc_idx]-ordinary[:,population.pgc_idx])>1


def test_shrinking_circle_fails_invariance_despite_perfect_input_order():
    phase=np.arange(128)*2*np.pi/128
    circle=np.c_[np.cos(phase),np.sin(phase)]
    result=module.geometry_precheck(circle,.5*circle,circle,np.ones(2))
    assert result['gates']['projected_phase_coverage_at_least_80percent']
    assert not result['gates']['fixed_curve_residual_p90_at_most_5percent']
    assert not result['passes_necessary_gates']
