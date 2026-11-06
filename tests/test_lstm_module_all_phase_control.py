import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('all_phase',Path(__file__).resolve().parents[1]/'scripts/lstm_module_all_phase_control.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_other_known_phase_is_explained_but_unexplained_memory_is_retained():
    n=8;state=np.zeros((n,10));state[:,0]=np.linspace(-1,1,n)
    phase=np.broadcast_to(np.eye(10)[:,:8],(n,10,8)).copy()
    normals=np.zeros((n,10,2));normals[:,2,0]=1;normals[:,9,1]=1
    result=module.compare_known_phases(state,state,phase,phase,normals,normals,0,.02)
    assert np.allclose(np.array(result['extra_phase_explained_fraction_final'])[:,0],1)
    assert np.allclose(np.array(result['extra_phase_explained_fraction_final'])[:,1],0)
    assert result['per_direction'][1]['eight_phase_normal_gain_median']==1
    assert result['per_direction'][0]['coefficient_locality_pass_fraction']==1
    phase[:,:,2]*=1e-4
    large=module.compare_known_phases(state,state,phase,phase,normals,normals,0,.02)
    assert large['per_direction'][0]['coefficient_locality_pass_fraction']==0


def test_cell_centroid_direction_keeps_hidden_block_exactly_unchanged():
    rng=np.random.default_rng(8);surface=rng.normal(size=(32,512));selected=surface[:8]
    tangent=rng.normal(size=(8,512,2));scale=np.r_[np.ones(256)*.2,np.ones(256)*7]
    direction=module.cell_centroid_normal(surface,selected,tangent,scale)
    assert np.array_equal(direction[:,:256],np.zeros((8,256)))
    np.testing.assert_allclose(np.einsum('nd,ndk->nk',direction/scale,tangent/scale[None,:,None]),0,atol=1e-12)


def test_all_phase_parameterization_changes_only_the_corresponding_module():
    pop=module.torus.one.GridPopulation(pgc_frac=.33,seed=3)
    phase=np.zeros((2,4,2));phase[1,2,1]=.3
    code,derivative=module.all_phase_codes(pop,np.array([.2,.1]),np.array([1.,0.]),phase)
    np.testing.assert_array_equal(code[0,pop.module!=2],code[1,pop.module!=2])
    for k in range(8):assert np.all(derivative[:,pop.module!=k//2,k]==0)


def test_large_selected_phase_compensation_cannot_hide_behind_small_extra_coefficients():
    n=8;state=np.zeros((n,10));state[:,0]=np.linspace(-1,1,n)
    phase=np.broadcast_to(np.eye(10)[:,:8],(n,10,8)).copy()
    phase[:,0,2]=100.
    normal=np.zeros((n,10,1));normal[:,2,0]=1.
    result=module.compare_known_phases(state,state,phase,phase,normal,normal,0,.02)
    coefficients=np.array(result['orthogonalized_eight_phase_coefficients_initial'])
    np.testing.assert_allclose(coefficients[:,2,0],.02)
    np.testing.assert_allclose(coefficients[:,0,0],-2.)
    assert result['per_direction'][0]['coefficient_locality_pass_fraction']==0
    # A full response along the actual extra-phase vector has a small total fit;
    # retain this distinct quantity rather than concealing the decomposition's compensation.
    normal[:,0,0]=100.
    full=module.compare_known_phases(state,state,phase,phase,normal,normal,0,.02)
    assert full['per_direction'][0]['coefficient_locality_pass_fraction']==0
    assert full['per_direction'][0]['total_fit_locality_pass_fraction']==1
