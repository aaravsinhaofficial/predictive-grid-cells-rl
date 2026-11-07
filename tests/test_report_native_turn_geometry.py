"""Independent saved-array reporter arithmetic; no trained policy execution."""
from pathlib import Path
import sys
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import native_turn_geometry as study
import report_native_turn_geometry as report
from test_native_turn_geometry_controls import two_section_system


def test_independent_dense_geometry_gates_match_saved_method():
    theta=np.arange(128)*2*np.pi/128
    reference=np.column_stack((np.cos(theta),np.sin(theta),.2*np.cos(2*theta)))
    for offset in (0.,12.):
        first=reference+offset
        for moved in (reference+offset,.8*reference+offset,np.roll(reference,-7,0)+offset,reference+offset+1):
            for center in (False,True):
                assert report.independent_gates(first,moved,center)==study.geometry_screen(first,moved,center)['gates']


def test_saved_normal_gains_and_bounded_metadata_are_independently_checked(tmp_path):
    circle,_,forward,backward,_,_=two_section_system()
    transition=lambda z,t:(forward,backward)[t%2](z)
    for method in (None,'shared','clip_equalize'):
        result,arrays=study.finite_probe(transition,circle.numpy(),np.ones(4),lambda z:z,(0,1),2,bounded=method)
        name=f'{method}.npz';np.savez_compressed(tmp_path/name,**arrays)
        row=report.audit_normal(tmp_path,result,name)
        assert row['physical_status']=='pass'and row['directions']==20
    arrays['normal_gain']=arrays['normal_gain'].copy();arrays['normal_gain'][0,0,0]+=.1
    np.savez_compressed(tmp_path/'corrupt.npz',**arrays)
    import pytest
    with pytest.raises(AssertionError):report.audit_normal(tmp_path,result,'corrupt.npz')


def test_bounded_audit_recomputes_rms_metric_and_rejects_direction_metadata_corruption(tmp_path):
    import copy
    import pytest
    circle,_,forward,backward,_,_=two_section_system()
    curve=circle.numpy();half=curve.shape[1]//2
    scale=np.repeat(np.sqrt(np.mean(curve.reshape(-1,2,half)**2,axis=(0,2))),half)
    transition=lambda z,t:(forward,backward)[t%2](z)
    for method in ('shared','clip_equalize'):
        result,arrays=study.finite_probe(transition,curve,scale,lambda z:z,(0,1),2,bounded=method)
        name=f'rms_{method}.npz';np.savez_compressed(tmp_path/name,**arrays)
        assert report.audit_normal(tmp_path,result,name,'h_c_RMS')['physical_status']=='pass'
        for field in ('actual_scaled_norms','actual_tangent_fraction','actual_normal_fraction','actual_intended_cosine'):
            corrupt={k:v.copy()for k,v in arrays.items()};corrupt[field][0,0,0]+=.05
            np.savez_compressed(tmp_path/'bad_geometry.npz',**corrupt)
            with pytest.raises(AssertionError):report.audit_normal(tmp_path,result,'bad_geometry.npz','h_c_RMS')
        bad=copy.deepcopy(result);bad['perturbation_design']['block_purity_exact']=False
        with pytest.raises(AssertionError):report.audit_normal(tmp_path,bad,name,'h_c_RMS')
        bad=copy.deepcopy(result);bad['perturbation_design']['bounded_target_norms'][0]*=1.1
        with pytest.raises(AssertionError):report.audit_normal(tmp_path,bad,name,'h_c_RMS')
        with pytest.raises(AssertionError):report.audit_normal(tmp_path,result,name,'raw')
