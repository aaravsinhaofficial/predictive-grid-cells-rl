import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('module_torus',Path(__file__).resolve().parents[1]/'scripts/lstm_module_phase_torus.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def clifford(n=24):
    u,v=np.meshgrid(np.arange(n)*2*np.pi/n,np.arange(n)*2*np.pi/n,indexing='ij')
    u,v=u.ravel(),v.ravel()
    state=np.c_[np.cos(u),np.sin(u),np.cos(v),np.sin(v)]
    phase=np.zeros((n*n,4,2))
    phase[:,0,0]=-np.sin(u);phase[:,1,0]=np.cos(u)
    phase[:,2,1]=-np.sin(v);phase[:,3,1]=np.cos(v)
    normal=np.zeros((n*n,4,2))
    normal[:,:2,0]=state[:,:2];normal[:,2:,1]=state[:,2:]
    return state,phase,normal


def test_true_torus_geometry_and_shrinking_torus_rejection():
    state,phase,normal=clifford()
    positive=module.geometry_metrics(state,state,24)
    assert all(positive['centering']['noncentered']['gates'].values())
    shrinking=module.geometry_metrics(state,.8*state,24)
    assert not shrinking['centering']['centered']['gates']['invariance_p90_at_most5percent_radius']
    recovered=module.differential_metrics(state,state,phase,phase,normal,.2*normal,.02)
    assert all(recovered['gates'].values())
    decaying=module.differential_metrics(state,.8*state,phase,.8*phase,normal,.8*normal,.02)
    assert not decaying['gates']['normal_phase_ratio_median_at_most_half']


def test_second_phase_is_not_misclassified_as_transverse_error():
    state,phase,normal=clifford()
    residual,_=module.normal_component(phase[:,:,1:],phase)
    assert np.max(np.abs(residual))<1e-12
    residual,_=module.normal_component(normal,phase)
    np.testing.assert_allclose(residual,normal,atol=1e-12)
    drifted=module.geometry_metrics(state,state+np.array([2.,-1.,3.,.5]),24)
    assert all(drifted['centering']['centered']['gates'].values())
    assert not drifted['centering']['noncentered']['gates']['invariance_p90_at_most5percent_radius']


def test_two_phase_codes_and_analytic_derivatives_match_both_periods():
    pop=module.one.GridPopulation(pgc_frac=.33,seed=8)
    phases=np.array([[.1,.3],[.6,.7]])
    pickup=np.array([.3,.2]);arrival=np.array([1.,0.])
    reference,derivative,_=module.torus_code(pop,pickup,arrival,1,phases)
    for axis in range(2):
        delta=np.eye(2)[axis]
        loop=module.torus_code(pop,pickup,arrival,1,phases+delta)[0]
        np.testing.assert_allclose(loop,reference,atol=1e-12)
        fd=(module.torus_code(pop,pickup,arrival,1,phases+1e-6*delta)[0]-module.torus_code(pop,pickup,arrival,1,phases-1e-6*delta)[0])/2e-6
        np.testing.assert_allclose(fd,derivative[:,:,axis],rtol=1e-5,atol=1e-8)


def test_persistent_direction_cannot_hide_in_pooled_median_and_silent_mode_cannot_reappear():
    state,phase,normal=clifford()
    six=np.concatenate((normal,normal,normal),axis=2)
    final=.1*six.copy();final[:,:,-1]=six[:,:,-1]
    result=module.differential_metrics(state,state,phase,phase,six,final,.02)
    assert result['gates']['normal_gain_median_at_most_half']
    assert not result['gates']['all_normal_directions_qualify_and_recover']
    six[:,:,-1]=0.;final=.1*six.copy()
    # A genuinely silent spectator factor stays silent and does not invalidate recovery.
    silent=module.differential_metrics(state,state,phase,phase,six,final,.02)
    assert all(silent['gates'].values())
    # Later purely tangential response is still an observable leak from a silent mode.
    final[:,:,-1]=phase[:,:,0]
    leaking=module.differential_metrics(state,state,phase,phase,six,final,.02)
    assert leaking['emerging_observable_fraction']>0
    assert not leaking['gates']['no_initially_silent_later_visible_kicks']
