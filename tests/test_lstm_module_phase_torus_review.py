"""Independent geometry/derivative-frame checks for the torus assay."""
import numpy as np
from scripts import lstm_module_phase_torus as torus


def sample_clifford(n):
    u,v=np.meshgrid(np.arange(n)*2*np.pi/n,np.arange(n)*2*np.pi/n,indexing='ij')
    u,v=u.ravel(),v.ravel()
    z=np.c_[np.cos(u),np.sin(u),np.cos(v),np.sin(v)]
    tangent=np.zeros((n*n,4,2));normal=np.zeros_like(tangent)
    tangent[:,0,0]=-np.sin(u);tangent[:,1,0]=np.cos(u)
    tangent[:,2,1]=-np.sin(v);tangent[:,3,1]=np.cos(v)
    normal[:,:2,0]=z[:,:2];normal[:,2:,1]=z[:,2:]
    return z,tangent,normal


def test_periodic_seam_triangle_preserves_local_coordinates_without_bridge():
    n=8;z,_,_=sample_clifford(n)
    triangles,uv=torus.mesh_indices(n)
    # The last cell touches both seams. Its local coordinates run toward1,
    # while vertex indices correctly wrap to0; interpolation must stay local.
    selected=np.array([2*(n*n-1),2*(n*n-1)+1])
    weights=np.array([.5,.2,.3])
    points=np.einsum('tjd,j->td',z[triangles[selected]],weights)
    expected=np.mod(np.einsum('tjd,j->td',uv[selected],weights),1)
    distance,actual=torus.nearest_torus_mesh(points,z,n)
    np.testing.assert_allclose(distance,0,atol=1e-7)
    np.testing.assert_allclose((actual-expected+.5)%1-.5,0,atol=1e-7)
    edges=z[triangles[:,1:]]-z[triangles[:,0,None]]
    assert np.linalg.norm(edges,axis=-1).max() <= np.sqrt(2)*2*np.sin(np.pi/n)+1e-12


def test_advected_two_phase_projection_tracks_rotated_ambient_frame():
    z,phase,normal=sample_clifford(8)
    rotation=np.linalg.qr(np.random.default_rng(7).normal(size=(4,4)))[0]
    evolved=np.einsum('ab,nbc->nac',rotation,phase)
    normal_final=.2*np.einsum('ab,nbc->nac',rotation,normal)
    result=torus.differential_metrics(z,z@rotation.T,phase,evolved,normal,normal_final,.02)
    np.testing.assert_allclose(result['phase_minimum_gains'],1,atol=1e-12)
    np.testing.assert_allclose(np.asarray(result['final_normal_norms'])/np.asarray(result['initial_normal_norms']),.2,atol=1e-12)
    assert all(result['gates'].values())


def test_collapsed_second_phase_fails_rank_gate_without_nan():
    z,phase,normal=sample_clifford(8)
    phase[:,:,1]=0
    result=torus.differential_metrics(z,z,phase,phase,normal,.2*normal,.02)
    assert result['phase_rank2_fraction']==0
    assert not result['gates']['phase_rank2_in90percent']
    assert np.isfinite(result['phase_minimum_gains']).all()


def test_hidden_and_cell_normals_remain_block_specific_after_projection():
    rng=np.random.default_rng(33)
    state=rng.normal(size=(8,512));tangent=rng.normal(size=(8,512,2))
    scale=np.r_[np.ones(256)*.2,np.ones(256)*4]
    normal,_=torus.normal_directions(state,tangent,scale,71)
    np.testing.assert_allclose(normal[:,256:,4],0,atol=1e-12)
    np.testing.assert_allclose(normal[:,:256,5],0,atol=1e-12)
    scaled=normal/scale[None,:,None];phase=tangent/scale[None,:,None]
    np.testing.assert_allclose(np.einsum('ndi,ndj->nij',phase,scaled),0,atol=1e-10)


def test_initially_silent_kick_becoming_tangentially_visible_is_flagged():
    z,phase,normal=sample_clifford(8)
    initial=np.concatenate((normal,np.zeros_like(normal[:,:,:1])),axis=2)
    # The new third direction is invisible in the initial output window,
    # then changes the later observable ONLY along phase1. A normal-only
    # denominator guard would conceal this failure of the chosen factor.
    final=np.concatenate((.2*normal,phase[:,:,:1]),axis=2)
    result=torus.differential_metrics(z,z,phase,phase,initial,final,.02)
    assert result['emerging_observable_fraction']>0
    assert not result['gates']['no_initially_silent_later_visible_kicks']
    assert not result['per_direction'][-1]['passes']
