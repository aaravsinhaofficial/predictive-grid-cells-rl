"""Private analytic/physical controls; never loads a trained policy checkpoint."""
from pathlib import Path
import json
import sys
import math
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_turn_geometry_resolution as study
from test_native_turn_teacher_preflight import fixture
torch.set_num_threads(1)


def row(context,metric,failures=(study.SPACING,),regime='fb_B',branch='native'):
    gates={k:True for k in ('radius_RMS_at_least_1e_minus4',study.SPACING,'nonlocal_separation_above_5percent',
        'evolved_radius_retains_5percent','p90_invariance_residual_below_5percent','phase_coverage_80percent','phase_degree_one','phase_order_preserved')}
    gates.update({k:False for k in failures})
    return dict(context=context,prefix=0,kind='probe_glimpse',branch=branch,regime=regime,window_updates=3,
        evolution_updates=2,observation_offsets=[0,1],observable='all_hidden',metric=metric,centered=True,
        gates=gates,family=f'context{context}_{branch}',exact_sham_of=None)


def test_eligibility_is_exact_spacing_failure_and_keeps_metric_counterpart():
    rows=[row(0,m)for m in('raw','h_c_RMS')]
    rows += [row(1,'raw'),row(1,'h_c_RMS',(study.SPACING,'phase_degree_one'))]
    rows += [row(2,m,())for m in('raw','h_c_RMS')]
    rows += [row(3,m,(study.SPACING,'phase_order_preserved'))for m in('raw','h_c_RMS')]
    selected,ledger=study.eligibility(rows)
    assert {(r['context'],r['metric'])for r in selected}=={(c,m)for c in(0,1)for m in('raw','h_c_RMS')}
    assert ledger['source_rows']==8 and ledger['selected_rows']==4
    assert ledger['spacing_only_rows']==3 and ledger['mandatory_metric_control_rows']==1
    assert ledger['excluded_rows']==4


def test_exact_sham_copy_preserves_distinct_forward_backward_sections():
    originals=[];selected=[]
    for regime in('fb_B','fb_F'):
        source=row(0,'raw',regime=regime)
        originals.append({**source,'source_family':source['family'],'payload':regime})
        alias={**source,'branch':'ordinary','family':'context0_ordinary','exact_sham_of':source['family']}
        selected.extend([source,alias])
    result=study.attach_shams(originals,selected)
    assert len(result)==4
    for r in result:assert r['payload']==r['regime']


def test_large_cell_baseline_cannot_hide_changed_angular_hidden_pattern():
    phase=np.arange(128)*2*np.pi/128
    reference=np.column_stack((.8+1e-4*np.cos(phase),.7+1e-4*np.sin(phase),np.full(128,1e6),np.full(128,1e6)))
    same=study.shared_state_agreement(reference.copy(),reference)
    assert same['passes']and same['blocks']['cell']['angular_RMS']==0
    changed=reference.copy();changed[:,0]+=1e-6*np.cos(phase)
    # Per-coordinate relative precision alone still misses this small angular signal.
    audit=study.shared_state_agreement(changed,reference)
    assert audit['blocks']['hidden']['coordinatewise_pass']
    assert not audit['blocks']['hidden']['angular_shape_pass']and not audit['passes']
    large_error=reference.copy();large_error[:,0]+=.01
    assert not study.shared_state_agreement(large_error,reference)['blocks']['hidden']['coordinatewise_pass']
    changed_cell=reference.copy();changed_cell[:,2]+=1e-4*np.sin(phase)
    cell_audit=study.shared_state_agreement(changed_cell,reference)
    assert cell_audit['blocks']['cell']['coordinatewise_pass']
    assert not cell_audit['blocks']['cell']['angular_shape_pass']and not cell_audit['passes']
    rounding=reference.copy();rounding[:,0]+=2e-8*np.cos(phase)
    assert study.shared_state_agreement(rounding,reference)['passes']


def steep_ring(n,sharpness=5):
    theta=np.arange(n)*2*np.pi/n
    phase=2*np.arctan2(sharpness*np.sin(theta/2),np.cos(theta/2))
    basis=np.array([[1.,0.],[0.,1.],[.6,.2],[-.2,.6]])
    basis,_=np.linalg.qr(basis)
    return .6*np.stack((np.cos(phase),np.sin(phase)),axis=1)@basis.T,basis


def ring_transition(basis,decay=.3):
    basis=torch.as_tensor(basis,dtype=torch.float64)
    def transition(z,step):
        coordinates=z@basis;r=coordinates.norm(dim=1,keepdim=True)
        plane=coordinates@basis.T
        return (coordinates*(.6+decay*(r-.6))/r)@basis.T+decay*(z-plane)
    return transition


def test_true_steep_ring_is_recovered_by_sampling_not_by_fitting(tmp_path):
    curve,basis=steep_ring(1024);transition=ring_transition(basis)
    states=study.old.trajectory(transition,curve,2)
    screens={}
    for n in(128,256,512,1024):
        ids=study.nested_indices(1024,n)
        screens[n]=study.old.geometry_screen(states[0][ids],states[2][ids])
    assert not screens[128]['gates'][study.SPACING]
    assert not screens[256]['gates'][study.SPACING]
    assert screens[512]['necessary_geometry_pass']and screens[1024]['necessary_geometry_pass']
    assert study.convergence(states[0][::2],states[2][::2],states[0],states[2])['passes']
    record=dict(observation_offsets=[0],evolution_updates=2)
    for n in(512,1024):
        probe=study.test_recovery(transition,curve[study.nested_indices(1024,n)],np.ones(4),lambda z:z,record,tmp_path,f'ring{n}')
        assert probe['physical_status']=='pass'and probe['passes']
        assert probe['nominal']['perturbation_design']['anchors']==8
        assert probe['nominal']['perturbation_design']['names'][-1]=='tangent'


def test_shrinking_plane_keeps_its_curve_but_fails_selective_recovery(tmp_path):
    curve,_=steep_ring(1024)
    transition=lambda z,step:.995*z
    states=study.old.trajectory(transition,curve,2)
    assert study.old.geometry_screen(states[0],states[2])['necessary_geometry_pass']
    probe=study.test_recovery(transition,curve,np.ones(4),lambda z:z,
        dict(observation_offsets=[0],evolution_updates=2),tmp_path,'plane')
    assert probe['physical_status']=='failed_recovery'and not probe['passes']


def test_finest_density_alone_cannot_qualify_an_unresolved_circle():
    curve,basis=steep_ring(1024,sharpness=8)
    states=study.old.trajectory(ring_transition(basis),curve,2)
    r=dict(observable='full_state',evolution_updates=2,observation_offsets=[0],metric='raw',centered=False)
    records,check,qualified=study.density_geometry(None,states,r,np.arange(1024)*2*np.pi/1024,np.zeros(1024))
    assert records['1024']['necessary_geometry_pass']
    assert not records['512']['gates'][study.SPACING]
    assert not qualified


def test_action_locked_point_erases_phase_and_fails_recovery(tmp_path):
    curve,_=steep_ring(1024)
    transition=lambda z,step:torch.zeros_like(z)
    probe=study.test_recovery(transition,curve,np.ones(4),lambda z:z,
        dict(observation_offsets=[0],evolution_updates=2),tmp_path,'point')
    assert not probe['nominal']['geometry']['False']['necessary_geometry_pass']
    assert not probe['nominal']['all_normal_directions_pass']
    assert not probe['passes']


def test_timeout_preserves_complete_cohort_ledger_and_refuses_retry(tmp_path,monkeypatch):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    source=tmp_path/'source.json';source.write_text('{}')
    rows=[row(0,m)for m in('raw','h_c_RMS')]
    monkeypatch.setattr(study,'BASE',tmp_path/'output')
    monkeypatch.setattr(study,'PROTOCOL',protocol)
    monkeypatch.setattr(study,'validate_design',lambda d:None)
    monkeypatch.setattr(study,'load_source',lambda c,e:({'rows':rows},source))
    def expire(*args):raise TimeoutError('private cap test')
    monkeypatch.setattr(study,'load_policy',expire)
    with pytest.raises(TimeoutError):study.run('gc100','clean')
    out=tmp_path/'output/gc100/clean'
    assert not(out/'summary.json').exists()
    incomplete=json.loads((out/'incomplete.json').read_text())
    assert not incomplete['complete']and incomplete['expected_cohort']['selected_rows']==2
    assert incomplete['completed_numerical_source_keys']==[]and incomplete['no_retry_performed']
    assert not json.loads((out/'execution.json').read_text())['complete']
    with pytest.raises(AssertionError,match='No selective retries'):study.run('gc100','clean')


def test_new_dense_physical_histories_preserve_shared_inputs_and_motor_tape():
    cfg,net,pop,context=fixture()
    coarse=study.old.prepare_family(net,pop,cfg,context,'probe_glimpse',1,phases=128)
    dense=study.old.prepare_family(net,pop,cfg,context,'probe_glimpse',1,phases=1024)
    audit=study.shared_input_audit(dense,coarse)
    assert audit['passes']
    assert dense['native_x'].shape[1]==1024
    assert torch.equal(dense['target_choice_heading'][::8],coarse['target_choice_heading'])
    # New intermediate headings are actual native trajectories, not copied states.
    delta=study.old.wrap_pi(dense['choice_heading'][1:]-dense['choice_heading'][:-1])
    assert torch.max(torch.abs(delta-2*math.pi/1024))<2e-5
    assert dense['fb_x'][:,1280:1286].argmax(1).tolist()==[2,3]
    # Only CNN accumulation can use tolerance; a nonvisual change is rejected.
    altered={**dense,'native_x':dense['native_x'].clone()}
    altered['native_x'][0,0,768]+=1e-6
    assert not study.shared_input_audit(altered,coarse)['passes']

