"""Frozen original-LSTM screen for grid-module phase memory.

The eliciting code is an explicit synthetic cross-module intervention. All
future inputs are common across phases. No neural angle decoder or circle fit
is used, and a periodic input is never itself classified as a neural ring.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from code.codes import GridPopulation
import lstm_goal_bearing_manifold as base
import lstm_heading_curve_flow as flow
from lstm_endogenous_curve_recovery import advance, describe

ANCHORS = (75,450,1500)
SNAPSHOTS = (0,1,15,75,150,450,525,1500,1575)
REGIMES = ('actual_goal_held','all_goal_withheld')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def module_phase_code(pop, pickup, arrival, module, axis, phases, shift_scale=1.):
    """Translate only one module through an exact primitive lattice period.

    Float64 source population parameters preserve the analytic period exactly;
    agreement with the actual float32 population is checked independently.
    The unchanged 384 cells always carry the same base code.
    """
    assert pop.cell_type == 'grid' and pop.pgc_shift_mode == 'space'
    phases = np.atleast_1d(phases).astype(float)
    k,phi,shifts = (pop._np[name] for name in ('K','phi','shifts'))
    selected = pop.module == module
    angle = pop.orientation[module] + (np.pi/6 if axis==0 else np.pi/2)
    lattice = pop.spacings[module]*np.array([np.cos(angle),np.sin(angle)])
    positions = np.asarray(pickup)[None] + shift_scale*shifts[:,None]*np.asarray(arrival)[None]
    positions = np.broadcast_to(positions,(len(phases),pop.n_cells,2)).copy()
    positions[:,selected] += phases[:,None,None]*lattice
    projections = np.einsum('bnd,nid->bni',positions-phi,k)
    rates = np.clip((np.cos(projections).sum(-1)+1.5)/4.5,0.,1.)
    code = rates**pop.sharpness
    wave_speed = np.einsum('nid,d->ni',k,lattice)*selected[:,None]
    derivative = -np.sum(np.sin(projections)*wave_speed[None],axis=-1)/4.5
    derivative *= pop.sharpness * rates**(pop.sharpness-1)
    return code, derivative, lattice


def input_controls(pop,pickup,arrival):
    results=[]
    for module in range(4):
        for axis in range(2):
            row={'module':module,'axis':axis,'module_size':int((pop.module==module).sum())}
            for label,scale in [('native_predictive_offsets',1.),('ordinary_same_maps',0.)]:
                code,derivative,lattice=module_phase_code(pop,pickup,arrival,module,axis,[0.,1.],scale)
                probe=np.array([.137,.423,.811])
                _,actual,_=module_phase_code(pop,pickup,arrival,module,axis,probe,scale)
                plus=module_phase_code(pop,pickup,arrival,module,axis,probe+1e-6,scale)[0]
                minus=module_phase_code(pop,pickup,arrival,module,axis,probe-1e-6,scale)[0]
                fd=(plus-minus)/2e-6
                turns=np.einsum('nid,d->ni',pop._np['K'][pop.module==module],lattice)/(2*np.pi)
                row[label]={'endpoint_code_max_error':float(np.max(np.abs(code[1]-code[0]))),
                    'endpoint_derivative_max_error':float(np.max(np.abs(derivative[1]-derivative[0]))),
                    'derivative_fd_relative_error':float(np.linalg.norm(fd-actual)/np.linalg.norm(actual)),
                    'wave_period_integer_max_error':float(np.max(np.abs(turns-np.round(turns))))}
                assert row[label]['endpoint_code_max_error']<1e-12
                assert row[label]['endpoint_derivative_max_error']<1e-11
                assert row[label]['derivative_fd_relative_error']<1e-7
            results.append(row)
    native=pop.activity(torch.tensor(pickup[None],dtype=torch.float32),
                        torch.tensor(arrival[None],dtype=torch.float32)).numpy()[0]
    analytic=module_phase_code(pop,pickup,arrival,0,0,[0.])[0][0]
    disagreement=float(np.max(np.abs(native-analytic)))
    assert disagreement<2e-5
    return {'families':results,'native_float32_population_max_disagreement':disagreement}


def geometry_precheck(start,future,initial,scale):
    start,future,initial=[x/scale for x in (start,future,initial)]
    g=flow.curve_geometry(start)
    initial_radius=flow.curve_geometry(initial)['radius']
    distance,phase,_,_=flow.nearest_closed_segments(future,start)
    _,initial_phase,_,_=flow.nearest_closed_segments(start,start)
    transport=flow.phase_transport(phase,initial_phase)
    retention=g['radius']/max(initial_radius,1e-30)
    departure=describe(distance/max(g['radius'],1e-30))
    gates={'family_retains_5percent_initial_radius':retention>=.05,
           'no_large_closing_bridges':g['max_edge_over_radius']<=.1,
           'nonlocal_segments_separated':g['nonlocal_distance_over_radius']['p10']>=.05,
           'fixed_curve_residual_p90_at_most_5percent':departure['p90']<=.05,
           'projected_phase_coverage_at_least_80percent':transport['circular_coverage']>=.8,
           'projected_phase_map_degree_one':abs(transport['winding_degree']-1)<.1,
           'projected_phase_order_preserved':transport['backward_edge_fraction']<=.05}
    return {'geometry':g,'radius_retention':retention,'fixed_curve_departure_over_radius':departure,
            'phase_transport':transport,'gates':gates,'passes_necessary_gates':all(gates.values())}


@torch.inference_mode()
def run_model(condition,seed,args):
    started=time.monotonic()
    output=args.out/f'{condition}_s{seed}';output.mkdir()
    cfg,checkpoint,net,pop=base.load_model(condition,seed,'cpu')
    net.double();cell=net.lstm
    source=ROOT/f'results/original_goal_bearing_manifold/pilot/{condition}_s{seed}/context0_states.npz'
    archive=np.load(source,allow_pickle=False)
    inputs=torch.tensor(archive['history_x'][:15],dtype=torch.float64)
    initial=torch.tensor(np.r_[archive['history_h_before'][0],archive['history_c_before'][0]],dtype=torch.float64)
    observer=archive['history_pos'][14].astype(float)
    pickup=observer+np.array([.75-.29,0.]);arrival=np.array([1.,0.])
    assert np.max(np.abs(pickup))<1.17
    controls=input_controls(pop,pickup,arrival)
    replay=initial[None];replay_error=0.
    for t,x in enumerate(inputs):
        replay=base.step_full(net,x[None],replay)
        expected=np.r_[archive['history_h_after'][t],archive['history_c_after'][t]]
        replay_error=max(replay_error,float(np.max(np.abs(replay.numpy()[0]-expected))))
    assert replay_error<3e-4
    phases=np.arange(args.phases+1)/args.phases
    labels=[(module,axis) for module in range(4) for axis in range(2)]
    codes=np.stack([module_phase_code(pop,pickup,arrival,m,j,phases)[0] for m,j in labels])
    goals=torch.tensor(codes.reshape(-1,512),dtype=torch.float64)
    state=initial[None].repeat(len(goals),1)
    for x in inputs:
        varied=x.repeat(len(goals),1);varied[:,768:1280]=goals
        assert torch.equal(varied[:,:768],x[:768].expand(len(goals),-1))
        assert torch.equal(varied[:,1280:],x[1280:].expand(len(goals),-1))
        drive=torch.nn.functional.linear(varied,cell.weight_ih,cell.bias_ih)
        state=advance(cell,drive,state)
    families=state.reshape(8,args.phases+1,512)
    endpoint_error=(families[:,0]-families[:,-1]).abs().amax(1).numpy()
    assert endpoint_error.max()<1e-10
    families=families[:,:-1].contiguous()
    scales=torch.stack([families[:,:,:256].square().mean((1,2)).sqrt(),
                        families[:,:,256:].square().mean((1,2)).sqrt()],axis=1).clamp_min(1e-8)
    native_drives=[]
    for mode in REGIMES:
        x=inputs[-1:].clone()
        if mode=='all_goal_withheld':x[:,768:1280]=0.
        native_drives.append(torch.nn.functional.linear(x,cell.weight_ih,cell.bias_ih))
    batched_drive=torch.cat([d.expand(8*args.phases,-1) for d in native_drives])
    state=families[None].expand(2,-1,-1,-1).reshape(-1,512).clone()
    saved={0:state.reshape(2,8,args.phases,512).numpy().copy()}
    last=time.monotonic()
    for step in range(1,max(SNAPSHOTS)+1):
        state=advance(cell,batched_drive,state)
        if step in SNAPSHOTS:saved[step]=state.reshape(2,8,args.phases,512).numpy().copy()
        if time.monotonic()-last>25:
            print(f'{condition} seed{seed}: common-input update{step}/{max(SNAPSHOTS)}',flush=True)
            last=time.monotonic()
    np.savez_compressed(output/'states.npz',**{f'states_{t}':v for t,v in saved.items()},
                        initial_codes=codes[:,:-1],scales=scales.numpy(),phase=phases[:-1],
                        observer=observer,pickup=pickup,arrival=arrival)
    records=[]
    for family,(module,axis) in enumerate(labels):
        metric_scales={'raw':np.ones(512),'relative_rms':np.repeat(scales[family].numpy(),256)}
        for regime,mode in enumerate(REGIMES):
            row={'module':module,'axis':axis,'regime':mode,'anchors':{}}
            for anchor in ANCHORS:
                measured={metric:geometry_precheck(saved[anchor][regime,family],saved[anchor+75][regime,family],
                    saved[0][regime,family],scale) for metric,scale in metric_scales.items()}
                necessary=all(m['passes_necessary_gates'] for m in measured.values())
                result={'metrics':measured,'passes_both_metric_necessary_gates':necessary,
                        'radial_recovery_tested':necessary,'candidate':False}
                if necessary:
                    checks={}
                    for metric,scale in metric_scales.items():
                        probe=flow.invariance_probe(cell,native_drives[regime],saved[anchor][regime,family],
                                                    scale,384)
                        measured[metric]['complete_radial_probe']=probe
                        f=probe['horizons']['75'];g=probe['geometry']
                        checks[metric]=all((g['max_edge_over_radius']<=.1,
                            g['nonlocal_distance_over_radius']['p10']>=.05,
                            f['residual_to_fixed_curve_over_radius']['p90']<=.05,
                            f['phase_transport']['circular_coverage']>=.8,
                            abs(f['phase_transport']['winding_degree']-1)<.1,
                            f['phase_transport']['backward_edge_fraction']<=.05,
                            f['radial_normal_gain']['median']<=.5,
                            f['radial_over_tangent_gain']['median']<=.5))
                    result['candidate']=all(checks.values())
                row['anchors'][str(anchor)]=result
            row['expression']={}
            for t in (0,75,450,1500):
                z=torch.tensor(saved[t][regime,family])
                probabilities=net.pi(z[:,:256]).softmax(1).numpy()
                h=z[:,:256].numpy();c=z[:,256:].numpy()
                row['expression'][str(t)]={'hidden_radius':float(np.linalg.norm(h-h.mean(0))/np.sqrt(args.phases)),
                    'cell_radius':float(np.linalg.norm(c-c.mean(0))/np.sqrt(args.phases)),
                    'mean_actor_TV_from_family_mean':float(np.abs(probabilities-probabilities.mean(0)).sum(1).mean()/2),
                    'maximum_absolute_cell':float(np.abs(c).max())}
            records.append(row)
    result={'condition':condition,'seed':seed,'context':0,'seconds':time.monotonic()-started,
            'checkpoint_sha256':sha(checkpoint),'history_sha256':sha(source),
            'input_controls':controls,'native_replay_max_error':replay_error,
            'elicited_state_endpoint_max_errors':endpoint_error.tolist(),'families':records}
    save(output/'result.json',result)
    candidates=sum(a['candidate'] for r in records for a in r['anchors'].values())
    print(f'{condition} seed{seed}: {candidates}/48 family-regime-anchor candidates; {result["seconds"]:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/lstm_module_phase_ring')
    parser.add_argument('--phases',type=int,default=128)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    protocol={'question':'Does original navigation LSTM retain a grid-module lattice phase as an attracting closed state family?',
        'conditions':['gc100','pgc33'],'seeds':[1,2],'context':0,'families':32,
        'elicitation':'15 recorded native updates; replace goalcode only; phase-shift one module, all its ordinary/predictive cells, through one lattice basis period',
        'synthetic_base':'pickup=recorded_endpoint_position+(.46,0); arrival=(1,0); corresponds prior radius.75,pickup_radius.29 geometry',
        'interpretation':'Synthetic cross-module-inconsistent intervention; not a physical goal trajectory and not proof of neural ring from input periodicity',
        'phases':args.phases,'lattice_vectors':'a1=lambda*u(theta+pi/6); a2=lambda*u(theta+pi/2)',
        'regimes':list(REGIMES),'future':'Repeated actual recorded endpoint input, identical across all phases; held means actual recorded goal not phase-varied goal',
        'anchors_updates':list(ANCHORS),'updates_per_second':15,'recovery_horizon_updates':75,
        'metric_scales':'raw and per-family fixed initial h/c RMS','controls':'analyticperiodandderivative; same-map ordinary shift_scale0; nativefloat32population agreement; nativeLSTMreplay; elicitedendpoint agreement',
        'staged_classifier':'All32families x2regimes x3anchors receive old geometry/invariance/phase gates. Only cases passing all necessary gates in BOTH metrics receive full384-arc old radial ±2% probes. Untested recovery is not counted as passing. All original gates must pass for a candidate.',
        'candidate_followup':'Only candidates trigger independent physically coherent native elicitation and held-out-seed confirmation; no favorable-condition search is silently pooled',
        'threads':args.threads,
        'source_sha256':{name:sha(ROOT/name) for name in ['scripts/lstm_module_phase_ring.py','scripts/lstm_heading_curve_flow.py',
            'scripts/lstm_endogenous_curve_recovery.py','scripts/lstm_goal_bearing_manifold.py','code/codes.py','code/nets.py']}}
    path=args.out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen protocol changed; choose a fresh output directory'
    else:save(path,protocol)
    if args.freeze_only:return
    if (args.out/'summary.json').exists():raise RuntimeError('Results already exist; choose fresh output directory')
    results=[run_model(condition,seed,args) for condition in ('gc100','pgc33') for seed in (1,2)]
    summary={'candidate_count':sum(a['candidate'] for model in results for family in model['families'] for a in family['anchors'].values()),
             'total_family_regime_anchors':192,'results':results}
    save(args.out/'summary.json',summary)


if __name__=='__main__':main()
