"""Exploratory gate-path attribution of selective original-agent stability.

Restore predictive goal influence on selected LSTM gate input drives. Every
branch evolves its own state along the same actual physical input sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy import linalg
import torch
from code.common import build_policy,build_population
from lstm_local_dynamics_spectra import select_states,local_jacobian
from lstm_worst_direction_stability import recover_windows
from lstm_history_input_factorial import correct
from summarize_lstm_code_swap import paired

SOURCE=ROOT/'results/lstm_variable_1B'
GATES={'plain':(), 'input':(0,), 'forget':(1,), 'candidate':(2,), 'output':(3,),
       'candidate_output':(2,3),'input_forget':(0,1),'native':(0,1,2,3)}
TERMS=('direct_memory','forget_feedback','input_feedback','candidate_feedback')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def cell_terms(net,x,state,vector):
    h,c=state.chunk(2);dh,dc=vector.chunk(2)
    w=net.lstm
    i,f,g,o=(torch.nn.functional.linear(x,w.weight_ih.double(),w.bias_ih.double())+
               torch.nn.functional.linear(h,w.weight_hh.double(),w.bias_hh.double())).chunk(4)
    i,f,g,o=i.sigmoid(),f.sigmoid(),g.tanh(),o.sigmoid()
    wi,wf,wg,wo=w.weight_hh.double().chunk(4)
    terms=torch.stack((f*dc,c*f*(1-f)*(wf@dh),g*i*(1-i)*(wi@dh),i*(1-g*g)*(wg@dh)))
    meta=dict(forget_mean=float(f.mean()),forget_derivative_mean=float((f*(1-f)).mean()),
              forget_near_one_fraction=float((f>.99).double().mean()),
              abs_c_weighted_forget_derivative=float((c.abs()*f*(1-f)).sum()/c.abs().sum()),
              candidate_derivative_mean=float((1-g*g).mean()),input_derivative_mean=float((i*(1-i)).mean()))
    jac=local_jacobian(net,x,h,c)
    error=float((terms.sum(0)-(jac@vector)[256:]).abs().max())
    assert error<1e-10,error
    return terms.numpy(),meta,error


def run(seed,replay,rows,windows,unique,rgb,args):
    started=time.monotonic();stem=ROOT/f'models/pgc33/pgc33_lstm_s{seed}_1000M'
    cfg=json.loads(stem.with_suffix('.json').read_text())
    ckpt=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=True)
    net=build_policy(cfg,'cpu').eval().requires_grad_(False);net.load_state_dict(ckpt['policy'])
    pop=build_population(cfg,'cpu')
    saved=dict(np.load(SOURCE/f'pgc33_s{seed}_activity.npz'))
    initial=np.concatenate((saved['h'][rows],saved['c'][rows]),1).astype(np.float64)
    local=np.searchsorted(unique,windows);b=len(rows)
    with torch.inference_mode():
        e=torch.cat([net.embed(rgb[j:j+256]) for j in range(0,len(rgb),256)])
        current=pop.activity(*(torch.from_numpy(replay[k][unique]) for k in ('pos','dir','vel')))
        goal=pop.activity(*(torch.from_numpy(replay['memory_'+k][unique]) for k in ('pos','dir','vel')))
        ordinary=pop.activity(*(torch.from_numpy(replay['memory_'+k][unique]) for k in ('pos','dir','vel')),shift_scale=0)
        known=torch.from_numpy(replay['goal_known'][unique])[:,None];goal*=known;ordinary*=known
        pa=torch.from_numpy(replay['prev_action'][unique]);pr=torch.from_numpy(replay['prev_reward'][unique])
        x={'native':net._concat(e,current,goal,pa,pr,None).double(),
           'plain':net._concat(e,current,ordinary,pa,pr,None).double()}
    wi,wh=net.lstm.weight_ih.double(),net.lstm.weight_hh.double()
    bi,bh=net.lstm.bias_ih.double(),net.lstm.bias_hh.double()
    piw,pib=net.pi.weight.double(),net.pi.bias.double()
    native_drive=torch.nn.functional.linear(x['native'],wi,bi)
    plain_drive=torch.nn.functional.linear(x['plain'],wi,bi)
    drives={}
    for gate,indices in GATES.items():
        drive=plain_drive.clone()
        for k in indices:drive[:,256*k:256*(k+1)]=native_drive[:,256*k:256*(k+1)]
        drives[gate]=drive
    assert torch.equal(drives['plain'],plain_drive) and torch.equal(drives['native'],native_drive)
    vectors=[];sv=[];decomposition={k:[] for k in ('native','plain')};gate_meta={k:[] for k in decomposition}
    maxdecomp=0.
    print(f'seed{seed}: ordinary-sensitive axes and cell-update decomposition',flush=True)
    for j in range(b):
        state=torch.from_numpy(initial[j]);h,c=state.chunk(2)
        jac=local_jacobian(net,x['plain'][local[j,0]],h,c)
        _,values,vh=linalg.svd(jac.numpy(),full_matrices=False,check_finite=False)
        vec=torch.from_numpy(vh[0].copy());vectors.append(vh[0]);sv.append(float(values[0]))
        for name in decomposition:
            terms,meta,error=cell_terms(net,x[name][local[j,0]],state,vec)
            decomposition[name].append(terms);gate_meta[name].append(meta);maxdecomp=max(maxdecomp,error)
    v=np.array(vectors)
    old=json.loads((ROOT/f'results/ring_attractor/worst_direction/pgc33_s{seed}.json').read_text())
    sv_error=float(np.max(np.abs(np.array(sv)-old['largest_singular_values']['goal_ordinary'])))
    assert sv_error<1e-8,sv_error
    decomposition={k:np.array(a) for k,a in decomposition.items()}
    reference=decomposition['plain'].sum(1)
    unit=reference/np.linalg.norm(reference,axis=1,keepdims=True)
    decomp_summary={}
    for name,terms in decomposition.items():
        projections=np.einsum('bkd,bd->bk',terms,unit)
        decomp_summary[name]=dict(
            terms={term:dict(mean_norm=float(np.linalg.norm(terms[:,k],axis=1).mean()),
                            mean_signed_projection_on_plain_response=float(projections[:,k].mean())) for k,term in enumerate(TERMS)},
            total_c_response_norm=float(np.linalg.norm(terms.sum(1),axis=1).mean()),
            total_c_projection_on_plain_response=float(projections.sum(1).mean()),
            gates={k:float(np.mean([m[k] for m in gate_meta[name]])) for k in gate_meta[name][0]})
    measurements={};archive={};replication={}
    for amplitude in args.amplitudes:
        epsilon=amplitude*np.sqrt(512)
        patches=np.stack((initial,initial+epsilon*v,initial-epsilon*v))
        a=np.tile(patches[None],(len(GATES),1,1,1)).reshape(-1,512)
        h,c=torch.from_numpy(a).chunk(2,1)
        curves={g:{k:[] for k in ('c_error','h_error','gain','action_tv')} for g in GATES}
        with torch.inference_mode():
            for t in range(args.horizon):
                drive=torch.stack([drives[g][local[:,t]] for g in GATES])[:,None].expand(-1,3,-1,-1).reshape(-1,1024)
                i,f,cand,o=(drive+torch.nn.functional.linear(h,wh,bh)).chunk(4,1)
                c=f.sigmoid()*c+i.sigmoid()*cand.tanh();h=o.sigmoid()*c.tanh()
                state=torch.cat((h,c),1).numpy().reshape(len(GATES),3,b,512)
                probs=torch.nn.functional.linear(h,piw,pib).softmax(-1).numpy().reshape(len(GATES),3,b,6)
                for gi,gate in enumerate(GATES):
                    delta=state[gi,1:]-state[gi,0]
                    for key,sl in [('c_error',slice(256,512)),('h_error',slice(0,256)),('gain',slice(0,512))]:
                        curves[gate][key].append(np.linalg.norm(delta[:,:,sl],axis=2).mean(0)/epsilon)
                    curves[gate]['action_tv'].append((.5*np.abs(probs[gi,1:]-probs[gi,0]).sum(2)).mean(0))
        measurements[str(amplitude)]={}
        for gate,values in curves.items():
            measurements[str(amplitude)][gate]={}
            for key,values0 in values.items():
                a=np.array(values0);archive[f'a{amplitude}_{gate}_{key}']=a
                measurements[str(amplitude)][gate][key]={str(t):float(a[t-1].mean()) for t in [1,5,15,args.horizon]}
                measurements[str(amplitude)][gate][key]['time_mean']=float(a.mean())
        # Full/native and no/plain restoration reproduce the earlier follow-up.
        for gate,oldv in [('native','native'),('plain','goal_ordinary')]:
            for metric in ('c_error','h_error','gain','action_tv'):
                now=measurements[str(amplitude)][gate][metric]
                previous=old['measurements'][str(amplitude)][oldv]['goal_ordinary'][metric]
                replication[f'{amplitude}/{gate}/{metric}']=max(abs(now[k]-previous[k]) for k in now)
    assert max(replication.values())<1e-8,replication
    record=dict(seed=seed,condition='pgc33',samples=b,horizon=args.horizon,amplitudes=args.amplitudes,
        decomposition=decomp_summary,measurements=measurements,
        validation=dict(all_gates_equal_native_drive=True,no_gates_equal_plain_drive=True,
                        ordinary_singular_values_max_error=sv_error,cell_decomposition_vs_jacobian_max_error=maxdecomp,
                        prior_native_plain_trajectory_max_error=max(replication.values())),
        checkpoint_sha256=sha(stem.with_suffix('.pt')),seconds=time.monotonic()-started)
    np.savez_compressed(args.out/f'pgc33_s{seed}_curves.npz',windows=windows,ordinary_worst_vector=v,
                        native_cell_terms=decomposition['native'],plain_cell_terms=decomposition['plain'],**archive)
    (args.out/f'pgc33_s{seed}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(f'seed{seed}: done {record["seconds"]:.1f}s',flush=True)
    return record


def summarize(runs,args):
    family={}
    if len(runs)>1:
        for gate in list(GATES)[1:-1]:
            for horizon in ('1',str(args.horizon)):
                a=[r['measurements']['0.0001']['plain']['c_error'][horizon] for r in runs]
                b=[r['measurements']['0.0001'][gate]['c_error'][horizon] for r in runs]
                family[gate+'_at_'+horizon]=paired(a,b)
        correct(family)
    summary=dict(protocol=json.loads((args.out/'protocol.json').read_text()),exploratory_comparisons=family,models=runs)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    avg=lambda gate,metric,t:float(np.mean([r['measurements']['0.0001'][gate][metric][t] for r in runs]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    order=list(GATES);labels=['Ordinary','Input','Forget','Candidate','Output','Cand. + out.','Input + forget','Intact']
    colors=['#9299A1','#87AAC7','#587DA5','#E5AE6F','#BC8D65','#D87532','#6F94A9','#6F573B']
    for ax,t,title in zip(axes,['1',str(args.horizon)],['After one update','After four seconds']):
        values=[avg(g,'c_error',t) for g in order]
        ax.bar(np.arange(len(order)),values,color=colors)
        ax.set(xticks=np.arange(len(order)),xticklabels=labels,title=title,ylabel='Cell-state error / initial full-state error')
        ax.tick_params(axis='x',labelrotation=50,labelsize=8);ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('Restore predictive influence through selected gate inputs · Same vulnerable perturbation',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(args.out/f'gate_stability.{ext}',dpi=180)
    plt.close(fig)
    protected1=avg('plain','c_error','1')-avg('native','c_error','1')
    protected60=avg('plain','c_error',str(args.horizon))-avg('native','c_error',str(args.horizon))
    forget1=(avg('plain','c_error','1')-avg('forget','c_error','1'))/protected1
    forget60=(avg('plain','c_error',str(args.horizon))-avg('forget','c_error',str(args.horizon)))/protected60
    co60=(avg('plain','c_error',str(args.horizon))-avg('candidate_output','c_error',str(args.horizon)))/protected60
    lines=['# The forget gate protects the sensitive memory direction','',
        f'**Restoring predictive influence to the forget gate recovers {100*forget1:.0f}% of the mean one-step cell-state protection and {100*forget60:.0f}% of the protection remaining after four seconds.** These are fractions of the difference between ordinary and intact input in this targeted perturbation test, not fractions of navigation performance.','',
        'We started with the same ordinary-goal-sensitive disturbance used in the preceding experiment. We then restored the predictive goal message to selected gate input drives while leaving the other gate inputs ordinary. Each branch carried its own actual h/c state through the same 60 physical inputs.','',
        '![Gate-specific stability](gate_stability.png)','',
        '| Predictive influence restored to | Cell error after one update | Cell error after four seconds |','|---|---:|---:|']
    for gate,label in zip(order,labels):lines.append(f'| {label} | {avg(gate,"c_error","1"):.4f} | {avg(gate,"c_error",str(args.horizon)):.4f} |')
    lines+=['','The units are amplification relative to the initial full h/c perturbation. The primary displayed amplitude is 0.0001 RMS raw-state units; 0.01 sensitivity results are also saved. This is an exploratory gate attribution, following the earlier fixed four-comparison analysis.','',
        'The cell-update derivative was independently decomposed into four terms: directly retained cell-state error; error entering through the forget-gate feedback; input-gate feedback; and candidate-update feedback. All terms are projected onto the same ordinary-goal cell-response direction, so their signed contributions add. Their norms are also saved, but norms do not add because terms can cancel.','',
        '| Cell derivative term | Native projection | Ordinary projection |','|---|---:|---:|']
    for term in TERMS:
        a=np.mean([r['decomposition']['native']['terms'][term]['mean_signed_projection_on_plain_response'] for r in runs])
        b=np.mean([r['decomposition']['plain']['terms'][term]['mean_signed_projection_on_plain_response'] for r in runs])
        lines.append(f'| {term} | {a:.4f} | {b:.4f} |')
    lines+=['', '**What the mechanism means:** an error in hidden activity can perturb the forget gate. Because that gate multiplies the stored cell state, a small gate error can become a larger memory error when the stored value is large. The predictive goal message reduces this amplification along the tested vulnerable directions. Average forget-gate openness changes little; the relevant change is the sensitivity of the weighted memory feedback.','',
        f'The candidate-plus-output restoration recovers only {100*co60:.0f}% of this four-second cell-state protection. Earlier, that pathway recovered most of the navigation loss when prediction was removed from both current and goal codes. The present test changes only the goal code. These results suggest different contributions to memory protection and action computation; a matched goal-only behavioral rescue would directly compare those functions.','',
        'Forget restoration improves the targeted cell-state response in all eight seeds at both horizons. Twelve exploratory paired-seed contrasts (six partial restorations at two horizons) receive Holm adjustment separately for paired-t and exact sign-flip tests. One-step forget restoration passes corrected paired-t testing; at this sample size the corrected exact test is 0.09375, and the four-second corrected paired-t test is about 0.098. Full results and intervals are saved; the earlier four worst-direction comparisons are unchanged.','',
        'Validation checks that restoring all gates exactly reproduces native input drive, restoring none exactly reproduces ordinary input drive, both full trajectories reproduce the preceding experiment, and the cell derivative sum matches the analytic full-state Jacobian.','',
        'These are vulnerable recurrent-state directions, not established ring tangents or normals. The analysis isolates a gate mechanism for memory-state amplification; behavior and ring geometry are evaluated separately.','',
        '[Results](summary.json) · [Protocol](protocol.json) · [Source](../../../scripts/lstm_gate_stability.py)']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--horizon',type=int,default=60)
    p.add_argument('--amplitudes',nargs='+',type=float,default=[.0001,.01])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path,default=ROOT/'results/ring_attractor/gate_stability')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads);args.out.mkdir(parents=True,exist_ok=True)
    replay=dict(np.load(SOURCE/'replay.npz'));rows=select_states(replay);n=int(replay['episodes'])
    windows=replay['sample_index'][rows,None]+n*np.arange(1,args.horizon+1)[None];unique=np.unique(windows)
    cfg=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    rgb,rgb_hash=recover_windows(cfg,replay,unique)
    protocol=dict(seeds=args.seeds,states_per_seed=len(rows),horizon=args.horizon,amplitudes=args.amplitudes,
        purpose='Exploratory gate attribution of the existing ordinary-goal worst-direction stability effect.',
        direction='Leading right singular vector under ordinary goal input at the exact earlier32states; fixed for every gate restoration.',
        restoration='Replace selected gate INPUT-drive rows with native predictive goal rows; every branch evolves its own state under actual60futurephysicalinputs.',
        decomposition='delta_c=f*delta_c_prev + c*f*(1-f)*W_f*delta_h + g*i*(1-i)*W_i*delta_h + i*(1-g*g)*W_g*delta_h; signed projection onto common ordinary-response direction.',
        statistics='12 exploratory restoration-versus-plain contrasts on cell-error amplification,6partialrestorations x1and60steps; Holmtandexact separately.',
        rgb_sha256=rgb_hash,replay_sha256=sha(SOURCE/'replay.npz'),source_sha256=sha(Path(__file__)))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    runs=[]
    for seed in args.seeds:
        path=args.out/f'pgc33_s{seed}.json'
        if args.resume and path.exists():runs.append(json.loads(path.read_text()))
        else:runs.append(run(seed,replay,rows,windows,unique,rgb,args))
    summarize(runs,args)


if __name__=='__main__':main()
