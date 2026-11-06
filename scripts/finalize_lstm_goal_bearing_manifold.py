"""Reproducible descriptive summary of the two-seed original RL state-family pilot."""
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import lstm_goal_bearing_manifold as m


def coordinate_audit(directory):
    results={}
    for ci in range(4):
        a=np.load(directory/f'context{ci}_states.npz');theta=a['theta']
        hd0=a['history_hd'][14];post=a['history_pos'][29];hdpost=a['history_hd'][29]
        for oi in range(3):
            for variant in ['native','ordinary_goal']:
                key=f'offset{oi}_{variant}'
                if key+'_states'not in a:continue
                center,w,test=m.fit_phase_readout(a[key+'_states'].astype(float),theta)
                points=a[key+'_geometry'][[0,1,3]]
                targetinitial=a[key+'_bearings_center_pickup_ahead']
                for mode in ['goal_held','goal_withheld']:
                    z=a[key+'_'+mode+'_t15'][0];y=(z-center)@w
                    phase=np.arctan2(y[:,1],y[:,0]);valid=test&(np.linalg.norm(y,axis=1)>=.1)
                    row=dict(context=ci,offset_index=oi)
                    for k,(name,p) in enumerate(zip(['goalcenter','pickup','predictivepoint'],points)):
                        bearing=np.arctan2((p-post)[:,1],(p-post)[:,0]);aligned=bearing-targetinitial[k]+theta
                        for coord,target in [('allocentric',aligned),('egocentric',aligned-hdpost+hd0)]:
                            row[name+'_'+coord]=float(np.rad2deg(m.angular_error(phase[valid],target[valid])).mean())
                    results.setdefault(variant+'_'+mode,[]).append(row)
    payload=dict(condition=directory.name,assay='Initial angular readout adjusted for exact native observer translation and facing change; initial offset aligned separately for center/pickup/predicted point.',metrics=results)
    m.save_json(directory/'coordinate_audit.json',payload)
    return payload


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/original_goal_bearing_manifold/pilot');args=ap.parse_args()
    records=[]
    for directory in sorted(args.root.glob('*_s*')):
        if not(directory/'summary.json').exists():continue
        r=json.loads((directory/'summary.json').read_text());q=json.loads((directory/'refinement.json').read_text());coord=coordinate_audit(directory)
        for variant in ['native','ordinary_goal']:
            families=[f for c in r['contexts']for f in c['families']if f['variant']==variant]
            refinements=[f for c in q['contexts']for f in c['families']if f['variant']==variant]
            if not families:continue
            end=[f['continuations']['goal_withheld']['15']for f in families]
            re=[f['continuations']['goal_withheld']['15']for f in refinements]
            coords=coord['metrics'][variant+'_goal_withheld']
            row=dict(condition=r['condition'],seed=r['seed'],variant=variant,
                contexts=len(r['contexts']),families=len(families),
                initial_decode_mae_deg=float(np.mean([f['phase_at_end']['heldout_mae_deg']for f in families])),
                withdrawal_radius_retention=float(np.mean([f['radius_retention']for f in end])),
                initial_readout_after_withdrawal_mae_deg=float(np.mean([f['phase']['heldout_mae_deg']for f in end])),
                refitted_readout_after_withdrawal_mae_deg=float(np.mean([f['newly_fitted_phase']['heldout_mae_deg']for f in re])),
                translation_corrected_mae_deg=float(np.mean([f['goalcenter_allocentric']for f in coords])),
                egocentric_corrected_mae_deg=float(np.mean([f['goalcenter_egocentric']for f in coords])),
                local_tangent_relative_gain=float(np.mean([f['tangent_error_gain']for f in re])),
                local_normal_relative_gain=float(np.mean([f['normal_error_gain']for f in re])),
                local_tangent_raw_gain=float(np.mean([f['tangent_raw_coordinate_error_gain']for f in re])),
                local_normal_raw_gain=float(np.mean([f['normal_raw_coordinate_error_gain']for f in re])))
            records.append(row)
    groups=[('Grid','gc100','native','#4D81B3'),('PGC + grid','pgc33','native','#DD7737'),('PGC, ordinary goal','pgc33','ordinary_goal','#8873A3')]
    means={label:{key:float(np.mean([r[key]for r in records if(r['condition'],r['variant'])==(condition,variant)])) for key in records[0]if key not in ('condition','variant','seed')}for label,condition,variant,color in groups}
    payload=dict(stage='Exploratory two-seed feasibility; no significance tests',records=records,group_means=means,
        comparison='PGC native/ordinary goal are matched counterfactuals in the same trained network/history. Grid histories are native to their own actor and provide descriptive context only.')
    m.save_json(args.root/'aggregate.json',payload)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.25),layout='constrained')
    theta=np.linspace(0,2*np.pi,100);ax=axes[0]
    ax.plot(.75*np.cos(theta),.75*np.sin(theta),color='.8',lw=1)
    center,pickup,arrival,ahead=m.snapshot_geometry([0,0],np.array([0]),np.pi/2)
    ax.add_patch(plt.Circle(center[0],.3,color='#DCE9D4',alpha=.8))
    ax.scatter([0],[0],color='k',s=30);ax.text(-.08,-.12,'Agent',ha='right',fontsize=8)
    for point,color,label,delta in [(center[0],'#61964C','Goal',(0.02,.09)),(pickup[0],'#4D81B3','Pickup',(.04,-.04)),(ahead[0],'#DD7737','Predictive point',(.04,.02))]:
        ax.scatter(*point,color=color,s=30);ax.annotate(label,point,xytext=(point[0]+delta[0],point[1]+delta[1]),fontsize=8)
    ax.annotate('',ahead[0],pickup[0],arrowprops=dict(arrowstyle='->',color='#DD7737'))
    ax.set_aspect('equal');ax.set(xlim=(-.9,1.3),ylim=(-.95,.95),title='Change only the goal message');ax.axis('off')
    x=np.array(m.HORIZONS)/15
    for label,condition,variant,color in groups:
        radius=[]
        for p in sorted(args.root.glob(condition+'_s*/summary.json')):
            r=json.loads(p.read_text());families=[f for c in r['contexts']for f in c['families']if f['variant']==variant]
            radius.append(np.mean([[f['continuations']['goal_withheld'][str(t)]['radius_retention']for t in m.HORIZONS]for f in families],axis=0))
        axes[1].plot(x,np.mean(radius,axis=0),color=color,label=label,lw=2)
        axes[1].scatter([x[-1]]*len(radius),[a[-1]for a in radius],color=color,s=12)
    axes[1].set(xlabel='Time after goal cue removed (s)',ylabel='Goal-specific state separation / initial',title='Goal-dependent state persists',ylim=(0,1.1));axes[1].legend(fontsize=7)
    for i,(label,condition,variant,color)in enumerate(groups):
        value=means[label]
        axes[2].bar(i-.17,value['egocentric_corrected_mae_deg'],.32,color=color,alpha=.35)
        axes[2].bar(i+.17,value['refitted_readout_after_withdrawal_mae_deg'],.32,color=color)
    axes[2].set(xticks=range(3),xticklabels=['Grid','PGC','Ordinary\ngoal'],ylabel='Heldout angle error (degrees)',title='Information remains; readout changes')
    axes[2].plot([],[],color='.5',lw=8,alpha=.35,label='Original readout, motion corrected')
    axes[2].plot([],[],color='.5',lw=8,label='Readout refitted at 1 s');axes[2].legend(fontsize=6.5)
    for ax in axes[1:]:ax.spines[['top','right']].set_visible(False)
    fig.savefig(args.root/'goal_information_persistence.png',dpi=190);plt.close(fig)
    lines=['# Original RL agents: what remains after the goal cue disappears?','',
        '**The original RNNs retain information about the goal. This pilot does not show that PGCs make that memory more stable, or that it is a ring attractor.**','',
        '![Goal message and recurrent-state experiment](goal_information_persistence.png)','',
        'We used the existing1-billion-frame Grid and PGC+grid LSTMs, with their original512 hexagonal maps, vision, six actions, and square arena. No network was trained or modified.','',
        'We collected four real native-policy histories from each of two seed pairs. At each history, we replayed1second of exactly the same physical inputs while changing only the saved goal message. The synthetic goals lay around the observer. Each message was made from a valid pickup point29cm from its goal, with predictive cells evaluated20cm along the inward arrival direction. Three arrival directions separate the observer bearings of the true center, pickup, and predictive point.','',
        'We then removed only the saved goal input for1second. Vision, current spatial code, previous action, and reward followed the same actual physical replay for every branch. Goal removal after discovery is an experimental intervention; the original training task supplies this message continuously.','',
        'The angle readouts are analysis tools fitted separately within each physical context and arrival-offset family, using alternating angles for training and the remaining angles for testing. They are not an extra input, the original actor, or evidence of a universal goal decoder. State separation means the RMS radius of the full(h,c) family, not percent variance retained.','',
        '|After1second without the saved goal cue|Grid|PGC+grid|Same PGC model, ordinary goal|',
        '|---|---:|---:|---:|']
    for name,key,fmt in [('Full-state goal separation retained','withdrawal_radius_retention',lambda x:f'{100*x:.1f}%'),('Original angle readout, corrected for movement/facing','egocentric_corrected_mae_deg',lambda x:f'{x:.1f}°'),('Angle readout refitted at the new time','refitted_readout_after_withdrawal_mae_deg',lambda x:f'{x:.1f}°')]:
        lines.append('|'+name+'|'+'|'.join(fmt(means[label][key])for label,*_ in groups)+'|')
    lines+=['','**Meaning:** the goal leaves a lasting trace in recurrent activity. A new readout can still recover which goal was supplied. The original readout does not remain aligned to that information, even after correcting the actual observer movement and facing. This suggests a changing goal representation, not a fixed angular memory coordinate.','',
        '**Predictive versus ordinary:** replacing only the169 predictive goal channels by their ordinary-grid values produces almost the same memory persistence. Current predictive input remains intact. The comparison is matched within the same PGC network and native history. Grid models contribute descriptive context using their own native histories.','',
        '**Why this is not a demonstrated ring attractor:** rotating a continuous input naturally produces a periodic state family. We additionally tested the full512-dimensional(h,c) state, exact angular tangents, and three equal-size normal perturbations. The tangent errors shrink; the normal perturbations do not show the preferential recovery expected around a stable ring. This result is sensitive to the h/c metric, so both raw and relative-RMS gains are retained in the data. We do not equate angular decoding, a projected loop, or residual state separation with an attractor.','',
        '**Checks and limits:** four scientific tests pass. Native replay reconstructs the recorded state within3e−4; original model files are unchanged;343 ordinary goal channels and all other input columns remain untouched. Exact angular Jacobian-vector products agree with central differences. Increasing angular resolution192→768 shrinks neighboring-state gaps. The first context of every model also has an explicitly artificial fixed-input check. No seed-level significance claim is made from two pairs. The synthetic goal messages are geometrically valid but were not experienced in the preceding native history; the continuation is forced physical replay, not a policy rollout under those false goals.','',
        'An arrival-direction transfer check did not reliably identify whether a simple readout represented the true center, pickup, or predictive point. All three semantic targets and the actual observer-coordinate corrections are saved; this assay does not establish a literal aiming-point decoder.','',
        'Reproduce:','',
        '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/lstm_goal_bearing_manifold.py --conditions gc100 pgc33 --seeds 1 2',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/refine_lstm_goal_bearing_manifold.py',
        'OPENBLAS_NUM_THREADS=1 python scripts/finalize_lstm_goal_bearing_manifold.py','python -m pytest -q tests/test_lstm_goal_bearing_manifold.py','```','',
        'Numerical results: [aggregate](aggregate.json), per-model summary/refinement/coordinate audits in the four model folders. Full native inputs and(h,c) arrays are retained locally in the compressed context archives(~1.1GB total); they are not required to view this report.']
    (args.root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
