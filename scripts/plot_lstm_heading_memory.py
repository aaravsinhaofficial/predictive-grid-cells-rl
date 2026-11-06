"""Display frozen heading results together with the raw-coordinate sensitivity."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from code.common import build_policy
from scripts.lstm_heading_memory_probe import recovery, readout_metrics, save, sha

OUT=ROOT/'results/lstm_heading_memory/original'


def phase_diagnostics(state,theta,decoder,movement=0.):
    center,weight,test=decoder
    y=(state-center)@weight
    phase=np.arctan2(y[:,1],y[:,0])
    error=phase[test]-theta[test]-movement
    result=readout_metrics(state,theta,decoder,movement)
    result.update(mean_phase_bias_deg=float(np.rad2deg(np.angle(np.exp(1j*error).mean()))),
                  phase_error_concentration=float(abs(np.exp(1j*error).mean())),
                  decoded_phase_concentration=float(abs(np.exp(1j*phase[test]).mean())))
    return result


def sensitivities(records):
    """Post-screen audits; the original frozen candidate gates stay untouched."""
    raw=[]; centered=[];fixed_bias=[];torch.set_num_threads(2)
    for r in records:
        folder=OUT/f'{r["condition"]}_s{r["seed"]}'
        a=np.load(folder/f'context{r["context"]}_states.npz')
        cfg=json.loads(Path(r['checkpoint']).with_suffix('.json').read_text())
        checkpoint=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
        net=build_policy(cfg,'cpu').eval().requires_grad_(False)
        net.load_state_dict(checkpoint['policy']);net.double()
        index=list(a['times']).index(75)
        z=torch.tensor(a['states'][index]);tangent=torch.tensor(a['tangents'][index])
        drive=torch.nn.functional.linear(torch.tensor(a['dark_input']),net.lstm.weight_ih,net.lstm.bias_ih)
        with torch.no_grad():
            result=recovery(net.lstm,drive,z,tangent,torch.ones(512,dtype=torch.float64),78091+100*r['seed']+r['context'])
        identity={k:r[k] for k in ('condition','seed','context')}
        raw.append(dict(identity,recovery=result)); times={};bias_times={}
        for i,t in enumerate(a['times']):
            state=a['states'][i]/a['scale'];train=np.arange(len(state))%2==0
            decoder=(state[train].mean(0),a['readout_weight'],~train)
            times[str(t)]=dict(initial_phase=phase_diagnostics(state,a['theta'],decoder),
                              heading=phase_diagnostics(state,a['theta'],decoder,r['physical_turn_offset_rad'][str(t)]))
            fixed=(a['readout_center'],a['readout_weight'],~train)
            bias_times[str(t)]=dict(initial_phase=phase_diagnostics(state,a['theta'],fixed),
                                   heading=phase_diagnostics(state,a['theta'],fixed,r['physical_turn_offset_rad'][str(t)]))
        centered.append(dict(identity,results=times))
        fixed_bias.append(dict(identity,results=bias_times))
    hashes={name:sha(ROOT/name) for name in ['scripts/lstm_heading_memory_probe.py','scripts/plot_lstm_heading_memory.py']}
    save(OUT/'raw_metric_sensitivity.json',dict(label='Post-screen sensitivity: raw(h,c) Euclidean perturbations and distances, after5s then1s; frozen primary screen unchanged',source_sha256=hashes,results=raw))
    save(OUT/'fixed_weights_time_centered_sensitivity.json',dict(label='Post-screen sensitivity: original readout weights remain fixed; only the phase-averaged even-training-angle baseline is subtracted at each time. Analysis-only baseline correction.',source_sha256=hashes,results=centered))
    save(OUT/'fixed_readout_phase_bias.json',dict(label='Post-screen circular-bias audit of the unchanged frozen readout: distinguish constant angular offset (error concentration near1) from a collapsed decoded phase (decoded concentration near1). Both initial and physically updated headings are retained.',source_sha256=hashes,results=fixed_bias))


def main():
    records=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_s*/context*.json'))]
    sensitivities(records)
    raw=json.loads((OUT/'raw_metric_sensitivity.json').read_text())['results']
    lookup={(r['condition'],r['seed'],r['context']):r for r in raw}
    colors={'gc100':'#2878b8','pgc33':'#e66b2f'}
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
    for condition,label in [('gc100','Grid'),('pgc33','PGC + grid')]:
        rows=[r for r in records if r['condition']==condition]
        for i,r in enumerate(rows):
            times=sorted(map(int,r['trajectory'])); seconds=np.array(times)*r['step_seconds']
            axes[0].plot(seconds,[r['trajectory'][str(t)]['scaled_radius_retention'] for t in times],color=colors[condition],alpha=.65,label=label if i==0 else None)
            axes[1].plot(seconds,[r['trajectory'][str(t)]['fixed_readout_movement_corrected']['heldout_mae_deg'] for t in times],color=colors[condition],alpha=.65)
            rr=lookup[(condition,r['seed'],r['context'])]['recovery']['horizons']['15']['normal_transverse_gain']['median']
            scaled=r['local_recovery']['75']['horizons']['15']['normal_transverse_gain']['median']
            axes[2].plot([0,1],[rr,scaled],'-o',color=colors[condition],alpha=.65,lw=1,ms=4)
    axes[0].set(xlabel='Seconds in darkness',ylabel='Remaining family size',title='Heading states remain distinct')
    axes[0].axhline(.2,color='.65',ls=':',lw=1);axes[0].legend(frameon=False)
    axes[1].set(xlabel='Seconds in darkness',ylabel='Heading error (degrees)',title='Fixed readout loses heading')
    axes[1].axhline(90,color='.65',ls=':',lw=1)
    axes[2].set(xticks=[0,1],xticklabels=['Raw state','h/c rescaled'],ylabel='Across-curve gain after 1 second',title='Recovery depends on state scaling',yscale='log',xlim=(-.2,1.2))
    axes[2].axhline(1,color='.65',ls=':',lw=1)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'heading_memory_checked.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'heading_memory_checked.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
