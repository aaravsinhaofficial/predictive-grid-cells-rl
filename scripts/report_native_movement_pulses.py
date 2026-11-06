"""Saved-data-only report of the fixed native movement impulse follow-up."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from native_movement_pulses import MODES,PULSES,MANIFEST,PROTOCOL
from rl_goal_memory import atomic_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean())if len(a)else None,median=float(np.median(a))if len(a)else None)




def joint_availability(active,replay_steps,horizon,pulse_index):
    closed=horizon<=len(active)and all(bool((active[:horizon,m*7+pulse_index]&active[:horizon,m*7]).all())for m in range(3))
    joint=horizon<=replay_steps and closed
    all_directions=horizon<=replay_steps and horizon<=len(active)and bool(active[:horizon].all())
    return closed,joint,all_directions


def render(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'native':'#2878b5','current_grid':'#e07b39','current_candidate_output_rescue':'#3a9b70'}
    labels={'native':'Native','current_grid':'Current grid','current_candidate_output_rescue':'g/o rescue'}
    fig,axes=plt.subplots(2,2,figsize=(9,7),sharex=True)
    for ax,key in zip(axes.flat,('gc100_s1','pgc33_s1','gc100_s2','pgc33_s2')):
        model=result['models'][key]
        for mode in MODES:
            for stream,style in [('closed','-'),('replay','--')]:
                x=[0];y=[1.]
                for horizon in (1,5,15,30):
                    metric=model['responses'][f'{stream}|{mode}|normal|{horizon}']['state_gain']
                    if metric['n']:x.append(horizon/15);y.append(metric['mean'])
                ax.plot(x,y,style,color=colors[mode],lw=2,label=labels[mode]+(' / replay'if stream=='replay'else' / closed'))
        ax.axhline(1,color='.65',lw=.7);ax.set_title(('Grid'if key.startswith('gc')else'PGC + grid')+f" · seed {key[-1]} · {model['eligible']} cases")
        ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    axes[0,0].set_ylabel('State error / initial error');axes[1,0].set_ylabel('State error / initial error')
    for ax in axes[-1]:ax.set_xlabel('Seconds after normal-direction pulse')
    handles,labels0=axes[0,1].get_legend_handles_labels();fig.legend(handles,labels0,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(bottom=.17,hspace=.3,wspace=.2)
    for ext in ('png','pdf'):fig.savefig(out/f'state_impulse_response.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),sharey=True)
    family_colors={'turn':'#8864ad','drift':'#d08d2c','normal':'#2878b5'}
    for ax,key in zip(axes,('pgc33_s1','pgc33_s2')):
        for family in ('turn','drift','normal'):
            for stream,style in [('closed','-'),('replay','--')]:
                x=[0];y=[1.]
                for horizon in (1,5,15,30):
                    metric=result['models'][key]['responses'][f'{stream}|native|{family}|{horizon}']['state_gain']
                    if metric['n']:x.append(horizon/15);y.append(metric['mean'])
                ax.plot(x,y,style,color=family_colors[family],lw=2,label=family+(' / replay'if stream=='replay'else' / closed'))
        ax.set_title('PGC + grid · seed '+key[-1]);ax.set_xlabel('Seconds after pulse');ax.grid(alpha=.15);ax.set_ylim(bottom=0)
    ymax=max(float(np.max(line.get_ydata()))for ax in axes for line in ax.lines)
    for ax in axes:ax.set_ylim(0,ymax*1.06)
    axes[0].set_ylabel('State error / initial error');handles,labels0=axes[1].get_legend_handles_labels();fig.legend(handles,labels0,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(bottom=.28,wspace=.2)
    for ext in ('png','pdf'):fig.savefig(out/f'observed_directions.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    lines=['# Native movement: state impulse responses','',
        '**The current predictive signal changes steering, but these turning states do not show a consistent predictive-input stabilization effect across the two discovery seeds.** Replacing current predictive features changed about 18–19% of sampled actions and reduced two-second goal progress by about 1.1–1.3 cm. Restoring their candidate/output-gate influence brought action choices closer to native; progress recovery was mixed.','',
        'The observed-turn nudges mostly faded. Equal-sized orthogonal random nudges produced much larger state and policy changes, including a large first-update amplification. These are finite interventions in the stated block-RMS metric, not an infinitesimal stability spectrum or a validated radial test. The input-removal effect on this response reversed between the two PGC seeds.','',
        'We nudged the original RNN state during naturally repeated turns. The three directions were an observed turning-state change, the drift between successive turns, and a random direction orthogonal to both. Each nudge had the same scaled size. No circle was fitted or imposed.','',
        'The state was then followed for two seconds with native input, current predictive features replaced by ordinary grid features, or predictive influence restored only to the candidate update and output gate. The saved goal cue stayed native throughout. Each pulse was compared with its own unpulsed input-route control.','',
        'Solid lines include the agent–environment feedback loop. Dashed lines replay the exact recorded native RNN inputs, including previous actions and rewards.','',
        '![State response](state_impulse_response.png)','',
        '| Model | Eligible / all episodes | Native normal-pulse gain, closed loop | Same gain, replay | Current-grid gain, closed loop |','|---|---:|---:|---:|---:|']
    for key,model in result['models'].items():
        def value(stream,mode):
            d=model['responses'][f'{stream}|{mode}|normal|30']['state_gain'];return f"{d['mean']:.3f} (n={d['n']})"if d['n']else'No uncensored cases'
        lines.append(f"| {key} | {model['eligible']}/32 | {value('closed','native')} | {value('replay','native')} | {value('closed','current_grid')} |")
    lines+=['','Gain = remaining scaled h/c error divided by the initial pulse. n counts signed pulse/case observations; opposite signs are not independent models. Both figures use the SAME cases for closed loop/replay, all three input routes, and all signed directions at each horizon. Cases must remain preterminal everywhere and have recorded replay available; per-horizon cohort sizes are saved. Less-restricted counts and responses remain separately available. All missing and censored cases remain in the saved report.','',
        '![Observed direction comparison](observed_directions.png)','',
        '| Model | Native goal progress, cm | Change after current-grid substitution, cm | Mean policy TV from substitution |','|---|---:|---:|---:|']
    for key,model in result['models'].items():
        effects=model['unpulsed_mode_effects'];a=effects['native']['goal_progress_cm']['mean'];b=effects['current_grid']['progress_difference_from_native_cm']['mean'];c=effects['current_grid']['mean_action_tv']['mean']
        lines.append(f'| {key} | {a:.2f} | {b:+.2f} | {c:.4f} |')
    lines+=['','| PGC model | Input route | Actions different from native | Policy TV from native | Goal progress, cm |','|---|---|---:|---:|---:|']
    for key in ('pgc33_s1','pgc33_s2'):
        for mode in MODES:
            row=result['models'][key]['unpulsed_mode_effects'][mode]
            lines.append(f"| {key} | {labels[mode]} | {100*row['action_difference_fraction']['mean']:.1f}% | {row['mean_action_tv']['mean']:.3f} | {row['goal_progress_cm']['mean']:.2f} |")
    lines+=['','These are short-window progress measurements from the mechanically selected turns, not complete-episode navigation scores. First goal hits terminate each branch; its endpoint is held for the fixed-time progress comparison, and teleportation never counts as travel.','',
        'The pulses were not clipped. The saved per-case metadata records any hidden-state coordinates that initially exceeded the normal [-1, 1] range; these are finite interventions in the stated coordinates, not an infinitesimal stability spectrum.','',
        'Every selected native-only fork exactly reproduced the archived inputs, h/c states, logits, sampled actions, physical transitions and random-state clocks before interventions. Grid substitution and rescue were exact shams. All four original checkpoints stayed unchanged. No training or cloud compute was used.','',
        'This measures local state and policy sensitivity along observed movement trajectories. A turn secant is not a validated ring phase; contraction in an orthogonal direction alone does not identify a ring. The report keeps full h/c, h-only, c-only, policy TV, action changes, movement progress and censored right-turn/right-strafe switching times separately. Two original seeds are a discovery sample, not a population significance test.','',
        'All branch summaries and raw-artifact hashes: [summary.json](summary.json). The frozen protocol and source manifest are under `docs/research/native_movement_pulse_*`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2);root=ROOT/'results/native_movement_pulses';out=root/'report';out.mkdir(exist_ok=True)
    aggregate={};inputs={};all_rows=[];case_meta=[];mode_effects=[];transition_rows=[]
    for condition,seed in [('gc100',1),('pgc33',1),('gc100',2),('pgc33',2)]:
        folder=root/f'{condition}_s{seed}';p=folder/'summary.json';summary=json.loads(p.read_text())
        assert summary['complete']and summary['manifest_sha256']==sha(MANIFEST)and summary['protocol_sha256']==sha(PROTOCOL)
        assert all(all(v==0 for v in row.values())for row in summary['fork_checks'])
        prechecks=json.loads((folder/'native_only_prechecks.json').read_text());assert prechecks['complete']and prechecks['before_any_counterfactual']
        key=f'{condition}_s{seed}';inputs[key]=dict(summary_sha256=sha(p),eligible=summary['eligible'],missing=len(summary['missing']),episodes=32,native_fork_exact=True)
        for name,meta in summary['raw_assets'].items():
            path=folder/name;assert path.stat().st_size==meta['bytes']and sha(path)==meta['sha256']
            raw=torch.load(path,map_location='cpu',weights_only=True);case=raw['case'];initial=raw['initial_state'];closed=raw['closed'];replay=raw['replay'];case_id=f"{key}_slot{case['slot']}"
            case_meta.append(dict(model=key,slot=case['slot'],anchor=case['anchor'],horizon=case['horizon'],replay_horizon=case['replay_horizon'],**case['metadata']))
            active=closed['active'];initial_pos=raw['initial_position'];goal=raw['goal']
            for mode_i,mode in enumerate(MODES):
                base=mode_i*7
                # Mode effects use unpulsed branches and the same fixed elapsed time.
                pos=closed['pos'][-1,base]
                shared=active[:,base]&active[:,0]
                tv=(closed['logits'][:,base].softmax(1)-closed['logits'][:,0].softmax(1)).abs().sum(1)/2
                mode_effects.append(dict(case_id=case_id,model=key,mode=mode,
                    goal_progress_cm=100*float((initial_pos-goal).norm()-(pos-goal).norm()),
                    progress_difference_from_native_cm=100*float((closed['pos'][-1,0]-goal).norm()-(pos-goal).norm()),
                    hit=bool(closed['hit'][:,base].any()),path_cm=100*float(closed['path_step'][:,base].sum()),
                    action_difference_fraction=float((closed['action'][shared,base]!=closed['action'][shared,0]).float().mean())if bool(shared.any())else None,
                    mean_action_tv=float(tv[shared].mean())if bool(shared.any())else None,shared_updates=int(shared.sum())))
                for pulse_i,pulse in enumerate(PULSES):
                    branch=base+pulse_i
                    actions=closed['action'][:,branch];valid=active[:,branch]
                    adjacent=valid[1:]&valid[:-1]&(((actions[:-1]==1)&(actions[1:]==5))|((actions[:-1]==5)&(actions[1:]==1)))
                    times=torch.nonzero(adjacent).flatten()
                    transition_rows.append(dict(case_id=case_id,model=key,mode=mode,pulse=pulse,
                        first_adjacent_right_turn_strafe_switch_update=int(times[0])+2 if len(times)else None,
                        switch_observed=bool(len(times)),observed_active_updates=int(valid.sum()),
                        definition='First adjacent1<->5 action change before terminal; 1=right turn,5=right strafe. Absent switches are censored, not zero latency.'))
                    if pulse_i==0:continue
                    family=pulse.split('_')[0]
                    delta0=initial[branch]-initial[base];norm0=float((delta0/case['scale']).square().mean().sqrt());h0=float(delta0[:256].square().mean().sqrt())
                    for stream,values in [('closed',closed),('replay',replay)]:
                        for horizon in (1,5,15,30):
                            if horizon>len(values['state']):continue
                            age=horizon-1
                            available=(stream=='replay')or bool((active[:horizon,branch]&active[:horizon,base]).all())
                            closed_common,joint,all_directions_joint=joint_availability(active,len(replay['state']),horizon,pulse_i)
                            common=(stream=='replay')or closed_common
                            delta=values['state'][age,branch]-values['state'][age,base]
                            tv=(values['logits'][age,branch].softmax(0)-values['logits'][age,base].softmax(0)).abs().sum()/2
                            actor_delta=values['logits'][age,branch]-values['logits'][age,base];actor_delta-=actor_delta.mean()
                            response=dict(case_id=case_id,model=key,mode=mode,pulse=pulse,family=family,stream=stream,horizon=horizon,
                                own_pair_available=available,all_modes_available=common,joint_streams_all_modes_available=joint,joint_all_directions_available=all_directions_joint,
                                state_gain=float((delta/case['scale']).square().mean().sqrt())/norm0,
                                h_gain=float(delta[:256].square().mean().sqrt())/h0 if h0>0 else None,
                                h_rms=float(delta[:256].square().mean().sqrt()),c_rms=float(delta[256:].square().mean().sqrt()),centered_actor_rms=float(actor_delta.square().mean().sqrt()),action_tv=float(tv))
                            if stream=='closed':
                                mask=active[:horizon,branch]&active[:horizon,base]
                                response['action_difference_fraction']=float((values['action'][:horizon,branch][mask]!=values['action'][:horizon,base][mask]).float().mean())if bool(mask.any())else None
                                response['pulse_goal_progress_difference_cm']=100*float((values['pos'][age,base]-goal).norm()-(values['pos'][age,branch]-goal).norm())
                            all_rows.append(response)
        # Fixed separate seeds, no trial-level population p-values.
        model_rows=[r for r in all_rows if r['model']==key]
        grouped={};joint_same_pulse={};all_available={}
        for stream in ('closed','replay'):
            for mode in MODES:
                for family in ('turn','drift','normal'):
                    for horizon in (1,5,15,30):
                        for destination,flag in ((grouped,'joint_all_directions_available'),(joint_same_pulse,'joint_streams_all_modes_available'),(all_available,'own_pair_available')):
                            rows=[r for r in model_rows if r['stream']==stream and r['mode']==mode and r['family']==family and r['horizon']==horizon and r[flag]]
                            destination[f'{stream}|{mode}|{family}|{horizon}']={metric:stats([r[metric]for r in rows if r.get(metric)is not None])for metric in ('state_gain','h_gain','h_rms','c_rms','centered_actor_rms','action_tv','action_difference_fraction','pulse_goal_progress_difference_cm')}
        effects={mode:{metric:stats([r[metric]for r in mode_effects if r['model']==key and r['mode']==mode and r.get(metric)is not None])for metric in ('goal_progress_cm','progress_difference_from_native_cm','hit','path_cm','action_difference_fraction','mean_action_tv')}for mode in MODES}
        switches={f'{mode}|{pulse}':dict(cases=sum(r['model']==key and r['mode']==mode and r['pulse']==pulse for r in transition_rows),observed=stats([r['first_adjacent_right_turn_strafe_switch_update']for r in transition_rows if r['model']==key and r['mode']==mode and r['pulse']==pulse and r['switch_observed']]))for mode in MODES for pulse in PULSES}
        aggregate[key]=dict(inputs[key],responses=grouped,joint_same_pulse_responses=joint_same_pulse,all_available_responses=all_available,unpulsed_mode_effects=effects,adjacent_turn_strafe_switches=switches)
    result=dict(complete=True,source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(MANIFEST),
        inputs=inputs,models=aggregate,case_metadata=case_meta,response_rows=all_rows,mode_effect_rows=mode_effects,transition_rows=transition_rows,
        interpretation='Observed turn-trajectory state impulse response. No imposed cyclic closure or validated ring phase; all eligible and missing episodes retained. Same-mode sham comparisons; primary curves retain the same cases across closed loop, replay, every input route and every signed direction at each horizon; less-restricted aggregates retained separately. Two discovery seeds, no population significance claims.')
    atomic_json(out/'summary.json',result)
    render(result,out)
    print(json.dumps(dict(complete=True,models={k:dict(eligible=v['eligible'],normal30=v['responses']['closed|native|normal|30'],modeeffects=v['unpulsed_mode_effects'])for k,v in aggregate.items()})),flush=True)


if __name__=='__main__':main()
