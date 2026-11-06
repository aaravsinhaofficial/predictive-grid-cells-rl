"""Read-only report of both fixed recurrent-policy distillation endpoints."""
import json,hashlib,math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from native_policy_distillation import PROTOCOL,BASE,VARIANTS,sha
from rl_goal_memory import atomic_json


def digest(values):
    result=hashlib.sha256()
    for name,value in sorted(values.items()):result.update(name.encode());result.update(value.cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def gap_value(replay,name):
    return replay['scheduled_gap_kl'][name]if replay['scheduled_gap_counts'][name]>0 else float('nan')


def number(value,places=5):
    return f'{value:.{places}f}'if math.isfinite(value)else'Not measured (zero exposure)'


def validate_evaluations(before,after,replay_before,replay_after,condition):
    expected=set(VARIANTS)
    for behavior in (before,after):
        assert behavior['complete']and behavior['episodes_per_variant']==16 and set(behavior['variants'])==expected
        assert set(behavior['scheduled_gap_counts'])==expected
        for name,v in behavior['variants'].items():
            assert v['episodes']==16 and len(v['rows'])==16 and {r['slot']for r in v['rows']}==set(range(16))
            assert v['mean_score']==sum(r['score']for r in v['rows'])/16
            assert 0<=behavior['scheduled_gap_counts'][name]<=16*1350
    for replay in (replay_before,replay_after):
        assert replay['complete']and replay['episodes']==32 and replay['evaluation_only']
        assert replay['mask_sha256']==sha(BASE/'masks'/f'{condition}.pt')
        assert replay['trace_sha256']==sha(ROOT/f'results/native_movement_cycles/{condition}_s1/native_trace.pt')
        for key in ('mean_kl','scheduled_gap_kl','mean_action_tv','scheduled_gap_counts'):assert set(replay[key])==expected
        assert len(set(replay['scheduled_gap_counts'].values()))==1
        assert all(0<=v<=32*1350 for v in replay['scheduled_gap_counts'].values())
    for key in ('mask_sha256','trace_sha256','scheduled_gap_counts'):assert replay_before[key]==replay_after[key]


def main():
    out=BASE/'report';out.mkdir(exist_ok=True);models={}
    design=json.loads(PROTOCOL.read_text())
    for key in ('source_sha256','asset_sha256'):
        for name,value in design[key].items():assert sha(ROOT/name)==value,name
    execution=json.loads((BASE/'execution.json').read_text());assert execution['complete']
    for condition in ('gc100','pgc33'):
        folder=BASE/f'{condition}_s1';summary=json.loads((folder/'summary.json').read_text())
        assert summary['complete']and summary['updates']==180 and summary['learning_physics_frames']==1036800 and summary['protocol_sha256']==sha(PROTOCOL)
        for name,record in summary['artifacts'].items():
            path=folder/name;assert sha(path)==record['sha256']and path.stat().st_size==record['bytes']
        metrics=[json.loads(row)for row in(folder/'metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==180 and [r['update']for r in metrics]==list(range(1,181))
        assert all(all(math.isfinite(r[k])for k in ('loss','grad_norm_before_clip','parameter_update_l2'))and r['replay_max_error']==0 for r in metrics)
        assert sum(len(r['completed_episode_scores'])for r in metrics)==192
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True);final=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        original=torch.load(ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt',map_location='cpu',weights_only=True)
        assert set(initial['policy'])==set(final['policy'])==set(original['policy'])
        assert {k for k in final['policy']if k.startswith('lstm.')}=={'lstm.weight_ih','lstm.weight_hh','lstm.bias_ih','lstm.bias_hh'}
        assert all(torch.equal(v,original['policy'][k])for k,v in initial['policy'].items())
        assert all(torch.equal(v,final['policy'][k])for k,v in initial['policy'].items()if not k.startswith('lstm.'))
        assert digest(final['policy'])==summary['student_final_policy_hash']
        assert len(final['optimizer']['state'])==4 and all(float(s['step'])==180 for s in final['optimizer']['state'].values())
        change={k:float((v-final['policy'][k]).norm())for k,v in initial['policy'].items()if k.startswith('lstm.')};assert sum(change.values())>0
        for stage,updates in enumerate((60,120,180),1):
            saved=torch.load(folder/f'stage{stage}.pt',map_location='cpu',weights_only=True)
            assert saved['updates']==updates and saved['extra_frames']==updates*16*90*4 and saved['protocol_sha256']==sha(PROTOCOL)
            assert set(saved['policy'])==set(initial['policy'])
            assert all(torch.equal(v,saved['policy'][k])for k,v in initial['policy'].items()if not k.startswith('lstm.'))
            if stage==3:assert all(torch.equal(v,final['policy'][k])for k,v in saved['policy'].items())
        models[condition]=dict(summary_sha256=sha(folder/'summary.json'),seconds=summary['seconds'],metrics=metrics,changed_lstm_tensor_l2=change,
            baseline_behavior=json.loads((folder/'baseline_behavior.json').read_text()),final_behavior=json.loads((folder/'final_behavior.json').read_text()),
            baseline_replay=json.loads((folder/'baseline_replay.json').read_text()),final_replay=json.loads((folder/'final_replay.json').read_text()),
            exact_initial_original_and_frozen_final_tensors=True,all180_losses_gradients_finite=True,all180_replays_exact=True)
        model=models[condition];validate_evaluations(model['baseline_behavior'],model['final_behavior'],model['baseline_replay'],model['final_replay'],condition)
    result=dict(complete=True,source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),models=models,
        execution_sha256=sha(BASE/'execution.json'),runtime_clock_audit_sha256=sha(BASE/'runtime_clock_audit.json'),
        interpretation='New recurrent-only policy-distillation continuations. Same original physical task, but new dense supervision and observation curriculum. Both fixed finals receive geometry regardless of behavior. Across-population comparisons are confounded by different teachers and effective PGC augmentation; primary comparisons are within-model input and erasure contrasts. Live erasure clocks share rules and draws but can diverge after branch-specific hits; fixed replay masks are exactly matched.')
    atomic_json(out/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    for ax,(condition,model)in zip(axes,models.items()):
        loss=np.array([r['loss']for r in model['metrics']]);x=np.arange(1,181)
        ax.plot(x,loss,color='.75',lw=.7);ax.plot(np.arange(5.5,176.5),np.convolve(loss,np.ones(10)/10,'valid'),color='#2878b5',lw=2)
        for boundary in (60,120):ax.axvline(boundary,color='.5',ls=':',lw=1)
        ax.set_title('Grid'if condition=='gc100'else'PGC + grid');ax.set_xlabel('Optimizer update');ax.grid(alpha=.15)
    axes[0].set_ylabel('Teacher → student KL');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'learning.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True);names=('seen','blind','blind_current_grid');x=np.arange(3)
    for ax,(condition,model)in zip(axes,models.items()):
        for offset,period,color,label in [(-.18,'baseline','#a3a3a3','Original'),(.18,'final','#2878b5','Continued')]:
            vals=[model[f'{period}_behavior']['variants'][name]['mean_score']for name in names];ax.bar(x+offset,vals,.36,color=color,label=label)
        ax.set_title('Grid'if condition=='gc100'else'PGC + grid');ax.set_xticks(x,['Vision','Blackout','Blackout +\nordinary current']);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Mean native episode score');axes[1].legend(frameon=False);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'behavior.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    names=VARIANTS;x=np.arange(len(names))
    for ax,(condition,model)in zip(axes,models.items()):
        for offset,period,color,label in [(-.18,'baseline','#a3a3a3','Original'),(.18,'final','#2878b5','Continued')]:
            vals=[gap_value(model[f'{period}_replay'],name)for name in names];ax.bar(x+offset,vals,.36,color=color,label=label)
        ax.set_title('Grid'if condition=='gc100'else'PGC + grid');ax.set_xticks(x,['Vision','Blackout','Blackout +\nordinary','Blackout +\nerase','Blackout +\nordinary + erase','Vision +\nerase'],rotation=35,ha='right',fontsize=8);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Teacher mismatch (KL)');axes[1].legend(frameon=False);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'fixed_history.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True);x=np.arange(3)
    for ax,(condition,model)in zip(axes,models.items()):
        for offset,period,color,label in [(-.18,'baseline','#a3a3a3','Original'),(.18,'final','#2878b5','Continued')]:
            values=model[f'{period}_behavior']['contrasts'];ax.bar(x+offset,[values[k]for k in ('blind_reset_cost','ordinary_blind_reset_cost','seen_scheduled_reset_cost')],.36,color=color,label=label)
        ax.axhline(0,color='.3',lw=.7);ax.set_title('Grid'if condition=='gc100'else'PGC + grid');ax.set_xticks(x,['Blackout','Blackout +\nordinary current','Vision']);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Score lost after erasure');axes[1].legend(frameon=False);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'erasure_cost.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    lead=[]
    for condition,model in models.items():
        name='Grid'if condition=='gc100'else'PGC + grid'
        a=gap_value(model['baseline_replay'],'blind');b=gap_value(model['final_replay'],'blind')
        c=model['baseline_behavior']['variants']['blind']['mean_score'];d=model['final_behavior']['variants']['blind']['mean_score']
        lead.append(f'{name}: teacher mismatch on the unchanged blackout histories went from **{number(a,2)} to {number(b,2)} KL**; navigation score during blackouts went from **{c:.2f} to {d:.2f}** on the new episodes.')
    lines=['# Recurrent policy distillation: fixed local pilot','',*sum(([line,'']for line in lead),[]),
        'These are two individual models, one original training seed per population. Every fixed endpoint and held-out episode is retained; the trial counts do not provide replication across independently trained models.','',
        'Each original agent taught a copy of itself to act through brief visual blackouts. Only the existing LSTM weights changed; the CNN, actor, critic, native maps and saved goal code stayed fixed. Training matched the full-vision teacher’s six-action probabilities on the action history actually executed.','',
        'Both models completed the fixed 180 updates (1,036,800 native physics frames each). All losses and gradients were finite; every stored training unroll replayed exactly. The original and all three stage checkpoints are retained.','',
        '![Learning](learning.png)','',
        'The training task became harder at updates 60 and 120. The curve combines changing blackout durations and teacher-control rates; the fixed held-out comparisons below provide the before/after test.','',
        '![Behavior](behavior.png)','',
        '| Model | Condition | Original score | Continued score |','|---|---|---:|---:|']
    for condition,model in models.items():
        for name in VARIANTS:
            a=model['baseline_behavior']['variants'][name]['mean_score'];b=model['final_behavior']['variants'][name]['mean_score'];lines.append(f'| {condition} | {name} | {a:.2f} | {b:.2f} |')
    lines+=['','Each score retains 16 held-out native episodes. Erasure clears h/c before every scheduled black input; its visible control follows the same rule with vision available. These live branches share duration draws, but their own pickups can shift renewal times. Exposure counts and every episode remain in the JSON.','',
        '![Erasure cost](erasure_cost.png)','',
        '| Model | Blackout erasure cost, original → continued | Visible erasure cost, original → continued |','|---|---:|---:|']
    for condition,model in models.items():
        a=model['baseline_behavior']['contrasts'];b=model['final_behavior']['contrasts']
        lines.append(f"| {condition} | {a['blind_reset_cost']:.2f} → {b['blind_reset_cost']:.2f} | {a['seen_scheduled_reset_cost']:.2f} → {b['seen_scheduled_reset_cost']:.2f} |")
    lines+=['','Erasure cost is the intact score minus the reset score; positive values mean the reset hurts navigation. This disruption can reflect stored information and a changed recurrent-state distribution.','',
        '| Model | Live condition | Original scheduled exposures | Continued scheduled exposures |','|---|---|---:|---:|']
    for condition,model in models.items():
        for name in VARIANTS:
            a=model['baseline_behavior']['scheduled_gap_counts'][name];b=model['final_behavior']['scheduled_gap_counts'][name]
            lines.append(f'| {condition} | {name} | {a:.0f} ({100*a/21600:.1f}%) | {b:.0f} ({100*b/21600:.1f}%) |')
    lines+=['','Each live branch has 21,600 total updates. The vision branches retain RGB during their scheduled intervals.','',
        '![Fixed-history test](fixed_history.png)','',
        '| Model | Fixed replay condition | Original gap KL | Continued gap KL |','|---|---|---:|---:|']
    for condition,model in models.items():
        for name in VARIANTS:
            a=gap_value(model['baseline_replay'],name);b=gap_value(model['final_replay'],name);lines.append(f'| {condition} | {name} | {number(a)} | {number(b)} |')
    lines+=['','Fixed replay uses all 32 archived original episodes per model, with exactly matched inputs, masks and teacher targets before/after. These trajectories and masks were evaluation-only. Lower KL means closer teacher imitation on this unchanged input history.','',
        'Native current and goal messages remain available. Improved imitation or navigation alone is not a memory test: recurrent erasure and the visible-control/replay comparisons measure how much the new controller uses its history. Different original teachers and the effective PGC input replacement also prevent interpreting the across-population learning difference as a clean PGC causal effect.','',
        'Both fixed final models receive the existing direction-memory geometry screen irrespective of their behavioral results. These are newly learned continuations, separate from the completed frozen-agent findings. No cloud compute or training extension was used.','',
        'The frozen limit was 1,800 seconds measured by `time.monotonic`, per model. Calendar time was longer during an apparent host suspension; execution duration is not a claim of completion within 30 calendar minutes. [Clock audit](../runtime_clock_audit.json), [execution record](../execution.json).','',
        'Full metrics, raw episode tables, exposure counts, checkpoint integrity and parameter changes: [summary.json](summary.json).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({condition:dict(seconds=model['seconds'],behavior_before={k:v['mean_score']for k,v in model['baseline_behavior']['variants'].items()},behavior_after={k:v['mean_score']for k,v in model['final_behavior']['variants'].items()},replay_before=model['baseline_replay']['scheduled_gap_kl'],replay_after=model['final_replay']['scheduled_gap_kl'])for condition,model in models.items()}),flush=True)
if __name__=='__main__':main()
