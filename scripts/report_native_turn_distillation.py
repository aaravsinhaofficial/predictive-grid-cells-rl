"""Saved-data report of every fixed native-turn distillation endpoint.

Never runs a policy, environment, optimization step, or geometry analysis.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import native_turn_distillation as train
import evaluate_native_turn_distillation as evaluation
import native_turn_original_memory as fixed
from rl_goal_memory import atomic_json,episode_summary

CONDITIONS=('gc100','pgc33')
COLORS={'gc100':'#2878b5','pgc33':'#e87524'}
NAMES={'gc100':'Grid','pgc33':'PGC + grid'}
ENDPOINTS=('original','clean','noise')
LABELS=('Original','180 updates\nclean','180 updates\nstate noise')


def digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,type(a))and len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b


def verify_artifacts(folder,result):
    for name,record in result['artifacts'].items():
        p=folder/name;assert p.is_file()and train.dist.sha(p)==record['sha256']and p.stat().st_size==record['bytes'],p


def complete_execution(condition,arm):
    path=train.BASE/f'{condition}_{arm}_execution.json';record=json.loads(path.read_text())
    assert record['condition']==condition and record['arm']==arm and record['complete']
    assert record['protocol_sha256']==train.dist.sha(train.PROTOCOL)
    assert record['exit_code']==0 and record['no_cloud']
    return dict(path=str(path.relative_to(ROOT)),sha256=train.dist.sha(path))


def paired_summary(folder,result,original_reference=None):
    assert result['complete']and len(result['rows'])==4096 and len(result['pairs'])==2048
    assert result['condition']in CONDITIONS
    verify_artifacts(folder,result);answer={}
    for kind in fixed.parent.KINDS:
        answer[kind]={};floor=result['exact_input_groups'][kind]['case_weighted_minimum_forward_kl_nats']
        if original_reference is not None:
            assert floor==original_reference['exact_input_groups'][kind]['case_weighted_minimum_forward_kl_nats']
        for variant in fixed.VARIANTS:
            rows=[r for r in result['rows']if r['kind']==kind and r['variant']==variant]
            pairs=[r for r in result['pairs']if r['kind']==kind and r['variant']==variant]
            assert len(rows)==256 and len(pairs)==128
            assert len({(r['context'],r['case'])for r in rows})==256
            assert len({(r['context'],r['pair'])for r in pairs})==128
            answer[kind][variant]=dict(cases=256,pairs=128,teacher_kl=float(np.mean([r['teacher_forward_kl_nats']for r in rows])),
                progress_mm=1000*float(np.mean([r['expected_progress_m']for r in rows])),
                common_ceiling_mm=1000*float(np.mean([r['best_common_action_ceiling_m']for r in pairs])),
                teacher_memoryless_kl_floor=floor,
                pair_total_variation=float(np.mean([r['student_pair_total_variation']for r in pairs])))
    # Check the plot's actual paired endpoint values against saved probabilities
    # and six-action physics, not just a precomputed JSON average.
    rows={(r['context'],r['kind'],r['variant'],r['case']):r for r in result['rows']}
    for context in range(16):
        for kind in fixed.parent.KINDS:
            raw=torch.load(folder/f'context{context:02d}_{kind}.pt',map_location='cpu',weights_only=True)
            original_raw=torch.load(fixed.parent.BASE/result['condition']/f'context{context:02d}_{kind}.pt',map_location='cpu',weights_only=True)
            assert torch.equal(raw['teacher_first_choice_probabilities'],original_raw['teacher_first_choice_probabilities'])
            assert equal(raw['first_update_physics'],original_raw['first_update_physics'])
            teacher=raw['teacher_first_choice_probabilities'].double();teacher/=teacher.sum(1,keepdim=True)
            physics=raw['first_update_physics']['progress_m'].double();tiny=torch.finfo(torch.float64).tiny
            for variant,saved in raw['variants'].items():
                assert torch.equal(saved['first_choice_input'],original_raw['student_native_x'][-1])
                p=saved['first_choice_probabilities'].double();p/=p.sum(1,keepdim=True)
                kl=(teacher*(teacher.clamp_min(tiny).log()-p.clamp_min(tiny).log())).sum(1)
                progress=(p*physics).sum(1)
                for case in range(16):
                    row=rows[context,kind,variant,case]
                    assert abs(float(kl[case])-row['teacher_forward_kl_nats'])<1e-10
                    assert abs(float(progress[case])-row['expected_progress_m'])<1e-12
            for variant in ('choice_reset_native','choice_reset_ordinary','pre_cue_reset_ordinary'):
                for key in ('first_choice_state_after_input','first_choice_logits'):
                    v=raw['variants'][variant][key];assert torch.equal(v,v[:1].expand_as(v))
    return answer


def live_summary(result,condition):
    assert result['complete']and result['episodes_per_variant']==16
    assert set(result['variants'])==set(evaluation.LIVE_VARIANTS)
    first_discovery=None
    for index,name in enumerate(evaluation.LIVE_VARIANTS):
        value=result['variants'][name]
        rows=value['rows'];assert len(rows)==16 and {r['slot']for r in rows}==set(range(16))
        assert episode_summary(rows,1350)==value
        discovered=[r['first_goal_step']for r in sorted(rows,key=lambda r:r['slot'])]
        if first_discovery is None:first_discovery=discovered
        else:assert discovered==first_discovery
        exposure=result['exposure'][name];assert sum(exposure.values())==16*1350
        phases=result['event_clock']['phase_update_counts'][index]
        assert len(phases)==7 and sum(phases)==16*1350
        assert phases[6]==exposure['choice']and sum(phases[2:6])==exposure['forced']
        assert sum(phases[:2])==exposure['other_control']
        assert result['choice_erasure_counts'][name]==(exposure['choice']if'reset'in name else 0)
        assert math.isclose(sum(result['reward_by_phase'][name]),sum(r['score']for r in rows),rel_tol=1e-10,abs_tol=1e-8)
        assert result['input_audit']['scheduled_blind_updates'][index]==sum(phases[2:])
        for kind,count in exposure.items():
            measured=result['mean_teacher_kl'][name][kind]
            assert (measured is None)==(count==0)
            if count:assert abs(measured-result['teacher_kl_sums'][name][kind]/count)<1e-12
    if condition=='gc100':
        for a,b in (('blind_native','blind_ordinary'),('blind_reset_every_choice','blind_ordinary_reset_every_choice')):
            ia,ib=evaluation.LIVE_VARIANTS.index(a),evaluation.LIVE_VARIANTS.index(b)
            for key in ('variants','exposure','mean_teacher_kl','teacher_kl_sums','choice_erasure_counts','reward_by_phase'):
                assert result[key][a]==result[key][b],(key,a,b)
            for key in ('event_clock','input_audit'):
                assert all(value[ia]==value[ib]for value in result[key].values()),(key,a,b)
    score={v:x['mean_score']for v,x in result['variants'].items()}
    seen_cost=score['seen_native']-score['seen_reset_every_choice']
    native_cost=score['blind_native']-score['blind_reset_every_choice']
    ordinary_cost=score['blind_ordinary']-score['blind_ordinary_reset_every_choice']
    return dict(scores=score,visible_erasure_cost=seen_cost,native_blind_erasure_cost=native_cost,
        ordinary_blind_erasure_cost=ordinary_cost,additional_native_blind_erasure_cost=native_cost-seen_cost,
        additional_ordinary_blind_erasure_cost=ordinary_cost-seen_cost,exposure=result['exposure'],
        choice_erasure_counts=result['choice_erasure_counts'],mean_teacher_kl=result['mean_teacher_kl'],
        variants=result['variants'],reward_by_phase=result['reward_by_phase'])


def load_all():
    design=json.loads(train.PROTOCOL.read_text());protocol_hash=train.dist.sha(train.PROTOCOL)
    for key in ('source_sha256','asset_sha256'):
        for name,sha in design[key].items():assert train.dist.sha(ROOT/name)==sha,name
    results={};references={};final_states={}
    for condition in CONDITIONS:
        original_path=ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
        original=torch.load(original_path,map_location='cpu',weights_only=True)['policy']
        original_hash=digest(original);cfg=json.loads(original_path.with_suffix('.json').read_text())
        initial_paired=json.loads((fixed.BASE/condition/'summary.json').read_text())
        initial_paired_values=paired_summary(fixed.BASE/condition,initial_paired)
        baseline=train.BASE/condition/'original_baseline';b=json.loads((baseline/'summary.json').read_text())
        assert b['complete']and b['policy_hash']==original_hash and b['protocol_sha256']==protocol_hash
        assert b['live_sha256']==train.dist.sha(baseline/'live.json')
        initial_live=live_summary(json.loads((baseline/'live.json').read_text()),condition)
        results[condition]={'original':dict(paired=initial_paired_values,live=initial_live)}
        references[str((baseline/'summary.json').relative_to(ROOT))]=train.dist.sha(baseline/'summary.json')
        for arm in train.ARMS:
            execution=complete_execution(condition,arm)
            folder=train.BASE/condition/arm;s=json.loads((folder/'summary.json').read_text())
            assert s['complete']and s['condition']==condition and s['arm']==arm
            assert s['updates']==180 and s['extra_frames']==1036800 and s['policy_decisions']==259200
            assert s['protocol_sha256']==protocol_hash and s['fixed_tensors_unchanged']and s['teacher_unchanged']
            assert s['original_checkpoint_sha256']==train.dist.sha(original_path)
            assert s['original_config_sha256']==train.dist.sha(original_path.with_suffix('.json'))
            verify_artifacts(folder,s)
            metrics=[json.loads(v)for v in(folder/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics)==180 and [r['update']for r in metrics]==list(range(1,181))
            for row in metrics:
                assert all(math.isfinite(row[k])for k in ('loss','grad_norm_before_clip','parameter_update_l2'))
                assert row['replay_max_error']==row['teacher_replay_max_error']==0
                assert row['choice_count']+row['control_count']+row['forced_count']==1440
                assert sum(row['phase_counts'])==1440 and row['phase_counts'][6]==row['choice_count']
                assert sum(row['phase_counts'][2:6])==row['forced_count']
                expected=.5*(row['choice_kl']+row['control_kl'])if row['choice_count']else row['control_kl']
                assert math.isclose(row['loss'],expected,rel_tol=2e-6,abs_tol=2e-6)
                assert row['no_choice_fallback']==(row['choice_count']==0)
            assert sum(len(r['completed_episode_scores'])for r in metrics)==192
            checkpoints={name:torch.load(folder/name,map_location='cpu',weights_only=True)for name in
                ('initial.pt','stage1.pt','stage2.pt','stage3.pt','final.pt')}
            assert equal(checkpoints['initial.pt']['policy'],original)
            for name,updates in (('initial.pt',0),('stage1.pt',60),('stage2.pt',120),('stage3.pt',180),('final.pt',180)):
                cp=checkpoints[name]
                assert cp['updates']==updates and cp['extra_frames']==updates*16*90*4 and cp['config']==cfg
                assert cp['protocol_sha256']==protocol_hash and cp['arm']==arm and cp['teacher_policy_hash']==original_hash
                assert cp['policy'].keys()==original.keys()
                assert all(torch.equal(v,cp['policy'][k])for k,v in original.items()if not k.startswith('lstm.'))
            final=checkpoints['final.pt'];assert equal(checkpoints['stage3.pt'],final)
            assert digest(final['policy'])==s['final_policy_hash']and s['initial_policy_hash']==original_hash
            assert len(final['optimizer']['state'])==4 and all(float(v['step'])==180 for v in final['optimizer']['state'].values())
            delta={k:float((v.double()-final['policy'][k].double()).norm())for k,v in original.items()if k.startswith('lstm.')}
            assert len(delta)==4 and sum(delta.values())>0
            final_states[condition,arm]={key:final[key]for key in ('action_rng','mixture_rng','event_tapes')}
            paired_result=json.loads((folder/'final_paired/summary.json').read_text())
            assert paired_result['policy_hash']==s['final_policy_hash']and paired_result['own_prefix_recomputed']
            pair=paired_summary(folder/'final_paired',paired_result,initial_paired)
            live=live_summary(json.loads((folder/'final_live.json').read_text()),condition)
            calendar=(dt.datetime.fromisoformat(s['end_utc'])-dt.datetime.fromisoformat(s['start_utc'])).total_seconds()
            results[condition][arm]=dict(paired=pair,live=live,metrics=metrics,changed_lstm_l2=delta,
                execution=execution,
                acquisition_monotonic_seconds=s['acquisition_monotonic_seconds'],evaluation_monotonic_seconds=s['evaluation_monotonic_seconds'],
                calendar_seconds=calendar,final_checkpoint_sha256=s['artifacts']['final.pt']['sha256'],summary_sha256=train.dist.sha(folder/'summary.json'))
            references[str((folder/'summary.json').relative_to(ROOT))]=train.dist.sha(folder/'summary.json')
            references[execution['path']]=execution['sha256']
    first=next(iter(final_states.values()))
    common={key:all(equal(first[key],s[key])for s in final_states.values())for key in first}
    return design,results,references,common


def plots(models,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    x=np.arange(3);width=.34
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for i,kind in enumerate(('probe_glimpse','turn_only')):
        for j,c in enumerate(CONDITIONS):
            values=[models[c][e]['paired'][kind]['native']for e in ENDPOINTS]
            axes[i,0].bar(x+(j-.5)*width,[v['progress_mm']for v in values],width,color=COLORS[c],label=NAMES[c])
            axes[i,0].axhline(values[0]['common_ceiling_mm'],color=COLORS[c],ls=':',lw=1)
            axes[i,1].bar(x+(j-.5)*width,[v['teacher_memoryless_kl_floor']-v['teacher_kl']for v in values],width,color=COLORS[c])
            teacher=json.loads((fixed.parent.BASE/c/'summary.json').read_text())
            target=1000*np.mean([p['teacher_mean_expected_progress_m']for p in teacher['pairs']if p['kind']==kind])
            axes[i,0].axhline(target,color=COLORS[c],ls='--',lw=1,label=NAMES[c]+' teacher')
        axes[i,0].set_ylabel(('Probe cue'if i==0 else'Turn only')+'\nGoalward first step (mm)')
        axes[i,1].set_ylabel('Memoryless KL floor − observed KL (nats)')
        for ax in axes[i]:
            ax.set_xticks(x,LABELS);ax.axhline(0,color='#555',lw=.8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('The same controlled first choice, before and after learning')
    fig.tight_layout();fig.savefig(out/'first_choice.png',dpi=180);fig.savefig(out/'first_choice.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,c in enumerate(CONDITIONS):
        for column,(a,b)in enumerate((('native','ordinary'),('pre_cue_reset_native','pre_cue_reset_ordinary'))):
            values=[models[c][e]['paired']['probe_glimpse'][a]['progress_mm']-models[c][e]['paired']['probe_glimpse'][b]['progress_mm']for e in ENDPOINTS]
            axes[column].bar(x+(j-.5)*width,values,width,color=COLORS[c],label=NAMES[c])
    for ax,title in zip(axes,('Predictive cue benefit','Predictive cue after state erasure')):
        ax.set_title(title);ax.set_xticks(x,LABELS);ax.set_ylabel('Native cue − ordinary cue (mm)')
        ax.axhline(0,color='#555',lw=.8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'cue_controls.png',dpi=180);fig.savefig(out/'cue_controls.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    keys=('visible_erasure_cost','native_blind_erasure_cost','ordinary_blind_erasure_cost')
    for ax,key,title in zip(axes,keys,('Visible, native cue','Blind, native cue','Blind, ordinary cue')):
        for j,c in enumerate(CONDITIONS):ax.bar(x+(j-.5)*width,[models[c][e]['live'][key]for e in ENDPOINTS],width,color=COLORS[c],label=NAMES[c])
        ax.set_title(title);ax.set_xticks(x,LABELS);ax.axhline(0,color='#555',lw=.8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Return lost after erasing each choice state');axes[0].legend()
    fig.tight_layout();fig.savefig(out/'live_erasure.png',dpi=180);fig.savefig(out/'live_erasure.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for c in CONDITIONS:
        for arm,style in (('clean','-'),('noise','--')):
            rows=models[c][arm]['metrics']
            for ax,field,count in zip(axes,('choice_kl','control_kl'),('choice_count','control_count')):
                values=[]
                for i in range(180):
                    rr=rows[max(0,i-9):i+1];den=sum(r[count]for r in rr)
                    values.append(sum((r[field]or 0)*r[count]for r in rr)/den if den else np.nan)
                ax.plot(np.arange(1,181),values,color=COLORS[c],ls=style,label=f'{NAMES[c]} / {arm}')
    for ax,title in zip(axes,('Ambiguous choices','Other policy-controlled decisions')):
        ax.set_title(title);ax.set_xlabel('Fixed optimizer update');ax.set_ylabel('Teacher KL,10-update exposure-weighted mean')
        for step in (60,120):ax.axvline(step,color='#888',ls=':',lw=1)
        ax.grid(alpha=.18);ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'learning.png',dpi=180);fig.savefig(out/'learning.pdf');plt.close(fig)


def main():
    design,models,references,common=load_all();out=train.BASE/'report';out.mkdir(parents=True,exist_ok=True)
    plots(models,out)
    record=dict(complete=True,protocol_sha256=train.dist.sha(train.PROTOCOL),producer_sha256=train.dist.sha(Path(__file__)),
        models=models,terminal_summary_sha256=references,saved_final_private_stream_equality=common,
        private_stream_scope='Final saved generators and indexed episode/event tapes, not a claim that action-dependent event exposures or trajectories matched.',
        scope='All four fixed180-update continuations from one original seed pair; primary causal comparisons are within-model cue/state interventions.',
        no_new_rollout=True,no_optimization=True,figures={p.name:train.dist.sha(p)for p in out.iterdir()if p.suffix in('.png','.pdf')})
    atomic_json(out/'summary.json',record)
    lines=['# Teaching angular history in the original navigation RNN','',
        'Four fixed continuations trained only the original LSTM: Grid and PGC+grid, each with clean or weakly perturbed recurrent states. The original fully informed policies supplied action targets on the actual visited trajectories. CNNs, action heads, critics, maps and six-action physics were retained. The noisy arm applies two small incoming-state perturbations during cue retention, at the single predeclared scale of10% of each original state block’s native one-update change.','',
        '![First choice](first_choice.png)','',
        'The measured first choice has identical current inputs across opposite headings. Positive goalward displacement above the saved common-action reference reflects useful history. The separate KL panel compares imitation with the best memoryless teacher-KL floor; positive values beat that floor. Teacher dashed lines put the movement scale in context.','',
        '| Model | Endpoint | Native cue step (mm) | Ordinary cue step (mm) | Erase before native cue (mm) | Erase before ordinary cue (mm) | Erase at choice (mm) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for e in ENDPOINTS:
            p=models[c][e]['paired']['probe_glimpse']
            lines.append('| '+NAMES[c]+' | '+e+' | '+' | '.join(f"{p[v]['progress_mm']:.6f}"for v in
                ('native','ordinary','pre_cue_reset_native','pre_cue_reset_ordinary','choice_reset_native'))+' |')
    lines+=['','![Cue controls](cue_controls.png)','',
        'The cue comparisons replace only the predictive current-code entries at the measured probe. Saved goal code remains native. The repair comparison erases state before delivering that cue, then withdraws it before the identical choice input. Ordinary-cue reset and choice-reset controls have exactly identical states/logits across all16 phases.','',
        '![Live erasure](live_erasure.png)','',
        'The visible and blind live controls share held-code/cue and forced-turn/brake scheduling rules. Visibility changes RGB only. These are modified-task controls, not original unmodified navigation scores. Actual event times/exposures can diverge after actions and hits; the table retains their denominators.','',
        '| Model | Endpoint | Live branch | Mean return | Choice / other / forced inputs | Erased choices |',
        '|---|---|---|---:|---:|---:|']
    for c in CONDITIONS:
        for e in ENDPOINTS:
            live=models[c][e]['live']
            for v in evaluation.LIVE_VARIANTS:
                count=live['exposure'][v]
                lines.append(f"| {NAMES[c]} | {e} | {v} | {live['scores'][v]:.3f} | {count['choice']} / {count['other_control']} / {count['forced']} | {live['choice_erasure_counts'][v]} |")
    lines+=['','![Learning](learning.png)','',
        'All180 update losses/gradients were finite and stored-input/noise replays were exact. Forced actions had zero direct loss, while later choices retained gradients through their full histories. Batches without ambiguous choices are retained and explicitly counted.','',
        '| Model | Arm | Choice inputs | Other control inputs | No-choice batches | Acquisition monotonic seconds | Evaluation monotonic seconds | Calendar seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for arm in train.ARMS:
            m=models[c][arm];rr=m['metrics']
            lines.append(f"| {NAMES[c]} | {arm} | {sum(r['choice_count']for r in rr)} | {sum(r['control_count']for r in rr)} | {sum(r['no_choice_fallback']for r in rr)} | {m['acquisition_monotonic_seconds']:.1f} | {m['evaluation_monotonic_seconds']:.1f} | {m['calendar_seconds']:.1f} |")
    lines+=['','## Every paired-history condition','',
        '| Model | Endpoint | History | Variant | First step (mm) | Common-action ceiling (mm) | Teacher KL | Memoryless KL floor | Cases / pairs |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for e in ENDPOINTS:
            for kind in fixed.parent.KINDS:
                for v,p in models[c][e]['paired'][kind].items():
                    lines.append(f"| {NAMES[c]} | {e} | {kind} | {v} | {p['progress_mm']:.6f} | {p['common_ceiling_mm']:.6g} | {p['teacher_kl']:.6f} | {p['teacher_memoryless_kl_floor']:.6f} |256 /128|")
    lines+=['','This is new recurrent policy distillation from one original seed pair. Across-population learning differences combine different teachers and initial weights. The within-model cue and erasure comparisons test predictive information and useful history. All four fixed finals also receive the separately registered native-turn geometry analysis, irrespective of these behavioral outcomes. No stage checkpoint was selected by performance.','',
        'Acquisition was capped at1800 monotonic seconds per run, with a separate600-second final evaluation cap. Shared original evaluations and geometry are separately timed; calendar durations are reported independently. No cloud compute was used.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
