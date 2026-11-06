"""Presentation-only summary of the two completed, fixed original-memory runs."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import native_turn_original_memory as study
from rl_goal_memory import atomic_json

LABELS=('Intact','Ordinary cue','Cue +60°','Cue −60°','Erase before native cue',
    'Erase before ordinary cue','Erase at choice: native','Erase at choice: ordinary')
COLORS={'gc100':'#2878b5','pgc33':'#e87524'}
NAMES={'gc100':'Grid','pgc33':'PGC + grid'}


def verified_results():
    design=json.loads(study.PROTOCOL.read_text());study.validate_frozen(design)
    models={};sources={}
    for condition in study.parent.CONDITIONS:
        folder=study.BASE/condition;path=folder/'summary.json';result=json.loads(path.read_text())
        assert result['complete']and result['condition']==condition
        assert result['protocol_sha256']==study.parent.sha(study.PROTOCOL)
        assert result['source_unchanged']and result['policy_unchanged']and result['global_cpu_rng_preserved']
        assert result['no_optimization']and result['no_cloud']and result['qualification']is None
        for name,record in result['artifacts'].items():
            p=folder/name;assert study.parent.sha(p)==record['sha256']and p.stat().st_size==record['bytes']
        for kind in study.parent.KINDS:
            for variant in study.VARIANTS:
                rows=[r for r in result['rows']if r['kind']==kind and r['variant']==variant]
                pairs=[r for r in result['pairs']if r['kind']==kind and r['variant']==variant]
                assert len(rows)==256 and len(pairs)==128
                assert len({(r['context'],r['case'])for r in rows})==256
                assert len({(r['context'],r['pair'])for r in pairs})==128
                declared=result['summary'][kind][variant]
                for key in ('teacher_forward_kl_nats','expected_progress_m','turn_weighted_body_goal_alignment_rad'):
                    assert abs(sum(r[key]for r in rows)/256-declared[key])<1e-12
                for key in ('best_common_action_ceiling_m','progress_above_common_ceiling_m'):
                    assert abs(sum(r[key]for r in pairs)/128-declared[key])<1e-12
        models[condition]=result;sources[str(path.relative_to(ROOT))]=study.parent.sha(path)
    return models,sources


def make_figure(models,kind,out):
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True,gridspec_kw={'wspace':.10})
    y=np.arange(len(study.VARIANTS));height=.34
    for j,condition in enumerate(study.parent.CONDITIONS):
        summary=models[condition]['summary'][kind];offset=(j-.5)*height
        progress=[summary[v]['expected_progress_m']*1000 for v in study.VARIANTS]
        advantage=[summary[v]['memoryless_teacher_kl_floor_nats']-summary[v]['teacher_forward_kl_nats']for v in study.VARIANTS]
        axes[0].barh(y+offset,progress,height,color=COLORS[condition],label=NAMES[condition])
        axes[1].barh(y+offset,advantage,height,color=COLORS[condition])
        # Thin reference is the actual mean saved best-common-action ceiling.
        ceiling=summary['native']['best_common_action_ceiling_m']*1000
        axes[0].axvline(ceiling,color=COLORS[condition],linestyle=':',lw=1.2,alpha=.8)
    axes[0].set_yticks(y,LABELS);axes[0].invert_yaxis()
    axes[0].set_xlabel('Expected first-step goalward displacement (mm)')
    axes[1].set_xlabel('Memoryless teacher-KL floor − observed KL (nats)')
    axes[0].set_title('Useful first-choice history')
    axes[1].set_title('Teacher imitation beyond current input')
    for ax in axes:
        ax.axvline(0,color='#444444',lw=.8);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='lower right',frameon=False)
    title='One predictive travel cue, then eight ordinary inputs'if kind=='probe_glimpse'else'Turn-only history: cue changes are exact shams'
    fig.suptitle(title,fontsize=14,y=.98)
    fig.text(.5,.025,'One original seed pair · 16 contexts × 8 opposite-heading pairs per condition · all cases retained',ha='center',fontsize=9)
    fig.subplots_adjust(left=.22,right=.98,bottom=.15,top=.88)
    for ext in ('png','pdf'):fig.savefig(out/f'{kind}.{ext}',dpi=180)
    plt.close(fig)


def main():
    models,sources=verified_results();out=study.BASE/'report';out.mkdir(parents=True,exist_ok=True)
    contrasts={}
    for condition,result in models.items():
        contrasts[condition]={}
        for kind,summary in result['summary'].items():
            contrasts[condition][kind]={}
            for name,a,b in (('predictive_cue','native','ordinary'),
                    ('predictive_cue_after_erasure','pre_cue_reset_native','pre_cue_reset_ordinary'),
                    ('history_before_choice','native','choice_reset_native')):
                contrasts[condition][kind][name]=dict(
                    expected_progress_difference_mm=1000*(summary[a]['expected_progress_m']-summary[b]['expected_progress_m']),
                    teacher_kl_reduction_nats=summary[b]['teacher_forward_kl_nats']-summary[a]['teacher_forward_kl_nats'])
    for kind in study.parent.KINDS:make_figure(models,kind,out)
    record=dict(complete=True,protocol_sha256=study.parent.sha(study.PROTOCOL),input_summary_sha256=sources,
        producer_sha256=study.parent.sha(Path(__file__)),all_fixed_variants=list(study.VARIANTS),
        summaries={c:r['summary']for c,r in models.items()},within_model_contrasts=contrasts,
        scientific_gate=None,no_optimization=True,no_new_environment_rollout=True,no_cloud=True,
        scope='One original seed pair; controlled identical-current-input first choice. No population inference or ring criterion.',
        figures={p.name:study.parent.sha(p)for p in sorted(out.iterdir())if p.suffix in ('.png','.pdf')})
    atomic_json(out/'summary.json',record)
    lines=['# Original recurrent memory and predictive cue use','',
        'We replayed the original networks through the same controlled turning histories. At the measured choice, every heading received the same complete input. Only the earlier cue or recurrent-state erasure changed. No weights were trained.','',
        '![Probe-cue results](probe_glimpse.png)','',
        '**Left:** expected goalward displacement on the next native action. The dotted lines mark the best shared action for each opposite-heading pair, averaged over all pairs. Exceeding this reference means history helps this controlled first choice.','',
        '**Right:** the best memoryless teacher-KL floor minus the observed teacher KL. Positive values mean the network imitates the fully informed teacher better than any common action distribution could. This is a stronger imitation comparison; useful movement does not require it.','',
        '| Model | Intact step (mm) | Ordinary cue (mm) | Native cue after erasure (mm) | Ordinary cue after erasure (mm) | Choice erasure (mm) |',
        '|---|---:|---:|---:|---:|---:|']
    for c in study.parent.CONDITIONS:
        s=models[c]['summary']['probe_glimpse']
        lines.append('| '+NAMES[c]+' | '+' | '.join(f"{s[v]['expected_progress_m']*1000:.4f}"for v in ('native','ordinary','pre_cue_reset_native','pre_cue_reset_ordinary','choice_reset_native'))+' |')
    lines+=['','The cue changes only the 169 predictive current-code entries in PGC models. Saved goal code remains native. Rotations use measured travel direction, which can differ from body heading after the common turns. All cue substitutions are exact shams in Grid models and in the turn-only histories.','',
        'Erasure before the ordinary cue leaves exactly identical states and logits across all 16 headings. Erasure at the choice does the same. These are direct controls for the information supplied by the predictive cue and the intervening recurrent computation.','',
        '## All fixed conditions','',
        '| History | Model | Condition | Step (mm) | Teacher KL (nats) | Memoryless KL floor (nats) | Cases / pairs |',
        '|---|---|---|---:|---:|---:|---:|']
    for kind in study.parent.KINDS:
        for c in study.parent.CONDITIONS:
            for v,label in zip(study.VARIANTS,LABELS):
                s=models[c]['summary'][kind][v]
                lines.append(f"| {kind} | {NAMES[c]} | {label} | {s['expected_progress_m']*1000:.4f} | {s['teacher_forward_kl_nats']:.5f} | {s['memoryless_teacher_kl_floor_nats']:.5f} | 256 / 128 |")
    lines+=['','![Turn-only controls](turn_only.png)','',
        'These comparisons describe useful history and cue processing in one original seed pair under the controlled input conditions. They do not by themselves identify a ring or establish its stability. All contexts and outcomes are retained; no statistical threshold selected a model or condition.','',
        'The initial recurrent state was recomputed from each checkpoint’s complete native input prefix, with the original terminal resets. The original cached states matched exactly. Raw first-choice states, logits, six-action physics, references and hashes are retained beside this report.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
