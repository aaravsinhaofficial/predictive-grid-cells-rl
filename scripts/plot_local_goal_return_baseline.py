"""Display every sealed-bank reset-head baseline after complete raw auditing.

No policy, environment or training is run. This report refuses partial inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/local_goal_return_baseline'
OUT=BASE/'report'
BEHAVIOR=ROOT/'results/local_goal_return_design/behavior_protocol.json'
BEHAVIOR_SHA='4c7ba90dd79c8c0b5f8ac13ba89e4a7d6ed5bf0287556982ade07f911fc16e48'
AUDIT=ROOT/'results/local_goal_return_design/completed_baseline_audit.json'
CONDITIONS=('gc100','pgc33')
BANKS=('near_baseline','final_hard')
LABELS={'gc100':'Grid','pgc33':'PGC + grid','uniform':'Uniform actions','planner':'Heading-aware planner'}
BEHAVIOR_FIELDS=('context','heading_index','hit','hit_step','free_steps','free_path_m',
                 'free_initial_distance_m','progress_m','action_counts')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def summarize_rows(rows):
    assert len(rows)==256
    keys=[(r['context'],r['heading_index']) for r in rows]
    assert sorted(keys)==[(c,h) for c in range(16) for h in range(16)]
    counts=[sum(bool(r['hit']) for r in rows if r['context']==c) for c in range(16)]
    valid=[r for r in rows if not r.get('invalid',False)]
    return dict(trials=256,successes=sum(counts),success_fraction=sum(counts)/256,
        contexts=16,context_successes=counts,context_success_fraction=[n/16 for n in counts],
        invalid_trials=sum(r.get('invalid',False) for r in rows),
        mean_initial_free_entropy=float(np.mean([r['first_free_entropy'] for r in valid])) if valid else None,
        mean_path_m=float(np.mean([r.get('free_path_m',0.) for r in rows])),
        mean_progress_m=float(np.mean([r.get('progress_m',0.) for r in rows])),
        action_counts=np.sum([r.get('action_counts',[0]*6) for r in rows],axis=0).tolist())


def load_complete():
    required=[BASE/c/'summary.json' for c in CONDITIONS]+[BASE/c/b/'summary.json' for c in CONDITIONS for b in BANKS]+[AUDIT]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing,f'Wait for all completed writers and independent audit: {missing}'
    assert sha(BEHAVIOR)==BEHAVIOR_SHA
    design=read(BEHAVIOR);audit=read(AUDIT)
    assert audit['complete'] and audit['passed'],'Require complete artifact audit, regardless of learning outcomes'
    sources={str(BEHAVIOR.relative_to(ROOT)):sha(BEHAVIOR),str(AUDIT.relative_to(ROOT)):sha(AUDIT)}
    models={};initializations=[]
    for condition in CONDITIONS:
        folder=BASE/condition;top=read(folder/'summary.json');initial=read(folder/'initialization.json')
        assert top['complete'] and initial['retained_all_nonhead_parameters_exactly']
        assert initial['seed']==291801 and initial['actor_gain']==.01 and initial['critic_gain']==1.
        assert initial['biases_zero'] and initial['original_frames']==1_000_000_000
        assert top['initialized_policy_sha256']==initial['initialized_policy_sha256']==sha(folder/'initialized_policy.pt')
        assert sha(ROOT/initial['parent_checkpoint'])==initial['parent_checkpoint_sha256']
        assert sha(ROOT/'scripts/local_goal_return_training_evaluate.py')==top['evaluation_source_sha256']
        assert sha(ROOT/'scripts/local_goal_return_training.py')==top['core_source_sha256']
        initializations.append(initial)
        for path in (folder/'summary.json',folder/'initialization.json',folder/'initialized_policy.pt'):
            sources[str(path.relative_to(ROOT))]=sha(path)
        model=dict(initialization=initial,banks={})
        for bank in BANKS:
            path=folder/bank/'summary.json';report=read(path)
            assert report['complete'] and report['no_optimization'] and report['clean_state_noise_sigma']==0
            assert report['condition']==condition and report['model_seed']==1
            assert report['source_sha256']==top['evaluation_source_sha256']
            assert report['core_sha256']==top['core_source_sha256']
            assert report['behavior_protocol_sha256']==BEHAVIOR_SHA
            assert report['bank_sha256']==design['banks'][bank]['sha256']
            assert report['checkpoint_policy_hash']==initial['full_initialized_policy_hash']
            assert len(report['contexts'])==16 and sorted(r['context'] for r in report['contexts'])==list(range(16))
            assert set(report['variants'])==set(design['variants']) and set(report['controls'])=={'uniform','planner'}
            assert len(report['rows'])==256*len(design['variants'])
            for variant,stats in report['variants'].items():
                actual=summarize_rows([r for r in report['rows'] if r['variant']==variant])
                for key in ('trials','successes','success_fraction','invalid_trials'):
                    assert actual[key]==stats[key],(condition,bank,variant,key)
            native=summarize_rows([r for r in report['rows'] if r['variant']=='blind_native'])
            controls={};control_rows={}
            for name in ('uniform','planner'):
                rows=[r for context in report['contexts'] for r in context['controls'][name]['rows']]
                controls[name]=summarize_rows(rows);control_rows[name]=rows
                for key in ('trials','successes','success_fraction','invalid_trials'):
                    assert controls[name][key]==report['controls'][name][key]
            certificate=report['certificate'];assert certificate['heading_trials']==256 and certificate['equally_weighted_pairs']==128
            assert certificate['certified_pairs']+certificate['uncertified_pairs']==128
            assert certificate['common_information_success_upper_bound']==(.5*certificate['certified_pairs']+certificate['uncertified_pairs'])/128
            near_expected=bank=='near_baseline' and report['all_physical_checks'] and native['successes']>=13
            assert report['gates']['near_baseline_pass']==near_expected
            model['banks'][bank]=dict(native=native,controls=controls,control_rows=control_rows,
                all_variants=report['variants'],gates=report['gates'],certificate=certificate,
                all_physical_checks=report['all_physical_checks'],
                forced_hits=sum(r.get('audit',{}).get('forced_hits',0) for r in report['contexts']),
                invalid_contexts=sum(not r['valid'] for r in report['contexts']))
            sources[str(path.relative_to(ROOT))]=sha(path)
        models[condition]=model
    assert initializations[0]['new_head_hashes']==initializations[1]['new_head_hashes']
    # Collapse duplicated physical baselines only after their actual outcomes agree.
    for bank in BANKS:
        for name in ('uniform','planner'):
            a=models['gc100']['banks'][bank]['control_rows'][name]
            b=models['pgc33']['banks'][bank]['control_rows'][name]
            assert [{k:r.get(k) for k in BEHAVIOR_FIELDS} for r in a]==[{k:r.get(k) for k in BEHAVIOR_FIELDS} for r in b]
    return models,sources


def main():
    models,sources=load_complete();OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for bank in BANKS:
        for name in (*CONDITIONS,'uniform','planner'):
            values=models[name]['banks'][bank]['native'] if name in CONDITIONS else models['gc100']['banks'][bank]['controls'][name]
            rows.append(dict(bank=bank,policy=name,**values))
    summary=dict(complete=True,untrained_on_local_task=True,original_feature_training_frames=1_000_000_000,
        original_model_seed=1,explicit_matched_action_value_head_reset=True,models=models,display_rows=rows,
        both_near_gates_pass=all(models[c]['banks']['near_baseline']['gates']['near_baseline_pass'] for c in CONDITIONS),
        source_sha256=sources,reporting_source_sha256=sha(__file__),
        controls_identical_across_population_inputs=True,
        interpretation='Reward-discovery baseline only. One source-seed pair,16physicalcontexts and256nestedheadings perbank; noPvalues ortrainedPGCbenefit inference.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.5),sharey=True)
    fig.subplots_adjust(left=.07,right=.98,top=.75,bottom=.23,wspace=.18)
    colors=['#357eb0','#d97536','#a4adb5','#6e9e88']
    labels=['Grid','PGC + grid','Uniform\nactions','Heading-aware\nplanner']
    for ax,bank,title in zip(axes,BANKS,['Near starts · 0.36–0.40 m','Hard starts · 0.58–0.62 m']):
        local=[r for r in rows if r['bank']==bank]
        for x,(record,color) in enumerate(zip(local,colors)):
            pct=100*record['success_fraction']
            ax.bar(x,pct,width=.63,color=color,alpha=.80,zorder=2)
            ax.scatter(x+np.linspace(-.18,.18,16),100*np.array(record['context_success_fraction']),
                s=14,color='#25353e',alpha=.44,zorder=3,edgecolors='none')
            ax.text(x,pct+3,f"{record['successes']}/256",ha='center',fontsize=10,color='#25353e')
        if bank=='near_baseline':
            ax.axhline(5,color='#69757e',ls='--',lw=1,zorder=1)
            ax.plot([],[],color='#69757e',ls='--',lw=1,label='5% initial-reward gate')
            ax.legend(loc='upper left',frameon=False,fontsize=9,labelcolor='#69757e')
        else:
            bounds=[models[c]['banks'][bank]['certificate']['common_information_success_upper_bound'] for c in CONDITIONS]
            bound=max(bounds);ax.axhline(100*bound,color='#69757e',ls='--',lw=1,zorder=1)
            ax.text(-.45,100*bound+3,f'Common-information bound: {100*bound:g}%',fontsize=9,color='#69757e')
        ax.set(title=title,ylim=(-3,112),xticks=range(4),xticklabels=labels,yticks=[0,25,50,75,100])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12,zorder=0)
        ax.tick_params(axis='x',length=0,pad=8,labelsize=10)
    axes[0].set_ylabel('Successful trials (%)')
    fig.text(.07,.94,'Before learning the new task',fontsize=17,color='#25353e')
    fig.text(.07,.865,'Original spatial/recurrent features + fresh matched action/value readouts',fontsize=11,color='#65727b')
    fig.text(.07,.055,'Dots: all 16 contexts. Bars: all 256 trials. No learning on the local task.',fontsize=10,color='#65727b')
    for suffix in ('png','pdf'):fig.savefig(OUT/f'reward_discovery.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)

    near={c:models[c]['banks']['near_baseline']['native'] for c in CONDITIONS}
    hard={c:models[c]['banks']['final_hard']['native'] for c in CONDITIONS}
    lead=('Both initialized populations pass the frozen 5% near-start reward-discovery gate.' if summary['both_near_gates_pass']
          else 'At least one initialized population fails the frozen near-start reward-discovery gate; the proposed training precondition is not met.')
    lines=['# Reward discovery before local-task training','',f'**{lead}**','',
        '![All baseline trials and contexts](reward_discovery.png)','',
        'These are the original 1B spatial/visual/recurrent features with explicitly reinitialized action and value readouts. Neither model has learned on this local task. The two populations receive identical new head weights, physical trials and random tapes. The figure asks whether sparse rewards are available at initialization; it is not a trained PGC-versus-Grid performance comparison.','',
        '| Controller | Near starts | Hard starts |','|---|---:|---:|']
    for name in (*CONDITIONS,'uniform','planner'):
        a=next(r for r in rows if r['bank']=='near_baseline' and r['policy']==name)
        b=next(r for r in rows if r['bank']=='final_hard' and r['policy']==name)
        lines.append(f"| {LABELS[name]} | {a['successes']}/256 ({100*a['success_fraction']:.1f}%) | {b['successes']}/256 ({100*b['success_fraction']:.1f}%) |")
    entropies=['unavailable' if near[c]['mean_initial_free_entropy'] is None else f"{near[c]['mean_initial_free_entropy']:.5f}" for c in CONDITIONS]
    lines+=['',f"The initialized Grid and PGC heads have measured mean initial free-action entropies of {entropies[0]} and {entropies[1]} nats on valid near-start trials (uniform six-action entropy: {np.log(6):.5f}). Invalid trials still count as failures in the success denominator. The original CNN/LSTM tensors are preserved exactly; the saved initialized policies and source hashes are independently audited.",'',
        'The uniform controller draws a fresh action from all six options at each update. The analysis-only planner uses the true physical pose to choose the action with best expected immediate progress; it does not see future motor noise. Both controls run the same native physics and stop at the first goal hit. Their physical outcomes were checked identical across the two population inputs before displaying one shared control series.','',
        '| Population | Near gate | Recorded forced hits: near / hard | Invalid contexts: near / hard | Certified hard pairs | Hard common-information bound |',
        '|---|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        a=models[condition]['banks']['near_baseline'];b=models[condition]['banks']['final_hard'];cert=b['certificate']
        lines.append(f"| {LABELS[condition]} | {'Pass' if a['gates']['near_baseline_pass'] else 'Fail'} | {a['forced_hits']} / {b['forced_hits']} | {a['invalid_contexts']} / {b['invalid_contexts']} | {cert['certified_pairs']}/128 | {100*cert['common_information_success_upper_bound']:.1f}% |")
    lines+=['',
        'The near gate was frozen at at least 13 successes out of 256 for **each** population, zero forced rewards and passing physical checks. The hard bound applies to equally weighted opposite-heading pairs with common inputs/state and the saved coupled body-frame tapes. Certified pairs contribute a maximum success rate of 50%; uncertified pairs remain in the denominator with bound 100%. It is conditional on these tapes and observation rules, and does not bound an agent retaining heading information. The easier near bank is not assigned the hard-bank 50% claim.','',
        'All 16 contexts and all 256 trials per controller/bank are retained, including failures. Context dots describe within-bank variation; headings share a physical context and random tape. These are one original model-seed pair, not 256 independent trained agents, and no p-values are reported. Every one of the 12 frozen vision/cue/state variants remains available in the source summaries and this report’s numerical summary; the figure displays the primary blind/native condition.','',
        '[Complete report data and input hashes](summary.json) · [Independent raw-artifact audit](../../local_goal_return_design/completed_baseline_audit.json) · [Frozen bank/gates](../../local_goal_return_design/behavior_protocol.json) · [Task design](../../local_goal_return_design/design/README.md) · [Plot source](../../../scripts/plot_local_goal_return_baseline.py)','',
        'Rebuild with `python scripts/plot_local_goal_return_baseline.py`. It refuses partial data or an incomplete independent audit and performs no policy/environment execution.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,both_near_gates_pass=summary['both_near_gates_pass'],output=str(OUT))))


if __name__=='__main__':main()
