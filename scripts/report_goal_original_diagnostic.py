"""Presentation of all completed original-only diagnostic cases; no rollout."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/goal_cue_memory/original_diagnostic'
OUT=BASE/'report'
CONDITIONS=('gc100','pgc33')
LABELS={'gc100':'Grid','pgc33':'PGC + grid'}
COLORS={'gc100':'#2878b5','pgc33':'#e87532'}
VARIANTS=('brief_native','continuous','brief_ordinary','no_cue','brief_reset',
    'donor_plus90','donor_minus90','continuous_reset','wrong_plus90','wrong_minus90','self_swap')
NAMES={'continuous':'Always goal','brief_native':'Brief native goal','brief_ordinary':'Brief ordinary goal',
    'no_cue':'No goal','brief_reset':'Brief goal + postcue erase','donor_plus90':'State donor +90°',
    'donor_minus90':'State donor −90°','continuous_reset':'Always goal + postcue erase',
    'wrong_plus90':'Goal cue from +90°','wrong_minus90':'Goal cue from −90°','self_swap':'Self-state swap'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def donor_contexts(rows):
    """The frozen early-horizon statistic, retaining all16 cases/context."""
    assert len(rows)==256 and {(r['context'],r['angle_index'])for r in rows}=={(c,a)for c in range(16)for a in range(16)}
    output=[]
    for context in range(16):
        chunk=[r for r in rows if r['context']==context]
        assert len(chunk)==16
        valid=[r for r in chunk if not r.get('invalid',False)]
        for row in valid:
            d=row['donor_behavior'];assert d['early_horizon_free_updates']==15
            assert d['early_terminal_position_held']==(d['early_observed_free_updates']<15)
        values={name:sum(r.get('donor_behavior',{}).get(f'early_15_free_progress_to_{name}_m',0)for r in chunk)/16*100
                for name in ('donor','recipient')}
        output.append(dict(context=context,trials=16,valid_trials=len(valid),
            recipient_progress_cm=values['recipient'],donor_progress_cm=values['donor'],
            donor_minus_recipient_cm=values['donor']-values['recipient'],
            early_terminal_held=sum(r.get('donor_behavior',{}).get('early_terminal_position_held',False)for r in chunk)))
    return output


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports={};evidence={};counts=[];directional=[]
    for condition in CONDITIONS:
        folder=BASE/condition;ap=BASE/'audits'/f'{condition}.json';audit=json.loads(ap.read_text())
        assert audit['passed'] and audit['complete'] and not audit['failures'] and not audit['pending']
        for relative,proof in audit['artifacts'].items():
            path=folder/relative;assert path.stat().st_size==proof['bytes'] and sha(path)==proof['sha256']
        report=json.loads((folder/'evaluation/summary.json').read_text());reports[condition]=report
        assert report['complete'] and report['variant_order']==list(VARIANTS)
        evidence[condition]=dict(audit_path=str(ap.relative_to(ROOT)),audit_sha256=sha(ap),
            evaluation_summary_sha256=sha(folder/'evaluation/summary.json'),policy_hash=report['checkpoint_policy_hash'],
            protocol_sha256=audit['protocol_sha256'])
        for variant in VARIANTS:
            v=report['variants'][variant];t=v['totals'];assert t['trials']==256 and len(v['rows'])==256
            assert t['successes']==sum(r['hit']for r in v['rows'])
            counts.append(dict(condition=condition,variant=variant,successes=t['successes'],
                trials=256,valid_trials=t['valid_trials'],invalid_trials=t['invalid_trials']))
        for variant in ('donor_plus90','donor_minus90'):
            directional.extend(dict(condition=condition,variant=variant,**r)
                for r in donor_contexts(report['variants'][variant]['rows']))
    assert len({e['protocol_sha256']for e in evidence.values()})==1
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('all_conditions.csv',counts),('donor_contexts.csv',directional)]:
        with(OUT/name).open('w')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    main_names=('continuous','brief_native','brief_reset','no_cue')
    fig,ax=plt.subplots(figsize=(8.2,3.7))
    for index,condition in enumerate(CONDITIONS):
        hits=[reports[condition]['variants'][name]['totals']['successes']for name in main_names]
        xs=[x+(index-.5)*.34 for x in range(4)]
        ax.bar(xs,[n/256*100 for n in hits],width=.32,color=COLORS[condition],label=LABELS[condition])
        for x,n in zip(xs,hits):
            y=n/256*100
            ax.text(x,y-9 if y>=15 else y+2,str(n),ha='center',color='white'if y>=15 else COLORS[condition],fontsize=9)
    ax.set(xticks=range(4),xticklabels=['Always goal','Brief goal','Brief + erase','No goal'],
        ylabel='Goals reached (%)',ylim=(0,100))
    ax.legend(frameon=False,ncols=2,loc='upper center',bbox_to_anchor=(.5,1.15))
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'goal_memory.png',dpi=180);fig.savefig(OUT/'goal_memory.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.3),sharex=True,sharey=True)
    limits=[r[k]for r in directional for k in ('donor_progress_cm','recipient_progress_cm')]
    lo,hi=min(0,min(limits))-2,max(0,max(limits))+2
    for i,condition in enumerate(CONDITIONS):
        for j,variant in enumerate(('donor_plus90','donor_minus90')):
            ax=axes[i,j];rows=[r for r in directional if r['condition']==condition and r['variant']==variant]
            for r in rows:
                offset=(r['context']-7.5)*.012
                ax.plot([offset,1+offset],[r['recipient_progress_cm'],r['donor_progress_cm']],
                    color=COLORS[condition],alpha=.3,lw=1,marker='o',ms=3)
            means=[sum(r[k]for r in rows)/16 for k in ('recipient_progress_cm','donor_progress_cm')]
            ax.plot([0,1],means,color=COLORS[condition],lw=2.5,marker='D',ms=6)
            ax.axhline(0,color='0.6',lw=.8,ls='--')
            ax.set(title=LABELS[condition]+' · '+('donor +90°'if j==0 else 'donor −90°'),
                xticks=[0,1],xticklabels=['Recipient goal','Donor goal'],xlim=(-.25,1.25),ylim=(lo,hi))
            if j==0:ax.set_ylabel('Progress in first second (cm)')
            ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'donor_progress.png',dpi=180);fig.savefig(OUT/'donor_progress.pdf');plt.close(fig)
    summaries={c:{name:reports[c]['variants'][name]['totals']for name in VARIANTS}for c in CONDITIONS}
    result=dict(version='original_goal_memory_diagnostic_report_v1',complete=True,producer_sha256=sha(__file__),
        evidence=evidence,all_conditions=counts,donor_contexts=directional,totals=summaries,
        descriptive_screens={c:reports[c]['gates']for c in CONDITIONS},
        one_original_seed_pair=True,no_training=True,no_cloud=True,
        no_automatic_neural_nomination=True,parent_paired_training_gate_still_failed=True,
        donor_summary='Each point is a 16-trial context mean; all contexts retained; the goal cue stays absent during navigation; first-second terminal position held as frozen. Invalid cases contribute zero and remain in denominators.',
        images={p.name:sha(p)for p in OUT.glob('*.png')})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join('| '+NAMES[name]+' | '+' | '.join(str(summaries[c][name]['successes'])+'/256'for c in CONDITIONS)+' |'for name in VARIANTS)
    simple=lambda c,name:summaries[c][name]['successes']
    invalid=sum(row['invalid_trials']for row in counts)
    validity=f'All {256*11*2:,} trial-condition cases were valid.'if not invalid else f'{invalid} invalid trial-condition cases remain as failures in the prescribed denominators.'
    early_advantages={c:{name:100*(summaries[c][name]['donor_behavior']['mean_early_15_free_progress_to_donor_m']-
        summaries[c][name]['donor_behavior']['mean_early_15_free_progress_to_recipient_m'])
        for name in ('donor_plus90','donor_minus90')}for c in CONDITIONS}
    # This report describes these fixed outcomes, rather than selecting an
    # explanation on a different endpoint. Guard the numeric interpretation.
    assert max(abs(v)for row in early_advantages.values()for v in row.values())<.005
    assert all(abs(simple(c,'brief_native')-simple(c,'no_cue'))<=7 and
               abs(simple(c,'brief_native')-simple(c,'brief_reset'))<=5 for c in CONDITIONS)
    assert all(not reports[c]['gates']['useful_memory']['passed']for c in CONDITIONS)
    text=f'''# Goal memory in the unchanged original networks

**For this fixed pair, successful navigation depended strongly on a continuing goal message.** After a brief cue and delay, performance was similar to the no-cue and erased-state controls: the differences were at most 7 and 5 goals out of 256, respectively. Neither sign of the 90° donor-state swap produced an average first-second steering advantage toward the donor goal (0.00 cm in both networks). Keeping the goal message available after erasure gave {simple('gc100','continuous_reset')}/256 Grid and {simple('pgc33','continuous_reset')}/256 PGC+grid successes, consistent with rebuilding useful state from the continuing cue. This pattern points to ongoing goal guidance as the useful mechanism in this task.

![Goal-input and state-erasure controls](goal_memory.png)

With the goal code continuously available, Grid reached {simple('gc100','continuous')}/256 goals and PGC+grid reached {simple('pgc33','continuous')}/256. After only a brief cue, Grid reached {simple('gc100','brief_native')}/256 and PGC+grid reached {simple('pgc33','brief_native')}/256. Erasing the recurrent state after that cue gave {simple('gc100','brief_reset')}/256 and {simple('pgc33','brief_reset')}/256, respectively. Numbers on bars are goals reached out of 256. [All counts linked to raw data](all_conditions.csv) include every prescribed trial.

We showed the native goal code for 15 stationary RNN updates, removed it for 45 more updates, then allowed 90 native actions (six seconds). The goal remained unmarked in the image. The networks retained their actor, CNN and LSTM trained for the original 1 billion frames; both used the same archived fresh critic. **No learning occurred.**

**Current-location code and clean native vision remain available throughout.** The ordinary-goal condition changes only the initial goal cue. Brief conditions remove all 512 goal-input slots after the cue. Postcue erasure clears both recurrent states immediately after the cue. In the always-goal erasure control, continued goal input can rebuild the state.

![Donor-state directional behavior](donor_progress.png)

Immediately before the first navigation input, we replaced the recipient's complete h/c state with a genuine state from the same context and a goal 90° away. We kept the recipient's physical goal and movement-noise tape; native vision and current-location code continued updating from its own path. Thin lines show all 16 context means; diamonds show their overall mean. Progress uses exactly the first 15 navigation updates (one second); an earlier terminal position is held. Positive progress means movement toward that goal. [Both signs and all context values](donor_contexts.csv) are retained; full-path donor entry is censored by the first actual recipient hit and is reported separately in [summary.json](summary.json).

| Condition | Grid | PGC+grid |
|---|---:|---:|
{table}

This was separately registered **after** the original qualification result (181 Grid / 236 PGC+grid) was known and **before** any memory-bank outcome. The qualification and diagnostic used **two different presealed trial banks**; the Grid continuous-goal score changing from 181 to {simple('gc100','continuous')} therefore reflects different trials, not learning. All 11 conditions and all 256 cases per condition were fixed. This is one trained seed pair, with no population-level significance claim. Both descriptive memory screens failed. The earlier paired training requirement remains failed; no further training or neural assay follows these results under the prescribed decision.

Both independent audits passed. {validity} They verified original-policy composition, unchanged weights, global random state and model mode across evaluation, saved physical/input traces, the full stationary neural prefix and first navigation transition, genuine donor provenance and the self-swap control. [Grid audit](../audits/gc100.json) · [PGC audit](../audits/pgc33.json) · [Exploratory protocol](../../../../docs/research/goal_cue_memory_original_diagnostic_protocol.json).

All outputs and exact source/audit hashes are in [the report record](summary.json). Cloud compute cost was $0.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps(dict(out=str(OUT),complete=True,all_conditions=counts)))


if __name__=='__main__':main()
