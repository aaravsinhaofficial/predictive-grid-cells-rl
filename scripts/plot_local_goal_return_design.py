"""Simple design-only figure from the sealed local-return behavioral protocol."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/local_goal_return_design'
PROTOCOL=OUT/'behavior_protocol.json'
PROTOCOL_SHA='4c7ba90dd79c8c0b5f8ac13ba89e4a7d6ed5bf0287556982ade07f911fc16e48'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(PROTOCOL)==PROTOCOL_SHA,'Use the unchanged prospective protocol'
    p=json.loads(PROTOCOL.read_text());stages=p['stages']
    assert len(stages)==4 and p['native_physics']['goal_radius']==.3
    assert [s['delay'] for s in stages]==[4,15,30,60]
    # Reviewed maximal native event. The cue occupies the first delay INPUT;
    # there is no extra cue action/update between the probe and delay.
    durations=[15,20,4,stages[-1]['delay'],30]
    edges=[0]
    for duration in durations:edges.append(edges[-1]+duration)
    assert edges==[0,15,35,39,99,129] and edges[-1]<160

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.titlesize':14,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(12.6,4.5),facecolor='white')
    gs=fig.add_gridspec(1,2,width_ratios=[1,1.65],left=.065,right=.975,
        bottom=.20,top=.77,wspace=.40)
    ax=fig.add_subplot(gs[0,0]);timeline=fig.add_subplot(gs[0,1])

    ax.set_title('A   Start nearer; remember longer',loc='left',pad=21)
    goal_radius=p['native_physics']['goal_radius']
    ax.axvspan(.28,goal_radius,color='#edf0f2',zorder=0)
    ax.axvline(goal_radius,color='#858d94',ls='--',lw=1.2,zorder=1)
    ax.text(goal_radius+.005,3.72,'Goal radius',fontsize=9,color='#66717a',va='bottom')
    for i,s in enumerate(stages):
        y=3-i;mid=(s['low']+s['high'])/2
        ax.plot([s['low'],s['high']],[y,y],color='#327c9d',lw=8,solid_capstyle='round')
        ax.scatter([mid],[y],s=25,color='white',zorder=3)
        ax.text(mid,y+.27,f"{s['low']:.2f}–{s['high']:.2f}",ha='center',fontsize=9,color='#2a596d')
        low=(s['delay']+1)//2
        ax.text(1.05,y,f"{low}–{s['delay']}",transform=ax.get_yaxis_transform(),
            ha='center',va='center',fontsize=11,color='#665b89',clip_on=False)
    ax.text(1.05,3.76,'Delay\n(updates)',transform=ax.get_yaxis_transform(),
        ha='center',va='bottom',fontsize=10,color='#665b89',clip_on=False)
    ax.set(xlim=(.28,.65),ylim=(-.45,4.0),xlabel='Distance to goal center (m)',
        yticks=[3,2,1,0],yticklabels=['Stage 1','Stage 2','Stage 3','Stage 4'],
        xticks=[.3,.4,.5,.6])
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#b6bdc1');ax.tick_params(axis='y',length=0,pad=8)
    ax.tick_params(axis='x',colors='#59646c');ax.grid(axis='x',color='#edf0f2',zorder=0)

    timeline.set_title('B   One trial, then stop',loc='left',pad=21)
    colors=['#7badd2','#cbd1d6','#e9b15d','#b6abd0','#7bb49d']
    y=.41;height=.26
    for start,stop,color in zip(edges[:-1],edges[1:],colors):
        timeline.add_patch(Rectangle((start,y),stop-start,height,facecolor=color,
            edgecolor='white',lw=1.5,zorder=2))
    for x,label in [(7.5,'15\nvisible'),(25,'20\nblind turns'),(69,'60\nleft turns'),(114,'30\nfree actions')]:
        timeline.text(x,y+height/2,label,ha='center',va='center',fontsize=10,color='#25353e',zorder=3)
    timeline.annotate('4 forward',xy=(37,y+height),xytext=(24,.95),
        ha='center',va='bottom',fontsize=10,color='#976a26',
        arrowprops=dict(arrowstyle='-',color='#b28a4d',lw=1,connectionstyle='angle,angleA=-90,angleB=0,rad=3'))
    # First input at update39; draw one small mark within the existing delay.
    timeline.plot([39,39],[y-.035,y+height+.055],color='#a14d36',lw=2.4,zorder=4)
    timeline.annotate('1 predictive cue',xy=(39,y+height+.055),xytext=(61,.95),
        ha='center',va='bottom',fontsize=10,color='#a14d36',
        arrowprops=dict(arrowstyle='-|>',color='#a14d36',lw=1,mutation_scale=9))
    for a,b,label,color in [(0,99,'Forced preparation · no reward','#69747b'),
                            (99,129,'Goal hit: +10 → end','#33785a')]:
        timeline.plot([a,b],[.24,.24],color=color,lw=1)
        timeline.plot([a,a],[.24,.28],color=color,lw=1)
        timeline.plot([b,b],[.24,.28],color=color,lw=1)
        timeline.text((a+b)/2,.17,label,ha='center',va='top',fontsize=9,color=color)
    timeline.set(xlim=(-2,131),ylim=(0,1.18),xticks=[0,15,39,99,129],yticks=[],
        xlabel='Action steps (15 per second)')
    timeline.spines[['top','right','left']].set_visible(False)
    timeline.spines['bottom'].set_color('#b6bdc1');timeline.tick_params(axis='x',colors='#59646c')
    fig.text(.065,.94,'Local goal-return curriculum',fontsize=18,weight='medium',color='#25353e')
    fig.text(.065,.875,'Same square arena, spatial maps, LSTM and six actions',fontsize=11,color='#64717a')
    fig.text(.975,.94,'DESIGN ONLY',ha='right',fontsize=9,color='#7a858c')
    fig.text(.065,.045,'Timeline shows the longest trial. The cue is the first delay input, not an extra step.',
        fontsize=10,color='#59646c')
    for suffix in ('png','pdf'):fig.savefig(OUT/f'trial_design.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)
    metadata=dict(protocol_sha256=PROTOCOL_SHA,source_sha256=sha(__file__),design_only=True,
        no_training_or_behavior_results=True,stages=stages,goal_radius_m=goal_radius,
        maximum_phase_updates=durations,phase_boundaries=edges,cue_input_update=39,
        free_start_update=99,maximum_total_updates=129,policy_updates_per_second=15,
        phase_names=['visible_forced_turns','blind_labelled_turns','forward_probe','left_delay','free_actor'],
        cue_counted_inside_delay=True,reward='Only an actual FREE goal hit receives+10; terminate then or after30FREEactions',
        preserved_archive_files='No archived bankREADME, protocol or PTbank was modified')
    (OUT/'trial_design_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    caption=OUT/'design/README.md';caption.parent.mkdir(exist_ok=True)
    caption.write_text('''# The proposed local goal-return task

![Four training stages and the native trial timeline](../trial_design.png)

**Design only; this figure contains no learning or neural results.** The agent starts progressively farther from the goal while the blind delay grows. Distances refer to the goal center; the unchanged goal radius is 0.30 m. The square arena, maps, LSTM and six actions remain the repository's native ones. The action/value heads are explicitly reinitialized before this new study.

The longest trial has 15 visible forced updates, at most 20 labelled blind-turn updates, four forward-probe updates, at most 60 LEFT-delay updates and 30 free actor updates: **129 total**, inside the 160-update credit window. The single native predictive cue is the **first delay input**, immediately after the actual forward probe; it adds no update. Blind turns vary from 10–20 updates total, and each stage's delay is uniform over the range shown. In Grid models the corresponding pulse is ordinary grid code.

Preparation is forced and supplies no task reward. During the final free interval, only an actual native goal hit earns +10; that hit or the 30-action timeout ends the trial. The separate setup pickup creates the saved goal code outside the scored trial. The actor receives no supervised heading or ring target.

[PDF](../trial_design.pdf) · [Frozen behavioral protocol](../behavior_protocol.json) · [Figure timing/provenance](../trial_design_metadata.json) · [Scientific design review](../../../docs/research/local_goal_return_study_review.md) · [Plot source](../../../scripts/plot_local_goal_return_design.py)

Rebuild this display with `python scripts/plot_local_goal_return_design.py`. The script verifies the frozen protocol hash and leaves the archived bank README and all bank bytes unchanged.
''')
    print(json.dumps(metadata))


if __name__=='__main__':main()
