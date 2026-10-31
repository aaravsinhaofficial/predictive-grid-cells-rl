"""Simple illustrations and result plots for the navigation mechanism guide."""
import numpy as np
from plot_lstm_mechanism_summary import GRID, PGC, GRAY, INK, read, save, horizontal_scores
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def geometry():
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.9),layout='constrained')
    for ax in axes:
        ax.add_patch(Circle((0,0),.3,facecolor='#F0F4F5',edgecolor='#A5B3BC',lw=1.4))
        ax.scatter(0,0,s=20,color=INK)
        ax.set(aspect='equal',xlim=(-.72,.4),ylim=(-.42,.48))
        ax.axis('off')
    hit=np.array([-.2919,0.]);direction=np.array([np.cos(.5),np.sin(.5)])
    ahead=hit+.2*direction
    ax=axes[0]
    ax.scatter(*hit,s=65,color=GRID,zorder=3)
    ax.scatter(*ahead,s=65,color=PGC,zorder=3)
    ax.annotate('',xy=ahead,xytext=hit,arrowprops=dict(arrowstyle='->',color=PGC,lw=2.5))
    ax.annotate('Pickup',hit,xytext=(-8,-22),textcoords='offset points',ha='right',color=GRID)
    ax.annotate('20 cm ahead',ahead,xytext=(-4,12),textcoords='offset points',ha='center',color=PGC)
    ax.text(.03,-.05,'Center',fontsize=10)
    ax.text(0,-.36,'Reward anywhere inside',ha='center',fontsize=10)
    ax.set_title('The saved goal cue points inward')
    ax=axes[1]
    start=np.array([-.65,.34])
    ax.scatter(*start,s=60,color=INK,zorder=4)
    ax.annotate('',xy=(.35,.34),xytext=start,arrowprops=dict(arrowstyle='->',color=GRAY,lw=2,linestyle='--'))
    ax.annotate('',xy=(.08,.02),xytext=start,arrowprops=dict(arrowstyle='->',color=PGC,lw=2.5,connectionstyle='arc3,rad=.14'))
    ax.text(-.08,.41,'Keep moving: miss',ha='center',fontsize=10,color=GRAY)
    ax.text(-.47,.02,'Correct course',ha='center',fontsize=10,color=PGC)
    ax.text(0,-.36,'Direction determines capture',ha='center',fontsize=10)
    ax.set_title('The live code helps with turn or move')
    fig.supxlabel('Illustrations, not recorded trajectories · goal radius 30 cm',fontsize=9,color='#687A86')
    save(fig,'07_navigation_cues')


def goal_recovery():
    d=read('lstm_phase_intervention_1B')['estimates']['mean_score']
    fig,ax=plt.subplots(figsize=(7.5,3.5),layout='constrained')
    horizontal_scores(ax,[('Ordinary goal throughout',d['goal_grid'],GRAY),
        ('Predictive goal only far',d['goal_near'],'#E8AD82'),
        ('Predictive goal only near',d['goal_far'],PGC),
        ('Predictive goal throughout',d['intact'],PGC)],PGC,xmax=124)
    ax.set_title('Restoring the predictive goal input restores navigation')
    ax.set_xlabel('90-second score (mean ± SEM)')
    fig.supxlabel('8 frozen PGC models · near = within 60 cm · live code stays predictive',fontsize=9,color='#687A86')
    save(fig,'08_goal_cue_recovery')


def goal_direction():
    d=read('lstm_goal_cue_mechanism_1B')['estimates']['mean_score']
    fig,ax=plt.subplots(figsize=(7.5,3.5),layout='constrained')
    horizontal_scores(ax,[('Original arrival direction',d['pgc33/intact'],PGC),
        ('Inward toward center*',d['pgc33/pgc_radial'],'#739B87'),
        ('Sideways*',d['pgc33/pgc_tangent'],GRAY),
        ('Outward from center*',d['pgc33/pgc_reverse_radial'],GRAY)],PGC,xmax=131)
    ax.set_title('The direction of the saved predictive cue matters')
    ax.set_xlabel('90-second score (mean ± SEM)')
    fig.supxlabel('8 PGC models · same 20 cm shift · ordinary pickup code intact\n*Uses the hidden true center for this diagnostic',fontsize=9,color='#687A86')
    save(fig,'09_goal_direction')


if __name__=='__main__':
    geometry()
    goal_recovery()
    goal_direction()
