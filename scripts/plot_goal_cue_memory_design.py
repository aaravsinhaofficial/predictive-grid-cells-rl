"""Task-design illustration from one physical fixture; no policy or study bank."""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import goal_cue_memory as task


def main():
    torch.set_num_threads(2)
    cfg_path = ROOT / 'models/pgc33/pgc33_lstm_s1_1000M.json'
    cfg = json.loads(cfg_path.read_text())
    plan = task.TrialStreams(699901).draw(1, cfg, delay_steps=45)
    plan['goal_bearing'].fill_(math.pi / 4)
    plan['goal'] = .75 * torch.stack((plan['goal_bearing'].cos(), plan['goal_bearing'].sin()), 1)
    plan['pickup_offset'].fill_(-math.pi / 2)
    plan['pickup_angle'] = plan['goal_bearing'] + plan['pickup_offset']
    batch = task.prepare_batch(cfg, task.make_source(cfg, 'cpu'), plan)
    goal = plan['goal'][0].numpy()
    pickup = batch['setup']['hit_position'][0].numpy()
    ahead = pickup + .2 * batch['setup']['hit_direction'][0].numpy()
    out = ROOT / 'results/goal_cue_memory_design'
    out.mkdir(parents=True, exist_ok=True)
    fig, (arena, timeline) = plt.subplots(1, 2, figsize=(10.2, 4.1),
                                         gridspec_kw={'width_ratios': [1, 1.6]})
    arena.add_patch(Rectangle((-1.25, -1.25), 2.5, 2.5, facecolor='#f5f7f9', edgecolor='#9aa4ad'))
    arena.add_patch(Circle(goal, .3, color='#e27932', alpha=.17))
    arena.plot(*goal, '+', color='#444444', ms=10, mew=1.5)
    arena.plot(*pickup, 'o', color='#2878b5', ms=8)
    arena.plot(*ahead, 'o', color='#e27932', ms=8)
    arena.annotate('', xy=ahead, xytext=pickup,
                   arrowprops={'arrowstyle': '->', 'color': '#e27932', 'lw': 1.7})
    arena.annotate('Grid', pickup, (-.75, .95), color='#2878b5', fontsize=12,
                   arrowprops={'arrowstyle': '-', 'color': '#2878b5'})
    arena.annotate('Predictive', ahead, (.64, .91), color='#bd5e24', fontsize=12,
                   arrowprops={'arrowstyle': '-', 'color': '#bd5e24'})
    arena.text(.86, .41, 'Goal', fontsize=12, ha='center')
    arena.plot(0, 0, 'o', color='#252a30', ms=6)
    arena.text(0, -.18, 'Common start', ha='center', fontsize=11)
    arena.set(xlim=(-1.34, 1.34), ylim=(-1.34, 1.34), aspect='equal')
    arena.axis('off')
    arena.set_title('Same arena and spatial maps', fontsize=14)

    timeline.set(xlim=(-.05, 3.05), ylim=(-.1, 2.8))
    timeline.axis('off')
    for i, label in enumerate(('Goal cue\n1 second', 'Wait\n0–3 seconds', 'Navigate\nup to 6 seconds')):
        timeline.add_patch(Rectangle((i + .04, 1.9), .92, .62,
            facecolor='#edf1f5', edgecolor='none'))
        timeline.text(i + .5, 2.21, label, ha='center', va='center', fontsize=12)
    timeline.text(0, 1.53, 'Saved goal code', fontsize=12)
    timeline.plot([.05, .95], [1.30, 1.30], color='#e27932', lw=8, solid_capstyle='butt')
    timeline.plot([1.05, 2.95], [1.30, 1.30], color='#dce1e6', lw=8, solid_capstyle='butt')
    timeline.text(2, 1.05, 'Removed', ha='center', color='#5d646c', fontsize=11)
    timeline.text(0, .69, 'Vision + current spatial code', fontsize=12)
    timeline.plot([.05, 2.95], [.46, .46], color='#2878b5', lw=8, solid_capstyle='butt')
    timeline.text(1, .04, 'Held still', ha='center', fontsize=11)
    timeline.text(2.5, .04, 'Agent chooses actions', ha='center', fontsize=11)
    fig.suptitle('Study design: remember the goal before moving', fontsize=16, y=.98)
    fig.text(.5, .025, 'Physical example only · No trained-agent result', ha='center', fontsize=10, color='#61666d')
    fig.tight_layout(rect=(0, .06, 1, .92))
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'task_design.{suffix}', dpi=180)
    plt.close(fig)
    record = dict(kind='physical_fixture_design_illustration', no_policy_or_study_bank=True,
        fixture_seed=699901, goal=goal.tolist(), actual_pickup=pickup.tolist(), predictive_point=ahead.tolist(),
        actual_setup_reward=float(batch['setup']['native_reward'][0]), learning_reward=0.,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest())
    (out / 'task_design_example.json').write_text(json.dumps(record, indent=2) + '\n')
    print(out / 'task_design.png')


if __name__ == '__main__':
    main()
