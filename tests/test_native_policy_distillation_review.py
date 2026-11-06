"""Independent reset/history checks; fresh native networks, no trained checkpoints."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_policy_distillation import (
    ROOT, N, advance_goal, trainable_student, train_unroll,
)
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena
from rl_visual_memory import NativeGoal, VisualClock


def test_actual_native_unroll_crosses_terminal_with_clean_history_and_state():
    torch.set_num_threads(2)
    torch.manual_seed(993301)
    cfg = json.loads((ROOT / 'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    teacher = build_policy(cfg, 'cpu').eval().requires_grad_(False)
    student = trainable_student(teacher)
    pop = build_population(cfg, 'cpu')
    env = MatchedArena(cfg, N, 1, 993302, 'cpu')
    obs = env.reset_all()
    # A synthetic terminal-time fixture, with actual native transition/reset code.
    env.t.fill_(cfg['episode_len'] - 1)
    goal = NativeGoal(N, 512, 'cpu')
    goal.saved.fill_(0.25)
    clock = VisualClock(N, 'cpu', 993303)
    clock.known.fill_(True)
    clock.blind.fill_(True)
    clock.remaining.fill_(30)
    initial_state = tuple(torch.randn(N, 256) * 0.1 for _ in range(2))
    runtime = dict(
        obs=obs, student_state=initial_state,
        teacher_state=tuple(z.clone() for z in initial_state),
        goal=goal, clock=clock, pa=torch.full((N,), 2, dtype=torch.long),
        pr=torch.ones(N), swap_interval=torch.ones(N, dtype=torch.bool),
        deadline=time.monotonic() + 30,
        black_embedding=teacher.embed(torch.zeros(N, 3, 84, 84)).detach(),
        action_rng=torch.Generator().manual_seed(993304),
        mixture_rng=torch.Generator().manual_seed(993305),
        substitution_rng=torch.Generator().manual_seed(993306),
    )
    optimizer = torch.optim.Adam(student.lstm.parameters(), lr=1e-4)
    _, probe = train_unroll(student, teacher, pop, cfg, env, runtime,
                            optimizer, gap=30, beta=0.5, length=2)
    assert bool(probe['done'][0].all())
    assert not bool(probe['done'][1].any())
    assert bool(probe['blind'][0].all())
    assert not bool(probe['blind'][1].any())
    assert not bool(probe['ordinary_current'][1].any())
    # Goal cache, previous action one-hot, and reward all clear on the next input.
    assert bool((probe['student_inputs'][1, :, 768:] == 0).all())
    # Teacher is frozen, so its second output can be independently checked after
    # the optimizer step. Student and teacher second inputs are both fully native.
    with torch.no_grad():
        expected_state = teacher.lstm(probe['student_inputs'][1],
                                      teacher.zero_state(N, 'cpu'))
        expected_logits = teacher.pi(expected_state[0])
    assert torch.equal(expected_logits, probe['teacher_logits'][1])
    assert torch.equal(expected_logits, probe['student_logits'][1])


def test_actual_hit_code_updates_only_hit_rows_and_terminal_hit_is_cleared():
    class PositionCode:
        def activity(self, position, direction, velocity):
            return position[:, :1].expand(-1, 512)

    goal = NativeGoal(4, 512, 'cpu')
    goal.saved.fill_(7.)
    info = dict(
        hit=torch.tensor([True, False, True, True]),
        hit_pos=torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]]),
        hit_dir=torch.zeros(4, 2), hit_vel=torch.zeros(4, 2),
    )
    advance_goal(PositionCode(), goal, info,
                 torch.tensor([False, False, True, False]), n=2)
    assert torch.equal(goal.saved[:, 0], torch.tensor([1., 7., 0., 4.]))
    assert torch.equal(goal.saved, goal.saved[:, :1].expand(-1, 512))
