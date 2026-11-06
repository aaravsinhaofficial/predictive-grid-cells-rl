"""Physical feasibility check only: no policy is loaded or evaluated."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from code.common import build_env, build_population


def main():
    torch.set_num_threads(2)
    records = []
    for condition in ('gc100', 'pgc33'):
        path = ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.json'
        cfg = json.loads(path.read_text())
        env = build_env(dict(cfg, img_noise=0.), 128, 984201, 'cpu', stagger=False)
        env.reset_all()
        env.noise_v = env.noise_r = 0.
        env.pos.zero_()
        env.goal[:] = torch.tensor([1., 1.])
        env.hd[:] = torch.arange(128) * (2 * math.pi / 128) - math.pi
        env.vs.zero_()
        env.vr.zero_()
        k = 1 - math.exp(-env.dt / env.tau)
        amplitude = env.v_max * k / (2 - k)
        env.vf.fill_(-amplitude)
        start_pos, start_hd = env.pos.clone(), env.hd.clone()
        pop = build_population(cfg, 'cpu')
        held = pop.activity(start_pos, env.mdir, torch.zeros_like(start_pos), shift_scale=0.)
        assert torch.equal(held, held[:1].expand_as(held))
        max_position = max_heading = max_motor = max_excursion = 0.
        for cycle in range(30):
            for action in (2, 3):
                obs, reward, done, info = env.step(torch.full((128,), action))
                assert not bool(reward.any() or done.any() or info['hit_wall'].any())
                max_excursion = max(max_excursion, float((env.pos-start_pos).norm(dim=1).max()))
            max_position = max(max_position, float((env.pos-start_pos).norm(dim=1).max()))
            max_heading = max(max_heading, float((env.hd-start_hd).abs().max()))
            max_motor = max(max_motor, float((env.vf+amplitude).abs().max()),
                            float(env.vs.abs().max()), float(env.vr.abs().max()))
        assert max_position < 1e-6 and max_heading < 1e-6 and max_motor < 1e-6
        records.append(dict(condition=condition, config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            headings=128, two_step_cycles=30, dt_seconds=env.dt, k=k,
            prescribed_initial_forward_velocity=-amplitude,
            maximum_stroboscopic_position_error_m=max_position,
            maximum_heading_error_rad=max_heading, maximum_motor_error=max_motor,
            maximum_within_cycle_excursion_m=max_excursion,
            held_ordinary_current_code_identical_across_headings=True))
    result = dict(scope='Physical feasibility only; no network, learned state or ring outcome evaluated.',
        intervention='Disable image/motor noise; set vf to the native two-action periodic solution, vs=vr=0; repeat actual forward/back actions. Proposed recurrent diagnostic would hold ordinary current input and black RGB explicitly.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), models=records)
    out = ROOT / 'results/native_turn_geometry_design/forward_backward_physics_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
