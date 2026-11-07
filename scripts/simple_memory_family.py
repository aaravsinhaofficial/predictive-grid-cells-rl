"""Conditional families from an actual saved visible prefix, without warm-up.

All rotations are sensory-history interventions. They preserve the saved goal,
incoming state and action/reward tape; they do not assert policy-generated
trajectories at the counterfactual angles or establish a ring by themselves.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import torch
from code.common import CodeSource, build_env, build_population
from collect_simple_memory_states import physical_eligibility, verify_visible_prefix

ATOL, RTOL, PHYSICAL_TOL = 1e-5, 1e-4, 1e-5


def rotate(vectors, phi):
    """Float64 angle arithmetic; float32 native physical vectors are retained."""
    angles = torch.remainder(phi, 2*math.pi)
    c, s = angles.cos().to(vectors.dtype), angles.sin().to(vectors.dtype)
    x, y = vectors[..., 0], vectors[..., 1]
    return torch.stack((c*x-s*y, s*x+c*y), -1)


def rotate_positions(points, endpoint, phi):
    delta = points-endpoint
    # Algebraically p*+R(p-p*), with literal identity-angle positions preserved.
    return points+(rotate(delta, phi)-delta)


def comparison(actual, expected):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return dict(qualified=finite and torch.allclose(actual, expected, rtol=RTOL, atol=ATOL),
        bitwise_equal=torch.equal(actual, expected),
        max_absolute_error=float((actual-expected).abs().max()) if finite else None,
        l2_error=float((actual-expected).norm()) if finite else None, atol=ATOL, rtol=RTOL)


@torch.inference_mode()
def build_family(policy, cfg, context, device='cpu', phases=768):
    """Return float32 full states/input tensors on device; theta is float64 phi.

    theta[j]=2*pi*j/phases is RELATIVE rotation, with phi0 the actual history.
    Model/render/recurrent batches retain the original batch size, padding only
    the final computation chunk. The extra2pi history is checked and excluded
    from returned family rows. No hidden-state or decoder-based selection.
    """
    if phases < 4 or cfg['policy'] != 'lstm' or policy.nh != 256 or policy.n_code != 512:
        raise ValueError('Use a full native256-LSTM and at least four phase histories')
    if cfg.get('extra_input','none') != 'none' or cfg['pgc_shift_mode'] != 'space' or cfg['code_noise'] != 0:
        raise ValueError('Native noiseless spatial lookup and no extra inputs required')
    slot, batch = context['selected_slot'], context['original_batch_size']
    rt, prefix = context['runtime'], context['visible_prefix']
    endpoint = rt['obs']['pos'][slot].cpu()
    goal_pos = rt['environment_tensors']['goal'][slot].cpu()
    goal = rt['goal'][slot].cpu()
    physical = physical_eligibility(prefix, endpoint, goal_pos, goal,
        half=rt['environment_scalars']['half'], wall_margin=rt['environment_scalars']['wall_margin'],
        goal_radius=cfg['goal_radius'], visible_steps=cfg.get('task_visible_steps',15))
    if not physical['eligible']:
        raise ValueError({'physical_context_unqualified': physical})
    if not bool(rt['clock']['known'][slot] and rt['clock']['blind'][slot]) or context['phase_age'] != 1:
        raise ValueError('Family must end immediately before the first blind input')
    if float(rt['pr'][slot]) != 0:
        raise ValueError('The fixed dark input requires actual onset reward0')
    before_mode = policy.training
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None
    policy.eval()
    try:
        replay = verify_visible_prefix(policy, context, device, strict=False)
        pop = build_population(cfg, device)
        source = CodeSource(cfg, pop, device)
        renderer = build_env(dict(cfg, img_noise=0.), batch, 741024, device, stagger=False)
        for key in ('land_pos','stripe_phase'):
            value = rt['environment_tensors'][key][slot].to(device)
            setattr(renderer, key, value[None].repeat(batch, 1))
        def clean_render(pos, hd):
            renderer.pos = pos.clone()
            renderer.hd = hd.clone()
            return renderer.render()
        native = {key: value.to(device) for key,value in prefix.items()}
        end = endpoint.to(device)
        clean_zero = []
        for t in range(len(prefix['rgb'])):
            clean_zero.append(clean_render(native['pos'][t].repeat(batch,1),
                                           native['body_heading'][t].repeat(batch))[0].clone())
        theta = torch.arange(phases, dtype=torch.float64, device=device)*(2*math.pi/phases)
        including_closure = torch.cat((theta, torch.tensor([2*math.pi],dtype=torch.float64,device=device)))
        output = []
        max_displacement_error, max_radius_error, max_heading_increment_error = 0., 0., 0.
        minimum_wall, minimum_goal = float('inf'), float('inf')
        phi0_rgb_error, phi0_code_error = 0., 0.
        all_inputs_finite = True
        initial = tuple(native[key][0].repeat(batch,1) for key in ('h_before','c_before'))
        fixed_goal = goal.to(device).repeat(batch,1)
        for start in range(0, len(including_closure), batch):
            phi = including_closure[start:start+batch]
            count = len(phi)
            if count < batch: phi = torch.cat((phi, phi[-1:].repeat(batch-count)))
            reduced = torch.remainder(phi, 2*math.pi)
            state = tuple(value.clone() for value in initial)
            previous_rotated, previous_heading = None, None
            for t in range(len(prefix['rgb'])):
                pos = rotate_positions(native['pos'][t].repeat(batch,1), end, phi)
                hd = native['body_heading'][t] + reduced.to(native['body_heading'].dtype)
                direction = rotate(native['travel_dir'][t].repeat(batch,1), phi)
                velocity = rotate(native['velocity'][t].repeat(batch,1), phi)
                clean = clean_render(pos, hd)
                # Equivalent to C_phi+(observed-C0); this association preserves
                # literal observed bytes at phi0 when the clean renders agree.
                rgb = (native['rgb'][t][None] + (clean-clean_zero[t][None])).clamp(0,1)
                current = source(pos, direction, velocity)
                if start == 0:
                    phi0_rgb_error = max(phi0_rgb_error,float((rgb[0]-native['rgb'][t]).abs().max()))
                    phi0_code_error = max(phi0_code_error,float((current[0]-native['current'][t]).abs().max()))
                all_inputs_finite &= bool(torch.isfinite(rgb).all() and torch.isfinite(current).all())
                _, _, state = policy.step(rgb, current, fixed_goal, native['pa'][t].repeat(batch),
                                          native['pr'][t].repeat(batch), state)
                radius = (pos-end).norm(dim=1)
                max_radius_error = max(max_radius_error,float((radius-(native['pos'][t]-end).norm()).abs().max()))
                minimum_wall = min(minimum_wall,float((renderer.half-renderer.wall_margin-pos.abs()).min()))
                minimum_goal = min(minimum_goal,float((pos-goal_pos.to(device)).norm(dim=1).min())-cfg['goal_radius'])
                if previous_rotated is not None:
                    expected = rotate((native['pos'][t]-native['pos'][t-1]).repeat(batch,1),phi)
                    max_displacement_error = max(max_displacement_error,float(((pos-previous_rotated)-expected).abs().max()))
                    delta_error = (hd-previous_heading)-(native['body_heading'][t]-native['body_heading'][t-1])
                    wrapped_error = torch.atan2(delta_error.sin(),delta_error.cos())
                    max_heading_increment_error = max(max_heading_increment_error,float(wrapped_error.abs().max()))
                previous_rotated, previous_heading = pos, hd
            expected_last = rotate((end-native['pos'][-1]).repeat(batch,1),phi)
            max_displacement_error = max(max_displacement_error,float(((end-previous_rotated)-expected_last).abs().max()))
            end_heading = rt['obs']['hd'][slot].to(device)+reduced.to(native['body_heading'].dtype)
            last_error = (end_heading-previous_heading)-(rt['obs']['hd'][slot].to(device)-native['body_heading'][-1])
            max_heading_increment_error=max(max_heading_increment_error,float(torch.atan2(last_error.sin(),last_error.cos()).abs().max()))
            output.append(torch.cat(state,1)[:count])
        full = torch.cat(output)
        states, closure = full[:-1], full[-1]
        expected_onset = torch.cat(rt['state'],1)[slot].to(device)
        phi0 = comparison(states[0], expected_onset)
        closure_check = comparison(closure, states[0])
        radius = float((states-states.mean(0)).square().sum(1).mean().sqrt()) if bool(torch.isfinite(states).all()) else None
        phi0['l2_over_family_radius'] = phi0['l2_error']/radius if radius and radius>0 and phi0['l2_error'] is not None else None
        closure_check['l2_over_family_radius'] = closure_check['l2_error']/radius if radius and radius>0 and closure_check['l2_error'] is not None else None
        endpoints = end.repeat(phases,1)
        travel = rotate(rt['obs']['dir'][slot].to(device).repeat(phases,1),theta)
        velocity = rotate(rt['obs']['vel'][slot].to(device).repeat(phases,1),theta)
        ordinary = pop.activity(endpoints,travel,velocity,shift_scale=0.)
        correct = source(endpoints,travel,velocity)
        codes = dict(ordinary=ordinary, correct=correct)
        for name, offset in (('wrong_plus', math.pi/3), ('wrong_minus', -math.pi/3)):
            wrong = source(endpoints,rotate(travel,torch.full_like(theta,offset)),velocity)
            codes[name] = torch.where(pop.is_pgc[None], wrong, ordinary)
        for code in codes.values():
            assert torch.equal(code[:,~pop.is_pgc],ordinary[:,~pop.is_pgc])
        if not pop.n_pgc: assert all(torch.equal(code,ordinary) for code in codes.values())
        black = torch.zeros(batch,3,84,84,device=device)
        embedding = policy.embed(black)[0].repeat(phases,1)
        goal_rows = goal.to(device).repeat(phases,1)
        pa = rt['pa'][slot].to(device).repeat(phases)
        pr = torch.zeros(phases,device=device)
        encode = lambda code: policy._concat(embedding,code,goal_rows,pa,pr,None)
        dark = encode(torch.zeros_like(ordinary))
        pulses = {name: encode(code) for name,code in codes.items()}
        assert torch.equal(dark,dark[:1].expand_as(dark))
        scale = torch.cat([states[:,k*policy.nh:(k+1)*policy.nh].square().mean().sqrt().clamp_min(1e-8).repeat(policy.nh) for k in range(2)])
        finite = bool(torch.isfinite(states).all() and torch.isfinite(dark).all() and all_inputs_finite)
        physical_valid = physical['eligible'] and minimum_wall>PHYSICAL_TOL and minimum_goal>PHYSICAL_TOL
        physical_valid &= max_displacement_error<=PHYSICAL_TOL and max_radius_error<=PHYSICAL_TOL and max_heading_increment_error<=PHYSICAL_TOL
        input_match = phi0_rgb_error<=ATOL and phi0_code_error<=ATOL
        qualified = physical_valid and finite and input_match and replay['qualified'] and phi0['qualified'] and closure_check['qualified']
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if cuda_rng is not None: assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
        metadata = dict(qualified=bool(qualified),phases=phases,state_timing='Before first blind input; no blind update performed',
            theta='Relative rigid rotation phi, phi0 is the native prefix; uniform [0,2pi), closure row excluded',
            physical=physical, all_angle_physical=dict(qualified=bool(physical_valid),
                minimum_wall_clearance=minimum_wall,minimum_goal_clearance=minimum_goal,
                max_displacement_rotation_error=max_displacement_error,max_radius_error=max_radius_error,
                max_heading_increment_error=max_heading_increment_error,tolerance=PHYSICAL_TOL),
            native_prefix_replay=replay,replay_phi0=phi0,closure_2pi=closure_check,
            numerical_guards=dict(batch_size=batch,phi0_rgb_max_error=phi0_rgb_error,phi0_current_max_error=phi0_code_error,
                finite=finite,phi0_input_qualified=input_match,global_rng_unchanged=True,atol=ATOL,rtol=RTOL),
            family_radius_raw=radius, image_noise='clip(observed_RGB+(clean_rotated-clean_native),0,1); effective observed residual, not reconstructed unclipped Gaussian noise',
            preparation='Exactly the saved visible history and saved initial h/c; no warm-up, no state averaging or phase interpolation',
            native_endpoint=dict(pos=endpoint.tolist(),body_heading=float(rt['obs']['hd'][slot]),
                travel_dir=rt['obs']['dir'][slot].tolist(),velocity=rt['obs']['vel'][slot].tolist(),
                goal_pos=goal_pos.tolist(),pa=int(rt['pa'][slot]),pr=float(rt['pr'][slot])),
            grid_pulse_shams_exact=not bool(pop.n_pgc), current_pgc_slots=pop.pgc_idx.tolist())
        return dict(theta=theta,states=states,current_codes=codes,dark_inputs=dark,pulse_inputs=pulses,
                    original_context=context,scale=scale,metadata=metadata)
    finally:
        policy.train(before_mode)
