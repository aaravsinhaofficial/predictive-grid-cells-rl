"""Private observation clock for the native sensory-memory task.

Physics, rendering, reward, goal snapshots and action history stay native.
Only RGB and the current population-code input are masked.
"""
import torch


def max_gap_at(frames_done, ramp_frames=200_000_000, max_gap=60):
    if frames_done < 0 or ramp_frames <= 0 or max_gap < 0:
        raise ValueError('Invalid curriculum parameters')
    return min(int(max_gap), int(max_gap) * int(frames_done) // int(ramp_frames))


def gate_observation(rgb, current_code, blind):
    """Pure, non-mutating gate; native rendering must already have occurred."""
    if blind.dtype != torch.bool or blind.ndim != 1:
        raise ValueError('Expected a one-dimensional Boolean observation mask')
    if rgb.shape[0] != len(blind) or current_code.shape[0] != len(blind):
        raise ValueError('Observation/mask batch mismatch')
    return (rgb * (~blind)[:, None, None, None],
            current_code * (~blind)[:, None])


class MemoryClock:
    """15 visible observations after a hit, then privately sampled blind gaps.

Call advance AFTER each native step. A hit renews visibility on the next
observation. A terminal resets knowledge even when it also reports a hit.
For D>0 a full base-slot duration vector is drawn every step, independent of
which slots hit or cross a phase boundary. Evaluation branches can repeat it.
Curriculum changes affect newly entered gaps only. D=0 is always visible and
does not consume the private duration generator.
"""
    def __init__(self, n, device, seed, groups=1, visible_steps=15):
        if min(n, groups, visible_steps) < 1:
            raise ValueError('Positive sizes required')
        self.n, self.groups, self.visible_steps = int(n), int(groups), int(visible_steps)
        self.known = torch.zeros(n * groups, dtype=torch.bool, device=device)
        self.blind = torch.zeros_like(self.known)
        self.remaining = torch.full((n * groups,), visible_steps, dtype=torch.long, device=device)
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.draw_steps = 0

    def mask(self):
        return self.known & self.blind

    def advance(self, hit, done, max_gap):
        if hit.shape != self.known.shape or done.shape != hit.shape:
            raise ValueError('Clock event shape mismatch')
        if hit.dtype != torch.bool or done.dtype != torch.bool or max_gap < 0:
            raise ValueError('Invalid clock events or gap')
        durations = None
        if max_gap:
            durations = torch.randint((int(max_gap) + 1) // 2, int(max_gap) + 1,
                                      (self.n,), generator=self.generator,
                                      device=self.remaining.device).repeat(self.groups)
            self.draw_steps += 1
        active = self.known & ~hit & ~done
        remaining = self.remaining - active.long()
        boundary = active & (remaining == 0)
        entering = boundary & ~self.blind & bool(max_gap)
        leaving = boundary & self.blind
        self.blind = (self.blind | entering) & ~leaving & ~(hit | done)
        if durations is not None:
            remaining = torch.where(entering, durations, remaining)
        renew = leaving | hit | done | (boundary & ~entering)
        self.remaining = torch.where(renew, torch.full_like(remaining, self.visible_steps), remaining)
        self.known = (self.known | hit) & ~done
        if not max_gap:
            self.blind = torch.zeros_like(self.blind)
        return entering

    def state_dict(self):
        return dict(n=self.n, groups=self.groups, visible_steps=self.visible_steps,
                    known=self.known.detach().cpu().clone(), blind=self.blind.detach().cpu().clone(),
                    remaining=self.remaining.detach().cpu().clone(),
                    rng=self.generator.get_state().cpu().clone(), draw_steps=self.draw_steps)

    def load_state_dict(self, state):
        if (state['n'], state['groups'], state['visible_steps']) != (self.n, self.groups, self.visible_steps):
            raise ValueError('Clock configuration changed')
        device = self.remaining.device
        for name in ('known', 'blind', 'remaining'):
            value = state[name]
            if value.shape != getattr(self, name).shape or value.dtype != getattr(self, name).dtype:
                raise ValueError('Invalid clock tensor')
            setattr(self, name, value.clone().to(device))
        self.generator.set_state(state['rng'].cpu().clone())
        self.draw_steps = int(state['draw_steps'])
