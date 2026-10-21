"""Policy network at the paper's full size (Banino et al. 2018, SI section 3c).

Vision: "four convolutional layers. The first had 16 filters of size 5x5 with
stride 2 and padding 2, the second 32 filters 5x5/2/2, the third 64 filters
5x5/2/2, the fourth 128 filters 5x5/2/2, all followed by a rectifier; the last
convolution was followed by a fully connected layer with 256 hidden units."
(84x84 -> 42 -> 21 -> 11 -> 6; 128*6*6 = 4608 -> 256.)

Policy: the conv embedding e_t is concatenated with the reward r_{t-1}, the
previous action (one-hot; all zeros on the first step of an episode), the
current population code g_t and the goal code g* (the code observed the last
time the goal was reached, zeros before that) and fed to an LSTM with 256
units; the actor is a linear layer with six units (softmax), the critic a
single linear unit.

`recurrent=False` swaps the LSTM for a two-layer MLP of the same width (256):
the agent then has NO memory across steps, so the supplied population code is
its only source of position. A policy LSTM fed its own previous action can
dead-reckon internally and ignore an external spatial code; the feed-forward
variant cannot, which makes it the cleaner test of code dependence.

`encoder_norm='layernorm'` bounds the embedding scale. Without it, RMSProp
with a small eps lets the conv/fc weights drift until |e| ~ 1e2, the LSTM
gates saturate (exactly 0/1 in float32) and the policy output becomes exactly
independent of every other input, including the population code (this happens
at lr >= 3e-4 with eps 1e-5).
"""
import torch
from torch import nn


def _ortho(layer, gain=1.0):
  nn.init.orthogonal_(layer.weight, gain)
  if layer.bias is not None:
    nn.init.zeros_(layer.bias)
  return layer


class ConvEncoder(nn.Module):
  """84x84 RGB in [0,1] -> 256-d embedding."""

  def __init__(self, arch='si', out_dim=256, img=84, norm='none'):
    super().__init__()
    if arch == 'si':
      convs = [nn.Conv2d(3, 16, 5, 2, 2), nn.Conv2d(16, 32, 5, 2, 2),
               nn.Conv2d(32, 64, 5, 2, 2), nn.Conv2d(64, 128, 5, 2, 2)]
    elif arch == 'dqn':   # the scaled-down net of banino/rl/nets.py
      convs = [nn.Conv2d(3, 16, 8, 4), nn.Conv2d(16, 32, 4, 2)]
    else:
      raise ValueError(f'unknown conv arch {arch!r}')
    layers = []
    for c in convs:
      layers += [_ortho(c, 2 ** 0.5), nn.ReLU()]
    self.conv = nn.Sequential(*layers)
    with torch.no_grad():
      flat = self.conv(torch.zeros(1, 3, img, img)).numel()
    self.fc = _ortho(nn.Linear(flat, out_dim), 2 ** 0.5)
    self.out_dim = out_dim
    self.norm = nn.LayerNorm(out_dim) if norm == 'layernorm' else nn.Identity()

  def forward(self, rgb):
    x = rgb * 2.0 - 1.0                      # the paper feeds images in [-1, 1]
    return torch.relu(self.norm(self.fc(self.conv(x).flatten(1))))


class PolicyNet(nn.Module):
  """Actor-critic over [e_t, g_t, g*, prev action, prev reward, extra]."""

  def __init__(self, n_actions=6, n_code=512, nh=256, recurrent=True,
               conv_arch='si', extra_in=0, encoder_norm='none'):
    super().__init__()
    self.encoder = ConvEncoder(conv_arch, 256, norm=encoder_norm)
    self.n_actions, self.n_code, self.nh = int(n_actions), int(n_code), int(nh)
    self.recurrent = bool(recurrent)
    self.extra_in = int(extra_in)
    in_dim = 256 + 2 * self.n_code + self.n_actions + 1 + self.extra_in
    if self.recurrent:
      self.lstm = nn.LSTMCell(in_dim, self.nh)
    else:
      self.mlp = nn.Sequential(_ortho(nn.Linear(in_dim, self.nh), 2 ** 0.5), nn.ReLU(),
                               _ortho(nn.Linear(self.nh, self.nh), 2 ** 0.5), nn.ReLU())
    self.pi = _ortho(nn.Linear(self.nh, self.n_actions), 0.01)
    self.v = _ortho(nn.Linear(self.nh, 1), 1.0)

  def zero_state(self, batch, device):
    return (torch.zeros(batch, self.nh, device=device),
            torch.zeros(batch, self.nh, device=device))

  def embed(self, rgb, chunk=1024):
    """rgb [B,3,84,84] in [0,1] -> e [B,256] (chunked for big batches)."""
    if rgb.shape[0] <= chunk:
      return self.encoder(rgb)
    return torch.cat([self.encoder(rgb[i:i + chunk]) for i in range(0, rgb.shape[0], chunk)])

  def _concat(self, e, code, goal_code, prev_action, prev_reward, extra):
    pa = torch.zeros(e.shape[0], self.n_actions, device=e.device)
    pa.scatter_(1, prev_action.clamp_min(0).unsqueeze(1), 1.0)
    pa = pa * (prev_action >= 0).float().unsqueeze(1)     # -1 = no previous action
    parts = [e, code, goal_code, pa, prev_reward.unsqueeze(1)]
    if self.extra_in:
      parts.append(extra)
    return torch.cat(parts, dim=-1)

  def step_from_embedding(self, e, code, goal_code, prev_action, prev_reward,
                          state, extra=None):
    x = self._concat(e, code, goal_code, prev_action, prev_reward, extra)
    if self.recurrent:
      h, c = self.lstm(x, state)
      return self.pi(h), self.v(h).squeeze(-1), (h, c)
    h = self.mlp(x)
    return self.pi(h), self.v(h).squeeze(-1), state

  def step(self, rgb, code, goal_code, prev_action, prev_reward, state, extra=None):
    """rgb [B,3,84,84] in [0,1]; code/goal_code [B,n_code]; prev_action [B]
    long (-1 = none); prev_reward [B]; state (h, c). Returns (logits, value,
    state)."""
    return self.step_from_embedding(self.embed(rgb), code, goal_code, prev_action,
                                    prev_reward, state, extra)

  @torch.no_grad()
  def diagnostics(self, rgb, code, goal_code, prev_action, prev_reward, state, extra=None):
    """Scale/saturation health on a batch: embedding magnitude, gate
    pre-activation magnitude, fraction of saturated gates (|pre| > 15 is
    exactly 0/1 after a float32 sigmoid), and the LSTM input drive from
    vision vs from the population code."""
    e = self.embed(rgb)
    x = self._concat(e, code, goal_code, prev_action, prev_reward, extra)
    out = dict(e_abs_mean=float(e.abs().mean()), e_abs_max=float(e.abs().max()))
    if self.recurrent:
      W = self.lstm.weight_ih
      pre = x @ W.T + state[0] @ self.lstm.weight_hh.T + self.lstm.bias_ih + self.lstm.bias_hh
      out['pre_abs_mean'] = float(pre.abs().mean())
      out['frac_saturated'] = float((pre.abs() > 15).float().mean())
      sl_v, sl_c = slice(0, 256), slice(256, 256 + self.n_code)
      out['drive_vision'] = float((x[:, sl_v] @ W[:, sl_v].T).abs().mean())
      out['drive_code'] = float((x[:, sl_c] @ W[:, sl_c].T).abs().mean())
    else:
      h = self.mlp(x)
      out['h_abs_mean'] = float(h.abs().mean())
      out['h_abs_max'] = float(h.abs().max())
    return out
