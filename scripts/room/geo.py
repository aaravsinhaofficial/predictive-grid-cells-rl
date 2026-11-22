"""Shortest walkable path between two points of a changed room.

The agent is a point that keeps `margin` away from the partition and the
obstacle, so the forbidden regions are the rectangles grown by `margin`. The
shortest path around axis-aligned rectangles runs through their corners:
visibility graph over the corners (all pairs solved once per room), then every
(start, goal) pair of a batch is connected to it in one vectorised pass.
In the open room the result is the straight-line distance.
"""
import numpy as np


def blocked(P, Q, rects, eps=1e-5):
  """True where the segment P-Q crosses the interior of a rectangle.
  P, Q [..., 2] (broadcast); rects [K, 4] = (xmin, xmax, ymin, ymax)."""
  P, Q = np.broadcast_arrays(np.asarray(P, float), np.asarray(Q, float))
  d = Q - P
  out = np.zeros(P.shape[:-1], bool)
  for x0, x1, y0, y1 in rects:
    if x1 - x0 <= 2 * eps or y1 - y0 <= 2 * eps:
      continue
    lo, hi = np.array([x0 + eps, y0 + eps]), np.array([x1 - eps, y1 - eps])
    par = np.abs(d) < 1e-12
    dd = np.where(par, 1.0, d)
    t1, t2 = (lo - P) / dd, (hi - P) / dd
    inside = (P > lo) & (P < hi)
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    enter = np.maximum(tmin[..., 0], tmin[..., 1])
    leave = np.minimum(tmax[..., 0], tmax[..., 1])
    out |= np.maximum(enter, 0.0) < np.minimum(leave, 1.0)
  return out


class Room:
  def __init__(self, boxes, margin=0.08, lim=1.17):
    b = np.asarray(boxes, float).reshape(-1, 4)
    b = b[(np.abs(b) < 50).all(1)]                       # unused slots sit far away
    self.rects = np.stack([b[:, 0] - margin, b[:, 1] + margin,
                           b[:, 2] - margin, b[:, 3] + margin], 1) if len(b) else np.zeros((0, 4))
    c = np.array([[x, y] for x0, x1, y0, y1 in self.rects for x in (x0, x1) for y in (y0, y1)]
                 ).reshape(-1, 2)
    self.corners = c[(np.abs(c) <= lim + 1e-9).all(1)] if len(c) else c
    C = len(self.corners)
    D = np.full((C, C), np.inf)
    if C:
      vis = ~blocked(self.corners[:, None], self.corners[None], self.rects)
      D = np.where(vis, np.linalg.norm(self.corners[:, None] - self.corners[None], axis=-1), np.inf)
      for k in range(C):                                  # Floyd-Warshall
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    self.D = D

  def shortest(self, S, G):
    """S, G [B, 2] -> [B] shortest path lengths."""
    S, G = np.asarray(S, float).reshape(-1, 2), np.asarray(G, float).reshape(-1, 2)
    direct = np.where(blocked(S, G, self.rects), np.inf, np.linalg.norm(S - G, axis=1))
    if not len(self.corners):
      return direct
    c = self.corners
    ds = np.where(blocked(S[:, None], c[None], self.rects), np.inf,
                  np.linalg.norm(S[:, None] - c[None], axis=-1))
    dg = np.where(blocked(G[:, None], c[None], self.rects), np.inf,
                  np.linalg.norm(G[:, None] - c[None], axis=-1))
    via = (ds[:, :, None] + self.D[None] + dg[:, None, :]).min((1, 2))
    return np.minimum(direct, via)
