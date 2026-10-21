"""Classic speckle filters on linear-power intensity, numpy/scipy only.

Notation: I = observed intensity, m and v = local mean and variance, L = equivalent number of
looks of the input, Cu = 1/sqrt(L) (speckle coefficient of variation), Ci = sqrt(v)/m.

  Lee (1980), "Digital image enhancement and noise filtering by use of local statistics",
    IEEE TPAMI 2(2):165-168.  Multiplicative model I = x*n, E[n]=1, var(n)=Cu^2:
      var_x = (v - m^2 Cu^2) / (1 + Cu^2),   k = clip(var_x / v, 0, 1),   out = m + k (I - m)
  Refined Lee (1981), "Refined filtering of image noise using local statistics", CGIP 15:380-389.
    Same estimator, but m and v come from one of 8 edge-aligned half-windows of a 7x7 window,
    picked from the 3x3 sub-means (strongest of 4 directional gradients, then the side of the
    edge closer to the centre).  Follows the widely used GEE/SNAP formulation.
  Gamma-MAP: Lopes, Nezry, Touzi, Laur (1990), "Maximum a posteriori speckle filtering and first
    order texture models in SAR images", IGARSS, 2409-2412.  Gamma prior on x, gamma speckle:
      Ci <= Cu       -> out = m                       (homogeneous)
      Ci >= sqrt(2)Cu -> out = I                       (point target / strong edge)
      else  a = (1 + Cu^2) / (Ci^2 - Cu^2),  b = a - L - 1,
            out = (b m + sqrt(m^2 b^2 + 4 a L I m)) / (2 a)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _fill(I):
    """Replace NaN by the global median so filters do not spread NaN; caller re-masks."""
    return np.where(np.isfinite(I), I, np.nanmedian(I)).astype("float64")


def boxcar(I, w):
    return ndi.uniform_filter(_fill(I), w, mode="reflect")


def _mv(I, w):
    m = ndi.uniform_filter(I, w, mode="reflect")
    v = np.maximum(ndi.uniform_filter(I * I, w, mode="reflect") - m * m, 0)
    return m, v


def lee(I, L, w=7):
    I = _fill(I)
    m, v = _mv(I, w)
    cu2 = 1.0 / L
    var_x = np.maximum((v - m * m * cu2) / (1 + cu2), 0)
    k = np.clip(var_x / np.maximum(v, 1e-12), 0, 1)
    return m + k * (I - m)


def _dir_masks():
    """8 half-window masks of a 7x7 window. Order matches the (direction, side) choice below."""
    r, c = np.mgrid[-3:4, -3:4]
    masks = {
        ("v", 0): c <= 0, ("v", 1): c >= 0,        # vertical edge: keep left / right half
        ("h", 0): r <= 0, ("h", 1): r >= 0,        # horizontal edge: keep upper / lower half
        ("d1", 0): r >= c, ("d1", 1): r <= c,      # edge along the main diagonal
        ("d2", 0): r + c <= 0, ("d2", 1): r + c >= 0,  # edge along the anti-diagonal
    }
    return masks


def refined_lee(I, L):
    I = _fill(I)
    m3 = ndi.uniform_filter(I, 3, mode="reflect")
    # 3x3 sub-means at offsets -2, 0, +2 inside the 7x7 window
    P = {}
    for dr in (-2, 0, 2):
        for dc in (-2, 0, 2):
            P[(dr, dc)] = ndi.shift(m3, (-dr, -dc), order=0, mode="nearest")
    g = {
        "v": np.abs(P[(-2, -2)] + P[(0, -2)] + P[(2, -2)] - P[(-2, 2)] - P[(0, 2)] - P[(2, 2)]),
        "h": np.abs(P[(-2, -2)] + P[(-2, 0)] + P[(-2, 2)] - P[(2, -2)] - P[(2, 0)] - P[(2, 2)]),
        "d1": np.abs(P[(0, -2)] + P[(2, -2)] + P[(2, 0)] - P[(-2, 0)] - P[(-2, 2)] - P[(0, 2)]),
        "d2": np.abs(P[(-2, 0)] + P[(-2, -2)] + P[(0, -2)] - P[(2, 0)] - P[(2, 2)] - P[(0, 2)]),
    }
    dirs = ["v", "h", "d1", "d2"]
    best = np.argmax(np.stack([g[d] for d in dirs]), 0)
    c0 = P[(0, 0)]
    # which side of the edge is closer to the centre sub-mean
    side_pairs = {"v": ((0, -2), (0, 2)), "h": ((-2, 0), (2, 0)), "d1": ((2, -2), (-2, 2)), "d2": ((-2, -2), (2, 2))}
    masks = _dir_masks()
    out_m = np.zeros_like(I)
    out_v = np.zeros_like(I)
    for di, d in enumerate(dirs):
        a, b = side_pairs[d]
        side = (np.abs(P[b] - c0) < np.abs(P[a] - c0)).astype(int)
        for s in (0, 1):
            k = masks[(d, s)].astype("float64")
            k /= k.sum()
            sel = (best == di) & (side == s)
            if not sel.any():
                continue
            mm = ndi.correlate(I, k, mode="reflect")
            vv = np.maximum(ndi.correlate(I * I, k, mode="reflect") - mm * mm, 0)
            out_m[sel], out_v[sel] = mm[sel], vv[sel]
    cu2 = 1.0 / L
    var_x = np.maximum((out_v - out_m ** 2 * cu2) / (1 + cu2), 0)
    k = np.clip(var_x / np.maximum(out_v, 1e-12), 0, 1)
    return out_m + k * (I - out_m)


def gamma_map(I, L, w=7):
    I = _fill(I)
    m, v = _mv(I, w)
    cu2 = 1.0 / L
    ci2 = v / np.maximum(m * m, 1e-12)
    cmax2 = 2 * cu2
    a = (1 + cu2) / np.maximum(ci2 - cu2, 1e-12)
    b = a - L - 1
    out = (b * m + np.sqrt(np.maximum(m * m * b * b + 4 * a * L * I * m, 0))) / (2 * a)
    out = np.where(ci2 <= cu2, m, out)
    out = np.where(ci2 >= cmax2, I, out)
    return out
