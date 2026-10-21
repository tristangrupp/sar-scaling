"""Data for IW (10 m) -> Stripmap (5 m) training and scoring.

Per tile (DATA/tiles/<tile_id>.npz): IW dB 320x320 (co-registered to the Stripmap), Stripmap dB
640x640 raw and Lee-filtered, WorldCover classes on the 5 m grid.

Following Amieva et al. (2026):
  - histogram matching: each Stripmap reference is mapped to its IW pair's distribution (per tile,
    fitted on valid land pixels), so the model learns detail, not the calibration difference
  - min-max normalization with the 0.1 and 99.9 percentiles of the training set, to [0, 1]
Water (WorldCover 80) and open ocean (no WorldCover) are masked: sea clutter changes between the
IW and Stripmap dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ROOT, TILES, hist_match

MIN_CORR = -1.0          # registration quality is enforced per scene in pairs_final.csv


def load_manifest(version="v1"):
    """v1: round 1 tiles (pairs_final.csv, splits.csv); v2: round 1 + round 2 inland tiles
    (pairs_final_v2.csv, splits_v2.csv, which keeps the round 1 assignments)."""
    sfx = "_v2" if version == "v2" else ""
    p = pd.read_csv(ROOT / f"pairs_final{sfx}.csv")          # finalize_pairs.py: scene-level registration
    p = p[p.use].copy()
    s = pd.read_csv(ROOT / f"splits{sfx}.csv")[["tile_id", "split"]]
    return p.merge(s, on="tile_id")


CACHE = TILES.parent / "tilecache"


def load_tile(tile_id):
    """Preprocessed tile (histogram matching is slow); cached uncompressed under DATA/tilecache.
    The cache key includes the tile npz mtime, so re-registered tiles are rebuilt."""
    src = TILES / f"{tile_id}.npz"
    cf = CACHE / f"{tile_id}_{int(src.stat().st_mtime)}.npz"
    if cf.exists():
        z = np.load(cf)
        return {**{k: z[k] for k in z.files}, "tile_id": tile_id}
    d = _load_tile(tile_id)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cf, **{k: v for k, v in d.items() if k != "tile_id"})
    return d


def _load_tile(tile_id):
    z = np.load(TILES / f"{tile_id}.npz")
    iw = z["iw"].astype("float32")
    sm = z["sm"].astype("float32")
    lee = z["lee"].astype("float32")
    wc = z["wc"]
    land = (wc != 80) & (wc != 0)
    m5 = np.isfinite(sm) & land
    iw_up_valid = np.kron(np.isfinite(iw), np.ones((2, 2), bool))
    m5 &= iw_up_valid
    # fit the histogram match on valid land pixels; IW land mask = 2x2 blocks that are all land
    land10 = land.reshape(320, 2, 320, 2).all(axis=(1, 3))
    iw_ref = np.where(land10 & np.isfinite(iw), iw, np.nan)
    sm_hm = hist_match(np.where(m5, sm, np.nan), iw_ref)
    lee_hm = hist_match(np.where(m5, lee, np.nan), iw_ref)
    return {"tile_id": tile_id, "iw": iw, "sm": sm, "lee": lee, "sm_hm": sm_hm, "lee_hm": lee_hm,
            "mask": m5, "wc": wc}


class Norm:
    """Global min-max from the training set's 0.1 / 99.9 percentiles (IW dB; the histogram-matched
    references share the IW distribution, so one pair of bounds serves both)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def __call__(self, a):
        return (a - self.lo) / (self.hi - self.lo)

    def inv(self, a):
        return a * (self.hi - self.lo) + self.lo

    @classmethod
    def fit(cls, tiles):
        v = np.concatenate([t["iw"][np.isfinite(t["iw"])][::7] for t in tiles])
        lo, hi = np.percentile(v, [0.1, 99.9])
        return cls(lo, hi)


def load_split(splits, man=None, min_corr=MIN_CORR):
    man = load_manifest() if man is None else man
    sel = man[man.split.isin(splits) & (man.corr10 >= min_corr)]
    return [dict(load_tile(r.tile_id), region=r.region, role=r.role, split=r.split) for r in sel.itertuples()]


def apply_change_mask(tiles):
    """Drop changed pixels (change_mask.py: |IW before - IW after| > 3 dB on the Stripmap date's
    bracketing same-track passes) from each tile's loss mask. Tiles without a mask keep theirs.
    Returns the share of previously valid pixels removed."""
    from common import DATA
    removed, total = 0, 0
    for t in tiles:
        f = DATA / "change" / f"{t['tile_id']}.npz"
        if not f.exists():
            continue
        ch = np.kron(np.load(f)["change10"], np.ones((2, 2), bool))
        removed += int((t["mask"] & ch).sum())
        total += int(t["mask"].sum())
        t["mask"] = t["mask"] & ~ch
    return removed / max(total, 1)
