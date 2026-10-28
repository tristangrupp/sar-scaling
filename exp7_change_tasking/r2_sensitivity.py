"""Round 2 sensitivity checks on the chance-corrected VV curve (r2_analyze.py holds the main result).

1. noisy truth out: drop pairs whose Capella log-ratio spread (robust sd) exceeds 2.5 dB. Above that,
   most of the window counts as near a change patch, so the truth is mostly diffuse (post hoc check;
   three such pairs hold 42% of all patches).
2. pairs equal: weight each patch 1 / (patches in its pair), so every pair counts once.
3. 4 m tolerance: the truth rebuilt with a +-2 px (4 m) shift tolerance (data/r2/*_patches_tol2.csv).
4. Spot-check-ready subset: spotlight pairs only with a Capella gap of 7 days or less.
Each reports the 50% and 90% areas of the chance-corrected fit with 95% site-bootstrap intervals (500
draws) and the binned excess detection. Writes r2_sensitivity.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import r2_analyze as A  # noqa: E402


def run(P, name, rng, w=None):
    x = np.log10(P.area_m2.values)
    y = P.vv_loose.values.astype(float)
    c = P.vv_chance.values
    t = A.fit_corrected(x, y, c, w)
    out = {"case": name, "patches": int(len(P)), "pairs": int(P.tag.nunique()), "sites": int(P.site_id.nunique()),
           "A50": A.area_at(t, 0.5), "A90": A.area_at(t, 0.9), "A10": A.area_at(t, 0.1)}
    bs = []
    for bw in A.boot_weights(P.site_id, rng, 500):
        bs.append(A.fit_corrected(x, y, c, bw if w is None else bw * w))
    for k, p in (("A50", .5), ("A90", .9), ("A10", .1)):
        v = np.array([A.area_at(b, p) for b in bs])
        v = v[np.isfinite(v)]
        out[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    ex = (P.vv_loose.astype(float) - P.vv_chance) / (1 - P.vv_chance.clip(upper=0.999))
    ww = np.ones(len(P)) if w is None else w
    b = A.wbinned(P.bin.values, ex.values, ww)
    out["excess_by_bin"] = {f"{A.EDGES[i]:.0f}-{A.EDGES[i + 1]:.0f}": round(float(v), 3) for i, v in enumerate(b) if np.isfinite(v)}
    print(json.dumps(out, default=lambda o: round(o, 1)), flush=True)
    return out


def main():
    rng = np.random.default_rng(1)
    pr, ok, P = A.load()
    res = [run(P, "primary (all pairs)", rng)]
    noisy = ok.loc[ok.lr_mad_sd_db > 2.5, "tag"]
    res.append(run(P[~P.tag.isin(noisy)], f"noisy truth out ({len(noisy)} pairs with sd > 2.5 dB)", rng))
    w = 1.0 / P.groupby("tag").tag.transform("size").values
    res.append(run(P, "pairs weighted equally", rng, w * len(P) / w.sum()))
    parts = []
    for r in ok.itertuples():
        f = A.OUT / f"{r.tag}_patches_tol2.csv"
        if f.exists() and f.stat().st_size > 5:
            q = pd.read_csv(f)
            if len(q):
                q["tag"], q["site_id"] = r.tag, r.site_id
                parts.append(q)
    T2 = pd.concat(parts, ignore_index=True)
    T2["bin"] = np.digitize(T2.area_m2, A.EDGES) - 1
    res.append(run(T2, "4 m shift tolerance", rng))
    sel = pd.read_csv(ROOT / "r2_pairs.csv")
    sel["tag"] = [f"site{a}_{b}_{c}" for a, b, c in zip(sel.site_id, sel.x1, sel.x2)]
    short = sel[(sel.x_gap_d <= 7) & (sel.mode1 == "spotlight")].tag
    res.append(run(P[P.tag.isin(short)], "spotlight, Capella gap <= 7 days", rng))
    res.append(run(P[~P.tag.isin(noisy) & (P.land != "vegetation/other")], "noisy truth out, vegetation out", rng))
    (ROOT / "r2_sensitivity.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
