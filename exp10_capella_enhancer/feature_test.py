"""Experiment 8's linear-feature visibility test, applied to any 2.5 m output (DESIGN.md, "Metrics").

Uses experiment 8's 803 accepted segments in its 22 headline scenes, its stored Sentinel-1 arrays (10 m, VV/VH,
single date and multi-date mean) and its matched-null contrast test (measure.contrast_stat / zscores, sensor cell
10 m, detected when |z| > 2 in the one output channel).

Approximations against experiment 8 (checked by `--check`, which reruns its multi-date VV test at 10 m):
segments are rebuilt as straight lines from their stored midpoint, tangent and length, and crossing stretches are
not masked (experiment 8 kept no line geometry in its tables).

  python feature_test.py --check                 # reproduce exp8's multi-date VV rate (39%) with this code
  python feature_test.py --baselines             # bicubic / n2n / degraded-Capella rates -> feature_baselines.csv
eval.py calls run(predict_fn) for a model.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import common  # noqa: F401
from common import BASE_RES, DATA, EXP8, HERE, OUT_RES, RES_LIST, bicubic_up, db, finish, n2n_apply, read_dem_slope

import argparse
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(EXP8))
from measure import D, contrast_stat, zscores  # noqa: E402

CACHE = DATA / "feature_test"


def segments():
    meta = pd.read_csv(EXP8 / "scene_meta.csv")
    good = set(meta[meta.osm_refined.astype(bool)].capella_id)
    s = pd.read_csv(EXP8 / "segments_all.csv")
    return s[(s.status == "ok") & s.capella_id.isin(good)].copy()


def frame(r):
    """Profile sample points of a straight segment: (len, len(D)) arrays X, Y at 1 m steps along it."""
    s = np.arange(0.5, r.length_m, 1.0) - r.length_m / 2
    t = np.array([r.tan_x, r.tan_y])
    n = np.array([-t[1], t[0]])
    Px = r.mid_x + s * t[0]
    Py = r.mid_y + s * t[1]
    X = Px[:, None] + D[None, :] * n[0]
    Y = Py[:, None] + D[None, :] * n[1]
    return X, Y


def sample(arr, gx0, gy1, res, X, Y):
    col = (X - gx0) / res - 0.5
    row = (gy1 - Y) / res - 0.5
    return ndimage.map_coordinates(arr, [row.ravel(), col.ravel()], order=1, mode="constant", cval=np.nan,
                                   prefilter=False).reshape(X.shape)


def detect(lin_img, gx0, gy1, res, seg, shift=(0.0, 0.0)):
    out = []
    for r in seg.itertuples():
        X, Y = frame(r)
        I = sample(lin_img, gx0, gy1, res, X + shift[0], Y + shift[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = np.nanmean(I, axis=0)
        o, nl, _ = contrast_stat(p, r.cap_center, r.cap_width, 10.0)
        z, _ = zscores(o, nl)
        out.append({"seg_id": r.seg_id, "capella_id": r.capella_id, "type": r.type, "cap_width": r.cap_width,
                    "z": float(z) if np.isfinite(z) else np.nan, "det": bool(np.isfinite(z) and abs(z) > 2)})
    return out


def scene_inputs(cid):
    """exp8 arrays plus a slope layer (cached). Returns dict with s1 (5, 440, 440) dB and grid origin."""
    z = np.load(EXP8 / "data" / cid / "arrays.npz")
    n = z["single_vv"].shape[0]
    gxS, gyS, epsg = float(z["gxS"]), float(z["gyS"]), int(z["epsg"])
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{cid}_slope.npy"
    if f.exists():
        slope = np.load(f)
    else:
        from pyproj import Transformer
        inv = Transformer.from_crs(epsg, 4326, always_xy=True)
        lo0, la0 = inv.transform(gxS - 200, gyS - n * 10 - 200)
        lo1, la1 = inv.transform(gxS + n * 10 + 200, gyS + 200)
        _, slope = read_dem_slope(epsg, gxS, gyS, n, n, [lo0, la0, lo1, la1])
        np.save(f, slope)
    # experiment 8 stored these arrays in dB already
    s1 = np.stack([z["single_vv"].astype("float32"), z["single_vh"].astype("float32"),
                   z["multi_vv"].astype("float32"), z["multi_vh"].astype("float32"), slope]).astype("float32")
    return {"s1": s1, "gxS": gxS, "gyS": gyS, "z": z}


def run(pred_fn, tag, seg=None):
    """pred_fn(s1 (5, H, W) dB at 10 m) -> dB at 2.5 m (4H, 4W). Returns per-segment detections."""
    seg = segments() if seg is None else seg
    rows = []
    for cid, g in seg.groupby("capella_id"):
        si = scene_inputs(cid)
        out = pred_fn(si["s1"])
        lin = 10 ** (out.astype("float64") / 10)
        for d in detect(lin, si["gxS"], si["gyS"], OUT_RES, g):
            rows.append({**d, "method": tag})
    return rows


def degraded_capella(cid, R):
    """exp8's 1 m Capella (coarse-shifted frame) blurred to effective resolution R; returns (lin, gx, gy, shift)."""
    z = np.load(EXP8 / "data" / cid / "arrays.npz")
    lin = 10 ** (z["cap_db"].astype("float64") / 10)
    m = np.isfinite(lin)
    from common import solve_sigma
    sig = solve_sigma(R, 1.0, 1.0)   # 1 m grid, about 1 m native: an approximation of the build's chain
    if sig > 0:
        num = ndimage.gaussian_filter(np.where(m, lin, 0), sig / 1.0, truncate=3.5)
        den = ndimage.gaussian_filter(m.astype(float), sig / 1.0, truncate=3.5)
        lin = np.where(den > 0.9, num / np.maximum(den, 1e-9), np.nan)
    return lin, float(z["gxC"]), float(z["gyC"]), (float(z["dx"]), float(z["dy"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    seg = segments()
    print(f"{len(seg)} segments in {seg.capella_id.nunique()} scenes", flush=True)
    rows = []
    if a.check:
        for cid, g in seg.groupby("capella_id"):
            z = np.load(EXP8 / "data" / cid / "arrays.npz")
            for d in detect(10 ** (z["multi_vv"].astype("float64") / 10), float(z["gxS"]), float(z["gyS"]), 10.0, g):
                rows.append({**d, "method": "exp8_multi_vv_10m"})
        df = pd.DataFrame(rows).merge(seg[["seg_id", "multi_vv_det"]], on="seg_id")
        print(f"this code: {df.det.mean():.3f}; exp8 table: {df.multi_vv_det.mean():.3f}; "
              f"agreement {(df.det == df.multi_vv_det).mean():.3f}", flush=True)
        df.to_csv(HERE / "feature_check.csv", index=False)
        return
    if a.baselines:
        rows += run(lambda s: bicubic_up(s[2]), "bicubic_multi", seg)
        rows += run(lambda s: bicubic_up(s[0]), "bicubic_single", seg)
        rows += run(lambda s: bicubic_up(db(n2n_apply(10 ** (s[2] / 10), 10 ** (s[3] / 10)))), "n2n_multi", seg)
        for R in RES_LIST:
            for cid, g in seg.groupby("capella_id"):
                lin, gx, gy, sh = degraded_capella(cid, R)
                for d in detect(lin, gx, gy, 1.0, g, shift=sh):
                    rows.append({**d, "method": f"capella_R{R:g}"})
            print("done capella", R, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(HERE / "feature_baselines.csv", index=False)
        print(df.groupby("method").det.agg(["mean", "size"]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
