"""Step 5: Capella-vs-Capella repeat ceiling at each target resolution (DESIGN.md, "Repeat-image ceiling").

Pairs: experiment 7's r2_pairs.csv (same polarization, look direction and incidence within 5 deg). Up to 12 land
chips per pair inside both footprints. Both images get their own degradation to each R. B is co-registered to A
(phase correlation of the 1.25 m base images in dB, median over the pair's chips, whole base pixels), then scored as
a prediction of A with the same metrics and masks as the baselines (water, slope > 10 deg, no data).

Writes ceiling_chips.csv and ceiling_summary.csv. CPU only. Run in the background with output to a log.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import common  # noqa: F401
from common import (BASE_RES, CHIP_M, HERE, MB, NB, NS, NT, PROJ, RES_LIST, S1_RES, SC, SLOPE_MAX, Bytes, block_mean,
                    capella_meta, capella_read, crop_c, db, degrade, finish, flat_mask, read_dem_slope,
                    read_worldcover, scores, solve_sigma, utm_epsg)

import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from build_chips import chip_positions, sub_at
from baselines import METRICS, site_boot

MAX_CHIPS = 12


def gr_az(cm):
    inc = cm["inc"] if np.isfinite(cm["inc"]) else 35.0
    gr = cm["gr_res"] if np.isfinite(cm["gr_res"]) else 1.0 / np.sin(np.radians(inc))
    az = cm["az_res"] if np.isfinite(cm["az_res"]) else 1.0
    return gr, az


def run_pair(p):
    from skimage.registration import phase_cross_correlation
    t0 = time.time()
    acc = Bytes()
    cmA, cmB = capella_meta(p["href1"]), capella_meta(p["href2"])
    epsg = utm_epsg(float(p["lon"]), float(p["lat"]))
    pa = set(chip_positions(p["href1"], cmA, epsg, acc))
    pb = set(chip_positions(p["href2"], cmB, epsg, acc))
    pos = sorted(pa & pb)
    if not pos:
        return [], {"pair": p["tag"], "status": "no_common_chips", "bytes": acc.n}
    rng = np.random.default_rng(0)
    if len(pos) > MAX_CHIPS:
        pos = [pos[i] for i in sorted(rng.choice(len(pos), MAX_CHIPS, replace=False))]
    xs, ys = [q[0] for q in pos], [q[1] for q in pos]
    ax0, ay1 = min(xs), max(ys)
    nc = int(round((max(xs) + CHIP_M - ax0) / S1_RES))
    nr = int(round((ay1 - (min(ys) - CHIP_M)) / S1_RES))
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    bb = list(shapely.box(*inv.transform(ax0 - 100, ay1 - nr * 10 - 100), *inv.transform(ax0 + nc * 10 + 100, ay1 + 100)).bounds)
    _, slope = read_dem_slope(epsg, ax0, ay1, nr, nc, bb, acc)
    wc = read_worldcover(epsg, ax0, ay1, nr, nc, bb, acc)
    A, B = [], []
    for x0, y1 in pos:
        A.append(capella_read(p["href1"], cmA, epsg, x0 - MB * BASE_RES, y1 + MB * BASE_RES, BASE_RES, NB, acc))
        B.append(capella_read(p["href2"], cmB, epsg, x0 - MB * BASE_RES, y1 + MB * BASE_RES, BASE_RES, NB, acc))
    shifts = []
    for a, b in zip(A, B):
        m = np.isfinite(a) & np.isfinite(b)
        if m.mean() < 0.8:
            continue
        da, dbb = db(np.where(m, a, np.nanmedian(a))), db(np.where(m, b, np.nanmedian(b)))
        lo, hi = np.percentile(da[m], [1, 99])
        sh, _, _ = phase_cross_correlation(np.clip(da, lo, hi), np.clip(dbb, lo, hi), upsample_factor=4,
                                           normalization=None)
        shifts.append(sh)
    if not shifts:
        return [], {"pair": p["tag"], "status": "no_overlap", "bytes": acc.n}
    sy, sx = np.median(np.array(shifts), axis=0)
    # phase_cross_correlation returns the shift that registers B onto A (rows, cols): B content at A index i sits
    # at B index i - shift, so B is read at fine offset fx = -sx * BASE_RES, fy = +sy * BASE_RES (north positive).
    fx = float(-np.round(sx) * BASE_RES)
    fy = float(np.round(sy) * BASE_RES)
    if abs(fx) > 40 or abs(fy) > 40:
        return [], {"pair": p["tag"], "status": "shift_too_large", "fx": fx, "fy": fy, "bytes": acc.n}
    gA, gB = gr_az(cmA), gr_az(cmB)
    sigA = {R: solve_sigma(R, float(np.sqrt(gA[0] * gA[1])), cmA["src_px"]) for R in RES_LIST}
    sigB = {R: solve_sigma(R, float(np.sqrt(gB[0] * gB[1])), cmB["src_px"]) for R in RES_LIST}
    rows = []
    for k, ((x0, y1), a, b) in enumerate(zip(pos, A, B)):
        r0 = int(round((ay1 - y1) / S1_RES))
        c0 = int(round((x0 - ax0) / S1_RES))
        wcc = wc[r0:r0 + NS, c0:c0 + NS]
        slc = slope[r0:r0 + NS, c0:c0 + NS] if slope is not None else np.full((NS, NS), np.nan)
        land10 = (wcc != 0) & (wcc != 80) & np.isfinite(slc) & (slc <= SLOPE_MAX)
        up = np.kron(land10, np.ones((SC, SC), bool))
        for R in RES_LIST:
            fa, wa = degrade(a, sigA[R])
            fb, wb = degrade(b, sigB[R])
            ta = block_mean(sub_at(fa, 0.0, 0.0), 2)
            tb = block_mean(sub_at(fb, fx, fy), 2)
            va = sub_at(wa, 0.0, 0.0).reshape(NT, 2, NT, 2).min(axis=(1, 3)) >= 0.9 if sigA[R] > 0 else np.isfinite(ta)
            vb = sub_at(wb, fx, fy).reshape(NT, 2, NT, 2).min(axis=(1, 3)) >= 0.9 if sigB[R] > 0 else np.isfinite(tb)
            m = va & vb & np.isfinite(ta) & np.isfinite(tb) & (ta > 0) & (tb > 0) & up
            if m.mean() < 0.5:
                continue
            tA, tB = db(np.where(m, ta, 1)), db(np.where(m, tb, 1))
            fl = flat_mask(np.where(m, tA, np.nan), m)
            s = scores(crop_c(tB), crop_c(tA), crop_c(m), flat=crop_c(fl & m))
            if s is None:
                continue
            rows.append({"pair": p["tag"], "site_id": p["site_id"], "chip": k, "x0": x0, "y1": y1, "R": R,
                         "gap_d": p["x_gap_d"], "pol": p["pol"], "d_inc": p["d_inc"], "valid": round(float(m.mean()), 3),
                         "fx": fx, "fy": fy, "lon": p["lon"], "lat": p["lat"], **s})
    info = {"pair": p["tag"], "status": "ok", "n_chips": len(pos), "bytes": acc.n, "fx": fx, "fy": fy,
            "seconds": round(time.time() - t0)}
    print(f"{p['tag']}: {len(pos)} chips, shift ({fx}, {fy}) m, {acc.n / 1e6:.0f} MB, {time.time() - t0:.0f}s", flush=True)
    return rows, info


def href_of(cid):
    ts = cid.split("_")[5]
    return f"https://capella-open-data.s3.amazonaws.com/data/{int(ts[:4])}/{int(ts[4:6])}/{int(ts[6:8])}/{cid}/{cid}.tif"


def cross_pairs(max_sites=40):
    """Different-geometry repeats (secondary ceiling): same site and polarization, 0.5-12 days apart, opposite orbit
    direction or incidence differing by at least 10 deg. One pair per site (smallest gap)."""
    x = pd.read_csv(PROJ / "exp7_change_tasking" / "xband_collects.csv")
    x = x[x.provider == "capella"].copy()
    x["t"] = pd.to_datetime(x.datetime, format="mixed", utc=True)
    out = []
    for sid, g in x.sort_values("t").groupby("site_id"):
        g = g.reset_index(drop=True)
        best = None
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                gap = (b.t - a.t).total_seconds() / 86400
                if gap > 12:
                    break
                if gap < 0.5 or a.pol != b.pol:
                    continue
                if a.orbit == b.orbit and abs(a.inc - b.inc) < 10:
                    continue
                if best is None or gap < best["x_gap_d"]:
                    best = {"site_id": sid, "xid1": a.xband_id, "xid2": b.xband_id, "href1": href_of(a.xband_id),
                            "href2": href_of(b.xband_id), "x_gap_d": round(gap, 2), "pol": a.pol,
                            "d_inc": round(abs(a.inc - b.inc), 1), "same_orbit": a.orbit == b.orbit,
                            "lon": (a.lon + b.lon) / 2, "lat": (a.lat + b.lat) / 2}
        if best:
            out.append(best)
    d = pd.DataFrame(out).sample(frac=1, random_state=0).head(max_sites).reset_index(drop=True)
    d["tag"] = [f"x{s}_{i}" for i, s in enumerate(d.site_id)]
    return d


def main():
    cross = "--cross" in sys.argv
    sfx = "_cross" if cross else ""
    if cross:
        pr = cross_pairs()
    else:
        pr = pd.read_csv(PROJ / "exp7_change_tasking" / "r2_pairs.csv")
        pr["tag"] = [f"e7s{s}_{i}" for i, s in enumerate(pr.site_id)]
    print(f"{len(pr)} pairs at {pr.site_id.nunique()} sites", flush=True)
    rows, infos = [], []
    with ProcessPoolExecutor(4) as ex:
        futs = [ex.submit(run_pair, r) for r in pr.to_dict("records")]
        for f in as_completed(futs):
            try:
                r, i = f.result()
                rows += r
                infos.append(i)
            except Exception:
                traceback.print_exc()
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"ceiling_chips{sfx}.csv", index=False)
    it = pd.DataFrame(infos)
    it.to_csv(HERE / f"ceiling_pairs{sfx}.csv", index=False)
    print("pairs by status:", it.status.value_counts().to_dict(), f"bytes {it.bytes.sum() / 1e9:.2f} GB", flush=True)
    out = []
    df["site_id"] = df.site_id.astype(str)
    for R, g in df.groupby("R"):
        rec = {"R": R, "n_chips": len(g), "n_pairs": g.pair.nunique(), "n_sites": g.site_id.nunique()}
        for mt in METRICS:
            if mt in g:
                med, lo, hi, _ = site_boot(g, mt)
                rec.update({mt: round(med, 4), mt + "_lo": round(lo, 4), mt + "_hi": round(hi, 4)})
        out.append(rec)
    s = pd.DataFrame(out)
    s.to_csv(HERE / f"ceiling_summary{sfx}.csv", index=False)
    print(s[["R", "n_chips", "n_pairs", "n_sites", "psnr", "ssim", "detail_r", "hp_r", "edge_f1", "gms",
             "flat_ratio"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
