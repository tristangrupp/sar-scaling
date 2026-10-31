"""Step 2: build chips per Capella item (DESIGN.md, "Targets", "Inputs", "Chips", "Alignment").

  python build_chips.py [--limit N] [--workers 6] [--ids CID ...]

Per item writes data/chips/<capella_id>.npz (accepted chips only) and data/chips/<capella_id>.json (every chip's
metadata, accepted or not, plus item-level facts). Finished items (json present) are skipped, so the run resumes.
merge_chips() at the end writes chips.csv (accepted chips) and chips_all.csv (all chips).
Run in the background with output to a log.
"""
from __future__ import annotations

import common  # noqa: F401  (GDAL env)
from common import (BASE_RES, CHIP_GAIN, CHIP_M, CHIP_R_MIN, CHIPS, COARSE_BIG, COARSE_Z, FINE_R, FINE_STEP, HERE,
                    ITEM_R_MIN, MB, MIN_VALID, NB, NS, NT, RES_LIST, S1_RES, SC, SLOPE_MAX, WC_NAMES, Bytes, _fwhm_1d,
                    acf_fwhm, block_mean, capella_meta, capella_read, corr, db, degrade, finish, highpass,
                    nominal_looks, pc_search, read_dem_slope, read_s1_pass, read_worldcover, s1_pass_groups,
                    solve_sigma, utm_epsg)

import argparse
import json
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

MAX_CHIPS = 40
WIN_CHIPS = 10   # 6.4 km
SINGLE_MAX_D = 12
MULTI_D = 30


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def chip_positions(href, cm, epsg, acc):
    """Chip origins (x0, y1) on the 640 m lattice whose square plus 80 m lies inside the Capella footprint."""
    from rasterio.warp import transform_bounds
    b = transform_bounds(cm["crs"], f"EPSG:{epsg}", *cm["bounds"], densify_pts=21)
    res = 40.0
    lvl = len(cm["overviews"]) - 1 if cm["overviews"] else None
    x0 = np.floor(b[0] / CHIP_M) * CHIP_M
    y1 = np.ceil(b[3] / CHIP_M) * CHIP_M
    nx = int(np.ceil((b[2] - x0) / res))
    ny = int(np.ceil((y1 - b[1]) / res))
    n = max(nx, ny)
    a = capella_read(href, cm, epsg, x0, y1, res, n, acc, level=lvl)
    v = np.isfinite(a)
    k = int(CHIP_M / res)
    m = 2  # 80 m margin in 40 m px
    out = []
    for i in range(0, n - k + 1, k):
        for j in range(0, n - k + 1, k):
            r0, r1, c0, c1 = i - m, i + k + m, j - m, j + k + m
            if r0 < 0 or c0 < 0 or r1 > n or c1 > n:
                continue
            if v[r0:r1, c0:c1].mean() >= 0.98:
                out.append((x0 + j * res, y1 - i * res))
    return out


def s1_select(epsg, ax0, ay1, nr, nc, poly_ll, t0, acc):
    """Single nearest pass (<= 12 d, >= 90% coverage preferred) and same-track passes within +-30 d."""
    items = pc_search(collections=["sentinel-1-rtc"], intersects=poly_ll.__geo_interface__,
                      datetime=f"{(t0 - pd.Timedelta(days=MULTI_D + 1)).date()}/{(t0 + pd.Timedelta(days=MULTI_D + 1)).date()}")
    g = s1_pass_groups(items)
    if not g:
        return None

    def dt(key):
        its = g[key]
        tt = pd.Timestamp(its[0].datetime).tz_localize(None)
        return (tt - t0).total_seconds() / 86400

    keys = sorted(g, key=lambda k: abs(dt(k)))
    single, sv = None, None
    best_cov = -1
    for k in keys:
        if abs(dt(k)) > SINGLE_MAX_D:
            break
        vv = read_s1_pass(g[k], "vv", epsg, ax0, ay1, nr, nc, acc)
        cov = 0 if vv is None else float(np.isfinite(vv).mean())
        if cov > best_cov:
            single, sv, best_cov = k, vv, cov
        if cov >= 0.9:
            break
    if single is None or best_cov < 0.5:
        return None
    rel, st = single[1], single[2]
    mk = sorted([k for k in g if k[1] == rel and k[2] == st and abs(dt(k)) <= MULTI_D], key=dt)
    out = {"single_key": single, "single_dt": dt(single), "single_ids": [i.id for i in g[single]],
           "multi_keys": mk, "rel": rel, "state": st, "single_cov": best_cov}
    S = {"vv1": sv, "vh1": read_s1_pass(g[single], "vh", epsg, ax0, ay1, nr, nc, acc)}
    stacks = {"vv": [], "vh": []}
    for k in mk:
        for p in ("vv", "vh"):
            a = S[p + "1"] if k == single else read_s1_pass(g[k], p, epsg, ax0, ay1, nr, nc, acc)
            if a is not None:
                stacks[p].append(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S["vvm"] = np.nanmean(np.array(stacks["vv"]), axis=0)
        S["vhm"] = np.nanmean(np.array(stacks["vh"]), axis=0)
        S["nm"] = np.sum(np.isfinite(np.array(stacks["vv"])), axis=0).astype("int16")
    out["n_multi"] = len(stacks["vv"])
    out["multi_dts"] = [round(dt(k), 2) for k in mk]
    return out, S


def coarse_align(href, cm, epsg, ax0, ay1, nr, nc, s1_vvm, acc):
    from skimage.feature import match_template
    lvl = None
    for i, f in enumerate(cm["overviews"]):
        if cm["native_px"] * f <= 5.0:
            lvl = i
    nb = int(COARSE_BIG / S1_RES)
    gxB, gyB = ax0 - COARSE_BIG, ay1 + COARSE_BIG
    n = max(nr, nc) + 2 * nb
    capB = capella_read(href, cm, epsg, gxB, gyB, S1_RES, n, acc, level=lvl)
    hpC = np.nan_to_num(highpass(db(capB)))
    hpS = np.nan_to_num(highpass(db(s1_vvm)))
    m = match_template(hpC, hpS)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    dx = gxB + S1_RES * j - ax0
    dy = (gyB - S1_RES * i) - ay1
    z = float((m[i, j] - m.mean()) / m.std())
    return float(dx), float(dy), float(m[i, j]), z


SH = np.arange(-FINE_R, FINE_R + 1e-6, FINE_STEP)   # fine shifts (m)


def sub_at(base, fx, fy):
    r0 = MB - int(round(fy / BASE_RES))
    c0 = MB + int(round(fx / BASE_RES))
    return base[r0:r0 + NS * 8, c0:c0 + NS * 8]


def fine_grid(base, s1_vvm_chip, s1ok):
    ref = highpass(db(np.where(s1ok, s1_vvm_chip, np.nan)))
    R = np.full((len(SH), len(SH)), -1.0)
    for a, fy in enumerate(SH):
        for b, fx in enumerate(SH):
            blk = block_mean(sub_at(base, fx, fy), 8)
            R[a, b] = corr(highpass(db(blk)), ref)
    return R


def process_item(r):
    t0 = time.time()
    cid = r["capella_id"]
    acc = Bytes()
    info = {"capella_id": cid, "site_id": int(r["site_id"]), "status": "ok"}
    href = r["capella_href"]
    cm = capella_meta(href)
    epsg = utm_epsg(float(r["lon"]), float(r["lat"]))
    inc = cm["inc"] if np.isfinite(cm["inc"]) else float(r["incidence_deg"])
    gr = cm["gr_res"] if np.isfinite(cm["gr_res"]) else float(r["res_range_m"]) / np.sin(np.radians(inc))
    az = cm["az_res"] if np.isfinite(cm["az_res"]) else float(r["res_azimuth_m"])
    info.update(epsg=epsg, cap_native_px=cm["native_px"], cap_src_px=cm["src_px"], cap_ov_level=cm["ov_level"],
                cap_scale=cm["scale_factor"], cap_radiometry=cm["radiometry"], cap_gr_res=gr, cap_az_res=az, cap_inc=inc)
    pos = chip_positions(href, cm, epsg, acc)
    info["n_positions"] = len(pos)
    if not pos:
        info["status"] = "no_chip_positions"
        return info, [], None
    rng = np.random.default_rng(0)
    if len(pos) > MAX_CHIPS:
        # amendment 2: keep reads compact. Take the WIN_CHIPS x WIN_CHIPS lattice window holding the most valid
        # positions (large Stripmap footprints otherwise cost up to 0.75 GB per item), then sample within it.
        P = np.array(pos)
        w = WIN_CHIPS * CHIP_M
        cnt = [int(((P[:, 0] >= x) & (P[:, 0] < x + w) & (P[:, 1] <= y) & (P[:, 1] > y - w)).sum()) for x, y in pos]
        x, y = pos[int(np.argmax(cnt))]
        pos = [q for q in pos if x <= q[0] < x + w and y - w < q[1] <= y]
        info["window_chips"] = len(pos)
        if len(pos) > MAX_CHIPS:
            pos = [pos[i] for i in sorted(rng.choice(len(pos), MAX_CHIPS, replace=False))]
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    ax0, ay1 = min(xs), max(ys)
    nc = int(round((max(xs) + CHIP_M - ax0) / S1_RES))
    nr = int(round((ay1 - (min(ys) - CHIP_M)) / S1_RES))
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    m = COARSE_BIG + 200
    poly = shapely.Polygon([inv.transform(x, y) for x, y in
                            [(ax0 - m, ay1 + m), (ax0 + nc * 10 + m, ay1 + m), (ax0 + nc * 10 + m, ay1 - nr * 10 - m),
                             (ax0 - m, ay1 - nr * 10 - m)]])
    bbox_ll = list(poly.bounds)
    tcap = pd.Timestamp(r["date"])
    capt = pd.to_datetime(cid.split("_")[5], format="%Y%m%d%H%M%S")
    sel = s1_select(epsg, ax0, ay1, nr, nc, shapely.box(*inv.transform(ax0, ay1 - nr * 10), *inv.transform(ax0 + nc * 10, ay1)),
                    capt, acc)
    if sel is None:
        info["status"] = "no_s1"
        return info, [], None
    s1i, S = sel
    info.update(s1_single_ids=";".join(s1i["single_ids"]), s1_gap_d=round(s1i["single_dt"], 2), s1_rel_orbit=s1i["rel"],
                s1_orbit_state=s1i["state"], s1_n_multi=s1i["n_multi"], s1_multi_dts=s1i["multi_dts"],
                s1_single_cov=round(s1i["single_cov"], 3), s1_platform=s1i["single_key"][0])
    if s1i["n_multi"] < 2:
        info["status"] = "multi_lt2"
        return info, [], None
    z, slope = read_dem_slope(epsg, ax0, ay1, nr, nc, bbox_ll, acc)
    wc = read_worldcover(epsg, ax0, ay1, nr, nc, bbox_ll, acc)
    print(f"{cid}: {len(pos)} chips, s1 single dt {s1i['single_dt']:.1f} d, multi {s1i['n_multi']}, "
          f"{time.time() - t0:.0f}s", flush=True)
    cdx, cdy, ncc, zc = coarse_align(href, cm, epsg, ax0, ay1, nr, nc, S["vvm"], acc)
    info.update(coarse_dx=cdx, coarse_dy=cdy, coarse_ncc=round(ncc, 4), coarse_z=round(zc, 2))
    print(f"  coarse dx={cdx:.0f} dy={cdy:.0f} ncc={ncc:.3f} z={zc:.1f} {time.time() - t0:.0f}s", flush=True)
    if zc < COARSE_Z:
        info["status"] = "coarse_fail"
        info["bytes"] = acc.n
        return info, [], None

    # per-chip reads and fine grids
    chips = []
    for k, (x0, y1) in enumerate(pos):
        r0 = int(round((ay1 - y1) / S1_RES))
        c0 = int(round((x0 - ax0) / S1_RES))
        sl = (slice(r0, r0 + NS), slice(c0, c0 + NS))
        s1c = {kk: S[kk][sl] for kk in ("vv1", "vh1", "vvm", "vhm")}
        s1ok = np.all([np.isfinite(v) for v in s1c.values()], axis=0)
        base = capella_read(href, cm, epsg, x0 - MB * BASE_RES + cdx, y1 + MB * BASE_RES + cdy, BASE_RES, NB, acc)
        G = fine_grid(base, s1c["vvm"], s1ok) if np.isfinite(base).mean() > 0.5 else np.full((len(SH), len(SH)), -1.0)
        chips.append({"k": k, "x0": x0, "y1": y1, "sl": sl, "base": base, "G": G, "s1ok": s1ok, "s1": s1c})
    best = [np.unravel_index(int(np.argmax(c["G"])), c["G"].shape) for c in chips]
    good = [(SH[b[1]], SH[b[0]]) for c, b in zip(chips, best) if c["G"][b] >= ITEM_R_MIN]
    if len(good) >= 3:
        ifx = float(np.round(np.median([g[0] for g in good]) / FINE_STEP) * FINE_STEP)
        ify = float(np.round(np.median([g[1] for g in good]) / FINE_STEP) * FINE_STEP)
    else:
        ifx = ify = 0.0
    info.update(fine_item_dx=ifx, fine_item_dy=ify, n_chips_r15=len(good))
    ia, ib = int(np.argmin(np.abs(SH - ify))), int(np.argmin(np.abs(SH - ifx)))
    print(f"  fine item shift ({ifx}, {ify}) from {len(good)} chips {time.time() - t0:.0f}s", flush=True)

    # degradation per item (sigma from the PSF budget)
    nat_geo = float(np.sqrt(gr * az))
    sig = {R: solve_sigma(R, nat_geo, cm["src_px"]) for R in RES_LIST}
    for R in RES_LIST:
        info[f"sigma_{R:g}"] = round(sig[R], 3)
        info[f"fwhm_fine_{R:g}"] = round(_fwhm_1d(min(gr, az), cm["src_px"], sig[R]), 2)
        info[f"fwhm_coarse_{R:g}"] = round(_fwhm_1d(max(gr, az), cm["src_px"], sig[R]), 2)
        info[f"looks_{R:g}"] = round(nominal_looks(sig[R], gr * az, cm["src_px"]), 1)

    rows, S1o, V10, TG, TV, WC, ST = [], [], [], [], [], [], []
    acf = {R: [] for R in RES_LIST}
    for c in chips:
        G = c["G"]
        b = np.unravel_index(int(np.argmax(G)), G.shape)
        r_best, r_item, r_zero = float(G[b]), float(G[ia, ib]), float(G[len(SH) // 2, len(SH) // 2])
        if r_best >= r_item + CHIP_GAIN:
            fx, fy, rf = float(SH[b[1]]), float(SH[b[0]]), r_best
        else:
            fx, fy, rf = ifx, ify, r_item
        tg, tv = [], []
        for R in RES_LIST:
            f, w = degrade(c["base"], sig[R])
            fs, ws = sub_at(f, fx, fy), sub_at(w, fx, fy)
            if len(acf[R]) < 4 and np.isfinite(fs).mean() > 0.95:
                acf[R].append(acf_fwhm(fs, BASE_RES))
            t = block_mean(fs, 2)
            okw = (ws.reshape(NT, 2, NT, 2).min(axis=(1, 3)) >= 0.9) if sig[R] > 0 else \
                np.isfinite(fs).reshape(NT, 2, NT, 2).all(axis=(1, 3))
            tg.append(np.where(okw & np.isfinite(t) & (t > 0), db(t), np.nan))
            tv.append(okw & np.isfinite(t) & (t > 0))
        tvalid = np.all(tv, axis=0)
        t10 = tvalid.reshape(NS, SC, NS, SC).all(axis=(1, 3))
        wcc = wc[c["sl"]]
        slc = slope[c["sl"]] if slope is not None else np.full((NS, NS), np.nan)
        land = (wcc != 0) & (wcc != 80)
        flat = np.isfinite(slc) & (slc <= SLOPE_MAX)
        valid10 = c["s1ok"] & land & flat & t10
        vf = float(valid10.mean())
        status = "ok"
        if rf < CHIP_R_MIN:
            status = "align_reject"
        elif vf < MIN_VALID:
            status = "low_valid"
        cls, cnt = np.unique(wcc, return_counts=True)
        shares = {WC_NAMES.get(int(a), "nodata"): float(n_) / wcc.size for a, n_ in zip(cls, cnt)}
        dom = max(shares, key=shares.get)
        x0, y1 = c["x0"], c["y1"]
        lon, lat = inv.transform(x0 + CHIP_M / 2, y1 - CHIP_M / 2)
        row = {"chip_id": f"{r['site_id']}_{c['k']:02d}", "capella_id": cid, "site_id": int(r["site_id"]),
               "country": r["country"], "lon": round(lon, 5), "lat": round(lat, 5), "epsg": epsg, "x0": x0, "y1": y1,
               "cap_date": str(tcap.date()), "mode": r["mode"], "pol": r["polarization"], "cap_inc": round(inc, 1),
               "s1_gap_d": info["s1_gap_d"], "s1_n_multi": info["s1_n_multi"], "s1_rel_orbit": info["s1_rel_orbit"],
               "s1_orbit_state": info["s1_orbit_state"], "align_dx": cdx + fx, "align_dy": cdy + fy,
               "fine_dx": fx, "fine_dy": fy, "r_best": round(r_best, 4), "r_item": round(r_item, 4),
               "r_noshift_fine": round(r_zero, 4), "r_final": round(rf, 4), "valid_frac": round(vf, 3),
               "slope_mean": round(float(np.nanmean(slc)), 2) if np.isfinite(slc).any() else np.nan,
               "slope_gt10_frac": round(float((slc > SLOPE_MAX).mean()), 3), "water_frac": round(shares.get("water", 0), 3),
               "built_frac": round(shares.get("built", 0), 3), "crop_frac": round(shares.get("crop", 0), 3),
               "tree_frac": round(shares.get("tree", 0), 3), "grass_frac": round(shares.get("grass", 0), 3),
               "bare_frac": round(shares.get("bare", 0), 3), "cover": dom, "status": status}
        rows.append(row)
        if status not in ("ok", "align_reject"):   # align rejects are stored for a sensitivity check only
            continue
        s1 = np.stack([db(c["s1"]["vv1"]), db(c["s1"]["vh1"]), db(c["s1"]["vvm"]), db(c["s1"]["vhm"]), slc]).astype("float16")
        S1o.append(s1)
        V10.append(valid10)
        TG.append(np.stack(tg).astype("float16"))
        TV.append(tvalid)
        WC.append(wcc)
        ST.append(status)
    for R in RES_LIST:
        v = [a for a in acf[R] if np.isfinite(a)]
        info[f"acf_fwhm_{R:g}"] = round(float(np.median(v)), 2) if v else np.nan
    info["bytes"] = acc.n
    info["seconds"] = round(time.time() - t0)
    info["n_ok"] = int(sum(1 for s_ in ST if s_ == "ok"))
    arrays = None
    if S1o:
        arrays = {"chip_id": np.array([rw["chip_id"] for rw in rows if rw["status"] in ("ok", "align_reject")]),
                  "status": np.array(ST),
                  "s1": np.stack(S1o), "valid10": np.stack(V10), "tgt": np.stack(TG), "tvalid": np.stack(TV),
                  "wc": np.stack(WC), "res_list": np.array(RES_LIST)}
    print(f"  {cid}: {info['n_ok']} of {len(rows)} chips ok, {acc.n / 1e6:.0f} MB read, {time.time() - t0:.0f}s", flush=True)
    return info, rows, arrays


def run_one(r):
    cid = r["capella_id"]
    try:
        info, rows, arrays = process_item(r)
    except Exception as ex:
        traceback.print_exc()
        info, rows, arrays = {"capella_id": cid, "site_id": int(r["site_id"]), "status": f"error: {str(ex)[:200]}"}, [], None
    if arrays is not None:
        np.savez(CHIPS / f"{cid}.npz", **arrays)
    (CHIPS / f"{cid}.json").write_text(json.dumps({"info": info, "rows": rows}, default=jdefault))
    return cid, info.get("status"), info.get("n_ok", 0), info.get("bytes", 0)


def merge_chips():
    infos, rows = [], []
    for f in sorted(CHIPS.glob("*.json")):
        j = json.loads(f.read_text())
        infos.append(j["info"])
        rows += j["rows"]
    it = pd.DataFrame(infos)
    it.to_csv(HERE / "items_built.csv", index=False)
    ch = pd.DataFrame(rows)
    if len(ch):
        keep = ["capella_id", "cap_native_px", "cap_src_px", "cap_gr_res", "cap_az_res", "cap_radiometry", "coarse_z"] + \
               [c for c in it.columns if c.startswith(("sigma_", "fwhm_", "looks_", "acf_"))]
        ch = ch.merge(it[[c for c in keep if c in it.columns]], on="capella_id", how="left")
        ch.to_csv(HERE / "chips_all.csv", index=False)
        old = HERE / "chips.csv"
        ok = ch[ch.status == "ok"].copy()
        if old.exists():   # keep split assignments made earlier
            prev = pd.read_csv(old)
            if "split" in prev.columns:
                ok = ok.merge(prev[["chip_id", "split", "block"]], on="chip_id", how="left")
        ok.to_csv(old, index=False)
    print("items by status:", it.status.value_counts().to_dict(), flush=True)
    if len(ch):
        print("chips by status:", ch.status.value_counts().to_dict(), flush=True)
    print(f"bytes read: {it.get('bytes', pd.Series(dtype=float)).fillna(0).sum() / 1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    CHIPS.mkdir(parents=True, exist_ok=True)
    if a.merge_only:
        merge_chips()
        return
    items = pd.read_csv(HERE / "items.csv")
    if a.ids:
        items = items[items.capella_id.isin(a.ids)]
    todo = [r for r in items.to_dict("records") if not (CHIPS / f"{r['capella_id']}.json").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} items to build ({len(items)} selected)", flush=True)
    t0 = time.time()
    tot = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, r) for r in todo]
        for i, f in enumerate(as_completed(futs)):
            try:
                cid, st, n, nb = f.result()
                tot += nb or 0
                print(f"[{i + 1}/{len(todo)}] {cid} {st} chips={n} {(nb or 0) / 1e6:.0f} MB; total {tot / 1e9:.2f} GB, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception:
                traceback.print_exc()
    merge_chips()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
