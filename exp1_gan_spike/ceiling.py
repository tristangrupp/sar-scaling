"""Metric ceiling for the S1 -> X-band super-resolution spike: score REAL X-band vs REAL X-band.

Two collects of the same place (different dates / satellites / geometries) are put on one 2.5 m
UTM grid (prep.tile_grid at the overlap centroid), converted to dB, B is phase-correlated onto A,
and B is scored as a "prediction" of A with evaluate.scores after train.norm on mutually valid
pixels. A degraded B (10 m block mean -> bicubic x4, as evaluate's bicubic baseline) shows what
"no fine detail" scores. A vs A shifted by 1 and 2 px shows how fast detail_r decays with pure
misregistration.

  python ceiling.py --stage pairs    # candidates + selection -> ceiling/pairs_selected.csv
  python ceiling.py --stage score    # read, register, score -> ceiling/pairs.csv, summary.csv, png/

Run in background with stdout to a file (this env's GDAL hangs foreground pipes on Windows).
"""
from __future__ import annotations

import prep  # noqa: I001  (sets GDAL env vars before rasterio loads)

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import torch.nn.functional as F
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import transform as shp_transform
from skimage.registration import phase_cross_correlation

from evaluate import scores
from train import norm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ceiling"
PROJ = prep.PROJ
NH, HR, TILE_M = prep.NH, prep.HR, prep.TILE_M
CAPELLA_GEO = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
               "capella-open-data-by-product-type/capella-open-data-geo/collection.json")
MAX_SHIFT = 8          # px at 2.5 m


# ------------------------------------------------------------------ items
def look_az_from_orbit(orbit, side):
    """Approximate look azimuth (deg, direction the radar looks) for a near-polar orbit."""
    if orbit not in ("ascending", "descending") or side not in ("left", "right"):
        return np.nan
    heading = 348.0 if orbit == "ascending" else 192.0
    return (heading + (90 if side == "right" else -90)) % 360


def load_items() -> gpd.GeoDataFrame:
    um = gpd.read_file(PROJ / "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson")
    ic = gpd.read_file(PROJ / "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson")
    ca = gpd.read_file(PROJ / "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson")
    um["provider"], ic["provider"], ca["provider"] = "umbra", "iceye", "capella"
    ca = ca.rename(columns={"capella_id": "xband_id"})
    for d in (ic, ca):
        if "view:azimuth" not in d:
            d["view:azimuth"] = np.nan
    ca["sar:observation_direction"] = None
    ic["href"] = ic["grd_href"]
    um["href"], ca["href"] = None, None
    cols = ["xband_id", "provider", "platform", "datetime", "sar:polarizations", "sar:instrument_mode",
            "view:incidence_angle", "sat:orbit_state", "sar:observation_direction", "view:azimuth",
            "href", "geometry"]
    df = gpd.GeoDataFrame(pd.concat([um[cols], ic[cols], ca[cols]], ignore_index=True), crs="EPSG:4326")
    df = df[df["sar:instrument_mode"].str.lower().isin(["spotlight", "sliding_spotlight"])]
    df = df[df["sar:polarizations"].isin(["VV", "HH"])].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="mixed")
    # view:azimuth is target->sensor (checked against orbit/side for Umbra); look = +180
    df["look_az"] = np.where(df["view:azimuth"].notna(), (df["view:azimuth"] + 180) % 360,
                             [look_az_from_orbit(o, s) for o, s in
                              zip(df["sat:orbit_state"], df["sar:observation_direction"])])
    c = df.geometry.centroid
    df["lon"], df["lat"] = c.x, c.y
    df["test_region"] = [prep.test_region(p) for p in c]
    df["cell"] = df.lon.round(1).astype(str) + "_" + df.lat.round(1).astype(str)
    return df


def candidates(df: gpd.GeoDataFrame, max_per_cell=25, seed=0) -> pd.DataFrame:
    """All same-pol pairs whose footprint intersection covers >= 90 % of a 3.2 km tile at its centroid."""
    rng = np.random.default_rng(seed)
    # cap very busy cells (repeat-tasked Umbra sites) so pair counts stay tractable
    keep = []
    for (cell, prov), g in df.groupby(["cell", "provider"]):
        n = max_per_cell * (3 if g.test_region.notna().any() else 1)
        keep.append(g if len(g) <= n else g.iloc[rng.choice(len(g), n, replace=False)])
    d = pd.concat(keep).reset_index(drop=True)
    d = gpd.GeoDataFrame(d, crs="EPSG:4326")
    j = gpd.sjoin(d[["geometry"]], d[["geometry"]], predicate="intersects")
    j = j[j.index < j["index_right"]]
    rows = []
    for i, k in zip(j.index, j["index_right"]):
        a, b = d.loc[i], d.loc[k]
        if a["sar:polarizations"] != b["sar:polarizations"]:
            continue
        if abs((a.datetime - b.datetime).total_seconds()) < 60:
            continue
        inter = a.geometry.intersection(b.geometry)
        if inter.is_empty or inter.area < 5e-4 ** 2:
            continue
        cen = inter.centroid
        epsg, x0, y1 = prep.tile_grid(cen.x, cen.y)
        tf = Transformer.from_crs(4326, epsg, always_xy=True).transform
        tile = box(x0, y1 - TILE_M, x0 + TILE_M, y1)
        cov = shp_transform(tf, inter).intersection(tile).area / tile.area
        if cov < 0.9:
            continue
        rows.append({"id_a": a.xband_id, "id_b": b.xband_id, "cov": cov, "lon": cen.x, "lat": cen.y,
                     "test_region": prep.test_region(cen), "cell": a.cell})
    return pd.DataFrame(rows)


def capella_item_hrefs() -> dict[str, str]:
    coll = requests.get(CAPELLA_GEO, timeout=60).json()
    out = {}
    for lk in coll["links"]:
        if lk["rel"] == "item":
            h = urljoin(CAPELLA_GEO, lk["href"])
            out[Path(h).stem] = h
    return out


def capella_meta(ids) -> dict[str, dict]:
    hrefs = capella_item_hrefs()

    cache = OUT / "capella_meta.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    sess = requests.Session()

    def one(i):
        if i in have:
            return i, have[i]
        for attempt in range(6):
            try:
                it = sess.get(hrefs[i], timeout=60).json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        else:
            return i, None
        p = it["properties"]
        asset = next(a["href"] for k, a in it["assets"].items() if k in ("HH", "VV", "HV", "VH"))
        return i, {"href": asset, "view:azimuth": p.get("view:azimuth"),
                   "sar:observation_direction": p.get("sar:observation_direction")}

    with ThreadPoolExecutor(6) as ex:
        res = {k: v for k, v in ex.map(one, [i for i in ids if i in hrefs]) if v}
    cache.write_text(json.dumps(res))
    return res


def geom_class(a, b):
    d_inc = abs(a["view:incidence_angle"] - b["view:incidence_angle"])
    d_az = abs((a.look_az - b.look_az + 180) % 360 - 180) if np.isfinite(a.look_az) and np.isfinite(b.look_az) else np.nan
    similar = bool(np.isfinite(d_az) and d_az <= 45 and d_inc <= 5)
    return d_inc, d_az, "similar" if similar else "different"


def stage_pairs(seed=0):
    df = load_items()
    print(f"items: {len(df)}  {df.groupby(['provider', 'sar:polarizations']).size().to_dict()}", flush=True)
    # Umbra STAC items from 2025 on carry empty asset hrefs (no resolvable GEC); leave them out
    df = df[~((df.provider == "umbra") & (df.datetime >= pd.Timestamp("2025-01-01", tz="UTC")))]
    cand = candidates(df, seed=seed)
    print(f"candidate pairs: {len(cand)}", flush=True)
    idx = df.set_index("xband_id")
    # fill Capella look geometry from item JSON before classifying
    cap_ids = set(cand.id_a[cand.id_a.isin(idx.index[idx.provider == "capella"])]) | \
        set(cand.id_b[cand.id_b.isin(idx.index[idx.provider == "capella"])])
    if cap_ids:
        meta = capella_meta(cap_ids)
        for i, m in meta.items():
            idx.loc[i, "href"] = m["href"]
            if m["view:azimuth"] is not None:
                idx.loc[i, "view:azimuth"] = m["view:azimuth"]
                idx.loc[i, "look_az"] = (m["view:azimuth"] + 180) % 360
            idx.loc[i, "sar:observation_direction"] = m["sar:observation_direction"]
    recs = []
    for r in cand.itertuples():
        a, b = idx.loc[r.id_a], idx.loc[r.id_b]
        # A = the earlier collect, as the "truth"
        if b.datetime < a.datetime:
            a, b, ia, ib = b, a, r.id_b, r.id_a
        else:
            ia, ib = r.id_a, r.id_b
        d_inc, d_az, gc = geom_class(a, b)
        combo = "-".join(sorted([a.provider, b.provider]))
        recs.append({"id_a": ia, "id_b": ib, "prov_a": a.provider, "prov_b": b.provider, "combo": combo,
                     "pol": a["sar:polarizations"], "plat_a": a.platform, "plat_b": b.platform,
                     "date_a": a.datetime.isoformat(), "date_b": b.datetime.isoformat(),
                     "dt_days": round((b.datetime - a.datetime).total_seconds() / 86400, 2),
                     "inc_a": a["view:incidence_angle"], "inc_b": b["view:incidence_angle"],
                     "orbit_a": a["sat:orbit_state"], "orbit_b": b["sat:orbit_state"],
                     "side_a": a["sar:observation_direction"], "side_b": b["sar:observation_direction"],
                     "look_az_a": a.look_az, "look_az_b": b.look_az, "d_inc": d_inc, "d_look_az": d_az,
                     "geometry": gc, "cov": r.cov, "lon": r.lon, "lat": r.lat,
                     "region": r.test_region if isinstance(r.test_region, str) else "train",
                     "site": r.test_region if isinstance(r.test_region, str) else r.cell,
                     "href_a": a.href, "href_b": b.href})
    c = pd.DataFrame(recs)
    c.to_csv(OUT / "candidates.csv", index=False)
    print(c.groupby(["region", "combo", "pol", "geometry"]).size().to_string(), flush=True)
    # 2025+ Umbra STAC items have empty asset hrefs; drop unresolvable images and reselect
    bad, known = set(), {}
    for _ in range(8):
        sel = select(c[~c.id_a.isin(bad) & ~c.id_b.isin(bad)].copy(), seed)
        known.update(resolve_umbra(sel, known))
        new_bad = {i for s in "ab" for i in sel.loc[sel[f"prov_{s}"] == "umbra", f"id_{s}"]
                   if not isinstance(known.get(i), str)}
        print(f"umbra hrefs: {len(new_bad)} unresolvable in this selection", flush=True)
        if not new_bad:
            break
        bad |= new_bad
    for s in "ab":
        u = sel[f"prov_{s}"] == "umbra"
        sel.loc[u, f"href_{s}"] = sel.loc[u, f"id_{s}"].map(known)
    sel = sel[sel.href_a.notna() & sel.href_b.notna()].reset_index(drop=True)
    print(f"selected {len(sel)}", flush=True)
    print(sel.groupby(["region", "combo", "geometry"]).size().to_string(), flush=True)
    sel.to_csv(OUT / "pairs_selected.csv", index=False)


# pairs wanted per (combo, pol, geometry); VV first since the spike was VV (Umbra + ICEYE)
QUOTA = {("umbra-umbra", "VV"): 8, ("iceye-umbra", "VV"): 5, ("iceye-iceye", "VV"): 4,
         ("capella-capella", "VV"): 2, ("capella-capella", "HH"): 2, ("capella-umbra", "HH"): 2,
         ("capella-umbra", "VV"): 1}


def resolve_umbra(sel, known):
    sc = pd.read_csv(ROOT / "selection.csv")
    out = {k: v for k, v in zip(sc.xband_id, sc.href) if isinstance(v, str)}
    ids = set(sel.loc[sel.prov_a == "umbra", "id_a"]) | set(sel.loc[sel.prov_b == "umbra", "id_b"])
    miss = {i for i in ids if i not in out and i not in known}
    print(f"resolving {len(miss)} Umbra GEC hrefs", flush=True)
    if miss:
        got = prep.umbra_gec_hrefs(miss)
        out.update({i: got.get(i) for i in miss})
    return out


def select(c: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fill QUOTA per geometry class, test regions first; each image used at most twice and at
    most 3 pairs per site (5 per test region), so busy Umbra sites do not dominate."""
    c = c.sample(frac=1, random_state=seed)
    c["pri"] = (c.region == "train").astype(int)
    c = c.sort_values("pri", kind="stable")
    used, per_site, chosen = {}, {}, []
    for (combo, pol), n in QUOTA.items():
        for geom in ("similar", "different"):
            q = c[(c.combo == combo) & (c.pol == pol) & (c.geometry == geom)]
            k = 0
            for i, r in q.iterrows():
                if k >= n:
                    break
                cap = 5 if r.region != "train" else 3
                if used.get(r.id_a, 0) >= 2 or used.get(r.id_b, 0) >= 2 or per_site.get(r.site, 0) >= cap:
                    continue
                chosen.append(i)
                k += 1
                used[r.id_a] = used.get(r.id_a, 0) + 1
                used[r.id_b] = used.get(r.id_b, 0) + 1
                per_site[r.site] = per_site.get(r.site, 0) + 1
    return c.loc[chosen].drop(columns="pri").reset_index(drop=True)


# ------------------------------------------------------------------ reading
def read_capella(href, epsg, x0, y1):
    with rasterio.open(href) as s0:
        native = abs(s0.res[0])
        ovs = s0.overviews(1)
    lvl = None
    for i, f in enumerate(ovs):
        if native * f <= HR / 2:
            lvl = i
    with rasterio.open(href, overview_level=lvl) if lvl is not None else rasterio.open(href) as s:
        return prep.warp(s, epsg, x0, y1, HR, NH)


def read_any(href, provider, epsg, x0, y1):
    if provider == "capella":
        return read_capella(href, epsg, x0, y1)
    return prep.read_xband(href, provider, epsg, x0, y1)


def to_db_masked(amp):
    v = amp > 0
    return prep.to_db(np.where(v, amp, np.nan), power=False), v


def degrade(p):
    """10 m block mean then bicubic x4: the evaluate.py bicubic baseline applied to B."""
    n = p.shape[0] // 4
    lo = p.reshape(n, 4, n, 4).mean(axis=(1, 3))
    t = torch.from_numpy(lo.astype("float32"))[None, None]
    return F.interpolate(t, scale_factor=4, mode="bicubic", align_corners=False)[0, 0].numpy()


def score_pair(A, B, m):
    """A, B in dB with NaN outside; m = mutual mask. Normalize on mutual pixels, score B vs A."""
    an = norm(np.where(m, A, np.nan))
    bn = norm(np.where(m, B, np.nan))
    t = np.nan_to_num(an)
    p = np.where(m, bn, 0.0)
    return scores(p, t, m), an, bn


def register(A, B, va, vb, max_shift):
    m = va & vb
    a = np.where(m, norm(np.where(m, A, np.nan)), 0)
    b = np.where(m, norm(np.where(m, B, np.nan)), 0)
    shift, err, _ = phase_cross_correlation(a, b, upsample_factor=4, normalization=None)
    return shift


def apply_shift(B, vb, shift):
    fill = np.nanmedian(B)
    bs = ndimage.shift(np.nan_to_num(B, nan=fill), shift, order=1, mode="nearest")
    vs = ndimage.shift(vb.astype("float32"), shift, order=0, mode="constant") > 0.5
    return np.where(vs, bs, np.nan), vs


def process(r):
    f = OUT / "tiles" / f"{r.id_a[-12:]}__{r.id_b[-12:]}.npz"
    epsg, x0, y1 = prep.tile_grid(r.lon, r.lat)
    if f.exists():
        z = np.load(f)
        A, B = z["A"].astype("float32"), z["B"].astype("float32")
    else:
        A, _ = to_db_masked(read_any(r.href_a, r.prov_a, epsg, x0, y1))
        B, _ = to_db_masked(read_any(r.href_b, r.prov_b, epsg, x0, y1))
        np.savez_compressed(f, A=A.astype("float16"), B=B.astype("float16"), epsg=epsg, x0=x0, y1=y1)
    va, vb = np.isfinite(A), np.isfinite(B)
    out = {"epsg": epsg, "valid_a": float(va.mean()), "valid_b": float(vb.mean()),
           "valid_mutual_raw": float((va & vb).mean())}
    if (va & vb).mean() < 0.3:
        return {**out, "status": "low_overlap"}, None

    n = NH // 4

    def block10(X):
        return np.nanmean(X.reshape(n, 4, n, 4), axis=(1, 3))

    def corr10(X, Y):
        a, b = block10(X), block10(Y)
        k = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 100 else np.nan

    # (1) spike-like: one phase correlation at 2.5 m, applied only if <= MAX_SHIFT px
    raw = register(A, B, va, vb, None)
    lim = raw if np.abs(raw).max() <= MAX_SHIFT else np.zeros(2)
    # (2) best effort: coarse phase correlation on 10 m block means (any size), then a 2.5 m
    #     refinement (<= MAX_SHIFT px) on the coarsely shifted image
    A10, B10 = block10(A), block10(B)
    coarse = register(A10, B10, np.isfinite(A10), np.isfinite(B10), None) * 4
    Bc, vbc = apply_shift(B, vb, coarse)
    fine = register(A, Bc, va, vbc, None)
    best = coarse + (fine if np.abs(fine).max() <= MAX_SHIFT else 0)
    out.update({"raw_dy_px": best[0], "raw_dx_px": best[1], "raw_off_m": float(np.hypot(*best) * HR),
                "pc25_off_m": float(np.hypot(*raw) * HR), "lim_shift_applied": bool(np.any(lim)),
                "corr10_noshift": corr10(A, B)})

    s0, _, _ = score_pair(A, B, va & vb)
    out.update({f"noshift_{k}": v for k, v in s0.items()})
    Bl, vbl = apply_shift(B, vb, lim)
    sl, _, _ = score_pair(A, Bl, va & vbl)
    out.update({f"lim_{k}": v for k, v in sl.items()})
    Bs, vbs = apply_shift(B, vb, best)
    m = va & vbs
    out["valid_mutual"] = float(m.mean())
    out["corr10_best"] = corr10(A, Bs)
    if m.mean() < 0.3:
        return {**out, "status": "low_overlap_after_shift"}, None
    s1, an, bn = score_pair(A, Bs, m)
    out.update(s1)
    # residual offset after the shift
    res = register(A, Bs, va, vbs, None)
    out["resid_off_m"] = float(np.hypot(*res) * HR)
    # degraded B
    bdeg = degrade(np.where(m, bn, 0.0))
    sd = scores(np.where(m, bdeg, 0.0), np.nan_to_num(an), m)
    out.update({f"deg_{k}": v for k, v in sd.items()})
    # A vs A shifted by 1 and 2 px (pure misregistration, same speckle)
    for k in (1, 2):
        As, vas = apply_shift(A, va, np.array([0.0, float(k)]))
        mm = va & vas
        sk, _, _ = score_pair(A, As, mm)
        out.update({f"self{k}px_{kk}": v for kk, v in sk.items()})
    out["status"] = "ok"
    return out, (an, bn, bdeg, m)


def panel(i, r, an, bn, bdeg, m, sc):
    x = np.where(m, an, np.nan)
    best, rc = -1, (0, 0)
    for rr in range(0, NH - 256, 128):
        for cc in range(0, NH - 256, 128):
            w = x[rr:rr + 256, cc:cc + 256]
            if np.isfinite(w).mean() > 0.99 and np.nanstd(w) > best:
                best, rc = np.nanstd(w), (rr, cc)
    rr, cc = rc
    sl = np.s_[rr:rr + 256, cc:cc + 256]
    lo, hi = np.nanpercentile(x[sl], [2, 98])
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.6), dpi=110)
    titles = [f"A: {r.prov_a} {r.date_a[:10]} inc {r.inc_a:.0f}",
              f"B: {r.prov_b} {r.date_b[:10]} inc {r.inc_b:.0f}",
              "B degraded (10 m -> bicubic)"]
    for a_, img, t in zip(ax, [x, np.where(m, bn, np.nan), np.where(m, bdeg, np.nan)], titles):
        a_.imshow(img[sl], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        a_.set_title(t, fontsize=8)
        a_.axis("off")
    fig.suptitle(f"pair {i:02d} {r.site} ({r.geometry} geometry)  detail_r={sc['detail_r']:.2f} "
                 f"hf={sc['hf_ratio']:.2f} ssim={sc['ssim']:.2f}  degraded detail_r={sc['deg_detail_r']:.2f}",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "png" / f"pair_{i:02d}_{r.site}.png")
    plt.close(fig)


def stage_score(workers):
    sel = pd.read_csv(OUT / "pairs_selected.csv")
    need = sel.prov_a.eq("umbra") & sel.href_a.isna() | sel.prov_b.eq("umbra") & sel.href_b.isna()
    if need.any():
        sc = pd.read_csv(ROOT / "selection.csv")
        known = dict(zip(sc.xband_id, sc.href))
        ids = set(sel.loc[sel.prov_a == "umbra", "id_a"]) | set(sel.loc[sel.prov_b == "umbra", "id_b"])
        miss = {i for i in ids if not isinstance(known.get(i), str)}
        print(f"resolving {len(miss)} Umbra GEC hrefs", flush=True)
        if miss:
            known.update(prep.umbra_gec_hrefs(miss))
        for s in "ab":
            u = sel[f"prov_{s}"] == "umbra"
            sel.loc[u, f"href_{s}"] = sel.loc[u, f"id_{s}"].map(known)
        sel.to_csv(OUT / "pairs_selected.csv", index=False)
    (OUT / "tiles").mkdir(exist_ok=True)
    (OUT / "png").mkdir(exist_ok=True)
    rows, extra, t0 = {}, {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(process, r): r for r in sel.itertuples()}
        for n, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                res, ex_ = fu.result()
            except Exception as e:
                res, ex_ = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}, None
            rows[r.Index] = res
            extra[r.Index] = ex_
            print(f"[{n}/{len(sel)}] {time.time() - t0:.0f}s pair {r.Index} {r.combo} {r.site} "
                  f"{res.get('status')} detail_r={res.get('detail_r', np.nan):.3f} "
                  f"raw_off={res.get('raw_off_m', np.nan):.1f}m {res.get('error', '')}", flush=True)
    res = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df = pd.concat([sel.drop(columns=["href_a", "href_b"]), res], axis=1)
    df.index.name = "pair"
    df.to_csv(OUT / "pairs.csv")
    ok = df[df.status == "ok"]
    # panels: up to 8, spread across combos
    # best and median registered pair per provider combo, plus the worst overall
    reg = ok[ok.corr10_best >= 0.5].sort_values("detail_r")
    pick = set(reg.groupby("combo").tail(1).index) | set(ok.sort_values("detail_r").index[:1])
    for _, g in reg.groupby("combo"):
        pick.add(g.index[len(g) // 2])
    for i in pick:
        an, bn, bdeg, m = extra[i]
        panel(i, ok.loc[i], an, bn, bdeg, m, ok.loc[i])
    summarize(ok)


def summarize(ok):
    mets = ["detail_r", "hf_ratio", "ssim", "mae"]
    cols = (mets + [f"deg_{k}" for k in mets] + [f"noshift_{k}" for k in mets] +
            [f"lim_{k}" for k in mets] + ["corr10_noshift", "corr10_best",
            "self1px_detail_r", "self2px_detail_r", "raw_off_m", "pc25_off_m", "resid_off_m", "dt_days"])

    def agg(g):
        s = g[cols].median()
        s["n"] = len(g)
        return s

    ok = ok.assign(reg=np.where(ok.corr10_best >= 0.5, "corr10>=0.5",
                                np.where(ok.corr10_best >= 0.3, "corr10 0.3-0.5", "corr10<0.3")))
    parts = [agg(ok).to_frame("all").T, agg(ok[ok.corr10_best >= 0.3]).to_frame("corr10>=0.3").T,
             agg(ok[ok.corr10_best >= 0.5]).to_frame("corr10>=0.5").T]
    t = ok.groupby(["reg", "geometry"])[cols].apply(agg)
    t.index = [f"reg,geometry={a},{b}" for a, b in t.index]
    parts.append(t)
    t = ok[ok.corr10_best >= 0.5].groupby("combo")[cols].apply(agg)
    t.index = [f"corr10>=0.5 combo={a}" for a in t.index]
    parts.append(t)
    for key in ["geometry", "combo", "region", "pol"]:
        t = ok.groupby(key)[cols].apply(agg)
        t.index = [f"{key}={v}" for v in t.index]
        parts.append(t)
    t = ok.groupby(["combo", "geometry"])[cols].apply(agg)
    t.index = [f"combo,geometry={a},{b}" for a, b in t.index]
    parts.append(t)
    summ = pd.concat(parts).round(3)
    summ.index.name = "group (medians)"
    cols_first = ["n"] + [c for c in summ.columns if c != "n"]
    summ = summ[cols_first]
    summ.to_csv(OUT / "summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ[["n"] + mets + ["deg_detail_r", "deg_hf_ratio", "lim_detail_r", "noshift_detail_r",
                                    "self1px_detail_r", "corr10_best", "raw_off_m", "resid_off_m"]].to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pairs", "score", "summary"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.stage == "pairs":
        stage_pairs(a.seed)
    elif a.stage == "score":
        stage_score(a.workers)
    else:
        df = pd.read_csv(OUT / "pairs.csv", index_col=0)
        summarize(df[df.status == "ok"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
