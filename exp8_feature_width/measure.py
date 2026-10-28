"""Step 2-3: per scene, measure linear-feature widths on Capella and test their visibility on Sentinel-1.

Implements DESIGN.md exactly (segments, alignment, Capella width/contrast, matched-null detection rule).
Per scene writes data/<capella_id>/segments.csv, profiles.npz (per-segment profiles) and arrays.npz
(Capella 1 m dB, Sentinel-1 VV/VH single and multi-date, for the spot-check panels).
Usage: measure.py [capella_id ...]   (default: every scene in scenes_selected.csv; finished scenes skipped)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import numpy as np
import pandas as pd
import rasterio
import shapely
import shapely.ops
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_scenes import OVDIR, osm_lines  # noqa: E402
import geopandas as gpd  # noqa: E402

WIN = 4000.0
MC, MS = 300, 200            # Capella / Sentinel-1 grid margins (m) around the window
NC = int(WIN + 2 * MC)       # Capella 1 m grid size
NS = int((WIN + 2 * MS) / 10)  # Sentinel-1 10 m grid size
EDGE = 150.0                 # segments stay this far inside the window
SEG, SEG_MIN = 200.0, 100.0
D = np.arange(-180.0, 180.0001, 0.5)   # across-line axis (m), positive = left of travel
S_STEP = 1.0
THR1, THR2 = 2.0, 2.24       # one polarization; either polarization
CAP_Z = 4.0


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


# ------------------------------------------------------------------ readers
def capella_read(href, epsg, gx0, gy1, res=1.0, n=None, max_src=0.75):
    """Amplitude on a res-m grid (average from the coarsest overview finer than max_src), squared to intensity.
    Returns (intensity, bytes of the source tiles touched, source resolution)."""
    n = NC if n is None else n
    with rasterio.open(href) as s0:
        native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
        ovs = s0.overviews(1)
    lvl = None
    for i, f in enumerate(ovs):
        if native * f <= max_src:
            lvl = i
    nbytes = 0
    with (rasterio.open(href, overview_level=lvl) if lvl is not None else rasterio.open(href)) as s:
        try:
            b = transform_bounds(f"EPSG:{epsg}", s.crs, gx0, gy1 - n * res, gx0 + n * res, gy1, densify_pts=21)
            w = rasterio.windows.from_bounds(*b, transform=s.transform)
            bh, bw = s.block_shapes[0]
            r0, r1 = max(0, int(w.row_off // bh)), min((s.height - 1) // bh, int((w.row_off + w.height) // bh))
            c0, c1 = max(0, int(w.col_off // bw)), min((s.width - 1) // bw, int((w.col_off + w.width) // bw))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    v = s.get_tag_item(f"BLOCK_SIZE_{c}_{r}", "TIFF", bidx=1)
                    nbytes += int(v) if v else bh * bw * np.dtype(s.dtypes[0]).itemsize
        except Exception:
            nbytes = -1
        with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(gx0, gy1, res, res), width=n, height=n,
                       resampling=Resampling.average, nodata=0, dtype="float32") as v:
            a = v.read(1)
        res_used = abs(s.res[0]) * (111_000 if s.crs.is_geographic else 1)
    a[~np.isfinite(a) | (a <= 0)] = np.nan
    return a.astype("float32") ** 2, nbytes, res_used


def s1_read(item, pol, epsg, gx0, gy1, n=None):
    n = NS if n is None else n
    with rasterio.open(item.assets[pol].href) as s, WarpedVRT(
            s, crs=f"EPSG:{epsg}", transform=from_origin(gx0, gy1, 10.0, 10.0), width=n, height=n,
            resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
        a = v.read(1)
    a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
    return a


def s1_items(ids):
    import planetary_computer as pc
    from pystac_client import Client
    c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    its = {i.id: i for i in c.search(collections=["sentinel-1-rtc"], ids=list(ids)).items()}
    return [its[i] for i in ids if i in its]


# ------------------------------------------------------------------ alignment
def block10(C, oy, ox):
    """Mean of Capella 1 m intensity over the 10 m Sentinel-1 cells, Capella offset (oy, ox) into C."""
    sub = C[oy:oy + NS * 10, ox:ox + NS * 10]
    m = np.isfinite(sub)
    s = np.where(m, sub, 0).reshape(NS, 10, NS, 10).sum(axis=(1, 3))
    n = m.reshape(NS, 10, NS, 10).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[n < 50] = np.nan
    return out


def highpass(a_db):
    m = np.isfinite(a_db)
    num = ndimage.uniform_filter(np.where(m, a_db, 0), 5)
    den = ndimage.uniform_filter(m.astype(float), 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        hp = a_db - num / den
    hp[~m | (den < 0.5)] = np.nan
    return hp


def corr(a, b, nmin=500):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < nmin:
        return -1.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


BIG = 1500.0   # coarse alignment search radius (m)


def coarse_align(href, epsg, gxS, gyS, s1_big, gxB, gyB):
    """Locate the Sentinel-1 window inside a wide Capella read at 10 m (normalized cross-correlation).
    Returns (dx, dy, r, cap_mb): Capella content for true point p sits at Capella grid p + (dx, dy)."""
    from skimage.feature import match_template
    nB = s1_big.shape[0]
    capB, nbytes, _ = capella_read(href, epsg, gxB, gyB, res=10.0, n=nB, max_src=5.0)
    hpC = np.nan_to_num(highpass(db(capB)))
    tmpl = s1_big[int(BIG / 10) - 20:int(BIG / 10) + NS + 20, int(BIG / 10) - 20:int(BIG / 10) + NS + 20]
    hpS = np.nan_to_num(highpass(db(tmpl)))
    m = match_template(hpC, hpS)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    # template top-left (true) = (gxS, gyS); found at Capella-grid (gxB + 10 j, gyB - 10 i)
    dx = gxB + 10 * j - gxS
    dy = (gyB - 10 * i) - gyS
    zpk = float((m[i, j] - m.mean()) / m.std())
    return float(dx), float(dy), float(m[i, j]), zpk, nbytes


def align(C, s1_db):
    """Shift (dx, dy) in m such that Capella content at true p sits at Capella grid p + (dx, dy)."""
    ref = highpass(s1_db)
    off = MC - MS

    def score(dx, dy):
        return corr(highpass(db(block10(C, off - dy, off + dx))), ref)

    best = (-2, 0, 0)
    for dy in range(-10, 11, 10):
        for dx in range(-10, 11, 10):
            r = score(dx, dy)
            if r > best[0]:
                best = (r, dx, dy)
    r0, bx, by = best
    for dy in range(by - 10, by + 11, 2):
        for dx in range(bx - 10, bx + 11, 2):
            if abs(dx) > 98 or abs(dy) > 98:
                continue
            r = score(dx, dy)
            if r > best[0]:
                best = (r, dx, dy)
    return best[1], best[2], best[0], r0


# ------------------------------------------------------------------ OSM -> segments
def build_lines(src, epsg, window):
    g = osm_lines(src, epsg)
    if g is None:
        return [], []
    g = g[g.geometry.intersects(window.buffer(300))]
    lines = []   # (type, geometry, src_index)
    neighbors = []
    for t in ("tree_line", "road", "track", "ditch_canal"):
        sub = g[g["ftype"] == t]
        if sub.empty:
            continue
        merged = shapely.line_merge(shapely.MultiLineString(list(sub.geometry.values)))
        for part in getattr(merged, "geoms", [merged]):
            if part.length > 0:
                lines.append((t, part, json.dumps(sorted(set(sub[sub.geometry.intersects(part.buffer(0.5))].osm_id.astype(str).tolist()))[:20])))
    nonlu = shapely.union_all(g[g["ftype"] != "landuse"].geometry.values) if (g["ftype"] != "landuse").any() else None
    lu = g[g["ftype"] == "landuse"]
    if not lu.empty:
        b = shapely.union_all(lu.geometry.values)
        if nonlu is not None:
            b = b.difference(nonlu.buffer(30))
        b = shapely.line_merge(b) if b.geom_type in ("MultiLineString", "LineString") else b
        for part in getattr(b, "geoms", [b]):
            if part.geom_type == "LineString" and part.length > 0:
                lines.append(("field_edge", part, "[]"))
    other = g[g["ftype"] == "other_line"]
    for i, (t, geom, _) in enumerate(lines):
        neighbors.append((i, geom))
    for geom in other.geometry.values:
        neighbors.append((-1, geom))
    return lines, neighbors


def unit_tangent(line, s):
    a = line.interpolate(max(s - 2, 0))
    b = line.interpolate(min(s + 2, line.length))
    v = np.array([b.x - a.x, b.y - a.y])
    n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def make_segments(lines, neighbors, window_inner):
    tree = STRtree([g for _, g in neighbors])
    src = np.array([i for i, _ in neighbors])
    out = []
    for li, (t, line, osm_ids) in enumerate(lines):
        L = line.length
        nseg = int(L // SEG) + (1 if (L % SEG) >= SEG_MIN else 0)
        for k in range(nseg):
            s0 = k * SEG
            s1 = min(s0 + SEG, L)
            seg = shapely.ops.substring(line, s0, s1)
            if not window_inner.contains(seg):
                continue
            ss = np.arange(0, seg.length + 1e-6, 5.0)
            par = np.zeros(len(ss), bool)
            cross = np.zeros(len(ss), bool)
            for qi, s in enumerate(ss):
                p = seg.interpolate(s)
                tv = unit_tangent(seg, s)
                nv = np.array([-tv[1], tv[0]])
                # parallel: a neighbor crossing the normal ray 2-30 m out on either side, running within 30 deg
                rays = [shapely.LineString([(p.x + a * nv[0], p.y + a * nv[1]), (p.x + b * nv[0], p.y + b * nv[1])])
                        for a, b in ((2, 30), (-2, -30))]
                for j in tree.query(p.buffer(30)):
                    if src[j] == li:
                        continue
                    other = neighbors[j][1]
                    dist = other.distance(p)
                    if dist > 30:
                        continue
                    for ray in rays:
                        x = other.intersection(ray)
                        if x.is_empty:
                            continue
                        q = x if x.geom_type == "Point" else shapely.get_geometry(x, 0)
                        q = q if q.geom_type == "Point" else q.centroid
                        to = unit_tangent(other, other.project(q))
                        if np.degrees(np.arccos(min(1.0, abs(float(tv @ to))))) < 30:
                            par[qi] = True
                    if dist < 20:
                        to = unit_tangent(other, other.project(p))
                        if np.degrees(np.arccos(min(1.0, abs(float(tv @ to))))) >= 30:
                            cross[qi] = True
            rec = {"type": t, "line_idx": li, "seg_k": k, "osm_ids": osm_ids, "length_m": round(seg.length, 1),
                   "par_frac": round(par.mean(), 3), "cross_frac": round(cross.mean(), 3), "geom": seg,
                   "cross_s": ss[cross]}
            if par.mean() > 0.25:
                rec["drop"] = "parallel_neighbor"
            elif cross.mean() > 0.40:
                rec["drop"] = "crossings"
            else:
                rec["drop"] = ""
            out.append(rec)
    return out


# ------------------------------------------------------------------ profiles
def frame(seg):
    s = np.arange(S_STEP / 2, seg.length, S_STEP)
    P = np.array([[seg.interpolate(v).x, seg.interpolate(v).y] for v in s])
    T = np.array([unit_tangent(seg, v) for v in s])
    N = np.c_[-T[:, 1], T[:, 0]]   # left normal
    X = P[:, 0:1] + D[None, :] * N[:, 0:1]
    Y = P[:, 1:2] + D[None, :] * N[:, 1:2]
    return s, X, Y, T


def sample(arr, gx0, gy1, res, X, Y):
    col = (X - gx0) / res - 0.5
    row = (gy1 - Y) / res - 0.5
    return ndimage.map_coordinates(arr, [row.ravel(), col.ravel()], order=1, mode="constant", cval=np.nan,
                                   prefilter=False).reshape(X.shape)


def along_mean(I, keep):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(I[keep], axis=0)


def contrast_stat(p_lin, c, w, r):
    """Center-vs-flank contrast (dB) at center c and on 16 parallel null lines. r = sensor cell (m)."""
    hc = max(r / 2, w / 2)
    f0 = max(1.5 * r, w / 2 + r)
    f1 = f0 + 15
    pdb = db(p_lin)

    def C(cc):
        dd = D - cc
        cen = p_lin[np.abs(dd) <= hc]
        L = p_lin[(dd >= -f1) & (dd <= -f0)]
        R = p_lin[(dd >= f0) & (dd <= f1)]
        if (np.isfinite(cen).mean() < 0.8 or np.isfinite(L).mean() < 0.8 or np.isfinite(R).mean() < 0.8):
            return np.nan
        return db(np.nanmean(cen)) - (db(np.nanmean(L)) + db(np.nanmean(R))) / 2

    obs = C(c)
    offs = f0 + 15 + w / 2 + 10 + 5 * np.arange(8)
    nulls = np.array([C(c + o) for o in offs] + [C(c - o) for o in offs])
    return obs, nulls, pdb


def zscores(obs, nulls):
    v = nulls[np.isfinite(nulls)]
    if len(v) < 10 or not np.isfinite(obs):
        return np.nan, np.full(len(nulls), np.nan)
    z = (obs - v.mean()) / v.std(ddof=1)
    zn = np.full(len(nulls), np.nan)
    for j in range(len(nulls)):
        if not np.isfinite(nulls[j]):
            continue
        o = np.delete(nulls, j)
        o = o[np.isfinite(o)]
        zn[j] = (nulls[j] - o.mean()) / o.std(ddof=1)
    return z, zn


def smooth(p, m):
    return ndimage.uniform_filter1d(p, int(round(m / 0.5)), mode="nearest")


def fwhm(pdb, c0, sign):
    """Full width at half contrast on a dB profile around c0. Returns dict or reason string."""
    ps = smooth(pdb, 1.0)
    idx0 = np.argmin(np.abs(D - c0))
    lo, hi = max(0, idx0 - 4), min(len(D), idx0 + 5)
    ic = lo + int(np.nanargmax(sign * ps[lo:hi]))
    P = ps[ic]
    dd = D - D[ic]
    bg = (np.abs(dd) >= 20) & (np.abs(dd) <= 40)
    B = np.nanmedian(ps[bg])
    xl = xr = np.nan
    BL = BR = B
    for _ in range(3):
        h = (P + B) / 2
        xr = xl = np.nan
        for j in range(ic, len(D)):
            if D[j] - D[ic] > 25:
                break
            if sign * (ps[j] - h) <= 0:
                a, b = ps[j - 1], ps[j]
                xr = D[j - 1] + (h - a) / (b - a) * 0.5 if b != a else D[j]
                break
        for j in range(ic, -1, -1):
            if D[ic] - D[j] > 25:
                break
            if sign * (ps[j] - h) <= 0:
                a, b = ps[j + 1], ps[j]
                xl = D[j + 1] - (h - a) / (b - a) * 0.5 if b != a else D[j]
                break
        if not (np.isfinite(xl) and np.isfinite(xr)):
            return "no_crossing"
        BL = np.nanmedian(ps[(D >= xl - 20) & (D <= xl - 5)])
        BR = np.nanmedian(ps[(D >= xr + 5) & (D <= xr + 20)])
        B = (BL + BR) / 2
    return {"center": (xl + xr) / 2, "width": xr - xl, "P": P, "B": B, "BL": BL, "BR": BR, "peak_d": D[ic]}


def capella_feature(I_cap, keep):
    """Width/contrast on the Capella block profiles. Returns (result dict, aligned dB profile)."""
    ns = I_cap.shape[0]
    nb = max(1, int(round(ns / 20)))
    edges = np.linspace(0, ns, nb + 1).astype(int)
    blocks = []
    for b in range(nb):
        kk = keep.copy()
        kk[:edges[b]] = False
        kk[edges[b + 1]:] = False
        blocks.append(along_mean(I_cap, kk) if kk.sum() >= 5 else np.full(len(D), np.nan))
    blocks = np.array(blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seg_lin = np.nanmean(blocks, axis=0)
    seg_db = db(seg_lin)
    ps = smooth(seg_db, 2.0)
    near = np.abs(D) <= 15
    base = np.nanmedian(ps[np.abs(D) <= 45])
    dev = np.where(near, ps - base, 0)
    if not np.isfinite(dev).any():
        return {"reject": "no_data"}, seg_db, seg_lin
    ic = int(np.nanargmax(np.abs(dev)))
    c0, sign = D[ic], float(np.sign(dev[ic]))
    # align blocks to the segment profile (+-4 m)
    win = np.abs(D - c0) <= 20
    al = []
    shifts = np.arange(-8, 9)
    for bp in blocks:
        bdb = db(bp)
        if not np.isfinite(bdb[win]).any():
            continue
        best, bs = -2, 0
        for sh in sorted(shifts, key=abs):
            r = corr(np.roll(bdb, sh)[win], seg_db[win], nmin=20)
            if r > best:
                best, bs = r, sh
        al.append(np.roll(bp, bs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        al_lin = np.nanmean(np.array(al), axis=0) if al else seg_lin
    al_db = db(al_lin)
    f = fwhm(al_db, c0, sign)
    if isinstance(f, str):
        return {"reject": f, "sign": sign, "c0": c0}, al_db, seg_lin
    # block consistency
    tol = max(3.0, f["width"] / 2)
    nb_ok = nb_valid = 0
    for bp in blocks:
        bdb = smooth(db(bp), 2.0)
        if not np.isfinite(bdb).any():
            continue
        nb_valid += 1
        bb = np.nanmedian(bdb[(np.abs(D - f["center"]) >= 20) & (np.abs(D - f["center"]) <= 40)])
        sel = np.abs(D - f["center"]) <= 15
        dv = np.where(sel, bdb - bb, np.nan)
        if not np.isfinite(dv).any():
            continue
        j = int(np.nanargmax(sign * dv))
        if sign * dv[j] > 0 and abs(D[j] - f["center"]) <= tol:
            nb_ok += 1
    res = {"sign": sign, "c0": c0, "cap_center": f["center"], "cap_width": f["width"],
           "cap_contrast": f["P"] - f["B"], "cap_bg_left": f["BL"], "cap_bg_right": f["BR"],
           "cap_blocks_ok": nb_ok, "cap_blocks_valid": nb_valid}
    return res, al_db, seg_lin


# ------------------------------------------------------------------ one scene
def run_scene(r, out):
    t0 = time.time()
    epsg = int(r.epsg)
    x0, y1 = float(r.win_x0), float(r.win_y1)
    window = shapely.box(x0, y1 - WIN, x0 + WIN, y1)
    inner = window.buffer(-EDGE)
    gxC, gyC = x0 - MC, y1 + MC
    gxS, gyS = x0 - MS, y1 + MS

    single_id = r.s1_single
    multi_ids = r.s1_multi.split(";")
    items = s1_items(sorted(set(multi_ids) | {single_id}))
    byid = {i.id: i for i in items}
    S = {}
    for pol in ("vv", "vh"):
        S[("single", pol)] = s1_read(byid[single_id], pol, epsg, gxS, gyS)
        stack = [S[("single", pol)] if i == single_id else s1_read(byid[i], pol, epsg, gxS, gyS) for i in multi_ids if i in byid]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            S[("multi", pol)] = np.nanmean(np.array(stack), axis=0)
    n_multi = len([i for i in multi_ids if i in byid])
    s1_bytes = NS * NS * 4 * 2 * (n_multi + (0 if single_id in multi_ids else 1))
    print(f"  s1 read {time.time() - t0:.0f}s, multi n={n_multi}", flush=True)
    # coarse alignment (amendment 2): multi-date VV over the window +- BIG, Capella at 10 m over the same area
    gxB, gyB = x0 - BIG, y1 + BIG
    nB = int((WIN + 2 * BIG) / 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s1_big = np.nanmean(np.array([s1_read(byid[i], "vv", epsg, gxB, gyB, n=nB) for i in multi_ids if i in byid]), axis=0)
    cdx, cdy, r_c, z_c, cb = coarse_align(r.capella_href, epsg, gxS, gyS, s1_big, gxB, gyB)
    s1_bytes += nB * nB * 4 * n_multi
    coarse_ok = z_c >= 6.0      # peak stands out of the match surface
    print(f"  coarse align dx={cdx:.0f} dy={cdy:.0f} ncc={r_c:.3f} peak_z={z_c:.1f} ok={coarse_ok} {time.time() - t0:.0f}s", flush=True)
    if not coarse_ok:
        cdx = cdy = 0.0
    C, cap_bytes, cap_res = capella_read(r.capella_href, epsg, gxC + cdx, gyC + cdy)
    cap_bytes += cb
    print(f"  capella read {time.time() - t0:.0f}s, {cap_bytes / 1e6:.1f} MB, src res {cap_res:.2f} m, "
          f"valid {np.isfinite(C).mean():.2f}", flush=True)

    dx_c, dy_c, r_al, r_coarse = align(C, db(S[("multi", "vv")]))
    r_noshift = corr(highpass(db(block10(C, MC - MS, MC - MS))), highpass(db(S[("multi", "vv")])))
    # fine shift (+-10 m, 2 m steps) after the coarse one (amendment 2)
    use = coarse_ok and r_al > r_noshift
    dx, dy = (dx_c, dy_c) if use else (0, 0)
    print(f"  align corr dx={dx_c} dy={dy_c} r={r_al:.3f} (no shift r={r_noshift:.3f}) applied={use} {time.time() - t0:.0f}s", flush=True)

    src = gpd.read_parquet(OVDIR / f"{r.capella_id}.parquet")
    lines, neighbors = build_lines(src, epsg, window)
    segs = make_segments(lines, neighbors, inner)
    print(f"  {len(segs)} segments ({sum(1 for s in segs if not s['drop'])} pass OSM checks) {time.time() - t0:.0f}s", flush=True)

    # flight direction (approximate ground-track heading, Earth rotation ignored)
    lat = float(r.lat)
    a = np.degrees(np.arcsin(np.clip(np.cos(np.radians(98.18)) / np.cos(np.radians(lat)), -1, 1)))
    heading = a if r.s1_orbit_state == "ascending" else 180 - a   # deg clockwise from north
    hv = np.array([np.sin(np.radians(heading)), np.cos(np.radians(heading))])

    def measure_all(dx, dy, with_s1):
        rows, prof = [], {}
        for si, sg in enumerate(segs):
            rec = {"capella_id": r.capella_id, "site_id": r.site_id, "country": r.country, "seg_id": f"{r.site_id}_{si}",
                   **{k: sg[k] for k in ("type", "osm_ids", "length_m", "par_frac", "cross_frac")}}
            mid = sg["geom"].interpolate(0.5, normalized=True)
            rec["mid_x"], rec["mid_y"] = round(mid.x, 1), round(mid.y, 1)
            tv = unit_tangent(sg["geom"], sg["geom"].length / 2)
            rec["tan_x"], rec["tan_y"] = round(float(tv[0]), 4), round(float(tv[1]), 4)
            rec["angle_to_flight"] = round(float(np.degrees(np.arccos(min(1.0, abs(float(tv @ hv)))))), 1)
            if sg["drop"]:
                rec["status"] = sg["drop"]
                rows.append(rec)
                continue
            s, X, Y, T = frame(sg["geom"])
            keep = np.ones(len(s), bool)
            for cs in sg["cross_s"]:
                keep &= np.abs(s - cs) > 2.5
            Ic = sample(C, gxC, gyC, 1.0, X + dx, Y + dy)
            cen = Ic[:, np.abs(D) <= 40]
            keep &= np.isfinite(cen).mean(axis=1) > 0.9
            if keep.mean() < 0.6:
                rec["status"] = "capella_nodata"
                rows.append(rec)
                continue
            try:
                cf, cap_al_db, cap_seg_lin = capella_feature(Ic, keep)
            except ValueError:          # all-NaN profile stretch
                cf, cap_al_db, cap_seg_lin = {"reject": "no_data"}, None, None
            rec.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cf.items()})
            if "reject" in cf:
                rec["status"] = "cap_" + cf["reject"]
                rows.append(rec)
                continue
            w, c = cf["cap_width"], cf["cap_center"]
            obs, nulls, _ = contrast_stat(cap_seg_lin, c, w, 1.0)
            zc, _ = zscores(obs, nulls)
            rec["cap_C"], rec["cap_z"] = round(float(obs), 3), round(float(zc), 2) if np.isfinite(zc) else np.nan
            ctr = abs(cf["cap_contrast"])
            reason = ""
            if not (np.isfinite(zc) and abs(zc) >= CAP_Z):
                reason = "cap_weak_z"
            elif ctr < 1.0:
                reason = "cap_low_contrast"
            elif abs(cf["cap_bg_left"] - cf["cap_bg_right"]) >= 0.5 * ctr:
                reason = "cap_step_edge"
            elif not (0.75 <= w <= 40):
                reason = "cap_width_range"
            elif abs(c) > 15:
                reason = "cap_center_far"
            elif cf["cap_blocks_ok"] < 6:
                reason = "cap_blocks_inconsistent"
            if reason:
                rec["status"] = reason
                rows.append(rec)
                continue
            rec["status"] = "ok"
            rec["n_along"] = int(keep.sum())
            if not with_s1:
                rows.append(rec)
                continue
            P = {"cap_db": cap_al_db.astype("float32")}
            zn_store = {}
            for prod in ("single", "multi"):
                for pol in ("vv", "vh"):
                    Is = sample(S[(prod, pol)], gxS, gyS, 10.0, X, Y)
                    p = along_mean(Is, keep)
                    o, nl, pdb = contrast_stat(p, c, w, 10.0)
                    z, zn = zscores(o, nl)
                    k = f"{prod}_{pol}"
                    rec[f"{k}_C"] = round(float(o), 3) if np.isfinite(o) else np.nan
                    rec[f"{k}_null_sd"] = round(float(np.nanstd(nl, ddof=1)), 3)
                    rec[f"{k}_z"] = round(float(z), 3) if np.isfinite(z) else np.nan
                    rec[f"{k}_det"] = bool(np.isfinite(z) and abs(z) > THR1)
                    rec[f"{k}_far_n"] = int(np.nansum(np.abs(zn) > THR1))
                    rec[f"{k}_far_m"] = int(np.isfinite(zn).sum())
                    zn_store[k] = zn
                    P[k] = pdb.astype("float32")
                zv, zh = rec[f"{prod}_vv_z"], rec[f"{prod}_vh_z"]
                rec[f"{prod}_either_det"] = bool(np.nanmax([abs(zv), abs(zh)]) > THR2) if np.isfinite([zv, zh]).any() else False
                ze = np.fmax(np.abs(zn_store[f"{prod}_vv"]), np.abs(zn_store[f"{prod}_vh"]))
                rec[f"{prod}_either_far_n"] = int(np.nansum(ze > THR2))
                rec[f"{prod}_either_far_m"] = int(np.isfinite(ze).sum())
            prof[rec["seg_id"]] = np.stack([P[k] for k in ("cap_db", "single_vv", "single_vh", "multi_vv", "multi_vh")])
            rows.append(rec)
        return rows, prof

    def ground_fit(rows):
        """Robust LS of Capella center offsets on ground-level features (road, track, ditch_canal)."""
        d = pd.DataFrame(rows)
        if not len(d) or "cap_center" not in d:
            return np.nan, np.nan, np.nan, 0
        g = d[(d.status == "ok") & d.type.isin(["road", "track", "ditch_canal"])]
        if len(g) < 8:
            return np.nan, np.nan, np.nan, len(g)
        Nn = np.c_[-g.tan_y.values, g.tan_x.values]
        cc = g.cap_center.values.astype(float)
        keep = np.ones(len(g), bool)
        for _ in range(3):
            sol, *_ = np.linalg.lstsq(Nn[keep], cc[keep], rcond=None)
            res = cc - Nn @ sol
            rms = np.sqrt(np.mean(res[keep] ** 2))
            keep = np.abs(res) <= max(2.5 * rms, 3.0)
        return float(sol[0]), float(sol[1]), float(rms), int(keep.sum())

    rows1, _ = measure_all(dx, dy, False)
    gdx, gdy, grms, gn = ground_fit(rows1)
    osm_refined = bool(np.isfinite(gdx))
    if osm_refined:
        dx, dy = dx + gdx, dy + gdy
    print(f"  ground-feature refinement: {gn} segments, shift {gdx:.1f},{gdy:.1f} m, rms {grms:.1f} m "
          f"-> total fine shift {dx:.1f},{dy:.1f} {time.time() - t0:.0f}s", flush=True)
    rows, prof = measure_all(dx, dy, True)

    df = pd.DataFrame(rows)
    # OSM-vs-Capella residual: fit center offsets c = delta . n over accepted segments (n = left normal)
    osm_dx = osm_dy = osm_rms = np.nan
    if len(df) and (df.status == "ok").sum() >= 10:
        okd = df[df.status == "ok"]
        Nn = np.c_[-okd.tan_y.values, okd.tan_x.values]
        sol, *_ = np.linalg.lstsq(Nn, okd.cap_center.values, rcond=None)
        osm_dx, osm_dy = float(sol[0]), float(sol[1])
        osm_rms = float(np.sqrt(np.mean((okd.cap_center.values - Nn @ sol) ** 2)))
    for k, v in {"coarse_dx": cdx, "coarse_dy": cdy, "coarse_ncc": round(r_c, 3), "coarse_peak_z": round(z_c, 1), "coarse_ok": coarse_ok, "align_dx": dx, "align_dy": dy, "align_corr_dx": dx_c, "align_corr_dy": dy_c, "align_r": round(r_al, 3),
                 "align_r_noshift": round(r_noshift, 3), "osm_refined": osm_refined, "ground_dx": gdx, "ground_dy": gdy,
                 "ground_rms": grms, "ground_n": gn, "osm_dx": osm_dx, "osm_dy": osm_dy, "osm_rms": osm_rms,
                 "s1_n_multi": n_multi, "cap_src_res_m": round(cap_res, 2)}.items():
        df[k] = v
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "segments.csv", index=False)
    np.savez_compressed(out / "profiles.npz", ids=np.array(list(prof.keys())),
                        prof=np.array(list(prof.values())) if prof else np.zeros((0, 5, len(D)), "float32"), D=D)
    np.savez_compressed(out / "arrays.npz", cap_db=db(C).astype("float16"),
                        **{f"{a}_{b}": db(S[(a, b)]).astype("float16") for (a, b) in S},
                        gxC=gxC, gyC=gyC, coarse_dx=cdx, coarse_dy=cdy, gxS=gxS, gyS=gyS, dx=dx, dy=dy, epsg=epsg)
    meta = {"capella_id": r.capella_id, "seconds": round(time.time() - t0), "capella_mb": round(cap_bytes / 1e6, 1),
            "s1_mb": round(s1_bytes / 1e6, 1), "align_dx": dx, "align_dy": dy, "align_corr_dx": dx_c, "align_corr_dy": dy_c,
            "coarse_dx": cdx, "coarse_dy": cdy, "coarse_ncc": r_c, "coarse_peak_z": z_c, "coarse_ok": coarse_ok, "align_applied": bool(use), "osm_refined": osm_refined,
            "ground_dx": gdx, "ground_dy": gdy, "ground_rms": grms, "ground_n": gn, "osm_dx": osm_dx, "osm_dy": osm_dy, "osm_rms": osm_rms, "align_r": r_al, "align_r_coarse": r_coarse,
            "align_r_noshift": r_noshift, "n_segments": len(df), "n_ok": int((df.status == "ok").sum()) if len(df) else 0}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"  done {meta}", flush=True)


def main():
    sc = pd.read_csv(HERE / "scenes_selected.csv")
    want = sys.argv[1:]
    for r in sc.itertuples():
        if want and r.capella_id not in want:
            continue
        out = HERE / "data" / r.capella_id
        if (out / "meta.json").exists():
            continue
        if not isinstance(r.s1_single, str) or not r.s1_single:
            print(r.capella_id, "no S1", flush=True)
            continue
        print(r.site_id, r.country, r.capella_id, flush=True)
        try:
            run_scene(r, out)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
