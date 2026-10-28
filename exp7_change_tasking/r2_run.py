"""Round 2: how small a change can Sentinel-1 catch? Capella change patches as truth.

Per pair in r2_pairs.csv (Capella x1 < x2, both inside one same-track Sentinel-1 interval s_b <= x1,
x2 <= s_a, so lasting change between x1 and x2 also lies between the S1 passes):

grid       2.56 km square at the centroid of the two Capella footprints' intersection, local UTM,
           origin on a 10 m multiple. S1 grid 256 x 256 at 10 m; Capella grid 1280 x 1280 at 2 m, so
           each S1 pixel holds exactly 5 x 5 Capella pixels.
Capella    GEO COG, windowed read of the coarsest overview with pixels <= 0.8 m (native if none),
           amplitude squared to intensity, block-averaged ("average" resampling) onto the 2 m grid.
co-reg     sub-pixel shift of x2 onto x1 by phase correlation of the 2 m dB images (upsample 20); x2 is
           re-gridded from the source with its geotransform moved by that shift (no second
           interpolation). Shifts over 5 px (10 m) are refused and the pair is marked.
masks      Capella valid = both dates have data, eroded 5 px (10 m). Edge mask: pixels where a half-pixel
           (1 m) misregistration alone would move either date by more than 1.5 dB (|grad dB| > 3 dB per
           px on the 3x3-smoothed image), dilated 1 px. Shadow/noise floor: both dates below the scene's
           2nd percentile. Stable bright scatterers (layover, sidelobes): both dates above the 99.5th
           percentile, dilated 2 px.
change     LR = 10 log10(I2 / I1) on the 2 m multilooked data, minus its scene median (the products are
           not radiometrically matched). The plain |LR| > 3 dB rule fires on slope and building edges that
           move by a pixel or two between dates (calibration: Kilauea crater wall, Pilbara pit benches), so
           the truth uses a shift-tolerant margin: brightening = after minus the maximum of before within
           +-TOL_R px, darkening = before minus the maximum of after within +-TOL_R px. Candidate pixels
           margin > 3 dB. A patch is an 8-connected group of candidates with >= MIN_PX pixels that contains a
           seed pixel with margin > SEED_DB (hysteresis). A 4 m tolerance (TOL_R = 2) set is scored too.
           The seed level and floor come from the calibration run (r2_run.py --calib) and the speckle
           simulation in r2_calib.log, fixed before any Sentinel-1 scoring. Patch area = pixels x 4 m^2;
           sign = sign of the patch's mean LR.
S1         Planetary Computer RTC gamma0, VV and VH, both passes of one track, bilinear to the 10 m grid,
           dB, 3x3 mean, difference minus its scene median, flag |d| > 3 dB, valid eroded by 1 (exp7 rule
           with the median correction pilot.md recommended).
S1-Capella integer shift (+-3 S1 px search) of the Capella grid onto S1 by correlation of 10 m mean dB
           images over land; applied only when the optimum lies inside +-2 px and beats zero shift by r >= 0.05.
water      ESA WorldCover 2021 permanent water (80) and no-data (open ocean) 10 m pixels leave everything:
           Capella truth, S1 flags, medians, false alarm and precision. Sea clutter changes with wind and
           flagged a third of the coastal Miami window in a first try.

Scoring (fixed before results):
  loose    patch caught when any S1-flagged 10 m pixel lies within 10 m (one S1 pixel) of the patch
           (Euclidean distance on the 2 m grid between flagged area and patch pixels <= 10 m).
  strict   patch caught when >= 50% of its pixels fall inside S1-flagged 10 m pixels (no dilation).
  chance   the loose rule after 20 random circular shifts of the S1 flag map (>= 30 px), per patch.
  Patches must lie wholly inside S1-valid ground.
  False alarm: S1-flagged fraction of Capella-stable 10 m pixels (all 25 Capella pixels valid and
  unmasked, and no patch pixel within 10 m). Region precision: S1 flag regions (>= 1 px and >= 20 px)
  explained when >= 30% of their pixels lie within 10 m of a patch; chance from 20 random shifts.
Flag sets: VV (primary), VH, VV or VH.

Writes data/r2/<tag>.npz, data/r2/<tag>_patches.csv, and r2_pair_results.csv. Bytes read per pair come
from the worker process's I/O counters (an upper bound: includes local file reads).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "r2"
N, RES = 256, 10.0
CN, CRES, K = 1280, 2.0, 5
MAX_SRC_RES = 0.8
THRESH = 3.0
MIN_PX, SEED_DB = 2, 4.0          # fixed from the calibration (see r2_calib.log)
TOL_R = 1                          # shift tolerance, px of 2 m
EDGE_DB_PER_PX = 3.0
MAX_COREG_PX = 5.0
FLAGSETS = ("vv", "vh", "any")


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


# ------------------------------------------------------------------ readers
BYTES = {"n": 0}


def window_bytes(ds, bounds, crs_epsg):
    """Compressed bytes of the TIFF blocks a read of these bounds (in EPSG crs_epsg) touches."""
    try:
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        b = transform_bounds(f"EPSG:{crs_epsg}", ds.crs, *bounds) if ds.crs.to_epsg() != crs_epsg else bounds
        w = from_bounds(*b, ds.transform)
        bh, bw = ds.block_shapes[0]
        r0, r1 = max(int(w.row_off // bh), 0), min(int((w.row_off + w.height) // bh), (ds.height - 1) // bh)
        c0, c1 = max(int(w.col_off // bw), 0), min(int((w.col_off + w.width) // bw), (ds.width - 1) // bw)
        n = sum(ds.block_size(1, i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1))
        BYTES["n"] += n
        return n
    except Exception:
        return 0


def read_capella(href, epsg, x0, y1):
    """Source intensity window (float32), its transform and pixel size. Grid it with grid_capella."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds
    dst_t = from_origin(x0, y1, CRES, CRES)
    with rasterio.open(href) as s0:
        native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
        ovs = s0.overviews(1)
        same = s0.crs.to_epsg() == epsg
    lvl, res = None, native
    for i, f in enumerate(ovs):
        if native * f <= MAX_SRC_RES:
            lvl, res = i, native * f
    kw = {"overview_level": lvl} if lvl is not None else {}
    pad = 40.0
    with rasterio.open(href, **kw) as s:
        window_bytes(s, (x0 - pad, y1 - CN * CRES - pad, x0 + CN * CRES + pad, y1 + pad), epsg)
        if same:
            win = from_bounds(x0 - pad, y1 - CN * CRES - pad, x0 + CN * CRES + pad, y1 + pad, s.transform)
            win = win.round_offsets().round_lengths()
            a = s.read(1, window=win, boundless=True, fill_value=0).astype("float32")
            st = s.window_transform(win)
        else:
            r = res
            n = int(round((CN * CRES + 2 * pad) / r))
            vt = from_origin(x0 - pad, y1 + pad, r, r)
            with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=vt, width=n, height=n, resampling=Resampling.nearest,
                           nodata=0) as v:
                a = v.read(1).astype("float32")
            st = vt
    a[a <= 0] = np.nan
    return a * a, st, res


def grid_capella(inten, st, epsg, x0, y1, dxy=(0.0, 0.0)):
    """Block-average source intensity onto the 2 m grid; dxy = (east, north) metres added to the source
    geotransform (the co-registration shift, applied without a second interpolation)."""
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    dst_t = from_origin(x0, y1, CRES, CRES)
    st = Affine(st.a, st.b, st.c + dxy[0], st.d, st.e, st.f + dxy[1])
    out = np.full((CN, CN), np.nan, "float32")
    reproject(inten, out, src_transform=st, src_crs=f"EPSG:{epsg}", dst_transform=dst_t, dst_crs=f"EPSG:{epsg}",
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def read_s1(items, epsg, x0, y1, pol):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    out = None
    for it in items:
        if pol not in it.assets:
            continue
        with rasterio.open(it.assets[pol].href) as s, WarpedVRT(
                s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, RES, RES), width=N, height=N,
                resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
            window_bytes(s, (x0 - 20, y1 - N * RES - 20, x0 + N * RES + 20, y1 + 20), epsg)
            a = v.read(1)
        a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
        out = a if out is None else np.where(np.isfinite(out), out, a)
        if np.isfinite(out).all():
            break
    return out


def pc_client():
    import planetary_computer as pc
    from pystac_client import Client
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def s1_items(bbox, t, rel, state):
    t = pd.Timestamp(t)
    rng = f"{(t - pd.Timedelta(minutes=10)):%Y-%m-%dT%H:%M:%SZ}/{(t + pd.Timedelta(minutes=10)):%Y-%m-%dT%H:%M:%SZ}"
    for k in range(4):
        try:
            return [i for i in pc_client().search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=rng).items()
                    if i.properties.get("sat:relative_orbit") == rel and i.properties.get("sat:orbit_state") == state]
        except Exception:
            time.sleep(5 * (k + 1))
    return []


def read_worldcover(epsg, x0, y1, bbox):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    out = np.zeros((N, N), "uint8")
    try:
        items = pc_client().search(collections=["esa-worldcover"], bbox=bbox,
                                   query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items()
        for it in items:
            with rasterio.open(it.assets["map"].href) as s, WarpedVRT(
                    s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, RES, RES), width=N, height=N,
                    resampling=Resampling.nearest, nodata=0) as v:
                window_bytes(s, (x0, y1 - N * RES, x0 + N * RES, y1), epsg)
                a = v.read(1)
            out = np.where(out == 0, a, out)
    except Exception:
        traceback.print_exc()
    return out


# ------------------------------------------------------------------ pieces
def coreg(a_db, b_db):
    from skimage.registration import phase_cross_correlation
    m = np.isfinite(a_db) & np.isfinite(b_db)
    if m.mean() < 0.2:
        return (0.0, 0.0), np.nan
    lo, hi = np.nanpercentile(a_db[m], [1, 99])
    f = lambda x: np.clip(np.where(m, x, np.nanmedian(x[m])), lo, hi)
    c = slice(CN // 8, CN - CN // 8)
    sh, err, _ = phase_cross_correlation(f(a_db)[c, c], f(b_db)[c, c], upsample_factor=20, normalization=None)
    return (float(sh[0]), float(sh[1])), float(err)


def capella_masks(a_db, b_db):
    from scipy import ndimage
    valid = np.isfinite(a_db) & np.isfinite(b_db)
    valid = ndimage.binary_erosion(valid, iterations=5)
    edge = np.zeros_like(valid)
    for x in (a_db, b_db):
        xs = ndimage.uniform_filter(np.where(np.isfinite(x), x, np.nanmedian(x)), 3)
        gy, gx = np.gradient(xs)
        edge |= np.hypot(gy, gx) > EDGE_DB_PER_PX
    edge = ndimage.binary_dilation(edge, iterations=1)
    lo = np.nanpercentile(np.fmin(a_db, b_db)[valid], 2)
    hi = np.nanpercentile(np.fmax(a_db, b_db)[valid], 99.5)
    shadow = (a_db < lo) & (b_db < lo)
    bright = ndimage.binary_dilation((a_db > hi) & (b_db > hi), iterations=2)
    use = valid & ~edge & ~shadow & ~bright
    return valid, use, {"edge": float((edge & valid).mean() / max(valid.mean(), 1e-9)),
                        "shadow": float((shadow & valid).mean() / max(valid.mean(), 1e-9)),
                        "bright": float((bright & valid).mean() / max(valid.mean(), 1e-9))}


def tol_change(a_db, b_db, use, r=TOL_R):
    """Shift-tolerant change margin (dB). Brightening: b minus the max of a within +-r px; darkening: a minus
    the max of b within +-r px. An edge displaced by <= r px finds its match and drops out; a new or removed
    object of any size keeps its full contrast. Returns a signed margin (positive = brightening)."""
    from scipy import ndimage
    fill = lambda x: np.where(np.isfinite(x), x, -99.0)
    amax = ndimage.maximum_filter(fill(a_db), size=2 * r + 1)
    bmax = ndimage.maximum_filter(fill(b_db), size=2 * r + 1)
    up_ = np.clip(b_db - amax, 0, None)
    dn_ = np.clip(a_db - bmax, 0, None)
    m = np.where(up_ >= dn_, up_, -dn_)
    m[~use] = np.nan
    return m


def patches(lr, use, min_px=MIN_PX, seed_db=SEED_DB):
    """lr here is the shift-tolerant margin from tol_change (signed dB)."""
    from scipy import ndimage
    cand = (np.abs(lr) > THRESH) & use
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return lab, np.zeros(0, int), np.zeros(0), np.zeros(0, bool)
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(lab), lab, idx).astype(int)
    peak = ndimage.maximum(np.abs(np.nan_to_num(lr)), lab, idx)
    mean = ndimage.mean(np.nan_to_num(lr), lab, idx)
    keep = (size >= min_px) & (peak > seed_db)
    remap = np.zeros(n + 1, int)
    remap[idx[keep]] = np.arange(1, keep.sum() + 1)
    return remap[lab], size[keep], mean[keep], keep


def s1_change(b_db, a_db, land=None):
    from scipy import ndimage
    m = np.isfinite(a_db) & np.isfinite(b_db)
    k = np.ones((3, 3)) / 9
    d = ndimage.convolve(np.where(m, a_db, 0), k) - ndimage.convolve(np.where(m, b_db, 0), k)
    ok = ndimage.binary_erosion(m, iterations=1)
    mk = ok & land if land is not None and (ok & land).sum() > 100 else ok
    med = float(np.median(d[mk])) if mk.any() else 0.0
    d = d - med
    return (np.abs(d) > THRESH) & ok, d, ok, med


def up(m):
    return np.kron(m, np.ones((K, K), dtype=m.dtype))


def best_int_shift(ref, mov, r=3):
    def z(a):
        return np.nan_to_num(a - np.nanmean(a))
    A, B = z(ref), z(mov)
    res = {}
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            Bs = np.roll(B, (dy, dx), axis=(0, 1))[r:-r, r:-r]
            res[(dy, dx)] = np.corrcoef(A[r:-r, r:-r].ravel(), Bs.ravel())[0, 1]
    best = max(res, key=res.get)
    if max(abs(best[0]), abs(best[1])) < r and res[best] - res[(0, 0)] >= 0.05:     # interior optimum only
        return best, res[best], res[(0, 0)]
    return (0, 0), res[(0, 0)], res[(0, 0)]


def roll_fill(a, s, fill):
    out = np.roll(a, s, axis=(0, 1))
    dy, dx = s
    if dy > 0:
        out[:dy] = fill
    elif dy < 0:
        out[dy:] = fill
    if dx > 0:
        out[:, :dx] = fill
    elif dx < 0:
        out[:, dx:] = fill
    return out


def score(lab, nP, flag10, s1ok10, rng_shifts):
    """Per-patch loose/strict catch and chance for one S1 flag map."""
    from scipy import ndimage
    f2 = up(flag10)
    dist = ndimage.distance_transform_edt(~f2) * CRES
    idx = np.arange(1, nP + 1)
    mind = ndimage.minimum(dist, lab, idx) if nP else np.zeros(0)
    inflag = ndimage.mean(f2.astype(float), lab, idx) if nP else np.zeros(0)
    chance = np.zeros(nP)
    for s in rng_shifts:
        fs = up(np.roll(flag10, s, axis=(0, 1)) & s1ok10)
        ds = ndimage.distance_transform_edt(~fs) * CRES
        chance += (ndimage.minimum(ds, lab, idx) <= RES) if nP else 0
    return mind <= RES, inflag >= 0.5, chance / len(rng_shifts), inflag


def region_precision(flag10, near10, min_px, shifts):
    from scipy import ndimage
    lab, n = ndimage.label(flag10, structure=np.ones((3, 3)))
    if n == 0:
        return 0, np.nan, np.nan
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(lab), lab, idx)
    keep = size >= min_px
    if not keep.any():
        return 0, np.nan, np.nan
    hit = ndimage.mean(near10.astype(float), lab, idx)
    nulls = [np.mean(ndimage.mean(np.roll(near10, s, axis=(0, 1)).astype(float), lab, idx)[keep] >= 0.3) for s in shifts]
    return int(keep.sum()), float(np.mean(hit[keep] >= 0.3)), float(np.mean(nulls))


# ------------------------------------------------------------------ one pair
def run_pair(pr, calib=False):
    import psutil
    from pyproj import Transformer
    from scipy import ndimage
    from shapely import wkt
    warnings.filterwarnings("ignore")
    proc = psutil.Process()
    io0 = proc.io_counters().read_bytes
    t0 = time.time()
    tag = f"site{pr['site_id']}_{pr['x1']}_{pr['x2']}"
    c = wkt.loads(pr["centroid_wkt"])
    zone = int((c.x + 180) // 6) + 1
    epsg = (32600 if c.y >= 0 else 32700) + zone
    cx, cy = Transformer.from_crs(4326, epsg, always_xy=True).transform(c.x, c.y)
    x0, y1 = round(cx - N * RES / 2, -1), round(cy + N * RES / 2, -1)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon0, lat0 = inv.transform(x0, y1 - N * RES)
    lon1, lat1 = inv.transform(x0 + N * RES, y1)
    bbox = [lon0, lat0, lon1, lat1]

    S1_, st1, src_res = read_capella(pr["href1"], epsg, x0, y1)
    S2_, st2, _ = read_capella(pr["href2"], epsg, x0, y1)
    I1, I2 = grid_capella(S1_, st1, epsg, x0, y1), grid_capella(S2_, st2, epsg, x0, y1)
    # permanent water and ocean (WorldCover 80 or no data) leave the analysis: sea clutter changes with wind
    wc = read_worldcover(epsg, x0, y1, bbox)
    land10 = (wc != 80) & (wc != 0) if wc.any() else np.ones_like(wc, bool)   # WorldCover read failed: keep all
    land2 = up(land10)
    a_db, b_db = db(I1), db(I2)
    (sy, sx), cerr = coreg(a_db, b_db)
    res = {"tag": tag, "epsg": epsg, "src_res_m": round(src_res, 3), "coreg_dy_px": round(sy, 2), "coreg_dx_px": round(sx, 2)}
    if max(abs(sy), abs(sx)) > MAX_COREG_PX:
        res["status"] = "coreg_refused"
        return res, None, (proc.io_counters().read_bytes - io0)
    if abs(sy) > 0.05 or abs(sx) > 0.05:
        I2 = grid_capella(S2_, st2, epsg, x0, y1, dxy=(sx * CRES, -sy * CRES))
        b_db = db(I2)
        (ry, rx), _ = coreg(a_db, b_db)
        res.update({"coreg_resid_dy": round(ry, 2), "coreg_resid_dx": round(rx, 2)})
    del S1_, S2_
    valid, use, mfrac = capella_masks(a_db, b_db)
    valid, use = valid & land2, use & land2
    if valid.mean() < 0.3:
        res.update({"status": "small_capella_overlap", "c_valid_frac": round(float(valid.mean()), 3)})
        return res, None, (proc.io_counters().read_bytes - io0)
    lr = b_db - a_db
    lr_med = float(np.nanmedian(lr[valid]))
    lr = lr - lr_med
    v = lr[use]
    mad_sd = float(1.4826 * np.median(np.abs(v - np.median(v))))
    res.update({"c_valid_frac": round(float(valid.mean()), 3), "c_use_frac": round(float(use.mean()), 3),
                "mask_edge": round(mfrac["edge"], 3), "mask_shadow": round(mfrac["shadow"], 3),
                "mask_bright": round(mfrac["bright"], 3), "lr_median_db": round(lr_med, 2), "lr_mad_sd_db": round(mad_sd, 3),
                "lr_frac_gt3": round(float((np.abs(v) > THRESH).mean()), 4)})
    if calib:
        out = {"lr": lr.astype("float32"), "use": use, "valid": valid, "a_db": (a_db + lr_med).astype("float16"),
               "b_db": b_db.astype("float16")}
        for tr in (0, 1, 2):
            mg = tol_change(a_db + lr_med, b_db, use, tr) if tr else np.where(use, lr, np.nan)
            res[f"frac_gt3_r{tr}"] = round(float((np.abs(mg[use]) > THRESH).mean()), 4)
            for mp in (1, 2, 4, 8):
                for sd in (3.0, 4.0, 5.0, 6.0):
                    lab, size, _, _ = patches(mg, use, mp, sd)
                    res[f"np_r{tr}_{mp}_{sd}"] = round(len(size) / (use.sum() * 4e-6), 1)
        mg = tol_change(a_db + lr_med, b_db, valid, 1)       # no edge/shadow/bright masks
        for mp in (2, 4):
            for sd in (4.0, 5.0):
                lab, size, _, _ = patches(mg, valid, mp, sd)
                res[f"np_r1nomask_{mp}_{sd}"] = round(len(size) / (valid.sum() * 4e-6), 1)
        res["status"] = "calib"
        return res, out, (proc.io_counters().read_bytes - io0)

    # Sentinel-1
    sb = s1_items(bbox, pr["s1_before_time"], int(pr["rel_orbit"]), pr["orbit_state"])
    sa = s1_items(bbox, pr["s1_after_time"], int(pr["rel_orbit"]), pr["orbit_state"])
    if not sb or not sa:
        res["status"] = "no_s1_item"
        return res, None, (proc.io_counters().read_bytes - io0)
    s1 = {}
    for pol in ("vv", "vh"):
        b = read_s1(sb, epsg, x0, y1, pol)
        a = read_s1(sa, epsg, x0, y1, pol)
        if b is None or a is None:
            continue
        s1[pol] = (db(b), db(a))
    if "vv" not in s1:
        res["status"] = "no_s1_vv"
        return res, None, (proc.io_counters().read_bytes - io0)
    flags, diffs, oks = {}, {}, {}
    for pol, (b, a) in s1.items():
        flags[pol], diffs[pol], oks[pol], med = s1_change(b, a, land10)
        oks[pol] = oks[pol] & land10
        flags[pol] = flags[pol] & land10
        res[f"s1_{pol}_median_db"] = round(med, 2)
    s1ok = oks["vv"] & (oks["vh"] if "vh" in oks else True)
    if "vh" in flags:
        flags["any"] = flags["vv"] | flags["vh"]

    mg = tol_change(a_db + lr_med, b_db, use, TOL_R)
    mg2 = tol_change(a_db + lr_med, b_db, use, 2)    # sensitivity: 4 m tolerance

    # S1 <-> Capella alignment (integer 10 m shift of the Capella grid)
    with np.errstate(invalid="ignore"):
        c10 = db(np.nanmean((0.5 * (I1 + I2)).reshape(N, K, N, K), axis=(1, 3)))
    s1m = 0.5 * (s1["vv"][0] + s1["vv"][1])
    (dy, dx), r_al, r0 = best_int_shift(np.where(land10, s1m, np.nan), np.where(land10, c10, np.nan))
    res.update({"s1c_shift_px": f"{dy},{dx}", "s1c_r": round(float(r_al), 3), "s1c_r0": round(float(r0), 3)})
    if (dy, dx) != (0, 0):
        sh = (dy * K, dx * K)
        a_db, b_db = roll_fill(a_db, sh, np.nan), roll_fill(b_db, sh, np.nan)
        lr, valid, use = roll_fill(lr, sh, np.nan), roll_fill(valid, sh, False), roll_fill(use, sh, False)
        mg, mg2 = roll_fill(mg, sh, np.nan), roll_fill(mg2, sh, np.nan)

    # patches inside S1-valid ground
    use = use & up(s1ok)
    lab, size, mean, _ = patches(mg, use)
    lab2, size2, mean2, _ = patches(mg2, use)
    nP = len(size)
    wc2 = up(wc)
    idx = np.arange(1, nP + 1)
    rng = np.random.default_rng(int(pr["site_id"]) * 1000 + 7)
    shifts = [(int(rng.integers(30, N - 30)), int(rng.integers(30, N - 30))) for _ in range(20)]
    near = ndimage.binary_dilation(lab > 0, structure=np.ones((3, 3)), iterations=K)   # within 10 m of a patch
    near10 = near.reshape(N, K, N, K).any(axis=(1, 3))
    block_use = use.reshape(N, K, N, K).all(axis=(1, 3))
    stable10 = block_use & ~near10 & s1ok
    prow = {"patch": idx, "area_m2": size * CRES * CRES, "lr_mean_db": np.round(mean, 2), "sign": np.where(mean > 0, "bright", "dark")}
    if nP:
        h = np.bincount(lab.ravel().astype(np.int64) * 256 + wc2.ravel(), minlength=(nP + 1) * 256).reshape(nP + 1, 256)
        cls = h[1:].argmax(axis=1)
        water = ndimage.binary_dilation(wc2 == 80, iterations=K)
        prow["wc_class"] = cls
        prow["near_water"] = ndimage.maximum(water.astype(int), lab, idx).astype(bool)
        com = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
        prow["row"] = [int(r) for r, _ in com]
        prow["col"] = [int(q) for _, q in com]
    for fs, fl in flags.items():
        fl = fl & s1ok
        loose, strict, chance, frac = score(lab, nP, fl, s1ok, shifts)
        prow[f"{fs}_loose"], prow[f"{fs}_strict"], prow[f"{fs}_chance"], prow[f"{fs}_flagfrac"] = loose, strict, np.round(chance, 3), np.round(frac, 3)
        if nP:
            d2 = up(diffs[fs if fs != "any" else "vv"])
            s1mean = ndimage.mean(np.nan_to_num(d2), lab, idx)
            prow[f"{fs}_sign_agree"] = np.sign(s1mean) == np.sign(mean)
        res[f"{fs}_flag_frac"] = round(float(fl[s1ok].mean()), 4)
        res[f"{fs}_fa_rate"] = round(float(fl[stable10].mean()), 4) if stable10.any() else np.nan
        for mp in (1, 20):
            n_r, prec, null = region_precision(fl, near10 & s1ok, mp, shifts)
            res[f"{fs}_nreg{mp}"], res[f"{fs}_regprec{mp}"], res[f"{fs}_regnull{mp}"] = n_r, round(prec, 3), round(null, 3)
        res[f"{fs}_recall_loose"] = round(float(np.mean(loose)), 3) if nP else np.nan
        if fs == "vv" and len(size2):
            l2, s2, c2, _ = score(lab2, len(size2), fl, s1ok, shifts)
            pd.DataFrame({"area_m2": size2 * CRES * CRES, "lr_mean_db": np.round(mean2, 2), "vv_loose": l2, "vv_strict": s2,
                          "vv_chance": np.round(c2, 3)}).to_csv(OUT / f"{tag}_patches_tol2.csv", index=False)
    res.update({"n_patches": nP, "patch_area_total_m2": int(size.sum() * 4), "stable10_frac": round(float(stable10.mean()), 3),
                "c_use_s1_frac": round(float(use.mean()), 3), "has_vh": "vh" in flags,
                "s1_items_before": ";".join(i.id for i in sb), "s1_items_after": ";".join(i.id for i in sa),
                "status": "ok", "secs": round(time.time() - t0)})
    pd.DataFrame(prow).to_csv(OUT / f"{tag}_patches.csv", index=False)
    np.savez_compressed(OUT / f"{tag}.npz", a_db=a_db.astype("float16"), b_db=b_db.astype("float16"),
                        lr=lr.astype("float16"), mg=mg.astype("float16"), lab=lab.astype("int32"), use=use, valid=valid,
                        s1b_vv=s1["vv"][0].astype("float16"), s1a_vv=s1["vv"][1].astype("float16"),
                        d_vv=diffs["vv"].astype("float16"), flag_vv=flags["vv"] & s1ok,
                        flag_vh=(flags["vh"] & s1ok) if "vh" in flags else np.zeros((N, N), bool),
                        s1ok=s1ok, stable10=stable10, wc=wc, epsg=epsg, x0=x0, y1=y1)
    return res, None, (proc.io_counters().read_bytes - io0)


def worker(pr, calib):
    try:
        r, extra, nb = run_pair(pr, calib)
    except Exception as e:
        traceback.print_exc()
        r, extra, nb = {"tag": f"site{pr['site_id']}_{pr['x1']}_{pr['x2']}", "status": "error",
                        "error": f"{type(e).__name__}: {e}"[:300]}, None, 0
    r["mb_read"] = round(BYTES["n"] / 1e6, 2)     # compressed TIFF blocks touched (re-reads counted again)
    r["mb_proc_io"] = round(nb / 1e6, 1)
    BYTES["n"] = 0
    if calib and extra is not None:
        np.savez_compressed(OUT / f"calib_{r['tag']}.npz", **extra)
    return {**pr, **r}


def main():
    import geopandas as gpd
    OUT.mkdir(parents=True, exist_ok=True)
    calib = "--calib" in sys.argv
    sel = pd.read_csv(ROOT / "r2_pairs.csv")
    foot = gpd.read_file(ROOT / "xband_collects.geojson").set_index("xband_id").geometry
    sel["centroid_wkt"] = [foot.loc[r.xid1].intersection(foot.loc[r.xid2]).centroid.wkt for r in sel.itertuples()]
    if calib:
        # quiet and busy pilot sites: Mojave, Kilauea, San Jose, Pilbara mine, Black Rock
        want = [32, 24, 31, 471, 0]
        sel = sel[sel.site_id.isin(want)].groupby("site_id").head(1)
    out_csv = ROOT / ("r2_calib_pairs.csv" if calib else "r2_pair_results.csv")
    rows = []
    with ProcessPoolExecutor(4) as ex:
        futs = [ex.submit(worker, r._asdict() if hasattr(r, "_asdict") else r, calib) for r in
                [dict(row) for _, row in sel.iterrows()]]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            keys = ("tag", "status", "secs", "mb_read", "src_res_m", "coreg_dy_px", "coreg_dx_px", "lr_mad_sd_db",
                    "lr_frac_gt3", "n_patches", "vv_flag_frac", "vv_fa_rate", "vv_recall_loose", "s1c_shift_px", "error")
            print({k: r.get(k) for k in keys if k in r}, flush=True)
            pd.DataFrame(rows).drop(columns=["centroid_wkt"], errors="ignore").to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    print(df.status.value_counts().to_dict(), "MB read total:", round(df.mb_read.sum(), 1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
