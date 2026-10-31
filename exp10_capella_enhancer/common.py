"""Experiment 10 shared pieces: paths, grids, readers, the degradation PSF, masks, metrics and baselines.

Everything here runs on CPU. train.py / eval.py import the metric and model pieces.
"""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
DATA = Path(os.environ.get("EXP10_DATA", HERE / "data"))
CHIPS = DATA / "chips"
EXP6 = PROJ / "exp6_capella_labels"
EXP8 = PROJ / "exp8_feature_width"

# ------------------------------------------------------------------ grids
S1_RES = 10.0
NS = 64                    # Sentinel-1 chip size (px)
CHIP_M = NS * S1_RES       # 640 m
BASE_RES = 1.25            # Capella base grid
OUT_RES = 2.5              # target / output grid
SC = int(S1_RES / OUT_RES)  # 4
NT = NS * SC               # 256
MARGIN_M = 50.0            # Capella base window margin (fine-shift search + blur support)
MB = int(MARGIN_M / BASE_RES)  # 40 base px
NB = int((CHIP_M + 2 * MARGIN_M) / BASE_RES)  # 592 base px
MAX_SRC_RES = 1.25
RES_LIST = (2.5, 5.0, 7.5, 10.0)   # target effective resolutions (m); 2.5 = stretch arm, no added blur
BORDER = 4                 # S1 px dropped on each side when scoring
SLOPE_MAX = 10.0
MIN_VALID = 0.8
COARSE_BIG = 1500.0
COARSE_Z = 6.0
FINE_R = 10.0
FINE_STEP = 2.5
ITEM_R_MIN = 0.15
CHIP_GAIN = 0.03
CHIP_R_MIN = 0.10
WC_NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water",
            90: "wetland", 95: "mangrove", 100: "moss"}


def finish():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


# ------------------------------------------------------------------ byte accounting (compressed COG blocks touched)
class Bytes:
    def __init__(self):
        self.seen = set()
        self.n = 0

    def add(self, ds, bounds, epsg, key):
        """bounds (x0, y0, x1, y1) in EPSG:epsg; adds the compressed sizes of the blocks a read touches, once each."""
        try:
            from rasterio.warp import transform_bounds
            from rasterio.windows import from_bounds
            b = transform_bounds(f"EPSG:{epsg}", ds.crs, *bounds, densify_pts=21) if ds.crs.to_epsg() != epsg else bounds
            w = from_bounds(*b, transform=ds.transform)
            bh, bw = ds.block_shapes[0]
            r0, r1 = max(int(w.row_off // bh), 0), min(int((w.row_off + w.height) // bh), (ds.height - 1) // bh)
            c0, c1 = max(int(w.col_off // bw), 0), min(int((w.col_off + w.width) // bw), (ds.width - 1) // bw)
            for i in range(r0, r1 + 1):
                for j in range(c0, c1 + 1):
                    k = (key, i, j)
                    if k in self.seen:
                        continue
                    self.seen.add(k)
                    try:
                        self.n += int(ds.block_size(1, i, j))
                    except Exception:
                        self.n += bh * bw * np.dtype(ds.dtypes[0]).itemsize
        except Exception:
            pass


# ------------------------------------------------------------------ Planetary Computer
_PC = {}


def pc_client():
    import planetary_computer as pc
    from pystac_client import Client
    if "c" not in _PC:
        _PC["c"] = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    return _PC["c"]


def pc_search(**kw):
    for k in range(5):
        try:
            return list(pc_client().search(**kw).items())
        except Exception as ex:
            print("  pc search retry", k, str(ex)[:120], flush=True)
            _PC.clear()
            time.sleep(5 * (k + 1))
    return []


def warp_read(href, epsg, x0, y1, res, n_rows, n_cols, resampling, nodata, acc=None, key=None, band=1):
    """Read href onto a north-up grid (origin x0, y1, pixel res) in EPSG:epsg; returns float32 with NaN for nodata."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    rs = getattr(Resampling, resampling)
    for k in range(4):
        try:
            with rasterio.open(href) as s:
                if acc is not None:
                    acc.add(s, (x0 - 2 * res, y1 - n_rows * res - 2 * res, x0 + n_cols * res + 2 * res, y1 + 2 * res), epsg,
                            key or href)
                with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, res, res), width=n_cols,
                               height=n_rows, resampling=rs, src_nodata=nodata, nodata=nodata, dtype="float32") as v:
                    a = v.read(band)
            a[(a == nodata) | ~np.isfinite(a)] = np.nan
            return a
        except Exception as ex:
            print("  read retry", k, str(ex)[:160], flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"read failed {href[:120]}")


def s1_pass_groups(items):
    """Group RTC items into passes (same platform, relative orbit, direction, date)."""
    import pandas as pd
    g = {}
    for it in items:
        p = it.properties
        t = pd.Timestamp(it.datetime).tz_localize(None)
        key = (p.get("platform"), p.get("sat:relative_orbit"), p.get("sat:orbit_state"), t.date())
        g.setdefault(key, []).append(it)
    return g


def read_s1_pass(items, pol, epsg, x0, y1, n_rows, n_cols, acc=None):
    out = None
    for it in items:
        if pol not in it.assets:
            continue
        a = warp_read(it.assets[pol].href, epsg, x0, y1, S1_RES, n_rows, n_cols, "bilinear", -32768.0, acc,
                      key=it.id + pol)
        a[a <= 0] = np.nan
        out = a if out is None else np.where(np.isfinite(out), out, a)
        if np.isfinite(out).all():
            break
    return out


def read_dem_slope(epsg, x0, y1, n_rows, n_cols, bbox_ll, acc=None):
    """Copernicus GLO-30 heights bilinear to the 10 m grid and slope in degrees."""
    its = pc_search(collections=["cop-dem-glo-30"], bbox=bbox_ll)
    z = None
    for it in its:
        a = warp_read(it.assets["data"].href, epsg, x0, y1, S1_RES, n_rows, n_cols, "bilinear", -32767.0, acc,
                      key=it.id)
        z = a if z is None else np.where(np.isfinite(z), z, a)
    if z is None:
        return None, None
    zz = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0)
    gy, gx = np.gradient(zz, S1_RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    slope[~np.isfinite(z)] = np.nan
    return z, slope


def read_worldcover(epsg, x0, y1, n_rows, n_cols, bbox_ll, acc=None):
    its = pc_search(collections=["esa-worldcover"], bbox=bbox_ll,
                    query={"esa_worldcover:product_version": {"eq": "2.0.0"}})
    out = np.zeros((n_rows, n_cols), "uint8")
    for it in its:
        a = warp_read(it.assets["map"].href, epsg, x0, y1, S1_RES, n_rows, n_cols, "nearest", 0.0, acc, key=it.id)
        a = np.nan_to_num(a, nan=0).astype("uint8")
        out = np.where(out == 0, a, out)
    return out


# ------------------------------------------------------------------ Capella
def capella_meta(href):
    """CRS, overview choice, scale factor and resolutions from the GEO COG header."""
    import rasterio
    with rasterio.open(href) as s:
        native = abs(s.res[0]) * (111_000 if s.crs.is_geographic else 1)
        ovs = s.overviews(1)
        desc = s.tags().get("TIFFTAG_IMAGEDESCRIPTION", "{}")
        bounds, crs = s.bounds, s.crs.to_string()
    try:
        meta = json.loads(desc)
    except Exception:
        meta = {}
    img = meta.get("collect", {}).get("image", {})
    lvl, src = None, native
    for i, f in enumerate(ovs):
        if native * f <= MAX_SRC_RES:
            lvl, src = i, native * f
    return {"native_px": native, "ov_level": lvl, "src_px": src, "overviews": ovs,
            "scale_factor": float(img.get("scale_factor", 1.0) or 1.0),
            "radiometry": img.get("radiometry", ""),
            "gr_res": float(img.get("ground_range_resolution", np.nan) or np.nan),
            "az_res": float(img.get("azimuth_resolution", np.nan) or np.nan),
            "inc": float(img.get("center_pixel", {}).get("incidence_angle", np.nan) or np.nan),
            "bounds": tuple(bounds), "crs": crs}


def capella_read(href, cm, epsg, x0, y1, res, n, acc=None, level=None):
    """Capella sigma0 intensity on an n x n grid (origin x0, y1, pixel res) in EPSG:epsg.
    Reads the source (overview `level`, default the one chosen in cm) by nearest at its own pixel size into the
    target CRS, squares DN * scale to intensity, then averages onto the grid. NaN = no data."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import reproject
    lvl = cm["ov_level"] if level is None else level
    src_px = cm["native_px"] * (cm["overviews"][lvl] if lvl is not None else 1)
    pad = 3 * src_px
    m = int(np.ceil((n * res + 2 * pad) / src_px))
    sx0, sy1 = x0 - pad, y1 + pad
    kw = {"overview_level": lvl} if lvl is not None else {}
    for k in range(4):
        try:
            with rasterio.open(href, **kw) as s:
                if acc is not None:
                    acc.add(s, (sx0, sy1 - m * src_px, sx0 + m * src_px, sy1), epsg, (href, lvl))
                with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(sx0, sy1, src_px, src_px), width=m,
                               height=m, resampling=Resampling.nearest, src_nodata=0, nodata=0) as v:
                    a = v.read(1).astype("float32")
            break
        except Exception as ex:
            print("  capella retry", k, str(ex)[:160], flush=True)
            time.sleep(5 * (k + 1))
    else:
        raise RuntimeError("capella read failed")
    a[a <= 0] = np.nan
    a = (a * cm["scale_factor"]) ** 2
    out = np.full((n, n), np.nan, "float32")
    reproject(a, out, src_transform=from_origin(sx0, sy1, src_px, src_px), src_crs=f"EPSG:{epsg}",
              dst_transform=from_origin(x0, y1, res, res), dst_crs=f"EPSG:{epsg}", resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan)
    return out


# ------------------------------------------------------------------ PSF budget
def _fwhm_1d(native_fwhm, src_px, sigma, dx=0.02):
    """FWHM (m) of native Gaussian IRF * box(src_px) * box(BASE_RES) * Gaussian(sigma) * box(OUT_RES)."""
    x = np.arange(-40, 40 + dx / 2, dx)
    sn = native_fwhm / 2.3548
    k = np.exp(-x ** 2 / (2 * sn ** 2))
    for w in (src_px, BASE_RES, OUT_RES):
        b = (np.abs(x) <= w / 2).astype(float)
        k = np.convolve(k, b / b.sum(), mode="same")
    if sigma > 0:
        g = np.exp(-x ** 2 / (2 * sigma ** 2))
        k = np.convolve(k, g / g.sum(), mode="same")
    k = k / k.max()
    above = np.nonzero(k >= 0.5)[0]
    i0, i1 = above[0], above[-1]
    # linear interpolation at both crossings
    left = x[i0 - 1] + (0.5 - k[i0 - 1]) / (k[i0] - k[i0 - 1]) * dx
    right = x[i1] + (k[i1] - 0.5) / (k[i1] - k[i1 + 1]) * dx
    return float(right - left)


def solve_sigma(R, native_fwhm, src_px):
    """Gaussian sigma (m) so that the chain FWHM equals R; 0 if the chain is already at least R."""
    if _fwhm_1d(native_fwhm, src_px, 0.0) >= R:
        return 0.0
    lo, hi = 0.0, R
    for _ in range(40):
        mid = (lo + hi) / 2
        if _fwhm_1d(native_fwhm, src_px, mid) < R:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def nominal_looks(sigma, native_cell_m2, src_px):
    """Kernel area (sum w)^2 / sum w^2 in m^2 over the native resolution cell area: speckle looks of the target."""
    from scipy import ndimage
    x = np.arange(-30, 30.001, 0.25)
    X, Y = np.meshgrid(x, x)
    k = ((np.abs(X) <= OUT_RES / 2) & (np.abs(Y) <= OUT_RES / 2)).astype(float)   # the 2.5 m output box
    if sigma > 0:
        k = ndimage.gaussian_filter(k, sigma / 0.25)
    area = k.sum() ** 2 / (k ** 2).sum() * 0.25 ** 2
    # an overview coarser than the native cell keeps about one independent sample per source pixel
    return float(max(1.0, area / max(native_cell_m2, src_px ** 2, 1e-6)))


def degrade(base, sigma):
    """Normalized Gaussian convolution of base-grid intensity (NaN = no data); returns (filtered, weight)."""
    from scipy import ndimage
    v = np.isfinite(base)
    b0 = np.where(v, base, 0).astype("float64")
    if sigma <= 0:
        return np.where(v, base, np.nan), v.astype("float32")
    s = sigma / BASE_RES
    num = ndimage.gaussian_filter(b0, s, truncate=3.5)
    den = ndimage.gaussian_filter(v.astype("float64"), s, truncate=3.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return out.astype("float32"), den.astype("float32")


def block_mean(a, k):
    h, w = a.shape[0] // k, a.shape[1] // k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:h * k, :w * k].reshape(h, k, w, k), axis=(1, 3))


def acf_fwhm(img_lin, res):
    """Empirical resolution check: FWHM of the autocorrelation of normalized intensity fluctuations / sqrt(2) (m).
    Real texture widens it, so this is an upper bound on the PSF width."""
    from scipy import ndimage
    a = np.where(np.isfinite(img_lin), img_lin, np.nan)
    if np.isfinite(a).mean() < 0.9:
        return np.nan
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    lm = ndimage.uniform_filter(a, int(max(9, 6 * 10 / res)))
    f = a / np.maximum(lm, 1e-12) - 1
    f = f - f.mean()
    F = np.fft.fft2(f)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = ac / ac[0, 0]
    prof = (ac[0, :40] + ac[:40, 0]) / 2
    j = np.nonzero(prof < 0.5)[0]
    if len(j) == 0 or j[0] == 0:
        return np.nan
    j = j[0]
    hw = (j - 1) + (prof[j - 1] - 0.5) / (prof[j - 1] - prof[j])
    return float(2 * hw * res / np.sqrt(2))


# ------------------------------------------------------------------ alignment helpers
def highpass(a_db, w=5):
    from scipy import ndimage
    m = np.isfinite(a_db)
    num = ndimage.uniform_filter(np.where(m, a_db, 0), w)
    den = ndimage.uniform_filter(m.astype(float), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        hp = a_db - num / den
    hp[~m | (den < 0.5)] = np.nan
    return hp


def corr(a, b, nmin=400):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < nmin:
        return -1.0
    x, y = a[m], b[m]
    if x.std() == 0 or y.std() == 0:
        return -1.0
    return float(np.corrcoef(x, y)[0, 1])


# ------------------------------------------------------------------ chips
def load_meta():
    import pandas as pd
    return pd.read_csv(HERE / "chips.csv")


def load_item(cid):
    z = np.load(CHIPS / f"{cid}.npz")
    return {k: z[k] for k in z.files}


def res_index(R):
    return RES_LIST.index(float(R))


# ------------------------------------------------------------------ upsampling
def bicubic_up(a10, k=SC):
    """Bicubic upsampling of a 10 m array (dB) by k (area-aligned, align_corners=False). NaN filled by the median."""
    import torch
    import torch.nn.functional as F
    a = np.where(np.isfinite(a10), a10, np.nanmedian(a10) if np.isfinite(a10).any() else 0).astype("float32")
    t = torch.from_numpy(a)[None, None]
    return F.interpolate(t, scale_factor=k, mode="bicubic", align_corners=False)[0, 0].numpy()


# ------------------------------------------------------------------ metrics
def _std(a, m):
    v = a[m]
    return (a - v.mean()) / (v.std() + 1e-9)


def gauss_fwhm(a, fwhm_px):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, fwhm_px / 2.3548, mode="reflect")


def hp_layer(a, m):
    """Image minus its Gaussian blur of 20 m FWHM (normalized for the mask)."""
    from scipy import ndimage
    s = 20.0 / OUT_RES / 2.3548
    num = ndimage.gaussian_filter(np.where(m, a, 0), s, mode="reflect")
    den = ndimage.gaussian_filter(m.astype(float), s, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        return a - num / np.maximum(den, 1e-6)


def detail_layer(a):
    k = SC
    n = a.shape[0] // k
    lo = a.reshape(n, k, n, k).mean(axis=(1, 3))
    return a - np.kron(lo, np.ones((k, k)))


def edges(a, q=0.90):
    from scipy import ndimage
    s = ndimage.gaussian_filter(a, 1.0)
    g = np.hypot(ndimage.sobel(s, 0), ndimage.sobel(s, 1))
    return g


def scores(pred, tgt, m, flat=None, light=False):
    """pred, tgt: dB on the 2.5 m chip (central crop applied by caller), m: valid mask. Returns a dict.
    light=True returns SSIM and hp r only (training-time validation)."""
    from scipy import ndimage
    from skimage.metrics import structural_similarity
    if m.sum() < 500:
        return None
    p = np.where(m, pred, np.nan)
    t = np.where(m, tgt, np.nan)
    pz, tz = _std(np.nan_to_num(p), m), _std(np.nan_to_num(t), m)
    pz[~m] = 0
    tz[~m] = 0
    lo, hi = np.percentile(tz[m], [1, 99])
    span = hi - lo
    rmse = float(np.sqrt(np.mean((pz[m] - tz[m]) ** 2)))
    psnr = float(20 * np.log10(span / max(rmse, 1e-9)))
    _, S = structural_similarity(np.clip(tz, lo, hi), np.clip(pz, lo, hi), data_range=span, full=True)
    ssim = float(S[m].mean())
    if light:
        me = ndimage.binary_erosion(m, iterations=2)
        hp_p, hp_t = hp_layer(pz, m), hp_layer(tz, m)
        return {"ssim": ssim, "hp_r": float(np.corrcoef(hp_p[me], hp_t[me])[0, 1]) if hp_p[me].std() > 0 else 0.0}
    # detail (exp1): minus the 10 m block mean, on 4x4 blocks fully valid
    k = SC
    m4 = m.reshape(m.shape[0] // k, k, -1, k).all(axis=(1, 3))
    mm = np.kron(m4, np.ones((k, k), bool))
    dp, dt = detail_layer(pz)[mm], detail_layer(tz)[mm]
    detail_r = float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 and dt.std() > 0 else 0.0
    hp_p, hp_t = hp_layer(pz, m), hp_layer(tz, m)
    me = ndimage.binary_erosion(m, iterations=2)
    hp_r = float(np.corrcoef(hp_p[me], hp_t[me])[0, 1]) if hp_p[me].std() > 0 else 0.0
    hf_ratio = float(hp_p[me].std() / max(hp_t[me].std(), 1e-9))
    # edge F1 with 2 px tolerance
    gp, gt = edges(pz), edges(tz)
    ep = (gp >= np.percentile(gp[me], 90)) & me
    et = (gt >= np.percentile(gt[me], 90)) & me
    dt_t = ndimage.distance_transform_edt(~et)
    dt_p = ndimage.distance_transform_edt(~ep)
    prec = float((dt_t[ep] <= 2).mean()) if ep.any() else 0.0
    rec = float((dt_p[et] <= 2).mean()) if et.any() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    # gradient magnitude similarity (Xue et al. 2014) on images scaled to [0, 1] by the target span
    sp, st = (np.clip(pz, lo, hi) - lo) / span, (np.clip(tz, lo, hi) - lo) / span
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    gmp = np.hypot(ndimage.convolve(sp, kx), ndimage.convolve(sp, kx.T))
    gmt = np.hypot(ndimage.convolve(st, kx), ndimage.convolve(st, kx.T))
    c = 0.0026
    gms = (2 * gmp * gmt + c) / (gmp ** 2 + gmt ** 2 + c)
    out = {"psnr": psnr, "ssim": ssim, "detail_r": detail_r, "hp_r": hp_r, "hf_ratio": hf_ratio,
           "edge_f1": float(f1), "gms": float(gms[me].mean())}
    if flat is not None:
        f = flat & me
        if f.sum() > 100:
            out["flat_ratio"] = float(np.sqrt(np.mean(hp_p[f] ** 2)) / max(np.sqrt(np.mean(hp_t[f] ** 2)), 1e-9))
    return out


def flat_mask(tgt_db, m, q=20):
    """Flattest q% of valid target pixels by local 5x5 standard deviation."""
    from scipy import ndimage
    if m.sum() < 100:
        return np.zeros_like(m)
    a = np.where(m, tgt_db, np.nanmean(tgt_db[m]))
    mu = ndimage.uniform_filter(a, 5)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(a * a, 5) - mu * mu, 0))
    thr = np.percentile(sd[m], q)
    return (sd <= thr) & m


def crop_c(a, k=SC):
    """Central crop dropping BORDER S1 px (BORDER*k output px) on each side."""
    b = BORDER * k
    return a[..., b:a.shape[-2] - b, b:a.shape[-1] - b]


# ------------------------------------------------------------------ baselines
LEE_L = 4.4


def lee(I, L=LEE_L, w=7):
    from scipy import ndimage
    I = np.where(np.isfinite(I), I, np.nanmedian(I)).astype("float64")
    mu = ndimage.uniform_filter(I, w, mode="reflect")
    v = np.maximum(ndimage.uniform_filter(I * I, w, mode="reflect") - mu * mu, 0)
    cu2 = 1.0 / L
    var_x = np.maximum((v - mu * mu * cu2) / (1 + cu2), 0)
    k = np.clip(var_x / np.maximum(v, 1e-12), 0, 1)
    return mu + k * (I - mu)


_N2N = {}


def n2n_model():
    """Experiment 4's r2_n2n_big_cond_vvvh despeckler, on CPU."""
    if "G" in _N2N:
        return _N2N["G"], _N2N["meta"]
    import torch
    import torch.nn as nn

    class RB(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

        def forward(self, x):
            return x + 0.2 * self.b(x)

    class DGenC(nn.Module):
        def __init__(self, cin=1, cout=1, c=64, nb=16):
            super().__init__()
            self.cout = cout
            self.head = nn.Conv2d(cin, c, 3, padding=1)
            self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
            self.tail = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, cout, 3, padding=1))

        def forward(self, s):
            h = self.head(s)
            h = h + self.body(h)
            return s[:, :self.cout] + self.tail(h)

    ck = PROJ / "exp4_despeckling" / "ckpt"
    meta = json.loads((ck / "r2_n2n_big_cond_vvvh.json").read_text())
    C = len(meta["pols"])
    G = DGenC(cin=C * 2, cout=C).eval()
    G.load_state_dict(torch.load(ck / "r2_n2n_big_cond_vvvh.pt", map_location="cpu", weights_only=True))
    _N2N["G"], _N2N["meta"] = G, meta
    return G, meta


def enl_hat(lin, w=7, q=90):
    from scipy import ndimage
    a = np.where(np.isfinite(lin), lin, np.nanmedian(lin)).astype("float64")
    m = ndimage.uniform_filter(a, w)
    v = np.maximum(ndimage.uniform_filter(a * a, w) - m * m, 1e-20)
    e = (m * m / v)[w // 2::w, w // 2::w]
    return float(np.clip(np.percentile(e, q), 1, 500))


def n2n_apply(vv_lin, vh_lin):
    """Despeckled VV (linear) from VV+VH linear inputs, exp4's build_input/apply_model on CPU."""
    import torch
    G, meta = n2n_model()
    xs, conds, stats = [], [], {}
    for p, I in (("vv", vv_lin), ("vh", vh_lin)):
        I = np.where(np.isfinite(I), I, np.nanmedian(I))
        d = 10 * np.log10(np.maximum(I, 1e-6))
        v = d[np.isfinite(d)]
        med, sd = float(np.median(v)), float(v.std() + 1e-6)
        stats[p] = (med, sd)
        xs.append((d - med) / sd)
        conds.append(np.full(d.shape, np.log(enl_hat(I)) - np.log(10.0)))
    x = torch.from_numpy(np.stack(xs + conds).astype("float32"))[None]
    with torch.no_grad():
        y = G(x)[0].numpy()
    med, sd = stats["vv"]
    return 10 ** ((y[0] * sd + med + meta.get("offset", {}).get("vv", 0.0)) / 10)


_ESRT = {}


def esrt_exp2():
    if "m" in _ESRT:
        return _ESRT["m"], _ESRT["norm"]
    import torch
    sys.path.insert(0, str(PROJ / "exp2_iw_to_stripmap"))
    from esrt import ESRT
    ck = torch.load(PROJ / "exp2_iw_to_stripmap" / "ckpt" / "esrt_v2_nomask_s0.pt", map_location="cpu",
                    weights_only=False)
    state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
    m = ESRT(upscale=2).eval()
    m.load_state_dict(state)
    norm = ck.get("norm") if isinstance(ck, dict) else None
    _ESRT["m"], _ESRT["norm"] = m, norm
    return m, norm


def esrt_apply(vv_db):
    """exp2 ESRT (x2 to 5 m) on single-date VV dB, then bicubic x2 to 2.5 m. Returns dB at 2.5 m."""
    import torch
    m, norm = esrt_exp2()
    lo, hi = norm
    a = np.where(np.isfinite(vv_db), vv_db, np.nanmedian(vv_db))
    x = torch.from_numpy(((a - lo) / (hi - lo)).astype("float32"))[None, None]
    with torch.no_grad():
        y = m(x)[0, 0].numpy() * (hi - lo) + lo
    return bicubic_up(y, 2)


BASELINES = ("bicubic_single", "bicubic_multi", "lee_single", "n2n_single", "n2n_multi", "esrt_exp2")


def baseline_preds(s1):
    """s1: (5, 64, 64) dB [VV1, VH1, VVm, VHm, slope]. Returns {name: 256x256 dB}."""
    vv1, vh1, vvm, vhm = s1[0].astype("float64"), s1[1].astype("float64"), s1[2].astype("float64"), s1[3].astype("float64")
    lin = lambda d: 10 ** (d / 10)  # noqa: E731
    out = {"bicubic_single": bicubic_up(vv1), "bicubic_multi": bicubic_up(vvm),
           "lee_single": bicubic_up(db(lee(lin(vv1)))),
           "n2n_single": bicubic_up(db(n2n_apply(lin(vv1), lin(vh1)))),
           "n2n_multi": bicubic_up(db(n2n_apply(lin(vvm), lin(vhm))))}
    try:
        out["esrt_exp2"] = esrt_apply(vv1)
    except Exception as ex:  # keep going without it
        print("esrt_exp2 failed:", str(ex)[:200], flush=True)
    return out


def shuffle_partners(ch, seed=0):
    """For each chip, a random chip of another site in the same split: {chip_id: (capella_id, chip_id)}.
    Shared by baselines.py and eval.py so the shuffled-input floor and the model's shuffled check pair alike."""
    out = {}
    for split, g in ch.sort_values("chip_id").groupby("split"):
        rng = np.random.default_rng([seed, {"train": 0, "val": 1, "test": 2}.get(split, 3)])
        ids = g[["chip_id", "capella_id", "site_id"]].values
        if g.site_id.nunique() < 2:
            continue
        for cid_, cap, site in ids:
            while True:
                k = ids[rng.integers(len(ids))]
                if k[2] != site:
                    out[cid_] = (k[1], k[0])
                    break
    return out
