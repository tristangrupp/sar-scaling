"""Shared configuration and helpers for Experiment 2 (Sentinel-1 IW 10 m -> Stripmap 5 m).

Paths: everything is relative to this folder (ROOT) or to DATA, so the project can move to
E:\\SAR_superresolution by changing DATA only.
"""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
DATA = Path(os.environ.get("S1SM_DATA", ROOT / "data"))      # SLCs + built tiles
TILES = DATA / "tiles"                                              # per-tile npz
SM_TIF = DATA / "sm5"                                               # per-tile 5 m Stripmap GeoTIFFs
TILE_M = 3200
NL, NH = TILE_M // 10, TILE_M // 5                                  # 320 px at 10 m, 640 at 5 m

PY_SLC = r"C:\Users\grupp\miniconda3\envs\s1slc\python.exe"
PY_TOOLS = r"C:\Users\grupp\miniconda3\envs\ftwtools\python.exe"

# Region roles. Held-out regions never enter training, validation, X-band loss or tuning.
HOLDOUT = {"houston": "holdout_xband", "owens": "holdout_near", "w_australia": "holdout_far",
           "amieva_test": "holdout_paper"}
# The paper's split, read off its Fig. 3 map (red = test, yellow = validation, green = training).
AMIEVA_SPLIT = {
    "S1A_S6_SLC__1SDV_20150305T180517_20150305T180535_004900_0061C2_0639": "test",       # Dorset, UK
    "S1B_S4_SLC__1SDV_20170124T170703_20170124T170714_003993_006E40_5CC7": "test",       # Franconia, DE
    "S1B_S4_SLC__1SDV_20170217T170609_20170217T170638_004343_0078B1_883A": "test",       # Tyrol, AT
    "S1A_S4_SLC__1SDV_20150307T175057_20150307T175130_004929_006281_2458": "val",        # Caithness, UK
    "S1A_S4_SLC__1SDV_20160825T172419_20160825T172447_012760_014185_0179": "val",        # Liege, BE
    "S1A_S4_SLC__1SDV_20160825T172447_20160825T172512_012760_014185_344D": "train",
    "S1A_S3_SLC__1SDV_20160820T171644_20160820T171709_012687_013EFB_B8A2": "train",
    "S1A_S3_SLC__1SDV_20160820T171616_20160820T171644_012687_013EFB_B6D5": "train",
    "S1B_S3_SLC__1SDV_20171116T141835_20171116T141858_008308_00EB31_4212": "train",      # Iran
    "S1B_S6_SLC__1SDV_20170210T171448_20170210T171513_004241_00759F_21CB": "train",
}


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def tile_grid(lon, lat, tile_m=TILE_M):
    """Same snapping as process_slc.tile_grid: UTM, origin on a 40 m lattice."""
    from pyproj import Transformer
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    return epsg, float(np.floor((x - tile_m / 2) / 40) * 40), float(np.floor((y + tile_m / 2) / 40) * 40)


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


_client = None


def pc_client():
    global _client
    if _client is None:
        import planetary_computer as pc
        from pystac_client import Client
        _client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    return _client


def pc_item(collection, item_id):
    import planetary_computer as pc
    it = pc_client().get_collection(collection).get_item(item_id)
    return pc.sign(it)


def read_iw(item, epsg, x0, y1, n, res=10.0, pol="vv"):
    """IW RTC (gamma0, linear power) warped bilinearly onto a UTM grid; NaN outside the swath.
    item may be a list of frames of one pass; they are mosaicked, first valid value wins."""
    if isinstance(item, (list, tuple)):
        out = None
        for it in item:
            a = read_iw(it, epsg, x0, y1, n, res, pol)
            out = a if out is None else np.where(np.isfinite(out), out, a)
            if np.isfinite(out).all():
                break
        return out
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    with rasterio.open(item.assets[pol].href) as s, WarpedVRT(
            s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, res, res), width=n, height=n,
            resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
        a = v.read(1)
    a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
    return a


# ------------------------------------------------------------------ co-registration
def block_mean(a, k):
    n0, n1 = a.shape[0] // k, a.shape[1] // k
    b = a[:n0 * k, :n1 * k].reshape(n0, k, n1, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def _fill(a):
    a = a.copy()
    a[~np.isfinite(a)] = np.nanmedian(a)
    return a


def coarse_offset(sm10_db, iw_big_db, margin):
    """Integer + parabolic subpixel shift (rows, cols in 10 m px) of SM inside a padded IW window.

    Returns (dr, dc, peak_ncc): the SM tile matches IW at nominal + (dr, dc).
    """
    from skimage.feature import match_template
    r = match_template(_fill(iw_big_db), _fill(sm10_db))
    i, j = np.unravel_index(np.argmax(r), r.shape)

    def para(v_1, v0, v1):
        d = v_1 - 2 * v0 + v1
        return 0.0 if d == 0 else 0.5 * (v_1 - v1) / d

    di = para(r[i - 1, j], r[i, j], r[i + 1, j]) if 0 < i < r.shape[0] - 1 else 0.0
    dj = para(r[i, j - 1], r[i, j], r[i, j + 1]) if 0 < j < r.shape[1] - 1 else 0.0
    return i + di - margin, j + dj - margin, float(r[i, j])


def fine_shift(sm5_db, iw5_db):
    """Subpixel (rows, cols) 5 m px shift that moves IW onto SM (phase correlation, smoothed SM)."""
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation
    a = ndimage.gaussian_filter(_fill(sm5_db), 1.0)
    b = _fill(iw5_db)
    a, b = a - a.mean(), b - b.mean()
    s, _, _ = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)
    return s


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ------------------------------------------------------------------ speckle filtering
def local_enl(I, size=7):
    from scipy import ndimage
    I = np.nan_to_num(I, nan=np.nanmedian(I))
    m = ndimage.uniform_filter(I, size)
    v = ndimage.uniform_filter(I * I, size) - m * m
    return m * m / np.maximum(v, 1e-12)


def estimate_looks(I):
    """Equivalent number of looks of a linear-power image: the 90th percentile of 7x7 local ENL,
    i.e. the ENL of its most homogeneous areas (texture lowers ENL elsewhere)."""
    e = local_enl(I)
    return float(np.clip(np.nanpercentile(e[3:-3, 3:-3], 90), 1.0, 30.0))


def lee_filter(I, size=7, looks=None):
    """Classic Lee filter on linear power (as SNAP's Lee filter); NaN preserved."""
    from scipy import ndimage
    nan = ~np.isfinite(I)
    J = np.where(nan, np.nanmedian(I), I).astype("float64")
    L = looks or estimate_looks(J)
    m = ndimage.uniform_filter(J, size)
    v = np.maximum(ndimage.uniform_filter(J * J, size) - m * m, 0)
    cu2 = 1.0 / L
    ci2 = v / np.maximum(m * m, 1e-12)
    w = np.clip(1 - cu2 / np.maximum(ci2, 1e-12), 0, 1)
    out = m + w * (J - m)
    out[nan] = np.nan
    return out.astype("float32"), L


# ------------------------------------------------------------------ histogram matching
def hist_match(src, ref):
    """Map src values so their distribution matches ref's (both may contain NaN)."""
    out = np.full_like(src, np.nan, dtype="float32")
    m = np.isfinite(src)
    r = np.sort(ref[np.isfinite(ref)])
    if m.sum() == 0 or r.size == 0:
        return out
    ranks = np.argsort(np.argsort(src[m]))
    q = (ranks + 0.5) / m.sum()
    out[m] = np.quantile(r, q)
    return out


def read_worldcover(epsg, x0, y1, n, res):
    """ESA WorldCover 2021 classes (nearest) on a UTM grid; 0 = no data (open ocean)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds
    b = transform_bounds(f"EPSG:{epsg}", 4326, x0, y1 - n * res, x0 + n * res, y1)
    out = np.zeros((n, n), "uint8")
    items = pc_client().search(collections=["esa-worldcover"], bbox=list(b),
                               query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items()
    for it in items:
        with rasterio.open(it.assets["map"].href) as s, WarpedVRT(
                s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, res, res), width=n, height=n,
                resampling=Resampling.nearest, nodata=0) as v:
            a = v.read(1)
        out = np.where(out == 0, a, out)
    return out
