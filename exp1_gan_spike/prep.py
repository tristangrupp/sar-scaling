"""Build paired tiles: Sentinel-1 RTC VV (10 m) <-> X-band VV (2.5 m), same UTM grid.

For each selected Umbra / ICEYE spotlight collect:
  - define a 3.2 km square tile at the footprint centroid, in local UTM, snapped to 40 m
  - X-band: warp to 2.5 m with average resampling (Umbra GEC via overview, ICEYE GRD via GCPs)
  - S1: nearest-date sentinel-1-rtc VV from Planetary Computer (+/- 30 d), warped to 10 m
  - co-register: phase correlation of X-band (block-averaged to 10 m) against S1; apply shift
Saves one .npz per tile in data/ and a manifest.csv.

Run in background with stdout to a file (this env's GDAL hangs foreground pipes on Windows).
"""
from __future__ import annotations

import os

os.environ.update(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                  GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

import argparse
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import requests
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
TILE_M, HR, LR = 3200, 2.5, 10.0
NH, NL = int(TILE_M / HR), int(TILE_M / LR)           # 1280, 320
UMBRA_BUCKET = "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog"

# Held-out test regions (lon_min, lat_min, lon_max, lat_max)
TEST_BOXES = {
    "houston": (-95.7, 29.4, -94.7, 30.1), "newark_ny": (-74.35, 40.45, -73.7, 40.95),
    "tokyo": (139.4, 35.3, 140.2, 35.95), "antwerp": (4.0, 51.1, 4.6, 51.45),
    "rotterdam": (3.9, 51.75, 4.7, 52.05),
}


def test_region(pt):
    for name, (a, b, c, d) in TEST_BOXES.items():
        if a <= pt.x <= c and b <= pt.y <= d:
            return name
    return None


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


# ------------------------------------------------------------ selection
def select(n_umbra: int, n_iceye: int, seed: int) -> pd.DataFrame:
    um = gpd.read_file(PROJ / "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson")
    ic = gpd.read_file(PROJ / "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson")
    um["provider"], ic["provider"] = "umbra", "iceye"
    cols = ["xband_id", "provider", "datetime", "sar:polarizations", "sar:instrument_mode",
            "view:incidence_angle", "geometry"]
    df = pd.concat([um[cols], ic[cols + ["grd_href"]]], ignore_index=True)
    df = gpd.GeoDataFrame(df, crs="EPSG:4326")
    df = df[(df["sar:polarizations"] == "VV") & (df["sar:instrument_mode"].str.lower() == "spotlight")]
    c = df.geometry.centroid
    df["lon"], df["lat"] = c.x, c.y
    df["test_region"] = [test_region(p) for p in c]
    df["cell"] = (df.lon.round(1).astype(str) + "_" + df.lat.round(1).astype(str))

    test = df[df.test_region.notna()]
    train = df[df.test_region.isna()].sample(frac=1, random_state=seed)
    train = train.drop_duplicates(["provider", "cell"])          # spread over places, one per 0.1 deg cell
    train = pd.concat([train[train.provider == "umbra"].head(n_umbra),
                       train[train.provider == "iceye"].head(n_iceye)])
    test = test.sample(frac=1, random_state=seed).groupby("provider").head(40)
    out = pd.concat([train.assign(split="train"), test.assign(split="test")])
    return out.drop(columns="geometry").reset_index(drop=True)


def umbra_gec_hrefs(ids: set[str]) -> dict[str, str]:
    """Map Umbra item id -> GEC COG href by listing STAC keys and fetching the needed items."""
    s = requests.Session()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, token = {}, None
    pat = re.compile(r"^stac/.+/([0-9a-f-]{36})/\1\.json$")
    while True:
        params = {"list-type": "2", "prefix": "stac/"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(s.get(UMBRA_BUCKET, params=params, timeout=60).content)
        for k in root.findall("s3:Contents/s3:Key", ns):
            m = pat.match(k.text)
            if m and m.group(1) in ids:
                keys[m.group(1)] = k.text
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)

    def gec(item_id):
        it = s.get(f"{UMBRA_BUCKET}/{keys[item_id]}", timeout=60).json()
        # newer items name it *_GEC.tif; 2023 items use a bare <timestamp>_UMBRA-NN.tif
        tifs = [a["href"] for a in it["assets"].values()
                if a["href"].endswith(".tif") and "_CSI" not in a["href"]]
        tifs.sort(key=lambda h: not h.endswith("_GEC.tif"))
        for h in tifs[:1]:
                base, path = h.split(".amazonaws.com/", 1)
                return item_id, base + ".amazonaws.com/" + quote(path)
        return item_id, None

    with ThreadPoolExecutor(16) as ex:
        return dict(ex.map(gec, list(keys)))


# ------------------------------------------------------------ per tile
def tile_grid(lon, lat):
    epsg = utm_epsg(lon, lat)
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    x0 = np.floor((x - TILE_M / 2) / 40) * 40
    y1 = np.floor((y + TILE_M / 2) / 40) * 40
    return epsg, x0, y1


def warp(src, epsg, x0, y1, res, n, resampling=Resampling.average):
    tr = from_origin(x0, y1, res, res)
    kw = dict(crs=f"EPSG:{epsg}", transform=tr, width=n, height=n, resampling=resampling,
              nodata=0, dtype="float32")
    if src.crs is None:      # ICEYE GRD: geolocated by GCPs only (WarpedVRT ignores them)
        return warp_gcps(src, epsg, tr, n, resampling)
    with WarpedVRT(src, **kw) as v:
        return v.read(1)


def warp_gcps(src, epsg, tr, n, resampling):
    """Read only the source window under the tile, then reproject with window-shifted GCPs."""
    from pyproj import Transformer
    from rasterio.control import GroundControlPoint
    from rasterio.warp import reproject
    g, gcrs = src.gcps
    fit = rasterio.transform.from_gcps(g)                    # affine approx, only to find the window
    to_geo = Transformer.from_crs(epsg, gcrs, always_xy=True)
    xs = [tr.c, tr.c + tr.a * n]
    ys = [tr.f, tr.f + tr.e * n]
    lon, lat = to_geo.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    rows, cols = rasterio.transform.rowcol(fit, lon, lat)
    pad = 300
    r0, r1 = max(min(rows) - pad, 0), min(max(rows) + pad, src.height)
    c0, c1 = max(min(cols) - pad, 0), min(max(cols) + pad, src.width)
    if r1 <= r0 or c1 <= c0:
        return np.zeros((n, n), "float32")
    a = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)).astype("float32")
    g2 = [GroundControlPoint(row=p.row - r0, col=p.col - c0, x=p.x, y=p.y, z=p.z) for p in g]
    dst = np.zeros((n, n), "float32")
    reproject(a, dst, gcps=g2, src_crs=gcrs, src_nodata=0, dst_crs=f"EPSG:{epsg}", dst_transform=tr,
              dst_nodata=0, resampling=resampling)
    return dst


def read_xband(href, provider, epsg, x0, y1):
    if provider in ("umbra", "capella"):      # georeferenced COGs with overviews
        with rasterio.open(href) as s0:
            native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
            ovs = s0.overviews(1)
        # pick the coarsest overview still at least 2x finer than the 2.5 m target
        lvl = None
        for i, f in enumerate(ovs):
            if native * f <= HR / 2:
                lvl = i
        with rasterio.open(href, overview_level=lvl) if lvl is not None else rasterio.open(href) as s:
            amp = warp(s, epsg, x0, y1, HR, NH)
    else:
        with rasterio.open(href) as s:
            amp = warp(s, epsg, x0, y1, HR, NH)
    return amp


_client = None


def s1_client():
    global _client
    if _client is None:
        _client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    return _client


def read_s1(lon, lat, when, epsg, x0, y1, window_days=30, pol="vv"):
    t = pd.Timestamp(when)
    rng = f"{(t - pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')}/{(t + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')}"
    d = 0.02
    items = list(s1_client().search(collections=["sentinel-1-rtc"], bbox=[lon - d, lat - d, lon + d, lat + d],
                                    datetime=rng).items())
    items = [i for i in items if pol in i.assets]
    if not items:
        return None, None, None
    items.sort(key=lambda i: abs(pd.Timestamp(i.datetime) - t))
    for it in items[:3]:
        with rasterio.open(it.assets[pol].href) as s:
            with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, LR, LR), width=NL, height=NL,
                           resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
                a = v.read(1)
        a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
        if np.isfinite(a).mean() > 0.9:
            return a, it.id, abs((pd.Timestamp(it.datetime) - t).days)
    return None, None, None


def to_db(a, power):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10 if power else 20) * np.log10(a)


def process(row, out_dir: Path):
    f = out_dir / f"{row.provider}_{row.xband_id}.npz"
    if f.exists():
        return {"status": "cached"}
    epsg, x0, y1 = tile_grid(row.lon, row.lat)
    pol = getattr(row, "s1_pol", "vv")
    pol = pol if isinstance(pol, str) else "vv"         # older selection rows have no s1_pol
    s1, s1_id, dt = read_s1(row.lon, row.lat, row.datetime, epsg, x0, y1, pol=pol)
    if s1 is None:
        return {"status": "no_s1"}
    amp = read_xband(row.href, row.provider, epsg, x0, y1)
    xval = amp > 0
    if xval.mean() < 0.6:
        return {"status": "x_sparse", "x_valid": float(xval.mean())}
    xdb = to_db(np.where(xval, amp, np.nan), power=False)
    sdb = to_db(s1, power=True)

    # co-register on the 10 m grid
    xl = np.nanmean(xdb.reshape(NL, 4, NL, 4), axis=(1, 3))
    m = np.isfinite(xl) & np.isfinite(sdb)
    a = np.where(m, xl - np.nanmean(xl[m]), 0)
    b = np.where(m, sdb - np.nanmean(sdb[m]), 0)
    shift, _, _ = phase_cross_correlation(b, a, upsample_factor=4, normalization=None)
    if np.abs(shift).max() > 4:
        shift = np.zeros(2)
    xs = ndimage.shift(np.nan_to_num(xdb, nan=np.nanmedian(xdb)), shift * 4, order=1, mode="nearest")
    vs = ndimage.shift(xval.astype("float32"), shift * 4, order=0, mode="constant") > 0.5
    xl2 = xs.reshape(NL, 4, NL, 4).mean(axis=(1, 3))
    m2 = m & vs.reshape(NL, 4, NL, 4).all(axis=(1, 3))
    corr = float(np.corrcoef(xl2[m2], sdb[m2])[0, 1]) if m2.sum() > 100 else np.nan

    np.savez_compressed(f, s1=sdb.astype("float16"), x=np.where(vs, xs, np.nan).astype("float16"),
                        epsg=epsg, x0=x0, y1=y1)
    return {"status": "ok", "s1_id": s1_id, "s1_dt_days": dt, "shift_px10": shift.tolist(),
            "corr10": corr, "x_valid": float(vs.mean()), "epsg": epsg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-umbra", type=int, default=350)
    ap.add_argument("--n-iceye", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    sel_f = ROOT / "selection.csv"
    if sel_f.exists():
        sel = pd.read_csv(sel_f)
    else:
        sel = select(a.n_umbra, a.n_iceye, a.seed)
        hrefs = umbra_gec_hrefs(set(sel.loc[sel.provider == "umbra", "xband_id"]))
        sel["href"] = np.where(sel.provider == "umbra", sel.xband_id.map(hrefs), sel.grd_href)
        sel = sel[sel.href.notna()]
        sel.to_csv(sel_f, index=False)
    print(f"selection: {len(sel)} collects  {sel.groupby(['split', 'provider']).size().to_dict()}", flush=True)

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, r, out_dir): r for r in sel.itertuples()}
        for n, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                res = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            rows.append({"xband_id": r.xband_id, "provider": r.provider, "split": r.split,
                         "test_region": r.test_region, "lon": r.lon, "lat": r.lat, **res})
            if n % 10 == 0 or n == len(futs):
                st = pd.Series([x["status"] for x in rows]).value_counts().to_dict()
                print(f"[{n}/{len(futs)}] {time.time() - t0:.0f}s {st}", flush=True)
    man = pd.DataFrame(rows)
    old = ROOT / "manifest.csv"
    if old.exists():
        prev = pd.read_csv(old)
        man = pd.concat([prev[~prev.xband_id.isin(man[man.status != "cached"].xband_id)],
                         man[man.status != "cached"]])
    man.to_csv(old, index=False)
    print("done", man.status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
