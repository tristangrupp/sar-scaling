"""X-band edge references on the 5 m tile grids (stage 1 validation and the arm B loss).

For each built tile with X-band coverage and each overlapping collect (up to MAX_PER_PROVIDER per
provider, nearest in time to the Stripmap date):
  read      Umbra GEC / Capella GEO COGs (overview just finer than 2.5 m) or ICEYE GRD (via GCPs),
            averaged onto the tile's 5 m grid; amplitude -> dB
  register  to the tile's IW (never to the Stripmap, so held-out tiles use only inference inputs):
            masked normalized cross-correlation at 5 m (X-band and IW both smoothed), +-8 px
  edges     Canny (sigma 2 px, quantile thresholds 0.85 / 0.95) on valid X-band; distance transform
  masks     infrastructure = OSM buildings and roads (Overpass, buffered 5-10 m) or WorldCover
            built-up; layover / shadow from the Copernicus DEM for the Stripmap geometry and, where
            known, the X-band geometry, plus slopes over 20 degrees; change = |IW at the X-band
            date - IW at the Stripmap date| > 3 dB (3x3 mean, 10 m); land only
Writes DATA/xband/<tile_id>__<xband_id>.npz and xband_pairs.csv.

  python xband_prep.py [--workers 6]
"""
from __future__ import annotations

import sys

from common import DATA, PROJ, ROOT, TILES, block_mean, db, pc_client, read_iw  # noqa: I001

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from shapely.geometry import LineString, Polygon, box, shape
from shapely.ops import transform as shp_tf

sys.path.insert(0, str(PROJ / "exp1_gan_spike"))
import prep  # noqa: E402  (exp1_gan_spike readers: warp, warp_gcps, umbra_gec_hrefs)

XB = DATA / "xband"
OSM = DATA / "osm"
N5, RES = 640, 5.0
MAX_PER_PROVIDER = 4
BEAM_INC = {"S1": 20.5, "S2": 24.5, "S3": 30.5, "S4": 35.5, "S5": 39.5, "S6": 43.5}


# ------------------------------------------------------------------ shared edge detector
def edges(img_db, mask, sigma=2.0, lo=0.85, hi=0.95):
    """Canny edges with quantile thresholds, so every image gives a comparable edge density."""
    from skimage.feature import canny
    a = np.where(mask, img_db, np.nanmedian(img_db[mask]) if mask.any() else 0)
    a = np.nan_to_num(a)
    return canny(a, sigma=sigma, low_threshold=lo, high_threshold=hi, use_quantiles=True, mask=mask)


# ------------------------------------------------------------------ items
def items():
    parts = []
    for prov, f, idc in [("umbra", "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson", "xband_id"),
                         ("iceye", "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson", "xband_id"),
                         ("capella", "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson", "capella_id")]:
        g = gpd.read_file(PROJ / f).rename(columns={idc: "xband_id"})
        g["provider"] = prov
        for c in ["sar:observation_direction", "grd_href", "view:azimuth"]:
            if c not in g:
                g[c] = None
        parts.append(g[["xband_id", "provider", "datetime", "sar:polarizations", "sar:instrument_mode",
                        "view:incidence_angle", "sat:orbit_state", "sar:observation_direction", "grd_href",
                        "geometry"]])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def resolve_hrefs(sel):
    cache = ROOT / "xband_hrefs.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    need_u = set(sel.loc[sel.provider == "umbra", "xband_id"]) - set(known)
    need_c = set(sel.loc[sel.provider == "capella", "xband_id"]) - set(known)
    if need_u:
        known.update({k: v for k, v in prep.umbra_gec_hrefs(need_u).items() if v})
    if need_c:
        import add_capella
        known.update({k: v for k, v in add_capella.hrefs(need_c).items() if v})
    for r in sel[sel.provider == "iceye"].itertuples():
        known[r.xband_id] = r.grd_href
    cache.write_text(json.dumps(known))
    return known


def read_xband(href, provider, epsg, x0, y1):
    if provider in ("umbra", "capella"):
        with rasterio.open(href) as s0:
            native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
            ovs = s0.overviews(1)
        lvl = None
        for i, f in enumerate(ovs):
            if native * f <= RES / 2:
                lvl = i
        with (rasterio.open(href, overview_level=lvl) if lvl is not None else rasterio.open(href)) as s:
            return prep.warp(s, epsg, x0, y1, RES, N5)
    with rasterio.open(href) as s:
        return prep.warp(s, epsg, x0, y1, RES, N5)


# ------------------------------------------------------------------ masks
def osm_infra(tile_id, epsg, x0, y1):
    """OSM buildings (filled, +5 m) and roads (+5 m each side) rasterized on the 5 m grid."""
    OSM.mkdir(parents=True, exist_ok=True)
    f = OSM / f"{tile_id}.json"
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    lo0, la0 = inv.transform(x0, y1 - N5 * RES)
    lo1, la1 = inv.transform(x0 + N5 * RES, y1)
    if f.exists():
        js = json.loads(f.read_text())
    else:
        q = (f"[out:json][timeout:90];(way[\"building\"]({la0},{lo0},{la1},{lo1});"
             f"way[\"highway\"~\"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|"
             f"motorway_link|trunk_link|primary_link\"]({la0},{lo0},{la1},{lo1}););out geom;")
        js = None
        for attempt in range(4):
            try:
                r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=120,
                                  headers={"User-Agent": "s1sm-exp2-research"})
                if r.status_code == 200:
                    js = r.json()
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(10 * (attempt + 1))
        if js is None:
            return None
        f.write_text(json.dumps(js))
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    shapes_b, shapes_r = [], []
    for el in js.get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        pts = [(p["lon"], p["lat"]) for p in g]
        tags = el.get("tags", {})
        if "building" in tags and len(pts) >= 4:
            shapes_b.append(shp_tf(fwd, Polygon(pts)).buffer(5))
        elif "highway" in tags:
            shapes_r.append(shp_tf(fwd, LineString(pts)).buffer(5))
    tr = from_origin(x0, y1, RES, RES)
    out = np.zeros((N5, N5), bool)
    for sh in (shapes_b, shapes_r):
        if sh:
            out |= rasterize([(s, 1) for s in sh if s.is_valid], out_shape=(N5, N5), transform=tr, fill=0,
                             dtype="uint8").astype(bool)
    return out, len(shapes_b), len(shapes_r)


def dem5(lon, lat, epsg, x0, y1):
    d = 0.1
    its = list(pc_client().search(collections=["cop-dem-glo-30"], bbox=[lon - d, lat - d, lon + d, lat + d]).items())
    dem = np.full((N5, N5), np.nan, "float32")
    for it in its:
        with rasterio.open(it.assets["data"].href) as s, WarpedVRT(
                s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, RES, RES), width=N5, height=N5,
                resampling=Resampling.bilinear, nodata=-9999, dtype="float32") as v:
            a = v.read(1)
        dem = np.where(np.isnan(dem) & (a != -9999), a, dem)
    return np.nan_to_num(dem, nan=0.0)


def layover_shadow(dem, look_az_deg, inc_deg):
    """True where the terrain slope toward the sensor exceeds the incidence (layover) or the slope
    away exceeds 90 - incidence (shadow). look_az: horizontal direction from sensor to ground."""
    gy, gx = np.gradient(ndimage.gaussian_filter(dem, 2), RES)
    gy = -gy                               # rows run south; make gy = dz/dnorth
    lx, ly = np.sin(np.radians(look_az_deg)), np.cos(np.radians(look_az_deg))
    toward = np.degrees(np.arctan(-(gx * lx + gy * ly)))     # > 0: terrain faces the sensor
    return (toward > inc_deg) | (-toward > 90 - inc_deg)


def look_azimuth(orbit_state, side="right"):
    """Approximate look azimuth for a near-polar sun-synchronous orbit."""
    if not isinstance(orbit_state, str) or orbit_state.lower() not in ("ascending", "descending"):
        return None
    heading = -12.0 if orbit_state.lower() == "ascending" else 192.0
    return (heading + (90 if side != "left" else -90)) % 360


# ------------------------------------------------------------------ registration
def register(xdb, xval, iw10):
    """Masked NCC of X-band (5 m, smoothed) against the IW upsampled to 5 m. Returns (dr, dc) px."""
    from skimage.registration import phase_cross_correlation
    iw5 = ndimage.zoom(np.where(np.isfinite(iw10), iw10, np.nanmedian(iw10)), 2, order=1)
    a = ndimage.gaussian_filter(np.where(xval, xdb, np.nanmedian(xdb[xval])), 2)
    m = ndimage.binary_erosion(xval, iterations=8)
    if m.sum() < 5000:
        return np.zeros(2), False
    sh = phase_cross_correlation(iw5, a, reference_mask=np.ones_like(m), moving_mask=m)
    sh = np.asarray(sh[0] if isinstance(sh, tuple) else sh, float)
    if np.abs(sh).max() > 8:
        return np.zeros(2), False
    return sh, True


def iw_near(lon, lat, when, epsg, x0, y1, days=12):
    t = pd.Timestamp(when)
    its = [i for i in pc_client().search(collections=["sentinel-1-rtc"], bbox=[lon - .02, lat - .02, lon + .02, lat + .02],
                                         datetime=f"{t - pd.Timedelta(days=days):%Y-%m-%d}/{t + pd.Timedelta(days=days):%Y-%m-%d}").items()
           if "vv" in i.assets]
    its.sort(key=lambda i: abs(pd.Timestamp(i.datetime) - t))
    for it in its[:3]:
        a = read_iw(it, epsg, x0, y1, 320)
        if np.isfinite(a).mean() > 0.8:
            return db(a), it, abs((pd.Timestamp(it.datetime) - t).days)
    return None, None, None


# ------------------------------------------------------------------ per pair
def process(job, href):
    t, x = job["tile"], job["x"]
    f = XB / f"{t.tile_id}__{x.xband_id}.npz"
    z = np.load(TILES / f"{t.tile_id}.npz")
    epsg, x0, y1 = int(z["epsg"]), float(z["x0"]), float(z["y1"])
    ix0, iy1 = float(z["iw_x0"]), float(z["iw_y1"])
    iw = z["iw"].astype("float32")
    wc = z["wc"]
    land = (wc != 80) & (wc != 0)
    amp = read_xband(href, x.provider, epsg, x0, y1)
    xval = amp > 0
    if xval.mean() < 0.15:
        return {"status": "x_sparse", "x_valid": float(xval.mean())}
    xdb = np.where(xval, db(amp.astype("float64") ** 2), np.nan).astype("float32")
    sh, ok = register(xdb, xval, iw)
    xdb = ndimage.shift(np.nan_to_num(xdb, nan=np.nanmedian(xdb)), sh, order=1, mode="nearest")
    xval = ndimage.shift(xval.astype("float32"), sh, order=0, mode="constant") > 0.5
    x10 = block_mean(np.where(xval, xdb, np.nan), 2)
    c10 = float(pd.Series(x10.ravel()).corr(pd.Series(iw.ravel())))
    # masks
    dem = dem5(t.lon, t.lat, epsg, x0, y1)
    beam = t.tile_id.split("_")[-2]
    sm_look = look_azimuth(t.orbit_dir)
    bad = np.zeros((N5, N5), bool)
    if sm_look is not None:
        bad |= layover_shadow(dem, sm_look, BEAM_INC.get(beam, 30))
    x_look = look_azimuth(x["sat:orbit_state"], x["sar:observation_direction"] or "right")
    x_inc = float(x["view:incidence_angle"]) if pd.notna(x["view:incidence_angle"]) else np.nan
    if x_look is not None and np.isfinite(x_inc):
        bad |= layover_shadow(dem, x_look, x_inc)
    gy, gx = np.gradient(dem, RES)
    bad |= np.degrees(np.arctan(np.hypot(gx, gy))) > 20
    iwx, itx, dtx = iw_near(t.lon, t.lat, x.datetime, epsg, ix0, iy1)
    if iwx is not None:
        k = np.ones((3, 3)) / 9
        d = np.abs(ndimage.convolve(np.nan_to_num(iwx, nan=0), k) - ndimage.convolve(np.nan_to_num(iw, nan=0), k))
        change = np.kron(d > 3.0, np.ones((2, 2), bool))
    else:
        change = np.zeros((N5, N5), bool)
    infra = osm_infra(t.tile_id, epsg, x0, y1)
    if infra is None:
        osm, nb, nr = np.zeros((N5, N5), bool), -1, -1
    else:
        osm, nb, nr = infra
    infra_m = osm | (wc == 50)
    valid = xval & land & ~bad & ~change
    e = edges(xdb, xval)
    dist = ndimage.distance_transform_edt(~e).astype("float32")
    np.savez_compressed(f, xdb=xdb.astype("float16"), edges=e, dist=np.minimum(dist, 50).astype("float16"),
                        infra=infra_m, osm=osm, valid=valid, bad=bad, change=change)
    sm_inc = BEAM_INC.get(beam, np.nan)
    same_side = (x_look is not None and sm_look is not None and
                 abs(((x_look - sm_look) + 180) % 360 - 180) < 45)
    return {"status": "ok", "x_valid": round(float(xval.mean()), 3), "shift_r": sh[0], "shift_c": sh[1],
            "reg_ok": ok, "corr10": round(c10, 3), "valid": round(float(valid.mean()), 3),
            "infra_valid": round(float((valid & infra_m).mean()), 3), "change": round(float(change.mean()), 3),
            "bad": round(float(bad.mean()), 3), "n_osm_bld": nb, "n_osm_road": nr, "osm_frac": round(float(osm.mean()), 3),
            "change_iw_dt": dtx, "sm_inc": sm_inc, "x_inc": x_inc, "same_look": same_side,
            "geom_match": bool(same_side and abs(sm_inc - x_inc) <= 5) if np.isfinite(x_inc) else False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    XB.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(ROOT / "tiles.csv")
    pairs = pd.read_csv(ROOT / "pairs_final.csv")
    splits = pd.read_csv(ROOT / "splits.csv")[["tile_id", "split"]]
    tiles = tiles.merge(pairs[pairs.use][["tile_id"]], on="tile_id").merge(splits, on="tile_id")
    tiles = tiles[tiles.n_xband > 0]
    xb = items()
    fwd_cache = {}
    jobs = []
    for t in tiles.itertuples():
        z = np.load(TILES / f"{t.tile_id}.npz")
        epsg, x0, y1 = int(z["epsg"]), float(z["x0"]), float(z["y1"])
        if epsg not in fwd_cache:
            fwd_cache[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True).transform
        tb = shp_tf(fwd_cache[epsg], box(x0, y1 - N5 * RES, x0 + N5 * RES, y1))
        hit = xb[xb.intersects(tb)].copy()
        if hit.empty:
            continue
        hit["cover"] = [g.intersection(tb).area / tb.area for g in hit.geometry]
        hit = hit[hit.cover >= 0.1]
        sm_t = pd.Timestamp(t.tile_id.split("_")[-3][:8], tz="UTC")
        hit["dt_x"] = (pd.to_datetime(hit.datetime, utc=True, format="mixed") - sm_t).abs().dt.days
        hit = hit.sort_values(["cover", "dt_x"], ascending=[False, True]).groupby("provider").head(MAX_PER_PROVIDER)
        for _, x in hit.iterrows():
            jobs.append({"tile": t, "x": x})
    sel = pd.DataFrame([{"xband_id": j["x"].xband_id, "provider": j["x"].provider, "grd_href": j["x"].grd_href}
                        for j in jobs]).drop_duplicates("xband_id")
    print(f"{len(tiles)} tiles with X-band, {len(jobs)} tile-collect pairs, {len(sel)} collects", flush=True)
    hrefs = resolve_hrefs(sel)
    out_f = ROOT / "xband_pairs.csv"
    rows, t0 = [], time.time()

    def run(j):
        h = hrefs.get(j["x"].xband_id)
        if not h:
            return {"status": "no_href"}
        f = XB / f"{j['tile'].tile_id}__{j['x'].xband_id}.npz"
        if f.exists():
            return {"status": "cached"}
        try:
            return process(j, h)
        except Exception as e:
            return {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for n, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            x = j["x"]
            rows.append({"tile_id": j["tile"].tile_id, "region": j["tile"].region, "split": j["tile"].split,
                         "xband_id": x.xband_id, "provider": x.provider, "pol": x["sar:polarizations"],
                         "x_date": x.datetime, "dt_days": x["dt_x"], "cover": round(x.cover, 3), **fu.result()})
            if n % 10 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s "
                      f"{pd.Series([r['status'] for r in rows]).value_counts().to_dict()}", flush=True)
                old = pd.read_csv(out_f) if out_f.exists() else pd.DataFrame(columns=["tile_id", "xband_id"])
                new = pd.DataFrame(rows)
                new = new[new.status != "cached"]
                key = set(zip(new.tile_id, new.xband_id))
                old = old[[k not in key for k in zip(old.tile_id, old.xband_id)]]
                pd.concat([old, new]).to_csv(out_f, index=False)
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
