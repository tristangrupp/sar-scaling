"""Choose 3.2 km tile centers per Stripmap scene: on land, with structure, inside both the
Stripmap and the paired IW RTC footprints, preferring X-band coverage.

Candidates sit on a 3.4 km lattice inside (SM footprint ∩ IW footprint), shrunk by 1 km.
Each is scored from ESA WorldCover 2021 (Planetary Computer, read at ~100 m):
  land     share of pixels that are neither water (80) nor no-data (open ocean); must be >= 0.35
  built    share of built-up (50)
  entropy  land-cover class entropy (mixed scenes have more edges)
  xband    number of Umbra / ICEYE / Capella collects whose footprint covers the tile center
score = 3*built + entropy/ln(6) + 0.5*min(xband, 4)/4 (+3 for xband > 0 in Houston, where stage 1
needs X-band). The top K non-overlapping candidates are kept.

Writes tiles.csv. Run in the ftwtools env, in the background.
"""
from __future__ import annotations

import sys

from common import AMIEVA_SPLIT, DATA, HOLDOUT, PROJ, ROOT, pc_client, pc_item, utm_epsg  # noqa: I001

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_tf, unary_union

K_DEFAULT, K_HOLDOUT = 6, 8
SPACING = 3400
WC_RES = 0.001          # deg, about 100 m


def scenes():
    """All Stripmap scenes with a SAFE on disk and their IW pair."""
    pil = pd.concat([pd.read_csv(ROOT / "pilot_scenes.csv"), pd.read_csv(ROOT / "replacement_scenes.csv")])
    logs = pd.concat([pd.read_csv(DATA / "fetch_log_pilot.csv"), pd.read_csv(DATA / "fetch_log.csv")])
    logs = logs[logs.status == "ok"].drop_duplicates("granule", keep="last")
    pil = pil.merge(logs[["granule", "slc", "safe"]], on="granule")
    sm = gpd.read_parquet(ROOT / "sm_scenes.parquet").set_index("granule")
    pil["geometry"] = [sm.loc[g, "geometry"] for g in pil.granule]
    pil["role"] = [HOLDOUT.get(r, "train") for r in pil.region]
    am = pd.read_csv(ROOT / "amieva_pairs.csv")
    fp = gpd.read_file(ROOT / "amieva_footprints.geojson").set_index("granule")
    am = am.assign(region="amieva_" + am.granule.map(AMIEVA_SPLIT).replace({"test": "test"}),
                   slc=am.granule, safe=[str(DATA / "amieva2026" / f"{g}.SAFE") for g in am.granule],
                   geometry=[fp.loc[g, "geometry"] for g in am.granule])
    am["role"] = am.granule.map(AMIEVA_SPLIT).map({"test": "holdout_paper", "val": "val", "train": "train"})
    am["region"] = am.granule.map(AMIEVA_SPLIT).map(
        {"test": "amieva_test", "val": "amieva_val", "train": "amieva_train"})
    am = am.rename(columns={"dir": "orbit_dir"})
    cols = ["granule", "slc", "safe", "region", "role", "iw_id", "iw_dt_days", "orbit_dir", "geometry"]
    out = pd.concat([pil[cols], am[cols]], ignore_index=True)
    out = out[[Path(s).joinpath(".complete").exists() for s in out.safe]]
    return gpd.GeoDataFrame(out, crs=4326)


def xband_items():
    parts = []
    for prov, f, idc in [("umbra", "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson", "xband_id"),
                         ("iceye", "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson", "xband_id"),
                         ("capella", "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson", "capella_id")]:
        g = gpd.read_file(PROJ / f).rename(columns={idc: "xband_id"})
        g["provider"] = prov
        parts.append(g[["xband_id", "provider", "datetime", "sar:polarizations", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def worldcover(bounds):
    """WorldCover classes on a WC_RES lat/lon grid over bounds (lon0, lat0, lon1, lat1)."""
    lon0, lat0, lon1, lat1 = bounds
    w, h = int(np.ceil((lon1 - lon0) / WC_RES)), int(np.ceil((lat1 - lat0) / WC_RES))
    tr = from_origin(lon0, lat1, WC_RES, WC_RES)
    out = np.zeros((h, w), "uint8")
    items = list(pc_client().search(collections=["esa-worldcover"], bbox=list(bounds),
                                    query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    for it in items:
        with rasterio.open(it.assets["map"].href) as s:
            ovs = s.overviews(1)
            lvl = max([i for i, f in enumerate(ovs) if f <= 8], default=None)
        with (rasterio.open(it.assets["map"].href, overview_level=lvl) if lvl is not None
              else rasterio.open(it.assets["map"].href)) as s, WarpedVRT(
                s, crs="EPSG:4326", transform=tr, width=w, height=h, resampling=Resampling.nearest,
                nodata=0) as v:
            a = v.read(1)
        out = np.where(out == 0, a, out)
    return out, tr


def score_candidates(fp, wc, tr, xb, region):
    c = fp.centroid
    epsg = utm_epsg(c.x, c.y)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    fpu = shp_tf(fwd, fp).buffer(-1000)
    if fpu.is_empty:
        return pd.DataFrame()
    x0, y0, x1, y1 = fpu.bounds
    rows = []
    half = 1600
    for x in np.arange(x0, x1, SPACING):
        for y in np.arange(y0, y1, SPACING):
            if not fpu.contains(Point(x, y).buffer(half, cap_style=3)):
                continue
            lon, lat = inv(x, y)
            # WorldCover window covering the tile (approximate square in degrees)
            dlat = half / 111_000
            dlon = half / (111_000 * np.cos(np.radians(lat)))
            r0, c0 = rasterio.transform.rowcol(tr, lon - dlon, lat + dlat)
            r1, c1 = rasterio.transform.rowcol(tr, lon + dlon, lat - dlat)
            win = wc[max(r0, 0):r1, max(c0, 0):c1]
            if win.size == 0:
                continue
            land = float(((win != 80) & (win != 0)).mean())
            built = float((win == 50).mean())
            vals, cnt = np.unique(win[(win != 0)], return_counts=True)
            p = cnt / max(cnt.sum(), 1)
            ent = float(-(p * np.log(p)).sum()) if cnt.sum() else 0.0
            pt = Point(lon, lat)
            hit = xb[xb.intersects(pt)]
            n = {k: int((hit.provider == k).sum()) for k in ("umbra", "iceye", "capella")}
            rows.append({"lon": lon, "lat": lat, "x": x, "y": y, "land": land, "built": built, "entropy": ent,
                         "n_xband": len(hit), **{f"n_{k}": v for k, v in n.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.land >= 0.35].copy()         # islands: water is masked per pixel later
    df["score"] = 3 * df.built + df.entropy / np.log(6) + 0.5 * np.minimum(df.n_xband, 4) / 4
    if region == "houston":
        df["score"] += 3 * (df.n_xband > 0)
    return df.sort_values("score", ascending=False)


def same_pass(iw_id, sm_geom):
    """All RTC frames of the IW pass (same day and absolute orbit) that touch the Stripmap scene;
    one frame often covers only part of an 80 x 170 km Stripmap scene."""
    it = pc_item("sentinel-1-rtc", iw_id)
    day = it.datetime.strftime("%Y-%m-%d")
    orbit = it.properties.get("sat:absolute_orbit")
    items = [i for i in pc_client().search(collections=["sentinel-1-rtc"], intersects=sm_geom.__geo_interface__,
                                           datetime=f"{day}/{day}").items()
             if i.properties.get("sat:absolute_orbit") == orbit and "vv" in i.assets]
    items.sort(key=lambda i: i.id != iw_id)          # the matched frame first
    return items or [it]


def pick(df, k):
    chosen = []
    for r in df.itertuples():
        if all(np.hypot(r.x - c.x, r.y - c.y) >= 3300 for c in chosen):
            chosen.append(r)
        if len(chosen) == k:
            break
    return pd.DataFrame(chosen)


def main():
    sc = scenes()
    prev = pd.read_csv(ROOT / "tiles.csv") if (ROOT / "tiles.csv").exists() and "--all" not in sys.argv else None
    if prev is not None:                       # append: keep existing tiles, add new scenes only
        sc = sc[~sc.slc.isin(prev.slc)]
    xb = xband_items()
    print(f"{len(sc)} scenes with SAFE on disk; {len(xb)} X-band items", flush=True)
    out = []
    for s in sc.itertuples():
        try:
            iws = same_pass(s.iw_id, s.geometry)
            fp = s.geometry.intersection(unary_union([shape(i.geometry) for i in iws]))
            xbs = xb[xb.intersects(fp)]
            wc, tr = worldcover(fp.bounds)
            cand = score_candidates(fp, wc, tr, xbs, s.region)
            k = K_HOLDOUT if s.role.startswith("holdout") else K_DEFAULT
            sel = pick(cand, k) if len(cand) else pd.DataFrame()
        except Exception as e:
            print(f"{s.region} {s.slc[:40]} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if sel.empty:
            print(f"{s.region:13s} {s.slc[:40]} no land candidates", flush=True)
            continue
        sel = sel.drop(columns=["Index"], errors="ignore").assign(
            region=s.region, role=s.role, slc=s.slc, safe=s.safe, iw_id=";".join(i.id for i in iws), iw_dt_days=s.iw_dt_days,
            orbit_dir=s.orbit_dir)
        sel["tile_id"] = [f"{s.region}_{s.slc[17:32]}_{s.slc[4:6]}_{i}" for i in range(len(sel))]
        out.append(sel)
        print(f"{s.region:13s} {s.slc[:40]} cand={len(cand):4d} picked={len(sel)} "
              f"built={sel.built.mean():.2f} xband_tiles={(sel.n_xband > 0).sum()}", flush=True)
    t = pd.concat(([prev] if prev is not None else []) + out, ignore_index=True)
    t.to_csv(ROOT / "tiles.csv", index=False)
    print(t.groupby(["role", "region"]).agg(n=("tile_id", "size"), xb=("n_xband", lambda v: (v > 0).sum())), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
