"""Stage 0: inventory Stripmap SLC scenes, pair them with IW RTC, and rank candidate regions.

1. ASF search (no login) for Sentinel-1 Stripmap SLC with VV, 2023-01 onward, worldwide.
2. Bin scene centers into 1-degree cells.
3. For the busiest cells, query Planetary Computer sentinel-1-rtc once per cell and count Stripmap
   scenes that have a VV IW scene within +/- 6 days (and how many share orbit direction).
4. Distance from each cell to the nearest open X-band collect (Umbra, ICEYE, Capella) for the
   near / far holdout split.
Outputs: sm_scenes.parquet, cells.csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import requests
from pystac_client import Client
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
ASF = "https://api.daac.asf.alaska.edu/services/search/param"


def cdse_sm(start="2023-01-01", end=None):
    """Global Stripmap inventory from the CDSE STAC GRD listing, month by month.

    ASF's search API times out on worldwide Stripmap queries, so the inventory uses CDSE GRD
    items (one per Stripmap acquisition); the matching SLC is resolved on ASF per region later.
    """
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    months = pd.date_range(start, end, freq="MS").tolist() + [pd.Timestamp(end)]
    feats = []
    for a, b in zip(months[:-1], months[1:]):
        body = {"collections": ["sentinel-1-grd"], "limit": 1000,
                "datetime": f"{a:%Y-%m-%dT00:00:00Z}/{b:%Y-%m-%dT00:00:00Z}",
                "filter": {"op": "=", "args": [{"property": "sar:instrument_mode"}, "SM"]},
                "filter-lang": "cql2-json"}
        url, fs = "https://stac.dataspace.copernicus.eu/v1/search", []
        while url:
            for attempt in range(6):
                r = requests.post(url, json=body, timeout=120)
                if r.status_code == 200:
                    break
                time.sleep(10 * (attempt + 1))
            d = r.json()
            fs += d["features"]
            nxt = [l for l in d.get("links", []) if l["rel"] == "next"]
            url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
            time.sleep(2)
        feats += fs
        print(f"{a:%Y-%m}: {len(fs)} Stripmap", flush=True)
    rows = []
    for f in feats:
        p = f["properties"]
        pols = p.get("sar:polarizations") or []
        if "VV" not in pols:
            continue
        rows.append({"granule": f["id"].replace("_COG", ""), "start": p["start_datetime"],
                     "beam": p["product:type"][:2], "pol": "+".join(pols),
                     "orbit_dir": (p.get("sat:orbit_state") or "").upper(), "path": p.get("sat:relative_orbit"),
                     "size_mb": 4400.0, "geometry": shape(f["geometry"])})
    g = gpd.GeoDataFrame(rows, crs="EPSG:4326").drop_duplicates("granule")
    c = g.geometry.centroid
    g["lon"], g["lat"] = c.x, c.y
    g["cell"] = np.floor(g.lon).astype(int).astype(str) + "_" + np.floor(g.lat).astype(int).astype(str)
    return g


def xband_points():
    pts = []
    for f in ["exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson", "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson",
              "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson"]:
        x = gpd.read_file(PROJ / f)
        c = x.geometry.centroid
        pts.append(np.c_[c.x, c.y])
    return np.vstack(pts)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def iw_pairs(cell_scenes, client, window=6):
    """Count Stripmap scenes in a cell with a VV IW RTC scene within +/- window days."""
    lon, lat = cell_scenes.lon.median(), cell_scenes.lat.median()
    t0 = pd.Timestamp(cell_scenes.start.min()) - pd.Timedelta(days=window)
    t1 = pd.Timestamp(cell_scenes.start.max()) + pd.Timedelta(days=window)
    items = list(client.search(collections=["sentinel-1-rtc"], bbox=[lon - .05, lat - .05, lon + .05, lat + .05],
                               datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}").items())
    iw = pd.DataFrame([{"t": pd.Timestamp(i.datetime), "dir": (i.properties.get("sat:orbit_state") or "").upper()}
                       for i in items if "vv" in i.assets])
    n_any = n_same = 0
    for s in cell_scenes.itertuples():
        if iw.empty:
            break
        d = (iw.t - pd.Timestamp(s.start)).abs() <= pd.Timedelta(days=window)
        n_any += bool(d.any())
        n_same += bool((d & (iw.dir == s.orbit_dir.upper())).any())
    return n_any, n_same, len(iw)


def main():
    f = ROOT / "sm_scenes.parquet"
    if f.exists():
        sm = gpd.read_parquet(f)
    else:
        sm = cdse_sm()
        sm.to_parquet(f)
    print(f"{len(sm)} Stripmap VV scenes, {sm.cell.nunique()} cells", flush=True)

    cells = sm.groupby("cell").agg(n_sm=("granule", "size"), lon=("lon", "median"), lat=("lat", "median"),
                                   first=("start", "min"), last=("start", "max"),
                                   gb=("size_mb", lambda s: s.sum() / 1e3)).reset_index()
    xb = xband_points()
    cells["km_to_xband"] = [haversine_km(r.lon, r.lat, xb[:, 0], xb[:, 1]).min() for r in cells.itertuples()]

    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    top = cells.sort_values("n_sm", ascending=False).head(200)
    res = {}
    for n, r in enumerate(top.itertuples(), 1):
        try:
            res[r.cell] = iw_pairs(sm[sm.cell == r.cell], client)
        except Exception as e:
            print("  iw fail", r.cell, e, flush=True)
        if n % 20 == 0:
            print(f"  IW pairing {n}/{len(top)}", flush=True)
    cells["n_pair_iw6"] = cells.cell.map(lambda c: res.get(c, (np.nan,) * 3)[0])
    cells["n_pair_iw6_samedir"] = cells.cell.map(lambda c: res.get(c, (np.nan,) * 3)[1])
    cells["n_iw"] = cells.cell.map(lambda c: res.get(c, (np.nan,) * 3)[2])
    cells.sort_values("n_pair_iw6", ascending=False).to_csv(ROOT / "cells.csv", index=False)
    print(cells.sort_values("n_pair_iw6", ascending=False).head(30).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
