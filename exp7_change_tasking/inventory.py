"""Inventory open X-band collects (Capella, Umbra, ICEYE) and find repeat-tasked sites.

Reads the footprint caches from exp0_overlap_atlas (read-only). Collects whose footprint centroids
lie within LINK_M of each other join one site (single-linkage). A site counts as a repeat site when
it holds collects on at least two distinct UTC days.

Writes:
  xband_collects.csv   one row per collect: provider, id, datetime, pol, mode, incidence, orbit,
                       centroid lon/lat, footprint area km2, site_id
  repeat_sites.csv     one row per site with >= 2 collect days: providers, n collects, n days,
                       first/last date, median gap, longest run of gaps <= 30 days
  xband_collects.geojson  footprints with site_id
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
EXP0 = ROOT.parent / "exp0_overlap_atlas"
LINK_M = 2000
EQ = "EPSG:6933"


def load():
    out = []
    c = gpd.read_file(EXP0 / "s1sm_capella" / "capella_geo_items.geojson").rename(columns={"capella_id": "xband_id"})
    c["provider"] = "capella"
    u = gpd.read_file(EXP0 / "s1sm_umbra" / "umbra_items.geojson")
    u["provider"] = "umbra"
    i = gpd.read_file(EXP0 / "s1sm_iceye" / "iceye_items.geojson")
    i["provider"] = "iceye"
    for g in (c, u, i):
        g = g.rename(columns={"sar:instrument_mode": "mode", "sar:polarizations": "pol",
                              "view:incidence_angle": "inc", "sat:orbit_state": "orbit",
                              "sar:product_type": "product", "view:azimuth": "azimuth"})
        for k in ("product", "orbit", "grd_href", "azimuth"):
            if k not in g:
                g[k] = None
        out.append(g[["provider", "xband_id", "datetime", "platform", "mode", "product", "pol", "inc",
                      "orbit", "azimuth", "grd_href", "geometry"]])
    g = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs="EPSG:4326")
    g["datetime"] = pd.to_datetime(g.datetime, utc=True)
    g["mode"] = g["mode"].str.lower()
    return g


def main():
    g = load()
    eq = g.to_crs(EQ)
    g["area_km2"] = (eq.area / 1e6).round(2)
    cen = eq.centroid
    xy = np.c_[cen.x, cen.y]
    pairs = cKDTree(xy).query_pairs(LINK_M, output_type="ndarray")
    from scipy.sparse import coo_matrix
    n = len(g)
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    g["site_id"] = lab
    c4326 = cen.to_crs("EPSG:4326")
    g["lon"], g["lat"] = c4326.x.round(5), c4326.y.round(5)
    g["day"] = g.datetime.dt.strftime("%Y-%m-%d")
    g.drop(columns="geometry").to_csv(ROOT / "xband_collects.csv", index=False)
    g.drop(columns=["datetime"]).assign(datetime=g.datetime.astype(str)).to_file(ROOT / "xband_collects.geojson", driver="GeoJSON")

    rows = []
    for sid, s in g.groupby("site_id"):
        days = sorted(pd.to_datetime(s.day.unique()))
        if len(days) < 2:
            continue
        gaps = np.diff(days).astype("timedelta64[D]").astype(int)
        # longest stretch of consecutive collect days with every gap <= 30 days
        run, best = 1, 1
        for d in gaps:
            run = run + 1 if d <= 30 else 1
            best = max(best, run)
        rows.append({"site_id": sid, "providers": ",".join(sorted(s.provider.unique())),
                     "n_collects": len(s), "n_days": len(days), "first": days[0].date(), "last": days[-1].date(),
                     "median_gap_d": int(np.median(gaps)), "max_run_le30d": best,
                     "pols": ",".join(sorted(s.pol.dropna().astype(str).unique())),
                     "lon": round(float(s.lon.median()), 4), "lat": round(float(s.lat.median()), 4)})
    r = pd.DataFrame(rows).sort_values(["n_days"], ascending=False)
    r.to_csv(ROOT / "repeat_sites.csv", index=False)
    print("collects by provider:", g.provider.value_counts().to_dict())
    print("sites:", g.site_id.nunique(), " repeat sites (>=2 days):", len(r))
    print(r.groupby("providers").size().to_string())
    print("repeat sites with >=3 days:", (r.n_days >= 3).sum(), " with a run of >=2 days gap<=30d:", (r.max_run_le30d >= 2).sum())
    print(r.head(40).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
