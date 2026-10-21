"""Find where Sentinel-1 Stripmap (SM) archive footprints overlap Capella Open Data.

Steps:
  1. Crawl the Capella Open Data static STAC (GEO product type) and cache item metadata.
  2. Merge overlapping Capella footprints into "sites" so each site costs one API query.
  3. For each site, search the Copernicus Data Space (CDSE) STAC API for S1 GRD items
     with sar:instrument_mode = SM, over the full archive.
  4. Pair every Capella item with every SM item that overlaps it, and score the pair on
     co-polarization match, incidence-angle gap, orbit direction and time gap.
  5. Optionally (--with-iw) count S1 IW scenes within +/- N days of each good SM scene,
     to build IW (input) / SM (validation) / Capella (target) triplets.

Outputs (in --out):
  capella_geo_items.geojson   Capella footprints + SAR metadata (cache)
  sites.geojson               merged Capella sites with Capella / SM counts
  pairs.csv                   one row per Capella x SM overlap, with match flags
  cdse_cache/                 raw CDSE responses per site (lets reruns resume)

Run:  python s1sm_capella_overlap.py --out ./s1sm_capella
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

CAPELLA_GEO = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
               "capella-open-data-by-product-type/capella-open-data-geo/collection.json")
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
EQUAL_AREA = "EPSG:6933"

CAPELLA_KEEP = ["datetime", "platform", "sar:instrument_mode", "sar:polarizations",
                "view:incidence_angle", "sat:orbit_state", "capella:use_case",
                "sar:resolution_range", "sar:resolution_azimuth"]


# ---------------------------------------------------------------- Capella
def load_capella(out: Path, workers: int) -> gpd.GeoDataFrame:
    cache = out / "capella_geo_items.geojson"
    if cache.exists():
        return gpd.read_file(cache)

    coll = requests.get(CAPELLA_GEO, timeout=60).json()
    hrefs = [urljoin(CAPELLA_GEO, l["href"]) for l in coll["links"] if l["rel"] == "item"]
    print(f"Capella GEO: fetching {len(hrefs)} item JSONs")

    session = requests.Session()

    def fetch(href):
        for attempt in range(4):
            try:
                r = session.get(href, timeout=60)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                time.sleep(2 ** attempt)
        print(f"  gave up on {href}")
        return None

    with ThreadPoolExecutor(workers) as ex:
        items = [i for i in ex.map(fetch, hrefs) if i]

    rows = []
    for it in items:
        p = it["properties"]
        row = {"capella_id": it["id"], "geometry": shape(it["geometry"])}
        row.update({k: p.get(k) for k in CAPELLA_KEEP})
        row["sar:polarizations"] = ",".join(p.get("sar:polarizations") or [])
        rows.append(row)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(cache, driver="GeoJSON")
    return gdf


def make_sites(cap: gpd.GeoDataFrame, merge_m: float) -> gpd.GeoDataFrame:
    """Merge Capella footprints closer than merge_m into one site polygon."""
    ea = cap.to_crs(EQUAL_AREA)
    merged = unary_union(ea.geometry.buffer(merge_m / 2))
    parts = list(getattr(merged, "geoms", [merged]))
    sites = gpd.GeoDataFrame({"site_id": range(len(parts))}, geometry=parts, crs=EQUAL_AREA)
    sites = sites.to_crs("EPSG:4326")
    # simplify so the CDSE intersects payload stays small
    sites["geometry"] = sites.geometry.simplify(0.001)
    return sites


# ---------------------------------------------------------------- CDSE
def search_with_backoff(client, **kw):
    for attempt in range(8):
        try:
            return [i.to_dict() for i in client.search(**kw).items()]
        except APIError as e:
            if "429" in str(e) or "Rate limit" in str(e):
                wait = min(120, 5 * 2 ** attempt)
                print(f"    rate limited, sleeping {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("CDSE kept rate-limiting; rerun later (cache resumes)")


def query_s1(client, geom, mode, cache_file: Path, datetime=None, pause=1.0):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    kw = dict(collections=["sentinel-1-grd"], intersects=mapping(geom),
              filter={"op": "=", "args": [{"property": "sar:instrument_mode"}, mode]},
              filter_lang="cql2-json", limit=100)
    if datetime:
        kw["datetime"] = datetime
    items = search_with_backoff(client, **kw)
    slim = [{"s1_id": i["id"], "geometry": i["geometry"],
             **{k: i["properties"].get(k) for k in
                ["datetime", "platform", "product:type", "sar:polarizations",
                 "view:incidence_angle", "sat:orbit_state", "sat:relative_orbit"]}}
            for i in items]
    cache_file.write_text(json.dumps(slim))
    time.sleep(pause)
    return slim


# ---------------------------------------------------------------- pairing
def co_pol(capella_pols: str, s1_pols) -> str | None:
    """Capella is single-pol (HH or VV); S1 SM carries HH+HV or VV+VH."""
    for p in (capella_pols or "").split(","):
        if p in (s1_pols or []):
            return p
    return None


def pair(cap, sm_items, site_id):
    if not sm_items:
        return []
    sm = gpd.GeoDataFrame(
        [{**{k: v for k, v in s.items() if k != "geometry"}, "geometry": shape(s["geometry"])}
         for s in sm_items], crs="EPSG:4326")
    cap_ea, sm_ea = cap.to_crs(EQUAL_AREA), sm.to_crs(EQUAL_AREA)
    rows = []
    for ci, c in cap_ea.iterrows():
        for si, s in sm_ea.iterrows():
            inter = c.geometry.intersection(s.geometry).area
            if inter <= 0:
                continue
            pol = co_pol(c["sar:polarizations"], s["sar:polarizations"])
            d_inc = (abs(c["view:incidence_angle"] - s["view:incidence_angle"])
                     if c["view:incidence_angle"] is not None and s["view:incidence_angle"] is not None
                     else None)
            dt_days = abs((pd.Timestamp(c["datetime"]) - pd.Timestamp(s["datetime"])).days)
            rows.append({
                "site_id": site_id, "capella_id": c["capella_id"], "s1_id": s["s1_id"],
                "capella_dt": c["datetime"], "s1_dt": s["datetime"], "dt_days": dt_days,
                "capella_mode": c["sar:instrument_mode"], "s1_product": s["product:type"],
                "capella_pol": c["sar:polarizations"], "s1_pol": ",".join(s["sar:polarizations"] or []),
                "co_pol": pol,
                "capella_inc": c["view:incidence_angle"], "s1_inc": s["view:incidence_angle"],
                "d_inc": d_inc,
                "capella_orbit": c["sat:orbit_state"], "s1_orbit": s["sat:orbit_state"],
                "same_orbit_dir": c["sat:orbit_state"] == s["sat:orbit_state"],
                "capella_cover": inter / c.geometry.area,
                "lon": cap.loc[ci].geometry.centroid.x, "lat": cap.loc[ci].geometry.centroid.y,
            })
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s1sm_capella")
    ap.add_argument("--merge-m", type=float, default=20000,
                    help="merge Capella footprints closer than this into one site (m)")
    ap.add_argument("--max-d-inc", type=float, default=5.0, help="'good' pair: max incidence gap (deg)")
    ap.add_argument("--max-dt-days", type=int, default=365, help="'good' pair: max time gap (days)")
    ap.add_argument("--min-cover", type=float, default=0.5, help="'good' pair: min fraction of Capella footprint covered")
    ap.add_argument("--with-iw", action="store_true",
                    help="count S1 IW scenes within +/- --iw-window days of each good SM scene")
    ap.add_argument("--iw-window", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-sites", type=int, default=None, help="debug: only query first N sites")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "cdse_cache").mkdir(parents=True, exist_ok=True)

    cap = load_capella(out, a.workers)
    sites = make_sites(cap, a.merge_m)
    cap = gpd.sjoin(cap, sites[["site_id", "geometry"]], predicate="intersects", how="left")
    cap = cap.drop(columns="index_right").drop_duplicates("capella_id")
    print(f"{len(cap)} Capella GEO items in {len(sites)} sites")

    client = Client.open(CDSE_STAC)
    sites_iter = sites.head(a.limit_sites) if a.limit_sites else sites
    all_pairs, sm_counts = [], {}
    for n, site in enumerate(sites_iter.itertuples(), 1):
        sm = query_s1(client, site.geometry, "SM",
                      out / "cdse_cache" / f"site_{site.site_id:04d}_SM.json")
        sm_counts[site.site_id] = len(sm)
        rows = pair(cap[cap.site_id == site.site_id], sm, site.site_id)
        all_pairs += rows
        if sm:
            print(f"  [{n}/{len(sites_iter)}] site {site.site_id}: {len(sm)} SM scenes, {len(rows)} pairs")

    pairs = pd.DataFrame(all_pairs)
    if pairs.empty:
        print("No Capella x S1 SM overlaps found.")
        return
    pairs["good"] = (pairs.co_pol.notna() & (pairs.d_inc <= a.max_d_inc)
                     & (pairs.dt_days <= a.max_dt_days) & (pairs.capella_cover >= a.min_cover))

    if a.with_iw:
        good_sm = pairs[pairs.good].drop_duplicates("s1_id")
        iw_counts = {}
        for r in good_sm.itertuples():
            t = pd.Timestamp(r.s1_dt)
            win = f"{(t - pd.Timedelta(days=a.iw_window)).isoformat()}/{(t + pd.Timedelta(days=a.iw_window)).isoformat()}"
            geom = cap.loc[cap.capella_id == r.capella_id].geometry.iloc[0]
            iw = query_s1(client, geom, "IW", out / "cdse_cache" / f"iw_{r.s1_id}.json", datetime=win)
            iw_counts[r.s1_id] = sum(1 for i in iw if r.co_pol in (i["sar:polarizations"] or []))
        pairs["iw_co_pol_in_window"] = pairs.s1_id.map(iw_counts)

    pairs.sort_values(["good", "d_inc", "dt_days"], ascending=[False, True, True]).to_csv(
        out / "pairs.csv", index=False)

    sites["n_capella"] = sites.site_id.map(cap.site_id.value_counts()).fillna(0).astype(int)
    sites["n_sm"] = sites.site_id.map(sm_counts)
    sites["n_pairs"] = sites.site_id.map(pairs.site_id.value_counts()).fillna(0).astype(int)
    sites["n_good_pairs"] = sites.site_id.map(pairs[pairs.good].site_id.value_counts()).fillna(0).astype(int)
    sites.to_file(out / "sites.geojson", driver="GeoJSON")

    print(f"\n{len(pairs)} Capella x SM overlaps; {pairs.good.sum()} pass "
          f"(co-pol, d_inc<={a.max_d_inc}, dt<={a.max_dt_days}d, cover>={a.min_cover})")
    print(f"sites with SM coverage: {(sites.n_sm > 0).sum()} / {len(sites_iter)}; "
          f"with a good pair: {(sites.n_good_pairs > 0).sum()}")
    print(f"outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
