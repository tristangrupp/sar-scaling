"""Find where Sentinel-1 Stripmap (SM) archive footprints overlap ICEYE Open Data (X-band).

Mirrors s1sm_capella_overlap.py; only the X-band loader differs.

Steps:
  1. Crawl the ICEYE Open Data static STAC (collection iceye-sar, one item per collect,
     each carrying GRD / QLK and sometimes SLC / CSI assets) and cache item metadata.
  2. Merge overlapping ICEYE footprints into "sites" so each site costs one API query.
  3. For each site, search the Copernicus Data Space (CDSE) STAC API for S1 GRD items
     with sar:instrument_mode = SM, over the full archive.
  4. Pair every ICEYE item with every SM item that overlaps it, and score the pair on
     co-polarization match, incidence-angle gap, orbit direction and time gap.
  5. Optionally (--with-iw) count S1 IW scenes within +/- N days of each good SM scene.

Outputs (in --out):
  iceye_items.geojson   ICEYE footprints + SAR metadata (cache)
  sites.geojson         merged ICEYE sites with ICEYE / SM counts
  pairs.csv             one row per ICEYE x SM overlap, with match flags
  cdse_cache/           raw CDSE responses per site (lets reruns resume)

Run:  python s1sm_iceye_overlap.py --out ./s1sm_iceye --with-iw
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

ICEYE_COLL = "https://iceye-open-data-catalog.s3.amazonaws.com/collections/iceye-sar.json"
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
EQUAL_AREA = "EPSG:6933"

XB_KEEP = ["datetime", "platform", "sar:instrument_mode", "sar:polarizations",
           "view:incidence_angle", "sat:orbit_state", "sar:observation_direction",
           "view:off_nadir"]


# ---------------------------------------------------------------- ICEYE
def load_iceye(out: Path, workers: int) -> gpd.GeoDataFrame:
    cache = out / "iceye_items.geojson"
    if cache.exists():
        return gpd.read_file(cache)

    coll = requests.get(ICEYE_COLL, timeout=60).json()
    hrefs = sorted({urljoin(ICEYE_COLL, l["href"]) for l in coll["links"] if l["rel"] == "item"})
    print(f"ICEYE: fetching {len(hrefs)} item JSONs")
    session = requests.Session()

    def fetch(href):
        for attempt in range(4):
            try:
                r = session.get(href, timeout=60)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        print(f"  gave up on {href}")
        return None

    with ThreadPoolExecutor(workers) as ex:
        items = [i for i in ex.map(fetch, hrefs) if i]

    rows, seen = [], set()
    for it in items:
        # skip the catalog's summary "footprints" pseudo-item and anything not a SAR collect
        if (it["id"] in seen or not it.get("geometry")
                or not it["properties"].get("sar:instrument_mode")):
            continue
        seen.add(it["id"])
        p, assets = it["properties"], it.get("assets", {})
        row = {"xband_id": it["id"], "geometry": shape(it["geometry"])}
        row.update({k: p.get(k) for k in XB_KEEP})
        row["sar:polarizations"] = ",".join(p.get("sar:polarizations") or [])
        row["grd_href"] = (assets.get("grd-cog") or {}).get("href")
        row["assets"] = ",".join(sorted(assets))
        rows.append(row)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(cache, driver="GeoJSON")
    return gdf


def make_sites(xb: gpd.GeoDataFrame, merge_m: float) -> gpd.GeoDataFrame:
    """Merge footprints closer than merge_m into one site polygon."""
    ea = xb.to_crs(EQUAL_AREA)
    merged = unary_union(ea.geometry.buffer(merge_m / 2))
    parts = list(getattr(merged, "geoms", [merged]))
    sites = gpd.GeoDataFrame({"site_id": range(len(parts))}, geometry=parts, crs=EQUAL_AREA)
    sites = sites.to_crs("EPSG:4326")
    sites["geometry"] = sites.geometry.simplify(0.001)
    return sites


# ---------------------------------------------------------------- CDSE
def search_with_backoff(client, **kw):
    for attempt in range(10):
        try:
            return [i.to_dict() for i in client.search(**kw).items()]
        except (APIError, requests.RequestException) as e:
            msg = str(e)
            transient = ("429" in msg or "Rate limit" in msg or "50" in msg[:40]
                         or isinstance(e, requests.RequestException))
            if not transient:
                raise
            wait = min(180, 10 * 2 ** attempt)
            print(f"    CDSE error ({msg[:60]!r}), sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("CDSE kept failing; rerun later (cache resumes)")


def query_s1(client, geom, mode, cache_file: Path, datetime=None, pause=2.5):
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
def co_pol(xb_pols: str, s1_pols) -> str | None:
    """ICEYE is single-pol (VV, occasionally HH); S1 SM carries HH+HV or VV+VH."""
    for p in (xb_pols or "").split(","):
        if p and p in (s1_pols or []):
            return p
    return None


def _num(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def _ts(v):
    t = pd.Timestamp(v)
    return t.tz_convert(None) if t.tzinfo else t


def pair(xb, sm_items, site_id):
    if not sm_items or xb.empty:
        return []
    sm = gpd.GeoDataFrame(
        [{**{k: v for k, v in s.items() if k != "geometry"}, "geometry": shape(s["geometry"])}
         for s in sm_items], crs="EPSG:4326")
    xb_ea, sm_ea = xb.to_crs(EQUAL_AREA), sm.to_crs(EQUAL_AREA)
    rows = []
    for ci, c in xb_ea.iterrows():
        for si, s in sm_ea.iterrows():
            inter = c.geometry.intersection(s.geometry).area
            if inter <= 0:
                continue
            pol = co_pol(c["sar:polarizations"], s["sar:polarizations"])
            ci_inc, s_inc = _num(c["view:incidence_angle"]), _num(s["view:incidence_angle"])
            d_inc = abs(ci_inc - s_inc) if ci_inc is not None and s_inc is not None else None
            dt_days = abs((_ts(c["datetime"]) - _ts(s["datetime"])).days)
            xo, so = c["sat:orbit_state"], s["sat:orbit_state"]
            rows.append({
                "provider": "iceye",
                "site_id": site_id, "xband_id": c["xband_id"], "s1_id": s["s1_id"],
                "xband_dt": c["datetime"], "s1_dt": s["datetime"], "dt_days": dt_days,
                "xband_mode": c["sar:instrument_mode"], "s1_product": s["product:type"],
                "xband_pol": c["sar:polarizations"], "s1_pol": ",".join(s["sar:polarizations"] or []),
                "co_pol": pol,
                "xband_inc": ci_inc, "s1_inc": s_inc, "d_inc": d_inc,
                "xband_orbit": xo, "s1_orbit": so,
                "same_orbit_dir": (xo == so) if xo and so else None,
                "xband_cover": inter / c.geometry.area,
                "lon": xb.loc[ci].geometry.centroid.x, "lat": xb.loc[ci].geometry.centroid.y,
            })
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s1sm_iceye")
    ap.add_argument("--merge-m", type=float, default=20000)
    ap.add_argument("--max-d-inc", type=float, default=5.0)
    ap.add_argument("--max-dt-days", type=int, default=365)
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--with-iw", action="store_true")
    ap.add_argument("--iw-window", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between CDSE queries")
    ap.add_argument("--limit-sites", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "cdse_cache").mkdir(parents=True, exist_ok=True)

    xb = load_iceye(out, a.workers)
    sites = make_sites(xb, a.merge_m)
    xb = gpd.sjoin(xb, sites[["site_id", "geometry"]], predicate="intersects", how="left")
    xb = xb.drop(columns="index_right").drop_duplicates("xband_id")
    print(f"{len(xb)} ICEYE items in {len(sites)} sites; modes: "
          f"{xb['sar:instrument_mode'].value_counts().to_dict()}; pols: "
          f"{xb['sar:polarizations'].value_counts().to_dict()}")

    client = Client.open(CDSE_STAC)
    sites_iter = sites.head(a.limit_sites) if a.limit_sites else sites
    all_pairs, sm_counts = [], {}
    for n, site in enumerate(sites_iter.itertuples(), 1):
        sm = query_s1(client, site.geometry, "SM",
                      out / "cdse_cache" / f"site_{site.site_id:04d}_SM.json", pause=a.pause)
        sm_counts[site.site_id] = len(sm)
        rows = pair(xb[xb.site_id == site.site_id], sm, site.site_id)
        all_pairs += rows
        if sm:
            print(f"  [{n}/{len(sites_iter)}] site {site.site_id}: {len(sm)} SM scenes, {len(rows)} pairs",
                  flush=True)
        elif n % 25 == 0:
            print(f"  [{n}/{len(sites_iter)}] ...", flush=True)

    pairs = pd.DataFrame(all_pairs)
    sites["n_xband"] = sites.site_id.map(xb.site_id.value_counts()).fillna(0).astype(int)
    sites["n_sm"] = sites.site_id.map(sm_counts)
    if pairs.empty:
        sites["n_pairs"] = 0
        sites["n_good_pairs"] = 0
        sites.to_file(out / "sites.geojson", driver="GeoJSON")
        print("No ICEYE x S1 SM overlaps found.")
        return
    pairs["good"] = (pairs.co_pol.notna() & (pairs.d_inc <= a.max_d_inc)
                     & (pairs.dt_days <= a.max_dt_days) & (pairs.xband_cover >= a.min_cover))

    if a.with_iw:
        good_sm = pairs[pairs.good].drop_duplicates("s1_id")
        iw_counts = {}
        for r in good_sm.itertuples():
            t = pd.Timestamp(r.s1_dt)
            win = f"{(t - pd.Timedelta(days=a.iw_window)).isoformat()}/{(t + pd.Timedelta(days=a.iw_window)).isoformat()}"
            geom = xb.loc[xb.xband_id == r.xband_id].geometry.iloc[0]
            iw = query_s1(client, geom, "IW", out / "cdse_cache" / f"iw_{r.s1_id}.json",
                          datetime=win, pause=a.pause)
            iw_counts[r.s1_id] = sum(1 for i in iw if r.co_pol in (i["sar:polarizations"] or []))
        pairs["iw_co_pol_in_window"] = pairs.s1_id.map(iw_counts)

    pairs.sort_values(["good", "d_inc", "dt_days"], ascending=[False, True, True]).to_csv(
        out / "pairs.csv", index=False)

    sites["n_pairs"] = sites.site_id.map(pairs.site_id.value_counts()).fillna(0).astype(int)
    sites["n_good_pairs"] = sites.site_id.map(pairs[pairs.good].site_id.value_counts()).fillna(0).astype(int)
    sites.to_file(out / "sites.geojson", driver="GeoJSON")

    print(f"\n{len(pairs)} ICEYE x SM overlaps; {pairs.good.sum()} pass "
          f"(co-pol, d_inc<={a.max_d_inc}, dt<={a.max_dt_days}d, cover>={a.min_cover})")
    print(f"sites with SM coverage: {(sites.n_sm > 0).sum()} / {len(sites_iter)}; "
          f"with a good pair: {(sites.n_good_pairs > 0).sum()}")
    print(f"outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
