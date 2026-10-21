"""Find where Sentinel-1 Stripmap (SM) archive footprints overlap Umbra Open Data (X-band).

Mirrors s1sm_capella_overlap.py; output schema uses provider-neutral xband_* columns.

Steps:
  1. List the Umbra Open Data static STAC (s3://umbra-open-data-catalog/stac/) and fetch
     every item JSON. Keep the GEC (geocoded) product, one item per umbra:collect_id.
  2. Merge Umbra footprints closer than --merge-m into "sites" (one CDSE query each).
  3. Per site, search CDSE STAC for sentinel-1-grd with sar:instrument_mode = SM.
  4. Pair each Umbra item with each overlapping SM item; score co-pol, incidence gap,
     orbit direction, time gap, cover fraction.
  5. --with-iw: count co-pol S1 IW scenes within +/- --iw-window days of each good SM scene.

Outputs (in --out):
  umbra_items.geojson   Umbra footprints + SAR metadata (cache)
  sites.geojson         site_id, n_xband, n_sm, n_pairs, n_good_pairs
  pairs.csv             one row per Umbra x SM overlap (provider = "umbra")
  cdse_cache/           raw slimmed CDSE responses per site (reruns resume)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

BUCKET = "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog"
UMBRA_STAC = f"{BUCKET}/stac/catalog.json"
CDSE_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
EQUAL_AREA = "EPSG:6933"
PROVIDER = "umbra"

UMBRA_KEEP = ["datetime", "platform", "sar:instrument_mode", "sar:product_type",
              "sar:polarizations", "view:incidence_angle", "sat:orbit_state",
              "sar:observation_direction", "sar:resolution_range", "sar:resolution_azimuth",
              "umbra:collect_id", "umbra:task_id", "umbra:squint_angle_degrees_off_broadside",
              "umbra:squint_angle_engineering_degrees", "view:azimuth"]
ITEM_KEY = re.compile(r"^stac/.+/([0-9a-f-]{36})/\1\.json$")


# ---------------------------------------------------------------- Umbra
def list_item_keys(session) -> list[str]:
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": "stac/"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(session.get(BUCKET, params=params, timeout=60).content)
        keys += [k.text for k in root.findall("s3:Contents/s3:Key", ns) if ITEM_KEY.match(k.text)]
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def load_umbra(out: Path, workers: int) -> gpd.GeoDataFrame:
    cache = out / "umbra_items.geojson"
    if cache.exists():
        return gpd.read_file(cache)

    session = requests.Session()
    keys = list_item_keys(session)
    print(f"Umbra STAC: fetching {len(keys)} item JSONs")

    def fetch_url(url):
        for attempt in range(4):
            try:
                r = session.get(url, timeout=60)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        print(f"  gave up on {url}")
        return None

    def fetch(key):
        return fetch_url(f"{BUCKET}/{key}")

    with ThreadPoolExecutor(workers) as ex:
        items = [i for i in ex.map(fetch, keys) if i]

    rows = []
    for it in items:
        p = it["properties"]
        if not it.get("geometry"):
            continue
        geom = shape(it["geometry"])
        if geom.has_z:  # Umbra footprints carry height; drop it
            from shapely import wkb
            geom = wkb.loads(wkb.dumps(geom, output_dimension=2))
        row = {"xband_id": it["id"], "geometry": geom}
        row.update({k: p.get(k) for k in UMBRA_KEEP})
        row["sar:polarizations"] = ",".join(p.get("sar:polarizations") or [])
        meta = [a.get("href", "") for a in it.get("assets", {}).values()
                if "metadata" in (a.get("roles") or []) and a.get("href", "").endswith("METADATA.json")]
        row["meta_href"] = meta[0] if meta and meta[0].startswith("http") else None
        row["orbit_source"] = "stac" if p.get("sat:orbit_state") else None
        rows.append(row)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    n0 = len(gdf)
    # GEC (geocoded, detected) only; one item per collect
    gdf = gdf[gdf["sar:product_type"].fillna("GEC").str.upper() == "GEC"]
    key = gdf["umbra:collect_id"].fillna(gdf.xband_id)
    gdf = gdf.loc[~key.duplicated()].reset_index(drop=True)
    print(f"  {n0} items -> {len(gdf)} GEC collects after dedupe")

    # older items lack sat:orbit_state / view:azimuth; the small METADATA.json has them
    need = gdf.index[gdf["sat:orbit_state"].isna() & gdf.meta_href.notna()]
    print(f"  reading METADATA.json for {len(need)} items without orbit state")

    def track(i):
        m = fetch_url(gdf.at[i, "meta_href"])
        if not m:
            return i, None, None
        cs = m.get("collects") or [{}]
        c = next((c for c in cs if c.get("id") == gdf.at[i, "umbra:collect_id"]), cs[0])
        return i, c.get("satelliteTrack"), c.get("angleAzimuthDegrees")

    with ThreadPoolExecutor(workers) as ex:
        for i, trk, az in ex.map(track, need):
            if trk:
                gdf.at[i, "sat:orbit_state"] = trk.lower()
                gdf.at[i, "orbit_source"] = "metadata"
            if az is not None and pd.isna(gdf.at[i, "view:azimuth"]):
                gdf.at[i, "view:azimuth"] = az
    gdf.to_file(cache, driver="GeoJSON")
    return gdf


def derive_orbit(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill missing sat:orbit_state from view:azimuth (target->satellite) and look side.

    Left-looking: satellite east of target (az < 180) when ascending; right-looking: west.
    The rule reproduces all 1505 Umbra items that carry sat:orbit_state.
    """
    az = gdf["view:azimuth"] % 360
    left = gdf["sar:observation_direction"].str.lower() == "left"
    asc = (left & (az < 180)) | (~left & (az >= 180))
    derived = asc.map({True: "ascending", False: "descending"}).where(az.notna())
    fill = gdf["sat:orbit_state"].isna() & derived.notna()
    gdf.loc[fill, "orbit_source"] = "derived"
    gdf.loc[fill, "sat:orbit_state"] = derived[fill]
    return gdf


def make_sites(xb: gpd.GeoDataFrame, merge_m: float) -> gpd.GeoDataFrame:
    ea = xb.to_crs(EQUAL_AREA)
    merged = unary_union(ea.geometry.buffer(merge_m / 2))
    parts = list(getattr(merged, "geoms", [merged]))
    sites = gpd.GeoDataFrame({"site_id": range(len(parts))}, geometry=parts, crs=EQUAL_AREA)
    sites = sites.to_crs("EPSG:4326")
    sites["geometry"] = sites.geometry.simplify(0.001)
    return sites


# ---------------------------------------------------------------- CDSE
_session = requests.Session()


def post_with_backoff(body, pause):
    for attempt in range(10):
        try:
            r = _session.post(CDSE_SEARCH, json=body, timeout=120)
            if r.status_code == 429 or "Rate limit" in r.text[:500]:
                raise RuntimeError("429")
            if r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except (RuntimeError, requests.ConnectionError, requests.Timeout, ValueError) as e:
            wait = min(180, 10 * 2 ** attempt)
            print(f"    CDSE {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("CDSE kept failing; rerun later (cache resumes)")


def query_s1(geom, mode, cache_file: Path, datetime=None, pause=2.5):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    body = {"collections": ["sentinel-1-grd"], "intersects": mapping(geom),
            "filter-lang": "cql2-json",
            "filter": {"op": "=", "args": [{"property": "sar:instrument_mode"}, mode]},
            "limit": 100}
    if datetime:
        body["datetime"] = datetime
    items = []
    while body is not None:
        page = post_with_backoff(body, pause)
        items += page.get("features", [])
        nxt = [l for l in page.get("links", []) if l.get("rel") == "next"]
        body = nxt[0].get("body") if nxt and page.get("features") else None
    slim, seen = [], set()
    for i in items:
        stem = i["id"].replace("_COG", "")
        if stem in seen:  # CDSE can list both SAFE and COG copies
            continue
        seen.add(stem)
        slim.append({"s1_id": i["id"], "geometry": i["geometry"],
                     **{k: i["properties"].get(k) for k in
                        ["datetime", "platform", "product:type", "sar:polarizations",
                         "view:incidence_angle", "sat:orbit_state", "sat:relative_orbit"]}})
    cache_file.write_text(json.dumps(slim))
    return slim


# ---------------------------------------------------------------- pairing
def co_pol(xb_pols: str, s1_pols) -> str | None:
    for p in (xb_pols or "").split(","):
        if p and p in (s1_pols or []):
            return p
    return None


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


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
            ci_inc, s_inc = _num(c["view:incidence_angle"]), _num(s["view:incidence_angle"])
            d_inc = abs(ci_inc - s_inc) if ci_inc is not None and s_inc is not None else None
            dt_days = abs((pd.Timestamp(c["datetime"]) - pd.Timestamp(s["datetime"])).days)
            rows.append({
                "provider": PROVIDER,
                "site_id": site_id, "xband_id": c["xband_id"], "s1_id": s["s1_id"],
                "xband_dt": c["datetime"], "s1_dt": s["datetime"], "dt_days": dt_days,
                "xband_mode": c["sar:instrument_mode"], "s1_product": s["product:type"],
                "xband_pol": c["sar:polarizations"], "s1_pol": ",".join(s["sar:polarizations"] or []),
                "co_pol": co_pol(c["sar:polarizations"], s["sar:polarizations"]),
                "xband_inc": ci_inc, "s1_inc": s_inc, "d_inc": d_inc,
                "xband_orbit": c["sat:orbit_state"], "s1_orbit": s["sat:orbit_state"],
                "same_orbit_dir": c["sat:orbit_state"] == s["sat:orbit_state"],
                "xband_orbit_source": c["orbit_source"],
                "xband_cover": inter / c.geometry.area,
                "lon": xb.loc[ci].geometry.centroid.x, "lat": xb.loc[ci].geometry.centroid.y,
            })
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s1sm_umbra")
    ap.add_argument("--merge-m", type=float, default=20000)
    ap.add_argument("--max-d-inc", type=float, default=5.0)
    ap.add_argument("--max-dt-days", type=int, default=365)
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--with-iw", action="store_true")
    ap.add_argument("--iw-window", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between CDSE requests")
    ap.add_argument("--limit-sites", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "cdse_cache").mkdir(parents=True, exist_ok=True)

    xb = derive_orbit(load_umbra(out, a.workers))
    sites = make_sites(xb, a.merge_m)
    xb = gpd.sjoin(xb, sites[["site_id", "geometry"]], predicate="intersects", how="left")
    xb = xb.drop(columns="index_right").drop_duplicates("xband_id")
    print(f"{len(xb)} Umbra GEC items in {len(sites)} sites")

    sites_iter = sites.head(a.limit_sites) if a.limit_sites else sites
    all_pairs, sm_counts = [], {}
    for n, site in enumerate(sites_iter.itertuples(), 1):
        sm = query_s1(site.geometry, "SM",
                      out / "cdse_cache" / f"site_{site.site_id:04d}_SM.json", pause=a.pause)
        sm_counts[site.site_id] = len(sm)
        rows = pair(xb[xb.site_id == site.site_id], sm, site.site_id)
        all_pairs += rows
        if sm:
            print(f"  [{n}/{len(sites_iter)}] site {site.site_id}: {len(sm)} SM scenes, {len(rows)} pairs",
                  flush=True)

    pairs = pd.DataFrame(all_pairs)
    sites["n_xband"] = sites.site_id.map(xb.site_id.value_counts()).fillna(0).astype(int)
    sites["n_sm"] = sites.site_id.map(sm_counts)
    if pairs.empty:
        print("No Umbra x S1 SM overlaps found.")
        sites["n_pairs"] = 0
        sites["n_good_pairs"] = 0
        sites.to_file(out / "sites.geojson", driver="GeoJSON")
        return
    pairs["good"] = (pairs.co_pol.notna() & (pairs.d_inc <= a.max_d_inc)
                     & (pairs.dt_days <= a.max_dt_days) & (pairs.xband_cover >= a.min_cover))

    if a.with_iw:
        good_sm = pairs[pairs.good].drop_duplicates("s1_id")
        print(f"IW check for {len(good_sm)} good SM scenes")
        iw_counts = {}
        for r in good_sm.itertuples():
            t = pd.Timestamp(r.s1_dt)
            win = f"{(t - pd.Timedelta(days=a.iw_window)).isoformat()}/{(t + pd.Timedelta(days=a.iw_window)).isoformat()}"
            geom = xb.loc[xb.xband_id == r.xband_id].geometry.iloc[0]
            iw = query_s1(geom, "IW", out / "cdse_cache" / f"iw_{r.s1_id}.json",
                          datetime=win, pause=a.pause)
            iw_counts[r.s1_id] = sum(1 for i in iw if r.co_pol in (i["sar:polarizations"] or []))
        pairs["iw_co_pol_in_window"] = pairs.s1_id.map(iw_counts)

    pairs.sort_values(["good", "d_inc", "dt_days"], ascending=[False, True, True]).to_csv(
        out / "pairs.csv", index=False)

    sites["n_pairs"] = sites.site_id.map(pairs.site_id.value_counts()).fillna(0).astype(int)
    sites["n_good_pairs"] = sites.site_id.map(pairs[pairs.good].site_id.value_counts()).fillna(0).astype(int)
    sites.to_file(out / "sites.geojson", driver="GeoJSON")

    print(f"\n{len(pairs)} Umbra x SM overlaps; {pairs.good.sum()} pass "
          f"(co-pol, d_inc<={a.max_d_inc}, dt<={a.max_dt_days}d, cover>={a.min_cover})")
    print(f"sites with SM coverage: {(sites.n_sm > 0).sum()} / {len(sites_iter)}; "
          f"with a good pair: {(sites.n_good_pairs > 0).sum()}")
    print(f"outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
