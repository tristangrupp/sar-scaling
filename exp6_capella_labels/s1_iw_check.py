"""For every Capella GEO item, find Sentinel-1 IW GRD scenes within +/-12 days (CDSE STAC).

Capella scenes in one exp0 site are grouped by date so that one CDSE query covers every
scene whose +/-12-day windows overlap. Raw responses are cached in cache/s1iw/.
Output: s1_iw_match.csv (one row per Capella item).
"""
import json, os, sys, time
from pathlib import Path
import geopandas as gpd, pandas as pd
from shapely.geometry import mapping, shape
from pystac_client import Client
from pystac_client.exceptions import APIError

HERE = Path(__file__).parent
EXP0 = HERE.parent / "exp0_overlap_atlas" / "s1sm_capella"
WIN = pd.Timedelta(days=12)
CDSE = "https://stac.dataspace.copernicus.eu/v1"

cap = gpd.read_file(EXP0 / "capella_geo_items.geojson")
sites = gpd.read_file(EXP0 / "sites.geojson")
cap = gpd.sjoin(cap, sites[["site_id", "geometry"]], predicate="intersects", how="left")
cap = cap.drop(columns="index_right").drop_duplicates("capella_id").reset_index(drop=True)
cap["t"] = pd.to_datetime(cap["datetime"], utc=True)

client = Client.open(CDSE)

def search(geom, t0, t1):
    kw = dict(collections=["sentinel-1-grd"], intersects=mapping(geom),
              datetime=f"{t0.strftime('%Y-%m-%dT%H:%M:%SZ')}/{t1.strftime('%Y-%m-%dT%H:%M:%SZ')}",
              filter={"op": "=", "args": [{"property": "sar:instrument_mode"}, "IW"]},
              filter_lang="cql2-json", limit=200)
    for a in range(8):
        try:
            return [i.to_dict() for i in client.search(**kw).items()]
        except Exception as e:
            wait = min(120, 5 * 2 ** a)
            print(f"   retry {a} after {type(e).__name__}: {str(e)[:120]}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("CDSE failed repeatedly")

rows = []
groups = []
for sid, g in cap.groupby("site_id"):
    g = g.sort_values("t")
    cur = [g.index[0]]
    for i in g.index[1:]:
        if cap.at[i, "t"] - cap.at[cur[-1], "t"] <= 2 * WIN and cap.at[i, "t"] - cap.at[cur[0], "t"] <= pd.Timedelta(days=120):
            cur.append(i)
        else:
            groups.append((sid, cur)); cur = [i]
    groups.append((sid, cur))
print(f"{len(cap)} scenes, {len(groups)} CDSE queries", flush=True)

for n, (sid, idx) in enumerate(groups, 1):
    sub = cap.loc[idx]
    t0, t1 = sub.t.min() - WIN, sub.t.max() + WIN
    cf = HERE / "cache" / "s1iw" / f"site{int(sid):04d}_{t0:%Y%m%d}_{t1:%Y%m%d}.json"
    if cf.exists():
        items = json.loads(cf.read_text())
    else:
        geom = sub.geometry.union_all().convex_hull.simplify(0.001)
        raw = search(geom, t0, t1)
        items = [{"id": i["id"], "geometry": i["geometry"],
                  **{k: i["properties"].get(k) for k in ["datetime", "sar:polarizations", "sat:orbit_state",
                                                          "sat:relative_orbit", "product:type"]}} for i in raw]
        cf.write_text(json.dumps(items)); time.sleep(0.3)
    if n % 25 == 0:
        print(f"  [{n}/{len(groups)}] site {sid}: {len(items)} IW items", flush=True)
    # keep one product per acquisition (CDSE lists _COG and non-COG twins)
    s1 = [dict(it, geom=shape(it["geometry"]), t=pd.Timestamp(it["datetime"])) for it in items]
    for i, c in sub.iterrows():
        cg = c.geometry
        hits = []
        for s in s1:
            dt = (s["t"] - c.t).total_seconds() / 86400
            if abs(dt) > 12: continue
            cov = s["geom"].intersection(cg).area / cg.area if cg.area > 0 else 0
            if cov < 0.5: continue
            hits.append((dt, s, cov))
        acq = {}
        for dt, s, cov in hits:
            acq.setdefault(s["datetime"][:19], (dt, s, cov))
        pols = sorted({",".join(s["sar:polarizations"] or []) for _, s, _ in acq.values()})
        best = min(acq.values(), key=lambda x: abs(x[0])) if acq else None
        rows.append({"capella_id": c.capella_id, "site_id": sid,
                     "s1_iw_n_acq_pm12d": len(acq),
                     "s1_iw_full_cover_n": sum(1 for v in acq.values() if v[2] > 0.99),
                     "s1_iw_nearest_dt_days": round(best[0], 2) if best else None,
                     "s1_iw_nearest_id": best[1]["id"] if best else None,
                     "s1_iw_pols": ";".join(pols),
                     "s1_iw_has_vv_vh": any("VV" in (s["sar:polarizations"] or []) and "VH" in (s["sar:polarizations"] or []) for _, s, _ in acq.values())})

pd.DataFrame(rows).to_csv(HERE / "s1_iw_match.csv", index=False)
print("done", len(rows), flush=True)
os._exit(0)
