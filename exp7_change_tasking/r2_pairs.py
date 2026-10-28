"""Round 2 pairing: Capella repeat pairs bracketed by two same-track Sentinel-1 passes.

Round 2 asks how small a change Sentinel-1 catches, using Capella change as truth. For recall,
the Sentinel-1 interval must contain the Capella interval:
    t(s_b) <= t(x1) < t(x2) <= t(s_a),   same S1 track (relative orbit + direction)
so any lasting change between the two Capella images also lies between the two S1 passes. This is
the reverse of pairs.py (whose tasking_pairs.csv puts the S1 passes inside the X-band interval), so
the pairs are rebuilt from xband_collects.csv. tasking_pairs.csv is counted for reference.

Capella geometry comes from each collect's STAC item (cached in data/capella_items.json):
polarization, observation direction (left/right), orbit state, view:azimuth, incidence, and the
repeat-ground-track id. A pair needs the same polarization, look direction and orbit state, an
azimuth within 10 deg, an incidence difference within DINC, and footprints overlapping >= 80%.

S1 passes: Planetary Computer sentinel-1-rtc metadata over each Capella repeat site, from 15 days
before its first collect to 15 days after its last (s1_passes.csv stops at the last collect, so it
misses "after" passes). Cached in data/r2_s1_passes.csv.

Writes r2_pair_candidates.csv (every qualifying pair at the loosest bounds), r2_pairs.csv (the
selected run set) and prints the count tables used in r2_pair_count.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROJ = ROOT.parent
CAPELLA_COLL = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
                "capella-open-data-by-product-type/capella-open-data-geo/collection.json")
CAPELLA_COLL_CACHE = PROJ / "exp1_gan_spike" / "capella_geo_collection.json"   # read-only
ITEMS = DATA / "capella_items.json"
S1P = DATA / "r2_s1_passes.csv"
MAX_GAP, DAZ, MIN_OV = 60, 10.0, 0.8
DINCS, WINDOWS = (1.0, 2.0, 5.0), (3, 6, 12)
CHOSEN_DINC, CHOSEN_W, MAX_XGAP_SEL, CAP_PER_SITE = 2.0, 6, 30, 5


def fetch_items(ids):
    known = json.loads(ITEMS.read_text()) if ITEMS.exists() else {}
    need = [i for i in ids if i not in known]
    if need:
        coll = json.loads(CAPELLA_COLL_CACHE.read_text())
        links = {Path(l["href"]).stem: urljoin(CAPELLA_COLL, l["href"]) for l in coll["links"] if l["rel"] == "item"}
        s = requests.Session()

        def one(i):
            if i not in links:
                return i, None
            for k in range(4):
                try:
                    it = s.get(links[i], timeout=60).json()
                    p = it["properties"]
                    tif = [a["href"] for a in it["assets"].values() if a["href"].endswith(".tif") and "preview" not in a["href"]]
                    keep = ["datetime", "sar:polarizations", "sar:observation_direction", "sat:orbit_state",
                            "sat:relative_orbit", "view:azimuth", "view:incidence_angle", "capella:rgt_id",
                            "capella:orbital_plane", "sar:instrument_mode", "capella:use_case", "capella:squint_angle",
                            "capella:layover_angle", "sar:pixel_spacing_range", "proj:epsg", "proj:shape"]
                    return i, {**{k2: p.get(k2) for k2 in keep}, "href": tif[0] if tif else None}
                except Exception:
                    time.sleep(3 * (k + 1))
            return i, None
        with ThreadPoolExecutor(12) as ex:
            for i, v in ex.map(one, need):
                known[i] = v
        ITEMS.write_text(json.dumps(known))
    return known


def fetch_s1(sites):
    if S1P.exists():
        return pd.read_csv(S1P)
    from pystac_client import Client
    c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    fields = {"include": ["properties.datetime", "properties.sat:relative_orbit", "properties.sat:orbit_state",
                          "properties.platform"], "exclude": ["assets", "links", "geometry"]}

    def one(r):
        t0 = pd.Timestamp(r.first) - pd.Timedelta(days=15)
        t1 = pd.Timestamp(r.last) + pd.Timedelta(days=15)
        for k in range(4):
            try:
                its = c.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [r.lon, r.lat]},
                               datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}T23:59:59Z", fields=fields).items_as_dicts()
                return [{"site_id": r.site_id, "t": i["properties"]["datetime"], "rel_orbit": i["properties"].get("sat:relative_orbit"),
                         "orbit_state": i["properties"].get("sat:orbit_state"), "platform": i["properties"].get("platform")}
                        for i in its]
            except Exception as e:
                time.sleep(5 * (k + 1))
                err = e
        print("S1 search failed", r.site_id, err, flush=True)
        return []
    with ThreadPoolExecutor(6) as ex:
        rows = [x for lst in ex.map(one, sites.itertuples()) for x in lst]
    p = pd.DataFrame(rows)
    p["t"] = pd.to_datetime(p.t, utc=True, format="ISO8601")
    p["date"] = p.t.dt.strftime("%Y-%m-%d")
    p = p.sort_values("t").drop_duplicates(["site_id", "date", "rel_orbit", "orbit_state"])
    p.to_csv(S1P, index=False)
    return p


def main():
    DATA.mkdir(exist_ok=True)
    col = pd.read_csv(ROOT / "xband_collects.csv")
    col = col[col.provider == "capella"].copy()
    col["d"] = pd.to_datetime(col.datetime, utc=True, format="ISO8601").dt.tz_localize(None)
    days = col.groupby("site_id").day.nunique()
    col = col[col.site_id.isin(days[days >= 2].index)]
    print(f"Capella collects at repeat sites: {len(col)} at {col.site_id.nunique()} sites", flush=True)
    items = fetch_items(sorted(col.xband_id))
    col["obs"] = col.xband_id.map(lambda i: (items.get(i) or {}).get("sar:observation_direction"))
    col["vaz"] = col.xband_id.map(lambda i: (items.get(i) or {}).get("view:azimuth"))
    col["vinc"] = col.xband_id.map(lambda i: (items.get(i) or {}).get("view:incidence_angle"))
    col["ostate"] = col.xband_id.map(lambda i: (items.get(i) or {}).get("sat:orbit_state"))
    col["rgt"] = col.xband_id.map(lambda i: (items.get(i) or {}).get("capella:rgt_id"))
    col["href"] = col.xband_id.map(lambda i: (items.get(i) or {}).get("href"))
    print("items with metadata:", col.obs.notna().sum(), "with href:", col.href.notna().sum(), flush=True)
    foot = gpd.read_file(ROOT / "xband_collects.geojson").set_index("xband_id").geometry
    foot = foot[foot.index.isin(col.xband_id)].to_crs(3857)

    sites = col.groupby("site_id").agg(first=("day", "min"), last=("day", "max"), lon=("lon", "first"),
                                       lat=("lat", "first")).reset_index()
    p = fetch_s1(sites)
    p["d"] = pd.to_datetime(p.t, utc=True, format="ISO8601").dt.tz_localize(None)
    print(f"S1 passes: {len(p)} at {p.site_id.nunique()} of {len(sites)} Capella repeat sites", flush=True)

    rows = []
    for sid, s in col.groupby("site_id"):
        s = s.sort_values("d").reset_index(drop=True)
        ps = p[p.site_id == sid]
        tracks = [(k, np.array(sorted(g.d.unique()), dtype="datetime64[s]")) for k, g in ps.groupby(["rel_orbit", "orbit_state"])]
        if not tracks:
            continue
        d = s.d.values.astype("datetime64[s]")
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                a, b = s.iloc[i], s.iloc[j]
                if a.day == b.day:
                    continue
                gap = (d[j] - d[i]).astype(int) / 86400
                if gap > MAX_GAP:
                    break
                if str(a.pol) != str(b.pol) or a.obs != b.obs or a.ostate != b.ostate or a.obs is None:
                    continue
                daz = abs((a.vaz - b.vaz + 180) % 360 - 180) if pd.notna(a.vaz) and pd.notna(b.vaz) else np.nan
                dinc = abs(a.vinc - b.vinc) if pd.notna(a.vinc) and pd.notna(b.vinc) else np.nan
                if not (np.isfinite(daz) and daz <= DAZ and np.isfinite(dinc) and dinc <= max(DINCS)):
                    continue
                best = None
                for (ro, st), td in tracks:
                    pb, pa = td[td <= d[i]], td[td >= d[j]]
                    if not len(pb) or not len(pa):
                        continue
                    sb, sa = pb[-1], pa[0]
                    slack_b, slack_a = (d[i] - sb).astype(int) / 86400, (sa - d[j]).astype(int) / 86400
                    key = max(slack_b, slack_a)
                    if best is None or key < best[0]:
                        best = (key, sb, sa, ro, st, slack_b, slack_a)
                if best is None or best[0] > max(WINDOWS):
                    continue
                fa, fb = foot.loc[a.xband_id], foot.loc[b.xband_id]
                ov = fa.intersection(fb).area / min(fa.area, fb.area)
                if ov < MIN_OV:
                    continue
                _, sb, sa, ro, st, slb, sla = best
                rows.append({"site_id": sid, "xid1": a.xband_id, "xid2": b.xband_id, "x1": a.day, "x2": b.day,
                             "x1_time": str(d[i]), "x2_time": str(d[j]), "x_gap_d": round(gap, 2), "pol": a.pol,
                             "mode1": a["mode"], "mode2": b["mode"], "obs": a.obs, "x_orbit": a.ostate,
                             "d_inc": round(float(dinc), 2), "d_az": round(float(daz), 1), "same_rgt": a.rgt == b.rgt,
                             "inc": a.vinc, "ov": round(float(ov), 3),
                             "s1_before_time": str(sb), "s1_after_time": str(sa), "s1_before": str(sb)[:10], "s1_after": str(sa)[:10],
                             "s1_gap_d": round((sa - sb).astype(int) / 86400, 1), "slack_before_d": round(slb, 2),
                             "slack_after_d": round(sla, 2), "rel_orbit": int(ro), "orbit_state": st,
                             "href1": a.href, "href2": b.href, "lon": a.lon, "lat": a.lat})
    c = pd.DataFrame(rows)
    c["slack_max"] = c[["slack_before_d", "slack_after_d"]].max(axis=1)
    c.to_csv(ROOT / "r2_pair_candidates.csv", index=False)

    # ---- count tables
    print("\n== counts: pairs (sites) by incidence bound x S1 window ==")
    tab = {}
    for di in DINCS:
        for w in WINDOWS:
            k = c[(c.d_inc <= di) & (c.slack_max <= w)]
            tab[(di, w)] = (len(k), k.site_id.nunique())
    print(pd.DataFrame({f"{w} d": [f"{tab[(di, w)][0]} ({tab[(di, w)][1]})" for di in DINCS] for w in WINDOWS},
                       index=[f"dinc<={di}" for di in DINCS]).to_string())
    for g in (7, 14, 30):
        k = c[(c.d_inc <= CHOSEN_DINC) & (c.slack_max <= CHOSEN_W) & (c.x_gap_d <= g)]
        print(f"Capella gap <= {g} d at chosen bounds: {len(k)} pairs, {k.site_id.nunique()} sites")
    k = c[(c.d_inc <= CHOSEN_DINC) & (c.slack_max <= CHOSEN_W)]
    print("chosen bounds: same rgt share", round(k.same_rgt.mean(), 3), "x gap median", k.x_gap_d.median(),
          "S1 gap median", k.s1_gap_d.median(), "pol", k.pol.value_counts().to_dict(), "mode", k.mode1.value_counts().to_dict())

    tp = pd.read_csv(ROOT / "tasking_pairs.csv")
    tp = tp[tp.provider == "capella"]
    print(f"\nreference, tasking_pairs.csv Capella (S1 inside the X-band interval): {len(tp)} pairs, {tp.site_id.nunique()} sites")

    # ---- selection: one pair per site per season, shortest Capella gap first, cap per site
    k = c[(c.d_inc <= CHOSEN_DINC) & (c.slack_max <= CHOSEN_W) & (c.x_gap_d <= MAX_XGAP_SEL) & c.href1.notna() & c.href2.notna()].copy()
    k["season"] = pd.to_datetime(k.x1).dt.year.astype(str) + "Q" + pd.to_datetime(k.x1).dt.quarter.astype(str)
    k = k.sort_values(["x_gap_d", "slack_max", "d_inc", "ov"], ascending=[True, True, True, False])
    sel = k.groupby(["site_id", "season"]).head(1)
    sel = sel.sort_values(["site_id", "x_gap_d"]).groupby("site_id").head(CAP_PER_SITE)
    sel = sel.sort_values(["site_id", "x1"]).reset_index(drop=True)
    sel.to_csv(ROOT / "r2_pairs.csv", index=False)
    print(f"\nselected: {len(sel)} pairs at {sel.site_id.nunique()} sites (site-seasons available: "
          f"{k.groupby(['site_id', 'season']).ngroups}); per-site counts: {sel.site_id.value_counts().value_counts().sort_index().to_dict()}")
    print("selected x gap median/max:", sel.x_gap_d.median(), sel.x_gap_d.max(), "slack median:", sel.slack_max.median())
    pd.set_option("display.width", 250)
    print(sel[["site_id", "x1", "x2", "x_gap_d", "pol", "mode1", "d_inc", "s1_before", "s1_after", "s1_gap_d", "lon", "lat"]].to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
