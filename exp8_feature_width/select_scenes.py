"""Step 1: pick scenes, place 4 km windows by OSM candidate length, find Sentinel-1 RTC items.

Rule (DESIGN.md "Sites"): exp6 items with fields_present and S1 VV+VH; one item per site (finest range
resolution, then smallest Capella-S1 gap); OSM lines over the footprint within a 12 km box; 4 km window on a
1 km grid with >= 60% inside the footprint maximizing candidate length; drop < 15 km; <= 4 sites per
country; up to 32 scenes.

OSM lines come from Overture Maps (release OVR), whose base and transportation themes are built from
OpenStreetMap; the public Overpass servers were at capacity on 2026-10-28.
Writes scenes_selected.csv, windows.geojson, cache/overture/<capella_id>.parquet.
Needs duckdb, installed outside the env (DUCKDB_PATH).
"""
import json, os, sys, traceback
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
import shapely.ops
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as stransform

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
OVDIR = HERE / "cache" / "overture"
OVR = "2026-10-21.0"
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", r"C:\Users\grupp\AppData\Local\Temp\claude\C--Users-grupp"
                             r"\335d88d1-5193-48a0-8750-c34519c54d86\scratchpad\pylib")
WIN, HALF_BOX = 4000.0, 6000.0
CAP_COLL = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
            "capella-open-data-by-product-type/capella-open-data-geo/collection.json")

PAVED = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link",
         "secondary_link", "tertiary_link", "unclassified", "residential", "service"}
TRACK = {"track", "path", "footway", "cycleway", "bridleway"}
DITCH = {"ditch", "drain", "canal"}
LANDUSE = ("farmland", "meadow", "orchard", "vineyard", "grass")
_CON = None


def _con():
    global _CON
    if _CON is None:
        sys.path.insert(0, DUCKDB_PATH)
        import duckdb
        _CON = duckdb.connect()
        _CON.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; SET threads=8; SET enable_progress_bar=false;")
    return _CON


def _q(theme, typ, cols, where, bb):
    w, s, e, n = bb
    sql = (f"SELECT id, class AS cls, {cols} ST_AsWKB(geometry) g FROM read_parquet('s3://overturemaps-us-west-2/release/"
           f"{OVR}/theme={theme}/type={typ}/*', hive_partitioning=0) WHERE bbox.xmin<{e} AND bbox.xmax>{w} "
           f"AND bbox.ymin<{n} AND bbox.ymax>{s} AND ({where})")
    return _con().execute(sql).df()


def _flag_pieces(line, flags):
    """Drop the parts of a segment flagged bridge/tunnel/covered (Overture road_flags with 'between')."""
    bad = []
    for f in flags:
        if any(v in ("is_bridge", "is_tunnel", "is_covered", "is_indoor") for v in (f.get("values") or [])):
            b = f.get("between")
            bad.append((0.0, 1.0) if b is None else (float(b[0]), float(b[1])))
    if not bad:
        return [line]
    bad.sort()
    keep, t = [], 0.0
    for a, b in bad:
        if a > t:
            keep.append((t, a))
        t = max(t, b)
    if t < 1:
        keep.append((t, 1.0))
    return [shapely.ops.substring(line, a, b, normalized=True) for a, b in keep if b - a > 1e-4]


def fetch_lines(cid, bb):
    """Lines inside bb=(w,s,e,n), cached. Columns osm_id (Overture id), type, candidate, tags, geometry (4326).
    type: tree_line, road, track, ditch_canal, landuse (outline), other_line (neighbor only)."""
    f = OVDIR / f"{cid}.parquet"
    if f.exists():
        return gpd.read_parquet(f)
    rows = []
    seg = _q("transportation", "segment", "subtype, road_flags,", "subtype IN ('road','rail')", bb)
    for r in seg.itertuples():
        g = shapely.from_wkb(bytes(r.g))
        t = "other_line" if r.subtype == "rail" else ("road" if r.cls in PAVED else ("track" if r.cls in TRACK else "other_line"))
        fl = [dict(x) for x in r.road_flags] if isinstance(r.road_flags, (list, tuple, np.ndarray)) else []
        for piece in _flag_pieces(g, fl):
            rows.append((r.id, t, t != "other_line", r.cls, piece))
    wat = _q("base", "water", "subtype, source_tags,", "class IN ('ditch','drain','canal','stream','river','tidal_channel')", bb)
    for r in wat.itertuples():
        st = dict(r.source_tags) if r.source_tags is not None else {}
        if any(st.get(k) not in (None, "no") for k in ("bridge", "tunnel", "culvert", "covered")):
            continue
        t = "ditch_canal" if r.cls in DITCH else "other_line"
        rows.append((r.id, t, t == "ditch_canal", r.cls, shapely.from_wkb(bytes(r.g))))
    inf = _q("base", "infrastructure", "subtype,", "class IN ('hedge','wall','retaining_wall','city_wall')", bb)
    for r in inf.itertuples():
        g = shapely.from_wkb(bytes(r.g))
        if g.geom_type in ("Polygon", "MultiPolygon"):
            g = g.boundary
        t = "tree_line" if r.cls == "hedge" else "other_line"
        rows.append((r.id, t, t == "tree_line", r.cls, g))
    land = _q("base", "land", "subtype,", "class = 'tree_row'", bb)
    for r in land.itertuples():
        rows.append((r.id, "tree_line", True, r.cls, shapely.from_wkb(bytes(r.g))))
    lu = _q("base", "land_use", "subtype,", "class IN ('" + "','".join(LANDUSE) + "')", bb)
    for r in lu.itertuples():
        g = shapely.from_wkb(bytes(r.g))
        if g.geom_type in ("Polygon", "MultiPolygon"):
            rows.append((r.id, "landuse", True, r.cls, g.boundary))
    out = []
    for oid, t, c, cls, g in rows:
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "LineString" and len(part.coords) >= 2:
                out.append({"osm_id": oid, "ftype": t, "candidate": c, "tags": json.dumps({"class": cls}), "geometry": part})
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=4326) if out else gpd.GeoDataFrame(
        {"osm_id": [], "ftype": [], "candidate": [], "tags": []}, geometry=[], crs=4326)
    gdf.to_parquet(f)
    return gdf


def osm_lines(g4326, epsg):
    if g4326 is None or g4326.empty:
        return None
    return g4326.to_crs(epsg)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def main():
    OVDIR.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(PROJ / "exp6_capella_labels" / "scenes.csv")
    cap = gpd.read_file(PROJ / "exp0_overlap_atlas" / "s1sm_capella" / "capella_geo_items.geojson").set_index("capella_id")
    c = sc[sc.fields_present & sc.s1_iw_has_vv_vh].copy()
    c["adt"] = c.s1_iw_nearest_dt_days.abs()
    c = c.sort_values(["site_id", "res_range_m", "adt"]).groupby("site_id").head(1)
    print("candidate sites", len(c), flush=True)
    rows = []
    for r in c.itertuples():
        try:
            foot = cap.loc[r.capella_id].geometry
            cen = foot.representative_point()
            epsg = utm_epsg(cen.x, cen.y)
            fw = Transformer.from_crs(4326, epsg, always_xy=True)
            inv = Transformer.from_crs(epsg, 4326, always_xy=True)
            fu = stransform(fw.transform, foot)
            cx, cy = fw.transform(cen.x, cen.y)
            qb = fu.intersection(box(cx - HALF_BOX, cy - HALF_BOX, cx + HALF_BOX, cy + HALF_BOX))
            ll = [inv.transform(x, y) for x, y in [(qb.bounds[0], qb.bounds[1]), (qb.bounds[2], qb.bounds[3]),
                                                   (qb.bounds[0], qb.bounds[3]), (qb.bounds[2], qb.bounds[1])]]
            lons, lats = [p[0] for p in ll], [p[1] for p in ll]
            src = fetch_lines(r.capella_id, (min(lons), min(lats), max(lons), max(lats)))
            g = osm_lines(src, epsg)
            best = (0.0, None)
            if g is not None:
                # rural score: hedges/tree rows, farm tracks (class track only), ditches/canals and the outlines of
                # farmland/meadow/orchard/vineyard; urban footways, streets and lawns don't count toward placement
                cls = g.tags.str.extract(r'"class": "([^"]+)"')[0]
                cand = g[g.ftype.isin(["tree_line", "ditch_canal"]) | ((g.ftype == "track") & (cls == "track"))
                         | ((g.ftype == "landuse") & cls.isin(["farmland", "meadow", "orchard", "vineyard"]))]
                x0, y0, x1, y1 = qb.bounds
                for xc in np.arange(np.floor(x0 / 1000) * 1000, x1 + 1, 1000):
                    for yc in np.arange(np.floor(y0 / 1000) * 1000, y1 + 1, 1000):
                        wb = box(xc - WIN / 2, yc - WIN / 2, xc + WIN / 2, yc + WIN / 2)
                        if wb.intersection(fu).area < 0.6 * wb.area:
                            continue
                        L = float(cand.geometry.intersection(wb).length.sum())
                        if L > best[0]:
                            best = (L, (xc, yc))
            rows.append({**r._asdict(), "epsg": epsg, "osm_lines": len(src),
                         "cand_km": round(best[0] / 1000, 1),
                         "win_x0": best[1][0] - WIN / 2 if best[1] else np.nan,
                         "win_y1": best[1][1] + WIN / 2 if best[1] else np.nan})
            print(r.site_id, r.country, r.capella_id, "lines", len(src), "cand_km", round(best[0] / 1000, 1), flush=True)
        except Exception as ex:
            traceback.print_exc()
            rows.append({**r._asdict(), "error": str(ex)[:200]})
    d = pd.DataFrame(rows).drop(columns=["Index"])
    d.to_csv(HERE / "cache" / "site_candidates.csv", index=False)
    d = d[d.cand_km >= 15].sort_values("cand_km", ascending=False)
    d = d.groupby("country").head(4).head(32).reset_index(drop=True)
    print("selected", len(d), flush=True)

    coll = json.loads((PROJ / "exp1_gan_spike" / "capella_geo_collection.json").read_text())
    links = {Path(l["href"]).stem: urljoin(CAP_COLL, l["href"]) for l in coll["links"] if l["rel"] == "item"}
    if not set(d.capella_id) <= set(links):   # the exp1 cache predates some items: use the live collection
        coll = requests.get(CAP_COLL, timeout=120).json()
        links.update({Path(l["href"]).stem: urljoin(CAP_COLL, l["href"]) for l in coll["links"] if l["rel"] == "item"})
    hrefs = []
    for cid in d.capella_id:
        if cid in links:
            try:
                it = requests.get(links[cid], timeout=60).json()
                tif = [a["href"] for k, a in it["assets"].items() if a["href"].endswith(".tif") and "preview" not in a["href"]]
                hrefs.append(tif[0])
                continue
            except Exception as ex:
                print(cid, "item JSON failed:", str(ex)[:80], flush=True)
        # not in the GEO collection index: the bucket layout is data/YYYY/M/D/<id>/<id>.tif
        ts = cid.split("_")[5]
        u = f"https://capella-open-data.s3.amazonaws.com/data/{int(ts[:4])}/{int(ts[4:6])}/{int(ts[6:8])}/{cid}/{cid}.tif"
        ok = requests.head(u, timeout=60).status_code == 200
        print(cid, "href by bucket layout", ok, flush=True)
        hrefs.append(u if ok else "")
    d["capella_href"] = hrefs
    d = d[d.capella_href != ""].reset_index(drop=True)

    import planetary_computer as pc
    from pystac_client import Client
    cl = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    single, multi, info = [], [], []
    for r in d.itertuples():
        inv = Transformer.from_crs(int(r.epsg), 4326, always_xy=True)
        m = 200
        wpoly = shapely.Polygon([inv.transform(x, y) for x, y in
                                 [(r.win_x0 - m, r.win_y1 + m), (r.win_x0 + WIN + m, r.win_y1 + m),
                                  (r.win_x0 + WIN + m, r.win_y1 - WIN - m), (r.win_x0 - m, r.win_y1 - WIN - m)]])
        t0 = pd.Timestamp(r.date)
        its = list(cl.search(collections=["sentinel-1-rtc"], intersects=wpoly.__geo_interface__,
                             datetime=f"{(t0 - pd.Timedelta(days=30)).date()}/{(t0 + pd.Timedelta(days=31)).date()}").items())
        its = [i for i in its if shape(i.geometry).contains(wpoly)]
        if not its:
            single.append(""); multi.append(""); info.append({}); continue
        dt = lambda i: abs((pd.Timestamp(i.datetime).tz_localize(None) - t0).total_seconds()) / 86400
        near = min(its, key=dt)
        rel, st = near.properties["sat:relative_orbit"], near.properties["sat:orbit_state"]
        same = sorted([i for i in its if i.properties["sat:relative_orbit"] == rel and i.properties["sat:orbit_state"] == st
                       and dt(i) <= 30], key=lambda i: i.datetime)
        single.append(near.id); multi.append(";".join(i.id for i in same))
        info.append({"s1_dt_days": round((pd.Timestamp(near.datetime).tz_localize(None) - t0).total_seconds() / 86400, 2),
                     "s1_rel_orbit": rel, "s1_orbit_state": st, "s1_n_multi": len(same),
                     "s1_multi_maxdt_days": round(max(dt(i) for i in same), 1)})
        print(r.capella_id, near.id, info[-1], flush=True)
    d["s1_single"], d["s1_multi"] = single, multi
    d = pd.concat([d, pd.DataFrame(info)], axis=1)
    d.to_csv(HERE / "scenes_selected.csv", index=False)
    geoms = []
    for r in d.itertuples():
        inv = Transformer.from_crs(int(r.epsg), 4326, always_xy=True)
        geoms.append(shapely.Polygon([inv.transform(x, y) for x, y in
                                      [(r.win_x0, r.win_y1), (r.win_x0 + WIN, r.win_y1), (r.win_x0 + WIN, r.win_y1 - WIN),
                                       (r.win_x0, r.win_y1 - WIN)]]))
    gpd.GeoDataFrame(d[["capella_id", "site_id", "country", "admin1", "date", "mode", "polarization", "cand_km"]],
                     geometry=geoms, crs=4326).to_file(HERE / "windows.geojson", driver="GeoJSON")
    print(d[["site_id", "country", "mode", "polarization", "cand_km", "s1_dt_days", "s1_n_multi"]].to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
