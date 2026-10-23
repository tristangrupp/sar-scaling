"""Count Overture building footprints inside every Capella GEO footprint.

One DuckDB query per exp0 site (bbox prefilter on Overture's bbox struct), returning building
centroids and source dataset; counts per Capella polygon are then made locally. Per-site results
are cached in cache/overture/ as parquet, so reruns resume.
Needs duckdb (installed outside the env; path in DUCKDB_PATH env var or the default below).
Output: overture_buildings.csv (one row per Capella item).
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, os.environ.get("DUCKDB_PATH", r"C:\Users\grupp\AppData\Local\Temp\claude\C--Users-grupp\335d88d1-5193-48a0-8750-c34519c54d86\scratchpad\pylib"))
import duckdb, geopandas as gpd, pandas as pd

HERE = Path(__file__).parent
EXP0 = HERE.parent / "exp0_overlap_atlas" / "s1sm_capella"
REL = "2026-10-21.0"
SRC = f"s3://overturemaps-us-west-2/release/{REL}/theme=buildings/type=building/*"

cap = gpd.read_file(EXP0 / "capella_geo_items.geojson")
sites = gpd.read_file(EXP0 / "sites.geojson")
cap = gpd.sjoin(cap, sites[["site_id", "geometry"]], predicate="intersects", how="left")
cap = cap.drop(columns="index_right").drop_duplicates("capella_id").reset_index(drop=True)

CON = duckdb.connect()
CON.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; SET threads=8; "
            "SET memory_limit='6GB'; SET enable_object_cache=true;")

# Index once: per-file bbox extent from the parquet footers, so each site query opens only the
# few files that can hold it (a glob over all files took ~40 s per site).
IDX = HERE / "cache" / "overture" / f"_file_index_{REL}.parquet"
if not IDX.exists():
    t = time.time()
    m = CON.execute(f"""SELECT file_name, path_in_schema, min(TRY_CAST(stats_min AS DOUBLE)) mn, max(TRY_CAST(stats_max AS DOUBLE)) mx
                       FROM parquet_metadata('{SRC}') WHERE path_in_schema IN ('bbox, xmin','bbox, xmax','bbox, ymin','bbox, ymax')
                       GROUP BY ALL""").df()
    m.to_parquet(IDX); print(f"file index built in {time.time()-t:.0f}s", flush=True)
m = pd.read_parquet(IDX)
FI = pd.DataFrame({"x0": m[m.path_in_schema == "bbox, xmin"].set_index("file_name").mn,
                   "x1": m[m.path_in_schema == "bbox, xmax"].set_index("file_name").mx,
                   "y0": m[m.path_in_schema == "bbox, ymin"].set_index("file_name").mn,
                   "y1": m[m.path_in_schema == "bbox, ymax"].set_index("file_name").mx})
print(f"{len(FI)} Overture building files indexed", flush=True)

def site_query(sid):
    out = HERE / "cache" / "overture" / f"site{int(sid):04d}.parquet"
    if out.exists():
        return sid
    x0, y0, x1, y1 = cap[cap.site_id == sid].total_bounds
    c = CON
    t = time.time()
    files = FI[(FI.x1 > x0) & (FI.x0 < x1) & (FI.y1 > y0) & (FI.y0 < y1)].index.tolist()
    if not files:
        pd.DataFrame({"x": [], "y": [], "src": []}).to_parquet(out); return sid
    flist = "[" + ",".join(f"'{f}'" for f in files) + "]"
    df = c.execute(f"""SELECT (bbox.xmin+bbox.xmax)/2 AS x, (bbox.ymin+bbox.ymax)/2 AS y,
                       sources[1].dataset AS src
                       FROM read_parquet({flist})
                       WHERE bbox.xmax > {x0} AND bbox.xmin < {x1} AND bbox.ymax > {y0} AND bbox.ymin < {y1}""").df()
    df.to_parquet(out)
    print(f"  site {sid}: {len(df)} buildings, {len(files)} files, {time.time()-t:.0f}s", flush=True)
    return sid

sids = sorted(cap.site_id.dropna().unique())
print(f"{len(sids)} sites", flush=True)
for sid in sids:  # serial: parallel DuckDB connections each refetch footers and blew up to 13 GB
    site_query(sid)

rows = []
for sid, g in cap.groupby("site_id"):
    b = pd.read_parquet(HERE / "cache" / "overture" / f"site{int(sid):04d}.parquet")
    pts = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.x, b.y), crs="EPSG:4326") if len(b) else None
    for _, c in g.iterrows():
        if pts is None:
            rows.append({"capella_id": c.capella_id, "overture_buildings": 0, "overture_bldg_sources": ""}); continue
        inside = pts[pts.within(c.geometry)] if len(pts) < 2_000_000 else pts.iloc[pts.sindex.query(c.geometry, predicate="contains")]
        vc = inside.src.value_counts()
        rows.append({"capella_id": c.capella_id, "overture_buildings": int(len(inside)),
                     "overture_bldg_sources": ";".join(f"{k}:{v}" for k, v in vc.items())})
pd.DataFrame(rows).to_csv(HERE / "overture_buildings.csv", index=False)
print("done", len(rows), flush=True)
os._exit(0)
