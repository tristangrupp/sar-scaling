r"""Per Capella GEO item: location, land cover, and which field-boundary labels cover it.

Reads (small windows only):
  - Natural Earth admin 0/1 (cache/ne) for country and state/province
  - ESA WorldCover 2021 v200 (10 m, read at 40 m) for cropland/built/water fractions
  - FTW Global (Robinson et al. 2026, source.coop/ftw/global-data) 500 m field-density and
    crop-consensus layers: share of each footprint the global field map calls field
  - FTW benchmark chips (H:\FTL_FTW\source, read-only copy of the public release): chips intersecting
  - FTW full national parcel sets (H:\FTL_FTW\source_full): parcels intersecting
Output: labels_raster.csv
"""
import json, math, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, shapely
import pyarrow.parquet as pq
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
HERE = Path(__file__).parent
EXP0 = HERE.parent / "exp0_overlap_atlas" / "s1sm_capella"
FTW = Path(r"H:\FTL_FTW")
FD = "https://data.source.coop/ftw/global-data/predictions/confidence/field-density/"
FD_UNF = FD + "prue_v1_field_area_500m.tif"
FD_C04 = FD + "prue_v1_field_area_500m_conf0.4.tif"
CC = "https://data.source.coop/ftw/global-data/predictions/confidence/crop-consensus/prue_v1_crop_count_mean_500m.tif"
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"

cap = gpd.read_file(EXP0 / "capella_geo_items.geojson")
adm0 = gpd.read_file(HERE / "cache/ne/ne_50m_admin_0_countries.zip")[["NAME", "geometry"]]
adm1 = gpd.read_file(HERE / "cache/ne/ne_10m_admin_1_states_provinces.zip")[["name", "admin", "geometry"]]
cent = gpd.GeoDataFrame(cap[["capella_id"]], geometry=cap.geometry.representative_point(), crs=4326)
c0 = gpd.sjoin_nearest(cent.to_crs(3857), adm0.to_crs(3857), how="left", max_distance=50_000).drop_duplicates("capella_id")
c1 = gpd.sjoin(cent, adm1, how="left", predicate="within").drop_duplicates("capella_id")
loc = pd.DataFrame({"capella_id": cap.capella_id,
                    "country": c0.set_index("capella_id").loc[cap.capella_id, "NAME"].values,
                    "admin1": c1.set_index("capella_id").loc[cap.capella_id, "name"].values})

def wc_tiles(b):
    x0, y0, x1, y1 = b
    out = []
    for la in range(int(math.floor(y0 / 3) * 3), int(math.floor(y1 / 3) * 3) + 1, 3):
        for lo in range(int(math.floor(x0 / 3) * 3), int(math.floor(x1 / 3) * 3) + 1, 3):
            out.append(f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
    return out

def read_masked(url, geom, decim=1, all_touched=False, band=1):
    with rasterio.open(url) as ds:
        w = from_bounds(*geom.bounds, ds.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        h, wd = max(1, int(w.height // decim)), max(1, int(w.width // decim))
        a = ds.read(band, window=w, out_shape=(h, wd), resampling=Resampling.nearest, boundless=False)
        t = ds.window_transform(w) * rasterio.Affine.scale(w.width / wd, w.height / h)
        m = geometry_mask([geom], out_shape=a.shape, transform=t, invert=True, all_touched=all_touched)
        return a[m], ds.nodata

def one(i):
    c = cap.iloc[i]; g = c.geometry
    r = {"capella_id": c.capella_id}
    try:
        vals = []
        for tile in wc_tiles(g.bounds):
            try:
                v, _ = read_masked(WC.format(tile), g, decim=4)
                vals.append(v)
            except rasterio.errors.RasterioIOError:
                pass  # no WorldCover tile = open ocean
        v = np.concatenate(vals) if vals else np.array([], dtype=np.uint8)
        n = max(1, int((v > 0).sum())) if len(v) else 0
        r["wc_tiles_found"] = len(vals)
        for k, cls in [("wc_water", 80), ("wc_crop", 40), ("wc_built", 50), ("wc_tree", 10), ("wc_grass", 30)]:
            r[k] = round(float((v == cls).sum()) / n, 4) if n else None
        r["wc_land"] = round(1 - (r["wc_water"] or 0), 4) if n else 0.0
    except Exception as e:
        r["wc_err"] = str(e)[:100]
    try:
        if -60 < g.centroid.y < 84:
            fu, _ = read_masked(FD_UNF, g, all_touched=True, band=1)
            bu, _ = read_masked(FD_UNF, g, all_touched=True, band=2)
            f4, _ = read_masked(FD_C04, g, all_touched=True, band=1)
            cc, ccnd = read_masked(CC, g, all_touched=True)
            r["ftwg_field_frac"] = round(float(fu.mean()) / 2500, 4) if fu.size else 0
            r["ftwg_boundary_frac"] = round(float(bu.mean()) / 2500, 4) if bu.size else 0
            r["ftwg_field_frac_conf04"] = round(float(f4.mean()) / 2500, 4) if f4.size else 0
            ccv = cc[cc != (ccnd if ccnd is not None else 0)]
            r["crop_consensus_mean"] = round(float(ccv.mean()), 2) if ccv.size else 0
    except Exception as e:
        r["ftwg_err"] = str(e)[:100]
    return r

RAS = HERE / "cache" / "raster_part.csv"
if RAS.exists():
    ras = pd.read_csv(RAS)
else:
    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(one, range(len(cap))))
    ras = pd.DataFrame(rows)
    ras.to_csv(RAS, index=False)
print("raster reads done", flush=True)

# FTW benchmark chips (public release copy)
chips = pd.concat([gpd.read_parquet(p).set_crs(4326, allow_override=True).assign(ftw_country=p.parent.name)
                   for p in sorted((FTW / "source").glob("*/chips_*.parquet"))], ignore_index=True)
chips = gpd.GeoDataFrame(chips, geometry="geometry", crs=4326)
j = gpd.sjoin(cap[["capella_id", "geometry"]], chips[["aoi_id", "split", "ftw_country", "geometry"]], predicate="intersects")
ch = j.groupby("capella_id").agg(ftw_chips=("aoi_id", "nunique"), ftw_chip_country=("ftw_country", "first"),
                                 ftw_chip_splits=("split", lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().items())))
print(f"chips: {len(chips)}; Capella items touching a chip: {len(ch)}", flush=True)

# FTW full national parcel sets: only open countries whose extent touches a Capella footprint
full = {}
for p in sorted((FTW / "source_full").glob("*/boundaries_*_full.parquet")):
    md = json.loads(pq.read_schema(p).metadata[b"geo"])
    bb = md["columns"]["geometry"].get("bbox")
    if bb is None:
        continue
    hit = cap[cap.intersects(shapely.box(*bb))]
    if hit.empty:
        continue
    t = pq.read_table(p, columns=["geometry"])
    geoms = shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False))
    tree = shapely.STRtree(geoms)
    for _, c in hit.iterrows():
        n = len(tree.query(c.geometry, predicate="intersects"))
        if n:
            full[c.capella_id] = (n, p.parent.name)
    print(f"  full parcels {p.parent.name}: {len(hit)} Capella in extent, {sum(1 for v in full.values() if v[1]==p.parent.name)} hit", flush=True)
fu = pd.DataFrame([{"capella_id": k, "ftw_full_parcels": v[0], "ftw_full_country": v[1]} for k, v in full.items()],
                  columns=["capella_id", "ftw_full_parcels", "ftw_full_country"])

out = loc.merge(ras, on="capella_id", how="left").merge(ch.reset_index(), on="capella_id", how="left").merge(fu, on="capella_id", how="left")
out["ftw_chips"] = out.ftw_chips.fillna(0).astype(int)
out["ftw_full_parcels"] = out.ftw_full_parcels.fillna(0).astype(int)
out.to_csv(HERE / "labels_raster.csv", index=False)
print("done", len(out), flush=True)
os._exit(0)
