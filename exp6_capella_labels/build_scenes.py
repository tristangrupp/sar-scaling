"""Merge the three checks into scenes.csv and print the coverage summary (summary.txt).

Flags (thresholds fixed here, before any training):
  s1_ok          Sentinel-1 IW GRD acquisition within +/-12 days covering >= 50% of the footprint
  fields_present ESA WorldCover cropland >= 10% AND FTW Global (unfiltered) field share >= 10%
  bldg_present   >= 200 Overture buildings inside the footprint
Chip yield: 2.56 km (256 px at 10 m) and 1.28 km (128 px) cells, EPSG:6933 grid, fully inside the
union of qualifying footprints (repeat passes over one spot count once).
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, shapely

HERE = Path(__file__).parent
EXP0 = HERE.parent / "exp0_overlap_atlas" / "s1sm_capella"
cap = gpd.read_file(EXP0 / "capella_geo_items.geojson")
s1 = pd.read_csv(HERE / "s1_iw_match.csv")
lab = pd.read_csv(HERE / "labels_raster.csv")
ov = pd.read_csv(HERE / "overture_buildings.csv")

g = cap.copy()
rp = g.geometry.representative_point()
g["lon"], g["lat"] = rp.x.round(4), rp.y.round(4)
g["date"] = pd.to_datetime(g["datetime"]).dt.strftime("%Y-%m-%d")
g["footprint_km2"] = (g.to_crs(6933).area / 1e6).round(1)
g = g.rename(columns={"sar:instrument_mode": "mode", "sar:polarizations": "polarization",
                      "sar:resolution_range": "res_range_m", "sar:resolution_azimuth": "res_azimuth_m",
                      "view:incidence_angle": "incidence_deg", "sat:orbit_state": "orbit"})
df = (g.drop(columns=["datetime", "capella:use_case"]).merge(s1, on="capella_id").merge(lab, on="capella_id")
      .merge(ov, on="capella_id"))
df["s1_ok"] = df.s1_iw_n_acq_pm12d > 0
df["fields_present"] = (df.wc_crop >= 0.10) & (df.ftwg_field_frac >= 0.10)
df["bldg_present"] = df.overture_buildings >= 200
df["ftw_benchmark_or_parcels"] = (df.ftw_chips > 0) | (df.ftw_full_parcels > 0)
cols = ["capella_id", "site_id", "country", "admin1", "lon", "lat", "date", "platform", "mode", "polarization",
        "res_range_m", "res_azimuth_m", "incidence_deg", "orbit", "footprint_km2",
        "s1_ok", "s1_iw_n_acq_pm12d", "s1_iw_nearest_dt_days", "s1_iw_pols", "s1_iw_has_vv_vh", "s1_iw_nearest_id",
        "wc_land", "wc_water", "wc_crop", "wc_built", "wc_tree", "wc_grass",
        "ftwg_field_frac", "ftwg_field_frac_conf04", "ftwg_boundary_frac", "crop_consensus_mean",
        "ftw_chips", "ftw_chip_country", "ftw_chip_splits", "ftw_full_parcels", "ftw_full_country",
        "overture_buildings", "overture_bldg_sources", "fields_present", "bldg_present", "ftw_benchmark_or_parcels"]
df[cols].sort_values(["country", "site_id", "date"]).to_csv(HERE / "scenes.csv", index=False)

def chips(sub, cell_m):
    """Grid each site's footprint union separately (a global grid would have billions of cells)."""
    if sub.empty: return 0, 0.0
    n, area = 0, 0.0
    for _, s in sub.to_crs(6933).groupby("site_id"):
        u = s.geometry.union_all()
        area += u.area
        x0, y0, x1, y1 = u.bounds
        xs = np.arange(x0, x1 - cell_m + 1, cell_m); ys = np.arange(y0, y1 - cell_m + 1, cell_m)
        if not (len(xs) and len(ys)): continue
        X, Y = [a.ravel() for a in np.meshgrid(xs, ys)]
        shapely.prepare(u)
        n += int(shapely.contains(u, shapely.box(X, Y, X + cell_m, Y + cell_m)).sum())
    return n, round(area / 1e6, 0)

gg = g.set_index("capella_id").loc[df.capella_id].reset_index().merge(s1[["capella_id", "site_id"]], on="capella_id")
lines = [f"Capella GEO items: {len(df)} in {df.site_id.nunique()} exp0 sites, "
         f"{df.date.min()} to {df.date.max()}",
         "Mode: " + ", ".join(f"{k} {v}" for k, v in df["mode"].value_counts().items()),
         "Polarization: " + ", ".join(f"{k} {v}" for k, v in df.polarization.value_counts().items()),
         f"Sentinel-1 IW within +/-12 days: {df.s1_ok.sum()} ({df.s1_iw_has_vv_vh.sum()} with VV+VH); "
         f"median {df.s1_iw_n_acq_pm12d.median():.0f} acquisitions per window",
         f"Mostly land (WorldCover water < 50%): {(df.wc_water < 0.5).sum()}", ""]
def block(name, m):
    s = df[m]
    n256, area = chips(gg[m.values & df.s1_ok.values], 2560)
    n128, _ = chips(gg[m.values & df.s1_ok.values], 1280)
    lines.append(f"{name}: {len(s)} items, {s.site_id.nunique()} sites, {(s.s1_ok).sum()} with S1; "
                 f"unique area with S1 {area:.0f} km2 -> {n256} chips of 256 px, {n128} of 128 px")
block("Fields present (crop>=10% & FTW-Global field>=10%)", df.fields_present)
block("Fields, loose (FTW-Global unfiltered field>=10%)", df.ftwg_field_frac >= 0.10)
block("Fields, strict (FTW-Global conf>=0.4 field>=10%)", df.ftwg_field_frac_conf04 >= 0.10)
block("FTW benchmark chip or FTW parcel set touches footprint", df.ftw_benchmark_or_parcels)
block("Buildings present (>=200 Overture)", df.bldg_present)
block("Buildings, dense (>=2000 Overture)", df.overture_buildings >= 2000)
block("Fields AND buildings", df.fields_present & df.bldg_present)
block("Fields present, Stripmap only", df.fields_present & (df["mode"] == "stripmap"))
lines.append("")
lines.append("Fields-present items by country: " + ", ".join(f"{k} {v}" for k, v in df[df.fields_present].country.value_counts().items()))
lines.append("Fields-present items by mode: " + ", ".join(f"{k} {v}" for k, v in df[df.fields_present]["mode"].value_counts().items()))
lines.append("FTW benchmark/parcel hits: " + "; ".join(f"{r.capella_id} ({r.country}, chips {r.ftw_chips}, parcels {r.ftw_full_parcels})" for r in df[df.ftw_benchmark_or_parcels].itertuples()))
src = {}
for s in df.overture_bldg_sources.dropna():
    for kv in s.split(";"):
        if kv:
            k, v = kv.rsplit(":", 1); src[k] = src.get(k, 0) + int(v)
lines.append("Overture building sources (summed over items): " + ", ".join(f"{k} {v}" for k, v in sorted(src.items(), key=lambda x: -x[1])))
(HERE / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
os._exit(0)
