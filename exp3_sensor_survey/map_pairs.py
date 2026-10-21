from config import DATA  # noqa: E402
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Polygon
p = pd.read_csv(str(DATA) + "/mixed_pairs.csv")
def poly(s):
    v = list(map(float, s.split())); return Polygon(list(zip(v[1::2], v[0::2])))
p["geometry"] = p.poly_a.map(poly)
g = gpd.GeoDataFrame(p, crs=4326)
g["lon"], g["lat"] = g.geometry.centroid.x, g.geometry.centroid.y
tf = g.groupby(["track", "frame"]).agg(n=("id_a", "size"), lon=("lon", "mean"), lat=("lat", "mean"), dir=("dir", "first"),
        full_a=("full_a", lambda s: (s == "F").mean()), geometry=("geometry", "first")).reset_index()
tf = gpd.GeoDataFrame(tf, crs=4326)
X = []
from config import PROJECT  # noqa: E402
root = str(PROJECT) + "/"
for prov, f in [("umbra", "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson"), ("iceye", "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson"), ("capella", "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson")]:
    x = gpd.read_file(root + f); x["provider"] = prov; x = x.rename(columns={"capella_id": "xband_id"}); X.append(x[["provider", "xband_id", "datetime", "geometry"] + [c for c in ["view:incidence_angle", "sar:observation_direction", "view:azimuth", "sar:instrument_mode", "sar:product_type"] if c in x.columns]])
    print(prov, len(x), list(x.columns)[:30], flush=True)
X = pd.concat(X, ignore_index=True); X = gpd.GeoDataFrame(X, crs=4326)
X.to_file(str(DATA) + "/xband_all.gpkg")
j = gpd.sjoin(X[["provider", "xband_id", "geometry"]], tf[["track", "frame", "geometry"]], predicate="within")
cnt = j.groupby(["track", "frame", "provider"]).size().unstack(fill_value=0)
tf = tf.merge(cnt, on=["track", "frame"], how="left").fillna(0)
tf["nx"] = tf[[c for c in ["umbra", "iceye", "capella"] if c in tf]].sum(axis=1)
tf.to_file(str(DATA) + "/pair_frames.gpkg")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(tf.drop(columns="geometry").sort_values("nx", ascending=False).head(60).round(2).to_string(), flush=True)
print("frames with xband:", (tf.nx > 0).sum(), "of", len(tf), flush=True)
# coarse geography
tf["reg"] = (np.round(tf.lat / 10) * 10).astype(int).astype(str) + "," + (np.round(tf.lon / 10) * 10).astype(int).astype(str)
print(tf.groupby("reg").agg(frames=("n", "size"), pairs=("n", "sum"), nx=("nx", "sum")).sort_values("frames", ascending=False).head(60).to_string(), flush=True)
sys.stdout.flush(); os._exit(0)
