import geopandas as gpd, pandas as pd, numpy as np
land = gpd.read_file("../exp0_overlap_atlas/land-110m.json").set_crs(4326, allow_override=True)
L = land.to_crs(8857).buffer(0).union_all()
g = gpd.read_parquet("sm_scenes_all.parquet")
print(g[["granule","beam","pol","orbit_dir","start"]].head(3).to_string())
g["pol"] = g.granule.str[14:16]; g["beam"] = g.granule.str[4:6]
g = g[g.pol.isin(["DV", "SV"])].copy()
pts = gpd.GeoSeries(gpd.points_from_xy(g.lon, g.lat), crs=4326).to_crs(8857)
inland = pts.within(L).values
d = pts.distance(L.boundary).values
g["inland_km"] = np.where(inland, d / 1000, -d / 1000)
g.to_parquet("sm_scenes_all_vv.parquet")
c = g[g.inland_km > 25].groupby("cell").agg(n=("granule","size"), lon=("lon","mean"), lat=("lat","mean"),
     km=("inland_km","median"), y0=("start","min"), y1=("start","max"),
     dirs=("orbit_dir", lambda s: str(s.value_counts().to_dict()))).reset_index().sort_values("n", ascending=False)
pd.set_option("display.width", 220)
print(len(c)); print(c.head(90).to_string())
c.to_csv("cells_inland.csv", index=False)
