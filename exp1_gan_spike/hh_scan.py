"""Count Capella HH collects with a Sentinel-1 IW RTC HH scene within 30 days (Planetary Computer)."""
import os, sys
import geopandas as gpd, numpy as np, pandas as pd, planetary_computer as pc
from pystac_client import Client
g = gpd.read_file("../exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson")
g = g[g["sar:polarizations"] == "HH"].copy()
c = g.geometry.centroid; g["lon"], g["lat"] = c.x, c.y
g["cell"] = np.floor(g.lon).astype(int).astype(str) + "_" + np.floor(g.lat).astype(int).astype(str)
g["t"] = pd.to_datetime(g.datetime)
cl = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
rows = []
cells = g.groupby("cell")
for n, (cell, s) in enumerate(cells, 1):
    lon, lat = s.lon.median(), s.lat.median()
    t0, t1 = s.t.min() - pd.Timedelta(days=30), s.t.max() + pd.Timedelta(days=30)
    try:
        items = [i for i in cl.search(collections=["sentinel-1-rtc"], bbox=[lon-.02, lat-.02, lon+.02, lat+.02],
                 datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}").items() if "hh" in i.assets]
    except Exception as e:
        print("fail", cell, e, flush=True); continue
    ts = pd.Series([pd.Timestamp(i.datetime) for i in items])
    for r in s.itertuples():
        d = (ts - r.t).abs().min() if len(ts) else pd.NaT
        rows.append({"xband_id": r.capella_id, "cell": cell, "lon": r.lon, "lat": r.lat, "n_hh": len(ts),
                     "dt_days": None if pd.isna(d) else d.days})
    if n % 25 == 0: print(f"{n}/{cells.ngroups}", flush=True)
df = pd.DataFrame(rows); df.to_csv("capella_hh_s1hh.csv", index=False)
ok = df[df.dt_days.notna() & (df.dt_days <= 30)]
print("HH collects", len(df), "with S1 HH <=30d:", len(ok), "cells:", ok.cell.nunique(), flush=True)
print(ok.groupby("cell").size().sort_values(ascending=False).head(25).to_dict(), flush=True)
sys.stdout.flush(); os._exit(0)
