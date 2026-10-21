"""Add Capella HH collects that have a Sentinel-1 IW HH scene within 30 days (co-pol HH pairs).

Only 38 of 2,088 Capella HH collects qualify: Sentinel-1 acquires HH over land mainly at high
latitudes (Greenland, Svalbard, Canada, Antarctica) and a few other places. Cells are split so
about a quarter are held out entirely (region "capella_hh_<cell>").
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
import add_capella

hh = pd.read_csv("capella_hh_s1hh.csv")
hh = hh[hh.dt_days.notna() & (hh.dt_days <= 30)]
g = gpd.read_file("../exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson").rename(columns={"capella_id": "xband_id"})
hh = hh.merge(pd.DataFrame(g.drop(columns="geometry")), on="xband_id")
cells = sorted(hh.cell.unique())
test_cells = set(np.random.default_rng(1).choice(cells, size=max(1, len(cells) // 4), replace=False))
hh["split"] = np.where(hh.cell.isin(test_cells), "test", "train")
hh["test_region"] = np.where(hh.split == "test", "capella_hh_" + hh.cell, None)
hh["provider"], hh["s1_pol"] = "capella", "hh"
h = add_capella.hrefs(set(hh.xband_id))
hh["href"] = hh.xband_id.map(h)
hh = hh[hh.href.notna()]
cols = ["xband_id", "provider", "datetime", "sar:polarizations", "sar:instrument_mode", "view:incidence_angle",
        "lon", "lat", "test_region", "cell", "split", "href", "s1_pol"]
sel = pd.read_csv("selection.csv")
sel = sel[~sel.xband_id.isin(hh.xband_id)]
pd.concat([sel, hh[cols]], ignore_index=True).to_csv("selection.csv", index=False)
print("HH added:", hh.groupby("split").size().to_dict(), "test cells:", sorted(test_cells), flush=True)
sys.stdout.flush(); os._exit(0)
