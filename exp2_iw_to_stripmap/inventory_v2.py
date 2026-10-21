"""Round 2 inventory: every Sentinel-1 Stripmap SLC on ASF (2015 onward), binned to 1-degree cells,
to find inland land the 2023+ pilot inventory lacks (it is dominated by islands and ocean).
Writes sm_scenes_all.parquet and cells_all.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import wkt

ROOT = Path(__file__).resolve().parent
ASF = "https://api.daac.asf.alaska.edu/services/search/param"


def main():
    parts = []
    import time
    months = pd.date_range("2014-04-01", "2026-11-01", freq="QS")
    for a, b in zip(months[:-1], months[1:]):
        p = {"platform": "SENTINEL-1", "processingLevel": "SLC", "beamSwath": "S1,S2,S3,S4,S5,S6",
             "start": f"{a:%Y-%m-%d}", "end": f"{b:%Y-%m-%d}", "output": "jsonlite2"}
        for attempt in range(5):
            js = requests.get(ASF, params=p, timeout=600).json()
            if "results" in js:
                break
            print("retry", a, str(js)[:200], flush=True)
            time.sleep(20 * (attempt + 1))
        r = js["results"]
        print(a.date(), len(r), flush=True)
        parts += r
    df = pd.DataFrame(parts)
    df.to_json(ROOT / "asf_sm_raw.json")
    print(df.columns.tolist(), flush=True)
    keep = {"gn": "granule", "st": "start", "b": "beam", "pa": "pol", "fd": "orbit_dir", "w": "wkt",
            "s": "size_mb", "p": "path"}
    df = df.rename(columns={k: v for k, v in keep.items() if k in df})
    g = gpd.GeoDataFrame(df, geometry=[wkt.loads(w) for w in df.wkt], crs=4326)
    c = g.geometry.representative_point()
    g["lon"], g["lat"] = c.x, c.y
    g["cell"] = (np.floor(g.lon).astype(int).astype(str) + "_" + np.floor(g.lat).astype(int).astype(str))
    g = g.drop(columns=["wkt"]).drop_duplicates("granule")
    g.to_parquet(ROOT / "sm_scenes_all.parquet")
    cells = g.groupby("cell").agg(n=("granule", "size"), lon=("lon", "mean"), lat=("lat", "mean"),
                                  first=("start", "min"), last=("start", "max"),
                                  pols=("pol", lambda s: ",".join(sorted(set(map(str, s)))))).reset_index()
    cells.sort_values("n", ascending=False).to_csv(ROOT / "cells_all.csv", index=False)
    print(len(g), "scenes", len(cells), "cells", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
