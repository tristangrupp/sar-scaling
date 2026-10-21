"""Add Capella to the spike's selection without disturbing the existing Umbra / ICEYE picks.

Capella open data is mostly HH (2,088 of 2,285 GEO collects); Sentinel-1 IW over land is VV, so
only the 197 VV collects can be co-pol training pairs. They are split by 1-degree cell:
  train         about 75% of VV cells
  test          the other VV cells, region "capella_<cell>" (held out entirely)
  test_xpol     HH collects inside the original test regions (Houston, Newark, Tokyo, Antwerp,
                Rotterdam). Cross-polarized against the VV input, so they are shown as visual
                examples and scored apart from the main test set.
Appends to selection.csv, then prep.py processes only the new collects.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

import prep

ROOT = Path(__file__).resolve().parent
COLL = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
        "capella-open-data-by-product-type/capella-open-data-geo/collection.json")


def hrefs(ids):
    coll = json.loads((ROOT / "capella_geo_collection.json").read_text())
    links = {Path(l["href"]).stem: urljoin(COLL, l["href"]) for l in coll["links"] if l["rel"] == "item"}
    s = requests.Session()

    def one(i):
        it = s.get(links[i], timeout=60).json()
        tif = [a["href"] for a in it["assets"].values() if a["href"].endswith(".tif") and "preview" not in a["href"]]
        return i, (tif[0] if tif else None)

    with ThreadPoolExecutor(16) as ex:
        return dict(ex.map(one, [i for i in ids if i in links]))


def main():
    g = gpd.read_file(PROJ := ROOT.parent / "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson")
    g = g.rename(columns={"capella_id": "xband_id"})
    c = g.geometry.centroid
    g["lon"], g["lat"] = c.x, c.y
    g["test_region"] = [prep.test_region(p) for p in c]
    g["cell"] = np.floor(g.lon).astype(int).astype(str) + "_" + np.floor(g.lat).astype(int).astype(str)
    g["provider"] = "capella"

    vv = g[(g["sar:polarizations"] == "VV") & g.test_region.isna()].copy()
    cells = sorted(vv.cell.unique())
    rng = np.random.default_rng(0)
    test_cells = set(rng.choice(cells, size=max(1, len(cells) // 4), replace=False))
    vv["split"] = np.where(vv.cell.isin(test_cells), "test", "train")
    vv.loc[vv.split == "test", "test_region"] = "capella_" + vv.cell
    # one collect per ~0.1 deg cell in training, as for Umbra / ICEYE
    vv["c01"] = vv.lon.round(1).astype(str) + "_" + vv.lat.round(1).astype(str)
    vv = pd.concat([vv[vv.split == "train"].drop_duplicates("c01"), vv[vv.split == "test"].drop_duplicates("c01")])

    xp = g[(g["sar:polarizations"] == "HH") & g.test_region.notna()].copy()
    xp["split"] = "test_xpol"
    new = pd.concat([vv, xp])
    h = hrefs(set(new.xband_id))
    new["href"] = new.xband_id.map(h)
    new = new[new.href.notna()]

    cols = ["xband_id", "provider", "datetime", "sar:polarizations", "sar:instrument_mode", "view:incidence_angle",
            "lon", "lat", "test_region", "cell", "split", "href"]
    sel = pd.read_csv(ROOT / "selection.csv")
    sel = sel[sel.provider != "capella"]
    out = pd.concat([sel, pd.DataFrame(new[cols])], ignore_index=True)
    out.to_csv(ROOT / "selection.csv", index=False)
    print("capella added:", new.groupby(["split"]).size().to_dict(), "test cells:", sorted(test_cells), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
