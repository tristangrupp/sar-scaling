"""Add tiles centered on X-band sites the structure score missed (Houston's Capella collects at
Baytown, -94.99 / 29.64), one per held-out scene whose Stripmap and IW footprints both cover it.
Appends to tiles.csv (tile ids end in _site)."""
import os
import sys

import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

import select_tiles as st
from common import ROOT

SITES = {"houston": [(-94.99, 29.64)]}


def main():
    t = pd.read_csv(ROOT / "tiles.csv")
    sc = st.scenes()
    xb = st.xband_items()
    new = []
    for reg, pts in SITES.items():
        for s in sc[sc.region == reg].itertuples():
            iw_ids = t[t.slc == s.slc].iw_id.iloc[0]
            iws = [st.pc_item("sentinel-1-rtc", i) for i in iw_ids.split(";")]
            fp = s.geometry.intersection(unary_union([shape(i.geometry) for i in iws]))
            for k, (lon, lat) in enumerate(pts):
                if not fp.buffer(-0.02).contains(Point(lon, lat)):
                    print(reg, s.slc[:40], "site outside footprint", flush=True)
                    continue
                hit = xb[xb.intersects(Point(lon, lat))]
                row = t[t.slc == s.slc].iloc[0].to_dict()
                row.update(lon=lon, lat=lat, x=None, y=None, land=None, built=None, entropy=None, score=None,
                           n_xband=len(hit), **{f"n_{p}": int((hit.provider == p).sum()) for p in ("umbra", "iceye", "capella")},
                           tile_id=f"{reg}_{s.slc[17:32]}_{s.slc[4:6]}_site{k}")
                new.append(row)
                print(reg, row["tile_id"], "xband", len(hit), flush=True)
    t = pd.concat([t[~t.tile_id.isin([r["tile_id"] for r in new])], pd.DataFrame(new)], ignore_index=True)
    t.to_csv(ROOT / "tiles.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
