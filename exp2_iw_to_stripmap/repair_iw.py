"""Re-pair scenes whose IW match failed co-registration (median 10 m correlation < 0.3).

Some early IW RTC frames (e.g. Sentinel-1C during commissioning, April 2025) do not line up with
the Stripmap at all. For each such scene, try every other same-direction IW pass within 6 days,
score it on up to three of the scene's tiles, and keep the best if its median correlation clears
0.3. Updates tiles.csv (iw_id, iw_dt_days) and drops the scene's rows from the pairs files, so
build_tiles.py re-pairs them (the Stripmap GeoTIFFs are cached).
"""
from __future__ import annotations

import sys

from common import NL, ROOT, SM_TIF, block_mean, corr, db, pc_client, read_iw  # noqa: I001

import os

import numpy as np
import pandas as pd
import rasterio

from select_tiles import same_pass


def pairs_all():
    fs = [ROOT / f for f in ("pairs.csv", "pairs_b.csv", "pairs_c.csv", "pairs_d.csv") if (ROOT / f).exists()]
    p = pd.concat([pd.read_csv(f) for f in fs])
    return p[p.status == "ok"].drop_duplicates("tile_id", keep="last")


def score_pass(items, tiles):
    cs = []
    for r in tiles.itertuples():
        with rasterio.open(SM_TIF / f"{r.tile_id}.tif") as s:
            sm = s.read(1)
            epsg, x0, y1 = s.crs.to_epsg(), s.transform.c, s.transform.f
        sm10 = db(block_mean(sm, 2))
        best = -1
        for dx in (-20, 0, 20):           # small search, enough to tell a usable pass from a broken one
            for dy in (-20, 0, 20):
                iw = db(read_iw(items, epsg, x0 + dx, y1 + dy, NL))
                if np.isfinite(iw).mean() > 0.8:
                    best = max(best, corr(sm10, iw))
        cs.append(best)
    return float(np.median(cs))


def main():
    t = pd.read_csv(ROOT / "tiles.csv")
    p = pairs_all()
    med = p.drop(columns=["slc"], errors="ignore").merge(t[["tile_id", "slc"]], on="tile_id").groupby("slc").corr10.median()
    bad = med[med < 0.3]
    print("scenes to repair:", bad.round(3).to_dict(), flush=True)
    for slc, c0 in bad.items():
        g = t[t.slc == slc]
        r0 = g.iloc[0]
        t_sm = pd.Timestamp(slc[17:32], tz="UTC")
        its = [i for i in pc_client().search(collections=["sentinel-1-rtc"], bbox=[r0.lon - .02, r0.lat - .02, r0.lon + .02, r0.lat + .02],
                                             datetime=f"{t_sm - pd.Timedelta(days=6):%Y-%m-%d}/{t_sm + pd.Timedelta(days=6):%Y-%m-%d}").items()
               if "vv" in i.assets and (i.properties.get("sat:orbit_state") or "").upper() == r0.orbit_dir
               and i.id not in r0.iw_id.split(";")]
        best = (c0, None, None)
        sample = g[g.tile_id.isin(p.tile_id)].head(3)
        for it in its:
            iws = same_pass(it.id, __import__("shapely.geometry", fromlist=["box"]).box(
                g.lon.min() - .05, g.lat.min() - .05, g.lon.max() + .05, g.lat.max() + .05))
            c = score_pass(iws, sample)
            print(f"  {slc[:40]} try {it.id[:40]} median corr {c:.3f}", flush=True)
            if c > best[0]:
                best = (c, iws, abs((pd.Timestamp(it.datetime) - t_sm).days))
        if best[1] is not None and best[0] >= 0.3:
            t.loc[g.index, "iw_id"] = ";".join(i.id for i in best[1])
            t.loc[g.index, "iw_dt_days"] = best[2]
            for f in ("pairs.csv", "pairs_b.csv", "pairs_c.csv", "pairs_d.csv"):
                if (ROOT / f).exists():
                    q = pd.read_csv(ROOT / f)
                    q[~q.tile_id.isin(g.tile_id)].to_csv(ROOT / f, index=False)
            print(f"  repaired {slc[:40]}: {c0:.3f} -> {best[0]:.3f} with {best[1][0].id}", flush=True)
        else:
            print(f"  no usable alternative for {slc[:40]} (best {best[0]:.3f}); its tiles stay excluded", flush=True)
    t.to_csv(ROOT / "tiles.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
