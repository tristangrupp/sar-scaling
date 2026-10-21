"""Scene-level co-registration check and the final pair list (pairs_final.csv).

Per-tile offsets are noisy where a tile has little contrast (forest, desert): its 10 m correlation
is low whether or not it is aligned. The Stripmap-to-IW offset is a property of the scene (orbit,
timing, DEM), so it is pooled per scene:
  reliable tile   corr10 >= 0.3 and the coarse search did not hit its +-200 m edge
  scene offset    median offset of its reliable tiles; the scene counts as registered with >= 2
  per tile        keeps its own offset if corr10 >= 0.4 and within 10 m of the scene offset,
                  otherwise the IW is re-read at the scene offset (Stripmap untouched)
Scenes with fewer than 2 reliable tiles are dropped (use = False); repair_iw.py tries another IW
pass for them first.
"""
from __future__ import annotations

import sys

from common import NL, ROOT, SM_TIF, TILES, block_mean, corr, db, pc_item, read_iw  # noqa: I001

import os

import numpy as np
import pandas as pd
import rasterio


def main():
    # v2: python finalize_pairs.py v2  -> tiles_v2.csv, adds pairs_v2.csv, writes pairs_final_v2.csv
    v2 = "v2" in sys.argv[1:]
    names = ["pairs.csv", "pairs_b.csv", "pairs_c.csv", "pairs_d.csv"] + (["pairs_v2.csv"] if v2 else [])
    fs = [ROOT / f for f in names if (ROOT / f).exists()]
    p = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("tile_id", keep="last")
    p = p[p.status == "ok"].copy()
    t = pd.read_csv(ROOT / ("tiles_v2.csv" if v2 else "tiles.csv"))[["tile_id", "slc", "iw_id"]]
    p = p.drop(columns=["iw_id", "slc"], errors="ignore").merge(t, on="tile_id")
    old = pd.read_csv(ROOT / "pairs_final.csv") if v2 else None
    if v2:                      # never rewrite round 1 tiles (exp5 reads them); finalize new scenes only
        p = p[~p.tile_id.isin(old.tile_id)]
    p["reliable"] = (p.corr10 >= 0.3) & ~p.coarse_edge.astype(bool)
    rows = []
    for slc, g in p.groupby("slc"):
        rel = g[g.reliable]
        ok = len(rel) >= 2
        me, mn = (rel.off_e_m.median(), rel.off_n_m.median()) if ok else (np.nan, np.nan)
        spread = float(np.hypot(rel.off_e_m - me, rel.off_n_m - mn).median()) if ok else np.nan
        items = None
        for r in g.itertuples():
            rec = r._asdict()
            rec.update(scene_ok=ok, scene_off_e_m=me, scene_off_n_m=mn, scene_spread_m=spread, use=ok,
                       offset_source="tile")
            if ok:
                own_ok = r.corr10 >= 0.4 and np.hypot(r.off_e_m - me, r.off_n_m - mn) <= 10
                if not own_ok:
                    if items is None:
                        items = [pc_item("sentinel-1-rtc", i) for i in r.iw_id.split(";")]
                    z = dict(np.load(TILES / f"{r.tile_id}.npz"))
                    x0, y1, epsg = float(z["x0"]), float(z["y1"]), int(z["epsg"])
                    iw = db(read_iw(items, epsg, x0 + me, y1 + mn, NL))
                    with rasterio.open(SM_TIF / f"{r.tile_id}.tif") as s:
                        sm10 = db(block_mean(s.read(1), 2))
                    z.update(iw=iw.astype("float16"), iw_x0=x0 + me, iw_y1=y1 + mn)
                    np.savez_compressed(TILES / f"{r.tile_id}.npz", **z)
                    rec.update(corr10_tile=r.corr10, corr10=round(corr(sm10, iw), 3), off_e_m=me, off_n_m=mn,
                               offset_source="scene")
            rows.append(rec)
        print(f"{slc[:45]} tiles={len(g)} reliable={len(rel)} ok={ok} offset=({me:.1f}, {mn:.1f}) m "
              f"spread={spread:.1f} m", flush=True)
    out = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    if v2:
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(ROOT / ("pairs_final_v2.csv" if v2 else "pairs_final.csv"), index=False)
    print(out.groupby("region").agg(tiles=("tile_id", "size"), used=("use", "sum"),
                                    corr10=("corr10", "median")).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
