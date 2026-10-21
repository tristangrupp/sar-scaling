"""Build paired tiles: Stripmap 5 m gamma0 (sarsen, terrain-flattened) <-> IW RTC VV 10 m.

For each row of tiles.csv:
  1. process_slc.py (s1slc env, subprocess) writes DATA/sm5/<tile_id>.tif, 5 m UTM gamma0
  2. co-register (here, ftwtools env), never resampling the Stripmap target:
       coarse  normalized cross-correlation of SM block-averaged to 10 m against a padded IW
               window (+-200 m), integer peak + parabolic subpixel
       fine    phase correlation at 5 m of the (Gaussian-smoothed) SM against the coarse-aligned
               IW upsampled bicubically, limited to +-2 px (10 m)
     The IW is then re-read (bilinear) on the tile grid shifted by the total offset.
  3. Lee-filtered (7x7) Stripmap reference, as in Amieva et al. 2026
  4. ESA WorldCover 2021 classes on the 5 m grid (water mask, built-up, homogeneous areas)
Saves DATA/tiles/<tile_id>.npz and pairs.csv (offsets, 10 m correlation before/after).

  python build_tiles.py [--workers 4] [--only REGEX]
"""
from __future__ import annotations

import sys

from common import (DATA, NL, PY_SLC, ROOT, SM_TIF, TILES, block_mean, coarse_offset, corr, db,  # noqa: I001
                    fine_shift, lee_filter, pc_item, read_iw, read_worldcover, NH)

import argparse
import os
import re
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

MARGIN = 20             # 10 m px searched each side in the coarse step
MAX_FINE = 2.0          # 5 m px


def run_sm(r):
    out = SM_TIF / f"{r.tile_id}.tif"
    if out.exists():
        return out, "cached", 0
    log = SM_TIF / f"{r.tile_id}.log"
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run([PY_SLC, "-u", str(ROOT / "process_slc.py"), r.safe, f"{r.lon:.6f}", f"{r.lat:.6f}",
                        "3200", str(out)], stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    return out, ("ok" if out.exists() else "sm_failed"), round(time.time() - t0)


_items = {}


def iw_item(iw_ids):
    """Signed items for the ';'-joined frames of the IW pass (re-signed after 30 min)."""
    now = time.time()
    if iw_ids not in _items or now - _items[iw_ids][0] > 1800:
        _items[iw_ids] = (now, [pc_item("sentinel-1-rtc", i) for i in iw_ids.split(";")])
    return _items[iw_ids][1]


def pair(r, sm_tif):
    with rasterio.open(sm_tif) as s:
        sm = s.read(1).astype("float32")
        epsg, x0, y1 = s.crs.to_epsg(), s.transform.c, s.transform.f
    sm[~np.isfinite(sm) | (sm <= 0)] = np.nan
    valid_sm = float(np.isfinite(sm).mean())
    if valid_sm < 0.8:
        return {"status": "sm_sparse", "valid_sm": valid_sm}
    it = iw_item(r.iw_id)
    sm_db = db(sm)
    sm10 = db(block_mean(sm, 2))
    big = db(read_iw(it, epsg, x0 - MARGIN * 10, y1 + MARGIN * 10, NL + 2 * MARGIN))
    if np.isfinite(big).mean() < 0.8:
        return {"status": "iw_sparse", "valid_sm": valid_sm}
    iw0 = big[MARGIN:-MARGIN, MARGIN:-MARGIN]
    c_nom = corr(sm10, iw0)
    dr, dc, ncc = coarse_offset(sm10, big, MARGIN)
    edge = max(abs(dr), abs(dc)) >= MARGIN - 1
    if edge:
        dr = dc = 0.0
    ox, oy = dc * 10, -dr * 10
    iwa = db(read_iw(it, epsg, x0 + ox, y1 + oy, NL))
    up = ndimage.zoom(np.where(np.isfinite(iwa), iwa, np.nanmedian(iwa)), 2, order=3)
    fs = fine_shift(sm_db, up)
    fs = np.clip(fs, -MAX_FINE, MAX_FINE)
    ox, oy = ox - fs[1] * 5, oy + fs[0] * 5
    iw = read_iw(it, epsg, x0 + ox, y1 + oy, NL)
    iw_db = db(iw)
    c_fin = corr(sm10, iw_db)
    lee, looks = lee_filter(sm, 7)
    wc = read_worldcover(epsg, x0, y1, NH, 5.0)
    np.savez_compressed(TILES / f"{r.tile_id}.npz", sm=sm_db.astype("float16"), lee=db(lee).astype("float16"),
                        iw=iw_db.astype("float16"), wc=wc, epsg=epsg, x0=x0, y1=y1, iw_x0=x0 + ox, iw_y1=y1 + oy)
    return {"status": "ok", "valid_sm": valid_sm, "valid_iw": float(np.isfinite(iw_db).mean()),
            "off_e_m": round(ox, 2), "off_n_m": round(oy, 2), "coarse_edge": edge, "ncc": round(ncc, 3),
            "corr10_nominal": round(c_nom, 3), "corr10": round(c_fin, 3), "looks": round(looks, 2),
            "land": round(float(((wc != 80) & (wc != 0)).mean()), 3),
            "sm_minus_iw_db": round(float(np.nanmedian(sm10) - np.nanmedian(iw_db)), 2), "epsg": epsg}


def one(r):
    try:
        sm_tif, st, sec = run_sm(r)
        if st == "sm_failed":
            tail = (SM_TIF / f"{r.tile_id}.log").read_text(errors="ignore")[-300:]
            return {"status": "sm_failed", "error": tail, "sm_sec": sec}
        res = pair(r, sm_tif)
        return {**res, "sm_sec": sec}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tiles", default="tiles.csv")
    ap.add_argument("--out", default="pairs.csv")
    a = ap.parse_args()
    SM_TIF.mkdir(parents=True, exist_ok=True)
    TILES.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / a.tiles)
    if a.only:
        t = t[t.tile_id.str.contains(a.only)]
    out_f = ROOT / a.out
    done = pd.read_csv(out_f) if out_f.exists() else pd.DataFrame(columns=["tile_id", "status"])
    skip = set(done[done.status == "ok"].tile_id)
    todo = t[~t.tile_id.isin(skip)]
    print(f"{len(t)} tiles, {len(todo)} to do", flush=True)
    rows, t0 = [], time.time()
    keep = ["tile_id", "region", "role", "slc", "iw_id", "iw_dt_days", "lon", "lat", "built", "n_xband"]
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, r): r for r in todo.itertuples()}
        for n, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            rows.append({k: getattr(r, k) for k in keep} | fu.result())
            print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s {r.tile_id} "
                  f"{ {k: rows[-1].get(k) for k in ('status', 'corr10_nominal', 'corr10', 'off_e_m', 'off_n_m', 'sm_sec')} }",
                  flush=True)
            if n % 5 == 0 or n == len(todo):
                allp = pd.concat([done[~done.tile_id.isin([x["tile_id"] for x in rows])], pd.DataFrame(rows)])
                allp.to_csv(out_f, index=False)
    print("done", pd.read_csv(out_f).status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
