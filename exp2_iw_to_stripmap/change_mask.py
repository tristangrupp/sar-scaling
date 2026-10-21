"""Change mask from the IW passes that bracket each Stripmap acquisition.

For every built tile (pairs_final_v2.csv): find the IW RTC VV passes on the paired IW's track (same
relative orbit) within +-36 days of the Stripmap date, and take the last one before and the first one after it
(they can include the pass paired with the tile). Both are read on the tile's co-registered IW
grid; a 10 m pixel counts as changed when its 3x3-mean backscatter differs by more than THRESH dB
between the two passes. Speckle alone (about 4.4 looks, 3x3 mean, so a difference sd near 1.1 dB)
puts roughly 1% of stable pixels over 3 dB; diff_sd_db records the observed spread per tile.
Shares are reported on land pixels.
Writes DATA/change/<tile_id>.npz (change10: bool 320x320, diff_db: float16) and change_v2.csv.
Tiles without a pass on one side get no mask (status one_sided).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import ndimage

from common import DATA, NL, ROOT, TILES, db, pc_client, pc_item, read_iw

OUT = DATA / "change"
THRESH = 3.0
WIN = 36


def passes(lon0, lat0, lon1, lat1, t, orbit_dir, rel_orbit):
    """Passes on the paired IW's track only: other tracks see the ground at another incidence and
    look direction, which reads as 'change' on slopes and in towns (first try: 20-35% flagged)."""
    its = [i for i in pc_client().search(collections=["sentinel-1-rtc"], bbox=[lon0, lat0, lon1, lat1],
                                         datetime=f"{t - pd.Timedelta(days=WIN):%Y-%m-%d}/{t + pd.Timedelta(days=WIN):%Y-%m-%d}").items()
           if "vv" in i.assets and (i.properties.get("sat:orbit_state") or "").upper() == orbit_dir
           and i.properties.get("sat:relative_orbit") == rel_orbit]
    g = {}
    for i in its:
        k = (i.datetime.strftime("%Y-%m-%d"), i.properties.get("sat:absolute_orbit"))
        g.setdefault(k, []).append(i)
    return [(pd.Timestamp(v[0].datetime), v) for v in g.values()]


def one(r, before, after):
    z = np.load(TILES / f"{r.tile_id}.npz")
    epsg, ix0, iy1 = int(z["epsg"]), float(z["iw_x0"]), float(z["iw_y1"])
    land10 = ((z["wc"] != 80) & (z["wc"] != 0)).reshape(NL, 2, NL, 2).all(axis=(1, 3))
    a = db(read_iw(before[1], epsg, ix0, iy1, NL))
    b = db(read_iw(after[1], epsg, ix0, iy1, NL))
    m = np.isfinite(a) & np.isfinite(b)
    if m.mean() < 0.8:
        return {"status": "iw_sparse"}
    k = np.ones((3, 3)) / 9
    d = ndimage.convolve(np.where(m, a, 0), k) - ndimage.convolve(np.where(m, b, 0), k)
    ok = ndimage.binary_erosion(m, iterations=1)
    ch = (np.abs(d) > THRESH) & ok
    np.savez_compressed(OUT / f"{r.tile_id}.npz", change10=ch, diff_db=d.astype("float16"), valid10=ok)
    lm = land10 & ok
    return {"status": "ok", "before": before[0].strftime("%Y-%m-%d"), "after": after[0].strftime("%Y-%m-%d"),
            "gap_days": (after[0] - before[0]).days, "change_land": round(float(ch[lm].mean()), 4) if lm.any() else np.nan,
            "change_all": round(float(ch[ok].mean()), 4), "diff_sd_db": round(float(np.std(d[lm])), 2) if lm.any() else np.nan}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / "tiles_v2.csv")
    pf = pd.read_csv(ROOT / "pairs_final_v2.csv")
    t = t[t.tile_id.isin(pf[pf.use].tile_id)]
    done = pd.read_csv(ROOT / "change_v2.csv") if (ROOT / "change_v2.csv").exists() else pd.DataFrame(columns=["tile_id", "status"])
    t = t[~t.tile_id.isin(done[done.status == "ok"].tile_id)]
    jobs = []
    for slc, g in t.groupby("slc"):
        ts = pd.Timestamp(slc[17:32], tz="UTC")
        rel = pc_item("sentinel-1-rtc", g.iw_id.iloc[0].split(";")[0]).properties.get("sat:relative_orbit")
        ps = passes(g.lon.min() - .03, g.lat.min() - .03, g.lon.max() + .03, g.lat.max() + .03, ts, g.orbit_dir.iloc[0], rel)
        bef = [p for p in ps if p[0] < ts - pd.Timedelta(hours=1)]
        aft = [p for p in ps if p[0] > ts + pd.Timedelta(hours=1)]
        b = max(bef, key=lambda p: p[0]) if bef else None
        a = min(aft, key=lambda p: p[0]) if aft else None
        print(f"{slc[:40]} passes={len(ps)} before={b[0].date() if b else None} after={a[0].date() if a else None}", flush=True)
        for r in g.itertuples():
            jobs.append((r, b, a))
    rows, t0 = [], time.time()

    def run(j):
        r, b, a = j
        if b is None or a is None:
            return {"status": "one_sided"}
        try:
            return one(r, b, a)
        except Exception as e:
            return {"status": "error", "error": f"{type(e).__name__}: {e}"[:200]}

    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for n, fu in enumerate(as_completed(futs), 1):
            r = futs[fu][0]
            rows.append({"tile_id": r.tile_id, "region": r.region, "role": r.role, **fu.result()})
            if n % 25 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    out = pd.concat([done[done.status == "ok"], pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(ROOT / "change_v2.csv", index=False)
    ok = out[out.status == "ok"]
    print(out.status.value_counts().to_dict(), flush=True)
    print(ok.groupby("region")[["gap_days", "change_land", "diff_sd_db"]].median().round(4).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
