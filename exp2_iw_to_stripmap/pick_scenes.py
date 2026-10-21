"""Pick the pilot Stripmap scenes: per region, scenes with a same-direction VV IW RTC within 6 days,
spread evenly over time. Writes pilot_scenes.csv (input to fetch_slc.py) with the IW match.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
from pystac_client import Client

ROOT = Path(__file__).resolve().parent
WINDOW = 6

# cell -> (region name, role, number of scenes)
REGIONS = {
    "43_-12": ("comoros", "train", 5), "128_-8": ("timor", "train", 5), "-152_-17": ("tahiti", "train", 5),
    "160_-12": ("solomon", "train", 5), "55_-22": ("reunion_s", "train", 4), "-82_19": ("cayman", "train", 4),
    "-25_14": ("fogo", "train", 4), "-118_35": ("mojave", "train", 4), "-93_-1": ("galapagos", "train", 3),
    "-3_40": ("madrid", "train", 3), "1_5": ("togo", "train", 3),
    "-96_29": ("houston", "holdout_xband", 6), "-118_37": ("owens", "holdout_near", 4),
    "115_-29": ("w_australia", "holdout_far", 6),
}


def main():
    sm = gpd.read_parquet(ROOT / "sm_scenes.parquet")
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    picks = []
    for cell, (name, role, n) in REGIONS.items():
        s = sm[sm.cell == cell].copy()
        s["t"] = pd.to_datetime(s.start)
        c = s.geometry.unary_union.centroid
        t0, t1 = s.t.min() - pd.Timedelta(days=WINDOW), s.t.max() + pd.Timedelta(days=WINDOW)
        items = [i for i in client.search(collections=["sentinel-1-rtc"],
                                          bbox=[c.x - .05, c.y - .05, c.x + .05, c.y + .05],
                                          datetime=f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}").items() if "vv" in i.assets]
        iw = pd.DataFrame([{"iw_id": i.id, "iw_t": pd.Timestamp(i.datetime),
                            "iw_dir": (i.properties.get("sat:orbit_state") or "").upper()} for i in items])
        rows = []
        for r in s.itertuples():
            if iw.empty:
                break
            cand = iw[(iw.iw_dir == r.orbit_dir) & ((iw.iw_t - r.t).abs() <= pd.Timedelta(days=WINDOW))]
            if len(cand):
                best = cand.iloc[(cand.iw_t - r.t).abs().argmin()]
                p = r.geometry.representative_point()
                rows.append({"granule": r.granule, "start": r.start, "lon": p.x, "lat": p.y, "orbit_dir": r.orbit_dir,
                             "beam": r.beam, "iw_id": best.iw_id, "iw_dt_days": abs((best.iw_t - r.t).days),
                             "t": r.t})
        ok = pd.DataFrame(rows).sort_values("t")
        if ok.empty:
            print(f"{name}: no same-direction pairs", flush=True)
            continue
        idx = np.unique(np.linspace(0, len(ok) - 1, min(n, len(ok))).round().astype(int))
        sel = ok.iloc[idx].assign(region=name, role=role, cell=cell)
        picks.append(sel)
        print(f"{name:12s} {role:14s} pairs={len(ok):3d} picked={len(sel)}", flush=True)
    out = pd.concat(picks).drop(columns="t")
    out.to_csv(ROOT / "pilot_scenes.csv", index=False)
    print(f"total {len(out)} scenes -> pilot_scenes.csv", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
