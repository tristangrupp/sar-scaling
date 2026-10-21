"""Round 2 scene picks: inland Stripmap VV over land-cover types the pilot lacks.

From sm_scenes_all_vv.parquet (inventory_v2.py + inland_cells.py), per chosen cell: scenes with a
same-direction IW RTC VV within 6 days on Planetary Computer, skipping Sentinel-1C before July 2025
and Sentinel-1D before July 2026, spread over time. Writes scenes_v2.csv (with the ASF url, so
fetch_slc.py downloads it directly) for fetch_slc.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from common import pc_client

ROOT = Path(__file__).resolve().parent
WINDOW = 6

# region: (cells, n scenes, role, land-cover theme)
REGIONS = {
    "darien":       (["-77_7"], 2, "train", "tropical forest"),
    "venezuela":    (["-61_8"], 1, "train", "tropical forest + river"),
    "roraima":      (["-64_2"], 2, "train", "tropical forest"),
    "w_sahara":     (["-10_26", "-11_26", "-14_22", "-15_21", "-10_27", "-12_26"], 3, "train", "arid interior"),
    "tibesti":      (["14_20", "14_19", "13_19", "13_20", "14_21", "13_21"], 3, "train", "desert mountains"),
    "yemen_east":   (["46_15", "46_17", "46_16"], 2, "train", "arid plateau + oasis towns"),
    "iran":         (["53_35"], 2, "train", "arid mountains + towns"),
    "sahel":        (["1_13", "1_12", "9_13", "9_14", "9_12", "8_14"], 3, "train", "Sahel cropland + towns"),
    "lake_chad":    (["13_12", "14_12", "13_11", "14_9", "13_9", "14_10"], 2, "train", "Sahel wetland + cropland"),
    "amur":         (["128_51", "127_51"], 2, "train", "boreal cropland + forest"),
    "montreal":     (["-74_45"], 1, "train", "city + cropland"),
    "yemen_west":   (["44_15", "44_17"], 2, "holdout_near2", "highland cities + terraces, ~200 km from yemen_east"),
    "bangladesh":   (["91_23"], 2, "holdout_far2", "dense villages + paddy"),
}


def ok_platform(g, t):
    sat = g[:3]
    return not ((sat == "S1C" and t < pd.Timestamp("2025-07-01", tz="UTC")) or
                (sat == "S1D" and t < pd.Timestamp("2026-07-01", tz="UTC")))


def main():
    g = gpd.read_parquet(ROOT / "sm_scenes_all_vv.parquet")
    raw = pd.read_json(ROOT / "asf_sm_raw.json")
    url = dict(zip(raw.gn, raw.du))
    g["t"] = pd.to_datetime(g.start, utc=True)
    rows = []
    for reg, (cells, n, role, theme) in REGIONS.items():
        s = g[g.cell.isin(cells) & (g.inland_km > 10)].copy()
        s = s[[ok_platform(a, b) for a, b in zip(s.granule, s.t)]].sort_values("t")
        if s.empty:
            print(reg, "no scenes", flush=True)
            continue
        # check candidates spread over time until n pairs found
        idx = np.unique(np.linspace(0, len(s) - 1, min(len(s), 30)).round().astype(int))
        found = []
        for i in idx:
            r = s.iloc[i]
            if any(abs((r.t - f["t"]).days) < 20 for f in found):
                continue
            p = r.geometry.representative_point()
            items = [it for it in pc_client().search(
                collections=["sentinel-1-rtc"], bbox=[p.x - .05, p.y - .05, p.x + .05, p.y + .05],
                datetime=f"{r.t - pd.Timedelta(days=WINDOW):%Y-%m-%d}/{r.t + pd.Timedelta(days=WINDOW):%Y-%m-%d}").items()
                if "vv" in it.assets and (it.properties.get("sat:orbit_state") or "").upper() == r.orbit_dir.upper()]
            if not items:
                continue
            best = min(items, key=lambda it: abs(pd.Timestamp(it.datetime) - r.t))
            found.append({"granule": r.granule, "url": url.get(r.granule), "start": r.start, "lon": p.x, "lat": p.y,
                          "orbit_dir": r.orbit_dir.upper(), "beam": r.granule[4:6], "iw_id": best.id,
                          "iw_dt_days": abs((pd.Timestamp(best.datetime) - r.t).days), "region": reg, "role": role,
                          "theme": theme, "cell": r.cell, "t": r.t, "size_mb": r.size_mb})
            if len(found) == n:
                break
        print(f"{reg:11s} {role:14s} candidates={len(s):4d} picked={len(found)}", flush=True)
        rows += found
    out = pd.DataFrame(rows).drop(columns="t")
    out.to_csv(ROOT / "scenes_v2.csv", index=False)
    print(out[["region", "granule", "iw_dt_days", "size_mb"]].to_string(), flush=True)
    print("total", len(out), "scenes,", round(out.size_mb.sum() / 1024 * 0.55), "GB approx (VV only)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
