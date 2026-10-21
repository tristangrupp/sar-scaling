"""Replace pilot scenes whose SLC is missing on ASF (S1C commissioning Apr-May 2025, S1D May 2026).

Same rule as pick_scenes.py (same-direction VV IW RTC within 6 days), restricted to scenes not
already picked, skipping S1C before 2025-07-01 and S1D before 2026-07-01, and confirmed on ASF
before selection. Madrid's cell holds only S1C April 2025 scenes, so it is replaced by the
Liberia coast cell (-11_6), another continental training region.
Writes replacement_scenes.csv (input to fetch_slc.py) and appends to pilot_scenes.csv rows.
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
sys.path.insert(0, str(ROOT))
from fetch_slc import resolve_slc  # noqa: E402

WINDOW = 6
FAILED_LOG = Path("E:/SAR_superresolution/exp2_iw_to_stripmap/data/fetch_log_pilot.csv")
SUBSTITUTE = {"madrid": ("-11_6", "liberia")}


def too_early(granule, t):
    sat = granule[:3]
    return (sat == "S1C" and t < pd.Timestamp("2025-07-01", tz="UTC")) or \
           (sat == "S1D" and t < pd.Timestamp("2026-07-01", tz="UTC"))


def main():
    pilot = pd.read_csv(ROOT / "pilot_scenes.csv")
    log = pd.read_csv(FAILED_LOG)
    failed = set(log[log.status != "ok"].granule)
    need = pilot[pilot.granule.isin(failed)].groupby(["region", "role", "cell"]).size()
    sm = gpd.read_parquet(ROOT / "sm_scenes.parquet")
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    picks = []
    for (region, role, cell), n in need.items():
        if region in SUBSTITUTE:
            cell, region = SUBSTITUTE[region]
        s = sm[(sm.cell == cell) & ~sm.granule.isin(pilot.granule)].copy()
        s["t"] = pd.to_datetime(s.start)
        s = s[[not too_early(g, t) for g, t in zip(s.granule, s.t)]]
        if s.empty:
            print(f"{region}: no candidates", flush=True)
            continue
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
                rows.append({"granule": r.granule, "start": r.start, "lon": p.x, "lat": p.y,
                             "orbit_dir": r.orbit_dir, "beam": r.beam, "iw_id": best.iw_id,
                             "iw_dt_days": abs((best.iw_t - r.t).days), "t": r.t})
        ok = pd.DataFrame(rows)
        if ok.empty:
            print(f"{region}: no same-direction pairs", flush=True)
            continue
        ok = ok.sort_values("t").reset_index(drop=True)
        # spread over time; check ASF in that order, take the first n that exist
        order = np.argsort(np.abs(np.linspace(0, 1, len(ok)) - 0.5))[::-1] if len(ok) > n else range(len(ok))
        order = list(dict.fromkeys(list(np.linspace(0, len(ok) - 1, min(len(ok), 8)).round().astype(int)) + list(order)))
        got = []
        for i in order:
            r = ok.iloc[i]
            gran, url = resolve_slc(r.start, r.lon, r.lat)
            if url:
                got.append(r)
            if len(got) == n:
                break
        sel = pd.DataFrame(got).assign(region=region, role=role, cell=cell)
        picks.append(sel)
        print(f"{region:12s} {role:14s} pairs={len(ok):3d} needed={n} picked={len(sel)}", flush=True)
    out = pd.concat(picks).drop(columns="t")
    out.to_csv(ROOT / "replacement_scenes.csv", index=False)
    print(out[["region", "granule", "iw_dt_days"]].to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
