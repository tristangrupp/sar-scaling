"""Round 2 tile selection: 3.2 km tiles for the inland scenes in scenes_v2.csv, appended to the
round 1 tiles in tiles_v2.csv (tiles.csv itself stays untouched; exp5 uses it).

Same scoring as select_tiles.py (WorldCover land >= 0.35, built-up share, class entropy, inside
the Stripmap and same-pass IW footprints). Held-out separation: no training tile within 40 km of a
tile of a round 2 held-out region (yemen_west vs yemen_east share a 200 km neighborhood).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union

import select_tiles as st
from common import DATA, ROOT

SEP_KM = 40


def main():
    sc = pd.read_csv(ROOT / "scenes_v2.csv")
    log = pd.read_csv(DATA / "fetch_log.csv")
    log = log[log.status == "ok"]
    sc = sc.merge(log[["granule", "slc", "safe"]], on="granule")
    sc = sc[[Path(s).joinpath(".complete").exists() for s in sc.safe]]
    allg = gpd.read_parquet(ROOT / "sm_scenes_all_vv.parquet").set_index("granule")
    xb = st.xband_items()
    prev = pd.read_csv(ROOT / "tiles.csv")
    have = pd.read_csv(ROOT / "tiles_v2.csv") if (ROOT / "tiles_v2.csv").exists() else prev
    sc = sc[~sc.slc.isin(have.slc)]
    out = []
    for s in sc.itertuples():
        try:
            geom = allg.loc[s.granule, "geometry"]
            iws = st.same_pass(s.iw_id, geom)
            fp = geom.intersection(unary_union([shape(i.geometry) for i in iws]))
            wc, tr = st.worldcover(fp.bounds)
            cand = st.score_candidates(fp, wc, tr, xb[xb.intersects(fp)], s.region)
            k = st.K_HOLDOUT if s.role.startswith("holdout") else st.K_DEFAULT
            sel = st.pick(cand, k) if len(cand) else pd.DataFrame()
        except Exception as e:
            print(f"{s.region} {s.slc[:40]} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if sel.empty:
            print(f"{s.region:12s} {s.slc[:40]} no land candidates", flush=True)
            continue
        sel = sel.drop(columns=["Index"], errors="ignore").assign(
            region=s.region, role=s.role, slc=s.slc, safe=s.safe, iw_id=";".join(i.id for i in iws),
            iw_dt_days=s.iw_dt_days, orbit_dir=s.orbit_dir, round=2, theme=s.theme)
        sel["tile_id"] = [f"{s.region}_{s.slc[17:32]}_{s.slc[4:6]}_{i}" for i in range(len(sel))]
        out.append(sel)
        print(f"{s.region:12s} {s.slc[:40]} cand={len(cand):4d} picked={len(sel)} built={sel.built.mean():.2f}", flush=True)
    t = pd.concat([have.assign(round=have.get("round", 1))] + out, ignore_index=True)
    # held-out separation for round 2 holdouts
    ho = t[t.role.isin(["holdout_near2", "holdout_far2"])][["lon", "lat"]].to_numpy()
    if len(ho):
        d = np.array([np.min(np.hypot((ho[:, 0] - a) * np.cos(np.radians(b)), ho[:, 1] - b)) * 111
                      for a, b in zip(t.lon, t.lat)])
        drop = (~t.role.str.startswith("holdout")) & (t.role != "val") & (d < SEP_KM)
        print("training tiles dropped for holdout separation:", int(drop.sum()), t[drop].region.value_counts().to_dict())
        t = t[~drop]
    t.to_csv(ROOT / "tiles_v2.csv", index=False)
    print(t[t["round"] == 2].groupby(["role", "region"]).size().to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
