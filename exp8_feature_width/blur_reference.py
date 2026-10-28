"""Reference: would X-band itself show the feature at Sentinel-1 resolution?

For each accepted segment, take the stored along-line Capella profile, blur it with a 20 m boxcar in linear
intensity (about the Sentinel-1 IW resolution) and apply the same pre-set test as for Sentinel-1 (10 m windows,
the same 16 null lines, |z| > 2). This separates the resolution limit from the C-band versus X-band
difference. Capella's speckle after along-line averaging is lower than Sentinel-1's, so the reference is
optimistic. Writes blur_reference.csv (per segment) and prints the rate by width bin.
"""
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import contrast_stat, zscores  # noqa: E402

EDGES = [0.75, 2, 3, 5, 8, 12, 20, 30, 40]


def main():
    seg = pd.read_csv(HERE / "segments_all.csv")
    meta = pd.read_csv(HERE / "scene_meta.csv")
    good = set(meta[meta.osm_refined.astype(bool)].capella_id)
    rows = []
    for cid, g in seg[(seg.status == "ok") & seg.capella_id.isin(good)].groupby("capella_id"):
        z = np.load(HERE / "data" / cid / "profiles.npz")
        P = dict(zip(z["ids"], z["prof"]))
        for r in g.itertuples():
            lin = 10 ** (P[r.seg_id][0].astype(float) / 10)
            m = np.isfinite(lin)
            num = ndimage.uniform_filter1d(np.where(m, lin, 0), 40, mode="nearest")   # 40 samples x 0.5 m = 20 m
            den = ndimage.uniform_filter1d(m.astype(float), 40, mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                bl = np.where(den > 0.8, num / den, np.nan)
            o, nl, _ = contrast_stat(bl, r.cap_center, r.cap_width, 10.0)
            zz, _ = zscores(o, nl)
            rows.append({"seg_id": r.seg_id, "capella_id": cid, "type": r.type, "cap_width": r.cap_width,
                         "cap_contrast": r.cap_contrast, "blur_C": o, "blur_z": zz, "blur_det": bool(np.isfinite(zz) and abs(zz) > 2)})
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "blur_reference.csv", index=False)
    d["wbin"] = pd.cut(d.cap_width, EDGES, right=False)
    print(d.groupby("wbin", observed=False).blur_det.agg(["mean", "size"]).round(3).to_string(), flush=True)
    print("overall", round(d.blur_det.mean(), 3), "n", len(d), flush=True)
    print(d.groupby("type").blur_det.mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
