"""Calibrate the Capella change rule before any Sentinel-1 scoring.

Inputs: r2_calib_pairs.csv and data/r2/calib_*.npz from `r2_run.py --calib` (five pairs: Mojave,
Kilauea, San Jose, Pilbara mine, Black Rock). For each pair: the spread of the median-corrected
log-ratio (robust sd from the MAD), the implied equivalent number of looks (ENL), and the density of
change patches (per km^2 of usable ground) for each (minimum size, seed level) combination.

Speckle-only reference: two independent gamma(ENL) intensity images on a 1280 x 1280 grid, same
log-ratio, same 3 dB candidate rule, same patch rule. The resulting patch density is what pure
speckle would produce; the chosen rule should keep that well under the densities seen at the
quiet sites. Spatially correlated speckle (oversampled sources) makes real noise clump more than
this independent-pixel simulation, so the quiet sites set the practical floor.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import brentq
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from r2_run import THRESH, patches, tol_change  # noqa: E402

DB = 10 / np.log(10)


def enl_from_sd(sd_db):
    """ENL whose log-ratio of two independent gamma(L) intensities has this sd (dB)."""
    f = lambda L: DB * np.sqrt(2 * polygamma(1, L)) - sd_db
    return brentq(f, 0.3, 500)


def sim(L, n=1280, seed=0):
    rng = np.random.default_rng(seed)
    a = DB * np.log(rng.gamma(L, 1 / L, (n, n)))
    b = DB * np.log(rng.gamma(L, 1 / L, (n, n)))
    return a, b


def main():
    c = pd.read_csv(ROOT / "r2_calib_pairs.csv")
    c = c[c.status == "calib"]
    pd.set_option("display.width", 250)
    cols = [k for k in c.columns if k.startswith("np_")]
    print("per-pair log-ratio spread and patch density (patches per km^2 of usable ground)")
    c["enl"] = [round(enl_from_sd(s), 1) for s in c.lr_mad_sd_db]
    print(c[["tag", "src_res_m", "coreg_dy_px", "coreg_dx_px", "c_use_frac", "mask_edge", "lr_mad_sd_db", "enl", "lr_frac_gt3"]].to_string(index=False))
    t = c.set_index("tag")[cols].T
    # np_r<tolerance px>_<min px>_<seed dB>; r0 = plain log-ratio, r1nomask = r1 without edge/shadow/bright masks
    print(t.to_string())
    print("\nspeckle-only simulation (independent pixels), patches per km^2:")
    for L in sorted(set(np.round(c.enl.clip(lower=3), 0))):
        a, b = sim(L)
        use = np.ones_like(a, bool)
        row = {"ENL": L}
        for tr in (0, 1):
            mg = tol_change(a, b, use, tr) if tr else b - a
            row[f"r{tr}_frac_gt3"] = round(float((np.abs(mg) > THRESH).mean()), 5)
            for mp in (1, 2, 4):
                for sd in (3.0, 4.0, 5.0):
                    lab, size, _, _ = patches(mg, use, mp, sd)
                    row[f"r{tr}_{mp}px/{sd}"] = round(len(size) / (use.size * 4e-6), 1)
        print(row)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
