"""Re-register training / validation X-band edge targets against the 5 m Stripmap.

xband_prep.py registers X-band to the IW (so held-out tiles use inference inputs only). Against
the smooth 10 m IW that failed for 85% of training pairs (volcanic slopes, sparse structure), so
their edges sat at the providers' own geolocation (ICEYE GRD is not terrain-corrected). Training
tiles have Stripmap, so here the X-band is registered to it (masked normalized cross-correlation,
both smoothed, +-10 px at 5 m) and edges / distance / validity are shifted to match. Pairs whose
peak is weak (shift at the limit) are dropped from the arm B loss (status reg_failed).
Updates the npz files in place and xband_pairs.csv (columns sm_shift_r/c, sm_reg_ok).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from common import ROOT, TILES
from xband_prep import XB, edges


def main():
    xp = pd.read_csv(ROOT / "xband_pairs.csv")
    for c in ("sm_shift_r", "sm_shift_c", "sm_reg_ok"):
        if c not in xp:
            xp[c] = np.nan
    todo = xp[(xp.status == "ok") & xp.split.isin(["train", "val"]) & xp.sm_reg_ok.isna()]
    for i, r in todo.iterrows():
        f = XB / f"{r.tile_id}__{r.xband_id}.npz"
        z = dict(np.load(f))
        sm = np.load(TILES / f"{r.tile_id}.npz")["sm"].astype("float32")
        xv0 = z["valid"] | (~z["bad"] & ~z["change"] & np.isfinite(z["xdb"].astype("float32")))
        xdb = z["xdb"].astype("float32")
        xval = np.isfinite(xdb) & (z["dist"] < 50) | z["edges"]
        xval = ndimage.binary_erosion(z["valid"] | z["edges"], iterations=6) if (z["valid"].sum() > 5000) else xval
        a = ndimage.gaussian_filter(np.nan_to_num(sm, nan=np.nanmedian(sm)), 1.5)
        b = ndimage.gaussian_filter(np.nan_to_num(xdb, nan=np.nanmedian(xdb)), 1.5)
        if xval.sum() < 5000:
            xp.loc[i, ["sm_reg_ok"]] = False
            continue
        sh = phase_cross_correlation(a, b, reference_mask=np.ones_like(xval), moving_mask=xval)
        sh = np.asarray(sh[0] if isinstance(sh, tuple) else sh, float)
        ok = np.abs(sh).max() < 10
        xp.loc[i, ["sm_shift_r", "sm_shift_c", "sm_reg_ok"]] = [sh[0], sh[1], ok]
        if not ok:
            continue
        xs = ndimage.shift(np.nan_to_num(xdb, nan=np.nanmedian(xdb)), sh, order=1, mode="nearest")
        valid = ndimage.shift(z["valid"].astype("float32"), sh, order=0) > 0.5
        e = edges(xs, valid | (ndimage.shift(np.isfinite(xdb).astype("float32"), sh, order=0) > 0.5))
        z.update(xdb=xs.astype("float16"), edges=e, valid=valid & ~z["bad"] & ~z["change"],
                 dist=np.minimum(ndimage.distance_transform_edt(~e), 50).astype("float16"))
        np.savez_compressed(f, **z)
    xp.loc[(xp.split.isin(["train", "val"])) & (xp.sm_reg_ok == False) & (xp.status == "ok"), "status"] = "reg_failed"  # noqa: E712
    xp.to_csv(ROOT / "xband_pairs.csv", index=False)
    t = xp[xp.split.isin(["train", "val"])]
    print(t.groupby(["split", "provider", "status"]).size().to_string())
    print("median |shift| px:", float(np.hypot(t.sm_shift_r, t.sm_shift_c).median()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
