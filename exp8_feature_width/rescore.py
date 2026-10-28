"""Post-hoc variant of the Sentinel-1 detection rule (results.md, "dB-domain contrast").

The pre-set rule takes dB(mean intensity in the center window) minus the mean of the two flank levels in dB.
Across a plain boundary between two fields of different brightness, that mixes a linear mean with a dB mean
and yields a positive contrast with no linear feature present (up to about 1 dB for a 6 dB step). The variant
averages the along-line profile in dB in all three windows, so a symmetric step cancels.

Reads data/<id>/profiles.npz (the stored along-line mean profiles, in dB) and segments_all.csv. First checks
that the stored profiles reproduce the pre-set statistic, then writes segments_all_db.csv with the same
columns recomputed under the variant. Nothing else changes: windows, null lines, thresholds.
"""
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import D, THR1, THR2, contrast_stat, zscores  # noqa: E402


def contrast_stat_db(pdb, c, w, r):
    hc = max(r / 2, w / 2)
    f0 = max(1.5 * r, w / 2 + r)
    f1 = f0 + 15

    def C(cc):
        dd = D - cc
        cen, L, R = pdb[np.abs(dd) <= hc], pdb[(dd >= -f1) & (dd <= -f0)], pdb[(dd >= f0) & (dd <= f1)]
        if np.isfinite(cen).mean() < 0.8 or np.isfinite(L).mean() < 0.8 or np.isfinite(R).mean() < 0.8:
            return np.nan
        return np.nanmean(cen) - (np.nanmean(L) + np.nanmean(R)) / 2

    offs = f0 + 15 + w / 2 + 10 + 5 * np.arange(8)
    return C(c), np.array([C(c + o) for o in offs] + [C(c - o) for o in offs])


def main():
    seg = pd.read_csv(HERE / "segments_all.csv")
    out = seg.copy()
    names = ["cap_db", "single_vv", "single_vh", "multi_vv", "multi_vh"]
    check = []
    for cid, g in seg[seg.status == "ok"].groupby("capella_id"):
        z = np.load(HERE / "data" / cid / "profiles.npz")
        P = dict(zip(z["ids"], z["prof"]))
        for i, r in g.iterrows():
            pr = P[r.seg_id]
            zn = {}
            for prod in ("single", "multi"):
                for pol in ("vv", "vh"):
                    k = f"{prod}_{pol}"
                    pdb = pr[names.index(k)].astype(float)
                    o0, n0, _ = contrast_stat(10 ** (pdb / 10), r.cap_center, r.cap_width, 10.0)
                    check.append((zscores(o0, n0)[0], r[f"{k}_z"]))
                    o, nl = contrast_stat_db(pdb, r.cap_center, r.cap_width, 10.0)
                    zz, znn = zscores(o, nl)
                    out.at[i, f"{k}_C"] = o
                    out.at[i, f"{k}_null_sd"] = np.nanstd(nl, ddof=1)
                    out.at[i, f"{k}_z"] = zz
                    out.at[i, f"{k}_det"] = bool(np.isfinite(zz) and abs(zz) > THR1)
                    out.at[i, f"{k}_far_n"] = int(np.nansum(np.abs(znn) > THR1))
                    out.at[i, f"{k}_far_m"] = int(np.isfinite(znn).sum())
                    zn[k] = znn
                zv, zh = out.at[i, f"{prod}_vv_z"], out.at[i, f"{prod}_vh_z"]
                out.at[i, f"{prod}_either_det"] = bool(np.nanmax([abs(zv), abs(zh)]) > THR2) if np.isfinite([zv, zh]).any() else False
                ze = np.fmax(np.abs(zn[f"{prod}_vv"]), np.abs(zn[f"{prod}_vh"]))
                out.at[i, f"{prod}_either_far_n"] = int(np.nansum(ze > THR2))
                out.at[i, f"{prod}_either_far_m"] = int(np.isfinite(ze).sum())
    c = np.array(check, dtype=float)
    m = np.isfinite(c).all(axis=1)
    print(f"reproduction of the pre-set z from stored profiles: n={m.sum()}, max |diff| = "
          f"{np.max(np.abs(c[m, 0] - c[m, 1])):.4f}, corr = {np.corrcoef(c[m, 0], c[m, 1])[0, 1]:.5f}", flush=True)
    out.to_csv(HERE / "segments_all_db.csv", index=False)
    ok = out.status == "ok"
    for k in ("single_either", "multi_either"):
        a = seg.loc[ok, f"{k}_det"].map({True: 1, False: 0, "True": 1, "False": 0}).mean()
        b = out.loc[ok, f"{k}_det"].map({True: 1, False: 0, "True": 1, "False": 0}).mean()
        print(k, "detected share: pre-set", round(a, 3), "dB-domain", round(b, 3), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
