"""Despeckling behaviour on real Sentinel-1, using Experiment 4's held-out sites and definitions.

For each exp4 test site (never used by any exp5 model), three single dates are the input; the
site's temporal mean (20-27 dates, single orbit) is the speckle-poor reference. Outputs are
mapped back to dB with a gain and offset fitted by least squares against the model's own input at
10 m (outputs live in target-standardized units, so de-standardizing with the noisy input's std would
inflate contrast; mean_ratio_raw reports that naive version), converted to power and averaged back
to 10 m blocks (O10).
  enl10        ENL of O10 in exp4's homogeneous windows (9x9 at 10 m, lowest 5 % temporal-mean CV)
  enl_native   same windows, at the output grid (9*s x 9*s pixels)
  mean_ratio   mean(O10) / mean(input) over valid pixels (1 = brightness preserved)
  mean_ratio_h same, homogeneous windows only
  ratio_enl    ENL of the ratio image input / O10 in homogeneous windows (should approach the input ENL)
  ratio_struct Pearson r between the 5x5-smoothed log ratio image and the log temporal mean:
               0 = the ratio is structureless speckle; > 0 = the model removed or altered real features
  mae_ref_db   |O10 - temporal mean| in dB, median over valid pixels (accuracy of the 10 m picture)
"""
from __future__ import annotations

import common  # noqa: F401
from common import (adapt, DEV, PROJ, ROOT, S1_TO_SENSOR_AZ, bicubic, enl_windows, finish, homog_windows, load_exp1,
                    load_model, meta_vec, norm_stats, run_model)

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

MODELS = ["x4_cond", "x4_unc", "x4_aug", "x2_cond", "x2_unc", "c4", "c2"]


def pool(a, f):
    n0, n1 = a.shape[0] // f, a.shape[1] // f
    return a[:n0 * f, :n1 * f].reshape(n0, f, n1, f).mean(axis=(1, 3))


def main():
    spec = importlib.util.spec_from_file_location("exp4_common", PROJ / "exp4_despeckling/common.py")
    c4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c4)
    models = {k: load_model(ROOT / "ckpt" / f"{k}.pt")[0] for k in MODELS if (ROOT / "ckpt" / f"{k}.pt").exists()}
    exp1, which = load_exp1()
    print("models", list(models), which, flush=True)
    rows = []
    for name, (lon, lat, lc, split, *_r) in c4.SITES.items():
        if split != "test":
            continue
        f = common.EXP4_DATA / f"{name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        vv = z["vv"].astype("float32")
        valid = np.isfinite(vv).all(0)
        ref = np.where(valid, np.nanmean(vv, 0), np.nan)
        reff = np.where(valid, ref, np.nanmedian(ref))
        sel = homog_windows(reff, valid)
        az = S1_TO_SENSOR_AZ.get(str(z["orbit"]), 258.0)
        mv = meta_vec(az, 37.0, "VV", "X")
        for t in np.linspace(0, vv.shape[0] - 1, 3).astype(int):
            I = vv[t]
            If = np.where(np.isfinite(I), I, np.nanmedian(I))
            db = 10 * np.log10(np.maximum(If, 1e-10))
            med, sd = norm_stats(db)
            s = (db - med) / sd
            outs = {"input (10 m)": (s, 1), "bicubic4": (bicubic(s, 4), 4),
                    "exp1_cnn": (run_model(exp1, s, kind="exp1"), 4)}
            for k, G in models.items():
                outs[k] = (run_model(G, s, mv), G.scale)
            if "x4_cond" in models:
                outs["x4_cond+adapt"] = (run_model(models["x4_cond"], adapt(s), mv), 4)
            for k, (o, sc) in outs.items():
                raw10 = pool(10 ** ((o * sd + med) / 10), sc) if sc > 1 else 10 ** ((o * sd + med) / 10)
                # back to dB: gain and offset fitted against the model's own input at 10 m (no reference used)
                o10 = pool(o, sc) if sc > 1 else o
                A = np.c_[o10[valid], np.ones(valid.sum())]
                a_, b_ = np.linalg.lstsq(A, db[valid], rcond=None)[0]
                lin = 10 ** ((a_ * o + b_) / 10)
                O10 = pool(lin, sc) if sc > 1 else lin
                ratio = If / np.maximum(O10, 1e-12)
                lr = ndi.uniform_filter(np.log(np.maximum(ratio, 1e-6)), 5)
                ok = valid & np.isfinite(lr)
                hs = ndi.binary_dilation(sel, np.ones((9, 9), bool)) & valid
                if sc > 1:
                    seln = np.zeros(lin.shape, bool)
                    rr, cc = np.nonzero(sel)
                    seln[rr * sc + sc // 2, cc * sc + sc // 2] = True
                    enl_n = enl_windows(lin, seln, w=9 * sc)
                else:
                    enl_n = enl_windows(lin, sel)
                rows.append(dict(site=name, landcover=lc, date_idx=int(t), model=k, gain_db=float(a_ * sd) if k != "input (10 m)" else 1.0,
                                 mean_ratio_raw=float(raw10[valid].mean() / If[valid].mean()),
                                 enl10=enl_windows(O10, sel), enl_native=enl_n,
                                 mean_ratio=float(O10[valid].mean() / If[valid].mean()),
                                 mean_ratio_h=float(O10[hs].mean() / If[hs].mean()),
                                 ratio_enl=enl_windows(ratio, sel) if k != "input (10 m)" else np.nan,
                                 ratio_struct=float(np.corrcoef(lr[ok], np.log(reff[ok]))[0, 1]) if k != "input (10 m)" else np.nan,
                                 mae_ref_db=float(np.median(np.abs(10 * np.log10(np.maximum(O10, 1e-10)) - 10 * np.log10(reff))[valid]))))
            print(name, t, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "despeckle_scores.csv", index=False)
    summ = df.groupby("model")[["mean_ratio_raw", "enl10", "enl_native", "mean_ratio", "mean_ratio_h", "ratio_enl", "ratio_struct",
                                "mae_ref_db"]].median().round(3)
    summ.to_csv(ROOT / "despeckle_summary.csv")
    print(summ.to_string(), flush=True)


if __name__ == "__main__":
    main()
    finish()
