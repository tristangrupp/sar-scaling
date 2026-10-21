"""Calibrate the synthetic speckle against real Sentinel-1 RTC.

Real: Experiment 4's single-orbit stacks (E:/SAR_superresolution/exp4_despeckling/data/<site>.npz, linear VV, 10 m). Homogeneous
  windows = 9x9 windows with the lowest 5 % CV in the temporal mean (exp 4's definition); ENL is
  measured on single dates there. Also the lag-1 autocorrelation of single-date / temporal-mean.
Synthetic: X-band training tiles degraded with K looks; homogeneous windows chosen on the
  speckle-free blurred image; same ENL and autocorrelation estimators.
Writes calib.csv and calib.json (chosen K).
"""
from __future__ import annotations

import common  # noqa: F401
from common import DEV, ROOT, SPIKE, blur_only, degrade, enl_windows, finish, homog_windows, tile_metadata

import json

import numpy as np
import pandas as pd
import torch


def lag1(r, sel_mask):
    """lag-1 autocorrelation (mean of x and y directions) of a ratio image within a mask."""
    r = r - r[sel_mask].mean()
    a = (r[:, :-1] * r[:, 1:])[sel_mask[:, :-1] & sel_mask[:, 1:]].mean()
    b = (r[:-1] * r[1:])[sel_mask[:-1] & sel_mask[1:]].mean()
    return float((a + b) / 2 / r[sel_mask].var())


def main():
    from scipy import ndimage as ndi
    rows = []
    from pathlib import Path
    for f in sorted(common.EXP4_DATA.glob("*.npz")):
        if f.stem.endswith("_stack"):
            continue
        try:
            z = np.load(f)
            vv = z["vv"].astype("float32")
        except Exception as e:
            print("skip", f.name, e, flush=True)
            continue
        valid = np.isfinite(vv).all(0)
        mean = np.where(valid, np.nanmean(vv, 0), np.nan)
        mf = np.where(valid, mean, np.nanmedian(mean))
        sel = homog_windows(mf, valid)
        Ls, acs = [], []
        hmask = ndi.binary_dilation(sel, np.ones((9, 9), bool)) & valid
        for t in range(0, vv.shape[0], max(1, vv.shape[0] // 8)):
            I = np.where(np.isfinite(vv[t]), vv[t], np.nanmedian(vv[t]))
            Ls.append(enl_windows(I, sel))
            acs.append(lag1(I / np.maximum(mf, 1e-12), hmask))
        rows.append(dict(source="real_s1", name=f.stem, n_dates=vv.shape[0], enl=np.median(Ls), lag1=np.median(acs)))
        print(rows[-1], flush=True)

    meta = tile_metadata()
    man = pd.read_csv(SPIKE / "manifest.csv")
    man = man[(man.status.isin(["ok", "cached"])) & (man.split == "train")].sample(40, random_state=0)
    for K in [3, 4, 5, 6]:
        Ls, acs = [], []
        for r in man.itertuples():
            fid = f"{r.provider}_{r.xband_id}"
            z = np.load(SPIKE / "data" / f"{fid}.npz")
            x = z["x"].astype("float32")
            if np.isfinite(x).mean() < 0.97:
                continue
            x = np.where(np.isfinite(x), x, np.nanmedian(x))
            az = meta.get(fid, {}).get("az", 270.0)
            xt = torch.from_numpy(x)[None, None].to(DEV)
            with torch.no_grad():
                I = (10 ** (degrade(xt, [az], looks=K) / 10))[0, 0].cpu().numpy().astype("float64")
                C = (10 ** (blur_only(xt, [az]) / 10))[0, 0].cpu().numpy().astype("float64")
            valid = np.ones_like(C, bool)
            sel = homog_windows(C, valid)
            hmask = ndi.binary_dilation(sel, np.ones((9, 9), bool))
            Ls.append(enl_windows(I, sel))
            acs.append(lag1(I / C, hmask))
        rows.append(dict(source="synthetic", name=f"K={K}", n_dates=len(Ls), enl=np.median(Ls), lag1=np.median(acs)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "calib.csv", index=False)
    real = df[df.source == "real_s1"]
    target = float(real.enl.median())
    syn = df[df.source == "synthetic"].copy()
    syn["k"] = syn.name.str[2:].astype(int)
    # ENL grows ~linearly in K; interpolate, but looks must be an integer in degrade()
    best = int(syn.iloc[(syn.enl - target).abs().argmin()].k)
    out = dict(real_enl_median=target, real_enl_range=[float(real.enl.min()), float(real.enl.max())],
               real_lag1_median=float(real.lag1.median()), looks=best,
               synthetic=syn[["k", "enl", "lag1"]].to_dict("records"))
    (ROOT / "calib.json").write_text(json.dumps(out, indent=1))
    print(out, flush=True)


if __name__ == "__main__":
    main()
    finish()
