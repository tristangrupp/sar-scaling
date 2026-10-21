"""Temporal statistics and change masks per site.

For each site's single-orbit stack (linear power):
  L      single-date ENL, measured in spatially homogeneous areas (9x9 windows whose
         temporal-mean CV is in the lowest 5 %), median over dates
  cv_t   per-pixel temporal coefficient of variation
  stable cv_t <= the 99th percentile of the CV of N independent gamma(L) samples (Monte Carlo with
         the site's own N and L): pixels that vary more than speckle alone allows are excluded
  sum, n per-pixel sums so any leave-one-out mean is (sum - I_t) / (n - 1)
Writes E:/SAR_superresolution/exp4_despeckling/data/<site>_stack.npz and stack_summary.csv in exp4_despeckling/.
"""
from __future__ import annotations

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def homog_windows(mean_lin, valid, w=9, q=5):
    m = ndi.uniform_filter(mean_lin, w)
    v = ndi.uniform_filter(mean_lin ** 2, w) - m ** 2
    cv = np.sqrt(np.maximum(v, 0)) / np.maximum(m, 1e-12)
    ok = ndi.minimum_filter(valid.astype(np.uint8), w).astype(bool)
    # non-overlapping centres
    centres = np.zeros_like(ok)
    centres[w // 2::w, w // 2::w] = True
    sel = ok & centres
    thr = np.percentile(cv[sel], q)
    return sel & (cv <= thr)


def enl_at(img, sel, w=9):
    m = ndi.uniform_filter(img, w)
    v = ndi.uniform_filter(img ** 2, w) - m ** 2
    e = m[sel] ** 2 / np.maximum(v[sel], 1e-20)
    return float(np.median(e))


def mc_cv_threshold(L, n, q=99, reps=200000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(L, 1.0 / L, size=(reps, n))
    cv = x.std(1, ddof=1) / x.mean(1)
    return float(np.percentile(cv, q))


def main():
    import sys
    pol = sys.argv[1] if len(sys.argv) > 1 else "vv"     # "vh": round-2 cross-pol stacks
    sfx = "" if pol == "vv" else "_vh"
    rows = []
    for name in SITES:
        f = DATA / f"{name}{sfx}.npz"
        if not f.exists():
            print("missing", name, flush=True)
            continue
        z = np.load(f)
        vv = z[pol]
        z0 = np.load(DATA / f"{name}.npz")
        valid_t = np.isfinite(vv)
        n = valid_t.sum(0)
        s = np.nansum(vv, 0)
        valid = n == vv.shape[0]
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        mfill = np.where(valid, mean, np.nanmedian(mean))
        sel = homog_windows(mfill, valid)
        Ls = [enl_at(np.where(np.isfinite(vv[t]), vv[t], np.nanmedian(vv[t])), sel) for t in range(vv.shape[0])]
        L = float(np.median(Ls))
        sd = np.nanstd(vv, 0, ddof=1)
        cv_t = sd / np.maximum(mean, 1e-12)
        thr = mc_cv_threshold(L, vv.shape[0])
        stable = valid & (cv_t <= thr)
        np.savez_compressed(DATA / f"{name}{sfx}_stack.npz", sum=s.astype("float32"), n=n.astype("int16"),
                            stable=stable, valid=valid, cv_t=cv_t.astype("float32"), homog=sel, L=L, thr=thr)
        lon, lat, lc, split, *_ = SITES[name]
        rows.append(dict(site=name, landcover=lc, split=split, n_dates=vv.shape[0], first=str(z0["dates"][0]),
                         last=str(z0["dates"][-1]), orbit=str(z0["orbit"]), L_single=round(L, 2),
                         cv_speckle=round(1 / np.sqrt(L), 3), cv_thr99=round(thr, 3),
                         stable_frac=round(float(stable.mean()), 3), valid_frac=round(float(valid.mean()), 3),
                         n_homog=int(sel.sum()), wc_classes=" ".join(map(str, np.unique(z0["wc"])))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / f"stack_summary{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
    finish()
