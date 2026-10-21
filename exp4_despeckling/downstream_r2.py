"""Round 2 downstream tests.

1. WorldCover land cover on the six held-out sites (as round 1: same blocks, same random forest),
   from E:/SAR_superresolution/exp4_despeckling/data/out_r2. Feature sets: "vv" (6 texture features of VV) for every method, and
   "vvvh" (6 per polarization) for methods that output both polarizations.
2. Fields of The World field boundaries on 20 FTW test-split chips (France, Spain, Netherlands,
   South Africa, Vietnam; labels from FTW_SRC, read-only). For three dates per chip, every method's
   output feeds
   a) a random forest over 3 classes (background, field, boundary; FTW's unlabelled class ignored),
      trained on two chips per country and tested on the other two, then swapped (2 folds);
      we report boundary-class F1, field F1 and macro F1;
   b) Canny edges (sigma 1.5) with the same edge density as FTW's boundary class inside the
      labelled area, scored against it with a 1 px tolerance (F1).
   The temporal mean here is the mean of every other date of the label year on the same orbit.
Writes downstream_r2_wc.csv, downstream_r2_ftw.csv.
"""
from __future__ import annotations

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import canny
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import eval4
import filters as flt
from downstream import NAMES, feats, split_masks
from train_r2 import apply_model, load_model

R2 = ["r2_sup_cond_vv", "r2_n2n_big_vv", "r2_n2n_big_cond_vv", "r2_sup_cond_vvvh", "r2_n2n_big_cond_vvvh"]


def rf():
    return RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=12, random_state=0)


# ------------------------------------------------------------------ WorldCover
def worldcover():
    test = [n for n, v in SITES.items() if v[3] == "test"]
    files = {n: sorted((DATA / "out_r2").glob(f"{n}_*.npz")) for n in test}
    keys = np.load(files[test[0]][0]).files
    methods = sorted({k.rsplit("_", 1)[0] for k in keys if k.endswith("_vv") and not k.startswith("stable")})
    rng = np.random.default_rng(0)
    rows = []
    for di in range(3):
        data = {n: (np.load(files[n][di]), np.load(DATA / f"{n}.npz")["wc"]) for n in test}
        allwc = np.concatenate([d[1].ravel() for d in data.values()])
        vals, cnt = np.unique(allwc[allwc > 0], return_counts=True)
        classes = vals[cnt / cnt.sum() >= 0.005]
        idx = {}
        for n, (z, wc) in data.items():
            tr, te = split_masks(wc.shape[0])
            ok = np.isin(wc, classes)
            itr, ite = np.flatnonzero((tr & ok).ravel()), np.flatnonzero((te & ok).ravel())
            idx[n] = (rng.choice(itr, min(8000, len(itr)), replace=False), rng.choice(ite, min(20000, len(ite)), replace=False))
        for m in methods:
            for fs in ("vv", "vvvh"):
                if fs == "vvvh" and f"{m}_vh" not in keys:
                    continue
                Xtr, ytr, Xte, yte = [], [], [], []
                for n, (z, wc) in data.items():
                    F = feats(z[f"{m}_vv"]).reshape(-1, 6)
                    if fs == "vvvh":
                        F = np.concatenate([F, feats(z[f"{m}_vh"]).reshape(-1, 6)], 1)
                    a, b = idx[n]
                    Xtr.append(F[a]); ytr.append(wc.ravel()[a]); Xte.append(F[b]); yte.append(wc.ravel()[b])
                c = rf().fit(np.concatenate(Xtr), np.concatenate(ytr))
                yt, pr = np.concatenate(yte), c.predict(np.concatenate(Xte))
                rows.append(dict(date_idx=di, method=m, features=fs, oa=accuracy_score(yt, pr),
                                 macro_f1=f1_score(yt, pr, average="macro")))
                print("wc", di, m, fs, round(rows[-1]["oa"], 3), round(rows[-1]["macro_f1"], 3), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "downstream_r2_wc.csv", index=False)


# ------------------------------------------------------------------ FTW
def ftw_outputs(models, L1):
    """{chip: [ (date index, {method_pol: image}) x 3 ], mask}"""
    res = {}
    for f in sorted((DATA / "ftw").glob("*.npz")):
        z = np.load(f)
        vv, vh, mask = z["vv"], z["vh"], z["mask"]
        T = vv.shape[0]
        per = []
        for t in np.round(np.linspace(0, T - 1, 5)[1:4]).astype(int):
            o = {}
            for p, stack in (("vv", vv), ("vh", vh)):
                I = stack[t].astype("float64")
                valid = np.isfinite(I)
                o[f"raw_{p}"] = np.where(valid, I, np.nanmedian(I))
                o[f"lee_{p}"] = flt.lee(I, L1, 7)
                o[f"refined_lee_{p}"] = flt.refined_lee(I, L1)
                o[f"temporal_mean_{p}"] = (np.nansum(stack, 0) - np.where(valid, I, 0)) / np.maximum(
                    np.isfinite(stack).sum(0) - valid, 1)
            for m in ("cnn_sup", "cnn_n2n"):
                o[f"{m}_vv"] = eval4.apply(m, vv[t].astype("float64"), L1)
            for n, (G, meta) in models.items():
                out = apply_model(G, meta, {"vv": vv[t], "vh": vh[t]})
                for p in meta["pols"]:
                    o[f"{n}_{p}"] = out[p]
            per.append((int(t), o))
        res[f.stem] = (per, mask, str(z["country"]))
        print("ftw outputs", f.stem, flush=True)
    return res


def edge_f1(lin, B, lab):
    db = 10 * np.log10(np.maximum(lin, 1e-6))
    dens = B[lab].mean()
    hi = min(max(1 - dens, 0.05), 0.995)
    E = canny(db, sigma=1.5, low_threshold=max(1 - 2.5 * dens, 0.5 * hi), high_threshold=hi, use_quantiles=True) & lab
    k = np.ones((3, 3), bool)
    p = (E & ndi.binary_dilation(B, k)).sum() / max(E.sum(), 1)
    r = (B & ndi.binary_dilation(E, k)).sum() / max(B.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9)


def ftw(models, L1):
    res = ftw_outputs(models, L1)
    chips = sorted(res)
    keys = res[chips[0]][0][0][1].keys()
    methods = sorted({k.rsplit("_", 1)[0] for k in keys if k.endswith("_vv")})
    by_c = {}
    for c in chips:
        by_c.setdefault(res[c][2], []).append(c)
    # two chips per country in each fold (a lazy generator here once captured the loop variable and
    # put the same chips in both folds, which gave a leaked boundary F1 of 0.99; keep this explicit)
    folds = []
    for f in (0, 1):
        folds.append([cs[i] for cs in by_c.values() for i in range(len(cs)) if i % 2 == f])
    assert not set(folds[0]) & set(folds[1])
    rows = []
    for di in range(3):
        for m in methods:
            for fs in ("vv", "vvvh"):
                if fs == "vvvh" and f"{m}_vh" not in keys:
                    continue
                X, Y = {}, {}
                for c in chips:
                    o = res[c][0][di][1]
                    F = feats(o[f"{m}_vv"]).reshape(-1, 6)
                    if fs == "vvvh":
                        F = np.concatenate([F, feats(o[f"{m}_vh"]).reshape(-1, 6)], 1)
                    y = res[c][1].ravel()
                    keep = y != 3
                    X[c], Y[c] = F[keep], y[keep]
                yt_all, pr_all = [], []
                for f in (0, 1):
                    tr, te = folds[f], folds[1 - f]
                    cl = rf().fit(np.concatenate([X[c] for c in tr]), np.concatenate([Y[c] for c in tr]))
                    yt_all.append(np.concatenate([Y[c] for c in te]))
                    pr_all.append(cl.predict(np.concatenate([X[c] for c in te])))
                yt, pr = np.concatenate(yt_all), np.concatenate(pr_all)
                f1s = f1_score(yt, pr, average=None, labels=[0, 1, 2])
                ef = [edge_f1(res[c][0][di][1][f"{m}_vv"], res[c][1] == 2, res[c][1] != 3) for c in chips] if fs == "vv" else [np.nan]
                rows.append(dict(date_idx=di, method=m, features=fs, oa=accuracy_score(yt, pr), macro_f1=f1s.mean(),
                                 f1_background=f1s[0], f1_field=f1s[1], f1_boundary=f1s[2], edge_f1=float(np.mean(ef))))
                print("ftw", di, m, fs, round(f1s[2], 3), round(f1s.mean(), 3), round(rows[-1]["edge_f1"], 3), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "downstream_r2_ftw.csv", index=False)


def main():
    import sys
    what = sys.argv[1:] or ["wc", "ftw"]
    L1 = eval4.global_L()
    models = {n: load_model(n) for n in R2 if (ROOT / "ckpt" / f"{n}.json").exists()}
    if "wc" in what:
        worldcover()
    if "ftw" in what:
        ftw(models, L1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
