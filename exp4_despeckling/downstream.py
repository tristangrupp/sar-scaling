"""Downstream tests on the held-out sites, same pixels and classifier for every method.

1. Land-cover classification against ESA WorldCover 2021 (10 m).
   Features from each method's output in dB: value, 3/7/15 px local means, 7 px local std, gradient
   magnitude of the 3 px mean. Spatial split: 64 px blocks in a checkerboard, train on one colour,
   test on the other, dropping 8 px along block borders so windows never straddle the split.
   Two feature sets: "pixel" (the dB value alone: how readable a single pixel is) and "texture"
   (all six). One random forest (200 trees, balanced class weights, seed 0) pooled over all six held-out
   sites, refitted for each method and each of three dates. Report overall accuracy and macro F1.
2. Boundary agreement. Boundaries of a 5x5 majority-filtered WorldCover (raw 10 m class noise
   removed; sites with < 0.5 % boundary pixels skipped) vs Canny edges of each output (sigma 1.5),
   with thresholds set by quantile so every method marks the same share of pixels as edges as
   WorldCover has boundary pixels. F1 with a 1-pixel tolerance.
Writes downstream_cls.csv, downstream_edges.csv.
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

OUT = DATA / "out"
NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water",
         90: "wetland", 95: "mangrove", 100: "moss"}


def feats(lin):
    db = 10 * np.log10(np.maximum(lin, 1e-6))
    m3, m7, m15 = (ndi.uniform_filter(db, w) for w in (3, 7, 15))
    s7 = np.sqrt(np.maximum(ndi.uniform_filter(db * db, 7) - m7 * m7, 0))
    g = np.hypot(ndi.sobel(m3, 0), ndi.sobel(m3, 1))
    return np.stack([db, m3, m7, m15, s7, g], -1).astype("float32")


def majority(wc, w=5):
    cls = np.unique(wc)
    votes = np.stack([ndi.uniform_filter((wc == c).astype("float32"), w) for c in cls])
    return cls[np.argmax(votes, 0)]


def boundaries(wc):
    m = majority(wc)
    return ndi.maximum_filter(m, 3) != ndi.minimum_filter(m, 3)


def split_masks(n, block=64, buf=8):
    r, c = np.mgrid[0:n, 0:n]
    br, bc = r // block, c // block
    inner = ((r % block) >= buf) & ((r % block) < block - buf) & ((c % block) >= buf) & ((c % block) < block - buf)
    tr = ((br + bc) % 2 == 0) & inner
    te = ((br + bc) % 2 == 1) & inner
    return tr, te


def edge_f1(lin, B):
    db = 10 * np.log10(np.maximum(lin, 1e-6))
    dens = B.mean()
    E = canny(db, sigma=1.5, low_threshold=1 - 2.5 * dens, high_threshold=1 - dens, use_quantiles=True)
    k = np.ones((3, 3), bool)
    p = (E & ndi.binary_dilation(B, k)).sum() / max(E.sum(), 1)
    r = (B & ndi.binary_dilation(E, k)).sum() / max(B.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def main():
    test = [n for n, v in SITES.items() if v[3] == "test"]
    files = {n: sorted(OUT.glob(f"{n}_*.npz")) for n in test}
    methods = [k for k in np.load(files[test[0]][0]).files if k != "stable"]
    print("methods", methods, flush=True)
    rng = np.random.default_rng(0)
    cls_rows, edge_rows = [], []
    for di in range(3):
        data = {}
        for n in test:
            z = np.load(files[n][di])
            wc = np.load(DATA / f"{n}.npz")["wc"]
            data[n] = (z, wc)
        # class set: classes holding >= 0.5 % of the pooled held-out pixels
        allwc = np.concatenate([d[1].ravel() for d in data.values()])
        vals, cnt = np.unique(allwc[allwc > 0], return_counts=True)
        classes = vals[cnt / cnt.sum() >= 0.005]
        # identical pixel samples for every method
        idx = {}
        for n, (z, wc) in data.items():
            tr, te = split_masks(wc.shape[0])
            ok = np.isin(wc, classes)
            itr, ite = np.flatnonzero((tr & ok).ravel()), np.flatnonzero((te & ok).ravel())
            idx[n] = (rng.choice(itr, min(8000, len(itr)), replace=False), rng.choice(ite, min(20000, len(ite)), replace=False))
        bounds = {n: boundaries(wc) for n, (z, wc) in data.items()}
        for m, fs in [(m, fs) for m in methods for fs in ("pixel", "texture")]:
            cols = [0] if fs == "pixel" else list(range(6))
            Xtr, ytr, Xte, yte = [], [], [], []
            for n, (z, wc) in data.items():
                F = feats(z[m]).reshape(-1, 6)[:, cols]
                y = wc.ravel()
                a, b = idx[n]
                Xtr.append(F[a]); ytr.append(y[a]); Xte.append(F[b]); yte.append(y[b])
                B = bounds[n]
                if fs == "texture" and B.mean() >= 0.005:
                    f1, p, r = edge_f1(z[m], B)
                    edge_rows.append(dict(date_idx=di, site=n, method=m, f1=f1, precision=p, recall=r,
                                          boundary_frac=B.mean()))
            rf = RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=12, random_state=0)
            rf.fit(np.concatenate(Xtr), np.concatenate(ytr))
            yt = np.concatenate(yte)
            pr = rf.predict(np.concatenate(Xte))
            row = dict(date_idx=di, method=m, features=fs, oa=accuracy_score(yt, pr), macro_f1=f1_score(yt, pr, average="macro"),
                       classes=" ".join(NAMES[c] for c in classes))
            for c, f in zip(classes, f1_score(yt, pr, average=None, labels=classes)):
                row[f"f1_{NAMES[c]}"] = f
            cls_rows.append(row)
            print(di, m, fs, round(row["oa"], 3), round(row["macro_f1"], 3), flush=True)
    pd.DataFrame(cls_rows).to_csv(ROOT / "downstream_cls.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(ROOT / "downstream_edges.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
