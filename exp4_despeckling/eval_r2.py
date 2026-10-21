"""Round 2 scoring on the same six held-out sites.

  VV, k = 1   every date: classic filters, round-1 CNNs, round-2 CNNs (VV channel of dual-pol models)
  VV, k = 2, 4  input = mean of k consecutive dates (every 3rd start date); reference excludes them.
              Tests whether a method over-smooths or biases an input that is already cleaner.
              Classic filters get L = k x 6.7.
  VH, k = 1   every date: classic filters and the dual-pol round-2 CNNs, against the VH temporal mean
              and VH change mask.
Metrics: eval4.metrics. Output: metrics_r2.csv, and for three dates per site (k = 1)
E:/SAR_superresolution/exp4_despeckling/data/out_r2/<site>_<t>.npz with "<method>_<pol>" outputs and ref_/stable_ per pol.
"""
from __future__ import annotations

import sys

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish

import numpy as np
import pandas as pd

import eval4
import filters as flt
from train_r2 import apply_model, load_model

OUT = DATA / "out_r2"
OUT.mkdir(exist_ok=True)
R2 = ["r2_sup_cond_vv", "r2_n2n_big_vv", "r2_n2n_big_cond_vv", "r2_sup_cond_vvvh", "r2_n2n_big_cond_vvvh"]
CLASSIC = ["raw", "lee", "refined_lee", "gamma_map"]


def classic(m, I, L):
    return {"raw": lambda: I.astype("float64"), "lee": lambda: flt.lee(I, L, 7),
            "refined_lee": lambda: flt.refined_lee(I, L), "gamma_map": lambda: flt.gamma_map(I, L, 7)}[m]()


def main():
    L1 = eval4.global_L()
    args = sys.argv[1:]
    base = not args or "base" in args            # classic filters + round-1 CNNs
    want = [a for a in args if a != "base"] or R2
    models = {n: load_model(n) for n in want if (ROOT / "ckpt" / f"{n}.json").exists()}
    print("round-2 models", list(models), flush=True)
    rows = []
    for name, v in SITES.items():
        if v[3] != "test":
            continue
        z = np.load(DATA / f"{name}.npz")
        img = {"vv": z["vv"], "vh": np.load(DATA / f"{name}_vh.npz")["vh"]}
        st = {"vv": np.load(DATA / f"{name}_stack.npz"), "vh": np.load(DATA / f"{name}_vh_stack.npz")}
        T = img["vv"].shape[0]
        keep_t = set(np.round(np.linspace(0, T - 1, 5)[1:4]).astype(int).tolist())
        for k in (1, 2, 4):
            starts = range(T) if k == 1 else range(0, T - k + 1, 3)
            for t in starts:
                sel = list(range(t, t + k))
                inp, ref, M = {}, {}, {}
                for p in ("vv", "vh"):
                    stack = img[p][sel]
                    I = np.nanmean(stack, 0) if k > 1 else stack[0]
                    valid = np.isfinite(I)
                    ref[p] = (st[p]["sum"] - np.nansum(stack, 0)) / np.maximum(st[p]["n"] - k, 1)
                    M[p] = st[p]["stable"] & valid
                    inp[p] = I.astype("float64")
                fill = {p: np.where(np.isfinite(inp[p]), inp[p], np.nanmedian(inp[p])) for p in inp}
                outs = {}
                for m in (CLASSIC if base else []):
                    for p in ("vv", "vh"):
                        if k > 1 and (p == "vh" or m in ("gamma_map",)):
                            continue
                        outs[(m, p)] = classic(m, inp[p], L1 * k)
                for m in (("cnn_sup", "cnn_n2n") if base else ()):
                    if k == 1 or m == "cnn_sup":
                        outs[(m, "vv")] = eval4.apply(m, inp["vv"], L1)
                for n, (G, meta) in models.items():
                    o = apply_model(G, meta, {p: inp[p] for p in meta["pols"]})
                    for p in meta["pols"]:
                        if k == 1 or p == "vv":
                            outs[(n, p)] = o[p]
                for (m, p), o in outs.items():
                    rows.append(dict(site=name, landcover=v[2], t=t, k=k, pol=p, method=m,
                                     **eval4.metrics(o, fill[p], ref[p], M[p], st[p]["homog"])))
                if k == 1 and t in keep_t:
                    save = {f"{m}_{p}": o.astype("float32") for (m, p), o in outs.items()}
                    for p in ("vv", "vh"):
                        save[f"ref_{p}"] = ref[p].astype("float32")
                        save[f"stable_{p}"] = M[p].astype("float32")
                    f = OUT / f"{name}_{t:02d}.npz"
                    old = dict(np.load(f)) if f.exists() else {}
                    old.update(save)
                    np.savez(f, **old)
            print("done", name, "k", k, flush=True)
    new = pd.DataFrame(rows)
    f = ROOT / "metrics_r2.csv"
    if f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old[~old.method.isin(new.method.unique())], new])
    new.to_csv(f, index=False)
    print("wrote", len(rows), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
