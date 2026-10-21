"""Round 2 evaluation.

A. NISAR held-out frames (exp 3's near and far holdouts): bicubic, zero-shot X-band models, every fine-tune
   run in ckpt/r2 (learning curve), the joint shared-trunk model and its no-embedding ablation.
   Metrics at 10 m: MAE, SSIM, detail_r, hf_ratio (standardized dB), edge chamfer and edge F (15 m Canny scale).
B. X-band held-out tiles at 5 m (synthetic input, and real Sentinel-1 input against real X-band), for the
   per-band X models against the joint model.
C. Cross index on real Sentinel-1: at the brightest local peaks of each 5 m output, mean standardized dB along
   the four axis-aligned arms minus along the four diagonals, 10 to 30 m out. Sidelobe crosses push it up.
Writes r2_nisar_scores.csv, r2_x_scores.csv, r2_cross.csv.
"""
from __future__ import annotations

import common  # noqa: F401
from common import (DEV, ROOT, SPIKE, bicubic, chamfer, degrade, edges, finish, fine_register, load_model, meta_vec,
                    nisar_tiles, norm, run_model, scores, tile_metadata)

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from e5r2_joint import code
from e5_eval_x import score_all


def run_c(G, s, c):
    """CondGen with an explicit conditioning code."""
    return run_model(G, s, c)


def models_r2():
    M = {}
    for f in sorted((ROOT / "ckpt" / "r2").glob("*.pt")):
        G, ck = load_model(f)
        M[f.stem] = (G, ck)
    return M


def cross_index(o, n_peaks=40, rmin=2, rmax=6):
    a = ndimage.gaussian_filter(o, 0.5)
    mx = ndimage.maximum_filter(a, 9)
    pk = np.argwhere((a == mx) & (a > np.percentile(a, 99.9)))
    pk = pk[(pk[:, 0] > rmax) & (pk[:, 0] < o.shape[0] - rmax - 1) & (pk[:, 1] > rmax) & (pk[:, 1] < o.shape[1] - rmax - 1)]
    if len(pk) == 0:
        return np.nan
    pk = pk[np.argsort(-a[pk[:, 0], pk[:, 1]])][:n_peaks]
    d = np.arange(rmin, rmax + 1)
    out = []
    for r, c in pk:
        ax = np.r_[o[r + d, c], o[r - d, c], o[r, c + d], o[r, c - d]].mean()
        dg = np.r_[o[r + d, c + d], o[r - d, c - d], o[r + d, c - d], o[r - d, c + d]].mean()
        out.append(ax - dg)
    return float(np.mean(out))


def main():
    M = models_r2()
    zs = {k: load_model(ROOT / "ckpt" / f"{k}.pt")[0] for k in ["x2_unc", "x2_cond", "x2_aug"] if (ROOT / "ckpt" / f"{k}.pt").exists()}
    print("r2 models:", list(M), "| zero-shot:", list(zs), flush=True)
    cn = code("nisar20", "nisar40")
    # ---------------- A. NISAR holdouts
    rows = []
    for role in ["near", "far"]:
        for t in nisar_tiles(role):
            x, y, m = t["x"], t["y"], t["m"]
            P = {"bicubic": bicubic(x, 2)}
            for k, G in zs.items():
                P[f"zeroshot_{k}"] = run_model(G, x, meta_vec(258, 37, "HH", "X"))
            for k, (G, ck) in M.items():
                if k.startswith("joint"):
                    P[k] = run_c(G, x, np.zeros(6, "float32") if ck.get("noembed") else cn)
                else:
                    P[k] = run_model(G, x)
            te = edges(y, m, 10.0, sigma_m=15.0)
            for k, p in P.items():
                r = dict(role=role, tile=t["id"], frame=f"{t['frame'][0]}_{t['frame'][1]}", model=k, **scores(p, y, m, 2))
                r.update(chamfer(edges(p, m, 10.0, sigma_m=15.0), te, m, 10.0, tols=(10.0, 20.0)))
                rows.append(r)
        print("nisar", role, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "r2_nisar_scores.csv", index=False)

    # ---------------- B. X-band held-out, 5 m, and C. cross index on real S1
    meta = tile_metadata()
    man = pd.read_csv(SPIKE / "manifest.csv")
    man = man[(man.split == "test") & man.status.isin(["ok", "cached"])]
    joint = {k: v for k, v in M.items() if k.startswith("joint")}
    gen = torch.Generator(device=DEV)
    xrows, crows = [], []
    for r in man.itertuples():
        fid = f"{r.provider}_{r.xband_id}"
        z = np.load(SPIKE / "data" / f"{fid}.npz")
        x = z["x"].astype("float32")
        m = np.isfinite(x)
        if m.mean() < 0.8:
            continue
        md = meta.get(fid, dict(inc=35.0, az=270.0, pol="VV"))
        truth = np.nan_to_num(norm(x))
        gen.manual_seed(zlib.crc32(fid.encode()))
        xf = np.where(m, x, np.nanmedian(x))
        with torch.no_grad():
            sdb = degrade(torch.from_numpy(xf)[None, None].to(DEV), [md["az"]], looks=6, gen=gen)[0, 0].cpu().numpy()
        s1 = z["s1"].astype("float32")
        s1 = np.where(np.isfinite(s1), s1, np.nanmedian(s1))
        base = dict(tile=fid, provider=r.provider, xpol=md["pol"], split=r.split, region=r.test_region)
        for inp, s in [("synthetic", norm(sdb)), ("real_s1", norm(s1))]:
            P = {"bicubic": (bicubic(s, 2), 2)}
            mv = meta_vec(md["az"] if inp == "synthetic" else 258, md["inc"] if inp == "synthetic" else 37, md["pol"], "X")
            for k, G in zs.items():
                P[k] = (run_model(G, s, mv), 2)
            for k, (G, ck) in joint.items():
                if ck.get("noembed"):
                    P[k] = (run_c(G, s, np.zeros(6, "float32")), 2)
                else:
                    P[k + " (target: X spotlight)"] = (run_c(G, s, code("xsyn", "xspot")), 2)
                    P[k + " (target: NISAR style)"] = (run_c(G, s, code("xsyn", "nisar40")), 2)
            tt, mm = truth, m
            if inp == "real_s1":
                sh = fine_register(bicubic(s, 4), truth, m)
                tt = ndimage.shift(truth, sh, order=1, mode="nearest")
                mm = ndimage.shift(m.astype("float32"), sh, order=0, mode="constant") > 0.5
                for k, (p, _) in P.items():
                    crows.append(dict(base, model=k, cross=cross_index(p)))
            xrows += score_all(P, tt, mm, dict(base, input=inp))
        print("x", fid, flush=True)
    pd.DataFrame(xrows).to_csv(ROOT / "r2_x_scores.csv", index=False)
    pd.DataFrame(crows).to_csv(ROOT / "r2_cross.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
    finish()
