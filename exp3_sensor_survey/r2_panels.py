"""Round 2 side-by-side panels (held-out tiles, one shared gray stretch per row).
  multi: input 20 MHz | bicubic | CNN single-date target | CNN multi-date target | truth single-date | truth multi-date
  g2:    input 10 m GCOV | bicubic | CNN | 5 m GSLC multi-date target
  armb:  X-band | 40 MHz truth | arm A' | arm B (best edge weight), on xtest windows, X-band edges in orange
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from config import DATA
from evaluate import bicubic, edges, predict
from train import DEV, Gen, to_n

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval2" / "panels"; OUT.mkdir(parents=True, exist_ok=True)


def G(name):
    g = Gen(2).to(DEV).eval()
    g.load_state_dict(torch.load(ROOT / f"ckpt/{name}.pt", weights_only=True, map_location=DEV))
    return g


def crop_best(ref, n):
    best, rc = -1, (0, 0)
    for r in range(0, ref.shape[0] - n + 1, n // 4):
        for c in range(0, ref.shape[1] - n + 1, n // 4):
            w = ref[r:r + n, c:c + n]
            if np.isfinite(w).mean() > 0.99 and np.nanstd(w) > best:
                best, rc = np.nanstd(w), (r, c)
    return rc


def row(ax, imgs, n, rc, ref_key, overlay=None):
    r, c = rc
    lo, hi = np.nanpercentile(imgs[ref_key][r:r + n, c:c + n], [2, 98])
    for a_, (k, v) in zip(ax, imgs.items()):
        a_.imshow(v[r:r + n, c:c + n], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if overlay is not None:
            yy, xx = np.nonzero(overlay[r:r + n, c:c + n])
            a_.scatter(xx, yy, s=0.4, c="#e8a33d", linewidths=0)
        a_.set_title(k, fontsize=8); a_.axis("off")


def fill(a):
    return np.where(np.isfinite(a), a, np.nanmedian(a))


def multi_panels(k=3):
    gs, gm = G(sys.argv[2] if len(sys.argv) > 2 else "s_w05_40"), G("m_multi")
    files = []
    for role in ["far", "near"]:
        for f in sorted(glob.glob(str(DATA / "tiles" / role / "*.npz"))):
            e = DATA / "extra" / "mt10" / Path(f).name
            if e.exists() and np.load(e)["m"].mean() > 0.95:
                files.append((role, f, e))
    rng = np.random.default_rng(1)
    pick = [files[i] for i in rng.choice(len(files), size=min(k, len(files)), replace=False)]
    for role, f, e in pick:
        z, m = np.load(f), np.load(e)
        x = fill(z["m20_HH"].astype("float32")); xn = to_n(x)
        imgs = {"Input 20 MHz (20 m)": np.kron(x, np.ones((2, 2))), "Bicubic": bicubic(xn, 2),
                "CNN, single-date target": predict(gs, xn), "CNN, multi-date target": predict(gm, xn),
                "Truth, single date": z["a10_HH"].astype("float32"), "Truth, multi-date mean": m["y"].astype("float32")}
        rc = crop_best(imgs["Truth, multi-date mean"], 192)
        fig, ax = plt.subplots(1, 6, figsize=(17, 3.3), dpi=95)
        row(ax, imgs, 192, rc, "Truth, multi-date mean")
        fig.suptitle(f"held-out {role} {Path(f).stem} (1.92 km)", fontsize=9); fig.tight_layout()
        fig.savefig(OUT / f"multi_{Path(f).stem}.png"); plt.close(fig)


def g2_panels(k=3):
    gg = G("g_g2")
    files = []
    for role in ["far", "near"]:
        for e in sorted(glob.glob(str(DATA / "extra" / "g5" / "*.npz"))):
            f = DATA / "tiles" / role / Path(e).name
            if f.exists() and np.load(e)["m"].mean() > 0.95:
                files.append((role, f, e))
    rng = np.random.default_rng(2)
    pick = [files[i] for i in rng.choice(len(files), size=min(k, len(files)), replace=False)]
    for role, f, e in pick:
        z, g = np.load(f), np.load(e)
        x = fill(z["a10_HH"].astype("float32")); xn = to_n(x)
        imgs = {"Input GCOV 10 m (single date)": np.kron(x, np.ones((2, 2))), "Bicubic": bicubic(xn, 2), "CNN": predict(gg, xn),
                "Target: GSLC 5 m, multi-date": g["y"].astype("float32")}
        rc = crop_best(imgs["Target: GSLC 5 m, multi-date"], 256)
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), dpi=95)
        row(ax, imgs, 256, rc, "Target: GSLC 5 m, multi-date")
        fig.suptitle(f"held-out {role} {Path(f).stem} (1.28 km at 5 m)", fontsize=9); fig.tight_layout()
        fig.savefig(OUT / f"g2_{Path(f).stem}.png"); plt.close(fig)


def armb_panels(best, k=3):
    ga, gb = G("b_a"), G(best)
    files = sorted(glob.glob(str(DATA / "xtrain" / "xtest" / "*.npz")))[:k]
    for f in files:
        z = np.load(f)
        x = fill(z["x"].astype("float32")); xn = to_n(x)
        ex = z["xe"] & z["infra"]
        imgs = {"X-band (registered, 10 m)": z["xband"].astype("float32"), "Truth 40 MHz": z["y"].astype("float32"),
                "Arm A' (no edge loss)": predict(ga, xn), f"Arm B ({best})": predict(gb, xn)}
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), dpi=95)
        for a_, (kk, v) in zip(ax, imgs.items()):
            lo, hi = np.nanpercentile(v, [2, 98])
            a_.imshow(v, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            yy, xx = np.nonzero(ex); a_.scatter(xx, yy, s=0.3, c="#e8a33d", linewidths=0)
            a_.set_title(kk, fontsize=8); a_.axis("off")
        fig.suptitle(f"xtest {Path(f).stem} (2.56 km; orange = X-band edges on infrastructure)", fontsize=9); fig.tight_layout()
        fig.savefig(OUT / f"armb_{Path(f).stem}.png"); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "multi":
        multi_panels()
    elif what == "g2":
        g2_panels()
    elif what == "armb":
        armb_panels(sys.argv[2])
    print("panels", what, flush=True)
    sys.stdout.flush(); os._exit(0)
