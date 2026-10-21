"""Round 3 panels (held-out tiles, shared gray stretch).
  loss  input | bicubic | CNN plain L1 | CNN L1 + gradient L1 | truth single date | truth multi-date
  g2v2  input 10 m | bicubic | CNN round 2 target | CNN deeper target | deeper 5 m GSLC target
"""
import glob, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import DATA
from evaluate import bicubic, predict
from train import to_n
from r2_panels import G, crop_best, fill, row

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval2" / "panels_r3"; OUT.mkdir(parents=True, exist_ok=True)


def pick(sub, key, k, seed):
    files = []
    for role in ["far", "near"]:
        for f in sorted(glob.glob(str(DATA / "tiles" / role / "*.npz"))):
            e = DATA / "extra" / sub / Path(f).name
            if e.exists() and np.load(e)["m"].mean() > 0.95:
                files.append((role, f, e))
    rng = np.random.default_rng(seed)
    return [files[i] for i in rng.choice(len(files), size=min(k, len(files)), replace=False)]


def loss_panels(win):
    g1, g2 = G("m_multi"), G(win)
    for role, f, e in pick("mt10", "y", 3, 1):
        z, m = np.load(f), np.load(e)
        x = fill(z["m20_HH"].astype("float32")); xn = to_n(x)
        imgs = {"Input 20 MHz (20 m)": np.kron(x, np.ones((2, 2))), "Bicubic": bicubic(xn, 2), "CNN, plain L1": predict(g1, xn),
                "CNN, L1 + gradient L1": predict(g2, xn), "Truth, single date": z["a10_HH"].astype("float32"),
                "Truth, multi-date mean": m["y"].astype("float32")}
        rc = crop_best(imgs["Truth, multi-date mean"], 192)
        fig, ax = plt.subplots(1, 6, figsize=(17, 3.3), dpi=95)
        row(ax, imgs, 192, rc, "Truth, multi-date mean")
        fig.suptitle(f"held-out {role} {Path(f).stem} (1.92 km)", fontsize=9); fig.tight_layout()
        fig.savefig(OUT / f"loss_{Path(f).stem}.png"); plt.close(fig)


def g2v2_panels():
    ga, gb = G("g_g2"), G("r3_g2v2")
    for role, f, e in pick("g5v2", "y", 3, 2):
        z, g = np.load(f), np.load(e)
        x = fill(z["a10_HH"].astype("float32")); xn = to_n(x)
        imgs = {"Input GCOV 10 m": np.kron(x, np.ones((2, 2))), "Bicubic": bicubic(xn, 2), "CNN, round 2 target": predict(ga, xn),
                "CNN, deeper target": predict(gb, xn), "Target: GSLC 5 m, deeper": g["y"].astype("float32")}
        rc = crop_best(imgs["Target: GSLC 5 m, deeper"], 256)
        fig, ax = plt.subplots(1, 5, figsize=(15, 3.5), dpi=95)
        row(ax, imgs, 256, rc, "Target: GSLC 5 m, deeper")
        fig.suptitle(f"held-out {role} {Path(f).stem} (1.28 km at 5 m)", fontsize=9); fig.tight_layout()
        fig.savefig(OUT / f"g2v2_{Path(f).stem}.png"); plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "loss":
        loss_panels(sys.argv[2])
    else:
        g2v2_panels()
    print("panels", sys.argv[1], flush=True)
    sys.stdout.flush(); os._exit(0)
