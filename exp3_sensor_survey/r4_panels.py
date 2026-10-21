"""Round 4 panels: does the adversarial term add real structure or hallucinated texture?
  tiles   held-out tiles: input | bicubic | control (adv 0) | GAN winner | truth single date | truth multi-date
  xband   X-band sites (xtest windows and held-out sites): X-band | 40 MHz truth | control | GAN, X-band edges in orange
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
from r3_panels import pick

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval2" / "panels_r4"; OUT.mkdir(parents=True, exist_ok=True)


def tiles(ctrl, gan):
    gc, gg = G(ctrl), G(gan)
    for role, f, e in pick("mt10", "y", 4, 4):
        z, m = np.load(f), np.load(e)
        x = fill(z["m20_HH"].astype("float32")); xn = to_n(x)
        imgs = {"Input 20 MHz (20 m)": np.kron(x, np.ones((2, 2))), "Bicubic": bicubic(xn, 2), "Control (no GAN)": predict(gc, xn),
                f"GAN ({gan})": predict(gg, xn), "Truth, single date": z["a10_HH"].astype("float32"),
                "Truth, multi-date mean": m["y"].astype("float32")}
        rc = crop_best(imgs["Truth, multi-date mean"], 160)
        fig, ax = plt.subplots(1, 6, figsize=(17, 3.3), dpi=95)
        row(ax, imgs, 160, rc, "Truth, multi-date mean")
        fig.suptitle(f"held-out {role} {Path(f).stem} (1.6 km)", fontsize=9); fig.tight_layout()
        fig.savefig(OUT / f"tile_{Path(f).stem}.png"); plt.close(fig)


def xband(ctrl, gan, k=4):
    gc, gg = G(ctrl), G(gan)
    for f in sorted(glob.glob(str(DATA / "xtrain" / "xtest" / "*.npz")))[:k]:
        z = np.load(f)
        x = fill(z["x"].astype("float32")); xn = to_n(x)
        ex = z["xe"] & z["infra"]
        imgs = {"X-band (registered, 10 m)": z["xband"].astype("float32"), "Truth 40 MHz": z["y"].astype("float32"),
                "Control (no GAN)": predict(gc, xn), f"GAN ({gan})": predict(gg, xn)}
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), dpi=95)
        for a_, (kk, v) in zip(ax, imgs.items()):
            lo, hi = np.nanpercentile(v, [2, 98])
            a_.imshow(v, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            yy, xx = np.nonzero(ex); a_.scatter(xx, yy, s=0.3, c="#e8a33d", linewidths=0)
            a_.set_title(kk, fontsize=8); a_.axis("off")
        fig.suptitle(f"X-band test site {Path(f).stem} (2.56 km; orange = X-band edges on infrastructure)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"xband_{Path(f).stem}.png"); plt.close(fig)


if __name__ == "__main__":
    tiles(sys.argv[1], sys.argv[2]); xband(sys.argv[1], sys.argv[2])
    print("panels done", flush=True)
    sys.stdout.flush(); os._exit(0)
