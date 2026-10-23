"""Example figures for the X-band geometry check: one row per provider example.
Columns: X-band (registered, 10 m) | 40 MHz truth | CNN | bicubic, with X-band edges on built-up pixels in orange."""
import os, sys, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluate import edges
from r2_panels import mark

ROOT = Path(__file__).resolve().parent
files = {}
for f in sorted(glob.glob(str(ROOT / "eval/xband/*.npz"))):
    prov = Path(f).stem.split("_")[0]
    files.setdefault(prov, []).append(f)
out = ROOT / "eval/xband_figs"; out.mkdir(exist_ok=True)
for prov, fs in files.items():
    fs = sorted(fs, key=lambda q: ("xmask" not in np.load(q).files, "cnn" not in np.load(q).files))
    f = fs[0]
    z = np.load(f)
    xb, built = z["xband"].astype("float32"), z["built"]
    xm = z["xmask"] if "xmask" in z.files else np.isfinite(xb)
    ex = edges(np.where(xm, xb, np.nanmedian(xb))) & built
    cols = [("X-band, registered (10 m)", xb), ("Truth 40 MHz (10 m)", z["truth40"].astype("float32"))]
    if "cnn" in z:
        cols.append(("CNN arm A", z["cnn"].astype("float32")))
    cols.append(("Bicubic", z["bicubic"].astype("float32")))
    fig, ax = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.6), dpi=100)
    for a_, (t, im) in zip(ax, cols):
        lo, hi = np.nanpercentile(im, [2, 98])
        a_.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        yy, xx = np.nonzero(ex)
        a_.scatter(xx, yy, s=0.3, c="#e8a33d", linewidths=0)
        a_.set_title(t, fontsize=9); a_.axis("off"); mark(a_, t)
    fig.suptitle(f"{prov}: {Path(f).stem}  (2.56 km; orange = X-band edges on built-up pixels)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"{prov}.png"); plt.close(fig)
    print("fig", prov, flush=True)
sys.stdout.flush(); os._exit(0)
