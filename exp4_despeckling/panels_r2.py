"""Round 2 panels: per held-out site, a 2.56 km crop (most stable structure), VV row and VH row."""
from __future__ import annotations

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from panels import db, pick

FIG = ROOT / "figs"
COLS = [("raw", "Raw"), ("lee", "Lee"), ("refined_lee", "Refined Lee"), ("cnn_sup", "R1 CNN sup."),
        ("r2_sup_cond_vv", "R2 sup. + cond."), ("r2_n2n_big_cond_vv", "R2 N2N 24 sites + cond."),
        ("r2_sup_cond_vvvh", "R2 sup. + cond., VV+VH"), ("r2_n2n_big_cond_vvvh", "R2 N2N + cond., VV+VH"),
        ("ref", "Temporal mean")]


def main():
    for name, v in SITES.items():
        if v[3] != "test":
            continue
        z = np.load(sorted((DATA / "out_r2").glob(f"{name}_*.npz"))[1])
        cols = [(k, lab) for k, lab in COLS if f"{k}_vv" in z.files]
        n = 256
        r0, c0 = pick(z["ref_vv"], z["stable_vv"].astype("float64"), n)
        fig, ax = plt.subplots(2, len(cols), figsize=(1.9 * len(cols), 4.1), dpi=110)
        for i, p in enumerate(("vv", "vh")):
            lo, hi = np.percentile(db(z[f"ref_{p}"]), [2, 98])
            for j, (k, lab) in enumerate(cols):
                a = ax[i, j]
                a.set_xticks([]); a.set_yticks([])
                if f"{k}_{p}" in z.files:
                    a.imshow(db(z[f"{k}_{p}"])[r0:r0 + n, c0:c0 + n], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                else:
                    a.axis("off")
                if i == 0:
                    a.set_title(lab, fontsize=7.5)
            ax[i, 0].set_ylabel(p.upper(), fontsize=8)
        fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.05)
        out = FIG / f"r2_panel_{name}.png"
        fig.savefig(out)
        plt.close(fig)
        Image.open(out).convert("RGB").quantize(64).save(out, optimize=True)
        print(name, out.stat().st_size // 1024, "KB", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
