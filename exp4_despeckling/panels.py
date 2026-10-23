"""Side-by-side panels per held-out site: a 2.56 km crop at native 10 m, plus a 4x zoom row.
All panels share the dB stretch of the temporal mean (p2..p98). Writes figs/panel_<site>.png."""
from __future__ import annotations

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage as ndi

import sys  # noqa: E402
sys.path.insert(0, str(ROOT.parent))
from panel_roles import legend, outline  # noqa: E402

FIG = ROOT / "figs"
FIG.mkdir(exist_ok=True)
MODELS = {"exp1_l1", "exp1_gan", "cnn_sup", "cnn_n2n", "cnn_gan"}   # learned despecklers: green outline
TRUTH = {"ref_loo"}                                                   # scoring reference: purple outline
ORDER = [("raw", "Raw single date"), ("lee", "Lee 7x7"), ("refined_lee", "Refined Lee"), ("gamma_map", "Gamma-MAP"),
         ("exp1_l1", "Exp 1 CNN, Capella (to 10 m)"), ("exp1_gan", "Exp 1 CNN+GAN, Capella"), ("cnn_sup", "New CNN (supervised)"),
         ("cnn_n2n", "New CNN (Noise2Noise)"), ("cnn_gan", "New CNN + GAN"), ("ref_loo", "Temporal mean (reference)")]


def db(a):
    return 10 * np.log10(np.maximum(a, 1e-6))


def pick(ref, stable, n, r_lim=None):
    """Top-left corner of the n px window with the most reference structure on stable pixels
    (windows less than 80 % stable are skipped, so the panel does not show a changed field)."""
    g = np.hypot(ndi.sobel(db(ref), 0), ndi.sobel(db(ref), 1)) * stable
    sc = ndi.uniform_filter(g, n) * (ndi.uniform_filter(stable, n) >= 0.8)
    H, W = ref.shape
    c = n // 2
    sub = sc[c:H - n + c + 1, c:W - n + c + 1]
    r0, c0 = np.unravel_index(np.argmax(sub), sub.shape)
    return int(r0), int(c0)


def main():
    for name, v in SITES.items():
        if v[3] != "test":
            continue
        files = sorted((DATA / "out").glob(f"{name}_*.npz"))
        z = np.load(files[1])
        cols = [(k, lab) for k, lab in ORDER if k in z.files]
        ref = z["ref_loo"]
        lo, hi = np.percentile(db(ref), [2, 98])
        st = z["stable"].astype("float64")
        n, zn = 256, 64
        r0, c0 = pick(ref, st, n)
        zr, zc = pick(ref[r0:r0 + n, c0:c0 + n], st[r0:r0 + n, c0:c0 + n], zn)
        k = len(cols)
        fig, ax = plt.subplots(2, k, figsize=(1.9 * k, 4.35), dpi=110)
        for j, (key, lab) in enumerate(cols):
            a = db(z[key])[r0:r0 + n, c0:c0 + n]
            ax[0, j].imshow(a, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax[0, j].add_patch(Rectangle((zc, zr), zn, zn, fill=False, ec="#ff8a4c", lw=0.8))
            ax[0, j].set_title(lab, fontsize=7.5)
            ax[1, j].imshow(a[zr:zr + zn, zc:zc + zn], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            for a_ in ax[:, j]:
                a_.set_xticks([]); a_.set_yticks([])
                if key in MODELS or key in TRUTH:
                    outline(a_, "model" if key in MODELS else "truth", lw=2.5)
        ax[1, 0].set_ylabel("zoom 640 m", fontsize=7.5)
        ax[0, 0].set_ylabel("2.56 km", fontsize=7.5)
        fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.06, wspace=0.04, hspace=0.06)
        legend(fig)
        p = FIG / f"panel_{name}.png"
        fig.savefig(p)
        plt.close(fig)
        im = Image.open(p).convert("RGB").quantize(96)
        im.save(p, optimize=True)
        print(name, files[1].name, p.stat().st_size // 1024, "KB", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
