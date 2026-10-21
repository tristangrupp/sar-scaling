"""Score every despeckling method on every date of the held-out sites.

Reference for date t: leave-one-out temporal mean (linear power) of all other dates of the same
orbit, scored only on valid, stable pixels (stack.py change mask).

Metrics (all per site and date; see the page for definitions):
  psnr, ssim, mae      output dB vs reference dB on stable pixels; PSNR/SSIM use the reference's
                       robust dB range (p0.5..p99.5) as the data range
  bias_db              mean(output dB - reference dB) on stable pixels
  enl                  ENL of the output in homogeneous 9x9 windows (lowest-5 % CV of the temporal mean)
  mean_ratio           mean(output) / mean(input), linear power, all valid pixels (ideal 1)
  ratio_enl            ENL of the ratio image input/output over stable pixels (ideal: the speckle ENL,
                       i.e. the value for the oracle ratio input/reference)
  ratio_struct         corr(|5x5 mean of the dB ratio image|, reference gradient magnitude): structure
                       the filter removed along edges (ideal: the oracle's value, near 0)
  grad_corr            corr(Sobel magnitude of output dB, Sobel magnitude of reference dB)
  epd_h, epd_v         EPD-ROA against the reference at reference-edge pixels (top 10 % gradient):
                       sum max(o1/o2, o2/o1) / sum max(r1/r2, r2/r1) for horizontal / vertical neighbours.
                       >1 = residual noise across edges, <1 = blurred edges, ideal 1

  python eval4.py                  all methods whose inputs exist
  python eval4.py cnn_sup cnn_n2n  only these methods (rows for them are replaced)
Outputs: metrics_dates.csv (one row per site x date x method) and, for three dates per site,
E:/SAR_superresolution/exp4_despeckling/data/out/<site>_<t>.npz with every method's output (float32 linear power).
"""
from __future__ import annotations

import hashlib
import sys

import common  # noqa: F401
from common import DATA, GAN_SPIKE, ROOT, SITES, finish, norm_stats

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

import filters as flt

DEV = "cuda"
OUT = DATA / "out"
OUT.mkdir(exist_ok=True)
CLASSIC = ["raw", "box3", "box5", "lee", "refined_lee", "gamma_map"]
CNN = ["exp1_l1", "exp1_gan", "cnn_sup", "cnn_n2n", "cnn_gan"]


def to_db(a):
    return 10 * np.log10(np.maximum(a, 1e-6))


def global_L():
    s = pd.read_csv(ROOT / "stack_summary.csv")
    return float(s[s.split == "train"].L_single.median())


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def exp1_ckpt_dir():
    """Use the retrained (Capella) weights only if they differ from v1 and training has finished."""
    v1, new = GAN_SPIKE / "ckpt_v1_no_capella", GAN_SPIKE / "ckpt"
    done = (new / "gan_log.csv").exists() and md5(new / "gan.pt") != md5(v1 / "gan.pt") and \
        md5(new / "l1.pt") != md5(v1 / "l1.pt")
    return (new, "retrained with Capella") if done else (v1, "v1, no Capella")


_models = {}


def models():
    if _models:
        return _models
    sys.path.insert(0, str(GAN_SPIKE))
    from train import Gen
    from train4 import DGen
    d, tag = exp1_ckpt_dir()
    _models["_exp1_tag"] = tag
    for k in ["l1", "gan"]:
        G = Gen().to(DEV).eval()
        G.load_state_dict(torch.load(d / f"{k}.pt", weights_only=True))
        _models[f"exp1_{k}"] = G
    off = {}
    if (ROOT / "ckpt" / "n2n_offset.txt").exists():
        off["cnn_n2n"] = float((ROOT / "ckpt" / "n2n_offset.txt").read_text())
    for k in ["sup", "n2n", "gan"]:
        p = ROOT / "ckpt" / f"{k}.pt"
        if p.exists():
            G = DGen().to(DEV).eval()
            G.load_state_dict(torch.load(p, weights_only=True))
            _models[f"cnn_{k}"] = G
    _models["_n2n_offset"] = off.get("cnn_n2n", 0.0)
    return _models


@torch.no_grad()
def run_exp1(G, x, tile=128, pad=16):
    """Tiled 4x inference; x is the standardized dB image (H, W). Returns (4H, 4W)."""
    H, W = x.shape
    out = np.zeros((4 * H, 4 * W), "float32")
    xp = np.pad(x, pad, mode="reflect")
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            t = xp[r:r + tile + 2 * pad, c:c + tile + 2 * pad]
            y = G(torch.from_numpy(np.ascontiguousarray(t))[None, None].to(DEV))[0, 0].cpu().numpy()
            h, w = min(tile, H - r), min(tile, W - c)
            out[4 * r:4 * (r + h), 4 * c:4 * (c + w)] = y[4 * pad:4 * (pad + h), 4 * pad:4 * (pad + w)]
    return out


def apply(method, I, L):
    if method == "raw":
        return I.astype("float64")
    if method == "box3":
        return flt.boxcar(I, 3)
    if method == "box5":
        return flt.boxcar(I, 5)
    if method == "lee":
        return flt.lee(I, L, 7)
    if method == "refined_lee":
        return flt.refined_lee(I, L)
    if method == "gamma_map":
        return flt.gamma_map(I, L, 7)
    M = models()
    if method not in M:
        return None
    db = to_db(np.where(np.isfinite(I), I, np.nanmedian(I)))
    med, sd = norm_stats(db)
    x = ((db - med) / sd).astype("float32")
    if method.startswith("exp1"):
        y = run_exp1(M[method], x)
        lin = 10 ** ((y * sd + med) / 10)            # back to S1 dB scale with the input's own stats
        H, W = x.shape
        return lin.reshape(H, 4, W, 4).mean((1, 3))  # block-average 2.5 m -> 10 m in linear power
    with torch.no_grad():
        y = M[method](torch.from_numpy(x)[None, None].to(DEV))[0, 0].cpu().numpy()
    off = M["_n2n_offset"] if method == "cnn_n2n" else 0.0
    return 10 ** ((y * sd + med + off) / 10)


def sobel_mag(a):
    return np.hypot(ndi.sobel(a, 0), ndi.sobel(a, 1))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-20))


def enl_windows(img, sel, w=9):
    m = ndi.uniform_filter(img, w)
    v = ndi.uniform_filter(img * img, w) - m * m
    return float(np.median(m[sel] ** 2 / np.maximum(v[sel], 1e-20)))


def epd(o, r, E, axis):
    sl1 = (slice(None), slice(0, -1)) if axis == 1 else (slice(0, -1), slice(None))
    sl2 = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
    e = E[sl1]
    q = lambda a: np.maximum(a[sl1] / a[sl2], a[sl2] / a[sl1])[e]  # noqa: E731
    return float(q(o).sum() / q(r).sum())


def metrics(o, I, ref, M, homog):
    o = np.maximum(o, 1e-6)
    odb, rdb = to_db(o), to_db(ref)
    rng = float(np.subtract(*np.percentile(rdb[M], [99.5, 0.5])))
    d = (odb - rdb)[M]
    _, smap = structural_similarity(odb, rdb, data_range=rng, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    gr = sobel_mag(rdb)
    Me = ndi.binary_erosion(M, iterations=2)
    E = Me & (gr >= np.percentile(gr[Me], 90))
    ratio = I / o
    rdbr = to_db(ratio)
    valid = np.isfinite(I)
    return dict(
        psnr=10 * np.log10(rng ** 2 / np.mean(d ** 2)), ssim=float(smap[M].mean()), mae=float(np.abs(d).mean()),
        bias_db=float(d.mean()), enl=enl_windows(o, homog),
        mean_ratio=float(o[valid].mean() / I[valid].mean()),
        ratio_enl=float(ratio[M].mean() ** 2 / ratio[M].var()),
        ratio_struct=corr(np.abs(ndi.uniform_filter(np.where(valid, rdbr, 0), 5))[Me], gr[Me]),
        grad_corr=corr(sobel_mag(odb)[Me], gr[Me]),
        epd_h=epd(o, ref, E, 1), epd_v=epd(o, ref, E, 0))


def main():
    only = sys.argv[1:]
    methods = [m for m in only if m != "oracle"] or (CLASSIC + CNN)
    L = global_L()
    print("global L", L, "methods", methods, flush=True)
    rows = []
    for name, v in SITES.items():
        if v[3] != "test":
            continue
        z, st = np.load(DATA / f"{name}.npz"), np.load(DATA / f"{name}_stack.npz")
        vv, S, n, stable, homog = z["vv"], st["sum"], st["n"], st["stable"], st["homog"]
        T = vv.shape[0]
        keep_t = set(np.round(np.linspace(0, T - 1, 5)[1:4]).astype(int).tolist())
        for t in range(T):
            I = vv[t].astype("float64")
            valid = np.isfinite(I)
            ref = (S - np.where(valid, I, 0)) / np.maximum(n - valid, 1)
            M = stable & valid
            Ifill = np.where(valid, I, np.nanmedian(I))
            saved = {}
            for m in methods:
                o = apply(m, I, L)
                if o is None:
                    continue
                rows.append(dict(site=name, landcover=v[2], t=t, date=str(z["dates"][t]), method=m,
                                 **metrics(o, Ifill, ref, M, homog)))
                if t in keep_t:
                    saved[m] = o.astype("float32")
            if t in keep_t:
                saved["ref_loo"] = ref.astype("float32")
                saved["stable"] = M.astype("float32")
                f = OUT / f"{name}_{t:02d}.npz"
                old = dict(np.load(f)) if f.exists() else {}
                old.update(saved)
                np.savez(f, **old)
            # the oracle ratio (input / reference) gives the ideal values of ratio_enl and ratio_struct
            if not only or "oracle" in only:
                ratio = Ifill / np.maximum(ref, 1e-6)
                Me = ndi.binary_erosion(M, iterations=2)
                gr = sobel_mag(to_db(ref))
                rows.append(dict(site=name, landcover=v[2], t=t, date=str(z["dates"][t]), method="oracle",
                                 ratio_enl=float(ratio[M].mean() ** 2 / ratio[M].var()),
                                 ratio_struct=corr(np.abs(ndi.uniform_filter(to_db(ratio), 5))[Me], gr[Me]),
                                 enl=enl_windows(ref, homog)))
        print("done", name, T, "dates", flush=True)
    new = pd.DataFrame(rows)
    f = ROOT / "metrics_dates.csv"
    if only and f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old[~old.method.isin(new.method.unique())], new])
    new.to_csv(f, index=False)
    if "_exp1_tag" in _models:
        (ROOT / "exp1_ckpt_used.txt").write_text(_models["_exp1_tag"])
    print("wrote", len(new), "rows", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
