"""Score bicubic / L1 / GAN on held-out test tiles against the real X-band, and save panels.

Metrics are computed in per-tile standardized dB on valid X-band pixels:
  mae         mean absolute error
  ssim        structural similarity (skimage, data range = truth 1-99 pct span)
  detail_r    correlation of the fine-detail layer (image minus its 10 m block mean)
              between prediction and truth: does the added detail land where real detail is?
  hf_ratio    fine-detail energy of prediction / truth (1 = as sharp as truth, <1 blurry)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from train import DEV, Gen, load_tiles

ROOT = Path(__file__).resolve().parent


def detail(a):
    n = a.shape[0] // 4
    lo = a.reshape(n, 4, n, 4).mean(axis=(1, 3))
    return a - np.kron(lo, np.ones((4, 4)))


def scores(p, t, m):
    tt = np.where(m, t, 0)
    pp = np.where(m, p, 0)
    lo, hi = np.percentile(t[m], [1, 99])
    ssim = structural_similarity(np.clip(tt, lo, hi), np.clip(pp, lo, hi), data_range=hi - lo)
    m4 = m.reshape(m.shape[0] // 4, 4, -1, 4).all(axis=(1, 3))
    mm = np.kron(m4, np.ones((4, 4), bool))
    dp, dt = detail(pp)[mm], detail(tt)[mm]
    return {"mae": float(np.abs(p - t)[m].mean()), "ssim": float(ssim),
            "detail_r": float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 else 0.0,
            "hf_ratio": float(dp.std() / dt.std())}


@torch.no_grad()
def predict(G, s1):
    s = torch.from_numpy(s1.astype("float32"))[None, None].to(DEV)
    return G(s)[0, 0].cpu().numpy()


def main():
    tiles = [dict(t, split="test") for t in load_tiles("test")]
    # Capella HH tiles in the original test regions: cross-pol against the VV input, scored apart
    tiles += [dict(t, split="test_xpol") for t in load_tiles("test_xpol", min_corr=-1)]
    models = {}
    for name in ["l1", "gan"]:
        f = ROOT / f"ckpt/{name}.pt"
        if f.exists():
            G = Gen().to(DEV).eval()
            G.load_state_dict(torch.load(f))
            models[name] = G
    out = ROOT / "eval"
    (out / "panels").mkdir(parents=True, exist_ok=True)

    rows, preds = [], {}
    for t in tiles:
        x = t["x"]
        m = np.isfinite(x)
        if m.mean() < 0.8:
            continue
        s = torch.from_numpy(t["s1"].astype("float32"))[None, None]
        bic = F.interpolate(s, scale_factor=4, mode="bicubic", align_corners=False)[0, 0].numpy()
        P = {"bicubic": bic, **{k: predict(G, t["s1"]) for k, G in models.items()}}
        preds[t["id"]] = P
        for k, p in P.items():
            rows.append({"tile": t["id"], "region": t["region"], "split": t["split"],
                         "provider": t["id"].split("_")[0], "model": k, **scores(p, np.nan_to_num(x), m)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    by_provider = df[df.split == "test"].groupby(["provider", "model"])[["mae", "ssim", "detail_r", "hf_ratio"]].median().round(3)
    by_provider.to_csv(out / "by_provider.csv")
    xpol = df[df.split == "test_xpol"].groupby("model")[["mae", "ssim", "detail_r", "hf_ratio"]].median().round(3)
    xpol.to_csv(out / "xpol_summary.csv")
    df = df[df.split == "test"]
    summ = df.groupby("model")[["mae", "ssim", "detail_r", "hf_ratio"]].median().round(3)
    by_region = df.groupby(["region", "model"])[["mae", "ssim", "detail_r"]].median().round(3)
    print(summ, "\n", by_region, flush=True)
    summ.to_csv(out / "summary.csv")
    by_region.to_csv(out / "by_region.csv")

    # panels: one 256 px (640 m) crop from each test tile, picked where the truth has most structure
    order = ["s1", "bicubic", "l1", "gan", "truth"]
    titles = {"s1": "Sentinel-1 IW (10 m)", "bicubic": "Bicubic", "l1": "CNN, pixel loss",
              "gan": "CNN + GAN", "truth": "Real X-band (2.5 m)"}
    panel_meta = []
    for t in tiles:
        if t["id"] not in preds:
            continue
        x = t["x"]
        best, rc = -1, (0, 0)
        for r in range(0, 1280 - 256, 128):
            for c in range(0, 1280 - 256, 128):
                w = x[r:r + 256, c:c + 256]
                if np.isfinite(w).mean() > 0.99 and np.nanstd(w) > best:
                    best, rc = np.nanstd(w), (r, c)
        r, c = rc
        imgs = {"s1": np.kron(t["s1"], np.ones((4, 4)))[r:r + 256, c:c + 256],
                "truth": x[r:r + 256, c:c + 256]}
        imgs.update({k: v[r:r + 256, c:c + 256] for k, v in preds[t["id"]].items()})
        lo, hi = np.nanpercentile(imgs["truth"], [2, 98])
        fig, ax = plt.subplots(1, len(order), figsize=(3 * len(order), 3.2), dpi=110)
        for a_, k in zip(ax, order):
            if k in imgs:
                a_.imshow(imgs[k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            a_.set_title(titles[k], fontsize=9)
            a_.axis("off")
        fig.tight_layout()
        fig.savefig(out / "panels" / f"{t['id']}.png")
        plt.close(fig)
        panel_meta.append({"id": t["id"], "region": t["region"], "split": t["split"]})
    (out / "panels.json").write_text(json.dumps(panel_meta))
    print("panels:", len(panel_meta), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
