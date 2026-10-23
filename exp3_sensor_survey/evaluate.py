"""Score bicubic and arm A on held-out NISAR tiles (near and far), against the real fine product.

Metrics, in gamma0 dB on valid target pixels:
  mae, ssim        as in Experiments 1-2 (ssim data range = truth 1-99 pct span)
  detail_r         corr of the fine-detail layer (image minus its block mean on the INPUT grid)
  hf_ratio         fine-detail energy of prediction / truth (1 = as sharp as truth)
  cons             mean |pool(pred) - input| dB: does the output still agree with what was measured
  ch_p2t, ch_t2p   edge chamfer, metres: mean distance from predicted edges to the nearest truth edge
                   (invented / misplaced edges), and from truth edges to the nearest predicted edge
                   (missed edges). Canny with quantile thresholds, so every image has the same edge density.
                   Shift-tolerant, unlike detail_r.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.metrics import structural_similarity
from train import ARMS, DEV, Gen, from_n, load, to_n

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))   # project root, for panel_roles
from panel_roles import outline         # noqa: E402


def pool_db(d, k):
    n0, n1 = d.shape[0] // k, d.shape[1] // k
    l = 10 ** (d / 10)
    return 10 * np.log10(l.reshape(n0, k, n1, k).mean(axis=(1, 3)))


def detail(a, k):
    n0, n1 = a.shape[0] // k, a.shape[1] // k
    lo = a.reshape(n0, k, n1, k).mean(axis=(1, 3))
    return a - np.kron(lo, np.ones((k, k)))


def edges(d, sigma=1.5):
    return canny(d, sigma=sigma, low_threshold=0.80, high_threshold=0.92, use_quantiles=True)


def chamfer(ep, et, m, px):
    """mean distance (m) from edges in ep to nearest edge in et, both directions, inside mask m"""
    if ep[m].sum() == 0 or et[m].sum() == 0:
        return np.nan, np.nan
    dt_t = ndi.distance_transform_edt(~et) * px
    dt_p = ndi.distance_transform_edt(~ep) * px
    return float(dt_t[ep & m].mean()), float(dt_p[et & m].mean())


def scores(p, t, m, x, k, px):
    lo, hi = np.percentile(t[m], [1, 99])
    tt, pp = np.where(m, t, np.median(t[m])), np.where(m, p, np.median(t[m]))
    ssim = structural_similarity(np.clip(tt, lo, hi), np.clip(pp, lo, hi), data_range=hi - lo)
    mk = m.reshape(m.shape[0] // k, k, -1, k).all(axis=(1, 3))
    mm = np.kron(mk, np.ones((k, k), bool))
    dp, dt = detail(pp, k)[mm], detail(tt, k)[mm]
    er = ndi.binary_erosion(m, iterations=6)
    c1, c2 = chamfer(edges(pp), edges(tt), er, px)
    dc = np.abs(np.diff(pp, axis=1)); dr = np.abs(np.diff(pp, axis=0))
    across = np.r_[dc[:, k - 1::k].ravel(), dr[k - 1::k, :].ravel()].mean()
    within = np.r_[np.delete(dc, np.s_[k - 1::k], axis=1).ravel(), np.delete(dr, np.s_[k - 1::k], axis=0).ravel()].mean()
    return {"grid": float(across / within), "mae": float(np.abs(p - t)[m].mean()), "ssim": float(ssim),
            "detail_r": float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 else 0.0,
            "hf_ratio": float(dp.std() / dt.std()), "cons": float(np.abs(pool_db(p, k) - x).mean()),
            "ch_p2t": c1, "ch_t2p": c2}


@torch.no_grad()
def predict(G, xn):
    return from_n(G(torch.from_numpy(xn.astype("float32"))[None, None].to(DEV))[0, 0].cpu().numpy())


def bicubic(xn, s):
    return from_n(F.interpolate(torch.from_numpy(xn.astype("float32"))[None, None], scale_factor=s,
                                mode="bicubic", align_corners=False)[0, 0].numpy())


def main():
    bad = set(json.loads((ROOT / "bad_tiles.json").read_text())) if (ROOT / "bad_tiles.json").exists() else set()
    tiles_csv = pd.read_csv(ROOT / "tiles.csv").set_index("tile")
    out = ROOT / "eval"
    (out / "panels").mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ["x2", "x4"]:
        f = ROOT / f"ckpt/{arm}.pt"
        if not f.exists():
            continue
        ki, kt, s = ARMS[arm]
        px = 10 if kt == "a10_HH" else 20
        G = Gen(s).to(DEV).eval()
        G.load_state_dict(torch.load(f))
        G0 = None
        if (ROOT / f"ckpt/{arm}_cons0.pt").exists():      # ablation: no input-consistency loss
            G0 = Gen(s).to(DEV).eval()
            G0.load_state_dict(torch.load(ROOT / f"ckpt/{arm}_cons0.pt"))
        for role in ["near", "far"]:
            for t in load(role, arm, bad):
                y, m, xn = from_n(t["y"]), t["m"], t["x"]
                if m.mean() < 0.9:
                    continue
                x = from_n(xn)
                P = {"bicubic": bicubic(xn, s), "cnn": predict(G, xn), "truth": y}
                if G0 is not None:
                    P["cnn_cons0"] = predict(G0, xn)
                reg = tiles_csv.loc[t["id"], "reg"] if t["id"] in tiles_csv.index else ""
                for k, p in P.items():
                    rows.append({"arm": arm, "role": role, "tile": t["id"], "pair": t["meta"]["pair"], "reg": reg,
                                 "model": k, **scores(p, y, m, x, s, px)})
                if arm == "x2" and len([r for r in rows if r["arm"] == "x2" and r["model"] == "cnn" and r["role"] == role]) % 9 == 1:
                    panel(t, x, {k: v for k, v in P.items() if k != "truth"}, y, s, out / "panels" / f"{arm}_{role}_{t['id']}.png")
        print(arm, "done", len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    cols = ["mae", "ssim", "detail_r", "hf_ratio", "cons", "ch_p2t", "ch_t2p", "grid"]
    summ = df.groupby(["arm", "role", "model"])[cols].median().round(3)
    summ["n"] = df.groupby(["arm", "role", "model"]).size()
    print(summ.to_string(), flush=True)
    summ.to_csv(out / "summary.csv")
    # paired win rate: cnn better than bicubic on the same tile
    w = df[df.model != "truth"].pivot_table(index=["arm", "role", "tile"], columns="model", values=cols)
    win = pd.DataFrame({c: (w[c]["cnn"] > w[c]["bicubic"]) if c in ("ssim", "detail_r") else (w[c]["cnn"] < w[c]["bicubic"])
                        for c in ["mae", "ssim", "detail_r", "ch_p2t", "ch_t2p"]}).groupby(level=[0, 1]).mean().round(3)
    print("cnn-better fraction\n", win.to_string(), flush=True)
    win.to_csv(out / "winrate.csv")
    df.groupby(["arm", "role", "reg", "model"])[cols].median().round(3).to_csv(out / "by_region.csv")


def panel(t, x, P, y, s, f, n=192):
    best, rc = -1, (0, 0)
    for r in range(0, y.shape[0] - n + 1, 64):
        for c in range(0, y.shape[1] - n + 1, 64):
            v = np.std(y[r:r + n, c:c + n])
            if v > best:
                best, rc = v, (r, c)
    r, c = rc
    imgs = {"Input 20 MHz (20 m)": np.kron(x, np.ones((s, s)))[r:r + n, c:c + n], "Bicubic": P["bicubic"][r:r + n, c:c + n],
            "CNN arm A": P["cnn"][r:r + n, c:c + n]}
    if "cnn_cons0" in P:
        imgs["CNN, no consistency"] = P["cnn_cons0"][r:r + n, c:c + n]
    imgs["Truth 40 MHz (10 m)"] = y[r:r + n, c:c + n]
    lo, hi = np.percentile(imgs["Truth 40 MHz (10 m)"], [2, 98])
    fig, ax = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3.3), dpi=100)
    for a_, (k, v) in zip(ax, imgs.items()):
        a_.imshow(v, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        a_.set_title(k, fontsize=9); a_.axis("off")
        if k.startswith("CNN"):
            outline(a_, "model")
        elif k.startswith("Truth"):
            outline(a_, "truth")
    fig.suptitle(f"{t['id']}  ({n * 10 / 1000:.2f} km)", fontsize=9)
    fig.tight_layout(); fig.savefig(f); plt.close(fig)


def panels_only():
    """Redraw the existing x2 side-by-side panels from the saved checkpoints, without rescoring (no CSVs written)."""
    bad = set(json.loads((ROOT / "bad_tiles.json").read_text())) if (ROOT / "bad_tiles.json").exists() else set()
    pdir = ROOT / "eval" / "panels"
    want = {q.stem for q in pdir.glob("x2_*.png")}
    ki, kt, s = ARMS["x2"]
    G = Gen(s).to(DEV).eval(); G.load_state_dict(torch.load(ROOT / "ckpt/x2.pt", map_location=DEV))
    G0 = None
    if (ROOT / "ckpt/x2_cons0.pt").exists():
        G0 = Gen(s).to(DEV).eval(); G0.load_state_dict(torch.load(ROOT / "ckpt/x2_cons0.pt", map_location=DEV))
    for role in ["near", "far"]:
        for t in load(role, "x2", bad):
            if f"x2_{role}_{t['id']}" not in want:
                continue
            y, xn = from_n(t["y"]), t["x"]
            P = {"bicubic": bicubic(xn, s), "cnn": predict(G, xn)}
            if G0 is not None:
                P["cnn_cons0"] = predict(G0, xn)
            panel(t, from_n(xn), P, y, s, pdir / f"x2_{role}_{t['id']}.png")
            print("panel", role, t["id"], flush=True)


if __name__ == "__main__":
    panels_only() if "--panels-only" in sys.argv else main()
    sys.stdout.flush()
    os._exit(0)
