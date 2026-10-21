"""Train IW (10 m) -> Stripmap (5 m), x2.

Arm A (baseline reproduction of Amieva et al. 2026): SSIM loss against the histogram-matched,
Lee-filtered Stripmap reference, 48 -> 96 px patches, min-max [0, 1] normalization, OneCycleLR
(max 1e-3). Optional input-consistency term (our design; the paper's distribution-preservation
loss uses the same term): L2 between the output average-pooled to 10 m and the IW input, weighted
0.15 against 0.85 for SSIM.

Arm B: arm A plus a tolerant X-band edge loss on infrastructure pixels (see edge_loss).

  python train_sr.py --model esrt --name esrt_a [--cons 0.15] [--edge-w 0.05] [--steps 12000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, ROOT
from esrt import ESRT
from sr_data import Norm, apply_change_mask, load_manifest, load_split

DEV = "cuda"
CKPT = ROOT / "ckpt"
XB = DATA / "xband"
LR_P, SC = 48, 2


# ------------------------------------------------------------------ models
class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + 0.2 * self.b(x)


class CNN(nn.Module):
    """The spike's EDSR-style residual generator (exp1_gan_spike/train.py Gen), one x2 pixel-shuffle step."""

    def __init__(self, c=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
        self.up = nn.Sequential(nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(True),
                                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, s):
        h = self.head(s)
        h = h + self.body(h)
        return self.up(h) + F.interpolate(s, scale_factor=2, mode="bicubic", align_corners=False)


def make_model(kind):
    return ESRT(upscale=2) if kind == "esrt" else CNN()


# ------------------------------------------------------------------ losses
def _gauss(win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[None]
    return (g.T @ g)[None, None]


_G = None


def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    global _G
    if _G is None or _G.device != x.device:
        _G = _gauss().to(x.device)
    f = lambda a: F.conv2d(a, _G, padding=5)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))


def masked_mean(v, m):
    return (v * m).sum() / m.sum().clamp(min=1)


def sobel(x):
    k = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=x.device)
    gx = F.conv2d(x, k[None, None], padding=1)
    gy = F.conv2d(x, k.T[None, None], padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def edge_loss(p, dist, w, tol=1.0, dmax=4.0):
    """Tolerant X-band edge loss: the share of the output's edge energy (Sobel magnitude) weighted by
    how far it sits from the nearest X-band edge (distance transform, 5 m px), within the weight mask
    (infrastructure, X-band valid, no layover / shadow / change). Edges within `tol` px cost nothing;
    the cost saturates at `dmax` px. Scale-invariant in edge strength, so blurring does not lower it;
    only moving edge energy onto X-band edges does."""
    e = sobel(p)
    d = ((dist - tol).clamp(min=0) / (dmax - tol)).clamp(max=1)
    num = (e * d * w).sum(dim=(1, 2, 3))
    den = (e * w).sum(dim=(1, 2, 3))
    ok = w.sum(dim=(1, 2, 3)) > 50
    if ok.sum() == 0:
        return p.sum() * 0
    return (num[ok] / den[ok].clamp(min=1e-6)).mean()


# ------------------------------------------------------------------ sampling
class Sampler:
    def __init__(self, tiles, norm, target="lee_hm", min_valid=0.8, seed=0):
        self.t, self.n, self.target, self.mv = tiles, norm, target, min_valid
        self.rng = np.random.default_rng(seed)

    def patch(self, t, r=None, c=None):
        for _ in range(50):
            rr = self.rng.integers(0, 320 - LR_P) if r is None else r
            cc = self.rng.integers(0, 320 - LR_P) if c is None else c
            m = t["mask"][rr * 2:(rr + LR_P) * 2, cc * 2:(cc + LR_P) * 2]
            if m.mean() >= self.mv or r is not None:
                break
        return rr, cc

    def batch(self, bs, xtiles=None, p_x=0.0):
        S, Y, M, D, W = [], [], [], [], []
        for _ in range(bs):
            use_x = xtiles and self.rng.random() < p_x
            if use_x:
                xt = xtiles[self.rng.integers(len(xtiles))]
                t = xt["tile"]
                rr, cc = self.patch_x(xt)
            else:
                t = self.t[self.rng.integers(len(self.t))]
                rr, cc = self.patch(t)
            s = t["iw"][rr:rr + LR_P, cc:cc + LR_P]
            y = t[self.target][rr * 2:(rr + LR_P) * 2, cc * 2:(cc + LR_P) * 2]
            m = t["mask"][rr * 2:(rr + LR_P) * 2, cc * 2:(cc + LR_P) * 2] & np.isfinite(y)
            if use_x:
                d = xt["dist"][rr * 2:(rr + LR_P) * 2, cc * 2:(cc + LR_P) * 2]
                w = xt["w"][rr * 2:(rr + LR_P) * 2, cc * 2:(cc + LR_P) * 2]
            else:
                d = np.zeros_like(y)
                w = np.zeros_like(m)
            k = self.rng.integers(4)
            fl = self.rng.random() < 0.5
            arrs = []
            for a in (s, y, m, d, w):
                a = np.rot90(a, k)
                arrs.append(a[:, ::-1] if fl else a)
            s, y, m, d, w = arrs
            S.append(np.nan_to_num(self.n(s), nan=0.0))
            Y.append(np.nan_to_num(self.n(y), nan=0.0))
            M.append(m)
            D.append(d)
            W.append(w & m)
        f = lambda L, dt=np.float32: torch.from_numpy(np.stack(L).astype(dt))[:, None].to(DEV)  # noqa: E731
        return f(S), f(Y), f(M), f(D), f(W)

    def patch_x(self, xt):
        """A patch centered on a random infrastructure pixel with X-band edges nearby."""
        ys, xs = xt["w_idx"]
        i = self.rng.integers(len(ys))
        rr = int(np.clip(ys[i] // 2 - LR_P // 2, 0, 320 - LR_P))
        cc = int(np.clip(xs[i] // 2 - LR_P // 2, 0, 320 - LR_P))
        return rr, cc


def load_xband_tiles(tiles, splits_ok):
    """Edge targets for training tiles: DATA/xband/<tile_id>__<xband_id>.npz from xband_prep.py."""
    by_id = {t["tile_id"]: t for t in tiles}
    xb = pd.read_csv(ROOT / "xband_pairs.csv")
    xb = xb[(xb.status == "ok") & xb.split.isin(splits_ok) & xb.tile_id.isin(by_id)]
    out = []
    for r in xb.itertuples():
        z = np.load(XB / f"{r.tile_id}__{r.xband_id}.npz")
        w = z["infra"] & z["valid"]
        if w.sum() < 200:
            continue
        out.append({"tile": by_id[r.tile_id], "dist": z["dist"].astype("float32"), "w": w,
                    "w_idx": np.nonzero(w), "xband_id": r.xband_id, "provider": r.provider})
    return out


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict(model, iw_n, win=LR_P, margin=8, bs=16):
    """Tiled inference on a normalized IW array (NaN filled with 0): windows of the training size,
    keeping each window's center (margin px dropped where neighbors exist)."""
    model.eval()
    H, W = iw_n.shape
    step = win - 2 * margin
    out = np.zeros((H * SC, W * SC), "float32")
    pos = []
    for r in list(range(0, H - win, step)) + [H - win]:
        for c in list(range(0, W - win, step)) + [W - win]:
            pos.append((r, c))
    x = torch.from_numpy(np.nan_to_num(iw_n, nan=0.0).astype("float32"))
    for i in range(0, len(pos), bs):
        chunk = pos[i:i + bs]
        b = torch.stack([x[r:r + win, c:c + win] for r, c in chunk])[:, None].to(DEV)
        y = model(b)[:, 0].float().cpu().numpy()
        for (r, c), yy in zip(chunk, y):
            r0 = 0 if r == 0 else margin
            c0 = 0 if c == 0 else margin
            out[(r + r0) * SC:(r + win) * SC, (c + c0) * SC:(c + win) * SC] = yy[r0 * SC:, c0 * SC:]
    model.train()
    return out


def val_ssim(model, tiles, norm, target="lee_hm"):
    from skimage.metrics import structural_similarity
    vals = []
    for t in tiles:
        p = predict(model, norm(t["iw"]))
        y = norm(t[target])
        m = t["mask"] & np.isfinite(y)
        if m.mean() < 0.3:
            continue
        _, S = structural_similarity(np.where(m, y, 0), np.where(m, p, 0), data_range=1.0, full=True)
        vals.append(float(S[m].mean()))
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["esrt", "cnn"], default="esrt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cons", type=float, default=0.0, help="input-consistency weight (SSIM gets 1-cons)")
    ap.add_argument("--edge-w", type=float, default=0.0, help="arm B X-band edge loss weight")
    ap.add_argument("--p-x", type=float, default=0.25, help="share of patches drawn on X-band infrastructure")
    ap.add_argument("--target", default="lee_hm", choices=["lee_hm", "sm_hm"])
    ap.add_argument("--init", default=None, help="start from this checkpoint name")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--splits", default="v1", choices=["v1", "v2"], help="tile set (v2 adds round 2 inland tiles)")
    ap.add_argument("--change-mask", action="store_true", help="drop IW-detected change from the training loss")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    man = load_manifest(a.splits)
    tr = load_split(["train"], man)
    va = load_split(["val"], man)
    norm = Norm.fit(tr)
    print(f"train tiles {len(tr)}  val tiles {len(va)}  norm [{norm.lo:.2f}, {norm.hi:.2f}] dB", flush=True)
    xt = load_xband_tiles(tr, ["train"]) if a.edge_w > 0 else []
    if a.edge_w > 0:
        print(f"x-band edge tiles: {len(xt)} ({pd.Series([x['provider'] for x in xt]).value_counts().to_dict()})",
              flush=True)
    if a.change_mask:          # training tiles only; validation stays unmasked so runs compare
        print(f"change mask removes {100 * apply_change_mask(tr):.1f}% of training loss pixels", flush=True)
    samp = Sampler(tr, norm, a.target, seed=a.seed)
    model = make_model(a.model).to(DEV)
    if a.init:
        model.load_state_dict(torch.load(CKPT / f"{a.init}.pt", map_location=DEV)["state"])
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    CKPT.mkdir(exist_ok=True)
    log, best, t0 = [], -1.0, time.time()
    for step in range(1, a.steps + 1):
        s, y, m, d, w = samp.batch(a.bs, xt, a.p_x if a.edge_w > 0 else 0.0)
        p = model(s)
        l_ssim = 1 - masked_mean(ssim_map(p, y), m)
        loss = (1 - a.cons) * l_ssim
        l_cons = l_edge = torch.zeros((), device=DEV)
        if a.cons > 0:
            m10 = F.avg_pool2d(m, 2) > 0.99
            l_cons = masked_mean((F.avg_pool2d(p, 2) - s) ** 2, m10.float())
            loss = loss + a.cons * l_cons
        if a.edge_w > 0:
            l_edge = edge_loss(p, d, w)
            loss = loss + a.edge_w * l_edge
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            rec = {"step": step, "ssim_loss": round(l_ssim.item(), 4), "cons": round(l_cons.item(), 5),
                   "edge": round(l_edge.item(), 4), "sec": round(time.time() - t0)}
            if step % a.val_every == 0 or step == a.steps:
                rec["val_ssim"] = round(val_ssim(model, va, norm, a.target), 4)
                if rec["val_ssim"] > best:
                    best = rec["val_ssim"]
                    torch.save({"state": model.state_dict(), "model": a.model, "norm": [norm.lo, norm.hi],
                                "args": vars(a), "step": step}, CKPT / f"{a.name}.pt")
            log.append(rec)
            print(rec, flush=True)
    torch.save({"state": model.state_dict(), "model": a.model, "norm": [norm.lo, norm.hi], "args": vars(a),
                "step": a.steps}, CKPT / f"{a.name}_last.pt")
    pd.DataFrame(log).to_csv(CKPT / f"{a.name}_log.csv", index=False)
    (CKPT / f"{a.name}.json").write_text(json.dumps({"best_val_ssim": best, **vars(a)}, indent=1))
    print("done best val ssim", best, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
