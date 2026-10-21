"""Train S1 (10 m) -> X-band-like (2.5 m) super-resolution: L1 baseline, then GAN fine-tune.

Normalization is per tile: each tile's S1 dB and X-band dB are standardized by their own
median and std, so the model learns spatial pattern, not absolute calibration (Umbra uint8
and ICEYE uint16 amplitudes are on different scales).

  python train.py --stage l1   # writes ckpt/l1.pt
  python train.py --stage gan  # starts from l1.pt, writes ckpt/gan.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DEV = "cuda"


# ---------------------------------------------------------------- data
def norm(a):
    v = a[np.isfinite(a)]
    med, sd = np.median(v), v.std() + 1e-6
    return (a - med) / sd


def load_tiles(split, min_corr=0.3):
    """min_corr drops pairs whose 10 m X-band and S1 disagree: misregistered scenes, or sea
    surfaces where wind state flips the contrast between the two dates."""
    man = pd.read_csv(ROOT / "manifest.csv")
    man = man[(man.status.isin(["ok", "cached"])) & (man.split == split) & (man.corr10 >= min_corr)]
    tiles = []
    for r in man.itertuples():
        f = ROOT / "data" / f"{r.provider}_{r.xband_id}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        s1, x = z["s1"].astype("float32"), z["x"].astype("float32")
        if not np.isfinite(s1).all():
            s1 = np.where(np.isfinite(s1), s1, np.nanmedian(s1))
        tiles.append({"id": f.stem, "s1": norm(s1), "x": norm(x), "region": r.test_region,
                      "corr10": getattr(r, "corr10", np.nan)})
    return tiles


class Patches(torch.utils.data.Dataset):
    def __init__(self, tiles, n, lr=32, scale=4, min_valid=0.97):
        self.t, self.n, self.lr, self.s, self.mv = tiles, n, lr, scale, min_valid
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        for _ in range(50):
            t = self.t[self.rng.integers(len(self.t))]
            L = t["s1"].shape[0]
            r, c = self.rng.integers(0, L - self.lr, 2)
            x = t["x"][r * self.s:(r + self.lr) * self.s, c * self.s:(c + self.lr) * self.s]
            if np.isfinite(x).mean() >= self.mv:
                break
        s = t["s1"][r:r + self.lr, c:c + self.lr]
        k = self.rng.integers(4)            # rotations / flips; SAR look direction varies anyway
        s, x = np.rot90(s, k), np.rot90(x, k)
        if self.rng.random() < 0.5:
            s, x = s[:, ::-1], x[:, ::-1]
        m = np.isfinite(x)
        return (torch.from_numpy(s.copy())[None], torch.from_numpy(np.nan_to_num(x).copy())[None],
                torch.from_numpy(m.copy())[None])


# ---------------------------------------------------------------- models
class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + 0.2 * self.b(x)


class Gen(nn.Module):
    """EDSR-style: residual trunk at 10 m, two pixel-shuffle x2 steps, plus a bicubic skip."""

    def __init__(self, c=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
        up = []
        for _ in range(2):
            up += [nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(True)]
        self.up = nn.Sequential(*up, nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, s):
        h = self.head(s)
        h = h + self.body(h)
        return self.up(h) + F.interpolate(s, scale_factor=4, mode="bicubic", align_corners=False)


class Disc(nn.Module):
    """Conditional PatchGAN: sees the bicubic-upsampled S1 next to the real or generated X-band."""

    def __init__(self, c=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        L = []
        cin = 2
        for i, co in enumerate([c, 2 * c, 4 * c, 4 * c]):
            L += [sn(nn.Conv2d(cin, co, 4, stride=2 if i < 3 else 1, padding=1)), nn.LeakyReLU(0.2, True)]
            cin = co
        L += [sn(nn.Conv2d(cin, 1, 4, padding=1))]
        self.net = nn.Sequential(*L)

    def forward(self, s, x):
        return self.net(torch.cat([F.interpolate(s, scale_factor=4, mode="bicubic", align_corners=False), x], 1))


def masked_l1(p, t, m):
    return (torch.abs(p - t) * m).sum() / m.sum().clamp(min=1)


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["l1", "gan"], required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adv-w", type=float, default=0.01)
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    tiles = load_tiles("train")
    print(f"train tiles: {len(tiles)}", flush=True)
    dl = torch.utils.data.DataLoader(Patches(tiles, a.steps * a.bs), batch_size=a.bs, num_workers=0)
    (ROOT / "ckpt").mkdir(exist_ok=True)

    G = Gen().to(DEV)
    oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    if a.stage == "gan":
        G.load_state_dict(torch.load(ROOT / "ckpt/l1.pt"))
        oG = torch.optim.Adam(G.parameters(), lr=a.lr / 2, betas=(0.5, 0.99))
        D = Disc().to(DEV)
        oD = torch.optim.Adam(D.parameters(), lr=a.lr / 2, betas=(0.5, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(oG, a.steps)

    t0, log = time.time(), []
    for step, (s, x, m) in enumerate(dl, 1):
        s, x, m = s.to(DEV), x.to(DEV), m.float().to(DEV)
        p = G(s)
        l1 = masked_l1(p, x, m)
        if a.stage == "l1":
            loss = l1
        else:
            # D step (LSGAN); masked pixels are filled with the prediction so D can't key on holes
            xr = x * m + p.detach() * (1 - m)
            dr, df = D(s, xr), D(s, p.detach())
            lD = ((dr - 1) ** 2).mean() + (df ** 2).mean()
            oD.zero_grad(set_to_none=True)
            lD.backward()
            oD.step()
            adv = ((D(s, p) - 1) ** 2).mean()
            loss = l1 + a.adv_w * adv
        oG.zero_grad(set_to_none=True)
        loss.backward()
        oG.step()
        sched.step()
        if step % 250 == 0:
            rec = {"step": step, "l1": l1.item(), "sec": round(time.time() - t0)}
            if a.stage == "gan":
                rec.update(adv=adv.item(), lD=lD.item())
            log.append(rec)
            print(rec, flush=True)
        if step >= a.steps:
            break
    torch.save(G.state_dict(), ROOT / f"ckpt/{a.stage}.pt")
    pd.DataFrame(log).to_csv(ROOT / f"ckpt/{a.stage}_log.csv", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
