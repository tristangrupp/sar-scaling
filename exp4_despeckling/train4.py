"""Train the 1x despeckling CNN (Experiment 1's residual trunk, no upsampling) on training sites only.

  --mode sup  single date t -> leave-one-out temporal mean (all other dates), masked L1 in dB,
              mask = valid and stable (change mask from stack.py)
  --mode n2n  Noise2Noise: date t -> another random date s of the same orbit, L1 in dB, no clean
              target and no change mask. L1 on dB estimates the median of log-speckle, which sits
              below the log of the mean; we add back one constant offset, fitted on training sites.
  --mode gan  start from sup, add a conditional PatchGAN (spectral norm, relativistic average loss,
              D learning rate 1/4 of G's, adversarial weight --adv_w) to the masked L1.

Inputs and targets are standardized by the input image's own median and std in dB (as in Exp 1).
"""
from __future__ import annotations

import argparse
import sys
import time

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish, norm_stats

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(ROOT.parent / "exp1_gan_spike"))
from train import RB  # noqa: E402  (same residual block as Exp 1)

DEV = "cuda"
CK = ROOT / "ckpt"
CK.mkdir(exist_ok=True)


class DGen(nn.Module):
    """EDSR-style residual trunk at 1x; predicts a correction added to the input."""

    def __init__(self, c=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
        self.tail = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, s):
        h = self.head(s)
        h = h + self.body(h)
        return s + self.tail(h)


class Disc(nn.Module):
    """Conditional PatchGAN on (input, candidate); spectral norm on every conv."""

    def __init__(self, c=48):
        super().__init__()
        sn = nn.utils.spectral_norm
        L, cin = [], 2
        for i, co in enumerate([c, 2 * c, 4 * c]):
            L += [sn(nn.Conv2d(cin, co, 4, stride=2 if i < 2 else 1, padding=1)), nn.LeakyReLU(0.2, True)]
            cin = co
        L += [sn(nn.Conv2d(cin, 1, 4, padding=1))]
        self.net = nn.Sequential(*L)

    def forward(self, s, x):
        return self.net(torch.cat([s, x], 1))


def load_train():
    sites = []
    for name, v in SITES.items():
        if v[3] != "train":
            continue
        z, st = np.load(DATA / f"{name}.npz"), np.load(DATA / f"{name}_stack.npz")
        vv = z["vv"]
        sites.append(dict(name=name, vv=vv, sum=st["sum"], n=st["n"], stable=st["stable"],
                          ns=[norm_stats(to_db(vv[t])) for t in range(vv.shape[0])]))
        print("loaded", name, vv.shape, flush=True)
    return sites


def to_db(a):
    return 10 * np.log10(np.maximum(a, 1e-6))


class Sampler:
    def __init__(self, sites, mode, P=96, seed=0):
        self.s, self.mode, self.P = sites, mode, P
        self.rng = np.random.default_rng(seed)

    def one(self):
        rng, P = self.rng, self.P
        while True:
            S = self.s[rng.integers(len(self.s))]
            T, H, W = S["vv"].shape
            t = rng.integers(T)
            r, c = rng.integers(0, H - P), rng.integers(0, W - P)
            I = S["vv"][t, r:r + P, c:c + P]
            if not np.isfinite(I).all():
                continue
            med, sd = S["ns"][t]          # whole-image stats, as at inference
            x = (to_db(I) - med) / sd
            if self.mode == "n2n":
                s = rng.integers(T - 1)
                s = s + (s >= t)
                J = S["vv"][s, r:r + P, c:c + P]
                m = np.isfinite(J)
                y = (to_db(np.where(m, J, 1)) - med) / sd
            else:
                n = S["n"][r:r + P, c:c + P]
                loo = (S["sum"][r:r + P, c:c + P] - I) / np.maximum(n - 1, 1)
                m = S["stable"][r:r + P, c:c + P] & (n == T)
                if m.mean() < 0.2:
                    continue
                y = (to_db(np.where(m, loo, 1)) - med) / sd
            k = rng.integers(4)
            x, y, m = np.rot90(x, k), np.rot90(y, k), np.rot90(m, k)
            if rng.random() < 0.5:
                x, y, m = x[:, ::-1], y[:, ::-1], m[:, ::-1]
            return x.copy(), y.copy(), m.copy()

    def batch(self, B):
        xs, ys, ms = zip(*[self.one() for _ in range(B)])
        f = lambda a: torch.from_numpy(np.stack(a)[:, None].astype("float32")).to(DEV)  # noqa: E731
        return f(xs), f(ys), f(ms)


def masked_l1(o, y, m):
    return (torch.abs(o - y) * m).sum() / m.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sup", choices=["sup", "n2n", "gan"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adv_w", type=float, default=5e-3)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    sites = load_train()
    smp = Sampler(sites, "n2n" if a.mode == "n2n" else "sup", seed={"sup": 1, "n2n": 2, "gan": 3}[a.mode])
    G = DGen().to(DEV)
    if a.mode == "gan":
        G.load_state_dict(torch.load(CK / "sup.pt", weights_only=True))
        D = Disc().to(DEV)
        oD = torch.optim.Adam(D.parameters(), lr=a.lr / 8, betas=(0.5, 0.99))
        oG = torch.optim.Adam(G.parameters(), lr=a.lr / 2, betas=(0.5, 0.99))
    else:
        oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(oG, a.steps)
    scG, scD = torch.amp.GradScaler(), torch.amp.GradScaler()   # fp16 autocast: the GPU is shared
    log = open(CK / f"{a.mode}_log.csv", "w")
    log.write("step,l1,adv,lD,dreal,dfake,sec\n")
    t0, acc = time.time(), []
    for step in range(1, a.steps + 1):
        x, y, m = smp.batch(a.batch)
        with torch.autocast("cuda", dtype=torch.float16):
            o = G(x)
        o = o.float()
        l1 = masked_l1(o, y, m)
        adv = lD = dr = df = torch.zeros(())
        if a.mode == "gan":
            real = torch.where(m > 0, y, o.detach())
            # D step (relativistic average)
            with torch.autocast("cuda", dtype=torch.float16):
                pr, pf = D(x, real), D(x, o.detach())
            pr, pf = pr.float(), pf.float()
            lD = 0.5 * (F.binary_cross_entropy_with_logits(pr - pf.mean(), torch.ones_like(pr)) +
                        F.binary_cross_entropy_with_logits(pf - pr.mean(), torch.zeros_like(pf)))
            oD.zero_grad(set_to_none=True)
            scD.scale(lD).backward()
            scD.step(oD)
            scD.update()
            with torch.autocast("cuda", dtype=torch.float16):
                pr, pf = D(x, real).detach(), D(x, o)
            pr, pf = pr.float(), pf.float()
            adv = 0.5 * (F.binary_cross_entropy_with_logits(pf - pr.mean(), torch.ones_like(pf)) +
                         F.binary_cross_entropy_with_logits(pr - pf.mean(), torch.zeros_like(pr)))
            dr, df = torch.sigmoid(pr).mean(), torch.sigmoid(pf).mean()
        loss = l1 + a.adv_w * adv
        oG.zero_grad(set_to_none=True)
        scG.scale(loss).backward()
        scG.step(oG)
        scG.update()
        sched.step()
        acc.append([l1.item(), adv.item(), lD.item(), dr.item(), df.item()])
        if step % 200 == 0:
            mu = np.mean(acc, 0)
            acc = []
            log.write(f"{step},{mu[0]:.4f},{mu[1]:.4f},{mu[2]:.4f},{mu[3]:.3f},{mu[4]:.3f},{time.time() - t0:.0f}\n")
            log.flush()
            print(step, np.round(mu, 4), f"{time.time() - t0:.0f}s", flush=True)
        if step % 2000 == 0 or step == a.steps:
            torch.save(G.state_dict(), CK / f"{a.mode}.pt")
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
