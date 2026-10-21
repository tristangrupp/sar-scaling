"""Round 2 despeckling CNNs.

  --mode sup   input -> leave-out temporal mean (dates in the input excluded), masked L1 in dB,
               round-1 training sites only (they have stable 20+ date stacks)
  --mode n2n   input -> one other single date of the same orbit, L1 in dB, valid mask only.
               --sites all adds the 18 round-2 sites (26 training sites); no clean target used.
               A constant per-polarization offset (log-domain median bias) is fitted afterwards on
               the round-1 training sites and stored in the checkpoint's meta file.
  --pols vv | vvvh   one channel, or VV and VH in and out (each standardized on its own)
  --cond       speckle-level conditioning: inputs are the mean of k = 1..4 random dates (so the
               network sees cleaner inputs too), plus one constant channel per polarization holding
               log(enl_hat(input)) from common.enl_hat, computed the same way at test time.

Checkpoint: ckpt/<name>.pt and ckpt/<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import common  # noqa: F401
from common import DATA, N2N_SITES, ROOT, SITES, enl_hat, finish, norm_stats

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(ROOT.parent / "exp1_gan_spike"))
from train import RB  # noqa: E402

DEV = "cuda"
CK = ROOT / "ckpt"
LOG_ENL0 = np.log(10.0)


def to_db(a):
    return 10 * np.log10(np.maximum(a, 1e-6))


class DGenC(nn.Module):
    """Experiment 1's residual trunk at 1x; cin = cout image channels + optional conditioning."""

    def __init__(self, cin=1, cout=1, c=64, nb=16):
        super().__init__()
        self.cout = cout
        self.head = nn.Conv2d(cin, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
        self.tail = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, cout, 3, padding=1))

    def forward(self, s):
        h = self.head(s)
        h = h + self.body(h)
        return s[:, :self.cout] + self.tail(h)


def load_sites(pols, which, need_ref):
    out = []
    for name, v in SITES.items():
        if v[3] != "train":
            continue
        z = np.load(DATA / f"{name}.npz")
        S = dict(name=name, img={"vv": z["vv"]})
        if "vh" in pols:
            S["img"]["vh"] = np.load(DATA / f"{name}_vh.npz")["vh"]
        if need_ref:
            for p in pols:
                st = np.load(DATA / (f"{name}_stack.npz" if p == "vv" else f"{name}_vh_stack.npz"))
                S[f"sum_{p}"], S[f"n_{p}"], S[f"stable_{p}"] = st["sum"], st["n"], st["stable"]
        out.append(S)
    if which == "all":
        for name in N2N_SITES:
            f = DATA / f"n2n_{name}.npz"
            if f.exists():
                z = np.load(f)
                out.append(dict(name=name, img={p: z[p] for p in pols}))
    for S in out:
        print("loaded", S["name"], S["img"]["vv"].shape, flush=True)
    return out


class Sampler:
    def __init__(self, sites, pols, mode, cond, kmax, P=96, C=256, seed=0):
        self.s, self.pols, self.mode, self.cond, self.kmax, self.P, self.C = sites, pols, mode, cond, kmax, P, C
        self.rng = np.random.default_rng(seed)

    def one(self):
        rng, P, C = self.rng, self.P, self.C
        while True:
            S = self.s[rng.integers(len(self.s))]
            T, H, W = S["img"]["vv"].shape
            k = int(rng.integers(1, self.kmax + 1)) if self.cond else 1
            k = min(k, T - 1)
            sel = rng.choice(T, k, replace=False)
            cr, cc = rng.integers(0, H - C + 1), rng.integers(0, W - C + 1)     # context window
            r, c = cr + rng.integers(0, C - P + 1), cc + rng.integers(0, C - P + 1)
            xs, ys, ms, conds = [], [], [], []
            ok = True
            other = None
            if self.mode == "n2n":
                rest = np.setdiff1d(np.arange(T), sel)
                other = int(rng.choice(rest))
            for p in self.pols:
                stack = S["img"][p]
                ctx = np.nanmean(stack[sel, cr:cr + C, cc:cc + C], 0) if k > 1 else stack[sel[0], cr:cr + C, cc:cc + C]
                I = ctx[r - cr:r - cr + P, c - cc:c - cc + P]
                if not np.isfinite(I).all():
                    ok = False
                    break
                med, sd = norm_stats(to_db(ctx))
                xs.append((to_db(I) - med) / sd)
                if self.cond:
                    conds.append(np.full((P, P), np.log(enl_hat(ctx)) - LOG_ENL0, "float32"))
                if self.mode == "n2n":
                    J = stack[other, r:r + P, c:c + P]
                    m = np.isfinite(J)
                    ys.append((to_db(np.where(m, J, 1)) - med) / sd)
                else:
                    n = S[f"n_{p}"][r:r + P, c:c + P]
                    Ssum = S[f"sum_{p}"][r:r + P, c:c + P] - np.nansum(stack[sel, r:r + P, c:c + P], 0)
                    ref = Ssum / np.maximum(n - k, 1)
                    m = S[f"stable_{p}"][r:r + P, c:c + P] & (n == T)
                    ys.append((to_db(np.where(m, ref, 1)) - med) / sd)
                ms.append(m)
            if not ok or (self.mode == "sup" and np.mean(ms) < 0.2):
                continue
            x, y, m = np.stack(xs + conds), np.stack(ys), np.stack(ms)
            kk = rng.integers(4)
            x, y, m = np.rot90(x, kk, (1, 2)), np.rot90(y, kk, (1, 2)), np.rot90(m, kk, (1, 2))
            if rng.random() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, :, ::-1], m[:, :, ::-1]
            return x.copy(), y.copy(), m.copy()

    def batch(self, B):
        xs, ys, ms = zip(*[self.one() for _ in range(B)])
        f = lambda a: torch.from_numpy(np.stack(a).astype("float32")).to(DEV)  # noqa: E731
        return f(xs), f(ys), f(ms)


def build_input(imgs, pols, cond):
    """imgs: {pol: linear image}. Returns (x tensor (1,C,H,W), stats per pol)."""
    xs, conds, stats = [], [], {}
    for p in pols:
        I = np.where(np.isfinite(imgs[p]), imgs[p], np.nanmedian(imgs[p]))
        db = to_db(I)
        med, sd = norm_stats(db)
        stats[p] = (med, sd)
        xs.append((db - med) / sd)
        if cond:
            conds.append(np.full(db.shape, np.log(enl_hat(I)) - LOG_ENL0))
    return torch.from_numpy(np.stack(xs + conds).astype("float32"))[None].to(DEV), stats


def load_model(name):
    meta = json.loads((CK / f"{name}.json").read_text())
    C = len(meta["pols"])
    G = DGenC(cin=C * (2 if meta["cond"] else 1), cout=C).to(DEV).eval()
    G.load_state_dict(torch.load(CK / f"{name}.pt", weights_only=True))
    return G, meta


@torch.no_grad()
def apply_model(G, meta, imgs):
    x, stats = build_input(imgs, meta["pols"], meta["cond"])
    y = G(x)[0].float().cpu().numpy()
    out = {}
    for i, p in enumerate(meta["pols"]):
        med, sd = stats[p]
        out[p] = 10 ** ((y[i] * sd + med + meta.get("offset", {}).get(p, 0.0)) / 10)
    return out


def calibrate(G, meta):
    """Median (reference dB - output dB) on stable pixels of the round-1 training sites, 4 dates each."""
    off = {}
    for p in meta["pols"]:
        diffs = []
        for name, v in SITES.items():
            if v[3] != "train":
                continue
            vv = np.load(DATA / f"{name}.npz")["vv"]
            imgs_all = {"vv": vv}
            if "vh" in meta["pols"]:
                imgs_all["vh"] = np.load(DATA / f"{name}_vh.npz")["vh"]
            st = np.load(DATA / (f"{name}_stack.npz" if p == "vv" else f"{name}_vh_stack.npz"))
            for t in np.linspace(0, vv.shape[0] - 1, 4).round().astype(int):
                imgs = {q: imgs_all[q][t] for q in meta["pols"]}
                o = apply_model(G, {**meta, "offset": {}}, imgs)[p]
                I = imgs_all[p][t]
                valid = np.isfinite(I)
                ref = (st["sum"] - np.where(valid, I, 0)) / np.maximum(st["n"] - valid, 1)
                M = st["stable"] & valid
                diffs.append(float(np.median((to_db(ref) - to_db(o))[M])))
        off[p] = float(np.median(diffs))
        print("offset", p, round(off[p], 3), flush=True)
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["sup", "n2n"], required=True)
    ap.add_argument("--pols", default="vv", choices=["vv", "vvvh"])
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--sites", default="r1", choices=["r1", "all"])
    ap.add_argument("--steps", type=int, default=4000)   # first two R2 runs used 6000
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    pols = ["vv"] if a.pols == "vv" else ["vv", "vh"]
    torch.backends.cudnn.benchmark = True
    sites = load_sites(pols, a.sites, need_ref=a.mode == "sup")
    smp = Sampler(sites, pols, a.mode, a.cond, a.kmax, seed=hash(a.name) % 1000)
    C = len(pols)
    G = DGenC(cin=C * (2 if a.cond else 1), cout=C).to(DEV)
    oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(oG, a.steps)
    sc = torch.amp.GradScaler()
    meta = dict(name=a.name, mode=a.mode, pols=pols, cond=a.cond, kmax=a.kmax if a.cond else 1, sites=a.sites,
                n_sites=len(sites), steps=a.steps)
    log = open(CK / f"{a.name}_log.csv", "w")
    log.write("step,l1,sec\n")
    t0, acc = time.time(), []
    for step in range(1, a.steps + 1):
        x, y, m = smp.batch(a.batch)
        with torch.autocast("cuda", dtype=torch.float16):
            o = G(x)
        l1 = (torch.abs(o.float() - y) * m).sum() / m.sum().clamp(min=1)
        oG.zero_grad(set_to_none=True)
        sc.scale(l1).backward()
        sc.step(oG)
        sc.update()
        sched.step()
        acc.append(l1.item())
        if step % 200 == 0:
            log.write(f"{step},{np.mean(acc):.4f},{time.time() - t0:.0f}\n")
            log.flush()
            print(step, round(float(np.mean(acc)), 4), f"{time.time() - t0:.0f}s", flush=True)
            acc = []
        if step % 2000 == 0 or step == a.steps:
            torch.save(G.state_dict(), CK / f"{a.name}.pt")
            (CK / f"{a.name}.json").write_text(json.dumps(meta, indent=1))
    G.eval()
    meta["offset"] = calibrate(G, meta) if a.mode == "n2n" else {}
    (CK / f"{a.name}.json").write_text(json.dumps(meta, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
