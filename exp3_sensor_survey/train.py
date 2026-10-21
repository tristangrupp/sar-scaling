"""Train arm A on NISAR simultaneous pairs: coarse mode -> fine mode of the same datatake.

  --arm x2   20 MHz mixed-mode GCOV (20 m) -> 40 MHz GCOV (10 m)       [main]
  --arm x4   5 MHz frequency B GCOV (80 m) -> 20 MHz GCOV (20 m)         [scale control]

Losses (HH, gamma0 dB, one fixed global normalisation so brightness stays physical):
  masked L1 against the fine product
  input consistency: the output, averaged back to the input grid in linear power, must match the input
Generator: EDSR-style residual CNN from Experiment 1, one pixel-shuffle step per factor 2, bicubic skip.
"""
from __future__ import annotations
from config import DATA  # noqa: E402
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
TILES = Path(str(DATA) + "/tiles")
DEV = os.environ.get("EXP3_DEVICE", "cuda")   # round 3: EXP3_DEVICE=cpu for evaluation while the GPU is busy
MU, SD = -12.0, 6.0                         # dB -> network units
ARMS = {"x2": ("m20_HH", "a10_HH", 2), "x4": ("b80_HH", "m20_HH", 4),
        "x2multi": ("m20_HH", "mt10", 2),        # round 2: multi-date 40 MHz target (DATA/extra/mt10/<tile>.npz)
        "g2": ("a10_HH", "g5", 2)}               # round 2: 10 m GCOV -> 5 m GSLC multi-date (DATA/extra/g5/<tile>.npz)


def to_n(d):
    return (d - MU) / SD


def from_n(n):
    return n * SD + MU


def frame_of(name):
    t, f = name.split("_")[:2]
    return int(t), int(f)


def load(role, arm, bad=(), frames_in=None, frames_out=None, target=None):
    """role: tile folder; frames_in / frames_out: keep only / drop these (track, frame); target overrides the arm's"""
    ki, kt, s = ARMS[arm]
    kt = target or kt
    out = []
    for f in sorted(glob.glob(str(TILES / role / "*.npz"))):
        name = Path(f).stem
        if name in bad:
            continue
        fr = frame_of(name)
        if (frames_in is not None and fr not in frames_in) or (frames_out is not None and fr in frames_out):
            continue
        z = np.load(f)
        x = z[ki].astype("float32")
        if kt in z.files:
            y = z[kt].astype("float32")
            mask = np.isfinite(y) & (z["a10_mask"] != 255) if kt == "a10_HH" else np.isfinite(y)
        else:
            ef = DATA / "extra" / kt / f"{name}.npz"
            if not ef.exists():
                continue
            e = np.load(ef)
            y = e["y"].astype("float32")
            mask = np.isfinite(y) & (e["m"] if "m" in e.files else True)
            if mask.mean() < 0.5:
                continue
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        out.append({"id": name, "x": to_n(x), "y": to_n(np.nan_to_num(y, nan=np.nanmedian(y))), "m": mask,
                    "meta": json.loads(str(z["meta"]))})
    return out


class Patches(torch.utils.data.Dataset):
    def __init__(self, tiles, n, lr, s):
        self.t, self.n, self.lr, self.s = tiles, n, lr, s
        self.rng = np.random.default_rng(np.random.randint(2**31))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        for _ in range(20):
            t = self.t[self.rng.integers(len(self.t))]
            L = t["x"].shape[0]
            r, c = self.rng.integers(0, L - self.lr + 1, 2)
            m = t["m"][r * self.s:(r + self.lr) * self.s, c * self.s:(c + self.lr) * self.s]
            if m.mean() > 0.95:
                break
        x = t["x"][r:r + self.lr, c:c + self.lr]
        y = t["y"][r * self.s:(r + self.lr) * self.s, c * self.s:(c + self.lr) * self.s]
        # flips only along axes, no 90 deg rotation: range and azimuth resolution differ (anisotropic PSF)
        if self.rng.random() < 0.5:
            x, y, m = x[:, ::-1], y[:, ::-1], m[:, ::-1]
        if self.rng.random() < 0.5:
            x, y, m = x[::-1], y[::-1], m[::-1]
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype="float32"))[None]
        return f(x), f(y), f(m)


class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + 0.2 * self.b(x)


class Gen(nn.Module):
    def __init__(self, scale, c=64, nb=16):
        super().__init__()
        self.s = scale
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
        up = []
        for _ in range(int(np.log2(scale))):
            up += [nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(True)]
        self.up = nn.Sequential(*up, nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x):
        h = self.head(x)
        h = h + self.body(h)
        return self.up(h) + F.interpolate(x, scale_factor=self.s, mode="bicubic", align_corners=False)


def pool_n(p, s):
    """network units -> linear power -> average over s x s -> back to network units"""
    lin = 10 ** (from_n(p) / 10)
    return to_n(10 * torch.log10(F.avg_pool2d(lin, s).clamp(min=1e-8)))


class XPatches(torch.utils.data.Dataset):
    """arm B: random 64 px (20 m) crops of the X-band training windows, with the X-band edge distance map"""

    def __init__(self, files, n, lr=64, s=2):
        self.d = [np.load(f) for f in files]
        self.d = [{k: z[k] for k in ["x", "y", "dt", "infra"]} for z in self.d]
        self.n, self.lr, self.s = n, lr, s
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        for _ in range(20):
            z = self.d[self.rng.integers(len(self.d))]
            L = z["x"].shape[0]
            r, c = self.rng.integers(0, L - self.lr + 1, 2)
            sl = np.s_[r * self.s:(r + self.lr) * self.s, c * self.s:(c + self.lr) * self.s]
            if z["infra"][sl].mean() > 0.05:
                break
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype="float32"))[None]
        x = to_n(z["x"][r:r + self.lr, c:c + self.lr].astype("float32"))
        y = to_n(z["y"][sl].astype("float32"))
        return f(x), f(y), f(z["dt"][sl]), f(z["infra"][sl])


SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8


def edge_loss(p, dt, infra, tol=1.0, cap=5.0):
    """tolerant X-band edge loss: output gradient energy weighted by its distance (px) beyond `tol` from the
    nearest X-band edge, capped at `cap`; infrastructure pixels only (built-up, X-band valid, no layover/shadow)"""
    k = SOBEL.to(p.device)
    gx = F.conv2d(p, k[None, None], padding=1)
    gy = F.conv2d(p, k.t()[None, None], padding=1)
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6) * infra
    d = (dt - tol).clamp(min=0, max=cap - tol) / (cap - tol)
    return (g * d).sum() / g.sum().clamp(min=1e-3)


class Disc(nn.Module):
    """round 4: conditional PatchGAN (spectral norm, LSGAN) on [bicubic input, image] at the output grid,
    as in Experiment 1. Receptive field about 34 px (340 m at 10 m)."""

    def __init__(self, s, c=48):
        super().__init__()
        self.s = s
        sn = nn.utils.spectral_norm
        L, cin = [], 2
        for i, co in enumerate([c, 2 * c, 4 * c]):
            L += [sn(nn.Conv2d(cin, co, 4, stride=2 if i < 2 else 1, padding=1)), nn.LeakyReLU(0.2, True)]
            cin = co
        L += [sn(nn.Conv2d(cin, 1, 4, padding=1))]
        self.net = nn.Sequential(*L)

    def forward(self, x, img):
        return self.net(torch.cat([F.interpolate(x, scale_factor=self.s, mode="bicubic", align_corners=False), img], 1))


def grad_l1(p, t, m):
    """round 3: L1 between Sobel gradients of prediction and target (x and y), on pixels valid in the target"""
    k = SOBEL.to(p.device)
    kx, ky = k[None, None], k.t()[None, None]
    d = (torch.abs(F.conv2d(p, kx, padding=1) - F.conv2d(t, kx, padding=1)) +
         torch.abs(F.conv2d(p, ky, padding=1) - F.conv2d(t, ky, padding=1)))
    mm = -F.max_pool2d(-m, 3, 1, 1)          # erode the mask by one pixel so gradients never touch invalid pixels
    return (d * mm).sum() / mm.sum().clamp(min=1)


def masked_l1(p, t, m):
    return (torch.abs(p - t) * m).sum() / m.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="x2", choices=list(ARMS))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cons", type=float, default=0.5, help="input-consistency weight")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cons_at", type=int, default=0, help="consistency grid in metres (0 = the input grid)")
    ap.add_argument("--val_out", type=int, default=0, help="1 = drop config.VAL_FRAMES from training")
    ap.add_argument("--target", default=None, help="override the arm's target key")
    ap.add_argument("--name", default=None)
    ap.add_argument("--edge_w", type=float, default=0.0, help="arm B: weight of the X-band edge loss (0 = arm A)")
    ap.add_argument("--xbs", type=int, default=8, help="arm B: X-band windows per step")
    ap.add_argument("--grad_w", type=float, default=0.0, help="round 3: weight of the gradient-L1 term")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init", default=None, help="round 4: start the generator from ckpt/<init>.pt")
    ap.add_argument("--adv_w", type=float, default=0.0, help="round 4: weight of the adversarial (LSGAN) term")
    ap.add_argument("--d_lr", type=float, default=1e-4)
    ap.add_argument("--xdata", type=int, default=0, help="1 = also train on the X-band windows (L1 + consistency) even with edge_w 0")
    a = ap.parse_args()
    ki, kt, s = ARMS[a.arm]
    name = a.name or (a.arm + a.tag)
    bad = set(json.loads((ROOT / "bad_tiles.json").read_text())) if (ROOT / "bad_tiles.json").exists() else set()
    torch.backends.cudnn.benchmark = True
    if a.seed is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
    from config import VAL_FRAMES
    tiles = load("train", a.arm, bad, frames_out=set(VAL_FRAMES) if a.val_out else None, target=a.target)
    in_px = {"x2": 20, "x4": 80, "x2multi": 20, "g2": 10}[a.arm]
    cf = 1 if not a.cons_at else a.cons_at // in_px     # extra pooling of input and output for the consistency grid
    print(f"{name}: train tiles {len(tiles)}", flush=True)
    lr_px = 64 if s == 2 else 32
    dl = torch.utils.data.DataLoader(Patches(tiles, a.steps * a.bs, lr_px, s), batch_size=a.bs, num_workers=0)
    G = Gen(s).to(DEV)
    if a.init:
        G.load_state_dict(torch.load(ROOT / f"ckpt/{a.init}.pt", weights_only=True, map_location=DEV))
        print("init from", a.init, flush=True)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    D = optD = None
    if a.adv_w > 0:
        D = Disc(s).to(DEV)
        optD = torch.optim.Adam(D.parameters(), lr=a.d_lr, betas=(0.5, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    (ROOT / "ckpt").mkdir(exist_ok=True)
    xit = None
    if a.edge_w > 0 or a.xdata:
        xf = sorted(glob.glob(str(DATA / "xtrain" / "xtrain" / "*.npz")))
        print(f"arm B: {len(xf)} X-band training windows, edge weight {a.edge_w}", flush=True)
        xit = iter(torch.utils.data.DataLoader(XPatches(xf, a.steps * a.xbs, 64, s), batch_size=a.xbs, num_workers=0))
    t0, log = time.time(), []
    for step, (x, y, m) in enumerate(dl, 1):
        x, y, m = x.to(DEV), y.to(DEV), m.to(DEV)
        p = G(x)
        l1 = masked_l1(p, y, m)
        if cf == 1:
            cons = torch.abs(pool_n(p, s) - x).mean()
        else:
            cons = torch.abs(pool_n(p, s * cf) - pool_n(x, cf)).mean()
        loss = l1 + a.cons * cons
        lg = grad_l1(p, y, m) if a.grad_w > 0 else torch.zeros(())
        loss = loss + a.grad_w * lg
        ladv = ld = torch.zeros(())
        if D is not None:
            # D step: real = target with invalid pixels filled by the prediction, so D cannot key on holes
            yr = y * m + p.detach() * (1 - m)
            ld = ((D(x, yr) - 1) ** 2).mean() + (D(x, p.detach()) ** 2).mean()
            optD.zero_grad(set_to_none=True)
            ld.backward()
            optD.step()
            ladv = ((D(x, p) - 1) ** 2).mean()
            loss = loss + a.adv_w * ladv
        le = torch.zeros(())
        if xit is not None:
            xx, yy, dt, inf = [q.to(DEV) for q in next(xit)]
            px_ = G(xx)
            le = edge_loss(px_, dt, inf)
            lx = torch.abs(px_ - yy).mean()
            if cf == 1:
                cx = torch.abs(pool_n(px_, s) - xx).mean()
            else:
                cx = torch.abs(pool_n(px_, s * cf) - pool_n(xx, cf)).mean()
            loss = loss + lx + a.cons * cx + a.edge_w * le
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            rec = {"step": step, "l1": round(l1.item(), 4), "cons": round(cons.item(), 4), "edge": round(le.item(), 4), "grad": round(lg.item(), 4), "adv": round(ladv.item(), 4), "lD": round(ld.item(), 4), "sec": round(time.time() - t0)}
            log.append(rec)
            print(rec, flush=True)
        if step >= a.steps:
            break
    torch.save(G.state_dict(), ROOT / f"ckpt/{name}.pt")
    pd.DataFrame(log).to_csv(ROOT / f"ckpt/{name}_log.csv", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
