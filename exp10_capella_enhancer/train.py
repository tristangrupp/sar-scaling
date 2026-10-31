"""Train the Sentinel-1 enhancer against degraded Capella at one target resolution (DESIGN.md, "Model").

  python train.py --res 5 --seed 0                     # GPU run (waits for and holds gpu_secondary.lock)
  python train.py --res 5 --device cpu --smoke         # CPU smoke test: tiny subset, a few steps, no lock

Inputs: 5 channels at 10 m (VV, VH single date; VV, VH multi-date mean; slope), scaled to [0, 1] with fixed bounds
from the training chips (ckpt/norm.json). Output: Capella dB at 2.5 m (x4), same scaling.
Loss: 0.16 L1 + 0.84 (1 - SSIM) on valid target pixels. AdamW, cosine decay with warmup.
Checkpoints: ckpt/<name>.pt (best validation SSIM), ckpt/<name>_last.pt (resume point, written at every
validation). Logs: logs/<name>.log and ckpt/<name>_log.csv. Rerunning the same command resumes from _last.pt.

GPU etiquette (project convention, see exp2_iw_to_stripmap/gpu_job.py): one SAR GPU job at a time. On cuda the
script waits while E:/SAR_superresolution/gpu_secondary.lock exists, then creates it (O_EXCL) with its name and pid
and deletes it on exit. It never starts without a coordinator slot: launch it only after the lead arranges one.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CHIPS, HERE as HEREP, PROJ, RES_LIST, SC, crop_c, res_index, scores  # noqa: E402

sys.path.insert(0, str(PROJ / "exp2_iw_to_stripmap"))
from esrt import ESRT  # noqa: E402

CKPT = HEREP / "ckpt"
LOGS = HEREP / "logs"
LOCK = PROJ / "gpu_secondary.lock"
NCH = 5


# ------------------------------------------------------------------ models
class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + 0.2 * self.b(x)


class CNN4(nn.Module):
    """Experiment 2's EDSR-style residual CNN, x4 by two pixel-shuffle steps, multi-channel input; the skip is a
    learned 1x1 mix of the bicubic-upsampled input channels."""

    def __init__(self, cin=NCH, c=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(cin, c, 3, padding=1)
        self.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv2d(c, c, 3, padding=1))
        self.up = nn.Sequential(nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(True),
                                nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(True),
                                nn.Conv2d(c, 1, 3, padding=1))
        self.skip = nn.Conv2d(cin, 1, 1)

    def forward(self, s):
        h = self.head(s)
        h = h + self.body(h)
        return self.up(h) + self.skip(F.interpolate(s, scale_factor=SC, mode="bicubic", align_corners=False))


def make_model(arch):
    if arch == "esrt":
        return ESRT(upscale=SC, in_ch=NCH, out_ch=1)
    return CNN4()


# ------------------------------------------------------------------ data
class Norm:
    def __init__(self, d):
        self.lo = np.array(d["in_lo"], "float32")[:, None, None]
        self.hi = np.array(d["in_hi"], "float32")[:, None, None]
        self.tlo = {float(k): v for k, v in d["t_lo"].items()}
        self.thi = {float(k): v for k, v in d["t_hi"].items()}

    def x(self, s1):
        return np.nan_to_num((s1 - self.lo) / (self.hi - self.lo), nan=0.0).astype("float32")

    def t(self, t, R):
        return (t - self.tlo[R]) / (self.thi[R] - self.tlo[R])

    def t_inv(self, y, R):
        return y * (self.thi[R] - self.tlo[R]) + self.tlo[R]


def fit_norm(ch):
    """0.1 / 99.9 percentiles of each input channel and of each target resolution over (a sample of) train chips."""
    rng = np.random.default_rng(0)
    xs, ts = [[] for _ in range(NCH)], {R: [] for R in RES_LIST}
    for cid, g in ch.groupby("capella_id"):
        z = np.load(CHIPS / f"{cid}.npz")
        idx = [list(z["chip_id"]).index(c) for c in g.chip_id]
        s1 = z["s1"][idx].astype("float32")
        tg = z["tgt"][idx].astype("float32")
        for c in range(NCH):
            v = s1[:, c][np.isfinite(s1[:, c])]
            xs[c].append(rng.choice(v, min(len(v), 4000), replace=False))
        for i, R in enumerate(RES_LIST):
            v = tg[:, i][np.isfinite(tg[:, i])]
            ts[R].append(rng.choice(v, min(len(v), 8000), replace=False))
    q = lambda L: np.percentile(np.concatenate(L), [0.1, 99.9]).tolist()  # noqa: E731
    xin = [q(L) for L in xs]
    return {"in_lo": [a[0] for a in xin], "in_hi": [a[1] for a in xin],
            "t_lo": {str(R): q(ts[R])[0] for R in RES_LIST}, "t_hi": {str(R): q(ts[R])[1] for R in RES_LIST},
            "channels": ["vv1_db", "vh1_db", "vvm_db", "vhm_db", "slope_deg"], "n_chips": int(len(ch))}


def load_split(ch, R, norm):
    """Arrays in RAM: x (n, 5, 64, 64) normalized, y (n, 256, 256) normalized target, m (n, 256, 256) loss mask."""
    ri = res_index(R)
    X, Y, M, ids = [], [], [], []
    for cid, g in ch.groupby("capella_id"):
        z = np.load(CHIPS / f"{cid}.npz")
        pos = {c: i for i, c in enumerate(z["chip_id"])}
        idx = [pos[c] for c in g.chip_id]
        s1 = z["s1"][idx].astype("float32")
        t = z["tgt"][idx, ri].astype("float32")
        up = np.repeat(np.repeat(z["valid10"][idx], SC, axis=1), SC, axis=2)
        m = z["tvalid"][idx] & up & np.isfinite(t)
        X.append(np.stack([norm.x(a) for a in s1]))
        Y.append(np.nan_to_num(norm.t(t, R), nan=0.0).astype("float32"))
        M.append(m)
        ids += list(g.chip_id)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M), ids


class Sampler:
    def __init__(self, X, Y, M, crop, seed):
        self.X, self.Y, self.M, self.c = X, Y, M, crop
        self.rng = np.random.default_rng(seed)

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, s):
        self.rng.bit_generator.state = s

    def batch(self, bs, dev):
        n, c = len(self.X), self.c
        xs, ys, ms = [], [], []
        while len(xs) < bs:
            i = int(self.rng.integers(n))
            r, q = self.rng.integers(0, 64 - c + 1, 2)
            m = self.M[i, r * SC:(r + c) * SC, q * SC:(q + c) * SC]
            if m.mean() < 0.5:
                continue
            x = self.X[i, :, r:r + c, q:q + c]
            y = self.Y[i, r * SC:(r + c) * SC, q * SC:(q + c) * SC]
            k = int(self.rng.integers(4))
            x, y, m = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(m, k)
            if self.rng.random() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
            xs.append(x.copy())
            ys.append(y.copy())
            ms.append(m.copy())
        f = lambda L: torch.from_numpy(np.stack(L).astype("float32")).to(dev)  # noqa: E731
        return f(xs), f(ys)[:, None], f(ms)[:, None]


# ------------------------------------------------------------------ loss
def _gauss(win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[None]
    return (g.T @ g)[None, None]


_G = {}


def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    if x.device not in _G:
        _G[x.device] = _gauss().to(x.device)
    G = _G[x.device]
    f = lambda a: F.conv2d(a, G, padding=5)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))


def loss_fn(p, y, m, w_l1=0.16):
    p, y = p.float(), y.float()
    den = m.sum().clamp(min=1)
    l1 = (torch.abs(p - y) * m).sum() / den
    ss = (ssim_map(p, y) * m).sum() / den
    return w_l1 * l1 + (1 - w_l1) * (1 - ss), l1.detach(), ss.detach()


# ------------------------------------------------------------------ inference / validation
@torch.no_grad()
def predict_chips(model, X, dev, bs=16, amp=False):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            y = model(x)
        out.append(y[:, 0].float().cpu().numpy())
    model.train()
    return np.concatenate(out)


@torch.no_grad()
def predict_large(model, s1n, dev, win=64, margin=8, amp=False):
    """Tiled inference on a normalized (5, H, W) array; keeps each window's center."""
    model.eval()
    _, H, W = s1n.shape
    step = win - 2 * margin
    out = np.zeros((H * SC, W * SC), "float32")
    rows = list(range(0, max(H - win, 0), step)) + [max(H - win, 0)]
    cols = list(range(0, max(W - win, 0), step)) + [max(W - win, 0)]
    x = torch.from_numpy(s1n)
    for r in rows:
        for c in cols:
            b = x[:, r:r + win, c:c + win][None].to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                y = model(b)[0, 0].float().cpu().numpy()
            r0 = 0 if r == 0 else margin
            c0 = 0 if c == 0 else margin
            out[(r + r0) * SC:(r + win) * SC, (c + c0) * SC:(c + win) * SC] = y[r0 * SC:, c0 * SC:]
    model.train()
    return out


def validate(model, V, norm, R, dev, amp):
    X, Y, M, _ = V
    P = predict_chips(model, X, dev, amp=amp)
    ss, hp = [], []
    for p, y, m in zip(P, Y, M):
        s = scores(crop_c(norm.t_inv(p, R)), crop_c(norm.t_inv(y, R)), crop_c(m), light=True)
        if s:
            ss.append(s["ssim"])
            hp.append(s["hp_r"])
    return float(np.median(ss)), float(np.median(hp))


# ------------------------------------------------------------------ lock
def take_lock(name, log):
    msg = False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"exp10 {name} pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M')}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            if not msg:
                log(f"waiting for {LOCK} ({LOCK.read_text(errors='ignore').strip()[:120]})")
                msg = True
            time.sleep(60)

    def release():
        try:
            if LOCK.exists() and f"pid={os.getpid()}" in LOCK.read_text(errors="ignore"):
                LOCK.unlink()
        except Exception:
            pass
    atexit.register(release)
    log("lock taken")
    return release


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, required=True, choices=list(RES_LIST))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["esrt", "cnn"], default="esrt")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-chips", type=int, default=300, help="cap on validation chips scored during training")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU check: 40 train / 20 val chips, few steps, no lock")
    a = ap.parse_args()
    name = a.name or f"{a.arch}_r{a.res:g}_s{a.seed}" + ("_smoke" if a.smoke else "")
    CKPT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    logf = open(LOGS / f"{name}.log", "a", buffering=1)

    def log(s):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")

    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        log("cuda requested but not available; refusing to fall back to CPU for a full run")
        return
    amp = dev.type == "cuda" and not a.no_amp
    if dev.type == "cuda" and not a.smoke:
        take_lock(name, log)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ch = pd.read_csv(HEREP / "chips.csv")
    tr, va = ch[ch.split == "train"], ch[ch.split == "val"]
    if a.smoke:
        if len(va) == 0:   # early partial builds may have no validation sites yet
            va = ch[ch.split != "train"]
        tr = tr.groupby("capella_id").head(4).head(40)
        va = va.groupby("capella_id").head(4).head(20)
    nf = CKPT / ("norm_smoke.json" if a.smoke else "norm.json")
    if not nf.exists():
        nf.write_text(json.dumps(fit_norm(tr), indent=1))
    norm = Norm(json.loads(nf.read_text()))
    vsub = va.sample(min(len(va), a.val_chips), random_state=0) if len(va) > a.val_chips else va
    T = load_split(tr, a.res, norm)
    V = load_split(vsub, a.res, norm)
    log(f"{name}: R={a.res} train {len(T[0])} chips ({tr.site_id.nunique()} sites), val {len(V[0])} chips "
        f"({vsub.site_id.nunique()} sites), device {dev}, amp {amp}")

    model = make_model(a.arch).to(dev)
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    smp = Sampler(*T[:3], a.crop, a.seed)
    step, best, hist = 0, -1.0, []
    last = CKPT / f"{name}_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        smp.set_state(ck["sampler"])
        torch.set_rng_state(ck["torch_rng"])
        step, best, hist = ck["step"], ck["best"], ck["hist"]
        log(f"resumed from step {step}, best val SSIM {best:.4f}")

    t0 = time.time()
    run_l = []
    while step < a.steps:
        x, y, m = smp.batch(a.bs, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            p = model(x)
        loss, l1, ss = loss_fn(p, y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run_l.append([float(loss), float(l1), float(ss)])
        if step % 100 == 0 or step == a.steps or (a.smoke and step % 5 == 0):
            r = np.mean(run_l, axis=0)
            run_l = []
            log(f"step {step} loss {r[0]:.4f} l1 {r[1]:.4f} ssim {r[2]:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{(time.time() - t0) / 60:.1f} min")
        if step % a.val_every == 0 or step == a.steps:
            vs, vh = validate(model, V, norm, a.res, dev, amp)
            hist.append({"step": step, "val_ssim": vs, "val_hp_r": vh, "min": round((time.time() - t0) / 60, 2)})
            if vs > best:
                best = vs
                torch.save({"model": model.state_dict(), "arch": a.arch, "res": a.res, "seed": a.seed, "step": step,
                            "norm": json.loads(nf.read_text()), "val_ssim": vs, "args": vars(a)}, CKPT / f"{name}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "sampler": smp.state(), "torch_rng": torch.get_rng_state(),
                        "step": step, "best": best, "hist": hist}, last)
            pd.DataFrame(hist).to_csv(CKPT / f"{name}_log.csv", index=False)
            log(f"val step {step}: SSIM {vs:.4f} hp_r {vh:.4f} (best {best:.4f})")
    (CKPT / f"{name}.json").write_text(json.dumps({"name": name, "best_val_ssim": best, "hist": hist, **vars(a)},
                                                  indent=1))
    (LOGS / f"{name}.done").write_text("done\n")
    log(f"finished {name}: best val SSIM {best:.4f}, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
