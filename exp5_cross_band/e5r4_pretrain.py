"""Round 4 pretraining with arm A's loss and schedule, on synthetic X-band (x) or X-band + NISAR L-band (xl).

Same model code (exp 2 make_model), same loss (1 - SSIM on the Lee-filtered, histogram-matched target, plus
0.15 x input consistency), same patches (48 -> 96, rot/flip), same min-max normalization (arm A's bounds),
same OneCycleLR and learning rate per architecture. Budget: 8000 steps at batch 16 for every variant;
xl draws 8 X-band and 8 NISAR patches per batch, so both variants see the same number of samples.
No Stripmap and no real IW enter pretraining. Saves the last state to ckpt/r4/pre_<model>_<src>.pt.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sr_data import Norm  # noqa: E402
from train_sr import DEV, LR_P, make_model, masked_mean, ssim_map  # noqa: E402

PRE = HERE / "data" / "r4_pre"


def load(src):
    out = []
    for f in sorted(glob.glob(str(PRE / src / "*.npz"))):
        z = np.load(f)
        ins = [z[k].astype("float32") for k in ("iw0", "iw1", "iw2") if k in z.files]
        out.append({"ins": ins, "y": z["lee_hm"].astype("float32"), "m": z["mask"]})
    return out


def sample(tiles, n, norm, rng):
    S, Y, M = [], [], []
    for _ in range(n):
        t = tiles[rng.integers(len(tiles))]
        s_full = t["ins"][rng.integers(len(t["ins"]))]
        L = s_full.shape[0]
        for _ in range(50):
            r, c = rng.integers(0, L - LR_P, 2)
            m = t["m"][r * 2:(r + LR_P) * 2, c * 2:(c + LR_P) * 2]
            if m.mean() >= 0.8:
                break
        s = s_full[r:r + LR_P, c:c + LR_P]
        y = t["y"][r * 2:(r + LR_P) * 2, c * 2:(c + LR_P) * 2]
        k, fl = rng.integers(4), rng.random() < 0.5
        arrs = []
        for a_ in (s, y, m):
            a_ = np.rot90(a_, k)
            arrs.append(a_[:, ::-1] if fl else a_)
        s, y, m = arrs
        S.append(np.nan_to_num(norm(s), nan=0.0))
        Y.append(np.nan_to_num(norm(y), nan=0.0))
        M.append(m & np.isfinite(y))
    f = lambda L_: torch.from_numpy(np.stack(L_).astype("float32"))[:, None].to(DEV)  # noqa: E731
    return f(S), f(Y), f(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["esrt", "cnn"], required=True)
    ap.add_argument("--src", choices=["x", "xl"], required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    lr = 1e-3 if a.model == "esrt" else 5e-4
    ck = torch.load(EXP2 / "ckpt" / f"{a.model}_a.pt", map_location="cpu", weights_only=False)
    norm = Norm(*ck["norm"])
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    xt = load("x")
    lt = load("l") if a.src == "xl" else []
    print(f"pre_{a.model}_{a.src}: {len(xt)} X-band tiles, {len(lt)} NISAR tiles, norm [{norm.lo:.2f}, {norm.hi:.2f}], lr {lr}", flush=True)
    model = make_model(a.model).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.steps)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        if a.src == "x":
            s, y, m = sample(xt, a.bs, norm, rng)
        else:
            s1, y1, m1 = sample(xt, a.bs // 2, norm, rng)
            s2, y2, m2 = sample(lt, a.bs // 2, norm, rng)
            s, y, m = torch.cat([s1, s2]), torch.cat([y1, y2]), torch.cat([m1, m2])
        p = model(s)
        l_ssim = 1 - masked_mean(ssim_map(p, y), m)
        m10 = F.avg_pool2d(m, 2) > 0.99
        l_cons = masked_mean((F.avg_pool2d(p, 2) - s) ** 2, m10.float())
        loss = 0.85 * l_ssim + 0.15 * l_cons
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print({"step": step, "ssim_loss": round(l_ssim.item(), 4), "cons": round(l_cons.item(), 5), "sec": round(time.time() - t0)}, flush=True)
    (HERE / "ckpt" / "r4").mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "model": a.model, "norm": [norm.lo, norm.hi], "src": a.src,
                "steps": a.steps, "bs": a.bs, "n_x": len(xt), "n_l": len(lt)}, HERE / "ckpt" / "r4" / f"pre_{a.model}_{a.src}.pt")
    print("saved", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
