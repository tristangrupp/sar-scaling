"""Round 4 fine-tune: Experiment 2's arm A recipe, unchanged, with a seed, a training-tile share and an
optional pretrained start. The loop is exp 2 train_sr.main() line for line (SSIM loss on the Lee-filtered,
histogram-matched Stripmap, input consistency 0.15, 48 -> 96 px patches, rot/flip, min-max norm fitted on
the full training split, OneCycleLR, grad clip 1, best-validation checkpoint every 1000 steps), importing
its functions so nothing drifts. Checkpoints go to exp5_cross_band/ckpt/r4/.

  python e5r4_train.py --model esrt --name r4_esrt_x_f1_s1 --seed 1 --frac 1.0 --init ckpt/r4/pre_esrt_x.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sr_data import Norm, load_manifest, load_split  # noqa: E402
from train_sr import DEV, Sampler, make_model, masked_mean, ssim_map, val_ssim  # noqa: E402

CK = HERE / "ckpt" / "r4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["esrt", "cnn"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="path to a pretrained state (ckpt/r4/pre_*.pt)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default: exp 2 arm A (esrt 1e-3, cnn 5e-4)")
    ap.add_argument("--cons", type=float, default=0.15)
    ap.add_argument("--val-every", type=int, default=1000)
    a = ap.parse_args()
    lr = a.lr or (1e-3 if a.model == "esrt" else 5e-4)
    target = "lee_hm"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    man = load_manifest()
    tr_all = load_split(["train"], man)
    va = load_split(["val"], man)
    norm = Norm.fit(tr_all)                     # arm A's normalization, from the full training split
    if a.frac < 1:
        idx = np.random.default_rng(1000 + a.seed).permutation(len(tr_all))[:max(4, int(round(a.frac * len(tr_all))))]
        tr = [tr_all[i] for i in sorted(idx)]
    else:
        tr = tr_all
    print(f"{a.name}: train tiles {len(tr)} of {len(tr_all)}  val {len(va)}  norm [{norm.lo:.2f}, {norm.hi:.2f}]  "
          f"init {a.init}  lr {lr}", flush=True)
    samp = Sampler(tr, norm, target, seed=a.seed)
    model = make_model(a.model).to(DEV)
    if a.init:
        model.load_state_dict(torch.load(HERE / a.init, map_location=DEV)["state"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.steps)
    CK.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], -1.0, time.time()
    for step in range(1, a.steps + 1):
        s, y, m, d, w = samp.batch(a.bs, [], 0.0)
        p = model(s)
        l_ssim = 1 - masked_mean(ssim_map(p, y), m)
        loss = (1 - a.cons) * l_ssim
        l_cons = torch.zeros((), device=DEV)
        if a.cons > 0:
            m10 = F.avg_pool2d(m, 2) > 0.99
            l_cons = masked_mean((F.avg_pool2d(p, 2) - s) ** 2, m10.float())
            loss = loss + a.cons * l_cons
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            rec = {"step": step, "ssim_loss": round(l_ssim.item(), 4), "cons": round(l_cons.item(), 5), "sec": round(time.time() - t0)}
            if step % a.val_every == 0 or step == a.steps:
                rec["val_ssim"] = round(val_ssim(model, va, norm, target), 4)
                if rec["val_ssim"] > best:
                    best = rec["val_ssim"]
                    torch.save({"state": model.state_dict(), "model": a.model, "norm": [norm.lo, norm.hi],
                                "args": vars(a), "step": step}, CK / f"{a.name}.pt")
            log.append(rec)
            print(rec, flush=True)
    pd.DataFrame(log).to_csv(CK / f"{a.name}_log.csv", index=False)
    (CK / f"{a.name}.json").write_text(json.dumps({"best_val_ssim": best, "n_train": len(tr), **vars(a), "lr_used": lr}, indent=1))
    print("done best val ssim", best, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
