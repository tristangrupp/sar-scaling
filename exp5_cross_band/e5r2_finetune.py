"""Round 2, part 1: does X-band pretraining help a same-band model, and most when real data is scarce?

Anchor: Experiment 3's NISAR same-datatake pairs (20 MHz GCOV HH 20 m -> 40 MHz GCOV HH 10 m, 2x).
  --init scratch       random initialization
  --init <ckpt name>   exp5_cross_band/ckpt/<name>.pt (e.g. x2_unc: X-band synthetic 2x, round 1), all weights
  --frac f             fraction of the NISAR training tiles (validation frames excluded), sampled per seed
Equal budget for every run: --steps (default 3000) at batch 16, Adam 2e-4 with cosine decay, masked L1,
flip augmentation only (NISAR's range and azimuth resolution differ, as in exp 3).
The validation frames give a held-aside L1 at the end (reported, not used for selection).
Writes ckpt/r2/<name>.pt and the log.
"""
from __future__ import annotations

import common  # noqa: F401
from common import DEV, EXP3_VAL_FRAMES, ROOT, CondGen, finish, nisar_tiles, sm_tiles

import argparse
import time

import numpy as np
import pandas as pd
import torch


def batch(tiles, bs, rng, lr=64, s=2):
    X, Y, M = [], [], []
    for _ in range(bs):
        for _ in range(20):
            t = tiles[rng.integers(len(tiles))]
            L = t["x"].shape[0]
            r, c = rng.integers(0, L - lr + 1, 2)
            m = t["m"][r * s:(r + lr) * s, c * s:(c + lr) * s]
            if m.mean() > 0.95:
                break
        x = t["x"][r:r + lr, c:c + lr]
        y = t["y"][r * s:(r + lr) * s, c * s:(c + lr) * s]
        if rng.random() < 0.5:
            x, y, m = x[:, ::-1], y[:, ::-1], m[:, ::-1]
        if rng.random() < 0.5:
            x, y, m = x[::-1], y[::-1], m[::-1]
        X.append(x.copy()); Y.append(y.copy()); M.append(m.copy())
    f = lambda a: torch.from_numpy(np.stack(a).astype("float32"))[:, None].to(DEV)
    return f(X), f(Y), f(M)


@torch.no_grad()
def val_l1(G, tiles):
    tot, n = 0.0, 0
    for t in tiles:
        p = G(torch.from_numpy(t["x"])[None, None].to(DEV))[0, 0].cpu().numpy()
        tot += float(np.abs(p - t["y"])[t["m"]].mean())
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="scratch")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--anchor", default="nisar", choices=["nisar", "sm"],
                    help="nisar: exp 3 L-band pairs (round 2); sm: exp 2 IW -> Stripmap pairs (round 3)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.anchor == "sm":
        allt, val = sm_tiles(["train"]), sm_tiles(["val"])
    else:
        allt = nisar_tiles("train", frames_out=EXP3_VAL_FRAMES)
    # nested subsets: the same seed's 10 % is contained in its 25 %, and so on
    order = np.random.default_rng(1000 + a.seed).permutation(len(allt))
    n = max(4, int(round(a.frac * len(allt))))
    tiles = [allt[i] for i in order[:n]]
    if a.anchor == "nisar":
        val = nisar_tiles("train", frames_in=EXP3_VAL_FRAMES)
    G = CondGen(scale=2, cond=False).to(DEV)
    if a.init != "scratch":
        ck = torch.load(ROOT / "ckpt" / f"{a.init}.pt", map_location=DEV, weights_only=False)
        G.load_state_dict({k: v for k, v in ck["state"].items() if not k.startswith("film")}, strict=False)
    v0 = val_l1(G.eval(), val)
    G.train()
    print(f"{a.name}: {len(tiles)} of {len(allt)} train tiles, {len(val)} val tiles, init {a.init}, val L1 at start {v0:.4f}", flush=True)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler()
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        x, y, m = batch(tiles, a.bs, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            p = G(x)
        loss = (torch.abs(p.float() - y) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            rec = {"step": step, "l1": round(loss.item(), 4), "sec": round(time.time() - t0)}
            if step % 1000 == 0:
                rec["val_l1"] = round(val_l1(G.eval(), val), 4)
                G.train()
            log.append(rec)
            print(rec, flush=True)
    out = ROOT / "ckpt" / ("r3" if a.anchor == "sm" else "r2")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": G.state_dict(), "scale": 2, "cond": False, "init": a.init, "frac": a.frac, "seed": a.seed,
                "n_tiles": len(tiles), "steps": a.steps, "anchor": a.anchor, "val_l1_start": v0, "val_l1": val_l1(G.eval(), val)},
               out / f"{a.name}.pt")
    pd.DataFrame(log).to_csv(out / f"{a.name}_log.csv", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
    finish()
