"""Round 2, part 2: one shared-trunk network for several radar bands.

One CondGen (2x, the round 1 residual trunk) trained jointly on:
  X-band: synthetic Sentinel-1-like 10 m input -> X-band 5 m target (round 1 pipeline, with band augmentation)
  L-band: NISAR 20 MHz GCOV HH (20 m) -> 40 MHz GCOV HH (10 m), real pairs, exp 3 validation frames excluded
  (C-band Stripmap joins when Experiment 2's tiles are verified.)
Each step takes one X-band batch and one NISAR batch (16 each); 3000 steps give NISAR the same exposure as
the per-band NISAR model at 100 % (ft_scratch_f1.0_s0).

Embeddings (FiLM, 6 numbers): input sensor (X-derived S1-like, NISAR 20 MHz, real Sentinel-1 IW) and target
sensor (X-band spotlight, NISAR 40 MHz, Sentinel-1 Stripmap). Separating the target sensor is the explicit
point-spread-function handling: the X-band spotlight look (sharp sidelobe crosses on bright points) belongs to
the "X target" code, so at inference the network can be asked for a sidelobe-free target style instead.
  --noembed 1   ablation: same data and schedule, conditioning vector all zeros (a plain shared trunk)
"""
from __future__ import annotations

import common  # noqa: F401
from common import DEV, EXP3_VAL_FRAMES, ROOT, CondGen, finish, nisar_tiles, sm_tiles

import argparse
import time

import numpy as np
import pandas as pd
import torch

import e5_train as T
import e5r2_finetune as FT

IN = {"xsyn": 0, "nisar20": 1, "s1iw": 2}
TG = {"xspot": 3, "nisar40": 4, "s1sm": 5}


def code(inp, tgt, n=6):
    v = np.zeros(n, "float32")
    v[IN[inp]] = 1
    v[TG[tgt]] = 1
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--noembed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sm", type=int, default=0, help="round 3: add exp 2 IW -> Stripmap pairs (input s1iw, target s1sm)")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(0)
    xt = T.prepare(T.load_x_tiles(), 2, "X")
    nt = nisar_tiles("train", frames_out=EXP3_VAL_FRAMES)
    val = nisar_tiles("train", frames_in=EXP3_VAL_FRAMES)
    st = sm_tiles(["train"]) if a.sm else []
    print(f"{a.name}: {len(xt)} X-band tiles, {len(nt)} NISAR tiles, {len(st)} Stripmap tiles, embed={not a.noembed}", flush=True)
    G = CondGen(scale=2, cond=True, nmeta=6).to(DEV)
    opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler()
    cx = torch.from_numpy(np.tile(code("xsyn", "xspot"), (a.bs, 1))).to(DEV)
    cn = torch.from_numpy(np.tile(code("nisar20", "nisar40"), (a.bs, 1))).to(DEV)
    cs = torch.from_numpy(np.tile(code("s1iw", "s1sm"), (a.bs, 1))).to(DEV)
    if a.noembed:
        cx, cn, cs = torch.zeros_like(cx), torch.zeros_like(cn), torch.zeros_like(cs)
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        s, y, m, _ = T.batch(xt, a.bs, rng, 2)
        s = T.band_aug(s)
        xn, yn, mn = FT.batch(nt, a.bs, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            px = G(s, cx)
            pn = G(xn, cn)
        lx = (torch.abs(px.float() - y) * m).sum() / m.sum().clamp(min=1)
        ln = (torch.abs(pn.float() - yn) * mn).sum() / mn.sum().clamp(min=1)
        loss = lx + ln
        ls = torch.zeros(())
        if a.sm:
            xs_, ys_, ms_ = FT.batch(st, a.bs, rng)
            with torch.autocast("cuda", dtype=torch.float16):
                ps = G(xs_, cs)
            ls = (torch.abs(ps.float() - ys_) * ms_).sum() / ms_.sum().clamp(min=1)
            loss = loss + ls
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0:
            log.append({"step": step, "l1_x": round(lx.item(), 4), "l1_nisar": round(ln.item(), 4), "l1_sm": round(float(ls), 4), "sec": round(time.time() - t0)})
            print(log[-1], flush=True)
    G.eval()
    with torch.no_grad():
        vl = float(np.mean([np.abs(G(torch.from_numpy(t["x"])[None, None].to(DEV), cn[:1])[0, 0].cpu().numpy() - t["y"])[t["m"]].mean() for t in val]))
    out = ROOT / "ckpt" / ("r3" if a.sm else "r2")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": G.state_dict(), "scale": 2, "cond": True, "nmeta": 6, "noembed": a.noembed,
                "steps": a.steps, "bs": a.bs, "val_l1": vl, "sm": a.sm}, out / f"{a.name}.pt")
    pd.DataFrame(log).to_csv(out / f"{a.name}_log.csv", index=False)
    print("saved, NISAR val L1", round(vl, 4), flush=True)


if __name__ == "__main__":
    main()
    finish()
