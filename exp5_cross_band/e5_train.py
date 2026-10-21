"""Train the generator on synthetic Sentinel-1-like degradations (no real Sentinel-1 involved).

  --source x      X-band 2.5 m tiles from the spike (train split only; held-out regions excluded)
                  plus extra train-region collects from the ceiling check's tiles.
  --source cband  C-band-only control: Experiment 4 temporal-mean Sentinel-1 VV (10 m, train sites
                  only), degraded with the same pipeline in pixel units (a scale-shift assumption:
                  10 m clean -> 40 m S1-like, then applied to real 10 m S1).
  --scale 4       10 m -> 2.5 m target;  --scale 2: 10 m -> 5 m target (2x2 power average of 2.5 m)
  --cond 1        FiLM conditioning on (look azimuth, incidence, polarization, band)

Each batch: random 176 px crops at the fine grid, random rotation / flip (azimuth follows),
degradation on the GPU with a fresh speckle draw (K looks, K uniform in 4..7 to bracket the
measured real ENL), center 128 px kept as target and the matching 32 px as input.
"""
from __future__ import annotations

import common  # noqa: F401
from common import (DEV, ROOT, SPIKE, CondGen, degrade, finish, meta_vec, aug_az, pool_db, tile_metadata,
                    heading_to_sensor_az, S1_TO_SENSOR_AZ)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

P, M = 128, 24          # fine-grid patch and margin (pixels at 2.5 m)


def test_points():
    man = pd.read_csv(SPIKE / "manifest.csv")
    t = man[man.split != "train"]
    return t[["lon", "lat"]].to_numpy()


def load_x_tiles():
    meta = tile_metadata()
    man = pd.read_csv(SPIKE / "manifest.csv")
    man = man[(man.status.isin(["ok", "cached"])) & (man.split == "train")]
    tiles = []
    for r in man.itertuples():
        fid = f"{r.provider}_{r.xband_id}"
        f = SPIKE / "data" / f"{fid}.npz"
        if not f.exists():
            continue
        x = np.load(f)["x"].astype("float32")
        if np.isfinite(x).mean() < 0.5:
            continue
        md = meta.get(fid, dict(inc=35.0, az=270.0, pol="VV", az_src="default"))
        tiles.append(dict(id=fid, x=x, provider=r.provider, **md))
    # extra collects from the ceiling check (train region only, not already in the manifest)
    ids = set(man.xband_id) | set(pd.read_csv(SPIKE / "manifest.csv").xband_id)
    pairs = pd.read_csv(SPIKE / "ceiling/pairs_selected.csv")
    pairs = pairs[pairs.region == "train"]
    tp = test_points()
    um = json.loads(common._gj_props(common.PROJ / "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson"))
    um_az = {p["xband_id"]: p.get("view:azimuth") for p in um}
    seen = set()
    for r in pairs.itertuples():
        if np.min(np.hypot(tp[:, 0] - r.lon, tp[:, 1] - r.lat)) < 0.3:      # keep away from held-out tiles
            continue
        hits = list((SPIKE / "ceiling/tiles").glob(f"*{r.id_a[-12:]}*{r.id_b[-12:]}*.npz"))
        if not hits:
            continue
        z = np.load(hits[0])
        for key, xid, prov, inc, orb, side in [("A", r.id_a, r.prov_a, r.inc_a, r.orbit_a, r.side_a),
                                                ("B", r.id_b, r.prov_b, r.inc_b, r.orbit_b, r.side_b)]:
            if xid in ids or xid in seen:
                continue
            x = z[key].astype("float32")
            if np.isfinite(x).mean() < 0.5:
                continue
            seen.add(xid)
            az = um_az.get(xid)
            az = float(az) % 360 if az is not None else heading_to_sensor_az(orb, side)
            tiles.append(dict(id=f"{prov}_{xid}", x=x, provider=prov, inc=float(inc), az=az, pol=r.pol,
                              az_src="ceiling"))
    return tiles


def load_c_tiles():
    from common import PROJ
    sys_path = str(PROJ / "exp4_despeckling")
    import sys
    sys.path.insert(0, sys_path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp4_common", PROJ / "exp4_despeckling/common.py")
    c4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c4)
    tiles = []
    for name, (lon, lat, lc, split, *_r) in c4.SITES.items():
        if split != "train":
            continue
        f = common.EXP4_DATA / f"{name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        vv = z["vv"].astype("float32")
        mean = np.nanmean(vv, 0)
        mean[~(np.isfinite(vv).sum(0) >= vv.shape[0] // 2)] = np.nan
        x = 10 * np.log10(np.where(mean > 0, mean, np.nan))
        tiles.append(dict(id=f"cband_{name}", x=x.astype("float32"), provider="s1_tmean", inc=37.0,
                          az=S1_TO_SENSOR_AZ.get(str(z["orbit"]), 258.0), pol="VV", az_src="orbit"))
    return tiles


def prepare(tiles, scale, band):
    """Per-tile normalization stats for the target and for a full-tile synthetic input."""
    for t in tiles:
        x = t["x"]
        med = np.nanmedian(x)
        xf = np.where(np.isfinite(x), x, med)
        n = (xf.shape[0] // 4) * 4
        xf = xf[:n, :n]
        t["xf"] = xf.astype("float16")
        t["valid"] = np.isfinite(x)[:n, :n]
        xt = torch.from_numpy(xf)[None, None].to(DEV)
        with torch.no_grad():
            inp = degrade(xt, [t["az"]], looks=6)[0, 0].cpu().numpy()
            tgt = xf if scale == 4 else pool_db(xt, 2)[0, 0].cpu().numpy()
        t["in_stats"] = (float(np.median(inp)), float(inp.std() + 1e-6))
        tv = tgt if scale == 4 else tgt
        t["tg_stats"] = (float(np.median(tv)), float(tv.std() + 1e-6))
        t["band"] = band
        del t["x"]
    return tiles


def batch(tiles, bs, rng, scale):
    S = P + 2 * M
    xs, vs, metas, azs, ist, tst = [], [], [], [], [], []
    for _ in range(bs):
        for _ in range(50):
            t = tiles[rng.integers(len(tiles))]
            L = t["xf"].shape[0]
            r, c = (rng.integers(0, (L - S) // 4, 2) * 4)
            v = t["valid"][r:r + S, c:c + S]
            if v.mean() >= 0.97:
                break
        x = t["xf"][r:r + S, c:c + S]
        k, fl = int(rng.integers(4)), bool(rng.random() < 0.5)
        x, v = np.rot90(x, k), np.rot90(v, k)
        if fl:
            x, v = x[:, ::-1], v[:, ::-1]
        az = aug_az(t["az"], k, fl)
        inc = t["inc"] + rng.normal(0, 1.0)
        xs.append(x.astype("float32"))
        vs.append(v.copy())
        azs.append(az)
        metas.append(meta_vec(az, inc, t["pol"], t["band"]))
        ist.append(t["in_stats"])
        tst.append(t["tg_stats"])
    x = torch.from_numpy(np.stack(xs))[:, None].to(DEV)
    v = torch.from_numpy(np.stack(vs))[:, None].to(DEV)
    K = int(rng.integers(4, 8))
    with torch.no_grad():
        s = degrade(x, azs, looks=K)[:, :, M // 4:-M // 4, M // 4:-M // 4]
        if scale == 4:
            y, m = x[:, :, M:-M, M:-M], v[:, :, M:-M, M:-M]
        else:
            y = pool_db(x, 2)[:, :, M // 2:-M // 2, M // 2:-M // 2]
            m = (torch.nn.functional.avg_pool2d(v.float(), 2) > 0.99)[:, :, M // 2:-M // 2, M // 2:-M // 2]
    ist = torch.tensor(ist, device=DEV)[:, :, None, None, None]
    tst = torch.tensor(tst, device=DEV)[:, :, None, None, None]
    s = (s - ist[:, 0]) / ist[:, 1]
    y = (y - tst[:, 0]) / tst[:, 1]
    return s, y, m.float(), torch.from_numpy(np.stack(metas)).to(DEV)


def band_aug(s, p=0.8):
    """Generic band-statistics augmentation of standardized inputs (B,1,h,w), applied per sample:
    a random radial spectral tilt (log power gain in the 20-50 m band ~ U(-0.8, 0.3), near the
    10 m pixel scale ~ U(-0.3, 0.7)) and an exponential tilt of the dB distribution
    (v -> (exp(l v) - 1) / l, l ~ U(-0.3, 0.3)), then re-standardization. Ranges are generic; they
    bracket, but are not fitted to, what C-band looks like."""
    B, _, h, w = s.shape
    fy = torch.fft.fftfreq(h, device=s.device)[:, None] * 2
    fx = torch.fft.fftfreq(w, device=s.device)[None, :] * 2
    r = torch.sqrt(fy ** 2 + fx ** 2).clamp(max=1)[None, None]
    gm = torch.empty(B, 1, 1, 1, device=s.device).uniform_(-0.8, 0.3)
    gh = torch.empty(B, 1, 1, 1, device=s.device).uniform_(-0.3, 0.7)
    logp = gm * torch.exp(-0.5 * ((r - 0.45) / 0.15) ** 2) + gh * ((r - 0.75).clamp(min=0) / 0.25) ** 2
    amp = torch.exp(0.5 * logp)
    amp[..., 0, 0] = 1
    mu = s.mean(dim=(2, 3), keepdim=True)
    o = torch.fft.ifft2(torch.fft.fft2(s - mu) * amp).real + mu
    lam = torch.empty(B, 1, 1, 1, device=s.device).uniform_(-0.3, 0.3)
    lam = torch.where(lam.abs() < 1e-3, torch.full_like(lam, 1e-3), lam)
    o = (torch.exp((lam * o).clamp(max=20)) - 1) / lam
    o = (o - o.median(dim=2, keepdim=True)[0].median(dim=3, keepdim=True)[0]) / (o.std(dim=(2, 3), keepdim=True) + 1e-6)
    keep = (torch.rand(B, 1, 1, 1, device=s.device) > p).float()
    return keep * s + (1 - keep) * o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["x", "cband"], default="x")
    ap.add_argument("--scale", type=int, choices=[2, 4], default=4)
    ap.add_argument("--cond", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bandaug", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(0)
    tiles = load_x_tiles() if a.source == "x" else load_c_tiles()
    tiles = prepare(tiles, a.scale, "X" if a.source == "x" else "C")
    prov = pd.Series([t["provider"] for t in tiles]).value_counts().to_dict()
    pols = pd.Series([t["pol"] for t in tiles]).value_counts().to_dict()
    print(f"{a.name}: {len(tiles)} tiles {prov} {pols}", flush=True)
    G = CondGen(scale=a.scale, cond=bool(a.cond)).to(DEV)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler()
    out = ROOT / "ckpt"
    out.mkdir(exist_ok=True)
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        s, y, m, meta = batch(tiles, a.bs, rng, a.scale)
        if a.bandaug:
            s = band_aug(s)
        with torch.autocast("cuda", dtype=torch.float16):
            p = G(s, meta)
        loss = (torch.abs(p.float() - y) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0:
            log.append({"step": step, "l1": round(loss.item(), 4), "sec": round(time.time() - t0)})
            print(log[-1], flush=True)
    torch.save({"state": G.state_dict(), "scale": a.scale, "cond": bool(a.cond), "source": a.source,
                "n_tiles": len(tiles), "providers": prov, "pols": pols, "steps": a.steps, "bs": a.bs, "bandaug": a.bandaug},
               out / f"{a.name}.pt")
    pd.DataFrame(log).to_csv(out / f"{a.name}_log.csv", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
    finish()
