"""Round 4: build pretraining tiles in Experiment 2's arm A format (input dB 10 m-like, Lee-filtered and
histogram-matched target, mask), so pretraining uses the very loss and preprocessing of the fine-tune.

  X-band (synthetic): exp 1 spike training tiles (held-out regions excluded). Input = round 1 Sentinel-1-like
    degradation (20 x 22 m PSF, K = 6 looks, 4x4 block mean to 10 m), three speckle draws per tile; target =
    the X-band 2 x 2 power mean at 5 m, Lee-filtered (exp 2's lee_filter, 7x7) and histogram-matched to the
    input's distribution, exactly as exp 2 builds lee_hm from Stripmap.
  L-band (NISAR): exp 3 training frames minus its validation frames. Input = 20 MHz GCOV HH (20 m); target =
    40 MHz GCOV HH (10 m), Lee-filtered and histogram-matched to the input. (x2 in pixels, like IW -> Stripmap.)
Writes data/r4_pre/x/*.npz and data/r4_pre/l/*.npz (float16).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import hist_match, lee_filter  # noqa: E402  (exp 2's)

spec = importlib.util.spec_from_file_location("e5common", HERE / "common.py")
E5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E5)

OUT = HERE / "data" / "r4_pre"


def lee_hm(y_db, ref_db, m):
    lee = lee_filter(np.where(m, 10 ** (y_db / 10), np.nan), 7)[0]
    lee_db = np.where(np.isfinite(lee) & (lee > 0), 10 * np.log10(np.maximum(lee, 1e-10)), np.nan)
    return hist_match(np.where(m, lee_db, np.nan), ref_db)


def main():
    shard, nsh = (int(v) for v in (sys.argv[1].split("/") if len(sys.argv) > 1 else ("0", "1")))
    (OUT / "x").mkdir(parents=True, exist_ok=True)
    (OUT / "l").mkdir(parents=True, exist_ok=True)
    meta = E5.tile_metadata()
    man = pd.read_csv(E5.SPIKE / "manifest.csv")
    man = man[man.status.isin(["ok", "cached"]) & (man.split == "train")]
    g = torch.Generator(device="cuda")
    g.manual_seed(4)
    n = 0
    for i, r in enumerate(man.itertuples()):
        if i % nsh != shard:
            continue
        fid = f"{r.provider}_{r.xband_id}"
        f = E5.SPIKE / "data" / f"{fid}.npz"
        of = OUT / "x" / f"{fid}.npz"
        if not f.exists() or of.exists():
            continue
        x = np.load(f)["x"].astype("float32")
        if np.isfinite(x).mean() < 0.9:
            continue
        v25 = np.isfinite(x)
        xf = np.where(v25, x, np.nanmedian(x))
        az = meta.get(fid, {}).get("az", 270.0)
        xt = torch.from_numpy(xf)[None, None].cuda()
        with torch.no_grad():
            ins = [E5.degrade(xt, [az], looks=6, gen=g)[0, 0].cpu().numpy() for _ in range(3)]
            y5 = E5.pool_db(xt, 2)[0, 0].cpu().numpy()
        m5 = v25.reshape(640, 2, 640, 2).all(axis=(1, 3))
        tgt = lee_hm(y5, ins[0], m5)
        np.savez(of, iw0=ins[0].astype("float16"), iw1=ins[1].astype("float16"), iw2=ins[2].astype("float16"),
                 lee_hm=tgt.astype("float16"), mask=m5 & np.isfinite(tgt))
        n += 1
        if n % 50 == 0:
            print("x", n, flush=True)
    print("x tiles", len(list((OUT / "x").glob("*.npz"))), flush=True)
    import glob
    for j, f in enumerate(sorted(glob.glob(str(E5.EXP3_DATA / "tiles" / "train" / "*.npz")))):
        if j % nsh != shard:
            continue
        name = Path(f).stem
        fr = tuple(int(v) for v in name.split("_")[:2])
        of = OUT / "l" / f"{name}.npz"
        if fr in set(E5.EXP3_VAL_FRAMES) or of.exists():
            continue
        z = np.load(f)
        xi = z["m20_HH"].astype("float32")
        y = z["a10_HH"].astype("float32")
        m = np.isfinite(y) & (z["a10_mask"] != 255)
        if m.mean() < 0.5 or not np.isfinite(xi).any():
            continue
        xi = np.where(np.isfinite(xi), xi, np.nanmedian(xi))
        tgt = lee_hm(y, xi, m)
        np.savez(of, iw0=xi.astype("float16"), lee_hm=tgt.astype("float16"), mask=m & np.isfinite(tgt))
    print("l tiles", len(list((OUT / "l").glob("*.npz"))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
