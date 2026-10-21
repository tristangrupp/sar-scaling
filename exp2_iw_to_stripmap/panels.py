"""Transfer panels: run the models beyond their truth footprints.

  neighbors  a 9.6 km IW window around one held-out tile per region (Houston, Owens, W Australia,
             paper test), truth tile outlined; plus a 1.6 km zoom inside the truth tile with the
             Stripmap next to the outputs
  contexts   landscapes the training set barely covers (IW only, no truth)
Normalization is the training set's global min-max, as in training.
Writes panels/<name>.png.

  python panels.py esrt_a cnn_a esrt_b
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from common import ROOT, TILES, db, pc_client, pc_item, read_iw, tile_grid
from score import bicubic
from sr_data import Norm
from train_sr import CKPT, DEV, make_model, predict

OUT = ROOT / "panels"
SPAN, N10 = 9600, 960
CONTEXTS = {
    "iowa_farmland": (-93.62, 42.03, "2025-07-15"),
    "rondonia_forest_edge": (-62.90, -10.10, "2025-07-15"),
    "swiss_alps": (7.75, 46.02, "2025-08-15"),
    "sahara_dunes": (8.50, 27.50, "2025-05-15"),
    "nile_delta_villages": (31.00, 30.80, "2025-04-15"),
    "navarre_pamplona": (-1.64, 42.81, "2025-06-15"),
}
LABEL = {"bicubic": "bicubic IW", "esrt_a": "ESRT arm A", "cnn_a": "CNN arm A", "esrt_b05": "ESRT arm B",
         "esrt_paper": "ESRT, round 1 set", "esrt_v2_mask_s0": "ESRT v2 + change mask", "esrt_v2_nomask_s0": "ESRT v2, no mask",
         "esrt_v2_mask_s1": "ESRT v2 + mask (seed 1)", "esrt_v2_nomask_s1": "ESRT v2, no mask (seed 1)"}




def load_models(names):
    nets, norms = {}, {}
    for n in names:
        ck = torch.load(CKPT / f"{n}.pt", map_location=DEV)
        m = make_model(ck["model"]).to(DEV)
        m.load_state_dict(ck["state"])
        nets[n], norms[n] = m, Norm(*ck["norm"])
    return nets, norms


def run_all(nets, norms, iw_db):
    n0 = norms[next(iter(norms))]
    out = {"bicubic": n0.inv(bicubic(n0(iw_db)))}
    for k, m in nets.items():
        out[k] = norms[k].inv(predict(m, norms[k](iw_db)))
    return out


def show(ax, a, lo, hi, title):
    ax.imshow(a, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def neighbors(nets, norm, regions=("houston", "owens", "w_australia", "amieva_test"), sfx="", prefix="neighbors"):
    tiles = pd.read_csv(ROOT / f"tiles{sfx}.csv")
    pairs = pd.read_csv(ROOT / f"pairs_final{sfx}.csv")
    ok = tiles.drop(columns=["iw_id"]).merge(pairs[pairs.use][["tile_id", "corr10", "iw_id"]], on="tile_id")
    for reg in regions:
        g = ok[ok.region == reg].sort_values(["n_xband", "corr10"], ascending=False)
        if g.empty:
            continue
        t = g.iloc[0]
        z = np.load(TILES / f"{t.tile_id}.npz")
        epsg, x0, y1, ix0, iy1 = int(z["epsg"]), float(z["x0"]), float(z["y1"]), float(z["iw_x0"]), float(z["iw_y1"])
        items = [pc_item("sentinel-1-rtc", i) for i in t.iw_id.split(";")]
        iw = db(read_iw(items, epsg, ix0 - 3200, iy1 + 3200, N10))
        if np.isfinite(iw).mean() < 0.7:
            continue
        iw = np.where(np.isfinite(iw), iw, np.nanmedian(iw))
        outs = run_all(nets, norm, iw)
        lo, hi = np.nanpercentile(iw, [2, 98])
        names = list(outs)
        fig, ax = plt.subplots(2, len(names) + 1, figsize=(3.1 * (len(names) + 1), 6.6), dpi=110)
        show(ax[0, 0], np.kron(iw, np.ones((2, 2))), lo, hi, "IW 10 m (input), 9.6 km")
        for j, k in enumerate(names, 1):
            show(ax[0, j], outs[k], lo, hi, LABEL.get(k, k))
        for a in ax[0]:
            a.add_patch(Rectangle((640, 640), 640, 640, fill=False, ec="#e4572e", lw=1.2))
        # zoom: 1.6 km in the truth tile, full 5 m resolution
        r0 = c0 = 640 + 160
        sl = (slice(r0, r0 + 320), slice(c0, c0 + 320))
        sm = z["sm"].astype("float32")[160:480, 160:480]
        show(ax[1, 0], sm, lo, hi, "Stripmap 5 m (truth), 1.6 km zoom")
        for j, k in enumerate(names, 1):
            show(ax[1, j], outs[k][sl], lo, hi, f"{LABEL.get(k, k)}, zoom")
        fig.suptitle(f"{reg}: {t.tile_id} (red box = truth tile)", fontsize=9)
        fig.tight_layout()
        OUT.mkdir(exist_ok=True)
        fig.savefig(OUT / f"{prefix}_{reg}.png")
        plt.close(fig)
        print("panel", reg, flush=True)


def contexts(nets, norm):
    for name, (lon, lat, when) in CONTEXTS.items():
        t = pd.Timestamp(when, tz="UTC")
        its = [i for i in pc_client().search(collections=["sentinel-1-rtc"], bbox=[lon - .02, lat - .02, lon + .02, lat + .02],
                                             datetime=f"{t - pd.Timedelta(days=20):%Y-%m-%d}/{t + pd.Timedelta(days=20):%Y-%m-%d}").items()
               if "vv" in i.assets]
        epsg, x0, y1 = tile_grid(lon, lat, SPAN)
        iw = None
        for it in sorted(its, key=lambda i: abs(pd.Timestamp(i.datetime) - t))[:3]:
            a = db(read_iw(it, epsg, x0, y1, N10))
            if np.isfinite(a).mean() > 0.9:
                iw = np.where(np.isfinite(a), a, np.nanmedian(a))
                break
        if iw is None:
            continue
        outs = run_all(nets, norm, iw)
        lo, hi = np.nanpercentile(iw, [2, 98])
        names = list(outs)
        fig, ax = plt.subplots(2, len(names) + 1, figsize=(3.1 * (len(names) + 1), 6.6), dpi=110)
        show(ax[0, 0], np.kron(iw, np.ones((2, 2))), lo, hi, "IW 10 m (input), 9.6 km")
        for j, k in enumerate(names, 1):
            show(ax[0, j], outs[k], lo, hi, LABEL.get(k, k))
        sl = (slice(800, 1120), slice(800, 1120))
        show(ax[1, 0], np.kron(iw, np.ones((2, 2)))[sl], lo, hi, "IW, 1.6 km zoom")
        for j, k in enumerate(names, 1):
            show(ax[1, j], outs[k][sl], lo, hi, f"{LABEL.get(k, k)}, zoom")
        fig.suptitle(f"{name.replace('_', ' ')} (no Stripmap truth)", fontsize=9)
        fig.tight_layout()
        OUT.mkdir(exist_ok=True)
        fig.savefig(OUT / f"context_{name}.png")
        plt.close(fig)
        print("panel", name, flush=True)


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        if "--r3" in args:          # round 3: new inland holdouts, v2 tile set
            args.remove("--r3")
            nets, norm = load_models(args)
            neighbors(nets, norm, ("yemen_west", "bangladesh"), "_v2", "r3_neighbors")
        else:
            nets, norm = load_models(args)
            neighbors(nets, norm)
            contexts(nets, norm)
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
