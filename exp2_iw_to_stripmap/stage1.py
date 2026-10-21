"""Stage 1: X-band geometry check of model outputs (no training).

Two sets of references, both outside training and validation ground:
  sites    X-band collects anywhere > 150 km from training tiles, IW only (stage1_sites.py);
           all three providers including Capella HH
  tiles    held-out Stripmap tiles that overlap X-band (xband_prep.py, split test); here the
           Stripmap itself is also scored, as the reference for how well real 5 m C-band edges
           match X-band edges
Metrics on infrastructure pixels (OSM buildings / roads or WorldCover built-up), excluding
layover, shadow, steep slopes, change and water; the same Canny detector for every image:
  p2x_m    mean distance from output edges to the nearest X-band edge (m, capped 50):
           edges in the wrong place or invented
  x2p_m    mean distance from X-band edges to the nearest output edge: structure missed
  osm_prec share of output edge pixels on OSM buildings / roads
Writes stage1_pairs.csv and stage1_summary.csv.

  python stage1.py esrt_paper esrt_a cnn_a esrt_b ...
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from common import DATA, ROOT, TILES
from score import PRED, bicubic
from sr_data import Norm
from train_sr import CKPT, DEV, make_model, predict
from xband_prep import XB, edges

S1DIR = DATA / "stage1"


def chamfer(pe, xe, m, cap=50.0):
    pe, xe = pe & m, xe & m
    if pe.sum() < 20 or xe.sum() < 20:
        return np.nan, np.nan
    dx = ndimage.distance_transform_edt(~xe) * 5.0
    dp = ndimage.distance_transform_edt(~pe) * 5.0
    return float(np.minimum(dx[pe], cap).mean()), float(np.minimum(dp[xe], cap).mean())


def score_images(imgs, z, meta):
    m = z["valid"] & z["infra"]
    if m.sum() < 500:
        return []
    xv = z["valid"] if "bad" not in z else z["valid"]
    xe = z["edges"] & xv
    osm = ndimage.binary_dilation(z["osm"], iterations=1)
    rows = []
    for name, im in imgs.items():
        mv = xv & np.isfinite(im)
        pe = edges(im, mv)
        p2x, x2p = chamfer(pe, xe, m & mv)
        pm = pe & m & mv
        rows.append({**meta, "model": name, "p2x_m": p2x, "x2p_m": x2p,
                     "osm_prec": float((pm & osm).sum() / max(pm.sum(), 1)) if z["osm"].any() else np.nan,
                     "n_px": int(m.sum())})
    return rows


def main(models):
    nets, norm = {}, None
    for n in models:
        ck = torch.load(CKPT / f"{n}.pt", map_location=DEV)
        net = make_model(ck["model"]).to(DEV)
        net.load_state_dict(ck["state"])
        nets[n], norm = net, Norm(*ck["norm"])
    rows = []
    # --- sites: IW only
    st = pd.read_csv(ROOT / "stage1_sites.csv")
    st = st[st.status.isin(["ok"])]
    for r in st.itertuples():
        f = S1DIR / f"{r.provider}_{r.xband_id}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        iw = z["iw"].astype("float32")
        iw = np.where(np.isfinite(iw), iw, np.nanmedian(iw))
        iw_n = norm(iw)
        imgs = {"bicubic": norm.inv(bicubic(iw_n))}
        for k, net in nets.items():
            imgs[k] = norm.inv(predict(net, iw_n))
        rows += score_images(imgs, z, {"set": "sites", "xband_id": r.xband_id, "provider": r.provider,
                                       "pol": r.pol, "region": "site", "km_to_train": r.km_to_train})
    # --- held-out Stripmap tiles with X-band
    xp = pd.read_csv(ROOT / "xband_pairs.csv")
    xp = xp[(xp.status == "ok") & (xp.split == "test")]
    for r in xp.itertuples():
        z = np.load(XB / f"{r.tile_id}__{r.xband_id}.npz")
        tz = np.load(TILES / f"{r.tile_id}.npz")
        imgs = {"stripmap": tz["sm"].astype("float32"), "stripmap_lee": tz["lee"].astype("float32")}
        for name in ["bicubic", *models]:
            f = PRED / name / f"{r.tile_id}.npy"
            if f.exists():
                imgs[name] = np.load(f).astype("float32")
        rows += score_images(imgs, z, {"set": "tiles", "xband_id": r.xband_id, "provider": r.provider, "pol": r.pol,
                                       "region": r.region, "tile_id": r.tile_id, "geom_match": r.geom_match})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "stage1_pairs.csv", index=False)
    agg = dict(n=("xband_id", "size"), p2x_m=("p2x_m", "median"), x2p_m=("x2p_m", "median"),
               osm_prec=("osm_prec", "median"))
    df["prov_pol"] = np.where(df.provider == "capella", "capella " + df.pol.astype(str), df.provider)
    parts = [df.groupby(["set", "prov_pol", "model"]).agg(**agg).reset_index(),
             df.assign(prov_pol="all").groupby(["set", "prov_pol", "model"]).agg(**agg).reset_index(),
             df[df.set == "tiles"].assign(prov_pol="region " + df.region).groupby(["set", "prov_pol", "model"]).agg(**agg).reset_index()]
    s = pd.concat(parts)
    s.to_csv(ROOT / "stage1_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
