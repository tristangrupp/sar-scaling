"""Run the trained CNNs beyond the X-band footprints, to see what they do where no truth exists.

Two sets:
  neighbors  for each held-out test tile, a 9.6 km square (3 x 3 tiles) of the same Sentinel-1
             scene centered on it; the X-band truth exists only in the middle tile (outlined)
  contexts   landscapes the training set barely covers: farmland, forest, mountains, desert

Normalization matches training: the 9.6 km Sentinel-1 window is standardized as a whole.
Outputs PNGs to eval/neighbors and eval/contexts.
"""
from __future__ import annotations

import os
import sys

import prep  # sets GDAL env before rasterio

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import torch
from matplotlib.patches import Rectangle
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from train import DEV, Gen, norm

ROOT = Path(__file__).resolve().parent
SPAN = 9600                      # m, 3 x 3 tiles
N10 = SPAN // 10                 # 960 px at 10 m

CONTEXTS = {
    "iowa_farmland": (-93.62, 42.03, "2025-07-15"),
    "rondonia_forest_edge": (-62.90, -10.10, "2025-07-15"),
    "swiss_alps": (7.75, 46.02, "2025-08-15"),
    "sahara_dunes": (8.50, 27.50, "2025-05-15"),
    "nile_delta_villages": (31.00, 30.80, "2025-04-15"),
    "rotterdam_port_wide": (4.20, 51.95, "2025-06-15"),
}


def client():
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def read_s1_window(item, epsg, x0, y1):
    pol = "vv" if "vv" in item.assets else "hh"     # Capella HH test tiles pair with S1 HH
    with rasterio.open(item.assets[pol].href) as s, WarpedVRT(
            s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, 10, 10), width=N10, height=N10,
            resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
        a = v.read(1)
    a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
    return 10 * np.log10(a)


def grid(lon, lat):
    epsg = prep.utm_epsg(lon, lat)
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    return epsg, np.floor((x - SPAN / 2) / 40) * 40, np.floor((y + SPAN / 2) / 40) * 40


@torch.no_grad()
def run(models, s1db):
    s = norm(s1db)
    s = np.where(np.isfinite(s), s, 0).astype("float32")
    t = torch.from_numpy(s)[None, None].to(DEV)
    out = {"s1": np.kron(s, np.ones((4, 4), "float32"))}
    out["bicubic"] = torch.nn.functional.interpolate(t, scale_factor=4, mode="bicubic",
                                                     align_corners=False)[0, 0].cpu().numpy()
    for k, G in models.items():
        out[k] = G(t)[0, 0].cpu().numpy()
    return out


def panel(imgs, title, path, box=None, zoom=None):
    order = [("s1", "Sentinel-1 IW input (10 m)"), ("bicubic", "Bicubic"), ("l1", "CNN, pixel loss"),
             ("gan", "CNN + GAN")]
    lo, hi = np.percentile(imgs["s1"], [2, 98])
    rows = 2 if zoom else 1
    fig, ax = plt.subplots(rows, 4, figsize=(16, 4.3 * rows), dpi=100, squeeze=False)
    for j, (k, name) in enumerate(order):
        a = ax[0, j]
        a.imshow(imgs[k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        a.set_title(name, fontsize=10)
        a.axis("off")
        if box:
            a.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ec="#ff8a4c", lw=1.2))
        if zoom:
            r, c, n = zoom
            a.add_patch(Rectangle((c, r), n, n, fill=False, ec="#6aa8ff", lw=1.2))
            b = ax[1, j]
            b.imshow(imgs[k][r:r + n, c:c + n], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            b.set_title(name + " (zoom, blue box)", fontsize=9)
            b.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    models = {}
    for k in ["l1", "gan"]:
        G = Gen().to(DEV).eval()
        G.load_state_dict(torch.load(ROOT / f"ckpt/{k}.pt", weights_only=True))
        models[k] = G
    c = client()
    man = pd.read_csv(ROOT / "manifest.csv")
    test = man[(man.split == "test") & (man.status == "ok") & (man.corr10 >= 0.3)]
    out_n, out_c = ROOT / "eval/neighbors", ROOT / "eval/contexts"
    out_n.mkdir(parents=True, exist_ok=True)
    out_c.mkdir(parents=True, exist_ok=True)

    seen_regions = {}
    for r in test.itertuples():
        if seen_regions.get(r.test_region, 0) >= 2:      # two per region keeps the page readable
            continue
        it = c.get_collection("sentinel-1-rtc").get_item(r.s1_id)
        epsg, x0, y1 = grid(r.lon, r.lat)
        s1 = read_s1_window(it, epsg, x0, y1)
        if np.isfinite(s1).mean() < 0.8:
            continue
        imgs = run(models, s1)
        # where the 3.2 km X-band tile sits inside the 9.6 km window, in 2.5 m pixels
        _, tx0, ty1 = prep.tile_grid(r.lon, r.lat)
        bx, by = (tx0 - x0) / 2.5, (y1 - ty1) / 2.5
        n = 512
        zr, zc = int(min(max(by + 1280 + 200, 0), 3840 - n)), int(min(max(bx - 600, 0), 3840 - n))
        panel(imgs, f"{r.test_region}: 9.6 km around held-out tile {r.xband_id[:20]} (orange = X-band tile; "
                    f"blue = zoom outside it). S1 {r.s1_id[:25]}",
              out_n / f"{r.test_region}_{r.xband_id[:12]}.png", box=(bx, by, 1280, 1280), zoom=(zr, zc, n))
        seen_regions[r.test_region] = seen_regions.get(r.test_region, 0) + 1
        print("neighbors", r.test_region, flush=True)

    for name, (lon, lat, date) in CONTEXTS.items():
        t = pd.Timestamp(date)
        items = [i for i in c.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                     datetime=f"{t - pd.Timedelta(days=20):%Y-%m-%d}/{t + pd.Timedelta(days=20):%Y-%m-%d}").items()
                 if "vv" in i.assets]
        if not items:
            print("no s1", name, flush=True)
            continue
        epsg, x0, y1 = grid(lon, lat)
        for it in items[:3]:
            s1 = read_s1_window(it, epsg, x0, y1)
            if np.isfinite(s1).mean() > 0.9:
                break
        imgs = run(models, s1)
        panel(imgs, f"{name.replace('_', ' ')}: 9.6 km, {it.id[:25]} (no X-band truth here)",
              out_c / f"{name}.png", zoom=(1664, 1664, 512))
        print("context", name, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
