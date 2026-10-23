"""Transfer panels: run the models on 9.6 km Sentinel-1 RTC VV windows where no truth exists.

  neighbors  9.6 km around one held-out X-band tile per test region (same S1 scene as the pair)
  contexts   unseen landscapes: farmland, forest, mountains, desert, villages, plus a city.
             Locations are Experiment 4's held-out sites (so the C-band-only control has not seen
             them either) and the Nile delta.
Per window: input consistency (r between the output averaged back to 10 m and the input) and the
fine-detail energy each model adds (std of output minus its 10 m block mean, in input-std units).
Writes panels_transfer/*.png and transfer_stats.csv.
"""
from __future__ import annotations

import common  # noqa: F401
from common import (adapt, DEV, ROOT, SPIKE, S1_TO_SENSOR_AZ, bicubic, detail, finish, load_exp1, load_model, meta_vec,
                    norm, run_model)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from matplotlib.patches import Rectangle
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import sys  # noqa: E402
sys.path.insert(0, str(ROOT.parent))
from panel_roles import legend, outline  # noqa: E402

# --panels-only: redraw the panels without rewriting transfer_stats.csv
PANELS_ONLY = "--panels-only" in sys.argv
MODEL_COLS = {"exp1_cnn", "x4_cond", "x4_cond+adapt", "x4_aug", "c4"}   # learned models: green; no truth exists here

SPAN, N10 = 9600, 960
CONTEXTS = {
    "matogrosso_farmland": (-55.90, -13.00, "2025-07-15"),
    "gabon_forest": (11.50, -0.50, "2025-03-15"),
    "rockies_mountains": (-106.10, 39.60, "2025-08-15"),
    "atacama_desert": (-69.30, -23.60, "2025-03-15"),
    "nile_delta_villages": (31.00, 30.80, "2025-04-15"),
    "madrid_city": (-3.68, 40.43, "2025-05-15"),
}
COLS = [("s1", "Sentinel-1 IW input (10 m)"), ("bicubic4", "Bicubic"), ("exp1_cnn", "Exp 1 CNN (real pairs)"),
        ("x4_cond", "X-synthetic 4x, cond."), ("x4_cond+adapt", "X-synth. 4x + band adapt"),
        ("x4_aug", "X-synth. 4x, band-aug training"), ("c4", "C-band-only 4x")]


def utm(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def grid(lon, lat):
    epsg = utm(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    return epsg, np.floor((x - SPAN / 2) / 40) * 40, np.floor((y + SPAN / 2) / 40) * 40


def read_win(item, epsg, x0, y1):
    with rasterio.open(item.assets["vv"].href) as s, WarpedVRT(
            s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, 10, 10), width=N10, height=N10,
            resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
        a = v.read(1)
    a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
    return 10 * np.log10(a)


def run_all(models, exp1, s1db, orbit):
    s = norm(s1db)
    s = np.where(np.isfinite(s), s, 0).astype("float32")
    mv = meta_vec(S1_TO_SENSOR_AZ.get(str(orbit), 258.0), 37.0, "VV", "X")
    out = {"s1": np.kron(s, np.ones((4, 4), "float32")), "bicubic4": bicubic(s, 4),
           "exp1_cnn": run_model(exp1, s, kind="exp1")}
    for k, G in models.items():
        out[k] = run_model(G, s, mv)
    if "x4_cond" in models:
        out["x4_cond+adapt"] = run_model(models["x4_cond"], adapt(s), mv)
    stats = {}
    for k, o in out.items():
        if k == "s1" or o.shape[0] != 4 * N10:
            continue
        lo = o.reshape(N10, 4, N10, 4).mean(axis=(1, 3))
        stats[k] = dict(consistency_r=float(np.corrcoef(lo.ravel(), s.ravel())[0, 1]),
                        added_detail=float(detail(o, 4).std()))
    return out, stats


def panel(imgs, title, path, box=None, zoom=None):
    lo, hi = np.percentile(imgs["s1"], [2, 98])
    fig, ax = plt.subplots(2, len(COLS), figsize=(3.2 * len(COLS), 7.8), dpi=80, squeeze=False)
    r, c, n = zoom
    for j, (k, name) in enumerate(COLS):
        a, b = ax[0, j], ax[1, j]
        for q in (a, b):
            q.axis("off")
        if k not in imgs:
            continue
        a.imshow(imgs[k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        a.set_title(name, fontsize=9)
        if box:
            a.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ec="#ff8a4c", lw=1.2))
        a.add_patch(Rectangle((c, r), n, n, fill=False, ec="#6aa8ff", lw=1.2))
        b.imshow(imgs[k][r:r + n, c:c + n], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        b.set_title(name + " (zoom)", fontsize=8)
        if k in MODEL_COLS:
            outline(a, "model", lw=3)
            outline(b, "model", lw=3)
    legend(fig)
    fig.suptitle(title, fontsize=10)
    # explicit spacing: tight_layout left the zoom-row titles over the top-row images
    fig.subplots_adjust(left=0.004, right=0.996, top=0.915, bottom=0.045, wspace=0.03, hspace=0.16)
    fig.savefig(path)
    plt.close(fig)


def main():
    models = {k: load_model(ROOT / "ckpt" / f"{k}.pt")[0] for k in ["x4_cond", "x4_unc", "x4_aug", "c4"]
              if (ROOT / "ckpt" / f"{k}.pt").exists()}
    exp1, which = load_exp1()
    print("models", list(models), which, flush=True)
    cl = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    col = cl.get_collection("sentinel-1-rtc")
    out = ROOT / "panels_transfer"
    out.mkdir(exist_ok=True)
    rows = []
    man = pd.read_csv(SPIKE / "manifest.csv")
    test = man[(man.split == "test") & (man.status == "ok") & (man.corr10 >= 0.3)].sort_values("corr10", ascending=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("spike_prep_tg", SPIKE / "prep.py")
    for reg, g in test.groupby("test_region"):
        if not (reg in ("houston", "newark_ny", "tokyo", "rotterdam") or len(rows) < 0):
            continue
        r = g.iloc[0]
        try:
            it = col.get_item(r.s1_id)
            if "vv" not in it.assets:
                continue
            epsg, x0, y1 = grid(r.lon, r.lat)
            s1 = read_win(it, epsg, x0, y1)
            if np.isfinite(s1).mean() < 0.8:
                continue
            imgs, st = run_all(models, exp1, s1, it.properties.get("sat:orbit_state"))
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(r.lon, r.lat)
            tx0, ty1 = np.floor((x - 1600) / 40) * 40, np.floor((y + 1600) / 40) * 40
            bx, by = (tx0 - x0) / 2.5, (y1 - ty1) / 2.5
            n = 512
            zr, zc = int(min(max(by + 1280 + 200, 0), 3840 - n)), int(min(max(bx - 600, 0), 3840 - n))
            fn = f"neighbor_{reg}.png"
            panel(imgs, f"{reg}: 9.6 km around held-out tile {r.xband_id[:18]} (orange = X-band tile, blue = zoom outside it)",
                  out / fn, box=(bx, by, 1280, 1280), zoom=(zr, zc, n))
            for k, v in st.items():
                rows.append(dict(kind="neighbor", place=reg, file=fn, model=k, **v))
            print("neighbor", reg, flush=True)
        except Exception as e:
            print("fail", reg, e, flush=True)

    for name, (lon, lat, date) in CONTEXTS.items():
        try:
            t = pd.Timestamp(date)
            items = [i for i in cl.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                          datetime=f"{t - pd.Timedelta(days=20):%Y-%m-%d}/{t + pd.Timedelta(days=20):%Y-%m-%d}").items()
                     if "vv" in i.assets]
            if not items:
                print("no s1", name, flush=True)
                continue
            epsg, x0, y1 = grid(lon, lat)
            for it in items[:3]:
                s1 = read_win(it, epsg, x0, y1)
                if np.isfinite(s1).mean() > 0.9:
                    break
            imgs, st = run_all(models, exp1, s1, it.properties.get("sat:orbit_state"))
            fn = f"context_{name}.png"
            panel(imgs, f"{name.replace('_', ' ')}: 9.6 km, {it.id[:32]} (no X-band truth here)", out / fn,
                  zoom=(1664, 1664, 512))
            for k, v in st.items():
                rows.append(dict(kind="context", place=name, file=fn, model=k, **v))
            print("context", name, flush=True)
        except Exception as e:
            print("fail", name, e, flush=True)
    if not PANELS_ONLY:
        pd.DataFrame(rows).to_csv(ROOT / "transfer_stats.csv", index=False)


if __name__ == "__main__":
    main()
    finish()
