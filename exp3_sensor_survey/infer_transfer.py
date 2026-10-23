"""Run arm A (x2) beyond the scored tiles, to see whether it adds plausible structure or only filters speckle.

  neighbors  9.6 km window (480 px input at 20 m -> 960 px at 10 m) centred on each of a few held-out tiles,
             read from the same mixed-mode product. The scored 5.12 km tile is outlined; the zoom row shows a
             1.6 km crop OUTSIDE it. The 40 MHz truth is shown too, since NISAR has it for the whole frame.
  contexts   landscapes the training set barely covers, read from NATIVE 20 MHz acquisitions (the global
             20+5 MHz mode; no 40 MHz truth exists). This is the real deployment case: the model was trained
             only on mixed-mode 20 MHz products split from 40 MHz echoes.
PNG layout: input | bicubic | CNN (| truth), one shared gray stretch per window, zoom row below.
"""
from __future__ import annotations
from config import DATA  # noqa: E402
import os, sys, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, pandas as pd, torch
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nisar_io import open_remote, search      # noqa: E402
from evaluate import bicubic, predict         # noqa: E402
from train import DEV, Gen, to_n              # noqa: E402
sys.path.insert(0, str(ROOT.parent))          # project root, for panel_roles
from panel_roles import outline               # noqa: E402

SPAN, NI = 9600, 480
CONTEXTS = {   # name: (lon, lat)
    "netherlands_polders_villages": (4.95, 52.45),
    "paris_basin_farmland": (2.20, 48.30),
    "austrian_alps": (11.40, 47.15),
    "chicago_suburbs": (-88.05, 41.85),
    "po_valley_farmland": (10.90, 45.10),
    "amazon_forest_edge": (-62.90, -10.10),
    "tabernas_desert_spain": (-2.40, 37.05),
    "nile_delta_villages": (31.00, 30.80),
}


def read_win(g, epsg_xy, n_px, res, pol="HHHH"):
    """window of n_px x n_px at res metres whose top-left edge is epsg_xy (x0, y1)"""
    x, y = g["xCoordinates"][:], g["yCoordinates"][:]
    x0e, y1e = x[0] - res / 2, y[0] + res / 2
    c = int(round((epsg_xy[0] - x0e) / res)); r = int(round((y1e - epsg_xy[1]) / res))
    if r < 0 or c < 0 or r + n_px > len(y) or c + n_px > len(x):
        return None
    a = g[pol][r:r + n_px, c:c + n_px].astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(a) & (a > 0), 10 * np.log10(a), np.nan)


def fill(a):
    return np.where(np.isfinite(a), a, np.nanmedian(a))


def figure(imgs, box_px, zoom, title, f):
    keys = list(imgs)
    ref = imgs[keys[-1]] if "Truth" in keys[-1] else imgs["Bicubic"]
    lo, hi = np.nanpercentile(ref, [2, 98])
    fig, ax = plt.subplots(2, len(keys), figsize=(3.2 * len(keys), 6.6), dpi=90)
    zr, zc, zn = zoom
    for j, k in enumerate(keys):
        ax[0, j].imshow(imgs[k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax[0, j].set_title(k, fontsize=9)
        if box_px:
            r, c, n = box_px
            ax[0, j].add_patch(Rectangle((c, r), n, n, fill=False, ec="#e8a33d", lw=1.2))
        ax[0, j].add_patch(Rectangle((zc, zr), zn, zn, fill=False, ec="#4fb3d9", lw=1.2))
        ax[1, j].imshow(imgs[k][zr:zr + zn, zc:zc + zn], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax[1, j].set_title("zoom (blue box)", fontsize=8, pad=8)
        for a_ in ax[:, j]:
            a_.axis("off")
            if k.startswith("CNN"):
                outline(a_, "model")
            elif k.startswith("Truth"):
                outline(a_, "truth")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(h_pad=2.0); fig.savefig(f); plt.close(fig)   # h_pad keeps zoom titles clear of the outlines


def zoom_outside(imgs, off, n, z=160):
    """160 px zoom outside the scored tile, on valid ground with the most structure"""
    ref = imgs.get("Truth 40 MHz (10 m)", imgs["Input 20 MHz (20 m)"])
    best, out = -1, (0, 0, z)
    for r in range(0, ref.shape[0] - z + 1, 40):
        for c in range(0, ref.shape[1] - z + 1, 40):
            if r + z > off and r < off + n and c + z > off and c < off + n:
                continue
            w = ref[r:r + z, c:c + z]
            if np.isfinite(w).mean() < 0.99:
                continue
            v = np.nanstd(w)
            if v > best:
                best, out = v, (r, c, z)
    return out


def run_window(G, prod_m, epsg, x0, y1, prod_a=None):
    gm = prod_m["science/LSAR/GCOV/grids/frequencyA"]
    x = read_win(gm, (x0, y1), NI, 20.0)
    if x is None or np.isfinite(x).mean() < 0.9:
        return None
    xn = to_n(fill(x))
    imgs = {"Input 20 MHz (20 m)": np.kron(fill(x), np.ones((2, 2))), "Bicubic": bicubic(xn, 2), "CNN arm A": predict(G, xn)}
    if G0 is not None:
        imgs["CNN, no consistency"] = predict(G0, xn)
    if prod_a is not None:
        t = read_win(prod_a["science/LSAR/GCOV/grids/frequencyA"], (x0, y1), NI * 2, 10.0)
        if t is not None and np.isfinite(t).mean() > 0.5:
            imgs["Truth 40 MHz (10 m)"] = t
    return imgs


G0 = None
PINNED = {}


def main():
    global G0
    G = Gen(2).to(DEV).eval()
    G.load_state_dict(torch.load(ROOT / "ckpt/x2.pt", map_location=DEV))
    if (ROOT / "ckpt/x2_cons0.pt").exists():
        G0 = Gen(2).to(DEV).eval()
        G0.load_state_dict(torch.load(ROOT / "ckpt/x2_cons0.pt", map_location=DEV))
    out = ROOT / "eval/transfer"; out.mkdir(parents=True, exist_ok=True)
    global PINNED
    if (out / "transfer.json").exists():
        PINNED = {m["name"]: m["granule"] for m in json.loads((out / "transfer.json").read_text()) if m.get("granule")}
    tiles = pd.read_csv(ROOT / "tiles.csv")
    meta = []
    # neighbors: two held-out tiles per holdout pair-region, first date only
    hold = tiles[tiles.role.isin(["near", "far"])].copy()
    hold["sd"] = [float(np.nanstd(np.load(f"{DATA}/tiles/{r.role}/{r.tile}.npz")["a10_HH"].astype("float32"))) for r in hold.itertuples()]
    pick = hold.sort_values("sd", ascending=False).groupby("reg").head(1)
    for t in pick.itertuples():
        try:
            M, A = open_remote(t.url_m), open_remote(t.url_a)
            pad = (SPAN - 5120) / 2
            x0, y1 = t.x0 - pad, t.y1 + pad
            imgs = run_window(G, M, t.epsg, x0, y1, A)
            if imgs is None:
                print("skip", t.tile, flush=True); continue
            figure(imgs, (int(pad / 10), int(pad / 10), 512), zoom_outside(imgs, int(pad / 10), 512),
                   f"held-out {t.role} {t.tile} ({t.lon:.2f}, {t.lat:.2f}): 9.6 km, scored tile outlined", out / f"nb_{t.tile}.png")
            meta.append({"kind": "neighbor", "name": t.tile, "role": t.role, "lon": t.lon, "lat": t.lat, "png": f"nb_{t.tile}.png"})
            print("ok", t.tile, flush=True)
        except Exception as e:
            print("ERR", t.tile, repr(e)[:200], flush=True)
    # contexts: native 20 MHz acquisitions
    for name, (lon, lat) in CONTEXTS.items():
        try:
            gs = [g for g in search(point=(lon, lat), page_size=200, extra={"temporal": "2026-06-17T00:00:00Z,2026-10-17T00:00:00Z"})
                  if g["bwA"] in ("20", 20, "20.0") or "_2005_" in g["id"]]
            nat = [g for g in gs if "_M_" not in g["id"]] or gs
            if not nat:
                print("no 20 MHz product at", name, flush=True); meta.append({"kind": "context", "name": name, "status": "no 20 MHz"}); continue
            g = sorted(nat, key=lambda q: q["start"])[-1]
            g = next((q for q in gs if q["id"] == PINNED.get(name)), g)   # redraws keep the granule shown on the page
            P = open_remote(g["url"])
            epsg = int(P["science/LSAR/GCOV/grids/frequencyA/projection"][()])
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
            x0, y1 = np.floor((x - SPAN / 2) / 80) * 80, np.ceil((y + SPAN / 2) / 80) * 80
            imgs = run_window(G, P, epsg, x0, y1)
            if imgs is None:
                print("window invalid", name, flush=True); continue
            native = "_M_" not in g["id"]
            figure(imgs, None, (400, 400, 160), f"{name} ({lon}, {lat}) {'native' if native else 'mixed-mode'} 20 MHz {g['id'][17:60]}", out / f"ctx_{name}.png")
            meta.append({"kind": "context", "name": name, "lon": lon, "lat": lat, "native": native, "granule": g["id"], "png": f"ctx_{name}.png"})
            print("ok", name, g["id"], flush=True)
        except Exception as e:
            print("ERR", name, repr(e)[:200], flush=True)
    (out / "transfer.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
