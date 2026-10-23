"""Score against Experiment 2's Sentinel-1 Stripmap tiles (5 m gamma0, same sensor and band).

For every exp2_iw_to_stripmap/sm_tiles/*.tif (read only):
  - IW RTC VV from Planetary Computer, nearest date within +/- 6 days, warped to 10 m on the tile's
    grid with a 160 m margin of context on each side.
  - co-register coarse then fine: the Stripmap averaged to 10 m (power) is phase-correlated with
    the IW at 10 m (bounded at 4 px = 40 m); then the shifted Stripmap at 5 m is phase-correlated
    with the bicubic IW at 5 m (bounded at 2 px = 10 m).
  - outputs scored at 5 m: 2x models directly, 4x models averaged 2x2. MAE / SSIM / detail_r /
    hf_ratio (spike metrics) and the edge chamfer / edge F at 5 m with a 10 m Canny scale
    (Stripmap is close to single-look, so finer edge scales mostly find speckle).
Tile date: YYYYMMDD in the file name, else exp2_iw_to_stripmap/test_scene.csv for test_* tiles.
Writes eval_sm_scores.csv and panels_sm/.
"""
from __future__ import annotations

import common  # noqa: F401
from common import (adapt, PROJ, ROOT, S1_TO_SENSOR_AZ, bicubic, chamfer, edges, finish, fine_register, load_exp1,
                    load_model, meta_vec, norm, run_model, scores)

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage

import sys  # noqa: E402
sys.path.insert(0, str(PROJ))
from panel_roles import legend, outline  # noqa: E402

SM = PROJ / "exp2_iw_to_stripmap/sm_tiles"
MARGIN = 16      # 10 m pixels of context
MODELS = ["x4_cond", "x4_unc", "x4_aug", "x2_cond", "x2_unc", "c4", "c2"]
TEST_BOXES = {"houston": (-95.7, 29.4, -94.7, 30.1), "newark_ny": (-74.35, 40.45, -73.7, 40.95),
              "tokyo": (139.4, 35.3, 140.2, 35.95), "antwerp": (4.0, 51.1, 4.6, 51.45),
              "rotterdam": (3.9, 51.75, 4.7, 52.05)}


def tile_date(f):
    m = re.search(r"(20\d{6})(T\d{6})?", f.stem)
    if m:
        return pd.Timestamp(m.group(1) + (m.group(2) or ""))
    ts = pd.read_csv(PROJ / "exp2_iw_to_stripmap/test_scene.csv")
    return pd.Timestamp(ts.start.iloc[0]).tz_localize(None)


def pool(a, f):
    n0, n1 = a.shape[0] // f, a.shape[1] // f
    return a[:n0 * f, :n1 * f].reshape(n0, f, n1, f).mean(axis=(1, 3))


def main():
    tifs = sorted(SM.glob("*.tif"))
    print("stripmap tiles:", len(tifs), flush=True)
    if not tifs:
        return
    models = {k: load_model(ROOT / "ckpt" / f"{k}.pt")[0] for k in MODELS if (ROOT / "ckpt" / f"{k}.pt").exists()}
    exp1, which = load_exp1()
    cl = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    rows, out = [], ROOT / "panels_sm"
    out.mkdir(exist_ok=True)
    for f in tifs:
        try:
            with rasterio.open(f) as s:
                sm = s.read(1).astype("float32")
                tr, crs, b = s.transform, s.crs, s.bounds
            n5 = sm.shape[0]
            n10 = n5 // 2
            lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)
            region = next((k for k, (a, bb, c, d) in TEST_BOXES.items() if a <= lon <= c and bb <= lat <= d), "")
            t = tile_date(f)
            items = [i for i in cl.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                          datetime=f"{t - pd.Timedelta(days=6):%Y-%m-%d}/{t + pd.Timedelta(days=6):%Y-%m-%d}").items()
                     if "vv" in i.assets]
            # same orbit direction first (S1 descending passes near 06 local solar time, ascending near 18)
            sm_orbit = "descending" if 2 <= (t.hour + t.minute / 60 + lon / 15) % 24 <= 10 else "ascending"
            items.sort(key=lambda i: (i.properties.get("sat:orbit_state") != sm_orbit,
                                      abs(pd.Timestamp(i.datetime).tz_localize(None) - t)))
            iw_big, BIG = None, 100
            for it in items[:3]:
                with rasterio.open(it.assets["vv"].href) as s, WarpedVRT(
                        s, crs=crs, transform=from_origin(b.left - BIG * 10, b.top + BIG * 10, 10, 10),
                        width=n10 + 2 * BIG, height=n10 + 2 * BIG, resampling=Resampling.bilinear,
                        nodata=-32768, dtype="float32") as v:
                    a = v.read(1)
                a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
                if np.isfinite(a).mean() > 0.95:
                    iw_big = 10 * np.log10(a)
                    break
            if iw_big is None:
                print("no IW for", f.name, flush=True)
                continue
            iw_big = np.where(np.isfinite(iw_big), iw_big, np.nanmedian(iw_big))
            m5 = np.isfinite(sm) & (sm > 0)
            smf = np.where(m5, sm, np.nanmedian(sm[m5]))
            sm10 = 10 * np.log10(pool(smf, 2))
            # wide coarse search (+/- 1 km): normalized cross-correlation of the Stripmap at 10 m in the IW
            from skimage.feature import match_template
            cc = match_template(iw_big, sm10)
            dr, dc = np.unravel_index(cc.argmax(), cc.shape)
            ncc0, ncc1 = float(cc[BIG, BIG]), float(cc.max())
            variants = [("nominal", 0, 0)]
            if (dr, dc) != (BIG, BIG):
                variants.append(("searched", int(dr) - BIG, int(dc) - BIG))
            for vname, oy, ox in variants:
                r0, c0_ = BIG + oy - MARGIN, BIG + ox - MARGIN
                if r0 < 0 or c0_ < 0 or r0 + n10 + 2 * MARGIN > iw_big.shape[0] or c0_ + n10 + 2 * MARGIN > iw_big.shape[1]:
                    continue
                iw = iw_big[r0:r0 + n10 + 2 * MARGIN, c0_:c0_ + n10 + 2 * MARGIN]
                s_in = norm(iw)
                iw_c = iw[MARGIN:-MARGIN, MARGIN:-MARGIN]
                sh10 = fine_register(iw_c - iw_c.mean(), sm10 - sm10.mean(), np.ones_like(sm10, bool), max_px=4)
                c0 = float(np.corrcoef(iw_c[4:-4, 4:-4].ravel(), sm10[4:-4, 4:-4].ravel())[0, 1])
                sm10s = ndimage.shift(sm10, sh10, order=1, mode="nearest")
                c1 = float(np.corrcoef(iw_c[4:-4, 4:-4].ravel(), sm10s[4:-4, 4:-4].ravel())[0, 1])
                smdb = 10 * np.log10(smf)
                smdb = ndimage.shift(smdb, sh10 * 2, order=1, mode="nearest")
                m5s = ndimage.shift(m5.astype("float32"), sh10 * 2, order=0, mode="constant") > 0.5
                m5s[:8], m5s[-8:], m5s[:, :8], m5s[:, -8:] = False, False, False, False
                truth = np.nan_to_num(norm(np.where(m5s, smdb, np.nan)))
                b2 = bicubic(s_in, 2)[2 * MARGIN:-2 * MARGIN, 2 * MARGIN:-2 * MARGIN]
                sh5 = fine_register(b2, truth, m5s, max_px=2)
                truth = ndimage.shift(truth, sh5, order=1, mode="nearest")
                m5s = m5s & (ndimage.shift(m5s.astype("float32"), sh5, order=0, mode="constant") > 0.5)
                orbit = it.properties.get("sat:orbit_state")
                mv = meta_vec(S1_TO_SENSOR_AZ.get(str(orbit), 258.0), 37.0, "VV", "X")
                P = {"bicubic": bicubic(s_in, 2), "exp1_cnn": pool(run_model(exp1, s_in, kind="exp1"), 2)}
                s_ad = adapt(s_in)
                for k, G in models.items():
                    o = run_model(G, s_in, mv)
                    P[k] = o if G.scale == 2 else pool(o, 2)
                    if k in ("x4_cond", "x2_cond"):
                        o = run_model(G, s_ad, mv)
                        P[k + "+adapt"] = o if G.scale == 2 else pool(o, 2)
                P = {k: v[2 * MARGIN:-2 * MARGIN, 2 * MARGIN:-2 * MARGIN] for k, v in P.items()}
                te = edges(truth, m5s, 5.0, sigma_m=10.0)
                base = dict(tile=f.stem, region=region, variant=vname, offset_row_m=oy * 10, offset_col_m=ox * 10,
                            ncc_nominal=ncc0, ncc_best=ncc1, iw_id=it.id,
                            dt_days=abs((pd.Timestamp(it.datetime).tz_localize(None) - t).days),
                            coarse_shift_m=float(np.hypot(*sh10) * 10), fine_shift_m=float(np.hypot(*sh5) * 5),
                            corr10_raw=c0, corr10_reg=c1)
                for k, p in P.items():
                    r = dict(base, model=k, **scores(p, truth, m5s, 2))
                    r.update(chamfer(edges(p, m5s, 5.0, sigma_m=10.0), te, m5s, 5.0, tols=(10.0, 20.0)))
                    rows.append(r)
                order = ["bicubic", "exp1_cnn", "x2_cond", "x4_cond", "c2", "c4"]
                lo, hi = np.percentile(truth[m5s], [2, 98])
                ims = [("IW input (10 m)", np.kron(s_in[MARGIN:-MARGIN, MARGIN:-MARGIN], np.ones((2, 2))))]
                ims += [(k, P[k]) for k in order if k in P] + [("Stripmap (5 m)", truth)]
                fig, ax = plt.subplots(1, len(ims), figsize=(2.6 * len(ims), 3.0), dpi=80)
                for i_, (a_, (nm, im)) in enumerate(zip(ax, ims)):
                    a_.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                    a_.set_title(nm, fontsize=8)
                    a_.axis("off")
                    if nm not in ("bicubic",) and 0 < i_ < len(ims) - 1:
                        outline(a_, "model", lw=3)        # learned models
                    elif i_ == len(ims) - 1:
                        outline(a_, "truth", lw=3)        # Stripmap reference
                legend(fig)
                fig.suptitle(f"{f.stem} ({vname} position, offset {oy * 10} m N-S, {ox * 10} m E-W; r10 = {c1:.2f})", fontsize=9)
                fig.tight_layout(rect=(0, 0.06, 1, 1))
                fig.savefig(out / f"{f.stem}_{vname}.png")
                plt.close(fig)
                print(f.name, region, vname, (oy, ox), "ncc", ncc0, ncc1, "shifts", sh10, sh5, "corr10", c0, c1, flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("fail", f.name, e, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "eval_sm_scores.csv", index=False)
    if len(df):
        print(df.groupby(["variant", "model"])[["mae", "ssim", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m"]]
              .median().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
    finish()
