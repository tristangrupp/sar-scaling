"""Spot check, version 2: the same 96 chips as r2_spotcheck/, redrawn with context so a person can judge
whether the Capella change patch is real ground change. See r2_spotcheck_v2/README.md.

Per chip (same chip_id order as r2_spotcheck/chips.csv):
  fresh windowed reads of both Capella GEO COGs (overview <= 0.8 m, as r2_run.py), block-averaged to 1 m
  over the context window; image 2 takes the pair's co-registration shift on its geotransform. The area
  to judge (Capella patch, or the S1 VV flag region for false-alarm chips) comes from data/r2/<tag>.npz,
  mapped back through the S1-Capella roll. After is corrected by the pair's scene-median log ratio, and
  both dates share one gray stretch (2-98th percentile of before over the context window).
Layout: top row context (before, after + purple outline, red/cyan composite), bottom row zoom (before,
after + outline, log ratio). No Sentinel-1, no stratum on the image.

Usage: python r2_spotcheck_v2.py [chip_id ...]    (no args = all 96)
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "r2"
OUT = ROOT / "r2_spotcheck_v2"
OLD = ROOT / "r2_spotcheck" / "chips.csv"
CRES, K = 2.0, 5            # round 2 Capella grid
R = 1.0                     # display grid, m
CTX_MIN, CTX_MAX = 480.0, 900.0
ZOOM_MIN = 100.0
MAX_SRC_RES = 0.8
PURPLE = "#7e3fc0"
FW, FH, DPI = 13.0, 9.4, 100
BOXC = "#ffd400"


def land_name(wc, near_water):
    if near_water and wc != 80:
        return "water edge"
    return {50: "urban", 40: "farm", 60: "bare/mine", 80: "water"}.get(int(wc), "vegetation/other")


def read_window(href, epsg, xa, ya, xb, yb, dxy=(0.0, 0.0)):
    """Intensity on a 1 m grid covering [xa, xb] x [ya, yb] (UTM epsg), NaN outside the image."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds
    from r2_run import window_bytes
    with rasterio.open(href) as s0:
        native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
        ovs = s0.overviews(1)
        same = s0.crs.to_epsg() == epsg
    lvl, res = None, native
    for i, f in enumerate(ovs):
        if native * f <= MAX_SRC_RES:
            lvl, res = i, native * f
    kw = {"overview_level": lvl} if lvl is not None else {}
    pad = 20.0
    nb = 0
    with rasterio.open(href, **kw) as s:
        nb = window_bytes(s, (xa - pad, ya - pad, xb + pad, yb + pad), epsg)
        if same:
            win = from_bounds(xa - pad, ya - pad, xb + pad, yb + pad, s.transform).round_offsets().round_lengths()
            a = s.read(1, window=win, boundless=True, fill_value=0).astype("float32")
            st = s.window_transform(win)
        else:
            n = int(round((xb - xa + 2 * pad) / res)), int(round((yb - ya + 2 * pad) / res))
            st = from_origin(xa - pad, yb + pad, res, res)
            with WarpedVRT(s, crs=f"EPSG:{epsg}", transform=st, width=n[0], height=n[1],
                           resampling=Resampling.nearest, nodata=0) as v:
                a = v.read(1).astype("float32")
    a[a <= 0] = np.nan
    inten = a * a
    st = Affine(st.a, st.b, st.c + dxy[0], st.d, st.e, st.f + dxy[1])
    W, H = int(round((xb - xa) / R)), int(round((yb - ya) / R))
    out = np.full((H, W), np.nan, "float32")
    reproject(inten, out, src_transform=st, src_crs=f"EPSG:{epsg}", dst_transform=from_origin(xa, yb, R, R),
              dst_crs=f"EPSG:{epsg}", resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out, nb, res


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


def block(a, f):
    if f == 1:
        return a
    H, W = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(a[:H, :W].reshape(H // f, f, W // f, f), axis=(1, 3))


def scalebar(ax, extent_m, color="white"):
    """Scale bar (lower left) in axes data units of metres."""
    nice = [10, 20, 25, 50, 100, 200, 250, 500]
    L = max(n for n in nice if n <= extent_m / 4)
    x0, y0 = extent_m * 0.05, extent_m * 0.05
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=5.5, solid_capstyle="butt", zorder=11)
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=3.5, solid_capstyle="butt", zorder=12)
    ax.text(x0 + L / 2, y0 + extent_m * 0.025, f"{L} m", color=color, ha="center", va="bottom", fontsize=9,
            fontweight="bold", zorder=12,
            path_effects=[__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(linewidth=2.5, foreground="black")])


def north(ax, extent_m):
    import matplotlib.patheffects as pe
    x, y = extent_m * 0.92, extent_m * 0.80
    ax.annotate("N", xy=(x, y + extent_m * 0.13), xytext=(x, y), ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=12,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


def make_chip(c, pair, prow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from pyproj import Transformer
    from scipy import ndimage
    from panel_roles import legend, outline
    warnings.filterwarnings("ignore")
    t0 = time.time()
    z = np.load(DATA / f"{c['pair']}.npz")
    epsg, x0, y1 = int(z["epsg"]), float(z["x0"]), float(z["y1"])
    sr, sc = [int(v) * K for v in str(prow["s1c_shift_px"]).split(",")]
    # target mask in the rolled 2 m frame
    if c["stratum"] == "false_alarm":
        lab10 = ndimage.label(z["flag_vv"], structure=np.ones((3, 3)))[0]
        # region id: the one whose centre matches the old chip (recompute via patches of same order)
        tgt = np.kron(lab10 == int(c["s1_region"]), np.ones((K, K), bool))
        wc_reg = z["wc"][lab10 == int(c["s1_region"])]
        wc = int(np.bincount(wc_reg).argmax())
        water = ndimage.binary_dilation(z["wc"] == 80, iterations=1)
        land = land_name(wc, bool(water[lab10 == int(c["s1_region"])].any()))
    else:
        tgt = z["lab"] == int(c["patch"])
        land = land_name(int(c["wc_class"]), bool(c["near_water"]))
    rr, cc = np.where(tgt)
    # map coords of target pixel centres (undo the S1 roll: rolled r -> original r - sr)
    xs = x0 + (cc - sc + 0.5) * CRES
    ys = y1 - (rr - sr + 0.5) * CRES
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    ext = max(np.ptp(xs), np.ptp(ys)) + CRES
    ctx = float(np.clip(max(CTX_MIN, 2.2 * ext + 120), CTX_MIN, CTX_MAX))
    ctx = np.ceil(ctx / 20) * 20
    zoom = float(np.clip(1.5 * ext + 50, ZOOM_MIN, ctx / 2))
    zoom = np.ceil(zoom / 10) * 10
    cx, cy = x0 + CRES * np.round((cx - x0) / CRES), y1 - CRES * np.round((y1 - cy) / CRES)   # snap to the 2 m grid
    xa, xb, ya, yb = cx - ctx / 2, cx + ctx / 2, cy - ctx / 2, cy + ctx / 2
    sy, sx = float(prow["coreg_dy_px"]), float(prow["coreg_dx_px"])
    dxy = (sx * CRES, -sy * CRES) if (abs(sy) > 0.05 or abs(sx) > 0.05) else (0.0, 0.0)
    I1, nb1, src_res = read_window(pair["href1"], epsg, xa, ya, xb, yb)
    I2, nb2, _ = read_window(pair["href2"], epsg, xa, ya, xb, yb, dxy)
    lrmed = float(prow["lr_median_db"])
    A, B = db(I1), db(I2) - lrmed
    n = A.shape[0]
    # target mask on the 1 m grid
    gx = xa + (np.arange(n) + 0.5) * R
    gy = yb - (np.arange(n) + 0.5) * R
    ci = np.floor((gx - x0) / CRES).astype(int) + sc
    ri = np.floor((y1 - gy) / CRES).astype(int) + sr
    okc, okr = (ci >= 0) & (ci < tgt.shape[1]), (ri >= 0) & (ri < tgt.shape[0])
    M = np.zeros((n, n), bool)
    M[np.ix_(okr, okc)] = tgt[np.ix_(ri[okr], ci[okc])]
    # alignment check: fresh 2 m block means vs cached round 2 images (rolled frame)
    f2 = int(CRES / R)
    A2, B2 = block(A, f2), block(B + lrmed, f2)
    r2c = ci[::f2][: A2.shape[1]] // 1
    r2r = ri[::f2][: A2.shape[0]] // 1
    okc2, okr2 = (r2c >= 0) & (r2c < 1280), (r2r >= 0) & (r2r < 1280)
    za = z["a_db"].astype(float)[np.ix_(r2r[okr2], r2c[okc2])]
    zb = z["b_db"].astype(float)[np.ix_(r2r[okr2], r2c[okc2])]
    fa_, fb_ = A2[np.ix_(okr2, okc2)], B2[np.ix_(okr2, okc2)]

    def corr(p, q):
        m = np.isfinite(p) & np.isfinite(q)
        return float(np.corrcoef(p[m], q[m])[0, 1]) if m.sum() > 100 else np.nan
    ra, rb = corr(fa_, za), corr(fb_, zb)
    # does a 2 m shift of the fresh read match the cache better? (should be (0, 0))
    zfa = z["a_db"].astype(float)
    best = max(((corr(fa_, zfa[np.ix_(np.clip(r2r[okr2] + dy, 0, 1279), np.clip(r2c[okc2] + dx, 0, 1279))]), (dy, dx))
                for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)), key=lambda t: t[0])

    # display arrays
    lo, hi = np.nanpercentile(A[np.isfinite(A)], [2, 98])
    cf = 2 if ctx > 500 else 1
    Ac, Bc = (block(A, cf), block(B, cf)) if cf > 1 else (A, B)
    # average in intensity for the context downsample
    if cf > 1:
        Ac, Bc = db(block(10 ** (A / 10), cf)), db(block(10 ** (B / 10), cf))
    nrm = lambda x: np.clip((x - lo) / (hi - lo), 0, 1)
    comp = np.dstack([nrm(Ac), nrm(Bc), nrm(Bc)])
    comp[~np.isfinite(Ac) | ~np.isfinite(Bc)] = 1.0
    zc0 = int(round((ctx - zoom) / 2 / R))
    zn = int(round(zoom / R))
    zs = (slice(zc0, zc0 + zn), slice(zc0, zc0 + zn))
    Az, Bz = A[zs], B[zs]
    sm = lambda x: ndimage.uniform_filter(np.nan_to_num(10 ** (x / 10)), 3)
    LR = db(sm(Bz)) - db(sm(Az))
    LR[~np.isfinite(Az) | ~np.isfinite(Bz)] = np.nan
    Mz = M[zs]
    rdil = max(2, int(round(2 / R)))
    disk = np.hypot(*np.mgrid[-rdil:rdil + 1, -rdil:rdil + 1]) <= rdil
    Md = ndimage.binary_dilation(M, structure=disk)
    Mzd = ndimage.binary_dilation(Mz, structure=disk)

    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = tr.transform(cx, cy)
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("#d9d9d9")
    rdbu = plt.get_cmap("RdBu").copy()
    rdbu.set_bad("#d9d9d9")
    fig, ax = plt.subplots(2, 3, figsize=(FW, FH))
    fig.subplots_adjust(left=0.008, right=0.955, top=0.915, bottom=0.035, wspace=0.03, hspace=0.09)
    E = [0, ctx, 0, ctx]
    Ez = [0, zoom, 0, zoom]
    kw = dict(cmap=gray, vmin=lo, vmax=hi, interpolation="nearest")
    ax[0, 0].imshow(Ac, extent=E, **kw)
    ax[0, 1].imshow(Bc, extent=E, **kw)
    ax[0, 2].imshow(comp, extent=E, interpolation="nearest")
    ax[0, 1].contour(np.flipud(Md), [0.5], colors=PURPLE, linewidths=1.6, extent=E)
    zb0 = (ctx - zoom) / 2
    for a_ in ax[0]:
        a_.add_patch(Rectangle((zb0, zb0), zoom, zoom, fill=False, edgecolor=BOXC, lw=1.4, ls="--", zorder=9))
    ax[1, 0].imshow(Az, extent=Ez, **kw)
    ax[1, 1].imshow(Bz, extent=Ez, **kw)
    ax[1, 1].contour(np.flipud(Mzd), [0.5], colors=PURPLE, linewidths=2.0, extent=Ez)
    im = ax[1, 2].imshow(LR, cmap=rdbu, vmin=-9, vmax=9, interpolation="nearest", extent=Ez)
    ax[1, 2].contour(np.flipud(Mzd), [0.5], colors=PURPLE, linewidths=1.6, linestyles="--", extent=Ez)
    x1, x2 = pair["x1"], pair["x2"]
    ax[0, 0].set_title(f"Capella before {x1}, context {ctx:.0f} m", fontsize=11)
    ax[0, 1].set_title(f"Capella after {x2}, area to judge outlined", fontsize=11)
    ax[0, 2].set_title("Red = before, cyan = after (cyan got brighter)", fontsize=11)
    ax[1, 0].set_title(f"Before, zoom {zoom:.0f} m (yellow box)", fontsize=11)
    ax[1, 1].set_title("After, zoom, area to judge outlined", fontsize=11)
    ax[1, 2].set_title("After / before, dB: blue brighter, red darker", fontsize=11)
    outline(ax[0, 1], "truth")
    outline(ax[1, 1], "truth")
    cb = fig.colorbar(im, cax=ax[1, 2].inset_axes([1.03, 0.1, 0.045, 0.8]))
    cb.ax.tick_params(labelsize=8)
    for a_ in ax.ravel():
        a_.set_xticks([])
        a_.set_yticks([])
    scalebar(ax[0, 0], ctx)
    scalebar(ax[1, 0], zoom)
    north(ax[0, 0], ctx)
    north(ax[1, 0], zoom)
    gap = (pd.Timestamp(x2) - pd.Timestamp(x1)).days
    fig.suptitle(f"{c['chip_id']}   Capella {pair['pol']}  {x1} to {x2} ({gap} days)   ({lat:.5f}, {lon:.5f})\n"
                 f"after image matched to before by the scene median ({-lrmed:+.1f} dB); both dates share one gray stretch",
                 fontsize=11, y=0.993)
    legend(fig)
    fn = f"{c['chip_id']}.png"
    tmp = OUT / f"_{fn}"
    for dpi in (DPI, 88, 76):          # busy scenes: drop resolution until the PNG fits in 400 KB
        fig.savefig(tmp, dpi=dpi)
        im_ = Image.open(tmp).convert("RGB")
        for ncol in (256, 192, 128):
            im_.quantize(colors=ncol, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(OUT / fn, optimize=True)
            if (OUT / fn).stat().st_size <= 400_000:
                break
        if (OUT / fn).stat().st_size <= 400_000:
            break
    plt.close(fig)
    tmp.unlink()
    return {"chip_id": c["chip_id"], "file": fn, "site_id": c["site_id"], "pair": c["pair"], "land": land,
            "area_m2": c["area_m2"], "capella_before": x1, "capella_after": x2, "_stratum": c["stratum"],
            "verdict": "", "note": "",
            "_mb": round((nb1 + nb2) / 1e6, 2), "_kb": round((OUT / fn).stat().st_size / 1e3), "_r_before": round(ra, 3),
            "_r_after": round(rb, 3), "_best_shift": best[1], "_colors": ncol, "_dpi": dpi, "_ctx": ctx, "_zoom": zoom, "_src_res": round(src_res, 3),
            "_target_px_2m": int(tgt.sum()), "_secs": round(time.time() - t0)}


def rebuild_selection():
    """Recreate the r2_spotcheck draw (same seed and order) to recover each chip's patch / region id."""
    from scipy import ndimage
    import r2_spotcheck as s
    rng = np.random.default_rng(42)
    pr = pd.read_csv(ROOT / "r2_pair_results.csv")
    ok = pr[pr.status == "ok"]
    P = []
    for r in ok.itertuples():
        f = DATA / f"{r.tag}_patches.csv"
        if f.exists() and f.stat().st_size > 5:
            p = pd.read_csv(f)
            if len(p):
                p["tag"], p["site_id"] = r.tag, r.site_id
                P.append(p)
    P = pd.concat(P, ignore_index=True)
    P["band"] = np.digitize(P.area_m2, s.BANDS) - 1
    rows = []
    for stratum, cond in (("caught", P.vv_loose.astype(bool)), ("missed", ~P.vv_loose.astype(bool))):
        for b in range(len(s.BANDS) - 1):
            sel = s.pick(P[cond & (P.band == b)], s.PER_BAND, rng)
            rows += [{"stratum": stratum, **r} for r in sel.to_dict("records")]
    fa = []
    for r in ok.itertuples():
        z = np.load(DATA / f"{r.tag}.npz")
        lab, n = ndimage.label(z["flag_vv"], structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        st = ndimage.minimum(z["stable10"].astype(int), lab, idx)
        size = ndimage.sum(np.ones_like(lab), lab, idx)
        com = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
        for i in np.where(st == 1)[0]:
            fa.append({"stratum": "false_alarm", "tag": r.tag, "site_id": r.site_id, "s1_region": int(i + 1),
                       "area_m2": float(size[i] * 100), "row": int(com[i][0] * K), "col": int(com[i][1] * K)})
    rows += s.pick(pd.DataFrame(fa), s.N_FA, rng).to_dict("records")
    C = pd.DataFrame(rows).reset_index(drop=True)
    C["chip_id"] = [f"chip{i:03d}" for i in range(len(C))]
    C = C.rename(columns={"tag": "pair"})
    return C


def worker(c, pair, prow):
    try:
        return make_chip(c, pair, prow)
    except Exception as e:
        traceback.print_exc()
        return {"chip_id": c["chip_id"], "error": f"{type(e).__name__}: {e}"[:300]}


def main():
    OUT.mkdir(exist_ok=True)
    old = pd.read_csv(OLD)
    C = rebuild_selection()
    # the redraw must reproduce the original chip list exactly
    chk = C[["chip_id", "stratum", "site_id", "pair"]].merge(old[["chip_id", "stratum", "site_id", "pair", "area_m2"]],
                                                          on="chip_id", suffixes=("", "_old"))
    bad = chk[(chk.stratum != chk.stratum_old) | (chk.pair != chk.pair_old) | (chk.site_id != chk.site_id_old)]
    assert len(C) == len(old) == 96 and bad.empty, f"selection mismatch: {len(bad)} rows"
    C["area_m2"] = old["area_m2"].values
    pairs = pd.read_csv(ROOT / "r2_pairs.csv")
    pairs["tag"] = [f"site{r.site_id}_{r.x1}_{r.x2}" for r in pairs.itertuples()]
    pairs = pairs.set_index("tag")
    pres = pd.read_csv(ROOT / "r2_pair_results.csv").set_index("tag")
    want = set(sys.argv[1:])
    todo = [r for r in C.to_dict("records") if not want or r["chip_id"] in want]
    print(f"{len(todo)} chips", flush=True)
    rows = []
    with ProcessPoolExecutor(4) as ex:
        futs = [ex.submit(worker, c, pairs.loc[c["pair"]].to_dict(), pres.loc[c["pair"]].to_dict()) for c in todo]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(r, flush=True)
    D = pd.DataFrame(rows).sort_values("chip_id")
    log = OUT / "_build_log.csv"
    if want and log.exists():             # partial rebuild: replace those rows in the full log
        prev = pd.read_csv(log)
        D = pd.concat([prev[~prev.chip_id.isin(D.chip_id)], D], ignore_index=True).sort_values("chip_id")
    D.to_csv(log, index=False)
    if not want:
        cols = ["chip_id", "file", "site_id", "pair", "land", "area_m2", "capella_before", "capella_after", "_stratum",
                "verdict", "note"]
        ok = D[D.get("error").isna()] if "error" in D else D
        ok[cols].to_csv(OUT / "chips.csv", index=False)
    print("made", int((D.get("error", pd.Series([np.nan] * len(D))).isna()).sum()), "of", len(D),
          "MB read", round(D.get("_mb", pd.Series([0])).sum(), 1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
