"""Pilot: does a Sentinel-1 change flag line up with change between two X-band images?

For each pilot pair (site, x1, x2) from tasking_pairs.csv (x1 <= S1 before < S1 after <= x2):
  grid      2.56 km square (256 x 256 at 10 m, local UTM) at the centroid of the two X-band footprints'
            intersection
  S1 flag   exp2's change_mask.py rule, unchanged: Planetary Computer IW RTC VV (gamma0) for the two
            passes of one track, warped bilinearly to the 10 m grid, dB, 3x3 mean, |difference| > 3 dB,
            valid pixels eroded by one
  X change  the same rule on the X-band pair: GEO/GEC COG read from the overview just finer than
            2.5 m, averaged to 2.5 m, intensity block-averaged to 10 m, dB, global median offset
            removed (the two products are not radiometrically matched), 3x3 mean, |difference| > 3 dB
  align     integer shift (+-6 px) of the X-band grid onto S1 by correlation of the after images
  metrics   pixel precision of S1 flags against X change dilated by 2 px (20 m), the X change base
            rate, lift = precision / base rate; region precision (S1 flag regions >= 4 px, explained
            when >= 30% of the region falls in dilated X change) and the same under 20 random
            circular shifts of the X change map (chance level); pixel recall of X change by dilated
            S1 flags; sign agreement where both flag
Writes pilot_pairs.csv, data/<pair>.npz and panels/<pair>.png. Only windowed COG reads (a few MB per pair).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, urljoin

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ))
from panel_roles import MODEL, TRUTH, outline  # noqa: E402

DATA, PANELS = ROOT / "data", ROOT / "panels"
N, RES = 256, 10.0
XRES, XN = 2.5, 1024
THRESH, DIL, MIN_REGION, EXPLAINED = 3.0, 2, 4, 0.30
HREFS = ROOT / "xband_hrefs.json"
CAPELLA_COLL = ("https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
                "capella-open-data-by-product-type/capella-open-data-geo/collection.json")
CAPELLA_COLL_CACHE = PROJ / "exp1_gan_spike" / "capella_geo_collection.json"   # read-only
UMBRA = "https://umbra-open-data-catalog.s3.amazonaws.com"
S3NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

PILOT = [(24, "2025-08-03", "2025-08-11", "Kilauea summit, Hawaii (Capella HH)"),
         (24, "2025-09-08", "2025-09-16", "Kilauea summit, Hawaii (Capella HH)"),
         (31, "2025-03-28", "2025-04-06", "San Jose, California (Capella HH)"),
         (32, "2025-04-12", "2025-04-21", "Mojave, California (Capella HH)"),
         (471, "2023-11-21", "2023-12-05", "Pilbara mine, Western Australia (Capella HH)"),
         (943, "2025-06-29", "2025-07-09", "Bangkok, Thailand (Umbra VV)"),
         (1312, "2024-05-03", "2024-05-18", "Indiana farmland, planting (Umbra VV)"),
         (1312, "2023-10-05", "2023-10-22", "Indiana farmland, harvest (Umbra VV)"),
         (1340, "2024-08-10", "2024-08-29", "Western Sydney airport works (Umbra VV)")]


def db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(a)


# ------------------------------------------------------------------ X-band hrefs
def _s3_list(prefix, pat):
    s, token, out = requests.Session(), None, []
    while True:
        p = {"list-type": "2", "prefix": prefix}
        if token:
            p["continuation-token"] = token
        root = ET.fromstring(s.get(UMBRA, params=p, timeout=120).content)
        out += [k.text for k in root.findall("s3:Contents/s3:Key", S3NS) if pat.search(k.text)]
        if root.findtext("s3:IsTruncated", namespaces=S3NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=S3NS)


def resolve_hrefs(col, ids):
    known = json.loads(HREFS.read_text()) if HREFS.exists() else {}
    need = [i for i in ids if not known.get(i)]
    cap = [i for i in need if col.loc[i, "provider"] == "capella"]
    umb = [i for i in need if col.loc[i, "provider"] == "umbra"]
    if cap:
        coll = json.loads(CAPELLA_COLL_CACHE.read_text()) if CAPELLA_COLL_CACHE.exists() else requests.get(CAPELLA_COLL, timeout=60).json()
        links = {Path(l["href"]).stem: urljoin(CAPELLA_COLL, l["href"]) for l in coll["links"] if l["rel"] == "item"}
        for i in cap:
            it = requests.get(links[i], timeout=60).json()
            tif = [a["href"] for a in it["assets"].values() if a["href"].endswith(".tif") and "preview" not in a["href"]]
            known[i] = tif[0] if tif else None
    if umb:
        # the STAC asset paths moved in the bucket (exp2 fix_umbra_hrefs.py); match GEC files by
        # platform and start time instead: <YYYY-MM-DD-HH-MM-SS>_UMBRA-NN[_GEC|_MM].tif. 2025 items carry
        # empty asset hrefs and ship <time>_UMBRA-NN_MM.tif in place of a GEC file.
        pat = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(UMBRA-\d{2})(_GEC|_MM)?\.tif$")
        keys = _s3_list("sar-data/", pat)
        rows = []
        for k in keys:
            m = pat.search(k)
            t = pd.Timestamp(m.group(1)[:10] + "T" + m.group(1)[11:].replace("-", ":"), tz="UTC")
            rows.append((k, t, m.group(2), {"_GEC": 2, "_MM": 1}.get(m.group(3), 0)))
        g = pd.DataFrame(rows, columns=["key", "t", "plat", "gec"])
        for i in umb:
            r = col.loc[i]
            plat = str(r.platform).replace("_", "-").upper()
            c = g[(g.plat == plat) & ((g.t - pd.Timestamp(r.datetime)).abs() < pd.Timedelta(seconds=90))]
            c = c.sort_values("gec", ascending=False)
            known[i] = f"{UMBRA}/{quote(c.key.iloc[0])}" if len(c) else None
    HREFS.write_text(json.dumps(known, indent=0))
    return known


# ------------------------------------------------------------------ readers
def read_x(href, epsg, x0, y1):
    """X-band amplitude on the 2.5 m grid (average resampling from the overview just finer than 2.5 m)."""
    with rasterio.open(href) as s0:
        native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
        ovs = s0.overviews(1)
    lvl = None
    for i, f in enumerate(ovs):
        if native * f <= XRES / 2:
            lvl = i
    with (rasterio.open(href, overview_level=lvl) if lvl is not None else rasterio.open(href)) as s, WarpedVRT(
            s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, XRES, XRES), width=XN, height=XN,
            resampling=Resampling.average, nodata=0, dtype="float32") as v:
        a = v.read(1)
    a[a <= 0] = np.nan
    return a


def read_s1(items, epsg, x0, y1, pol="vv"):
    out = None
    for it in items:
        with rasterio.open(it.assets[pol].href) as s, WarpedVRT(
                s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, RES, RES), width=N, height=N,
                resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
            a = v.read(1)
        a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
        out = a if out is None else np.where(np.isfinite(out), out, a)
        if np.isfinite(out).all():
            break
    return out


def s1_items(lon0, lat0, lon1, lat1, day, rel, state):
    import planetary_computer as pc
    from pystac_client import Client
    c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    its = [i for i in c.search(collections=["sentinel-1-rtc"], bbox=[lon0, lat0, lon1, lat1], datetime=day).items()
           if i.properties.get("sat:relative_orbit") == rel and i.properties.get("sat:orbit_state") == state]
    return its


# ------------------------------------------------------------------ change rule (exp2 change_mask.py)
def change(a_db, b_db, offset=0.0):
    m = np.isfinite(a_db) & np.isfinite(b_db)
    k = np.ones((3, 3)) / 9
    d = ndimage.convolve(np.where(m, b_db - offset, 0), k) - ndimage.convolve(np.where(m, a_db, 0), k)
    ok = ndimage.binary_erosion(m, iterations=1)
    return (np.abs(d) > THRESH) & ok, d, ok


def best_shift(ref, mov, r=10):
    """Integer (dy, dx) that best aligns mov to ref (both dB, NaN allowed), by correlation."""
    def z(a):
        a = ndimage.uniform_filter(np.nan_to_num(a - np.nanmean(a)), 3)
        return a
    A, B = z(ref), z(mov)
    best, arg = -2, (0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            Bs = np.roll(B, (dy, dx), axis=(0, 1))[r:-r, r:-r]
            Ac = A[r:-r, r:-r]
            c = np.corrcoef(Ac.ravel(), Bs.ravel())[0, 1]
            if c > best:
                best, arg = c, (dy, dx)
    if best < 0.15:          # no reliable match: keep the geocoding as is
        return (0, 0), best
    return arg, best


def shift(a, s, fill):
    out = np.roll(a, s, axis=(0, 1))
    dy, dx = s
    if dy > 0:
        out[:dy] = fill
    elif dy < 0:
        out[dy:] = fill
    if dx > 0:
        out[:, :dx] = fill
    elif dx < 0:
        out[:, dx:] = fill
    return out


def region_precision(flag, xd, valid, min_region=MIN_REGION):
    lab, n = ndimage.label(flag, structure=np.ones((3, 3)))
    if n == 0:
        return 0, np.nan
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    hit = ndimage.mean(xd.astype(float), lab, index=np.arange(1, n + 1))
    keep = sizes >= min_region
    return int(keep.sum()), (float((hit[keep] >= EXPLAINED).mean()) if keep.any() else np.nan)


# ------------------------------------------------------------------ one pair
def run_pair(pr, col, hrefs, foot):
    a, b = col.loc[pr.xid1], col.loc[pr.xid2]
    inter = foot.loc[pr.xid1].intersection(foot.loc[pr.xid2])
    c = inter.centroid
    zone = int((c.x + 180) // 6) + 1
    epsg = (32600 if c.y >= 0 else 32700) + zone
    fw = Transformer.from_crs(4326, epsg, always_xy=True)
    cx, cy = fw.transform(c.x, c.y)
    x0, y1 = round(cx - N * RES / 2, -1), round(cy + N * RES / 2, -1)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon0, lat0 = inv.transform(x0, y1 - N * RES)
    lon1, lat1 = inv.transform(x0 + N * RES, y1)

    xa = read_x(hrefs[pr.xid1], epsg, x0, y1)
    xb = read_x(hrefs[pr.xid2], epsg, x0, y1)
    to10 = lambda amp: db(np.nanmean((amp ** 2).reshape(N, 4, N, 4), axis=(1, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xa10, xb10 = to10(xa), to10(xb)
    sb = s1_items(lon0, lat0, lon1, lat1, pr.s1_before, int(pr.rel_orbit), pr.orbit_state)
    sa = s1_items(lon0, lat0, lon1, lat1, pr.s1_after, int(pr.rel_orbit), pr.orbit_state)
    if not sb or not sa:
        return {"status": "no_s1_item"}
    s1b, s1a = db(read_s1(sb, epsg, x0, y1)), db(read_s1(sa, epsg, x0, y1))
    s1_flag, s1_d, s1_ok = change(s1b, s1a)

    (dy, dx), r_al = best_shift(s1a, xb10)
    xa10s, xb10s = shift(xa10, (dy, dx), np.nan), shift(xb10, (dy, dx), np.nan)
    both = np.isfinite(xa10s) & np.isfinite(xb10s)
    off = float(np.nanmedian((xb10s - xa10s)[both]))
    x_flag, x_d, x_ok = change(xa10s, xb10s, off)

    valid = s1_ok & x_ok
    if valid.mean() < 0.3:
        return {"status": "small_overlap", "valid_frac": round(float(valid.mean()), 3)}
    sf, xf = s1_flag & valid, x_flag & valid
    xd = ndimage.binary_dilation(xf, iterations=DIL) & valid
    sd = ndimage.binary_dilation(sf, iterations=DIL) & valid
    n_s1 = int(sf.sum())
    prec = float(xd[sf].mean()) if n_s1 else np.nan
    base = float(xd[valid].mean())
    n_reg, reg_prec = region_precision(sf, xd, valid)
    n_big, big_prec = region_precision(sf, xd, valid, 20)
    rng = np.random.default_rng(0)
    null, null_big = [], []
    for _ in range(20):
        s = (int(rng.integers(20, N - 20)), int(rng.integers(20, N - 20)))
        xr = np.roll(xd, s, axis=(0, 1)) & valid
        null.append(region_precision(sf, xr, valid)[1])
        null_big.append(region_precision(sf, xr, valid, 20)[1])
    both_flag = sf & xf
    sign = float((np.sign(s1_d[both_flag]) == np.sign(x_d[both_flag])).mean()) if both_flag.sum() >= 5 else np.nan
    res = {"status": "ok", "epsg": epsg, "lon": round(c.x, 4), "lat": round(c.y, 4), "valid_frac": round(float(valid.mean()), 3),
           "x_shift_px": f"{dy},{dx}", "x_align_r": round(r_al, 2), "x_offset_db": round(off, 2),
           "s1_diff_sd_db": round(float(np.std(s1_d[valid])), 2), "x_diff_sd_db": round(float(np.std(x_d[valid])), 2),
           "s1_flag_frac": round(float(sf.sum() / valid.sum()), 4), "x_change_frac": round(float(xf.sum() / valid.sum()), 4),
           "n_s1_flag_px": n_s1, "pix_precision": round(prec, 3), "x_base_rate_dil": round(base, 3),
           "lift": round(prec / base, 2) if base > 0 and n_s1 else np.nan,
           "n_s1_regions": n_reg, "region_precision": round(reg_prec, 3) if np.isfinite(reg_prec) else np.nan,
           "region_precision_null": round(float(np.nanmean(null)), 3) if np.isfinite(np.nanmean(null)) else np.nan,
           "n_s1_regions_ge20": n_big, "region_precision_ge20": round(big_prec, 3) if np.isfinite(big_prec) else np.nan,
           "region_precision_ge20_null": round(float(np.nanmean(null_big)), 3) if np.isfinite(np.nanmean(null_big)) else np.nan,
           "s1_median_diff_db": round(float(np.median(s1_d[valid])), 2),
           "pix_recall_of_x": round(float(sd[xf].mean()), 3) if xf.any() else np.nan, "sign_agree": round(sign, 3) if np.isfinite(sign) else np.nan,
           "s1_items_before": ";".join(i.id for i in sb), "s1_items_after": ";".join(i.id for i in sa)}
    tag = f"site{pr.site_id}_{pr.x1}_{pr.x2}"
    np.savez_compressed(DATA / f"{tag}.npz", s1b=s1b.astype("float16"), s1a=s1a.astype("float16"), s1_flag=sf, s1_d=s1_d.astype("float16"),
                        xa10=xa10s.astype("float16"), xb10=xb10s.astype("float16"), x_flag=xf, x_d=x_d.astype("float16"), valid=valid,
                        epsg=epsg, x0=x0, y1=y1)
    panel(tag, pr, res, xa, xb, (dy, dx), s1b, s1a, sf, s1_d, xf, x_d, valid)
    return res


def _show(ax, a, title, lo=2, hi=98, cmap="gray"):
    v = a[np.isfinite(a)]
    ax.imshow(a, cmap=cmap, vmin=np.percentile(v, lo) if v.size else 0, vmax=np.percentile(v, hi) if v.size else 1,
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def panel(tag, pr, res, xa, xb, s, s1b, s1a, sf, s1_d, xf, x_d, valid):
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.6))
    with np.errstate(divide="ignore", invalid="ignore"):
        xa_db, xb_db = db(xa ** 2), db(xb ** 2)
    sh = (s[0] * 4, s[1] * 4)
    xa_db, xb_db = shift(xa_db, sh, np.nan), shift(xb_db, sh, np.nan)
    _show(ax[0, 0], xa_db, f"X-band before {pr.x1} (2.5 m)")
    _show(ax[0, 1], xb_db, f"X-band after {pr.x2} (2.5 m)")
    xm = np.where(valid, np.clip(x_d, -8, 8), np.nan)
    ax[0, 2].imshow(xm, cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest")
    ax[0, 2].contour(xf, levels=[0.5], colors="k", linewidths=0.5)
    ax[0, 2].set_title(f"X-band change, 10 m (|d|>3 dB: {res['x_change_frac']:.1%})", fontsize=9)
    ax[0, 2].axis("off")
    outline(ax[0, 2], "truth")
    _show(ax[1, 0], s1b, f"Sentinel-1 VV {pr.s1_before}")
    _show(ax[1, 1], s1a, f"Sentinel-1 VV {pr.s1_after}")
    sm = np.where(valid, np.clip(s1_d, -8, 8), np.nan)
    ax[1, 2].imshow(sm, cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest")
    ax[1, 2].contour(sf, levels=[0.5], colors="k", linewidths=0.5)
    ax[1, 2].set_title(f"Sentinel-1 change flags (|d|>3 dB: {res['s1_flag_frac']:.1%})", fontsize=9)
    ax[1, 2].axis("off")
    outline(ax[1, 2], "model")
    # overlay: X-band after (10 m) with both masks
    _show(ax[0, 3], xb_db, "X-band after, S1 flags (green) over X change (purple)")
    big = lambda m: np.kron(m, np.ones((4, 4), bool))
    ov = np.zeros(xb_db.shape + (4,))
    ov[big(xf)] = matplotlib.colors.to_rgba(TRUTH, 0.55)
    ov[big(sf)] = matplotlib.colors.to_rgba(MODEL, 0.75)
    ax[0, 3].imshow(ov, interpolation="nearest")
    ax[1, 3].axis("off")
    txt = (f"{PILOT_LABEL.get((pr.site_id, pr.x1, pr.x2), '')}\nsite {pr.site_id}, {pr.provider} {pr.pol}, d_inc {pr.d_inc} deg\n"
           f"X-band {pr.x1} -> {pr.x2} ({pr.x_gap_d} d)\nS1 track {int(pr.rel_orbit)} {pr.orbit_state}: "
           f"{pr.s1_before} -> {pr.s1_after}\nlatency S1 after -> X after: {pr.latency_d} d\n\n"
           f"S1 flagged {res['n_s1_flag_px']} px ({res['s1_flag_frac']:.1%})\n"
           f"pixel precision {res['pix_precision']} vs base rate {res['x_base_rate_dil']} (lift {res['lift']})\n"
           f"regions >= {MIN_REGION} px: {res['n_s1_regions']}, explained {res['region_precision']} "
           f"(chance {res['region_precision_null']})\nX change recalled by S1: {res['pix_recall_of_x']}\n"
           f"X shift {res['x_shift_px']} px (r={res['x_align_r']}), X offset {res['x_offset_db']} dB\n"
           f"diff sd: S1 {res['s1_diff_sd_db']} dB, X {res['x_diff_sd_db']} dB")
    ax[1, 3].text(0, 1, txt, va="top", ha="left", fontsize=9, family="monospace", transform=ax[1, 3].transAxes)
    fig.text(0.5, 0.005, "green outline = Sentinel-1 change flags (detector output)     "
             "purple outline = X-band change (reference it is compared against)", ha="center", va="bottom",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(PANELS / f"{tag}.png", dpi=90)
    plt.close(fig)


PILOT_LABEL = {(s, a, b): lab for s, a, b, lab in PILOT}


def main():
    DATA.mkdir(exist_ok=True)
    PANELS.mkdir(exist_ok=True)
    col = pd.read_csv(ROOT / "xband_collects.csv").set_index("xband_id")
    foot = gpd.read_file(ROOT / "xband_collects.geojson").set_index("xband_id").geometry
    tp = pd.read_csv(ROOT / "tasking_pairs.csv")
    sel = []
    for sid, x1, x2, _ in PILOT:
        c = tp[(tp.site_id == sid) & (tp.x1 == x1) & (tp.x2 == x2)].copy()
        c["ov"] = [foot.loc[r.xid1].intersection(foot.loc[r.xid2]).area / min(foot.loc[r.xid1].area, foot.loc[r.xid2].area)
                   for r in c.itertuples()]
        sel.append(c.sort_values(["ov", "d_inc"], ascending=[False, True]).iloc[0])
    sel = pd.DataFrame(sel).reset_index(drop=True)
    hrefs = resolve_hrefs(col, set(sel.xid1) | set(sel.xid2))
    rows = []
    for pr in sel.itertuples():
        t0 = time.time()
        if not hrefs.get(pr.xid1) or not hrefs.get(pr.xid2):
            r = {"status": "no_xband_href"}
        else:
            try:
                r = run_pair(pr, col, hrefs, foot)
            except Exception as e:
                traceback.print_exc()
                r = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        print(pr.site_id, pr.x1, pr.x2, f"{time.time() - t0:.0f}s", {k: r.get(k) for k in
              ("status", "s1_flag_frac", "x_change_frac", "pix_precision", "x_base_rate_dil", "lift", "n_s1_regions",
               "region_precision", "region_precision_null", "region_precision_ge20", "region_precision_ge20_null", "error")}, flush=True)
        rows.append({**pr._asdict(), **r})
    out = pd.DataFrame(rows).drop(columns=["Index"])
    out.to_csv(ROOT / "pilot_pairs.csv", index=False)
    print(out.status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
