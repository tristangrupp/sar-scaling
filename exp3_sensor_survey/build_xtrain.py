"""Round 2 step 4: training data for arm B (tolerant X-band edge loss).

Sites: open X-band collects (Umbra, ICEYE, Capella HH and VV) whose centre lies inside a TRAINING frame (validation
frames excluded) plus extra X-band-rich frames added only for arm B, and outside every holdout frame (round 1
near/far plus the geometry-only frames). Sites (about 2 km cells) are split by hash: 60 % xtrain, 20 % xval (for
tuning the edge weight), 20 % xtest (scored, never trained on). Up to 4 collects per site.

Per collect (2.56 km window on the frame's 10 m grid, as in xband_check):
  x    20 MHz mixed-mode HH dB, 128 x 128 (20 m)        y   40 MHz HH dB, 256 x 256 (10 m)
  xe   X-band edges (Canny on the registered X-band)     dt  distance to the nearest X-band edge (px)
  infra  ESA WorldCover built-up (class 50) & X-band valid & not layover/shadow
Layover/shadow: NISAR static layers are not yet published, so they come from the Copernicus 30 m DEM and the
pass direction: layover where the slope facing the radar exceeds the incidence angle, shadow where the slope
away from it exceeds 90 deg minus incidence (incidence from the GCOV's radar-grid cube when present, else 40 deg).
Writes DATA/xtrain/<split>/<provider>_<site>_<collect>.npz and xtrain.csv.
"""
from __future__ import annotations
import os, sys, json, zlib, traceback
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import xband_check as xc                     # noqa: E402  read_x, worldcover, register, read_win, hrefs helpers
from config import DATA, VAL_FRAMES          # noqa: E402
from evaluate import edges                   # noqa: E402
from nisar_io import open_remote             # noqa: E402

N, PX = xc.N, xc.PX
EXTRA_FRAMES = [(23, 73), (154, 21), (135, 70), (55, 20),    # X-band-rich frames added for arm B only
                (139, 19), (35, 75), (107, 75), (43, 24), (43, 26), (109, 64), (111, 14), (133, 76),
                (119, 22), (120, 75), (104, 75), (40, 4), (126, 79), (25, 79), (95, 70)]
# sites inside any holdout frame are always dropped (see sites()), so these frames cannot leak holdout ground
GEO_HOLD = [(172, 15), (122, 76), (133, 81), (82, 22)]


def split_of(site):
    h = zlib.crc32(site.encode()) % 10
    return "xtrain" if h < 6 else ("xval" if h < 8 else "xtest")


def dem_mask(client, epsg, x0, y1, bbox_ll, ascending, inc_deg=40.0):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    items = list(client.search(collections=["cop-dem-glo-30"], bbox=bbox_ll).items())
    dem = np.full((N, N), np.nan, "float32")
    for it in items:
        with rasterio.open(it.assets["data"].href) as s, WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, PX, PX),
                                                                   width=N, height=N, resampling=Resampling.bilinear) as v:
            a = v.read(1).astype("float32")
        dem = np.where(np.isfinite(dem), dem, np.where(a > -1000, a, np.nan))
    if not np.isfinite(dem).any():
        return np.zeros((N, N), bool), np.nan
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    gy, gx = np.gradient(dem, PX)               # gy: d/drow (south), gx: d/dcol (east)
    # NISAR looks left. Heading about -12 deg (ascending) or 192 deg (descending); look azimuth = heading - 90
    head = np.deg2rad(-12.0 if ascending else 192.0)
    look = head - np.pi / 2
    ex, ny = np.sin(look), np.cos(look)          # unit vector from radar to ground, east / north components
    slope_toward = -(gx * ex + (-gy) * ny)       # rise toward the radar (positive = faces the radar)
    ang = np.rad2deg(np.arctan(slope_toward))
    inc = inc_deg
    lay = ang > inc
    sh = ang < -(90 - inc)
    return lay | sh, float(np.nanmax(np.abs(ang)))


def sites():
    sel = pd.read_csv(ROOT / "selection.csv")
    sel = sel[(sel.role == "train")]
    sel = sel[~sel.apply(lambda r: (r.track, r.frame) in set(VAL_FRAMES), axis=1)]
    sel = sel.sort_values("date").groupby(["track", "frame"]).head(1)
    pairs = pd.read_csv(DATA / "mixed_pairs.csv")
    pf = gpd.read_file(DATA / "pair_frames.gpkg")
    add = []
    for tr, fr in EXTRA_FRAMES:
        p = pairs[(pairs.track == tr) & (pairs.frame == fr)].sort_values("t0_a")
        if len(p):
            add.append({"role": "train", "track": tr, "frame": fr, "url_m": p.iloc[0].url_m, "url_a": p.iloc[0].url_a, "dir_": p.iloc[0].dir})
    sel = pd.concat([sel, pd.DataFrame(add)], ignore_index=True)
    frames = gpd.read_file(DATA / "selected_frames.gpkg")
    hold = list(frames[frames.role != "train"].geometry) + list(pf[pf.apply(lambda r: (r.track, r.frame) in set(GEO_HOLD), axis=1)].geometry)
    hold_u = gpd.GeoSeries(hold, crs=4326).union_all()
    X = gpd.read_file(DATA / "xband_all.gpkg")
    ic = gpd.read_file(xc.ROOT.parent / "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson")[["xband_id", "grd_href"]]
    X = X.merge(ic, on="xband_id", how="left")
    X["c"] = X.geometry.representative_point()
    out = []
    for s in sel.itertuples():
        fp = pf[(pf.track == s.track) & (pf.frame == s.frame)].geometry.iloc[0]
        xi = X[X.c.within(fp) & ~X.c.within(hold_u)].copy()
        xi["site"] = xi.provider + "_" + (xi.c.x / 0.02).round().astype(int).astype(str) + "_" + (xi.c.y / 0.02).round().astype(int).astype(str)
        d = getattr(s, "dir_", None)
        asc = (d == "A") if isinstance(d, str) else ("_A_" in s.url_a.split("/")[-1][20:30])
        for x in xi.itertuples():
            out.append({"track": s.track, "frame": s.frame, "url_m": s.url_m, "url_a": s.url_a, "asc": asc,
                        "xband_id": x.xband_id, "provider": x.provider, "x_dt": str(x.datetime), "lon": x.c.x, "lat": x.c.y,
                        "grd_href": x.grd_href, "site": x.site, "split": split_of(x.site)})
    return pd.DataFrame(out)


def main():
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer
    from scipy import ndimage as ndi
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    cand = sites()
    print("collects in training frames:", len(cand), cand.groupby(["provider", "split"]).size().to_dict(), flush=True)
    hc = DATA / "xband_hrefs.json"
    hrefs = json.loads(hc.read_text()) if hc.exists() else {}
    um = set(cand[cand.provider == "umbra"].xband_id) - set(hrefs)
    if um:
        hrefs.update(xc.spike_prep.umbra_gec_hrefs(um))
    ca = set(cand[cand.provider == "capella"].xband_id) - set(hrefs)
    if ca:
        hrefs.update(xc.add_capella.hrefs(ca))
    hc.write_text(json.dumps({k: v for k, v in hrefs.items() if v}))
    cand["href"] = np.where(cand.provider == "iceye", cand.grd_href, cand.xband_id.map(hrefs))
    cand = cand[cand.href.notna()].sort_values("x_dt", ascending=False).groupby("site").head(4)
    cand = cand.sort_values(["url_a", "site"])
    print("with hrefs:", len(cand), cand.groupby(["provider", "split"]).size().to_dict(), flush=True)
    prev = pd.read_csv(ROOT / "xtrain.csv") if (ROOT / "xtrain.csv").exists() else pd.DataFrame(columns=["xband_id"])
    rows, opened = prev.to_dict("records"), {}
    cand = cand[~cand.xband_id.isin(set(prev.xband_id))]
    print("new collects to process:", len(cand), flush=True)
    wc_cache = {}
    for i, c in enumerate(cand.itertuples()):
        rec = {"site": c.site, "split": c.split, "provider": c.provider, "xband_id": c.xband_id, "track": c.track, "frame": c.frame}
        try:
            if c.url_a not in opened:
                opened = {c.url_a: (open_remote(c.url_m), open_remote(c.url_a))}
            M, A = opened[c.url_a]
            gm, ga = M["science/LSAR/GCOV/grids/frequencyA"], A["science/LSAR/GCOV/grids/frequencyA"]
            epsg = int(ga["projection"][()])
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(c.lon, c.lat)
            xe0 = ga["xCoordinates"][0] - 5; ye0 = ga["yCoordinates"][0] + 5
            x0 = xe0 + np.round((x - N * PX / 2 - xe0) / 80) * 80; y1 = ye0 - np.round((ye0 - y - N * PX / 2) / 80) * 80
            xin = xc.read_win(gm, (x0, y1), N // 2, 20.0); truth = xc.read_win(ga, (x0, y1), N, 10.0)
            if xin is None or truth is None or np.isfinite(truth).mean() < 0.95 or np.isfinite(xin).mean() < 0.95:
                rec["status"] = "nisar_outside"; rows.append(rec); continue
            to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
            lo0, la0 = to_ll.transform(x0, y1 - N * PX); lo1, la1 = to_ll.transform(x0 + N * PX, y1)
            key = (c.url_a, x0, y1)
            if key not in wc_cache:
                wc = xc.worldcover(client, epsg, x0, y1, (lo0, la0, lo1, la1))
                lsm, maxang = dem_mask(client, epsg, x0, y1, (lo0, la0, lo1, la1), bool(c.asc))
                wc_cache = {key: (wc, lsm, maxang)}
            wc, lsm, maxang = wc_cache[key]
            rec["built_frac"] = float((wc == 50).mean()); rec["layover_shadow_frac"] = float(lsm.mean())
            if rec["built_frac"] < 0.10:
                rec["status"] = "little_builtup"; rows.append(rec); continue
            amp = xc.read_x(c.href, epsg, x0, y1)
            xm = np.isfinite(amp) & (amp > 0)
            if xm.mean() < 0.4:
                rec["status"] = "xband_partial"; rows.append(rec); continue
            xdb = np.where(xm, 20 * np.log10(np.where(xm, amp, 1)), np.nan)
            tr = np.where(np.isfinite(truth), truth, np.nanmedian(truth))
            xr, xmr, rec["shift_m"], rec["r40"] = xc.register(xdb, tr, xm)
            if not (rec["r40"] >= 0.2):
                rec["status"] = "low_corr"; rows.append(rec); continue
            ex = edges(np.where(xmr, xr, np.nanmedian(xr)))
            dt = ndi.distance_transform_edt(~ex).astype("float32")
            infra = (wc == 50) & xmr & ~lsm
            rec["infra_px"] = int(infra.sum())
            if infra.sum() < 2000:
                rec["status"] = "little_infra"; rows.append(rec); continue
            rec["status"] = "ok"
            d = DATA / "xtrain" / c.split; d.mkdir(parents=True, exist_ok=True)
            name = f"{c.provider}_{c.site}_{str(c.xband_id)[-12:]}"
            np.savez_compressed(d / f"{name}.npz", x=xin.astype("float16"), y=tr.astype("float16"), xe=ex, dt=dt.astype("float16"),
                                infra=infra, xband=xr.astype("float16"), xmask=xmr,
                                meta=json.dumps({"epsg": epsg, "x0": float(x0), "y1": float(y1), "site": c.site, "provider": c.provider}))
            rec["file"] = name
            rows.append(rec)
            print(i, c.split, c.provider, c.site, "ok", round(rec["r40"], 2), flush=True)
        except Exception as e:
            rec.update(status="error", err=repr(e)[:200]); rows.append(rec)
            print(i, "ERROR", repr(e)[:200], traceback.format_exc()[-300:], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "xtrain.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "xtrain.csv", index=False)
    print(df.groupby(["provider", "split", "status"]).size().to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
