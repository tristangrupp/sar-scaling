"""Stage 1 geometry check: do arm A's output edges sit where open X-band (Umbra, ICEYE, Capella) puts them?

Sites: every X-band collect whose footprint centre lies inside a held-out NISAR frame (near or far), on land
with ESA WorldCover built-up >= 10 % of a 2.56 km window, one collect per ~2 km site per provider. Near-holdout
sites inside any training frame are dropped. For each site:
  1. read a 2.56 km window from the held-out NISAR pair (20 MHz input 128 px, 40 MHz truth 256 px), remotely
  2. warp X-band amplitude onto the same 10 m UTM grid (average resampling from an overview), to dB
  3. register X-band to the 40 MHz truth, coarse then fine: phase correlation on 40 m block means
     (integer, <= 400 m), then at 10 m with 4x upsampling (<= 30 m). Reject if the 40 m correlation after the
     shift stays below 0.2. The shift is fit against the truth, never against a model output.
  4. Canny edges (quantile thresholds, sigma 1.5 px) of X-band, bicubic, CNN and the 40 MHz truth
  5. chamfer distance in metres on built-up pixels only, both directions:
       p2x  from the image's edges to the nearest X-band edge   (invented or misplaced edges)
       x2p  from X-band edges to the nearest image edge         (missed edges)
The 40 MHz truth's own chamfer is the floor a 10 m L-band image can reach against X-band geometry.
"""
from __future__ import annotations
from config import DATA  # noqa: E402
import os
os.environ.update(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                  GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
import json, sys, traceback
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import rasterio, torch
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


spike_prep = _load("prep", ROOT.parent / "exp1_gan_spike/prep.py")                 # Umbra GEC hrefs, ICEYE GCP warp
add_capella = _load("add_capella", ROOT.parent / "exp1_gan_spike/add_capella.py")  # Capella GEO hrefs
del sys.modules["prep"]
sys.path.insert(0, str(ROOT))
from evaluate import bicubic, chamfer, edges, predict   # noqa: E402
from train import DEV, Gen, to_n                        # noqa: E402
from nisar_io import open_remote                        # noqa: E402
from infer_transfer import read_win                     # noqa: E402

N, PX = 256, 10.0
MAX_PER_PROVIDER = 60


def read_x(href, epsg, x0, y1):
    tr = from_origin(x0, y1, PX, PX)
    kw = dict(crs=f"EPSG:{epsg}", transform=tr, width=N, height=N, resampling=Resampling.average, nodata=0, dtype="float32")
    with rasterio.open(href) as s0:
        if s0.crs is None:                       # ICEYE GRD: GCPs only
            return spike_prep.warp_gcps(s0, epsg, tr, N, Resampling.average)
        native = abs(s0.res[0]) * (111_000 if s0.crs.is_geographic else 1)
        ovs = s0.overviews(1)
    lvl = None
    for i, f in enumerate(ovs):
        if native * f <= PX / 2:
            lvl = i
    with (rasterio.open(href, overview_level=lvl) if lvl is not None else rasterio.open(href)) as s, WarpedVRT(s, **kw) as v:
        return v.read(1)


def worldcover(client, epsg, x0, y1, bbox_ll):
    items = list(client.search(collections=["esa-worldcover"], bbox=bbox_ll).items())
    items.sort(key=lambda it: it.properties.get("esa_worldcover:product_version", ""), reverse=True)
    out = np.zeros((N, N), "uint8")
    for it in items:
        with rasterio.open(it.assets["map"].href) as s, WarpedVRT(s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, PX, PX),
                                                                   width=N, height=N, resampling=Resampling.nearest) as v:
            a = v.read(1)
        out = np.where(out == 0, a, out)
    return out


def register(xdb, tdb, xm):
    """coarse (40 m) then fine (10 m) shift of xdb onto tdb; returns shifted xdb, its mask, shift (m), corr at 40 m"""
    tm = np.isfinite(tdb)
    X = np.where(xm, xdb, np.nanmedian(xdb[xm]))
    T = np.where(tm, tdb, np.nanmedian(tdb[tm]))
    b = lambda a: a.reshape(N // 4, 4, N // 4, 4).mean(axis=(1, 3))
    sh, _, _ = phase_cross_correlation(b(T) - b(T).mean(), b(X) - b(X).mean(), upsample_factor=1)
    sh = np.clip(sh, -10, 10) * 4
    X1 = ndi.shift(X, sh, order=1, mode="nearest"); m1 = ndi.shift(xm.astype(float), sh, order=0) > 0.5
    sh2, _, _ = phase_cross_correlation(T - T.mean(), X1 - X1.mean(), upsample_factor=4)
    sh2 = np.clip(sh2, -3, 3)
    X2 = ndi.shift(X1, sh2, order=1, mode="nearest"); m2 = ndi.shift(m1.astype(float), sh2, order=0) > 0.5
    vb = (m2 & tm).reshape(N // 4, 4, N // 4, 4).all(axis=(1, 3))
    r = np.corrcoef(b(X2)[vb], b(T)[vb])[0, 1] if vb.sum() > 50 else np.nan
    tot = (sh + sh2) * PX
    return X2, m2, float(np.hypot(*tot)), float(r)


def sites():
    sel = pd.read_csv(ROOT / "selection.csv")
    sel = sel[sel.role.isin(["near", "far"])].sort_values("date").groupby(["track", "frame"]).head(1)   # one date per frame
    frames = gpd.read_file(str(DATA) + "/selected_frames.gpkg")
    # geometry-only holdout frames (never used for tiles or training), to raise the X-band sample:
    # Saudi Arabia x2 and Philippines (>= 800 km from any training frame), north China (about 340 km away)
    extra = {(172, 15): "far", (122, 76): "far", (133, 81): "far", (82, 22): "near"}
    pf = gpd.read_file(str(DATA) + "/pair_frames.gpkg")
    pairs = pd.read_csv(str(DATA) + "/mixed_pairs.csv")
    add = []
    for (tr, fr), role in extra.items():
        p = pairs[(pairs.track == tr) & (pairs.frame == fr)].sort_values("t0_a").iloc[0]
        g = pf[(pf.track == tr) & (pf.frame == fr)]
        add.append({"role": role, "reg": f"geo_{tr}_{fr}", "track": tr, "frame": fr, "url_m": p.url_m, "url_a": p.url_a})
        frames = pd.concat([frames, g.assign(role=role)[["track", "frame", "role", "geometry"]]], ignore_index=True)
    sel = pd.concat([sel, pd.DataFrame(add)], ignore_index=True)
    frames = gpd.GeoDataFrame(frames, crs=4326)
    # training frames, but only X-band sites at least 3 km from every training tile ("unseen ground")
    tr_sel = pd.read_csv(ROOT / "selection.csv")
    tr_sel = tr_sel[tr_sel.role == "train"].sort_values("date").groupby(["track", "frame"]).head(1).assign(role="train_frame_unseen")
    sel = pd.concat([sel, tr_sel], ignore_index=True)
    tt = pd.read_csv(ROOT / "tiles.csv"); tt = tt[tt.role == "train"]
    from shapely.geometry import box as _box
    polys = []
    for e, g in tt.groupby("epsg"):
        gg = gpd.GeoSeries([_box(r.x0, r.y1 - 5120, r.x0 + 5120, r.y1) for r in g.itertuples()], crs=int(e)).to_crs(4326)
        polys += list(gg)
    seen = gpd.GeoSeries(polys, crs=4326).to_crs(3857).buffer(3000).to_crs(4326).union_all()
    trn = frames[frames.role == "train"].union_all()
    X = gpd.read_file(str(DATA) + "/xband_all.gpkg")
    ic = gpd.read_file(ROOT.parent / "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson")[["xband_id", "grd_href"]]
    X = X.merge(ic, on="xband_id", how="left")
    X["c"] = X.geometry.representative_point()
    out = []
    for s in sel.itertuples():
        fp = frames[(frames.track == s.track) & (frames.frame == s.frame)].geometry.iloc[0]
        if s.role == "train_frame_unseen":
            xi = X[X.c.within(fp) & ~X.c.within(seen)].copy()
        else:
            xi = X[X.c.within(fp) & ~X.c.within(trn)].copy()
        xi["site"] = xi.provider + "_" + (xi.c.x / 0.02).round().astype(int).astype(str) + "_" + (xi.c.y / 0.02).round().astype(int).astype(str)
        xi = xi.sort_values("datetime", ascending=False)
        for x in xi.itertuples():
            out.append({"role": s.role, "reg": s.reg, "track": s.track, "frame": s.frame, "url_m": s.url_m, "url_a": s.url_a,
                        "xband_id": x.xband_id, "provider": x.provider, "x_dt": str(x.datetime), "lon": x.c.x, "lat": x.c.y,
                        "grd_href": x.grd_href, "site": x.site})
    return pd.DataFrame(out)


def main():
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    cand = sites()
    print("candidate sites:", len(cand), cand.groupby(["role", "provider"]).size().to_dict(), flush=True)
    hc = Path(str(DATA) + "/xband_hrefs.json")
    hrefs = json.loads(hc.read_text()) if hc.exists() else {}
    um = set(cand[cand.provider == "umbra"].xband_id) - set(hrefs)
    if um:
        hrefs.update(spike_prep.umbra_gec_hrefs(um))
    ca = set(cand[cand.provider == "capella"].xband_id) - set(hrefs)
    if ca:
        hrefs.update(add_capella.hrefs(ca))
    hc.write_text(json.dumps({k: v for k, v in hrefs.items() if v}))
    cand["href"] = np.where(cand.provider == "iceye", cand.grd_href, cand.xband_id.map(hrefs))
    cand = cand[cand.href.notna()]
    cand = cand.sort_values("x_dt", ascending=False).groupby("site").head(4)
    print("with hrefs:", len(cand), cand.groupby("provider").size().to_dict(), flush=True)

    G = None
    if (ROOT / "ckpt/x2.pt").exists():
        G = Gen(2).to(DEV).eval()
        G.load_state_dict(torch.load(ROOT / "ckpt/x2.pt"))
    rows, opened = [], {}
    (ROOT / "eval/xband").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cand = cand.iloc[rng.permutation(len(cand))]
    # cap per provider and role before reading, then group by pair so each remote pair opens once
    cand = cand.groupby(["provider", "role"], group_keys=False).head(30).sort_values(["url_a", "site"])
    for i, c in enumerate(cand.itertuples()):
        if sum(r.get("status") == "ok" and r["provider"] == c.provider for r in rows) >= MAX_PER_PROVIDER:
            continue
        rec = {"site": c.site, "role": c.role, "reg": c.reg, "xband_id": c.xband_id, "provider": c.provider, "x_dt": c.x_dt}
        try:
            if c.url_a not in opened:
                opened = {c.url_a: (open_remote(c.url_m), open_remote(c.url_a))}
            M, A = opened[c.url_a]
            gm, ga = M["science/LSAR/GCOV/grids/frequencyA"], A["science/LSAR/GCOV/grids/frequencyA"]
            epsg = int(ga["projection"][()])
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(c.lon, c.lat)
            # window aligned to the frame's 80 m grid (both products share the origin)
            xe = ga["xCoordinates"][0] - 5; ye = ga["yCoordinates"][0] + 5
            x0 = xe + np.round((x - N * PX / 2 - xe) / 80) * 80; y1 = ye - np.round((ye - y - N * PX / 2) / 80) * 80
            xin = read_win(gm, (x0, y1), N // 2, 20.0); truth = read_win(ga, (x0, y1), N, 10.0)
            if xin is None or truth is None or np.isfinite(truth).mean() < 0.95 or np.isfinite(xin).mean() < 0.95:
                rec["status"] = "nisar_outside"; rows.append(rec); continue
            to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
            lo0, la0 = to_ll.transform(x0, y1 - N * PX); lo1, la1 = to_ll.transform(x0 + N * PX, y1)
            wc = worldcover(client, epsg, x0, y1, (lo0, la0, lo1, la1))
            rec["built_frac"] = float((wc == 50).mean())
            if rec["built_frac"] < 0.10:
                rec["status"] = "little_builtup"; rows.append(rec); continue
            xin = np.where(np.isfinite(xin), xin, np.nanmedian(xin))
            truth = np.where(np.isfinite(truth), truth, np.nanmedian(truth))
            imgs = {"bicubic": bicubic(to_n(xin), 2), **({"cnn": predict(G, to_n(xin))} if G is not None else {}), "truth40": truth}
            amp = read_x(c.href, epsg, x0, y1)
            xm = np.isfinite(amp) & (amp > 0)
            rec["x_cov"] = float(xm.mean())
            if xm.mean() < 0.4:
                rec["status"] = "xband_partial"; rows.append(rec); continue
            xdb = np.where(xm, 20 * np.log10(np.where(xm, amp, 1)), np.nan)
            xr, xmr, rec["shift_m"], rec["r40"] = register(xdb, truth, xm)
            built = (wc == 50) & xmr
            rec["built_px"] = int(built.sum())
            if not (rec["r40"] >= 0.2):
                rec["status"] = "low_corr"; rows.append(rec); print(i, rec, flush=True); continue
            rec["status"] = "ok"
            ex = edges(np.where(xmr, xr, np.nanmedian(xr)))
            for k, im in imgs.items():
                rec[f"{k}_p2x"], rec[f"{k}_x2p"] = chamfer(edges(im), ex, built, PX)
            rows.append(rec)
            print(i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
            if sum(r.get("status") == "ok" and r["provider"] == c.provider for r in rows) <= 3:
                np.savez_compressed(ROOT / f"eval/xband/{c.provider}_{c.site}.npz", xband=xr.astype("float16"), xmask=xmr,
                                    built=built, inp=xin.astype("float16"), **{k: v.astype("float16") for k, v in imgs.items()})
        except Exception as e:
            rec.update(status="error", err=repr(e)[:200]); rows.append(rec)
            print(i, "ERROR", repr(e)[:200], traceback.format_exc()[-400:], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "eval/xband_scores.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "eval/xband_scores.csv", index=False)
    ok = df[df.status == "ok"]
    cols = [f"{k}_{d}" for k in ["bicubic", "cnn", "truth40"] for d in ["p2x", "x2p"] if f"{k}_{d}" in df]
    print(df.groupby(["provider", "status"]).size().to_string(), flush=True)
    if len(ok):
        s = ok.groupby(["provider", "role"])[cols + ["shift_m", "r40"]].median().round(2)
        s["n"] = ok.groupby(["provider", "role"]).size()
        allp = ok.groupby("provider")[cols + ["shift_m", "r40"]].median().round(2)
        allp["n"] = ok.groupby("provider").size(); allp["role"] = "all"
        s = pd.concat([s.reset_index(), allp.reset_index()]).sort_values(["provider", "role"])
        print(s.to_string(), flush=True)
        s.to_csv(ROOT / "eval/xband_summary.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
