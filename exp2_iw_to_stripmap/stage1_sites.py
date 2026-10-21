"""Stage 1 sites: X-band collects anywhere outside the training regions, paired with IW only.

Stage 1 needs no Stripmap: it asks whether the edges a model draws from IW sit where X-band sees
edges. That lets every provider in, including Capella's HH collects (the Houston Capella site at
Baytown lies outside the Houston Stripmap swath, so Stripmap tiles alone give no Capella test).

Selection: per provider up to N collects, one per 0.5 degree cell, more than 150 km from every
training or validation tile; Capella split between HH and VV.
Per site: 3.2 km tile at the footprint centroid; IW RTC VV nearest in time (+-30 days, >= 90%
valid); X-band averaged to 5 m, registered to the IW (xband_prep.register); Canny edges; masks:
land (WorldCover), slopes over 20 degrees, layover / shadow for the X-band geometry where known;
infrastructure = OSM buildings / roads or WorldCover built-up.
Writes DATA/stage1/<provider>_<xband_id>.npz and stage1_sites.csv.
"""
from __future__ import annotations

import sys

from common import DATA, NL, ROOT, db, pc_client, read_iw, read_worldcover, tile_grid  # noqa: I001

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import ndimage

import xband_prep as xp

OUT = DATA / "stage1"
N_PER = {"capella_HH": 25, "capella_VV": 20, "umbra": 45, "iceye": 45}


def select():
    it = xp.items()
    c = it.geometry.centroid
    it["lon"], it["lat"] = c.x, c.y
    t = pd.read_csv(ROOT / "tiles.csv").merge(pd.read_csv(ROOT / "splits.csv")[["tile_id", "split"]], on="tile_id")
    tr = t[t.split.isin(["train", "val"])][["lon", "lat"]].drop_duplicates().to_numpy()
    allt = t[["lon", "lat"]].to_numpy()

    def km(lon, lat, pts):
        d = np.hypot((pts[:, 0] - lon) * np.cos(np.radians(lat)), pts[:, 1] - lat) * 111
        return float(d.min())

    it["km_to_train"] = [km(a, b, tr) for a, b in zip(it.lon, it.lat)]
    it["km_to_any_tile"] = [km(a, b, allt) for a, b in zip(it.lon, it.lat)]
    it = it[it.km_to_train > 150]
    it["cell"] = (it.lon // 0.5).astype(int).astype(str) + "_" + (it.lat // 0.5).astype(int).astype(str)
    it["grp"] = np.where(it.provider == "capella", "capella_" + it["sar:polarizations"].astype(str), it.provider)
    out = []
    for g, n in N_PER.items():
        s = it[it.grp == g].sample(frac=1, random_state=0).drop_duplicates("cell").head(n)
        out.append(s)
    return pd.concat(out)


def process(x, href):
    f = OUT / f"{x.provider}_{x.xband_id}.npz"
    if f.exists():
        return {"status": "cached"}
    epsg, x0, y1 = tile_grid(x.lon, x.lat)
    t = pd.Timestamp(x.datetime)
    its = [i for i in pc_client().search(collections=["sentinel-1-rtc"], bbox=[x.lon - .02, x.lat - .02, x.lon + .02, x.lat + .02],
                                         datetime=f"{t - pd.Timedelta(days=30):%Y-%m-%d}/{t + pd.Timedelta(days=30):%Y-%m-%d}").items()
           if "vv" in i.assets]
    its.sort(key=lambda i: abs(pd.Timestamp(i.datetime) - t))
    iw, iw_it = None, None
    for i in its[:4]:
        a = db(read_iw(i, epsg, x0, y1, NL))
        if np.isfinite(a).mean() > 0.9:
            iw, iw_it = a, i
            break
    if iw is None:
        return {"status": "no_iw"}
    amp = xp.read_xband(href, x.provider, epsg, x0, y1)
    xval = amp > 0
    if xval.mean() < 0.15:
        return {"status": "x_sparse"}
    xdb = np.where(xval, db(amp.astype("float64") ** 2), np.nan).astype("float32")
    sh, ok = xp.register(xdb, xval, iw)
    xdb = ndimage.shift(np.nan_to_num(xdb, nan=np.nanmedian(xdb)), sh, order=1, mode="nearest")
    xval = ndimage.shift(xval.astype("float32"), sh, order=0, mode="constant") > 0.5
    x10 = xp.block_mean(np.where(xval, xdb, np.nan), 2)
    c10 = float(pd.Series(x10.ravel()).corr(pd.Series(iw.ravel())))
    wc = read_worldcover(epsg, x0, y1, xp.N5, 5.0)
    land = (wc != 80) & (wc != 0)
    dem = xp.dem5(x.lon, x.lat, epsg, x0, y1)
    gy, gx = np.gradient(dem, xp.RES)
    bad = np.degrees(np.arctan(np.hypot(gx, gy))) > 20
    x_look = xp.look_azimuth(x["sat:orbit_state"], x["sar:observation_direction"] or "right")
    x_inc = float(x["view:incidence_angle"]) if pd.notna(x["view:incidence_angle"]) else np.nan
    if x_look is not None and np.isfinite(x_inc):
        bad |= xp.layover_shadow(dem, x_look, x_inc)
    infra = xp.osm_infra(f"stage1_{x.provider}_{x.xband_id}", epsg, x0, y1)
    osm = infra[0] if infra else np.zeros((xp.N5, xp.N5), bool)
    infra_m = osm | (wc == 50)
    valid = xval & land & ~bad
    e = xp.edges(xdb, xval)
    np.savez_compressed(f, iw=iw.astype("float16"), xdb=xdb.astype("float16"), edges=e, valid=valid,
                        infra=infra_m, osm=osm, wc=wc, epsg=epsg, x0=x0, y1=y1)
    return {"status": "ok", "iw_id": iw_it.id, "iw_dt_days": abs((pd.Timestamp(iw_it.datetime) - t).days),
            "iw_dir": iw_it.properties.get("sat:orbit_state"), "shift_r": sh[0], "shift_c": sh[1], "reg_ok": ok,
            "corr10": round(c10, 3), "valid": round(float(valid.mean()), 3),
            "infra_valid": round(float((valid & infra_m).mean()), 3), "osm_frac": round(float(osm.mean()), 3),
            "x_inc": x_inc}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select()
    print(sel.groupby("grp").size().to_dict(), flush=True)
    hrefs = json.loads((ROOT / "xband_hrefs.json").read_text())
    need_c = set(sel[sel.provider == "capella"].xband_id) - set(hrefs)
    if need_c:
        sys.path.insert(0, str(xp.PROJ / "exp1_gan_spike"))
        import add_capella
        hrefs.update({k: v for k, v in add_capella.hrefs(need_c).items() if v})
    for r in sel[sel.provider == "iceye"].itertuples():
        hrefs[r.xband_id] = r.grd_href
    (ROOT / "xband_hrefs.json").write_text(json.dumps(hrefs))
    rows, t0 = [], time.time()

    def run(x):
        h = hrefs.get(x.xband_id)
        if not h:
            return {"status": "no_href"}
        try:
            return process(x, h)
        except Exception as e:
            return {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}

    with ThreadPoolExecutor(5) as ex:
        futs = {ex.submit(run, x): x for _, x in sel.iterrows()}
        for n, fu in enumerate(as_completed(futs), 1):
            x = futs[fu]
            rows.append({"xband_id": x.xband_id, "provider": x.provider, "grp": x.grp, "pol": x["sar:polarizations"],
                         "lon": round(x.lon, 4), "lat": round(x.lat, 4), "x_date": x.datetime,
                         "km_to_train": round(x.km_to_train), "km_to_any_tile": round(x.km_to_any_tile), **fu.result()})
            if n % 10 == 0 or n == len(futs):
                print(f"[{n}/{len(futs)}] {time.time() - t0:.0f}s {pd.Series([r['status'] for r in rows]).value_counts().to_dict()}",
                      flush=True)
    old = ROOT / "stage1_sites.csv"
    df = pd.DataFrame(rows)
    if old.exists():
        prev = pd.read_csv(old)
        df = pd.concat([prev[~prev.xband_id.isin(df[df.status != "cached"].xband_id)], df[df.status != "cached"]])
    df.to_csv(old, index=False)
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
