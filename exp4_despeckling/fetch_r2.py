"""Round 2 downloads.

  vh      VH for the 14 round-1 sites, same items and dates as their VV stacks -> <site>_vh.npz
  n2n     18 extra Noise2Noise training sites (VV+VH, 12 dates of one orbit, 10.24 km)
          -> n2n_<site>.npz. No clean target is built for these; only date pairs are used.
  ftw     Fields of The World test-split chips (read-only from FTW_SRC) re-gridded to a 2.56 km
          UTM window, with up to 24 VV+VH dates of one orbit from the label year
          -> ftw/<country>_<aoi>.npz (vv, vh, dates, mask: 0 bg, 1 field, 2 boundary, 3 unlabelled)
"""
from __future__ import annotations

import common  # noqa: F401
from common import DATA, FTW_SRC, N2N_SITES, SITES, finish, utm_epsg

import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from fetch import STAC, read_window


def search(cl, lon, lat, t0, t1):
    return [i for i in cl.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                 datetime=f"{t0}/{t1}").items() if "vv" in i.assets]


def okey(i):
    return (i.properties.get("sat:relative_orbit"), i.properties.get("sat:orbit_state"))


def read_pol(items, pol, epsg, x0, y1, n):
    def rd(it):
        if pol not in it.assets:
            return None
        a = read_window(pc.sign(it.assets[pol].href), epsg, x0, y1, n, Resampling.bilinear, -32768, "float32")
        if a is None:
            return None
        a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
        return a
    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(rd, items))


def one_per_date(items):
    seen, out = set(), []
    for i in sorted(items, key=lambda i: i.datetime):
        d = i.datetime.strftime("%Y-%m-%d")
        if d not in seen:
            seen.add(d)
            out.append(i)
    return out


# ----------------------------------------------------------------- VH for round-1 sites
def do_vh(cl, name):
    out = DATA / f"{name}_vh.npz"
    if out.exists():
        return print("cached", out.name, flush=True)
    z = np.load(DATA / f"{name}.npz")
    lon, lat = SITES[name][:2]
    dates = [str(d) for d in z["dates"]]
    orbit = eval(str(z["orbit"]))  # noqa: S307  (our own tuple string)
    t0 = (pd.Timestamp(dates[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t1 = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    its = {i.datetime.strftime("%Y-%m-%d"): i for i in one_per_date([i for i in search(cl, lon, lat, t0, t1) if okey(i) == orbit])}
    items = [its.get(d) for d in dates]
    n = z["vv"].shape[1]
    found = [i for i in items if i is not None]
    arrs = iter(read_pol(found, "vh", int(z["epsg"]), float(z["x0"]), float(z["y1"]), n))
    vh = []
    for i in items:
        a = next(arrs) if i is not None else None
        vh.append(a if a is not None else np.full((n, n), np.nan, "float32"))
    vh = np.stack(vh)
    np.savez_compressed(out, vh=vh.astype("float32"), dates=np.array(dates))
    print(f"vh {name}: {vh.shape}, finite {np.isfinite(vh).mean():.3f}", flush=True)


# ----------------------------------------------------------------- extra N2N sites
def do_n2n(cl, name):
    out = DATA / f"n2n_{name}.npz"
    if out.exists():
        return print("cached", out.name, flush=True)
    lon, lat, lc, t0, t1 = N2N_SITES[name]
    epsg = utm_epsg(lon, lat)
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    n = 1024
    x0, y1 = float(np.floor((x - 5120) / 10) * 10), float(np.floor((y + 5120) / 10) * 10)
    items = search(cl, lon, lat, t0, t1)
    if not items:
        return print("no items", name, flush=True)
    nd = {k: len({i.datetime.date() for i in items if okey(i) == k}) for k in {okey(i) for i in items}}
    best = max(nd, key=nd.get)
    its = one_per_date([i for i in items if okey(i) == best and "vh" in i.assets])
    if len(its) > 12:
        its = [its[k] for k in np.linspace(0, len(its) - 1, 12).round().astype(int)]
    vv, vh = read_pol(its, "vv", epsg, x0, y1, n), read_pol(its, "vh", epsg, x0, y1, n)
    keep = [k for k in range(len(its)) if vv[k] is not None and vh[k] is not None
            and np.isfinite(vv[k]).mean() > 0.9 and np.isfinite(vh[k]).mean() > 0.9]
    if len(keep) < 4:
        return print("too few valid dates", name, len(keep), nd, flush=True)
    np.savez_compressed(out, vv=np.stack([vv[k] for k in keep]), vh=np.stack([vh[k] for k in keep]),
                        dates=np.array([its[k].datetime.strftime("%Y-%m-%d") for k in keep]), orbit=str(best),
                        epsg=epsg, x0=x0, y1=y1, landcover=lc)
    print(f"n2n {name} ({lc}): {len(keep)} dates, orbit {best}", flush=True)


# ----------------------------------------------------------------- FTW chips
FTW_COUNTRIES = {"france": "2020", "spain": "2020", "netherlands": "2022", "south_africa": "2018", "brazil": "2020",
                 "kenya": "2022", "vietnam": "2021"}
PER_COUNTRY = 4


def north_up_mask(path):
    with rasterio.open(path) as s:
        a, tr, crs = s.read(1), s.transform, s.crs
    if tr.e > 0:  # FTW chips are stored south-up
        a = a[::-1]
        tr = rasterio.Affine(tr.a, tr.b, tr.c, tr.d, -tr.e, tr.f + tr.e * a.shape[0])
    return a, tr, crs


def ftw_candidates(country, rng):
    import geopandas as gpd
    g = gpd.read_parquet(FTW_SRC / country / f"chips_{country}.parquet")
    g = g[g.split == "test"].sample(frac=1, random_state=0)
    picked, used = [], set()
    for r in g.itertuples():
        f = FTW_SRC / country / "label_masks" / "semantic_3class" / f"{r.aoi_id}.tif"
        if not f.exists():
            continue
        grp = r.aoi_id.rsplit("_", 1)[0]          # neighbouring chips share a prefix; take one per group
        if grp in used:
            continue
        a, _, _ = north_up_mask(f)
        fb, fu, fbg = (a == 2).mean(), (a == 3).mean(), (a == 0).mean()
        if fb >= 0.04 and fu <= 0.02 and fbg <= 0.7:
            picked.append((r.aoi_id, f, r.geometry))
            used.add(grp)
        if len(picked) >= PER_COUNTRY:
            break
    return picked


def do_ftw(cl, country, year):
    od = DATA / "ftw"
    od.mkdir(exist_ok=True)
    for aoi, f, geom in ftw_candidates(country, None):
        out = od / f"{country}_{aoi}.npz"
        if out.exists():
            print("cached", out.name, flush=True)
            continue
        try:
            lon, lat = geom.centroid.x, geom.centroid.y
            epsg = utm_epsg(lon, lat)
            from pyproj import Transformer
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
            n = 160   # FTW masks are ~5.4e-5 deg pixels: a chip spans ~1.0-1.5 km, so a 1.6 km window holds it
            x0, y1 = float(np.floor((x - 800) / 10) * 10), float(np.floor((y + 800) / 10) * 10)
            a, tr, crs = north_up_mask(f)
            # area fractions of each class per 10 m cell; boundary wins at >= 25 % so thin ribbons survive
            frac = {}
            for k in (0, 1, 2, 3):
                dst = np.full((n, n), np.nan, "float32")
                reproject((a == k).astype("float32"), dst, src_transform=tr, src_crs=crs,
                          dst_transform=from_origin(x0, y1, 10, 10), dst_crs=f"EPSG:{epsg}",
                          resampling=Resampling.average, src_nodata=None, dst_nodata=np.nan)
                frac[k] = np.nan_to_num(dst, nan=0.0)
            cover = frac[0] + frac[1] + frac[2] + frac[3]
            m = np.where(frac[1] >= frac[0], 1, 0).astype(np.uint8)
            m[frac[2] >= 0.25] = 2
            m[(cover < 0.9) | (frac[3] > 0.5)] = 3
            items = search(cl, lon, lat, f"{year}-01-01", f"{year}-12-31")
            nd = {k: len({i.datetime.date() for i in items if okey(i) == k}) for k in {okey(i) for i in items}}
            best = max(nd, key=nd.get)
            its = one_per_date([i for i in items if okey(i) == best and "vh" in i.assets])
            if len(its) > 24:
                its = [its[k] for k in np.linspace(0, len(its) - 1, 24).round().astype(int)]
            vv, vh = read_pol(its, "vv", epsg, x0, y1, n), read_pol(its, "vh", epsg, x0, y1, n)
            keep = [k for k in range(len(its)) if vv[k] is not None and vh[k] is not None
                    and np.isfinite(vv[k]).mean() > 0.95 and np.isfinite(vh[k]).mean() > 0.95]
            if len(keep) < 10:
                print("too few dates", country, aoi, len(keep), flush=True)
                continue
            np.savez_compressed(out, vv=np.stack([vv[k] for k in keep]), vh=np.stack([vh[k] for k in keep]),
                                dates=np.array([its[k].datetime.strftime("%Y-%m-%d") for k in keep]), mask=m,
                                orbit=str(best), epsg=epsg, x0=x0, y1=y1, country=country, aoi=aoi, year=year)
            print(f"ftw {country} {aoi}: {len(keep)} dates, boundary {(m == 2).mean():.3f} unlabelled {(m == 3).mean():.3f}",
                  flush=True)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()


def main():
    what = sys.argv[1:] or ["vh", "n2n", "ftw"]
    cl = Client.open(STAC, modifier=pc.sign_inplace)
    if "vh" in what:
        for name in SITES:
            try:
                do_vh(cl, name)
            except Exception:
                traceback.print_exc()
    if "n2n" in what:
        for name in N2N_SITES:
            try:
                do_n2n(cl, name)
            except Exception:
                traceback.print_exc()
    if "ftw" in what:
        for c, y in FTW_COUNTRIES.items():
            try:
                do_ftw(cl, c, y)
            except Exception:
                traceback.print_exc()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
    finish()
