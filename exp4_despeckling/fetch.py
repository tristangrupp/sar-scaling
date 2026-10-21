"""Download single-date Sentinel-1 RTC VV windows for every site (one relative orbit per site)
and ESA WorldCover labels. Writes E:/SAR_superresolution/exp4_despeckling/data/<site>.npz with
  vv  (T, H, W) float32 linear power (NaN = nodata), dates, orbit, epsg, x0, y1, wc (H, W) uint8
"""
from __future__ import annotations

import common  # noqa: F401  (GDAL env first)
from common import DATA, MAX_DATES, MIN_DATES, SITES, finish, site_grid

import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
FORCE_ORBIT = {"louisiana_wetland": 63}   # orbit 165 clips the site at its swath edge (38 % nodata)


def read_window(href, epsg, x0, y1, n, resampling, nodata, dtype):
    for attempt in range(4):
        try:
            with rasterio.open(href) as s, WarpedVRT(
                    s, crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, 10, 10), width=n, height=n,
                    resampling=resampling, nodata=nodata, dtype=dtype) as v:
                return v.read(1)
        except Exception as e:  # noqa: BLE001
            print("  retry", attempt, str(e)[:120], flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def save_wc(cl, name):
    """WorldCover 2021 (v200), mosaicked over every 3x3 degree tile the window touches."""
    from pyproj import Transformer
    epsg, x0, y1, n = site_grid(name)
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = tr.transform([x0, x0 + 10 * n, x0, x0 + 10 * n], [y1, y1, y1 - 10 * n, y1 - 10 * n])
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    items = [i for i in cl.search(collections=["esa-worldcover"], bbox=bbox).items() if "v200" in i.id or "2021" in i.id]
    wc = np.zeros((n, n), np.uint8)
    for it in items:
        a = read_window(pc.sign(it.assets["map"].href), epsg, x0, y1, n, Resampling.nearest, 0, "uint8")
        if a is not None:
            wc = np.where(wc == 0, a, wc)
    return wc


def do_site(cl, name):
    out = DATA / f"{name}.npz"
    if out.exists():
        print("cached", name, flush=True)
        return
    lon, lat, lc, split, t0, t1 = SITES[name]
    epsg, x0, y1, n = site_grid(name)
    import pandas as pd
    a, b = pd.Timestamp(t0), pd.Timestamp(t1)
    for widen in range(5):      # widen the window by 2 months each side until one orbit has >= MIN_DATES
        items = [i for i in cl.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                      datetime=f"{a:%Y-%m-%d}/{b:%Y-%m-%d}").items() if "vv" in i.assets]
        orb = Counter((i.properties.get("sat:relative_orbit"), i.properties.get("sat:orbit_state")) for i in items)
        ndates = {k: len({i.datetime.date() for i in items if (i.properties.get("sat:relative_orbit"), i.properties.get("sat:orbit_state")) == k}) for k in orb}
        best = max(ndates, key=ndates.get)
        if name in FORCE_ORBIT:
            best = next(k for k in ndates if k[0] == FORCE_ORBIT[name])
        print(name, f"{a:%Y-%m-%d}..{b:%Y-%m-%d}", "dates per orbit", ndates, flush=True)
        if ndates[best] >= MIN_DATES + 2:
            break
        a, b = a - pd.DateOffset(months=2), b + pd.DateOffset(months=2)
    its = sorted([i for i in items if (i.properties.get("sat:relative_orbit"), i.properties.get("sat:orbit_state")) == best],
                 key=lambda i: i.datetime)
    # one per date (frames overlap at slice edges)
    seen, uniq = set(), []
    for i in its:
        d = i.datetime.strftime("%Y-%m-%d")
        if d not in seen:
            seen.add(d)
            uniq.append(i)
    if len(uniq) > MAX_DATES:
        idx = np.linspace(0, len(uniq) - 1, MAX_DATES).round().astype(int)
        uniq = [uniq[k] for k in idx]

    def rd(it):
        a = read_window(pc.sign(it.assets["vv"].href), epsg, x0, y1, n, Resampling.bilinear, -32768, "float32")
        if a is None:
            return None
        a[(a == -32768) | ~np.isfinite(a) | (a <= 0)] = np.nan
        return a

    with ThreadPoolExecutor(6) as ex:
        arrs = list(ex.map(rd, uniq))
    keep = [(it, a) for it, a in zip(uniq, arrs) if a is not None and np.isfinite(a).mean() > 0.95]
    vv = np.stack([a for _, a in keep]).astype("float32")
    dates = np.array([it.datetime.strftime("%Y-%m-%d") for it, _ in keep])
    plats = np.array([it.properties.get("platform", "") for it, _ in keep])

    wc = save_wc(cl, name)
    np.savez_compressed(out, vv=vv, dates=dates, platforms=plats, orbit=str(best), epsg=epsg, x0=x0, y1=y1,
                        wc=wc)
    print(f"saved {name}: {vv.shape[0]} dates {dates[0]}..{dates[-1]} orbit {best} "
          f"platforms {dict(Counter(plats))} wc classes {np.unique(wc).tolist()}", flush=True)


def main():
    cl = Client.open(STAC, modifier=pc.sign_inplace)
    names = sys.argv[1:] or list(SITES)
    for name in names:
        try:
            do_site(cl, name)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
    finish()
