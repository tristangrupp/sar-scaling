"""Step 1: pick one Capella GEO item per site (DESIGN.md, "Sites and items"). Writes items.csv.

Run in the background with output to a log (GDAL can hang on a foreground pipe).
"""
from __future__ import annotations

import common  # noqa: F401  (GDAL env)
from common import EXP6, HERE, finish, pc_search, utm_epsg, warp_read

import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

BUILT_CAP = 0.45


def flat_share(lon, lat):
    """Share of a 4 km box at (lon, lat) with slope <= 10 deg on the 30 m DEM."""
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    x0, y1, n = x - 2000, y + 2000, 134
    d = 0.03
    its = pc_search(collections=["cop-dem-glo-30"], bbox=[lon - d, lat - d, lon + d, lat + d])
    z = None
    for it in its:
        a = warp_read(it.assets["data"].href, epsg, x0, y1, 30.0, n, n, "bilinear", -32767.0)
        z = a if z is None else np.where(np.isfinite(z), z, a)
    if z is None or np.isfinite(z).mean() < 0.5:
        return np.nan
    zz = np.where(np.isfinite(z), z, np.nanmean(z))
    gy, gx = np.gradient(zz, 30.0)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))[np.isfinite(z)]
    return float((s <= 10).mean())


def href_for(cid):
    ts = cid.split("_")[5]
    u = f"https://capella-open-data.s3.amazonaws.com/data/{int(ts[:4])}/{int(ts[4:6])}/{int(ts[6:8])}/{cid}/{cid}.tif"
    for k in range(3):
        try:
            return u if requests.head(u, timeout=60).status_code == 200 else ""
        except Exception:
            pass
    return ""


def main():
    sc = pd.read_csv(EXP6 / "scenes.csv")
    mt = pd.read_csv(EXP6 / "s1_iw_match.csv")[["capella_id", "s1_iw_full_cover_n"]]
    d = sc.merge(mt, on="capella_id", how="left")
    n0 = len(d)
    d = d[d.s1_iw_has_vv_vh.fillna(False).astype(bool)]
    d = d[d.s1_iw_nearest_dt_days.abs() <= 12]
    n1 = len(d)
    d = d[(d.wc_land >= 0.7) & (d.wc_water <= 0.3)]
    n2 = len(d)
    print(f"items {n0}; VV+VH within 12 d {n1}; land>=0.7 & water<=0.3 {n2} at {d.site_id.nunique()} sites", flush=True)
    d = d.assign(gap=d.s1_iw_nearest_dt_days.abs(), near6=(d.s1_iw_nearest_dt_days.abs() <= 6).astype(int),
                 res=np.sqrt(d.res_range_m * d.res_azimuth_m))
    d = d.sort_values(["site_id", "near6", "gap", "res"], ascending=[True, False, True, True])
    first = d.groupby("site_id").head(1)

    with ThreadPoolExecutor(8) as ex:
        shares = list(ex.map(lambda r: _safe(flat_share, r.lon, r.lat), first.itertuples()))
    first = first.assign(flat_share=shares)
    ok_sites = first[first.flat_share >= 0.5].site_id
    print(f"sites with flat share >= 0.5: {len(ok_sites)} of {len(first)}", flush=True)
    d = d[d.site_id.isin(ok_sites)]
    d = d.merge(first[["site_id", "flat_share"]], on="site_id")

    # one item per site, with a working GeoTIFF
    picks = []
    for sid, g in d.groupby("site_id", sort=False):
        for r in g.itertuples():
            h = href_for(r.capella_id)
            if h:
                picks.append({**r._asdict(), "capella_href": h})
                break
    p = pd.DataFrame(picks).drop(columns=["Index"])
    other = (p.wc_land - p[["wc_crop", "wc_built", "wc_tree", "wc_grass"]].sum(axis=1)).clip(lower=0)
    shares = pd.DataFrame({"built": p.wc_built, "crop": p.wc_crop, "tree": p.wc_tree, "grass": p.wc_grass,
                           "bare_other": other})
    p["cover"] = shares.idxmax(axis=1)
    print("dominant cover before cap:", p.cover.value_counts().to_dict(), flush=True)
    # built cap: drop the built-dominated sites with the largest S1 gap first
    nb_max = int(BUILT_CAP * len(p[p.cover != "built"]) / (1 - BUILT_CAP))
    b = p[p.cover == "built"].sort_values(["near6", "gap", "res"], ascending=[False, True, True])
    keep = pd.concat([p[p.cover != "built"], b.head(nb_max)]).sort_values("site_id").reset_index(drop=True)
    print(f"built cap {BUILT_CAP}: kept {min(nb_max, len(b))} of {len(b)} built sites; total {len(keep)}", flush=True)
    print("dominant cover:", keep.cover.value_counts().to_dict(), flush=True)
    print("mode:", keep["mode"].value_counts().to_dict(), "pol:", keep.polarization.value_counts().to_dict(), flush=True)
    print("gap<=6:", int(keep.near6.sum()), "countries:", keep.country.nunique(), flush=True)
    keep.to_csv(HERE / "items.csv", index=False)


def _safe(f, *a):
    try:
        return f(*a)
    except Exception:
        traceback.print_exc()
        return np.nan


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
