"""Experiment 9 prep: site split and the Sentinel-1 VV/VH arrays the cluster rules need.

1. Split the 33 sites with finished experiment 7 round 2 pairs into tune and test halves, at random with a
   fixed seed, stratified by the site's dominant ESA WorldCover land class (pooled over its windows).
   Writes sites_split.csv.
2. For every finished pair, re-read the Sentinel-1 RTC gamma0 VV and VH windows (both passes) on the exact
   experiment 7 10 m grid (EPSG, x0, y1 from the cached npz), as float32 linear intensity. The experiment 7
   cache keeps VV in dB but no raw VH, and the speckle-aware rule needs linear intensity. Planetary
   Computer items come from the ids experiment 7 recorded. Writes data/s1/<tag>.npz and data/s1_fetch.csv
   (bytes = compressed TIFF blocks the reads touch).

Experiment 7 files are read only.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
E7 = ROOT.parent / "exp7_change_tasking"
sys.path.insert(0, str(E7))
import r2_run as R  # noqa: E402  (read-only reuse of the experiment 7 readers)

OUT = ROOT / "data" / "s1"
SEED = 9


def land_group(wc):
    land = wc[(wc != 80) & (wc != 0)]
    if not land.size:
        return "other"
    cls = np.bincount(land, minlength=256).argmax()
    return {50: "urban", 60: "bare", 40: "farm", 10: "vegetation", 20: "vegetation", 30: "vegetation",
            90: "vegetation", 95: "vegetation", 100: "vegetation"}.get(int(cls), "other")


def split(ok):
    rows = []
    for sid, g in ok.groupby("site_id"):
        wc = np.concatenate([np.load(E7 / "data" / "r2" / f"{t}.npz")["wc"].ravel() for t in g.tag])
        rows.append({"site_id": int(sid), "land": land_group(wc), "n_pairs": len(g)})
    s = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    s["half"] = ""
    start = int(rng.integers(0, 2))
    for land in sorted(s.land.unique()):
        idx = s.index[s.land == land].to_numpy()
        idx = idx[rng.permutation(len(idx))]
        for k, i in enumerate(idx):           # alternate within stratum; the odd site alternates across strata
            s.loc[i, "half"] = ("tune", "test")[(k + start) % 2]
        start = (start + len(idx)) % 2
    s.to_csv(ROOT / "sites_split.csv", index=False)
    print(s.groupby(["land", "half"]).size().unstack(fill_value=0), flush=True)
    print(s.groupby("half").agg(sites=("site_id", "size"), pairs=("n_pairs", "sum")), flush=True)
    return s


def items_by_id(ids):
    import planetary_computer as pc
    from pystac_client import Client
    for k in range(4):
        try:
            c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
            return list(c.search(collections=["sentinel-1-rtc"], ids=ids).items())
        except Exception:
            time.sleep(5 * (k + 1))
    return []


def fetch(r):
    f = OUT / f"{r.tag}.npz"
    if f.exists():
        return {"tag": r.tag, "status": "cached", "mb": 0.0}
    z = np.load(E7 / "data" / "r2" / f"{r.tag}.npz")
    epsg, x0, y1 = int(z["epsg"]), float(z["x0"]), float(z["y1"])
    R.BYTES["n"] = 0
    out = {}
    for when, col in (("b", "s1_items_before"), ("a", "s1_items_after")):
        ids = str(getattr(r, col)).split(";")
        its = items_by_id(ids)
        its = sorted(its, key=lambda i: ids.index(i.id)) if its else its
        for pol in ("vv", "vh"):
            a = R.read_s1(its, epsg, x0, y1, pol) if its else None
            out[f"{pol}_{when}"] = a.astype("float32") if a is not None else np.full((R.N, R.N), np.nan, "float32")
    # check: the re-read VV must match the experiment 7 cache
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10 * np.log10(out["vv_b"]) - z["s1b_vv"].astype("float32")
    mad = float(np.nanmedian(np.abs(d)))
    np.savez_compressed(f, **out)
    return {"tag": r.tag, "status": "ok", "mb": round(R.BYTES["n"] / 1e6, 2), "vv_check_med_abs_db": round(mad, 4),
            "vh_finite_b": round(float(np.isfinite(out["vh_b"]).mean()), 3),
            "vh_finite_a": round(float(np.isfinite(out["vh_a"]).mean()), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pr = pd.read_csv(E7 / "r2_pair_results.csv")
    ok = pr[pr.status == "ok"].copy()
    print(f"finished pairs {len(ok)}, sites {ok.site_id.nunique()}", flush=True)
    split(ok)
    rows = []
    # BYTES is a module global; one pair at a time per thread would race, so run serially
    for r in ok.itertuples():
        try:
            res = fetch(r)
        except Exception as e:
            traceback.print_exc()
            res = {"tag": r.tag, "status": f"error {type(e).__name__}: {e}"[:200], "mb": 0.0}
        print(res, flush=True)
        rows.append(res)
        pd.DataFrame(rows).to_csv(ROOT / "data" / "s1_fetch.csv", index=False)
    d = pd.DataFrame(rows)
    print("status", d.status.value_counts().to_dict(), "MB", round(d.mb.sum(), 1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
