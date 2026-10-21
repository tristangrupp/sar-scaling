"""Round 2 step 3: a 5 m target from 40 MHz GSLC (5 m x 5 m posting), for the 10 m GCOV -> 5 m arm.

A single-date GSLC is about one look, so the target is a multi-date mean, as in step 2:
  - read the tile's 1024 x 1024 HH window at 5 m from the input-date GSLC and up to K other dates (remote)
  - calibrate: gamma0 = |DN|^2 / LUT^2 (geometry/gamma0 LUT, bilinear onto the window)
  - change mask per 80 m block and date: keep a date where its block mean is within TOL dB of the input date's
  - mean of kept dates in linear power
  - anchor the radiometry to the input-date GCOV (RTC, 10 m): multiply by the ratio of the two, both smoothed
    to about 80 m, so terrain-flattening and calibration match the input; only the fine structure comes from GSLC
Writes DATA/extra/g5/<tile>.npz: y (dB), n (dates kept), m (n >= 3).
"""
from __future__ import annotations
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA, VAL_FRAMES  # noqa: E402

K, TOL, N5 = 5, 1.5, 1024
PER_PAIR = {"train": 10, "near": 12, "far": 12}
OUT = DATA / "extra" / "g5"


def block(a, k):
    n = a.shape[0] // k
    return np.nanmean(a.reshape(n, k, n, k), axis=(1, 3))


def gamma_lut(h, xs, ys):
    from scipy.interpolate import RegularGridInterpolator
    c = h["science/LSAR/GSLC/metadata/calibrationInformation/geometry"]
    lx, ly, g = c["xCoordinates"][:], c["yCoordinates"][:], c["gamma0"][:].astype("float64")
    f = RegularGridInterpolator((ly[::-1], lx), g[::-1], bounds_error=False, fill_value=np.nan)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return f((Y, X))


def read_cal(h, x0, y1):
    g = h["science/LSAR/GSLC/grids/frequencyA"]
    if abs(g["xCoordinateSpacing"][()] - 5) > 1e-6 or abs(abs(g["yCoordinateSpacing"][()]) - 5) > 1e-6:
        return None
    x = g["xCoordinates"][:]; y = g["yCoordinates"][:]
    xe, ye = x[0] - 2.5, y[0] + 2.5
    c = (x0 - xe) / 5; r = (ye - y1) / 5
    if abs(c - round(c)) > 0.01 or abs(r - round(r)) > 0.01:
        return None
    c, r = int(round(c)), int(round(r))
    if r < 0 or c < 0 or r + N5 > len(y) or c + N5 > len(x):
        return None
    z = g["HH"][r:r + N5, c:c + N5]
    p = (z.real.astype("float64") ** 2 + z.imag.astype("float64") ** 2)
    lut = gamma_lut(h, x[c:c + N5], y[r:r + N5])
    a = p / lut ** 2
    a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
    return a if np.isfinite(a).mean() > 0.9 else None


def frame_job(args):
    (track, frame), tiles, prods = args
    sys.path.insert(0, str(ROOT))
    from scipy import ndimage as ndi
    from nisar_io import open_remote
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tiles if not (OUT / f"{t['tile']}.npz").exists()]
    if not todo:
        return f"{track}_{frame}: cached"
    st = {t["tile"]: {"k": 0, "dates": []} for t in todo}
    # own date first (the tile's own 40 MHz datatake)
    for t in todo:
        try:
            a = read_cal(open_remote(t["url_a"].replace("GCOV", "GSLC")), t["x0"], t["y1"])
        except Exception as e:
            a = None
            print(t["tile"], "own ERR", repr(e)[:120], flush=True)
        if a is not None:
            st[t["tile"]].update(own=a, sum=np.nan_to_num(a), n=np.isfinite(a).astype("int16"))
    todo = [t for t in todo if "own" in st[t["tile"]]]
    for pr in prods:
        need = [t for t in todo if st[t["tile"]]["k"] < K and pr["t0"][:8] != t["date"]]
        if not need:
            break
        try:
            h = open_remote(pr["url"].replace("GCOV", "GSLC"))
        except Exception as e:
            print(track, frame, "open ERR", repr(e)[:120], flush=True)
            continue
        for t in need:
            s = st[t["tile"]]
            try:
                a = read_cal(h, t["x0"], t["y1"])
            except Exception:
                a = None
            if a is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = 10 * np.log10(block(a, 16) / block(s["own"], 16))
            keep = np.kron(np.abs(diff) <= TOL, np.ones((16, 16), bool)) & np.isfinite(a)
            s["sum"] += np.where(keep, a, 0); s["n"] += keep; s["k"] += 1; s["dates"].append(pr["t0"][:8])
    for t in todo:
        s = st[t["tile"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean5 = s["sum"] / s["n"]
            z = np.load(DATA / "tiles" / t["role"] / f"{t['tile']}.npz")
            g10 = 10 ** (z["a10_HH"].astype("float32") / 10)
            pooled = mean5.reshape(512, 2, 512, 2).mean(axis=(1, 3))
            fill = lambda q: np.where(np.isfinite(q), q, np.nanmedian(q))
            ratio = ndi.gaussian_filter(fill(g10), 8) / ndi.gaussian_filter(fill(pooled), 8)
            y = 10 * np.log10(mean5 * np.kron(ratio, np.ones((2, 2))))
        np.savez_compressed(OUT / f"{t['tile']}.npz", y=y.astype("float16"), n=s["n"].astype("uint8"), m=s["n"] >= 3,
                            dates=json.dumps(s["dates"]))
    return f"{track}_{frame}: tiles {len(todo)} extra dates median {np.median([st[t['tile']]['k'] for t in todo]) if todo else 0}"


def main():
    tiles = pd.read_csv(ROOT / "tiles.csv")
    tiles["date"] = tiles.pair.str.split("_").str[2]
    tiles = tiles[~tiles.xtile]
    tiles = pd.concat([g.head(PER_PAIR[r]) for (r, _), g in tiles.groupby(["role", "pair"])])
    idx = pd.read_csv(DATA / "gcov_index.csv")
    idx = idx[(idx.mm == "A") & (idx.bw == 4005)]
    jobs = []
    for (tr, fr), g in tiles.groupby(["track", "frame"]):
        p = idx[(idx.track == tr) & (idx.frame == fr)].copy()
        mid = pd.to_datetime(g.date).median()
        p["dt"] = (pd.to_datetime(p.t0.str[:8]) - mid).abs()
        jobs.append(((tr, fr), g[["tile", "role", "date", "x0", "y1", "url_a"]].to_dict("records"),
                     p.sort_values("dt")[["url", "t0"]].to_dict("records")))
    print("frames", len(jobs), "tiles", len(tiles), flush=True)
    with ProcessPoolExecutor(6) as ex:
        for msg in ex.map(frame_job, jobs):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
