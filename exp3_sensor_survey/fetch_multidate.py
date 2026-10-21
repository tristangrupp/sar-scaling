"""Round 2 step 2: a cleaner 10 m target from several 12-day repeats of the same track/frame (40 MHz GCOV).

For every tile: read the same 10 m window (HH) from up to K other 40 MHz acquisitions of its track/frame,
remotely. Change mask, per 80 m block and date: keep a date only where its block mean is within TOL dB of the
input date's block mean (the input date is always kept). Target = mean of kept dates in linear power, in dB.
Writes DATA/extra/mt10/<tile>.npz with y (dB, float16), n (kept dates per pixel), m (n >= 3).

  python fetch_multidate.py            (all tiles in tiles.csv; resumable)
"""
from __future__ import annotations
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

K, TOL, NPX = 6, 1.5, 512
OUT = DATA / "extra" / "mt10"


def block(a, k=8):
    n = a.shape[0] // k
    return np.nanmean(a.reshape(n, k, n, k), axis=(1, 3))


def frame_job(args):
    (track, frame), tiles, prods = args
    sys.path.insert(0, str(ROOT))
    from nisar_io import open_remote
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tiles if not (OUT / f"{t['tile']}.npz").exists()]
    if not todo:
        return f"{track}_{frame}: cached"
    # own-date stacks start from the saved tile (input date)
    stacks = {}
    for t in todo:
        z = np.load(DATA / "tiles" / t["role"] / f"{t['tile']}.npz")
        own = 10 ** (z["a10_HH"].astype("float32") / 10)
        stacks[t["tile"]] = {"date": t["date"], "own": own, "sum": np.nan_to_num(own), "n": np.isfinite(own).astype("int16")}
    used = 0
    for pr in prods:
        try:
            h = open_remote(pr["url"])
            g = h["science/LSAR/GCOV/grids/frequencyA"]
            if abs(g["xCoordinateSpacing"][()] - 10) > 1e-6:
                continue
            x = g["xCoordinates"][:]; y = g["yCoordinates"][:]
            xe, ye = x[0] - 5, y[0] + 5
            for t in todo:
                st = stacks[t["tile"]]
                d = pr["t0"][:8]
                if d == st["date"] or st.get("k", 0) >= K:
                    continue
                c = (t["x0"] - xe) / 10; r = (ye - t["y1"]) / 10
                if abs(c - round(c)) > 0.01 or abs(r - round(r)) > 0.01:
                    continue
                c, r = int(round(c)), int(round(r))
                if r < 0 or c < 0 or r + NPX > len(y) or c + NPX > len(x):
                    continue
                a = g["HHHH"][r:r + NPX, c:c + NPX].astype("float32")
                a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
                if np.isfinite(a).mean() < 0.9:
                    continue
                diff = 10 * np.log10(block(a) / block(st["own"]))
                keep = np.kron(np.abs(diff) <= TOL, np.ones((8, 8), bool)) & np.isfinite(a)
                st["sum"] += np.where(keep, a, 0)
                st["n"] += keep
                st["k"] = st.get("k", 0) + 1
                st.setdefault("dates", []).append(d)
            used += 1
        except Exception as e:
            print(track, frame, pr["t0"], "ERR", repr(e)[:150], flush=True)
        if all(stacks[t["tile"]].get("k", 0) >= K for t in todo):
            break
    for t in todo:
        st = stacks[t["tile"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            yy = 10 * np.log10(st["sum"] / st["n"])
        np.savez_compressed(OUT / f"{t['tile']}.npz", y=yy.astype("float16"), n=st["n"].astype("uint8"),
                            m=st["n"] >= 3, dates=json.dumps(st.get("dates", [])))
    ks = [stacks[t["tile"]].get("k", 0) for t in todo]
    return f"{track}_{frame}: tiles {len(todo)} products opened {used} extra dates per tile median {np.median(ks):.0f}"


def main():
    tiles = pd.read_csv(ROOT / "tiles.csv")
    tiles["date"] = tiles.pair.str.split("_").str[2]
    idx = pd.read_csv(DATA / "gcov_index.csv")
    idx = idx[(idx.mm == "A") & (idx.bw == 4005)] if idx.bw.dtype != object else idx[(idx.mm == "A") & (idx.bw == "4005")]
    jobs = []
    for (tr, fr), g in tiles.groupby(["track", "frame"]):
        p = idx[(idx.track.astype(int) == tr) & (idx.frame.astype(int) == fr)].copy()
        if p.empty:
            continue
        p = p[p.full == "F"] if (p.full == "F").sum() >= 4 else p
        mid = pd.to_datetime(g.date).median()
        p["dt"] = (pd.to_datetime(p.t0.str[:8]) - mid).abs()
        prods = p.sort_values("dt")[["url", "t0"]].to_dict("records")
        jobs.append(((tr, fr), g[["tile", "role", "date", "x0", "y1"]].to_dict("records"), prods))
    print("frames", len(jobs), flush=True)
    with ProcessPoolExecutor(6) as ex:
        for msg in ex.map(frame_job, jobs):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
