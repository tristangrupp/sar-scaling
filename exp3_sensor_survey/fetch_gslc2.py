"""Round 3 step 1: a deeper 5 m GSLC target (more dates per tile), from a real GSLC index instead of GCOV-derived names.

Round 2 derived each GSLC URL from its GCOV twin's name and stopped at 5 extra dates; some files 404'd and 25 tiles
got no extra date. Here:
  - list every 40 MHz GSLC granule (PROVISIONAL) for each track/frame from NASA CMR, full and partial frames
  - read up to K extra dates per tile (nearest in time first), retrying transient errors 3 times
  - same calibration, change mask and RTC anchoring as fetch_gslc.py
Writes DATA/extra/g5v2/<tile>.npz and gslc2_coverage.csv (dates tried / read / outside / 404 / error per tile).
"""
from __future__ import annotations
import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
import fetch_gslc as fg   # noqa: E402  read_cal, block, PER_PAIR

K, TOL = 10, 1.5
OUT = DATA / "extra" / "g5v2"
GSLC = "C2854332392-ASF"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"


def gslc_index(track, frame):
    out, sa = [], None
    while True:
        p = {"collection_concept_id": GSLC, "page_size": 2000, "options[readable_granule_name][pattern]": "true",
             "readable_granule_name[]": f"NISAR_L2_PR_GSLC_*_{track:03d}_?_{frame:03d}_4005_*"}
        r = requests.get(CMR, params=p, headers={"CMR-Search-After": sa} if sa else {}, timeout=120)
        e = r.json()["feed"]["entry"]
        for it in e:
            url = next((l["href"] for l in it["links"] if l["href"].endswith(".h5") and l["href"].startswith("https")), None)
            if url:
                out.append({"id": it["title"], "url": url, "t0": it["title"].split("_")[11]})
        sa = r.headers.get("CMR-Search-After")
        if not e or not sa or len(e) < 2000:
            break
    return pd.DataFrame(out).drop_duplicates("t0") if out else pd.DataFrame(columns=["id", "url", "t0"])


def open_retry(url, tries=3):
    from nisar_io import open_remote
    last = None
    for k in range(tries):
        try:
            return open_remote(url), "ok"
        except Exception as e:
            last = repr(e)
            if "404" in last:
                return None, "404"
            time.sleep(5 * (k + 1))
    return None, "error:" + last[:80]


def frame_job(args):
    (track, frame), tiles = args
    sys.path.insert(0, str(ROOT))
    from scipy import ndimage as ndi
    OUT.mkdir(parents=True, exist_ok=True)
    idx = gslc_index(track, frame)
    cov = []
    todo = [t for t in tiles if not (OUT / f"{t['tile']}.npz").exists()]
    if not todo:
        return f"{track}_{frame}: cached", cov
    st = {t["tile"]: {"k": 0, "dates": [], "tried": 0, "outside": 0, "e404": 0, "err": 0} for t in todo}
    # own date: the GSLC of the tile's own datatake (match by start time), else skip the tile
    for t in todo:
        own = idx[idx.t0.str[:15] == t["t0a"][:15]]
        url = own.url.iloc[0] if len(own) else t["url_a"].replace("GCOV", "GSLC")
        h, s = open_retry(url)
        a = fg.read_cal(h, t["x0"], t["y1"]) if h is not None else None
        if a is not None:
            st[t["tile"]].update(own=a, sum=np.nan_to_num(a), n=np.isfinite(a).astype("int16"))
        else:
            st[t["tile"]]["own_status"] = s
    todo2 = [t for t in todo if "own" in st[t["tile"]]]
    mid = pd.to_datetime(pd.Series([t["date"] for t in todo]), format="%Y%m%d").median()
    idx = idx.assign(dt=(pd.to_datetime(idx.t0.str[:8], format="%Y%m%d") - mid).abs()).sort_values("dt")
    for pr in idx.itertuples():
        need = [t for t in todo2 if st[t["tile"]]["k"] < K and pr.t0[:8] != t["date"]]
        if not need:
            break
        h, s = open_retry(pr.url)
        for t in need:
            S = st[t["tile"]]; S["tried"] += 1
            if h is None:
                S["e404" if s == "404" else "err"] += 1
                continue
            try:
                a = fg.read_cal(h, t["x0"], t["y1"])
            except Exception:
                a = None; S["err"] += 1
                continue
            if a is None:
                S["outside"] += 1
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = 10 * np.log10(fg.block(a, 16) / fg.block(S["own"], 16))
            keep = np.kron(np.abs(diff) <= TOL, np.ones((16, 16), bool)) & np.isfinite(a)
            S["sum"] += np.where(keep, a, 0); S["n"] += keep; S["k"] += 1; S["dates"].append(pr.t0[:8])
    for t in todo:
        S = st[t["tile"]]
        rec = {"tile": t["tile"], "role": t["role"], "frame": f"{track}_{frame}", "gslc_in_index": len(idx),
               "own": "own" in S, "own_status": S.get("own_status", "ok"), "extra_dates": S["k"], "tried": S["tried"],
               "outside": S["outside"], "e404": S["e404"], "err": S["err"]}
        if "own" in S:
            with np.errstate(divide="ignore", invalid="ignore"):
                mean5 = S["sum"] / S["n"]
                z = np.load(DATA / "tiles" / t["role"] / f"{t['tile']}.npz")
                g10 = 10 ** (z["a10_HH"].astype("float32") / 10)
                pooled = mean5.reshape(512, 2, 512, 2).mean(axis=(1, 3))
                fill = lambda q: np.where(np.isfinite(q), q, np.nanmedian(q))
                ratio = ndi.gaussian_filter(fill(g10), 8) / ndi.gaussian_filter(fill(pooled), 8)
                y = 10 * np.log10(mean5 * np.kron(ratio, np.ones((2, 2))))
            m = S["n"] >= 3
            np.savez_compressed(OUT / f"{t['tile']}.npz", y=y.astype("float16"), n=S["n"].astype("uint8"), m=m,
                                dates=json.dumps(S["dates"]))
            rec["mask_frac"] = float(m.mean()); rec["looks_median"] = float(np.median(S["n"]))
        cov.append(rec)
    ks = [st[t["tile"]]["k"] for t in todo2]
    return f"{track}_{frame}: gslc granules {len(idx)} tiles {len(todo)} extra dates median {np.median(ks) if ks else 0}", cov


def main():
    tiles = pd.read_csv(ROOT / "tiles.csv")
    tiles["date"] = tiles.pair.str.split("_").str[2]
    tiles = tiles[~tiles.xtile]
    tiles = pd.concat([g.head(fg.PER_PAIR[r]) for (r, _), g in tiles.groupby(["role", "pair"])])   # same 468 tiles as round 2
    tiles["t0a"] = tiles.url_a.str.split("/").str[-1].str.split("_").str[11]
    jobs = [((tr, fr), g[["tile", "role", "date", "x0", "y1", "url_a", "t0a"]].to_dict("records"))
            for (tr, fr), g in tiles.groupby(["track", "frame"])]
    print("frames", len(jobs), "tiles", len(tiles), flush=True)
    cov = []
    with ProcessPoolExecutor(6) as ex:
        for msg, c in ex.map(frame_job, jobs):
            print(msg, flush=True)
            cov += c
            pd.DataFrame(cov).to_csv(ROOT / "gslc2_coverage.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
