"""Build paired NISAR tiles from one datatake, read remotely by HDF5 window (no full downloads).

Per tile (5120 m square, aligned to the 80 m frequency-B grid of the frame):
  a10  40 MHz frequency A GCOV, 10 m, 512x512    (the fine target)
  m20  20 MHz mixed-mode frequency A GCOV, 20 m, 256x256   (the coarse input, same echoes)
  b80  5 MHz frequency B GCOV, 80 m, 64x64       (the coarsest band, same echoes)
Bands HH and HV, stored as gamma0 dB float16; NaN where invalid. Also the a10 mask layer.

  python prep.py selection.csv            -> <DATA>/tiles/<role>/<pair>_<r>_<c>.npz, tiles.csv
"""
from __future__ import annotations
from config import DATA  # noqa: E402
import os, sys, json, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = Path(str(DATA) + "/tiles")
NB = 64                                    # tile size in 80 m pixels
K = {"train": 20, "near": 24, "far": 24}   # random tiles per pair
KX = 16                                    # extra X-band-centred tiles per holdout pair


def db(a):
    a = a.astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(a) & (a > 0), 10 * np.log10(a), np.nan).astype("float16")


def grid(g):
    x, y = g["xCoordinates"][:], g["yCoordinates"][:]
    dx, dy = x[1] - x[0], y[1] - y[0]
    return x[0] - dx / 2, y[0] - dy / 2, dx, dy, len(x), len(y)   # pixel edges


def work(args):
    row, excl_wkb, xb = args
    sys.path.insert(0, str(ROOT))
    from shapely import wkb
    from shapely.geometry import Point
    from pyproj import Transformer
    from nisar_io import open_remote
    excl = [wkb.loads(bytes.fromhex(e)) for e in excl_wkb]
    pid = f"{row['track']}_{row['frame']}_{row['date']}"
    res = []
    try:
        A, M = open_remote(row["url_a"]), open_remote(row["url_m"])
        ga, gm = A["science/LSAR/GCOV/grids/frequencyA"], M["science/LSAR/GCOV/grids/frequencyA"]
        gb, gbm = A["science/LSAR/GCOV/grids/frequencyB"], M["science/LSAR/GCOV/grids/frequencyB"]
        epsg = int(ga["projection"][()])
        Ea, Em, Eb, Ebm = grid(ga), grid(gm), grid(gb), grid(gbm)
        assert abs(Ea[2] - 10) < 1e-6 and abs(Em[2] - 20) < 1e-6 and abs(Eb[2] - 80) < 1e-6, (Ea[2], Em[2], Eb[2])
        info = {k: str(v)[:200] for k, v in ga["mask"].attrs.items() if k != "DIMENSION_LIST"}
        hb = gb["HHHH"][:]
        hbm = gbm["HHHH"][:]
        ob = (round((Ebm[0] - Eb[0]) / 80), round((Eb[1] - Ebm[1]) / 80))   # (col,row) of M-B grid inside A-B grid
        vb = np.isfinite(hb) & (hb > 0)
        vm = np.zeros_like(vb)
        r0, c0 = max(ob[1], 0), max(ob[0], 0)
        r1, c1 = min(vb.shape[0], ob[1] + hbm.shape[0]), min(vb.shape[1], ob[0] + hbm.shape[1])
        sub = hbm[r0 - ob[1]:r1 - ob[1], c0 - ob[0]:c1 - ob[0]]
        vm[r0:r1, c0:c1] = np.isfinite(sub) & (sub > 0)
        v = vb & vm
        cs = np.cumsum(np.cumsum(np.pad(v.astype(np.int32), ((1, 0), (1, 0))), 0), 1)
        R, C = v.shape
        to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
        from_ll = Transformer.from_crs(4326, epsg, always_xy=True)

        def block_ok(r, c):
            if r < 0 or c < 0 or r + NB > R or c + NB > C:
                return False
            return cs[r + NB, c + NB] - cs[r, c + NB] - cs[r + NB, c] + cs[r, c] == NB * NB

        def ll_of(r, c):
            return to_ll.transform(Eb[0] + (c + NB / 2) * 80, Eb[1] - (r + NB / 2) * 80)

        def excluded(r, c):
            p = Point(*ll_of(r, c)).buffer(0.05)
            return any(p.intersects(e) for e in excl)

        cands = [(r, c, "") for r in range(0, R - NB, NB // 2) for c in range(0, C - NB, NB // 2)
                 if block_ok(r, c) and not excluded(r, c)]
        rng = np.random.default_rng(zlib.crc32(pid.encode()))
        pick = []
        for i in rng.permutation(len(cands)):
            r, c, _ = cands[i]
            if all(abs(r - r2) >= NB or abs(c - c2) >= NB for r2, c2, _ in pick):
                pick.append(cands[i])
            if len(pick) >= K[row["role"]]:
                break
        if row["role"] != "train" and xb:
            nx = 0
            for xid, prov, lon, lat in xb:
                x, y = from_ll.transform(lon, lat)
                r = int(round((Eb[1] - y) / 80 - NB / 2)); c = int(round((x - Eb[0]) / 80 - NB / 2))
                if block_ok(r, c) and not excluded(r, c) and all(abs(r - r2) >= NB // 2 or abs(c - c2) >= NB // 2 for r2, c2, _ in pick):
                    pick.append((r, c, "x")); nx += 1
                if nx >= KX:
                    break
        oa = (round((Eb[0] - Ea[0]) / 10), round((Ea[1] - Eb[1]) / 10))
        om = (round((Eb[0] - Em[0]) / 20), round((Em[1] - Eb[1]) / 20))
        for r, c, tag in pick:
            ra, ca = oa[1] + r * 8, oa[0] + c * 8
            rm, cm = om[1] + r * 4, om[0] + c * 4
            rb, cb = r - ob[1], c - ob[0]
            if min(ra, ca, rm, cm, rb, cb) < 0:
                continue
            d = {}
            for pol in ["HH", "HV"]:
                t = pol + pol
                if t not in ga or t not in gm or t not in gb:      # single-pol HH acquisitions
                    d[f"a10_{pol}"] = np.full((NB * 8, NB * 8), np.nan, "float16")
                    d[f"m20_{pol}"] = np.full((NB * 4, NB * 4), np.nan, "float16")
                    d[f"b80_{pol}"] = np.full((NB, NB), np.nan, "float16")
                    continue
                d[f"a10_{pol}"] = db(ga[t][ra:ra + NB * 8, ca:ca + NB * 8])
                d[f"m20_{pol}"] = db(gm[t][rm:rm + NB * 4, cm:cm + NB * 4])
                d[f"b80_{pol}"] = db(gb[t][r:r + NB, c:c + NB])
            d["b80m_HH"] = db(gbm["HHHH"][rb:rb + NB, cb:cb + NB])
            d["a10_mask"] = ga["mask"][ra:ra + NB * 8, ca:ca + NB * 8]
            if d["a10_HH"].shape != (512, 512) or d["m20_HH"].shape != (256, 256) or d["b80m_HH"].shape != (64, 64):
                continue
            va = float(np.isfinite(d["a10_HH"].astype("float32")).mean())
            vm2 = float(np.isfinite(d["m20_HH"].astype("float32")).mean())
            if va < 0.98 or vm2 < 0.98:
                continue
            x0, y1 = Eb[0] + c * 80, Eb[1] - r * 80
            name = f"{pid}_{r:04d}_{c:04d}"
            (OUT / row["role"]).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / row["role"] / f"{name}.npz", **d,
                                meta=json.dumps({"epsg": epsg, "x0": x0, "y1": y1, "role": row["role"], "pair": pid, "tag": tag}))
            lon, lat = ll_of(r, c)
            res.append({"tile": name, "role": row["role"], "pair": pid, "track": row["track"], "frame": row["frame"],
                        "reg": row["reg"], "epsg": epsg, "x0": x0, "y1": y1, "lon": lon, "lat": lat, "xtile": tag == "x",
                        "valid_a": va, "valid_m": vm2, "url_a": row["url_a"], "url_m": row["url_m"]})
        print(pid, row["role"], "cands", len(cands), "saved", len(res), "mask:", json.dumps(info)[:300], flush=True)
    except Exception as e:
        import traceback
        print(pid, "ERROR", repr(e)[:300], traceback.format_exc()[-800:], flush=True)
    return res


def main():
    import geopandas as gpd
    sel = pd.read_csv(sys.argv[1])
    frames = gpd.read_file(str(DATA) + "/selected_frames.gpkg")
    hold = frames[frames.role != "train"].geometry.tolist()
    trn = frames[frames.role == "train"].geometry.tolist()
    X = gpd.read_file(str(DATA) + "/xband_all.gpkg")
    X["c"] = X.geometry.representative_point()
    jobs = []
    for _, r in sel.iterrows():
        row = r.to_dict()
        excl = hold if row["role"] == "train" else trn
        xb = []
        if row["role"] != "train":
            fpoly = frames[(frames.track == row["track"]) & (frames.frame == row["frame"])].geometry.iloc[0]
            xi = X[X.c.within(fpoly)].sample(frac=1, random_state=0)
            xb = [(a.xband_id, a.provider, a.c.x, a.c.y) for a in xi.itertuples()]
        jobs.append((row, [e.wkb_hex for e in excl], xb))
    rows = []
    if len(sys.argv) > 2:                  # append to an existing tiles.csv (rerun of failed pairs)
        rows = pd.read_csv(ROOT / sys.argv[2]).to_dict("records")
    with ProcessPoolExecutor(6) as ex:
        for res in ex.map(work, jobs):
            rows += res
            pd.DataFrame(rows).to_csv(ROOT / "tiles.csv", index=False)
    print("TOTAL", len(rows), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
