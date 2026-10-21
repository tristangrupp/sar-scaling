"""Round 2 step 4: score arm A' / arm B against X-band edges.

  windows   DATA/xtrain/<xval|xtest>/*.npz (built by build_xtrain.py): xval tunes the edge weight, xtest is scored
  holdout   the round 1 held-out X-band sites that passed (eval/xband_scores.csv, roles near / far, status ok),
            re-read and re-registered exactly as in xband_check.py, now scored for several checkpoints
Chamfer (m) on infrastructure pixels only: p2x = image edges to nearest X-band edge, x2p = X-band edges to nearest
image edge. Writes eval2/xb_<split>_scores.csv and eval2/xb_summary.csv (per provider and split).

  python xb_eval.py --runs b_a b_e005 b_e02
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA                                  # noqa: E402
from evaluate import bicubic, chamfer, edges, predict   # noqa: E402
from train import DEV, Gen, to_n                        # noqa: E402

PX = 10.0


def score_window(models, xin, truth, ex, infra):
    xin = np.where(np.isfinite(xin), xin, np.nanmedian(xin))
    truth = np.where(np.isfinite(truth), truth, np.nanmedian(truth))
    imgs = {"bicubic": bicubic(to_n(xin), 2), "truth40": truth, **{k: predict(G, to_n(xin)) for k, G in models.items()}}
    rec = {}
    for k, im in imgs.items():
        rec[f"{k}_p2x"], rec[f"{k}_x2p"] = chamfer(edges(im), ex, infra, PX)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--label", default="", help="prefix for output files (round 4: r4_)")
    a = ap.parse_args()
    models = {}
    for r in a.runs:
        G = Gen(2).to(DEV).eval()
        G.load_state_dict(torch.load(ROOT / f"ckpt/{r}.pt", weights_only=True, map_location=DEV))
        models[r] = G
    out = ROOT / "eval2"; out.mkdir(exist_ok=True)
    allrows = []
    for split in ["xval", "xtest"]:
        rows = []
        for f in sorted(glob.glob(str(DATA / "xtrain" / split / "*.npz"))):
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            rec = {"split": split, "file": Path(f).stem, "provider": meta["provider"], "site": meta["site"]}
            rec.update(score_window(models, z["x"].astype("float32"), z["y"].astype("float32"), z["xe"], z["infra"]))
            rows.append(rec)
        pd.DataFrame(rows).to_csv(out / f"{a.label}xb_{split}_scores.csv", index=False)
        allrows += rows
        print(split, len(rows), flush=True)
    if a.holdout:
        # held-out X-band sites from round 1 (near / far / geometry-only frames): re-read with xband_check
        import xband_check as xc
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer
        from scipy import ndimage as ndi  # noqa: F401
        from nisar_io import open_remote
        client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        prev = pd.read_csv(ROOT / "eval/xband_scores.csv")
        prev = prev[(prev.status == "ok") & prev.role.isin(["near", "far"])]
        cand = xc.sites()
        cand = cand[cand.xband_id.isin(set(prev.xband_id)) & cand.role.isin(["near", "far"])].drop_duplicates("xband_id")
        hrefs = json.loads((DATA / "xband_hrefs.json").read_text())
        cand["href"] = np.where(cand.provider == "iceye", cand.grd_href, cand.xband_id.map(hrefs))
        cand = cand.sort_values(["url_a", "site"])
        rows, opened = [], {}
        for c in cand.itertuples():
            try:
                if c.url_a not in opened:
                    opened = {c.url_a: (open_remote(c.url_m), open_remote(c.url_a))}
                M, A = opened[c.url_a]
                gm, ga = M["science/LSAR/GCOV/grids/frequencyA"], A["science/LSAR/GCOV/grids/frequencyA"]
                epsg = int(ga["projection"][()])
                x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(c.lon, c.lat)
                xe0 = ga["xCoordinates"][0] - 5; ye0 = ga["yCoordinates"][0] + 5
                N = xc.N
                x0 = xe0 + np.round((x - N * PX / 2 - xe0) / 80) * 80; y1 = ye0 - np.round((ye0 - y - N * PX / 2) / 80) * 80
                xin = xc.read_win(gm, (x0, y1), N // 2, 20.0); truth = xc.read_win(ga, (x0, y1), N, 10.0)
                to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
                lo0, la0 = to_ll.transform(x0, y1 - N * PX); lo1, la1 = to_ll.transform(x0 + N * PX, y1)
                wc = xc.worldcover(client, epsg, x0, y1, (lo0, la0, lo1, la1))
                amp = xc.read_x(c.href, epsg, x0, y1)
                xm = np.isfinite(amp) & (amp > 0)
                xdb = np.where(xm, 20 * np.log10(np.where(xm, amp, 1)), np.nan)
                tr = np.where(np.isfinite(truth), truth, np.nanmedian(truth))
                xr, xmr, sh, r40 = xc.register(xdb, tr, xm)
                if not (r40 >= 0.2):
                    continue
                ex = edges(np.where(xmr, xr, np.nanmedian(xr)))
                infra = (wc == 50) & xmr
                rec = {"split": "holdout_" + c.role, "file": c.xband_id, "provider": c.provider, "site": c.site}
                rec.update(score_window(models, xin, truth, ex, infra))
                rows.append(rec)
                print("holdout", c.provider, c.site, "ok", flush=True)
            except Exception as e:
                print("holdout ERR", c.xband_id, repr(e)[:150], flush=True)
        pd.DataFrame(rows).to_csv(out / f"{a.label}xb_holdout_scores.csv", index=False)
        allrows += rows
    df = pd.DataFrame(allrows)
    cols = [c for c in df.columns if c.endswith(("_p2x", "_x2p"))]
    s = df.groupby(["split", "provider"])[cols].median().round(2)
    s["n"] = df.groupby(["split", "provider"]).size()
    s2 = df.groupby("split")[cols].median().round(2); s2["n"] = df.groupby("split").size(); s2["provider"] = "all"
    s = pd.concat([s.reset_index(), s2.reset_index()]).sort_values(["split", "provider"])
    s.to_csv(out / f"{a.label}xb_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
