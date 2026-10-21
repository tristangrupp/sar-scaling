"""Round 2 scoring: any checkpoints, any split, any target, same metrics as round 1 (evaluate.scores).

  python eval2.py --label sweep --arm x2 --runs s_w0 s_w005 --splits val
  python eval2.py --label multi --arm x2 --runs a x2multi --splits near far --target mt10
Splits: val = training-role tiles from config.VAL_FRAMES; near / far = the round 1 holdouts.
Writes eval2/<label>_scores.csv and eval2/<label>_summary.csv (medians per split and model).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from config import VAL_FRAMES
from evaluate import bicubic, predict, scores
from train import ARMS, DEV, Gen, from_n, load

ROOT = Path(__file__).resolve().parent
COLS = ["mae", "ssim", "detail_r", "hf_ratio", "cons", "ch_p2t", "ch_t2p", "grid"]


def tiles_for(split, arm, target):
    if split == "val":
        return load("train", arm, frames_in=set(VAL_FRAMES), target=target)
    return load(split, arm, target=target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", default="x2")
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoint names in ckpt/ (without .pt)")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--target", default=None)
    ap.add_argument("--only", default=None, help="file listing tile ids to keep (optional)")
    a = ap.parse_args()
    ki, kt, s = ARMS[a.arm]
    px = {"x2": 10, "x4": 20, "x2multi": 10, "g2": 5}[a.arm]
    models = {}
    for r in a.runs:
        G = Gen(s).to(DEV).eval()
        G.load_state_dict(torch.load(ROOT / f"ckpt/{r}.pt", weights_only=True, map_location=DEV))
        models[r] = G
    keep = set(Path(a.only).read_text().split()) if a.only else None
    rows = []
    for split in a.splits:
        for t in tiles_for(split, a.arm, a.target):
            if keep is not None and t["id"] not in keep:
                continue
            y, m, xn = from_n(t["y"]), t["m"], t["x"]
            if m.mean() < 0.9:
                continue
            x = from_n(xn)
            P = {"bicubic": bicubic(xn, s), "truth": y, **{k: predict(G, xn) for k, G in models.items()}}
            for k, p in P.items():
                rows.append({"split": split, "tile": t["id"], "model": k, **scores(p, y, m, x, s, px)})
        print(split, "done", len(rows), flush=True)
    out = ROOT / "eval2"; out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{a.label}_scores.csv", index=False)
    sm = df.groupby(["split", "model"])[COLS].median().round(3)
    sm["n"] = df.groupby(["split", "model"]).size()
    sm.to_csv(out / f"{a.label}_summary.csv")
    pd.set_option("display.width", 200)
    print(sm.to_string(), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
