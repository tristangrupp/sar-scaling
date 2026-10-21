"""Round 3: score exp 5 predictions with Experiment 2's own scorer, so numbers sit beside its arm A.

Runs Experiment 2's code (score.tile_scores, homogeneous_windows, Norm from its cnn_a checkpoint) on its test
split, reading exp 5 predictions from data/r3_preds. Aggregation as in exp 2: mean per holdout group, plus
"all pilot holdouts" (everything except the paper test scenes). Bicubic is recomputed as a check against
exp 2's scores_summary.csv. Writes r3_scores_tiles.csv and r3_scores_summary.csv here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)            # exp 2's modules import its own common.py, not this folder's

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import score as S  # noqa: E402
from sr_data import Norm, load_manifest, load_split  # noqa: E402


def main():
    ck = torch.load(EXP2 / "ckpt" / "cnn_a.pt", map_location="cpu", weights_only=False)
    norm = Norm(*ck["norm"])
    test = load_split(["test"], load_manifest())
    PRED = HERE / "data" / "r3_preds"
    models = sorted(p.name for p in PRED.iterdir() if p.is_dir())
    rows = []
    for k, t in enumerate(test):
        wins = S.homogeneous_windows(t)
        preds = {"bicubic": S.bicubic(norm(t["iw"]))}
        for m in models:
            f = PRED / m / f"{t['tile_id']}.npy"
            if f.exists():
                preds[m] = norm(np.load(f).astype("float32"))
        for name, p in preds.items():
            rows.append({"tile_id": t["tile_id"], "region": t["region"], "group": S.GROUP.get(t["region"], t["region"]),
                         **S.tile_scores(name, p, t, norm, wins)})
        if k % 20 == 0:
            print(k, len(test), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "r3_scores_tiles.csv", index=False)
    num = df.select_dtypes("number").columns
    summ = pd.concat([df.groupby(["group", "model"])[num].mean().reset_index(),
                      df[df.region != "amieva_test"].assign(group="all pilot holdouts").groupby(["group", "model"])[num].mean().reset_index()])
    summ.to_csv(HERE / "r3_scores_summary.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
