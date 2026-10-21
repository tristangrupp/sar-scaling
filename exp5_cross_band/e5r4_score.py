"""Round 4 scoring with Experiment 2's own pipeline (predict, tile_scores, homogeneous windows, norm).

Models: bicubic, exp 2 arm A (esrt_a, cnn_a: the scratch seed 0 of each architecture), and every round 4
fine-tune in ckpt/r4 (r4_<model>_<init>_f<1|25>_s<seed>). All predicted here on exp 2's 152 test tiles.
Writes r4_scores_tiles.csv (one row per tile x model, all exp 2 metrics incl. ENL / MoI / MoR / Cx).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import score as S  # noqa: E402
from sr_data import Norm, load_manifest, load_split  # noqa: E402
from train_sr import DEV, make_model, predict  # noqa: E402


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""          # e.g. "cnn": score only that architecture
    nets, norm = {}, None
    for name, path in [("esrt_a", EXP2 / "ckpt" / "esrt_a.pt"), ("cnn_a", EXP2 / "ckpt" / "cnn_a.pt")] + \
            [(p.stem, p) for p in sorted((HERE / "ckpt" / "r4").glob("r4_*.pt")) if p.with_suffix(".json").exists()]:
        if only and only not in name:
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        nets[name] = None                    # predictions come from data/r4_preds (e5r4_predict.py)
        n = Norm(*ck["norm"])
        assert norm is None or (abs(n.lo - norm.lo) < 1e-6 and abs(n.hi - norm.hi) < 1e-6), "norm mismatch"
        norm = n
    print("models:", list(nets), flush=True)
    test = load_split(["test"], load_manifest())
    rows = []
    for k, t in enumerate(test):
        wins = S.homogeneous_windows(t)
        iw_n = norm(t["iw"])
        preds = {"bicubic": S.bicubic(iw_n)}
        for name in nets:
            preds[name] = np.load(HERE / "data" / "r4_preds" / name / f"{t['tile_id']}.npy").astype("float32")
        for name, p in preds.items():
            rows.append({"tile_id": t["tile_id"], "region": t["region"], "group": S.GROUP.get(t["region"], t["region"]),
                         **S.tile_scores(name, p, t, norm, wins)})
        if k % 10 == 0:
            print(k, len(test), flush=True)
    pd.DataFrame(rows).to_csv(HERE / (f"r4_scores_tiles_{only}.csv" if only else "r4_scores_tiles.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
