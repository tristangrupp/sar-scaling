"""Round 4 predictions on Experiment 2's test tiles, resumable and device-selectable (CPU while the GPU is held).

Uses exp 2's predict() and normalization unchanged. Saves the normalized prediction per model and tile to
data/r4_preds/<model>/<tile_id>.npy (float32); existing files are skipped, so several processes can split
the model list (--shard i/n) and a stopped run resumes.

  CUDA_VISIBLE_DEVICES= python e5r4_predict.py --device cpu --threads 4 --shard 0/5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent / "exp2_iw_to_stripmap"
sys.path[0] = str(EXP2)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_sr as TS  # noqa: E402
from sr_data import Norm, load_manifest, load_split  # noqa: E402

OUT = HERE / "data" / "r4_preds"


def model_list():
    L = [("esrt_a", EXP2 / "ckpt" / "esrt_a.pt"), ("cnn_a", EXP2 / "ckpt" / "cnn_a.pt")]
    L += [(p.stem, p) for p in sorted((HERE / "ckpt" / "r4").glob("r4_*.pt")) if p.with_suffix(".json").exists()]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="", help="predict only this model name")
    a = ap.parse_args()
    TS.DEV = a.device                         # exp 2's predict() reads this module global
    torch.set_num_threads(a.threads)
    i, n = (int(v) for v in a.shard.split("/"))
    todo = [m for k, m in enumerate(model_list()) if (m[0] == a.only if a.only else k % n == i)]
    test = load_split(["test"], load_manifest())
    for name, path in todo:
        d = OUT / name
        if d.exists() and len(list(d.glob("*.npy"))) >= len(test):
            continue
        d.mkdir(parents=True, exist_ok=True)
        ck = torch.load(path, map_location=a.device, weights_only=False)
        net = TS.make_model(ck["model"]).to(a.device)
        net.load_state_dict(ck["state"])
        norm = Norm(*ck["norm"])
        with torch.no_grad():
            for t in test:
                f = d / f"{t['tile_id']}.npy"
                if f.exists():
                    continue
                np.save(f, TS.predict(net, norm(t["iw"])).astype("float32"))
        print("done", name, flush=True)
    print("shard finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
