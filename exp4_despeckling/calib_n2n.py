"""Noise2Noise with L1 in dB learns the median of the log of a speckled date, which sits below the
log of the mean intensity. Fit one constant offset (dB) on TRAINING sites only: the median of
(reference dB - N2N output dB) over stable pixels of 4 dates per site. Writes ckpt/n2n_offset.txt."""
from __future__ import annotations

import common  # noqa: F401
from common import DATA, ROOT, SITES, finish, norm_stats

import numpy as np
import torch

from train4 import DEV, DGen, to_db


def main():
    G = DGen().to(DEV).eval()
    G.load_state_dict(torch.load(ROOT / "ckpt" / "n2n.pt", weights_only=True))
    diffs = []
    for name, v in SITES.items():
        if v[3] != "train":
            continue
        z, st = np.load(DATA / f"{name}.npz"), np.load(DATA / f"{name}_stack.npz")
        vv = z["vv"]
        for t in np.linspace(0, vv.shape[0] - 1, 4).round().astype(int):
            I = vv[t]
            valid = np.isfinite(I)
            ref = (st["sum"] - np.where(valid, I, 0)) / np.maximum(st["n"] - valid, 1)
            db = to_db(np.where(valid, I, np.nanmedian(I)))
            med, sd = norm_stats(db)
            with torch.no_grad():
                y = G(torch.from_numpy(((db - med) / sd).astype("float32"))[None, None].to(DEV))[0, 0].cpu().numpy()
            M = st["stable"] & valid
            diffs.append(np.median((to_db(ref) - (y * sd + med))[M]))
        print(name, np.round(diffs[-4:], 3), flush=True)
    off = float(np.median(diffs))
    (ROOT / "ckpt" / "n2n_offset.txt").write_text(f"{off:.4f}")
    print("offset dB", off, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finish()
