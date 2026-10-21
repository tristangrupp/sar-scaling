"""Round 3: predict Experiment 2's held-out Stripmap tiles with every exp 5 model, saved in dB for e5r3_score.py.

Models: the Stripmap learning-curve runs (ckpt/r3/sm_*), the three-band joint model with each target code,
and zero-shot models that never saw Stripmap (round 1 X-band 2x, round 2 X+NISAR joint).
Outputs: data/r3_preds/<model>/<tile_id>.npy (dB, float16), in this experiment's folder.
"""
from __future__ import annotations

import common  # noqa: F401
from common import ROOT, finish, load_model, meta_vec, run_model, sm_tiles

import numpy as np

from e5r2_joint import code

OUT = ROOT / "data" / "r3_preds"


def main():
    M = {}
    for f in sorted((ROOT / "ckpt" / "r3").glob("*.pt")):
        G, ck = load_model(f)
        if f.stem.startswith("joint"):
            for tg, nm in [("s1sm", "stripmap"), ("nisar40", "NISAR"), ("xspot", "X spotlight")]:
                M[f"{f.stem} [target {nm}]"] = (G, code("s1iw", tg))
        else:
            M[f.stem] = (G, None)
    for k in ["x2_unc", "x2_cond"]:
        G, ck = load_model(ROOT / "ckpt" / f"{k}.pt")
        M[f"zeroshot_{k}"] = (G, meta_vec(258, 37, "VV", "X"))
    jf = ROOT / "ckpt" / "r2" / "joint.pt"
    if jf.exists():
        G, ck = load_model(jf)
        M["zeroshot_joint_r2 [target NISAR]"] = (G, code("xsyn", "nisar40"))
        M["zeroshot_joint_r2 [target X spotlight]"] = (G, code("xsyn", "xspot"))
    print("models:", list(M), flush=True)
    tiles = sm_tiles(["test"])
    for i, t in enumerate(tiles):
        for k, (G, c) in M.items():
            o = run_model(G, t["x"], c)
            d = OUT / k.replace(" ", "_").replace("[", "").replace("]", "")
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{t['id']}.npy", (o * t["sd"] + t["med"]).astype("float16"))
        if i % 20 == 0:
            print(i, len(tiles), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
    finish()
