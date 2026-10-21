"""splits_v2.csv: round 1 assignments copied unchanged from splits.csv (exp5 uses that file), plus
round 2 tiles: held-out regions (yemen_west = near2, bangladesh = far2) -> test; about 15% of tile
locations per new training region -> val; the rest -> train. Adds round and land-cover theme.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import ROOT
from make_splits import clusters


def main():
    t = pd.read_csv(ROOT / "tiles_v2.csv")
    pf = pd.read_csv(ROOT / "pairs_final_v2.csv")
    old = pd.read_csv(ROOT / "splits.csv")
    new = t[t["round"] == 2].copy()
    rng = np.random.default_rng(1)
    new["split"] = np.where(new.role.str.startswith("holdout"), "test", "train")
    new["cluster"] = -1
    for reg, g in new[new.split == "train"].groupby("region"):
        c = clusters(g)
        new.loc[g.index, "cluster"] = c
        k = max(1, int(round(0.15 * len(np.unique(c)))))
        pick = rng.choice(np.unique(c), size=k, replace=False)
        new.loc[g.index[np.isin(c, pick)], "split"] = "val"
    cols = ["tile_id", "region", "role", "split", "cluster", "lon", "lat", "n_xband"]
    out = pd.concat([old[cols].assign(round=1), new[cols].assign(round=2)], ignore_index=True)
    out = out.merge(t[["tile_id", "theme"]], on="tile_id", how="left")
    out["used"] = out.tile_id.isin(pf[pf.use].tile_id)
    out.to_csv(ROOT / "splits_v2.csv", index=False)
    print(out[out.used].groupby(["round", "region", "split"]).size().unstack(fill_value=0).to_string())
    assert not set(out[out.split != "test"].region) & set(out[out.split == "test"].region)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
