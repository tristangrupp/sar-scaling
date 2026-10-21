"""Assign each tile to train / val / test.

test   every tile of a held-out region (houston, owens, w_australia) and of the paper's three test
       scenes (amieva_test); nothing from these places enters training, tuning or the X-band loss
val    the paper's two validation scenes, plus about 15% of tile *locations* in each training region
       (all dates of a location go together, so the same ground never sits in both train and val)
train  the rest
Writes splits.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import ROOT


def clusters(df, radius_deg=0.02):
    """Group tiles of one region whose centers lie within ~2 km of each other (same ground)."""
    cid = -np.ones(len(df), int)
    pts = df[["lon", "lat"]].to_numpy()
    k = 0
    for i in range(len(df)):
        if cid[i] >= 0:
            continue
        near = np.hypot(*(pts - pts[i]).T) < radius_deg
        cid[near & (cid < 0)] = k
        k += 1
    return cid


def main():
    t = pd.read_csv(ROOT / "tiles.csv")
    rng = np.random.default_rng(0)
    split = pd.Series("train", index=t.index)
    split[t.role.str.startswith("holdout")] = "test"
    split[t.role == "val"] = "val"
    t["cluster"] = -1
    for reg, g in t[t.role == "train"].groupby("region"):
        c = clusters(g)
        t.loc[g.index, "cluster"] = c
        n = len(np.unique(c))
        k = max(1, int(round(0.15 * n)))
        pick = rng.choice(np.unique(c), size=k, replace=False)
        split[g.index[np.isin(c, pick)]] = "val"
    t["split"] = split
    t[["tile_id", "region", "role", "split", "cluster", "lon", "lat", "n_xband"]].to_csv(ROOT / "splits.csv",
                                                                                       index=False)
    print(t.groupby(["region", "split"]).size().unstack(fill_value=0).to_string())
    # leakage check: no test region name appears in train/val
    assert not set(t[t.split != "test"].region) & set(t[t.split == "test"].region)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
