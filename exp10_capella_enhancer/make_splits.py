"""Step 3: split sites into train / val / test by 3 x 3 degree regional blocks (DESIGN.md, "Splits").

Blocks holding any experiment 8 scene go to test. The rest, shuffled with seed 0, go to whichever split sits
furthest below its share of chips (70 / 15 / 15). Adds `block` and `split` to chips.csv and writes splits.csv
(one row per site).
"""
from __future__ import annotations

import common  # noqa: F401
from common import EXP8, HERE, finish

import traceback

import numpy as np
import pandas as pd

SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}


def block_of(lon, lat):
    return f"{int(np.floor(lat / 3))}_{int(np.floor(lon / 3))}"


def main():
    ch = pd.read_csv(HERE / "chips.csv").drop(columns=["split", "block"], errors="ignore")
    site = ch.groupby("site_id").agg(lon=("lon", "mean"), lat=("lat", "mean"), n=("chip_id", "size"),
                                     country=("country", "first")).reset_index()
    site["block"] = [block_of(a, b) for a, b in zip(site.lon, site.lat)]
    e8 = pd.read_csv(EXP8 / "scenes_selected.csv")
    forced = {block_of(a, b) for a, b in zip(e8.lon, e8.lat)}
    blocks = site.groupby("block").n.sum()
    total = blocks.sum()
    cur = {k: 0 for k in SHARES}
    assign = {}
    for b in blocks.index:
        if b in forced:
            assign[b] = "test"
            cur["test"] += blocks[b]
    rng = np.random.default_rng(0)
    rest = [b for b in blocks.index if b not in forced]
    rng.shuffle(rest)
    for b in rest:
        s = max(SHARES, key=lambda k: SHARES[k] * total - cur[k])
        assign[b] = s
        cur[s] += blocks[b]
    site["split"] = site.block.map(assign)
    site["exp8_block"] = site.block.isin(forced)
    site.to_csv(HERE / "splits.csv", index=False)
    ch = ch.merge(site[["site_id", "block", "split"]], on="site_id", how="left")
    ch.to_csv(HERE / "chips.csv", index=False)
    print("forced test blocks (exp8):", len(forced & set(blocks.index)), "of", len(forced), flush=True)
    print("chips by split:", ch.split.value_counts().to_dict(), flush=True)
    print("sites by split:", site.split.value_counts().to_dict(), flush=True)
    print("blocks by split:", pd.Series(assign).value_counts().to_dict(), flush=True)
    print(pd.crosstab(ch.cover, ch.split).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()
