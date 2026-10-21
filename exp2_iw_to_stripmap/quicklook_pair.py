"""Quicklook of one built tile: co-registered IW 10 m, Stripmap 5 m, Lee-filtered Stripmap.

  python quicklook_pair.py <tile_id> <out.png>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import TILES

tid, out = sys.argv[1], sys.argv[2]
z = np.load(TILES / f"{tid}.npz")
sm, lee, iw = (z[k].astype("float32") for k in ("sm", "lee", "iw"))
lo, hi = np.nanpercentile(sm, [2, 98])
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), dpi=90)
for a, im, t in zip(ax, [np.kron(iw, np.ones((2, 2))), sm, lee],
                    ["Sentinel-1 IW RTC VV, 10 m (co-registered)", "Stripmap SLC → 5 m gamma0 (sarsen, terrain-flattened)",
                     "Stripmap, Lee 7x7 (the paper's reference)"]):
    a.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    a.set_title(t, fontsize=9)
    a.axis("off")
fig.suptitle(f"{tid}: 3.2 km tile, north up", fontsize=10)
fig.tight_layout()
fig.savefig(out)
sys.stdout.flush()
os._exit(0)
