"""Check enl_hat against the stack-based single-date ENL, for 1, 2 and 4 averaged dates."""
import common  # noqa: F401
from common import DATA, SITES, enl_hat, finish
import numpy as np, pandas as pd
rows = []
for name in SITES:
    vv = np.load(DATA / f"{name}.npz")["vv"]
    L = float(np.load(DATA / f"{name}_stack.npz")["L"])
    for k in (1, 2, 4):
        e = [enl_hat(np.nanmean(vv[t:t + k], 0)) for t in range(0, vv.shape[0] - k, 4)]
        rows.append(dict(site=name, L_stack=round(L, 2), k=k, enl_hat=round(float(np.median(e)), 2)))
d = pd.DataFrame(rows).pivot(index=["site", "L_stack"], columns="k", values="enl_hat")
print(d.to_string(), flush=True)
finish()
