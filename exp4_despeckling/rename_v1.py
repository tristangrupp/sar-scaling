"""One-off: the Exp 1 rows scored so far used the v1 (no Capella) weights; rename them exp1v1_* so the
Capella-retrained weights can be scored as exp1_* beside them."""
import common  # noqa: F401
from common import DATA, ROOT, finish
import numpy as np, pandas as pd
d = pd.read_csv(ROOT / "metrics_dates.csv")
d["method"] = d.method.replace({"exp1_l1": "exp1v1_l1", "exp1_gan": "exp1v1_gan"})
d.to_csv(ROOT / "metrics_dates.csv", index=False)
for f in (DATA / "out").glob("*.npz"):
    z = dict(np.load(f))
    for k in ["l1", "gan"]:
        if f"exp1_{k}" in z:
            z[f"exp1v1_{k}"] = z.pop(f"exp1_{k}")
    np.savez(f, **z)
print(d.method.value_counts(), flush=True)
finish()
