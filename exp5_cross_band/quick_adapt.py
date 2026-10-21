"""Quick check: X-synthetic 4x on real S1 with and without the band adaptation, 20 held-out tiles."""
import common
from common import *
import numpy as np, pandas as pd
from scipy import ndimage
G, _ = load_model(ROOT / "ckpt/x4_cond.pt")
man = pd.read_csv(SPIKE / "manifest.csv"); man = man[(man.split == "test") & man.status.isin(["ok", "cached"])].head(20)
rows = []
for r in man.itertuples():
    z = np.load(SPIKE / "data" / f"{r.provider}_{r.xband_id}.npz"); x = z["x"].astype("float32"); m = np.isfinite(x)
    if m.mean() < 0.8: continue
    truth = np.nan_to_num(norm(x)); s1 = z["s1"].astype("float32"); s1 = np.where(np.isfinite(s1), s1, np.nanmedian(s1)); s = norm(s1)
    bic = bicubic(s, 4); sh = fine_register(bic, truth, m); truth = ndimage.shift(truth, sh, order=1, mode="nearest")
    te = edges(truth, m, 2.5); mv = meta_vec(258, 37, "VV", "X")
    for k, p in {"bicubic": bic, "bicubic+adapt": bicubic(adapt(s), 4), "x4": run_model(G, s, mv), "x4+adapt": run_model(G, adapt(s), mv)}.items():
        rows.append(dict(tile=r.xband_id, model=k, **scores(p, truth, m, 4), **chamfer(edges(p, m, 2.5), te, m, 2.5)))
print(pd.DataFrame(rows).groupby("model")[["detail_r", "hf_ratio", "chamfer_m", "edgeF_10m"]].median().round(3).to_string(), flush=True)
finish()
