"""Pick train / near-holdout / far-holdout NISAR track-frames that have simultaneous 20 MHz (mixed-mode)
and 40 MHz GCOV products, and up to two acquisition dates per frame."""
from config import DATA  # noqa: E402
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
tf = gpd.read_file(str(DATA) + "/pair_frames.gpkg")
pairs = pd.read_csv(str(DATA) + "/mixed_pairs.csv")
tf["key"] = list(zip(tf.track, tf.frame))
far = [(82, 106), (45, 108), (143, 164)]          # Buenos Aires, Melbourne, Johannesburg: X-band rich
near = [(6, 76), (39, 14)]                        # Abu Dhabi (near Dubai), north Taiwan (near south Taiwan)
train_fixed = [(118, 71), (56, 14), (61, 77), (44, 17), (82, 20), (116, 56)]
tfm = tf.to_crs("ESRI:54034")                      # world cylindrical equal area: distances only roughly
far_geo = tfm[tfm.key.isin(far)]
def dist_km(g, others):
    return min(g.distance(o) for o in others.geometry) / 1000
tf["d_far"] = [dist_km(g, far_geo) for g in tfm.geometry]
tf["reg"] = (np.round(tf.lat / 10) * 10).astype(int).astype(str) + "," + (np.round(tf.lon / 10) * 10).astype(int).astype(str)
cand = tf[(tf.n >= 4) & (tf.d_far > 800) & (~tf.key.isin(far + near + train_fixed)) & (tf.lat.abs() < 62)]
extra_regions = ["10,0", "-20,-50", "40,0", "50,10", "40,-90", "20,100", "0,30", "-10,130", "10,-70", "30,50", "-20,50", "50,-90"]
extra = [cand[cand.reg == r].sort_values("n", ascending=False).key.iloc[0] for r in extra_regions if (cand.reg == r).any()]
train = train_fixed + extra
tr_geo = tfm[tfm.key.isin(train)]
tf["d_train"] = [dist_km(g, tr_geo) for g in tfm.geometry]
role = {k: "train" for k in train} | {k: "near" for k in near} | {k: "far" for k in far}
sel = tf[tf.key.isin(role)].copy()
sel["role"] = sel.key.map(role)
print(sel[["track", "frame", "role", "reg", "lon", "lat", "n", "nx", "d_train", "d_far"]].round(1).to_string(), flush=True)
rows = []
for r in sel.itertuples():
    p = pairs[(pairs.track == r.track) & (pairs.frame == r.frame)].sort_values("t0_a")
    p = p.iloc[[0, -1]] if len(p) > 1 else p     # earliest and latest date: two seasons, independent speckle
    for q in p.drop_duplicates("id_a").itertuples():
        rows.append({"role": r.role, "track": r.track, "frame": r.frame, "reg": r.reg, "lon": r.lon, "lat": r.lat,
                     "id_m": q.id_m, "url_m": q.url_m, "id_a": q.id_a, "url_a": q.url_a, "date": q.t0_a[:8], "full_a": q.full_a})
out = pd.DataFrame(rows)
out.to_csv("selection.csv", index=False)
print(out.groupby("role").size(), flush=True)
sel.drop(columns=["key"]).to_file(str(DATA) + "/selected_frames.gpkg")
sys.stdout.flush(); os._exit(0)
