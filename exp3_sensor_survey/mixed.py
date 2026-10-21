from config import DATA  # noqa: E402
import os, sys, collections, json
import pandas as pd
from nisar_io import search
pts = {"Miami": (-80.19, 25.76), "Tokyo": (139.69, 35.69), "Dubai": (55.27, 25.20), "Delhi": (77.21, 28.61), "Bengaluru": (77.59, 12.97),
       "Iowa_farm": (-93.5, 42.0), "Kansas": (-98.0, 38.5), "Chicago": (-87.63, 41.88)}
rows = []
for n, p in pts.items():
    for g in search(point=p, page_size=500):
        g["site"] = n; rows.append(g)
df = pd.DataFrame(rows)
df.to_csv(str(DATA) + "/mixed_sites.csv", index=False)
df["date"] = df.start.str[:10]
for (s, tr, fr), d in df.groupby(["site", "track", "frame"]):
    print(s, tr, fr, d.dir.iloc[0], " ".join(f"{a[5:]}:{b}{'q' if 'VV' in str(c) else ''}" for a, b, c in sorted(zip(d.date, d.bwA, d.polA))), flush=True)
sys.stdout.flush(); os._exit(0)
