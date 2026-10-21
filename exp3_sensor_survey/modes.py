import os, sys, collections
from nisar_io import search
pts = {"Houston": (-95.36, 29.76), "LosAngeles": (-118.25, 34.05), "Denver": (-104.99, 39.74), "Chicago": (-87.63, 41.88),
       "NewYork": (-74.0, 40.71), "Seattle": (-122.33, 47.61), "Miami": (-80.19, 25.76), "Phoenix": (-112.07, 33.45),
       "Toronto": (-79.38, 43.65), "MexicoCity": (-99.13, 19.43), "SaoPaulo": (-46.63, -23.55), "Madrid": (-3.70, 40.42),
       "Berlin": (13.40, 52.52), "Rotterdam": (4.48, 51.92), "Lagos": (3.38, 6.52), "Nairobi": (36.82, -1.29), "Delhi": (77.21, 28.61),
       "Bengaluru": (77.59, 12.97), "Tokyo": (139.69, 35.69), "Sydney": (151.21, -33.87), "Dubai": (55.27, 25.20),
       "Jakarta": (106.85, -6.21), "Iowa_farm": (-93.5, 42.0), "Kansas": (-98.0, 38.5)}
for n, p in pts.items():
    g = search(point=p, page_size=200, extra={"temporal": "2026-06-17T00:00:00Z,2026-10-17T00:00:00Z"})
    c = collections.Counter(f"{x['bwA']}+{x['bwB']} {x['polA']}/{x['polB']}" for x in g)
    tr = sorted({(x['track'], x['dir']) for x in g})
    print(f"{n:12s} n={len(g):3d} modes={dict(c)} tracks={tr[:6]}", flush=True)
sys.stdout.flush(); os._exit(0)
