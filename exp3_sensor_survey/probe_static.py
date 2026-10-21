"""Round 2 step 4 probe: is the NISAR L2 static-layers product (layover/shadow) reachable by windowed read?"""
import os, sys
from nisar_io import open_remote, TOKEN
import pandas as pd, h5py, requests
from nisar_io import open_remote, TOKEN
t = pd.read_csv("tiles.csv").iloc[0]
gid = t.url_a.split("/")[-1]
u = f"https://nisar-services.earthdata.nasa.gov/redirect/NISAR_L2_STATIC/{gid}"
r = requests.get(u, headers={"Authorization": f"Bearer {TOKEN}"}, allow_redirects=False, timeout=60)
print("redirect status", r.status_code, r.headers.get("Location", "")[:160], flush=True)
loc = r.headers.get("Location")
if loc:
    h = open_remote(loc)
    def show(n, o):
        if isinstance(o, h5py.Dataset) and o.ndim >= 2:
            print(n, o.shape, o.dtype, o.chunks, flush=True)
    h.visititems(show)
sys.stdout.flush(); os._exit(0)
