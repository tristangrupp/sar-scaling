import os, sys
from nisar_io import TOKEN
import pandas as pd, requests
t = pd.read_csv("tiles.csv").iloc[0]
gid = t.url_a.split("/")[-1]
u = f"https://nisar-services.earthdata.nasa.gov/redirect/NISAR_L2_STATIC/{gid}"
r = requests.get(u, headers={"Authorization": f"Bearer {TOKEN}"}, allow_redirects=False, timeout=60)
print(r.status_code, r.headers.get("content-type"), len(r.content), flush=True)
print(r.text[:1500], flush=True)
sys.stdout.flush(); os._exit(0)
