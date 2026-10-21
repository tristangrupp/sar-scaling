"""Quicklook: Stripmap 5 m tile next to the nearest-date IW RTC on the same grid."""
import os, sys
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="60")
import numpy as np, rasterio, pandas as pd, planetary_computer as pc, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pystac_client import Client
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
tif, date, out = sys.argv[1], pd.Timestamp(sys.argv[2], tz="UTC"), sys.argv[3]
with rasterio.open(tif) as s:
    sm = 10 * np.log10(s.read(1)); crs, tr, w, h = s.crs, s.transform, s.width, s.height
    from rasterio.warp import transform_bounds
    b = transform_bounds(crs, 4326, *s.bounds)
c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
items = sorted([i for i in c.search(collections=["sentinel-1-rtc"], bbox=b, datetime=f"{date - pd.Timedelta(days=6):%Y-%m-%d}/{date + pd.Timedelta(days=6):%Y-%m-%d}").items() if "vv" in i.assets],
               key=lambda i: abs(pd.Timestamp(i.datetime) - date))
it = items[0]
with rasterio.open(it.assets["vv"].href) as s, WarpedVRT(s, crs=crs, transform=rasterio.Affine(10, 0, tr.c, 0, -10, tr.f), width=w // 2, height=h // 2, resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
    iw = v.read(1); iw[iw <= 0] = np.nan; iw = 10 * np.log10(iw)
lo, hi = np.nanpercentile(sm, [2, 98])
fig, ax = plt.subplots(1, 2, figsize=(11, 5.6), dpi=100)
ax[0].imshow(np.kron(iw, np.ones((2, 2))), cmap="gray", vmin=lo, vmax=hi); ax[0].set_title(f"Sentinel-1 IW RTC, 10 m ({it.id[:25]})", fontsize=9)
ax[1].imshow(sm, cmap="gray", vmin=lo, vmax=hi); ax[1].set_title("Sentinel-1 Stripmap SLC, sarsen geocoded, 5 m (4x4 of 1.25 m)", fontsize=9)
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig(out); print("ok", it.id, flush=True)
sys.stdout.flush(); os._exit(0)
