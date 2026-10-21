"""Diagnose Stripmap/IW registration: normalized cross-correlation over +/-800 m shifts at 10 m."""
import common
from common import PROJ, ROOT, finish
import numpy as np, pandas as pd, rasterio, planetary_computer as pc
from pystac_client import Client
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from pyproj import Transformer
from skimage.feature import match_template
f = PROJ / "exp2_iw_to_stripmap/sm_tiles/test_houston.tif"
with rasterio.open(f) as s:
    sm = s.read(1); crs, b = s.crs, s.bounds
sm10 = 10*np.log10(np.nanmean(np.where(sm > 0, sm, np.nan).reshape(160, 2, 160, 2), axis=(1, 3)))
cl = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform((b.left+b.right)/2, (b.bottom+b.top)/2)
items = [i for i in cl.search(collections=["sentinel-1-rtc"], intersects={"type":"Point","coordinates":[lon,lat]}, datetime="2026-10-01/2026-10-20").items() if "vv" in i.assets]
M = 200
for it in items:
    with rasterio.open(it.assets["vv"].href) as s, WarpedVRT(s, crs=crs, transform=from_origin(b.left-M*10, b.top+M*10, 10, 10), width=160+2*M, height=160+2*M, resampling=Resampling.bilinear, nodata=-32768, dtype="float32") as v:
        a = v.read(1)
    a[(a==-32768)|(a<=0)] = np.nan
    if np.isfinite(a).mean() < 0.9: continue
    iw = 10*np.log10(np.where(np.isfinite(a), a, np.nanmedian(a)))
    t = np.nan_to_num(sm10, nan=np.nanmedian(sm10))
    cc = match_template(iw, t)
    r, c = np.unravel_index(cc.argmax(), cc.shape)
    print(it.id[:40], it.properties.get("sat:orbit_state"), "zero-shift ncc", round(float(cc[M, M]), 3), "best", round(float(cc.max()), 3), "offset m (row,col)", ((r-M)*10, (c-M)*10), flush=True)
finish()
