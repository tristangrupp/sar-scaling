"""Terrain-correct a Stripmap SLC (VV) onto a 5 m UTM tile with sarsen.

sarsen samples the SLC at each DEM pixel, so the DEM grid sets the output grid. To multilook, the
DEM is resampled to 1.25 m over the tile and the gamma0 output is averaged 4 x 4 to 5 m.

  python process_slc.py <SAFE dir> <lon> <lat> <tile_m> <out.tif>
Runs in the s1slc env (sarsen).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

FINE, OUT = 1.25, 5.0
# Terrain flattening (gamma_bilinear) gives gamma0_T, consistent with the IW RTC input and with the
# published baseline (SNAP terrain flattening). On the uncropped SLC it ran >35 min for a 1.6 km
# tile; with the SLC cropped to the tile it takes about 80 s for 3.2 km. Set SM_RADIOMETRY=none
# to get uncorrected beta0 instead (about 4 dB brighter at S1-beam incidence, tan(theta)).
_r = os.environ.get("SM_RADIOMETRY", "gamma_bilinear")
RADIOMETRY = None if _r.lower() == "none" else _r


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def tile_grid(lon, lat, tile_m):
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    x0 = np.floor((x - tile_m / 2) / 40) * 40
    y1 = np.floor((y + tile_m / 2) / 40) * 40
    return epsg, x0, y1


def geoid_undulation(lon, lat):
    """EGM2008 geoid height above WGS84 (m), via the PROJ network grid."""
    import pyproj
    pyproj.network.set_network_enabled(True)
    t = Transformer.from_crs("EPSG:9518", "EPSG:4979", always_xy=True)
    return float(t.transform(lon, lat, 0.0)[2])


def write_dem(lon, lat, epsg, x0, y1, n, path):
    """Copernicus GLO-30 DEM from Planetary Computer, bilinear to the fine UTM grid."""
    c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    d = 0.1
    items = list(c.search(collections=["cop-dem-glo-30"], bbox=[lon - d, lat - d, lon + d, lat + d]).items())
    tr = from_origin(x0, y1, FINE, FINE)
    dem = np.full((n, n), np.nan, "float32")
    for it in items:
        with rasterio.open(it.assets["data"].href) as s, WarpedVRT(
                s, crs=f"EPSG:{epsg}", transform=tr, width=n, height=n, resampling=Resampling.bilinear,
                nodata=-9999, dtype="float32") as v:
            a = v.read(1)
            dem = np.where(np.isnan(dem) & (a != -9999), a, dem)
    dem = np.nan_to_num(dem, nan=0.0)          # sea level where no tile (open ocean)
    # GLO-30 heights are EGM2008; sarsen treats DEM heights as ellipsoidal. Without this the
    # geoid undulation (e.g. -28 m at Houston) shifts the geocoded image by N / tan(incidence),
    # about 75 m at 20 degrees. One value per tile: the geoid varies slowly over 3 km.
    dem = dem + geoid_undulation(lon, lat)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs=f"EPSG:{epsg}", transform=tr) as o:
        o.write(dem, 1)


def cropped_product(safe, lon_min, lat_min, lon_max, lat_max, margin_px=600):
    """Sentinel-1 product whose measurement is sliced to the tile, so sarsen reads only that window.

    Uncropped, sarsen loaded the whole 4.2 GB SLC (6-8 GB RAM, >15 min for a 1.6 km tile).
    The window comes from the annotation GCP grid, padded for terrain displacement.
    """
    from sarsen import sentinel1

    class Cropped(sentinel1.Sentinel1SarProduct):
        @property
        def measurement(self):
            ds = super().measurement
            return ds.isel(azimuth_time=slice(*self._rows), slant_range_time=slice(*self._cols))

    prod = Cropped(str(safe), measurement_group=sentinel1_group(safe))
    full = sentinel1.Sentinel1SarProduct.measurement.fget(prod)
    g = prod.gcp
    lat_g, lon_g = g.latitude.values, g.longitude.values
    inside = (lon_g >= lon_min) & (lon_g <= lon_max) & (lat_g >= lat_min) & (lat_g <= lat_max)
    if not inside.any():          # tile smaller than GCP spacing: take the nearest GCP cells
        d = (lon_g - (lon_min + lon_max) / 2) ** 2 + (lat_g - (lat_min + lat_max) / 2) ** 2
        inside = d <= np.partition(d.ravel(), 4)[4]
    ia, ir = np.nonzero(inside)
    at, srt = g.azimuth_time.values, g.slant_range_time.values
    a_idx = np.searchsorted(full.azimuth_time.values, at[[ia.min(), ia.max()]])
    r_idx = np.searchsorted(full.slant_range_time.values, srt[[ir.min(), ir.max()]])
    # one GCP cell beyond the hit range on each side, plus a margin
    da = int(np.diff(np.searchsorted(full.azimuth_time.values, at[:2]))[0]) if len(at) > 1 else 0
    dr = int(np.diff(np.searchsorted(full.slant_range_time.values, srt[:2]))[0]) if len(srt) > 1 else 0
    na, nr = full.sizes["azimuth_time"], full.sizes["slant_range_time"]
    prod._rows = (max(int(a_idx[0]) - da - margin_px, 0), min(int(a_idx[1]) + da + margin_px, na))
    prod._cols = (max(int(r_idx[0]) - dr - margin_px, 0), min(int(r_idx[1]) + dr + margin_px, nr))
    print("SLC window rows", prod._rows, "cols", prod._cols, "of", (na, nr), flush=True)
    return prod


def main(safe, lon, lat, tile_m, out_tif):
    from sarsen import apps
    lon, lat, tile_m = float(lon), float(lat), float(tile_m)
    epsg, x0, y1 = tile_grid(lon, lat, tile_m)
    n_fine = int(tile_m / FINE)
    tmp = Path(tempfile.mkdtemp())
    dem = tmp / "dem.tif"
    write_dem(lon, lat, epsg, x0, y1, n_fine, dem)
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    lons, lats = to_ll.transform([x0, x0 + tile_m, x0, x0 + tile_m], [y1, y1, y1 - tile_m, y1 - tile_m])
    product = cropped_product(safe, min(lons), min(lats), max(lons), max(lats))
    gtc = tmp / "gtc.tif"
    apps.terrain_correction(product, str(dem), output_urlpath=str(gtc), correct_radiometry=RADIOMETRY,
                            interp_method="nearest", grouping_area_factor=(4.0, 4.0), chunks=1024)
    with rasterio.open(gtc) as s:
        g = s.read(1).astype("float64")
        # sarsen writes the DEM's y coordinate as ascending, i.e. a south-up raster (positive
        # y pixel size). Flip it to north-up before writing on the north-up tile grid.
        if s.transform.e > 0:
            g = g[::-1]
        if s.transform.a < 0:
            g = g[:, ::-1]
    g[~np.isfinite(g) | (g <= 0)] = np.nan
    n = int(tile_m / OUT)
    k = int(OUT / FINE)
    blocks = g.reshape(n, k, n, k)
    valid = np.isfinite(blocks).sum(axis=(1, 3))
    g5 = np.where(valid >= k * k // 2, np.nanmean(blocks, axis=(1, 3)), np.nan).astype("float32")
    with rasterio.open(out_tif, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, OUT, OUT), nodata=np.nan,
                       compress="deflate") as o:
        o.write(g5, 1)
    print("wrote", out_tif, "valid", float(np.isfinite(g5).mean()), flush=True)


def sentinel1_group(safe):
    """xarray-sentinel group for the VV Stripmap measurement, e.g. 'S3/VV'."""
    name = Path(safe).name
    return f"{name.split('_')[1]}/VV"


if __name__ == "__main__":
    try:
        main(*sys.argv[1:6])
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)
