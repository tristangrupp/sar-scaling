"""Shared settings for Experiment 4: sites, paths, I/O helpers."""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
GAN_SPIKE = ROOT.parent / "exp1_gan_spike"

# name: (lon, lat, land cover, split, start, end)
# Windows: one season of 2025, chosen to avoid the main snow or flood transitions where we could.
SITES = {
    # ---- training sites (10.24 km)
    "iowa_crop":         (-93.55, 42.05, "cropland", "train", "2025-05-01", "2025-10-31"),
    "punjab_crop":       (75.60, 30.70, "cropland", "train", "2025-01-01", "2025-06-30"),
    "beauce_crop":       (1.65, 48.20, "cropland", "train", "2025-04-01", "2025-08-31"),
    "rondonia_forest":   (-62.90, -10.10, "forest", "train", "2025-05-01", "2025-10-31"),
    "losangeles_urban":  (-118.25, 34.02, "urban", "train", "2025-01-01", "2025-06-30"),
    "alps_mountain":     (7.75, 46.02, "mountain", "train", "2025-06-01", "2025-10-15"),
    "sahara_desert":     (8.50, 27.50, "desert", "train", "2025-01-01", "2025-06-30"),
    "pantanal_wetland":  (-57.00, -18.50, "wetland", "train", "2025-05-01", "2025-10-31"),
    # ---- held-out sites (5.12 km), never seen in training
    "matogrosso_crop":   (-55.90, -13.00, "cropland", "test", "2025-05-01", "2025-10-31"),
    "gabon_forest":      (11.50, -0.50, "forest", "test", "2025-01-01", "2025-06-30"),
    "madrid_urban":      (-3.68, 40.43, "urban", "test", "2025-03-01", "2025-08-31"),
    "rockies_mountain":  (-106.10, 39.60, "mountain", "test", "2025-06-15", "2025-10-15"),
    "atacama_desert":    (-69.30, -23.60, "desert", "test", "2025-01-01", "2025-06-30"),
    "louisiana_wetland": (-90.60, 29.55, "wetland", "test", "2025-01-01", "2025-06-30"),
}
SPAN = {"train": 10240, "test": 5120}   # m

# Round 2: extra Noise2Noise training sites (date pairs only; no clean target is built).
# name: (lon, lat, land cover, start, end). All far from the six held-out sites.
N2N_SITES = {
    "namib_dunes":       (15.20, -24.70, "desert", "2025-01-01", "2025-08-31"),
    "sonoran_desert":    (-112.90, 32.90, "desert", "2025-01-01", "2025-08-31"),
    "gobi_desert":       (105.50, 42.50, "desert", "2025-04-01", "2025-10-31"),
    "thar_desert":       (71.00, 27.00, "desert", "2025-01-01", "2025-08-31"),
    "tokyo_urban":       (139.75, 35.68, "urban", "2025-01-01", "2025-08-31"),
    "cairo_urban":       (31.25, 30.05, "urban", "2025-01-01", "2025-08-31"),
    "saopaulo_urban":    (-46.63, -23.55, "urban", "2025-01-01", "2025-08-31"),
    "lagos_urban":       (3.38, 6.50, "urban", "2025-01-01", "2025-08-31"),
    "okavango_wetland":  (22.80, -19.40, "wetland", "2025-01-01", "2025-08-31"),
    "everglades_wetland": (-80.70, 25.80, "wetland", "2025-01-01", "2025-08-31"),
    "sundarbans_wetland": (89.20, 21.90, "wetland", "2025-01-01", "2025-08-31"),
    "andes_mountain":    (-72.50, -13.30, "mountain", "2025-04-01", "2025-10-31"),
    "norway_mountain":   (6.90, 61.20, "mountain", "2025-06-01", "2025-09-30"),
    "taiwan_mountain":   (121.20, 23.80, "mountain", "2025-01-01", "2025-08-31"),
    "ukraine_crop":      (33.00, 49.00, "cropland", "2025-04-01", "2025-09-30"),
    "kansas_crop":       (-98.00, 38.50, "cropland", "2025-04-01", "2025-10-31"),
    "pampas_crop":       (-61.00, -34.00, "cropland", "2025-01-01", "2025-08-31"),
    "borneo_forest":     (114.50, 0.50, "forest", "2025-01-01", "2025-08-31"),
}
FTW_SRC = Path(r"H:/FTL_FTW/source")      # Fields of The World labels, read-only
MAX_DATES = 40
MIN_DATES = 20


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def site_grid(name):
    from pyproj import Transformer
    lon, lat, _, split, _, _ = SITES[name]
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    s = SPAN[split]
    return epsg, float(np.floor((x - s / 2) / 10) * 10), float(np.floor((y + s / 2) / 10) * 10), s // 10


def norm_stats(db):
    v = db[np.isfinite(db)]
    return float(np.median(v)), float(v.std() + 1e-6)


def finish():
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def enl_hat(lin, w=7, q=90):
    """No-reference ENL estimate of one image (linear power): the q-th percentile of local w x w
    ENL (mean^2/var) on a stride-w grid. Homogeneous windows carry the highest local ENL, so a
    high percentile tracks the speckle level; texture pulls it down. Used as the conditioning
    signal in round 2 and computed the same way at training and test time."""
    from scipy import ndimage as ndi
    a = np.where(np.isfinite(lin), lin, np.nanmedian(lin)).astype("float64")
    m = ndi.uniform_filter(a, w)
    v = np.maximum(ndi.uniform_filter(a * a, w) - m * m, 1e-20)
    e = (m * m / v)[w // 2::w, w // 2::w]
    return float(np.clip(np.percentile(e, q), 1, 500))
