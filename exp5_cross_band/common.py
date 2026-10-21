"""Experiment 5 shared code: data, Sentinel-1-like degradation, conditioned generator, metrics.

Conventions
  - X-band tiles (exp1_gan_spike/data/*.npz, 'x') are 20*log10(amplitude) at 2.5 m, i.e. dB of power.
  - Inputs and targets are standardized per tile (median / std of the tile's own dB), as in the spike.
  - Metadata vector (7): cos(az), sin(az), (inc-35)/15, pol VV, pol HH, band X, band C.
    az = azimuth from the ground to the sensor, degrees clockwise from north, in image coordinates
    (north-up), so layover leans toward az. Rotations / flips of a patch transform az with it.
"""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
SPIKE = PROJ / "exp1_gan_spike"
DEV = "cuda"
HR = 2.5
# Experiment 4 stacks live outside the project folder; override with EXP4_DATA when the project moves.
EXP4_DATA = Path(os.environ.get("EXP4_DATA", PROJ / "exp4_despeckling" / "data"))
# Experiment 3 NISAR tiles (EXP3_DATA env var, same default as exp3_sensor_survey/config.py) and Experiment 2 Stripmap tiles
EXP3_DATA = Path(os.environ.get("EXP3_DATA", PROJ / "exp3_sensor_survey" / "data"))
SM_TILES = Path(__file__).resolve().parent.parent / "exp2_iw_to_stripmap" / "sm_tiles"
EXP3_VAL_FRAMES = [(4, 22), (152, 84), (8, 66)]          # copied from exp3_sensor_survey/config.py VAL_FRAMES

# Sentinel-1 IW true resolution (intensity PSF FWHM), metres
S1_RES_RANGE, S1_RES_AZ = 20.0, 22.0


def finish():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def spike_module(name="train"):
    """Load a exp1_gan_spike module by path without clashing with exp5 module names."""
    spec = importlib.util.spec_from_file_location(f"spike_{name}", SPIKE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SPIKE))
    spec.loader.exec_module(m)
    return m


def norm_stats(a):
    v = a[np.isfinite(a)]
    return float(np.median(v)), float(v.std() + 1e-6)


def norm(a):
    med, sd = norm_stats(a)
    return (a - med) / sd


# ---------------------------------------------------------------- metadata
S1_TO_SENSOR_AZ = {"ascending": 258.0, "descending": 102.0}   # right-looking, heading ~348 / ~192


def heading_to_sensor_az(orbit, side):
    head = 348.0 if str(orbit).lower().startswith("asc") else 192.0
    return (head + (90.0 if str(side).lower() == "left" else -90.0)) % 360


def tile_metadata():
    """Per X-band collect: incidence, azimuth to sensor, pol, and how az was obtained."""
    sel = pd.read_csv(SPIKE / "selection.csv")
    um = pd.DataFrame(json.loads(_gj_props(PROJ / "exp0_overlap_atlas/s1sm_umbra/umbra_items.geojson")))
    ic = pd.DataFrame(json.loads(_gj_props(PROJ / "exp0_overlap_atlas/s1sm_iceye/iceye_items.geojson")))
    cap_meta = json.loads((SPIKE / "ceiling/capella_meta.json").read_text())
    capg = pd.DataFrame(json.loads(_gj_props(PROJ / "exp0_overlap_atlas/s1sm_capella/capella_geo_items.geojson")))
    um_az = dict(zip(um.xband_id, um["view:azimuth"]))
    ic_os = dict(zip(ic.xband_id, zip(ic["sat:orbit_state"], ic["sar:observation_direction"])))
    cap_os = dict(zip(capg.capella_id, capg["sat:orbit_state"])) if "capella_id" in capg else {}
    rows = {}
    for r in sel.itertuples():
        inc = float(getattr(r, "_6")) if not pd.isna(getattr(r, "_6")) else 35.0
        pol = str(getattr(r, "_4")).upper()
        az, src = 270.0, "default"
        if r.provider == "umbra" and r.xband_id in um_az and not pd.isna(um_az[r.xband_id]):
            az, src = float(um_az[r.xband_id]) % 360, "stac view:azimuth"
        elif r.provider == "iceye" and r.xband_id in ic_os:
            o, s = ic_os[r.xband_id]
            az, src = heading_to_sensor_az(o, s), "orbit+side"
        elif r.provider == "capella":
            if r.xband_id in cap_meta and cap_meta[r.xband_id].get("view:azimuth") is not None:
                az, src = float(cap_meta[r.xband_id]["view:azimuth"]) % 360, "stac view:azimuth"
            elif r.xband_id in cap_os:
                az, src = heading_to_sensor_az(cap_os[r.xband_id], "right"), "orbit, side assumed right"
        rows[f"{r.provider}_{r.xband_id}"] = dict(inc=inc, az=az, pol=pol if pol in ("VV", "HH") else "VV",
                                                   az_src=src)
    return rows


def _gj_props(path):
    g = json.loads(Path(path).read_text())
    return json.dumps([f["properties"] for f in g["features"]])


def meta_vec(az, inc, pol="VV", band="X"):
    a = math.radians(az)
    return np.array([math.cos(a), math.sin(a), (inc - 35.0) / 15.0, pol == "VV", pol == "HH",
                     band == "X", band == "C"], "float32")


def aug_az(az, k, flip):
    """Azimuth after np.rot90(.., k) (counter-clockwise) then optional left-right flip (north-up)."""
    az = (az - 90.0 * k) % 360
    if flip:
        az = (360.0 - az) % 360
    return az


# ---------------------------------------------------------------- degradation (torch, GPU)
def psf_kernels(az_deg, res_r=S1_RES_RANGE, res_a=S1_RES_AZ, px=HR, size=33):
    """Amplitude PSF per sample: anisotropic Gaussian, range axis along az, sum(h^2)=1.
    Intensity PSF FWHM = res; amplitude sigma = sqrt(2) * intensity sigma."""
    az = torch.as_tensor(az_deg, dtype=torch.float32, device=DEV)
    s_r = math.sqrt(2) * res_r / 2.3548 / px
    s_a = math.sqrt(2) * res_a / 2.3548 / px
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(size, device=DEV) - c, torch.arange(size, device=DEV) - c, indexing="ij")
    th = torch.deg2rad(az)[:, None, None]
    # range unit vector in image coords: east = +x, north = -y
    ur_x, ur_y = torch.sin(th), -torch.cos(th)
    dr = xx * ur_x + yy * ur_y
    da = -xx * ur_y + yy * ur_x
    h = torch.exp(-0.5 * ((dr / s_r) ** 2 + (da / s_a) ** 2))
    h = h / torch.sqrt((h ** 2).sum(dim=(1, 2), keepdim=True))
    return h


def degrade(xdb, az, looks=5, factor=4, gen=None):
    """xdb: (B,1,H,W) dB of power at 2.5 m. Returns (B,1,H/factor,W/factor) dB at 10 m with
    S1-like resolution and speckle: |h * (sqrt(I) n)|^2 averaged over `looks` independent looks,
    then block-averaged to 10 m. Mean intensity is preserved (sum h^2 = 1, E|n|^2 = 1)."""
    B, _, H, W = xdb.shape
    amp = torch.sqrt(10 ** (xdb / 10)).float()
    h = psf_kernels(az)[:, None]                        # (B,1,k,k)
    pad = h.shape[-1] // 2
    acc = torch.zeros_like(amp)
    for _ in range(looks):
        nr = torch.randn(amp.shape, device=DEV, generator=gen) / math.sqrt(2)
        ni = torch.randn(amp.shape, device=DEV, generator=gen) / math.sqrt(2)
        zr = F.conv2d(F.pad((amp * nr).transpose(0, 1), (pad,) * 4, mode="reflect"), h, groups=B).transpose(0, 1)
        zi = F.conv2d(F.pad((amp * ni).transpose(0, 1), (pad,) * 4, mode="reflect"), h, groups=B).transpose(0, 1)
        acc += zr ** 2 + zi ** 2
    I = F.avg_pool2d(acc / looks, factor)
    return 10 * torch.log10(I.clamp(min=1e-10))


def blur_only(xdb, az, factor=4):
    """Same PSF, no speckle (the 'clean' 10 m image an infinite-look S1 would see)."""
    I = 10 ** (xdb / 10)
    B = I.shape[0]
    h2 = (psf_kernels(az) ** 2)[:, None]
    pad = h2.shape[-1] // 2
    o = F.conv2d(F.pad(I.transpose(0, 1), (pad,) * 4, mode="reflect"), h2, groups=B).transpose(0, 1)
    return 10 * torch.log10(F.avg_pool2d(o, factor).clamp(min=1e-10))


def pool_db(xdb, f):
    """Average power over f x f blocks, return dB (2.5 m -> 5 m target with f=2)."""
    return 10 * torch.log10(F.avg_pool2d(10 ** (xdb / 10), f).clamp(min=1e-10))


# ---------------------------------------------------------------- model
class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x, film=None):
        y = x + 0.2 * self.b(x)
        if film is not None:
            g, b = film
            y = y * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return y


class CondGen(nn.Module):
    """The spike's EDSR-style Gen (residual trunk at 10 m, pixel-shuffle x2 steps, bicubic skip),
    generalized to scale 2 or 4, with optional FiLM conditioning on the 7-d metadata vector.
    The FiLM head is zero-initialized, so training starts from the unconditioned network."""

    def __init__(self, scale=4, cond=False, c=64, nb=16, nmeta=7):
        super().__init__()
        self.scale, self.cond, self.nb, self.c = scale, cond, nb, c
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.blocks = nn.ModuleList([RB(c) for _ in range(nb)])
        self.tail = nn.Conv2d(c, c, 3, padding=1)
        up = []
        for _ in range(int(round(math.log2(scale)))):
            up += [nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(True)]
        self.up = nn.Sequential(*up, nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, 1, 3, padding=1))
        if cond:
            self.film = nn.Sequential(nn.Linear(nmeta, 128), nn.ReLU(True), nn.Linear(128, 2 * c * nb))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(self, s, meta=None):
        h = self.head(s)
        t = h
        fl = None
        if self.cond:
            fl = self.film(meta).view(-1, self.nb, 2, self.c)
        for i, blk in enumerate(self.blocks):
            t = blk(t, None if fl is None else (fl[:, i, 0], fl[:, i, 1]))
        h = h + self.tail(t)
        return self.up(h) + F.interpolate(s, scale_factor=self.scale, mode="bicubic", align_corners=False)


def load_model(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    G = CondGen(scale=ck["scale"], cond=ck["cond"], nmeta=ck.get("nmeta", 7)).to(DEV).eval()
    G.load_state_dict(ck["state"])
    return G, ck


def load_exp1(prefer_capella=True):
    """Exp 1 CNN (spike Gen, 4x). Returns (model, which)."""
    st = spike_module("train")
    new, old = SPIKE / "ckpt/l1.pt", SPIKE / "ckpt_v1_no_capella/l1.pt"
    use, which = old, "ckpt_v1_no_capella/l1.pt (before the Capella retrain)"
    if prefer_capella and new.exists() and new.stat().st_mtime > old.stat().st_mtime + 60:
        use, which = new, "ckpt/l1.pt (Capella retrain)"
    G = st.Gen().to(DEV).eval()
    G.load_state_dict(torch.load(use, map_location=DEV, weights_only=True))
    return G, which


@torch.no_grad()
def run_model(G, s_norm, meta=None, kind="cond", chunk=160, pad=16):
    """s_norm: (H,W) standardized dB at 10 m (NaN-free). Runs in overlapping chunks to bound GPU memory."""
    H, W = s_norm.shape
    sc = 4 if kind == "exp1" else G.scale
    m = None if (meta is None or kind == "exp1" or not G.cond) else torch.from_numpy(meta)[None].to(DEV)
    out = np.zeros((H * sc, W * sc), "float32")
    sp = np.pad(s_norm.astype("float32"), pad, mode="reflect")
    for r in range(0, H, chunk):
        for c in range(0, W, chunk):
            h, w = min(chunk, H - r), min(chunk, W - c)
            t = torch.from_numpy(np.ascontiguousarray(sp[r:r + h + 2 * pad, c:c + w + 2 * pad]))[None, None].to(DEV)
            o = (G(t) if kind == "exp1" else G(t, m))[0, 0].cpu().numpy()
            out[r * sc:(r + h) * sc, c * sc:(c + w) * sc] = o[pad * sc:(pad + h) * sc, pad * sc:(pad + w) * sc]
    return out


def bicubic(s_norm, scale):
    s = torch.from_numpy(s_norm.astype("float32"))[None, None]
    return F.interpolate(s, scale_factor=scale, mode="bicubic", align_corners=False)[0, 0].numpy()


# ---------------------------------------------------------------- metrics
def detail(a, f):
    n0, n1 = a.shape[0] // f, a.shape[1] // f
    lo = a[:n0 * f, :n1 * f].reshape(n0, f, n1, f).mean(axis=(1, 3))
    return a[:n0 * f, :n1 * f] - np.kron(lo, np.ones((f, f)))


def scores(p, t, m, f):
    """Spike metrics; f = upscale factor relative to 10 m (4 at 2.5 m, 2 at 5 m)."""
    from skimage.metrics import structural_similarity
    tt, pp = np.where(m, t, 0), np.where(m, p, 0)
    lo, hi = np.percentile(t[m], [1, 99])
    ssim = structural_similarity(np.clip(tt, lo, hi), np.clip(pp, lo, hi), data_range=hi - lo)
    n0 = m.shape[0] // f
    mf = m[:n0 * f, :n0 * f].reshape(n0, f, n0, f).all(axis=(1, 3))
    mm = np.kron(mf, np.ones((f, f), bool))
    dp, dt = detail(pp, f)[mm], detail(tt, f)[mm]
    return {"mae": float(np.abs(p - t)[m].mean()), "ssim": float(ssim),
            "detail_r": float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 else 0.0,
            "hf_ratio": float(dp.std() / max(dt.std(), 1e-9))}


def edges(img, m, px, sigma_m=5.0, q=(0.80, 0.90)):
    """Canny edges with quantile thresholds, so every image gets a similar edge density and the
    comparison is about where edges sit, not how many there are. sigma in metres."""
    from skimage.feature import canny
    a = np.where(m, img, np.median(img[m]))
    return canny(a, sigma=sigma_m / px, low_threshold=q[0], high_threshold=q[1], use_quantiles=True, mask=m)


def chamfer(pe, te, m, px, cap_m=50.0, tols=(5.0, 10.0)):
    """Shift-tolerant edge agreement. out->truth: distance from each output edge to the nearest
    truth edge (invented edges raise it); truth->out: the reverse (missed edges raise it).
    Distances capped at cap_m. Also edge precision / recall within each tolerance."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    inner = binary_erosion(m, iterations=6)
    pe, te = pe & inner, te & inner
    if pe.sum() < 20 or te.sum() < 20:
        return {}
    dt_t = distance_transform_edt(~te) * px
    dt_p = distance_transform_edt(~pe) * px
    a, b = np.minimum(dt_t[pe], cap_m), np.minimum(dt_p[te], cap_m)
    out = {"ch_out2true_m": float(a.mean()), "ch_true2out_m": float(b.mean()),
           "chamfer_m": float((a.mean() + b.mean()) / 2)}
    for tol in tols:
        P, R = float((a <= tol).mean()), float((b <= tol).mean())
        out[f"edgeF_{int(tol)}m"] = 2 * P * R / max(P + R, 1e-9)
    return out


def fine_register(ref_up, truth, m, max_px=4):
    """Sub-pixel shift (rows, cols) that aligns truth to ref_up at the fine grid, bounded."""
    from skimage.registration import phase_cross_correlation
    a = np.where(m, ref_up - ref_up[m].mean(), 0)
    b = np.where(m, truth - truth[m].mean(), 0)
    sh, _, _ = phase_cross_correlation(a, b, upsample_factor=4, normalization=None)
    if np.abs(sh).max() > max_px:
        return np.zeros(2)
    return sh


def enl_windows(I, sel, w=9):
    """Exp 4's ENL definition: mean^2/var in w x w windows at the selected (homogeneous) centres, median."""
    from scipy import ndimage as ndi
    m = ndi.uniform_filter(I, w)
    v = ndi.uniform_filter(I ** 2, w) - m ** 2
    e = m[sel] ** 2 / np.maximum(v[sel], 1e-20)
    return float(np.median(e))


def homog_windows(clean, valid, w=9, q=5):
    """Exp 4's selection: non-overlapping w x w windows whose CV in the clean image is in the lowest q %."""
    from scipy import ndimage as ndi
    m = ndi.uniform_filter(clean, w)
    v = ndi.uniform_filter(clean ** 2, w) - m ** 2
    cv = np.sqrt(np.maximum(v, 0)) / np.maximum(m, 1e-12)
    ok = ndi.minimum_filter(valid.astype(np.uint8), w).astype(bool)
    centres = np.zeros_like(ok)
    centres[w // 2::w, w // 2::w] = True
    sel = ok & centres
    thr = np.percentile(cv[sel], q)
    return sel & (cv <= thr)


# ---------------------------------------------------------------- test-time band adaptation
_GAP = None


def adapt(s_norm):
    """Map a standardized real-S1 image toward the statistics of the synthetic (X-band-derived)
    inputs the X-trained models saw: (1) quantile map real -> synthetic, (2) radial spectral
    equalization by sqrt(P_syn / P_real) per frequency band (from e5_input_gap.py, training tiles
    only, unpaired), (3) re-standardize. A fixed per-image transform; nothing is trained on real data."""
    global _GAP
    if _GAP is None:
        _GAP = json.loads((ROOT / "input_gap.json").read_text())
    qr, qs = np.array(_GAP["qmap_real"]), np.array(_GAP["qmap_syn"])
    a = np.interp(s_norm, qr, qs, left=None, right=None)
    lo, hi = s_norm < qr[0], s_norm > qr[-1]
    a[lo] = qs[0] + (s_norm[lo] - qr[0])
    a[hi] = qs[-1] + (s_norm[hi] - qr[-1])
    ratio = np.array(_GAP["spectrum_ratio_real_over_syn"])
    nb = len(ratio)
    centers = (np.arange(nb) + 0.5) / nb
    H, W = a.shape
    fy = np.fft.fftfreq(H)[:, None] * 2
    fx = np.fft.fftfreq(W)[None, :] * 2
    r = np.clip(np.hypot(fy, fx), 0, 1)
    gain = np.sqrt(1.0 / np.interp(r, centers, ratio))
    gain[0, 0] = 1.0
    m = a.mean()
    out = np.real(np.fft.ifft2(np.fft.fft2(a - m) * gain)) + m
    return ((out - np.median(out)) / (out.std() + 1e-6)).astype("float32")


# ---------------------------------------------------------------- NISAR (Experiment 3) pairs
def nisar_tiles(role, frames_out=(), frames_in=None):
    """NISAR same-datatake pairs: 20 MHz GCOV HH (20 m, 256 px) -> 40 MHz GCOV HH (10 m, 512 px).
    Both standardized per tile (exp 5 convention). role: train / near / far."""
    import glob
    out = []
    for f in sorted(glob.glob(str(EXP3_DATA / "tiles" / role / "*.npz"))):
        name = Path(f).stem
        fr = tuple(int(v) for v in name.split("_")[:2])
        if fr in set(frames_out) or (frames_in is not None and fr not in set(frames_in)):
            continue
        z = np.load(f)
        x = z["m20_HH"].astype("float32")
        y = z["a10_HH"].astype("float32")
        m = np.isfinite(y) & (z["a10_mask"] != 255)
        if m.mean() < 0.5 or not np.isfinite(x).any():
            continue
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        out.append(dict(id=name, frame=fr, x=norm(x).astype("float32"),
                        y=np.nan_to_num(norm(np.where(m, y, np.nan)), nan=0.0).astype("float32"), m=m))
    return out


# ---------------------------------------------------------------- Stripmap (Experiment 2) pairs
EXP2_DIR = PROJ / "exp2_iw_to_stripmap"


def sm_manifest():
    p = pd.read_csv(EXP2_DIR / "pairs_final.csv")
    p = p[p.use]
    s = pd.read_csv(EXP2_DIR / "splits.csv")[["tile_id", "split"]]
    return p.merge(s, on="tile_id")


def sm_tiles(splits, target="lee_hm"):
    """Experiment 2's IW (10 m, 320 px) -> Stripmap (5 m, 640 px) pairs, read from exp 2's own tile cache
    (histogram-matched, Lee-filtered reference as in its arm A; exp 2's splits exactly). Input and target are
    standardized with the IW tile's median / std: the reference is histogram-matched to the IW, so one pair of
    statistics serves both and maps outputs back to dB."""
    import glob
    man = sm_manifest()
    man = man[man.split.isin(splits)]
    out = []
    for r in man.itertuples():
        src = EXP2_DIR / "data" / "tiles" / f"{r.tile_id}.npz"
        cf = EXP2_DIR / "data" / "tilecache" / f"{r.tile_id}_{int(src.stat().st_mtime)}.npz"
        if not cf.exists():
            continue
        z = np.load(cf)
        iw = z["iw"].astype("float32")
        y = z[target].astype("float32")
        m = z["mask"] & np.isfinite(y)
        med, sd = norm_stats(iw)
        x = np.where(np.isfinite(iw), iw, med)
        out.append(dict(id=r.tile_id, region=r.region, split=r.split, med=med, sd=sd,
                        x=((x - med) / sd).astype("float32"),
                        y=np.nan_to_num((y - med) / sd, nan=0.0).astype("float32"), m=m))
    return out
