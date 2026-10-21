"""Input-domain gap: how real Sentinel-1 at 10 m differs from X-band degraded to S1-like 10 m.

Uses TRAINING tiles only (never the held-out ones), and only as unpaired statistics: for each tile,
the real S1 tile and the synthetic input made from its X-band, both standardized.
  spectrum   radially averaged power spectrum of the standardized dB image, per frequency band
  hf_energy  std of the detail layer (image minus a 3x3 box low-pass), in std units
  pXX        percentiles of the standardized dB distribution (point targets live in the upper tail)
Also the quantile map real -> synthetic (pooled), used by the test-time contrast adaptation in
e5_eval_x.py ("+adapt" rows): a per-image transform, no training on real data.
Writes input_gap.csv and input_gap.json.
"""
from __future__ import annotations

import common  # noqa: F401
from common import DEV, ROOT, SPIKE, degrade, finish, norm, tile_metadata

import json

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi


def spectrum(a, nb=8):
    a = a - a.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    n = a.shape[0]
    y, x = np.indices(F.shape)
    r = np.hypot(y - n // 2, x - n // 2) / (n / 2)          # 0..1 = Nyquist
    edges = np.linspace(0, 1, nb + 1)
    return np.array([F[(r >= edges[i]) & (r < edges[i + 1])].mean() for i in range(nb)])


def hf(a):
    return float((a - ndi.uniform_filter(a, 3)).std())


def main():
    meta = tile_metadata()
    man = pd.read_csv(SPIKE / "manifest.csv")
    man = man[(man.status.isin(["ok", "cached"])) & (man.split == "train") & (man.corr10 >= 0.3)]
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    rows, Sr, Ss, qr, qs = [], [], [], [], []
    for r in man.itertuples():
        fid = f"{r.provider}_{r.xband_id}"
        z = np.load(SPIKE / "data" / f"{fid}.npz")
        x = z["x"].astype("float32")
        if np.isfinite(x).mean() < 0.95:
            continue
        s1 = z["s1"].astype("float32")
        s1 = np.where(np.isfinite(s1), s1, np.nanmedian(s1))
        xf = np.where(np.isfinite(x), x, np.nanmedian(x))
        az = meta.get(fid, {}).get("az", 270.0)
        with torch.no_grad():
            syn = degrade(torch.from_numpy(xf)[None, None].to(DEV), [az], looks=6, gen=gen)[0, 0].cpu().numpy()
        a, b = norm(s1), norm(syn)
        Sr.append(spectrum(a))
        Ss.append(spectrum(b))
        qr.append(np.percentile(a, np.linspace(0.5, 99.5, 199)))
        qs.append(np.percentile(b, np.linspace(0.5, 99.5, 199)))
        rows.append(dict(tile=fid, provider=r.provider, hf_real=hf(a), hf_syn=hf(b),
                         p99_real=float(np.percentile(a, 99)), p99_syn=float(np.percentile(b, 99)),
                         p1_real=float(np.percentile(a, 1)), p1_syn=float(np.percentile(b, 1)),
                         skew_real=float(((a - a.mean()) ** 3).mean()), skew_syn=float(((b - b.mean()) ** 3).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "input_gap.csv", index=False)
    Sr, Ss = np.median(np.array(Sr), 0), np.median(np.array(Ss), 0)
    out = dict(n_tiles=len(df), spectrum_real=Sr.tolist(), spectrum_syn=Ss.tolist(),
               spectrum_ratio_real_over_syn=(Sr / Ss).tolist(),
               hf_real=float(df.hf_real.median()), hf_syn=float(df.hf_syn.median()),
               hf_gain=float(df.hf_syn.median() / df.hf_real.median()),
               p99_real=float(df.p99_real.median()), p99_syn=float(df.p99_syn.median()),
               skew_real=float(df.skew_real.median()), skew_syn=float(df.skew_syn.median()),
               qmap_real=np.median(np.array(qr), 0).tolist(), qmap_syn=np.median(np.array(qs), 0).tolist(),
               by_provider=df.groupby("provider")[["hf_real", "hf_syn", "p99_real", "p99_syn"]].median().round(3).to_dict())
    (ROOT / "input_gap.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if not k.startswith("qmap")}, flush=True)


if __name__ == "__main__":
    main()
    finish()
