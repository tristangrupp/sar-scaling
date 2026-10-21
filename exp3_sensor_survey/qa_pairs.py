"""QA of the simultaneous pairs: are the three resolution levels consistent and co-registered?

For each tile:
  b_identical   max |b80 - b80m| (frequency B from the 40 MHz and the mixed-mode product; same echoes)
  r_m_a         corr of m20 dB with a10 averaged (linear) to 20 m
  bias_m_a      median(m20 - avg(a10)) dB
  l1_m_a        mean |m20 - avg(a10)| dB  (how far the truth itself is from perfect input consistency)
  shift_m_a     phase-correlation shift (m, at 20 m with upsampling) between m20 and avg(a10)
  r_b_m, bias_b_m, shift_b_m: same for b80 vs m20 averaged to 80 m
"""
from config import DATA  # noqa: E402
import os, sys, glob
import numpy as np, pandas as pd
from skimage.registration import phase_cross_correlation


def lin(d):
    return 10 ** (d.astype("float64") / 10)


def pool_db(d, k):
    l = lin(d)
    n = l.shape[0] // k
    return 10 * np.log10(np.nanmean(l.reshape(n, k, n, k), axis=(1, 3)))


def comp(x, y, res):
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(m, x, np.nanmedian(x)), np.where(m, y, np.nanmedian(y))
    sh, _, _ = phase_cross_correlation(yy - yy.mean(), xx - xx.mean(), upsample_factor=10)
    return (float(np.corrcoef(x[m], y[m])[0, 1]), float(np.median(x[m] - y[m])), float(np.abs(x[m] - y[m]).mean()),
            float(np.hypot(*sh) * res), float(sh[0] * res), float(sh[1] * res))


rows = []
files = sorted(glob.glob(str(DATA) + "/tiles/*/*.npz"))
for f in files:
    z = np.load(f)
    a, m, b, bm = [z[k].astype("float32") for k in ["a10_HH", "m20_HH", "b80_HH", "b80m_HH"]]
    r = {"tile": os.path.basename(f)[:-4], "role": f.split(os.sep)[-2] if os.sep in f else f.split("/")[-2]}
    r["b_identical"] = float(np.nanmax(np.abs(b - bm)))
    r["r_m_a"], r["bias_m_a"], r["l1_m_a"], r["shift_m_a"], r["dy_m_a"], r["dx_m_a"] = comp(m, pool_db(a, 2), 20)
    r["r_b_m"], r["bias_b_m"], r["l1_b_m"], r["shift_b_m"], r["dy_b_m"], r["dx_b_m"] = comp(b, pool_db(m, 4), 80)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("qa_pairs.csv", index=False)
pd.set_option("display.width", 200)
print(df.describe().round(3).to_string(), flush=True)
sys.stdout.flush(); os._exit(0)
