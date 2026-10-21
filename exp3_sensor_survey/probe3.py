from config import DATA  # noqa: E402
import os, sys
import numpy as np
import pandas as pd
from nisar_io import open_remote
import h5py
d = pd.read_csv(str(DATA) + "/mixed_sites.csv")
ids = ["NISAR_L2_PR_GCOV_023_056_A_014_2005_DHDH_M_20260618T014303_20260618T014340_P05023_N_F_J_001",
       "NISAR_L2_PR_GCOV_023_056_A_014_4005_DHDH_A_20260618T014305_20260618T014336_P05023_N_F_J_001"]
H = {}
for i in ids:
    url = d[d.id == i].url.iloc[0]
    h = open_remote(url); H[i] = h
    idn = h["science/LSAR/identification"]
    print("=====", i[-60:], flush=True)
    for k, v in idn.items():
        if isinstance(v, h5py.Dataset) and v.size < 20:
            print("  ", k, v[()], flush=True)
    sd = h["science/LSAR/GCOV/metadata/sourceData"]
    def show(n, o):
        if isinstance(o, h5py.Dataset) and o.size < 10:
            print("   src", n, o[()], flush=True)
    sd.visititems(show)
    for f in "AB":
        g = h[f"science/LSAR/GCOV/grids/frequency{f}"]
        print("  ", f, g["HHHH"].shape, g["xCoordinates"][0], g["yCoordinates"][0], g["xCoordinateSpacing"][()], flush=True)
# compare freq B pixels and freq A pixels over a common window
a, b = [H[i] for i in ids]
for f in "AB":
    ga, gb = a[f"science/LSAR/GCOV/grids/frequency{f}"], b[f"science/LSAR/GCOV/grids/frequency{f}"]
    xa, ya, xb, yb = ga["xCoordinates"][:], ga["yCoordinates"][:], gb["xCoordinates"][:], gb["yCoordinates"][:]
    x0 = max(xa[0], xb[0]); y0 = min(ya[0], yb[0])
    ia, ja = int(np.argmin(abs(ya - y0))) + 200, int(np.argmin(abs(xa - x0))) + 200
    ib, jb = int(np.argmin(abs(yb - y0))) + 200, int(np.argmin(abs(xb - x0))) + 200
    n = 300
    A = ga["HHHH"][ia:ia + n, ja:ja + n]; B = gb["HHHH"][ib:ib + n, jb:jb + n]
    m = np.isfinite(A) & np.isfinite(B) & (A > 0) & (B > 0)
    print(f, "valid", m.mean(), "corr dB", np.corrcoef(10*np.log10(A[m]), 10*np.log10(B[m]))[0, 1], "mean dB", 10*np.log10(A[m]).mean(), 10*np.log10(B[m]).mean(), flush=True)
sys.stdout.flush(); os._exit(0)
