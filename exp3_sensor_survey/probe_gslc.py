"""Round 2 step 3 probe: GSLC structure, chunking, calibration layers, and read speed for a 1024 x 1024 window."""
import os, sys, time
import pandas as pd
from nisar_io import open_remote
import h5py
t = pd.read_csv("tiles.csv").query("role == 'far'").iloc[0]
url = t.url_a.replace("GCOV", "GSLC")
print(url[-110:], flush=True)
h = open_remote(url)
g = h["science/LSAR/GSLC/grids/frequencyA"]
for k, v in g.items():
    if isinstance(v, h5py.Dataset):
        print(k, v.shape, v.dtype, v.chunks, v.compression, flush=True)
x, y = g["xCoordinates"][:], g["yCoordinates"][:]
print("x", x[0], x[1] - x[0], "y", y[0], y[1] - y[0], flush=True)
cal = h["science/LSAR/GSLC/metadata/calibrationInformation"]
def show(n, o):
    if isinstance(o, h5py.Dataset):
        print("cal", n, o.shape, o.dtype, flush=True)
cal.visititems(show)
t0 = time.time()
r0 = int((y[0] - t.y1) / 5); c0 = int((t.x0 - x[0]) / 5)
a = g["HH"][r0:r0 + 1024, c0:c0 + 1024]
print("read 1024^2", a.dtype, round(time.time() - t0, 1), "s", flush=True)
sys.stdout.flush(); os._exit(0)
