import os, sys, time
from nisar_io import open_remote, search
g = search(point=(-95.36, 29.76), page_size=3)[0]
print(g, flush=True)
t = time.time()
h = open_remote(g["url"])
def show(name, obj):
    if isinstance(obj, h5py.Dataset) and ("grids" in name or "geolocation" in name.lower()) :
        print(name, obj.shape, obj.dtype, obj.chunks, obj.compression, flush=True)
import h5py
h["science/LSAR/GCOV/grids"].visititems(show)
for f in ["frequencyA", "frequencyB"]:
    gg = h[f"science/LSAR/GCOV/grids/{f}"]
    x, y = gg["xCoordinates"][:], gg["yCoordinates"][:]
    print(f, x[0], x[-1], x[1]-x[0], y[0], y[-1], y[1]-y[0], gg["projection"][()], dict(gg["projection"].attrs).get("epsg_code"), flush=True)
print("listOfFreq", h["science/LSAR/identification/listOfFrequencies"][()], time.time()-t, flush=True)
sys.stdout.flush(); os._exit(0)
