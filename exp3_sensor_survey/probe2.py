import os, sys, time
from nisar_io import open_remote, search
g = search(point=(-95.36, 29.76), page_size=3)[0]
import h5py
h = open_remote(g["url"])
for f in ["frequencyA", "frequencyB"]:
    gg = h[f"science/LSAR/GCOV/grids/{f}"]
    for k, v in gg.items():
        if isinstance(v, h5py.Dataset):
            print(f, k, v.shape, v.dtype, v.chunks, v.compression, flush=True)
    print({k: v for k, v in gg["HHHH"].attrs.items()} if "HHHH" in gg else "", flush=True)
md = h["science/LSAR/GCOV/metadata"]
print(list(md.keys()), flush=True)
pi = h["science/LSAR/GCOV/metadata/processingInformation/parameters"]
print(list(pi.keys()), flush=True)
t=time.time(); a = h["science/LSAR/GCOV/grids/frequencyA/HHHH"][15000:15512, 15000:15512]; print("readA", a.mean(), time.time()-t, flush=True)
sys.stdout.flush(); os._exit(0)
